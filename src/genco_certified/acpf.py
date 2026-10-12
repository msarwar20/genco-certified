"""AC power flow in float64 on exactly the network GENCO sees.

`extract_problems` turns a *denormalized* graphkit batch into one `PFProblem` per
scenario. The bus admittance matrix is assembled from the same directed-edge
admittances (Yff/Yft) and shunts (GS/BS) that graphkit's `ComputeBranchFlow` /
`ComputeNodeResiduals` use, so our mismatch equals GENCO's power-balance residual
(up to sign) in MW / MVAr.

`newton_pf` is a standard polar Newton-Raphson (MATPOWER `newtonpf` formulation).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from torch_scatter import scatter_add

from gridfm_graphkit.datasets.globals import (
    BS, GS, MAX_VM_H, MIN_VM_H, PD_H, PG_H, QD_H, RATE_A, VA_H, VM_H, YFF_TT_I, YFF_TT_R, YFT_TF_I, YFT_TF_R,
)

EDGE = ("bus", "connects", "bus")
GEN_EDGE = ("gen", "connected_to", "bus")


@dataclass
class PFProblem:
    Ybus: sp.csr_matrix          # complex, physical units (S in MVA when V in p.u.)
    Sbus: np.ndarray             # specified injection Pg - Pd + j(Qg - Qd); only pv/pq P and pq Q are used
    ref: np.ndarray
    pv: np.ndarray
    pq: np.ndarray
    V_true: np.ndarray           # ground-truth complex voltage (datakit AC solution)
    Vm_set: np.ndarray           # Vm used at PV/REF buses, Va at REF (from inputs)
    Va_set: np.ndarray
    meta: dict = field(default_factory=dict)
    # Directed branch edges (one per direction) and limits. Limits are only valid if the
    # batch went through graphkit's RemovePFMask after inverse_transform.
    f: np.ndarray | None = None
    t: np.ndarray | None = None
    yff: np.ndarray | None = None
    yft: np.ndarray | None = None
    rate_a: np.ndarray | None = None
    vmin: np.ndarray | None = None
    vmax: np.ndarray | None = None

    @property
    def pvpq(self) -> np.ndarray:
        return np.r_[self.pv, self.pq]

    def V_from(self, Vm: np.ndarray, Va: np.ndarray) -> np.ndarray:
        """Assemble a full voltage vector, enforcing setpoints at PV/REF buses."""
        Vm = Vm.astype(np.float64).copy()
        Va = Va.astype(np.float64).copy()
        Vm[self.pv] = self.Vm_set[self.pv]
        Vm[self.ref] = self.Vm_set[self.ref]
        Va[self.ref] = self.Va_set[self.ref]
        return Vm * np.exp(1j * Va)

    def flat_start(self) -> np.ndarray:
        n = len(self.Vm_set)
        return self.V_from(np.ones(n), np.full(n, self.Va_set[self.ref].mean()))

    def dc_start(self) -> np.ndarray:
        """Classical DC power-flow initialisation (B theta = P, series susceptances only, |V| = 1 at PQ)."""
        n = len(self.Vm_set)
        b = self.Ybus.imag.tolil()
        b.setdiag(0)
        b = b.tocsr()  # off-diagonal Im(Y_ij) = 1/x_ij for a lossless line
        B = (sp.diags(np.asarray(b.sum(axis=1)).ravel()) - b).tocsr()
        pvpq = self.pvpq
        Va = np.full(n, self.Va_set[self.ref].mean())
        rhs = self.Sbus.real[pvpq] - B[pvpq][:, self.ref] @ Va[self.ref]
        Va[pvpq] = spla.spsolve(B[pvpq][:, pvpq].tocsc(), rhs)
        return self.V_from(np.ones(n), Va)


def extract_problems(batch) -> list[PFProblem]:
    """Split a denormalized PF batch into per-scenario AC power-flow problems."""
    xb = batch.x_dict["bus"].double().cpu().numpy()
    yb = batch.y_dict["bus"].double().cpu().numpy()
    ei = batch.edge_index_dict[EDGE].cpu().numpy()
    ea = batch.edge_attr_dict[EDGE].double().cpu().numpy()
    _, gen_bus = batch.edge_index_dict[GEN_EDGE]
    n_total = xb.shape[0]
    pg_bus = scatter_add(batch.y_dict["gen"][:, PG_H].double(), gen_bus, dim=0, dim_size=n_total).cpu().numpy()
    is_pq = batch.mask_dict["PQ"].cpu().numpy()
    is_pv = batch.mask_dict["PV"].cpu().numpy()
    is_ref = batch.mask_dict["REF"].cpu().numpy()
    ptr = batch["bus"].ptr.cpu().numpy()
    bus_graph = batch.batch_dict["bus"].cpu().numpy()
    edge_graph = bus_graph[ei[0]]

    problems = []
    for g in range(len(ptr) - 1):
        b0, b1 = ptr[g], ptr[g + 1]
        n = b1 - b0
        e = edge_graph == g
        f, t = ei[0, e] - b0, ei[1, e] - b0
        yff = ea[e, YFF_TT_R] + 1j * ea[e, YFF_TT_I]
        yft = ea[e, YFT_TF_R] + 1j * ea[e, YFT_TF_I]
        # Each directed edge f->t contributes I_f += Yff V_f + Yft V_t (ComputeBranchFlow).
        Y = sp.coo_matrix((np.r_[yff, yft], (np.r_[f, f], np.r_[f, t])), shape=(n, n)).tocsr()
        sl = slice(b0, b1)
        Y = Y + sp.diags(xb[sl, GS] + 1j * xb[sl, BS])
        Sbus = (pg_bus[sl] - xb[sl, PD_H]) + 1j * (yb[sl, 2] - xb[sl, QD_H])  # QG_H == 2
        Vm_t, Va_t = yb[sl, VM_H], yb[sl, VA_H]
        problems.append(
            PFProblem(
                Ybus=Y.tocsr(),
                Sbus=Sbus,
                ref=np.flatnonzero(is_ref[sl]),
                pv=np.flatnonzero(is_pv[sl]),
                pq=np.flatnonzero(is_pq[sl]),
                V_true=Vm_t * np.exp(1j * Va_t),
                Vm_set=Vm_t.copy(),
                Va_set=Va_t.copy(),
                meta={"graph": g, "n_bus": int(n), "n_edges": int(e.sum()), "load_MW": float(xb[sl, PD_H].sum())},
                f=f, t=t, yff=yff, yft=yft, rate_a=ea[e, RATE_A],
                vmin=xb[sl, MIN_VM_H], vmax=xb[sl, MAX_VM_H],
            )
        )
    return problems


def mismatch(p: PFProblem, V: np.ndarray) -> np.ndarray:
    """F(V) = [Re(mis)[pvpq], Im(mis)[pq]] with mis = V conj(Ybus V) - Sbus (MW / MVAr)."""
    mis = V * np.conj(p.Ybus @ V) - p.Sbus
    return np.r_[mis[p.pvpq].real, mis[p.pq].imag]


def jacobian(p: PFProblem, V: np.ndarray) -> sp.csr_matrix:
    Ibus = p.Ybus @ V
    diagV, diagI = sp.diags(V).tocsr(), sp.diags(Ibus).tocsr()
    diagVn = sp.diags(V / np.abs(V)).tocsr()
    dS_dVm = diagV @ np.conj(p.Ybus @ diagVn) + np.conj(diagI) @ diagVn
    dS_dVa = 1j * diagV @ np.conj(diagI - p.Ybus @ diagV)
    pvpq, pq = p.pvpq, p.pq
    J11 = dS_dVa[pvpq][:, pvpq].real
    J12 = dS_dVm[pvpq][:, pq].real
    J21 = dS_dVa[pq][:, pvpq].imag
    J22 = dS_dVm[pq][:, pq].imag
    return sp.bmat([[J11, J12], [J21, J22]], format="csc")


def newton_step(p: PFProblem, V: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One Newton step from V. Returns (V_new, dx, F(V)); dx = [dVa[pvpq], dVm[pq]]."""
    F = mismatch(p, V)
    dx = spla.spsolve(jacobian(p, V), -F)
    return apply_dx(p, V, dx), dx, F


def apply_dx(p: PFProblem, V: np.ndarray, dx: np.ndarray) -> np.ndarray:
    Va, Vm = np.angle(V), np.abs(V)
    npvpq = len(p.pvpq)
    Va = Va.copy(); Vm = Vm.copy()
    Va[p.pvpq] += dx[:npvpq]
    Vm[p.pq] += dx[npvpq:]
    return Vm * np.exp(1j * Va)


def state_error(p: PFProblem, V: np.ndarray, V_ref: np.ndarray | None = None) -> np.ndarray:
    """x - x_ref on the unknowns [Va[pvpq], Vm[pq]] (rad, p.u.)."""
    V_ref = p.V_true if V_ref is None else V_ref
    dVa = np.angle(V[p.pvpq] * np.conj(V_ref[p.pvpq]))  # wrap-safe angle difference
    dVm = np.abs(V[p.pq]) - np.abs(V_ref[p.pq])
    return np.r_[dVa, dVm]


def newton_pf(p: PFProblem, V0: np.ndarray, tol_mw: float = 1e-6, max_it: int = 30) -> dict:
    """Run Newton-Raphson from V0. Converged when ||F||_inf < tol_mw (MW/MVAr)."""
    V = V0.copy()
    F = mismatch(p, V)
    hist = [float(np.max(np.abs(F)))]
    it = 0
    while hist[-1] > tol_mw and it < max_it:
        try:
            V, _, _ = newton_step(p, V)
        except Exception:  # singular Jacobian
            break
        it += 1
        F = mismatch(p, V)
        hist.append(float(np.max(np.abs(F))) if np.all(np.isfinite(F)) else np.inf)
        if not np.isfinite(hist[-1]) or hist[-1] > 1e8:
            break
    converged = hist[-1] <= tol_mw
    return {"V": V, "iterations": it, "converged": bool(converged), "history": hist}


def branch_flows(p: PFProblem, V: np.ndarray) -> np.ndarray:
    """Complex sending-end flow S_ft (MVA) on every directed edge (ComputeBranchFlow)."""
    return V[p.f] * np.conj(p.yff * V[p.f] + p.yft * V[p.t])


def branch_keys(p: PFProblem) -> np.ndarray:
    """Undirected branch id per directed edge (parallel circuits share an id)."""
    lo, hi = np.minimum(p.f, p.t), np.maximum(p.f, p.t)
    _, key = np.unique(lo * (len(p.Vm_set) + 1) + hi, return_inverse=True)
    return key


def branch_loading(p: PFProblem, S_edge: np.ndarray) -> np.ndarray:
    """Max |S| / rate over both directions of each undirected branch."""
    key = branch_keys(p)
    with np.errstate(divide="ignore", invalid="ignore"):
        load_edge = np.where(p.rate_a > 0, np.abs(S_edge) / p.rate_a, 0.0)
    out = np.zeros(key.max() + 1)
    np.maximum.at(out, key, load_edge)
    return out


def dc_branch_flows(p: PFProblem, Va: np.ndarray) -> np.ndarray:
    """DC flow P_ft = Im(yft) (theta_f - theta_t) on every directed edge."""
    return p.yft.imag * (Va[p.f] - Va[p.t])


# ---------------------------------------------------------------------------
# Problems built directly from datakit raw parquet (independent of graphkit).
# Used to reconstruct the pre-contingency base case of each load scenario, the
# industry-standard warm start for contingency analysis.
# Raw units: Pd/Qd/Pg in MW, admittances and GS/BS in p.u. on baseMVA.
# ---------------------------------------------------------------------------

def problem_from_raw(bus, branch, gen, base_mva: float = 100.0, restore_all: bool = False) -> PFProblem:
    """Build a PFProblem from one scenario's raw rows (bus ids must be 0..n-1).

    restore_all=True puts every branch back in service (Y taken from `branch`, which
    must then hold the in-service admittances for all branches) -> base case.
    """
    bus = bus.sort_values("bus")
    n = len(bus)
    assert (bus.bus.to_numpy() == np.arange(n)).all()
    br = branch if restore_all else branch[branch.br_status == 1]
    f, t = br.from_bus.to_numpy().astype(int), br.to_bus.to_numpy().astype(int)
    yff = (br.Yff_r + 1j * br.Yff_i).to_numpy() * base_mva
    yft = (br.Yft_r + 1j * br.Yft_i).to_numpy() * base_mva
    ytf = (br.Ytf_r + 1j * br.Ytf_i).to_numpy() * base_mva
    ytt = (br.Ytt_r + 1j * br.Ytt_i).to_numpy() * base_mva
    F, T = np.r_[f, t], np.r_[t, f]
    YFF, YFT = np.r_[yff, ytt], np.r_[yft, ytf]
    Y = sp.coo_matrix((np.r_[YFF, YFT], (np.r_[F, F], np.r_[F, T])), shape=(n, n)).tocsr()
    Y = (Y + sp.diags((bus.GS + 1j * bus.BS).to_numpy() * base_mva)).tocsr()
    g = gen[gen.in_service == 1]
    pg = np.bincount(g.bus.to_numpy().astype(int), weights=g.p_mw.to_numpy(), minlength=n)
    Sbus = (pg - bus.Pd.to_numpy()) + 1j * (bus.Qg.to_numpy() - bus.Qd.to_numpy())
    Vm, Va = bus.Vm.to_numpy(), np.deg2rad(bus.Va.to_numpy())
    rate = br.rate_a.to_numpy()
    return PFProblem(
        Ybus=Y, Sbus=Sbus,
        ref=np.flatnonzero(bus.REF.to_numpy() == 1), pv=np.flatnonzero(bus.PV.to_numpy() == 1),
        pq=np.flatnonzero(bus.PQ.to_numpy() == 1),
        V_true=Vm * np.exp(1j * Va), Vm_set=Vm.copy(), Va_set=Va.copy(),
        meta={"n_bus": n, "load_MW": float(bus.Pd.sum())},
        f=F, t=T, yff=YFF, yft=YFT, rate_a=np.r_[rate, rate],
        vmin=bus.min_vm_pu.to_numpy(), vmax=bus.max_vm_pu.to_numpy(),
    )


def base_case_voltages(bus_ls, branch_ls, gen_ls, base_mva: float = 100.0) -> dict:
    """Solve the pre-contingency base case of one load scenario from its topology variants.

    Each variant takes one (or k) extra element(s) out of service. Some grids (e.g. GOC 2000)
    also have elements that are out of service in the base case itself, so the intact state of
    every element is its *majority* status across the variants. Same for bus types. Admittances,
    dispatch and setpoints are taken from variants where the element is in service.
    Returns {"V": ..., "converged": ..., "iterations": ...}.
    """
    nvar = bus_ls.scenario.nunique()
    s0 = int(bus_ls.scenario.min())
    # branches: majority status; in-service admittances from any variant where in service
    st = branch_ls.groupby("idx").br_status.mean() > 0.5
    br0 = branch_ls[branch_ls.scenario == s0].set_index("idx").copy()
    yin = branch_ls[branch_ls.br_status == 1].drop_duplicates("idx").set_index("idx")
    cols = ["Yff_r", "Yff_i", "Yft_r", "Yft_i", "Ytf_r", "Ytf_i", "Ytt_r", "Ytt_i", "rate_a"]
    on = st[st].index.intersection(yin.index)
    br0.loc[on, cols] = yin.loc[on, cols].to_numpy()
    br0["br_status"] = st.reindex(br0.index).fillna(False).astype(int)
    # generators: majority in-service; dispatch from a variant where in service
    gs = gen_ls.groupby("idx").in_service.mean() > 0.5
    g0 = gen_ls[gen_ls.scenario == s0].set_index("idx").copy()
    gin = gen_ls[gen_ls.in_service == 1].drop_duplicates("idx").set_index("idx")
    gon = gs[gs].index.intersection(gin.index)
    g0.loc[gon, ["p_mw", "q_mvar"]] = gin.loc[gon, ["p_mw", "q_mvar"]].to_numpy()
    g0["in_service"] = gs.reindex(g0.index).fillna(False).astype(int)
    # buses: majority type; PV/REF voltage setpoints from a variant where the bus has that type
    b0 = bus_ls[bus_ls.scenario == s0].set_index("bus").copy()
    typ = bus_ls.assign(t=bus_ls.PQ * 0 + bus_ls.PV * 1 + bus_ls.REF * 2).groupby("bus").t.agg(lambda x: x.mode().iloc[0])
    for col, code in [("PQ", 0), ("PV", 1), ("REF", 2)]:
        b0[col] = (typ.reindex(b0.index) == code).astype(int)
    pvref = bus_ls[(bus_ls.PV == 1) | (bus_ls.REF == 1)].drop_duplicates("bus").set_index("bus")
    idx = b0.index[(b0.PV == 1) | (b0.REF == 1)].intersection(pvref.index)
    b0.loc[idx, "Vm"] = pvref.loc[idx, "Vm"]
    p = problem_from_raw(b0.reset_index(), br0.reset_index(), g0.reset_index(), base_mva)
    r = newton_pf(p, p.flat_start())
    return {"V": r["V"], "converged": r["converged"], "iterations": r["iterations"], "variants": int(nvar)}
