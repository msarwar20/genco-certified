"""Industry-style contingency solvers that reuse one factorization per operating point.

Full Newton-Raphson refactorizes the Jacobian at every iteration of every contingency. Production
contingency analysis does not: it factorizes once per operating point (here: per *load scenario*, the
pre-contingency base case) and re-uses that factorization for every contingency of the operating point.
This module implements the two standard ways of doing so on `acpf.PFProblem` objects, so that the work
saved by a good starting point (GENCO's prediction) can be priced against a baseline that is already
cheap.

(a) `ChordContext`: chord (simplified) Newton. The Jacobian of the BASE topology at the base-case
    solution is LU-factorized once; every contingency iterates  x <- x - J0^{-1} F_c(x)  on its own
    mismatch F_c. Branch outages keep the unknown set. A generator outage that turns a PV bus into PQ
    changes the unknown set: one extra factorization per distinct (pv, pq) pattern is created and cached
    for the load scenario (counted, `FastResult.new_factors`). Optional `bordered=True` instead keeps the
    base factorization and solves the enlarged system with a Schur complement (no new factorization, one
    extra base solve per iteration plus one setup solve per new PQ bus).

(b) `FDLFContext`: fast-decoupled load flow, XB variant (Stott & Alsac; MATPOWER makeB alg 2).
        B'  = Laplacian of 1/x over PV+PQ buses (r, shunts, taps and line charging ignored),
        B'' = -Im(Ybus) over PQ buses (full Ybus: taps, charging, shunts, r; phase shifts would be zeroed,
              the data has none).
    Both are factorized once for the BASE topology. A branch outage changes B' by a rank-1 term and B''
    by a rank <= 2 term per outaged branch; the outaged systems are solved with Sherman-Morrison-Woodbury
    compensation on the base factors (`LowRank`): one base solve per half-step plus a small dense update,
    and k setup solves per contingency for the compensation vectors. Iterations alternate P-theta and Q-V
    half-steps until max |mismatch| (MW / MVAr) falls below the tolerance. A generator outage changes the
    PQ set, hence B'': one extra B'' factorization per distinct PQ pattern, cached (counted).

Units. `acpf.PFProblem.Ybus` is in MVA (per-unit admittance times baseMVA), so B' and B'' are scaled by
baseMVA and the P/Q mismatches are in MW / MVAr: F = V conj(Ybus V) - Sbus, exactly the mismatch that
`acpf.newton_pf` tests. All counters are explicit: `solves` counts forward+backward substitutions with one
LU factor (one "triangular-solve pair") inside the iteration, `setup_solves` those spent on compensation
vectors / bordering, `new_factors` the LU factorizations created for a contingency (the per-load-scenario
base factorizations are counted once in `base_factors`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from genco_certified import acpf

TOLS = (1e-2, 1e-4)  # MW / MVAr: practical contingency-analysis tolerances
MAX_IT = 50
DIVERGED = 1e8


# ---------------------------------------------------------------------------
# Base case
# ---------------------------------------------------------------------------

@dataclass
class BaseCase:
    """Pre-contingency state of one load scenario (see `acpf.base_case_voltages`)."""

    p: acpf.PFProblem           # base-topology problem built from the raw parquet (float64)
    V: np.ndarray               # converged base-case voltages
    converged: bool
    iterations: int
    branch: pd.DataFrame        # branch table handed to problem_from_raw: base br_status, x, r, tap, Y entries
    base_mva: float = 100.0

    def start_for(self, p_c: acpf.PFProblem) -> np.ndarray:
        """Base-case voltages as a start for contingency problem p_c (setpoints of p_c enforced, as in exp02)."""
        return p_c.V_from(np.abs(self.V), np.angle(self.V))

    @property
    def in_service(self) -> pd.DataFrame:
        return self.branch[self.branch.br_status == 1]


def solve_base_case(bus_ls, branch_ls, gen_ls, base_mva: float = 100.0) -> BaseCase:
    """`acpf.base_case_voltages`, additionally returning the base-topology problem and branch table.

    The base case is reconstructed by `acpf.base_case_voltages` itself (majority statuses across the
    variants of the load scenario); this wrapper only intercepts the call it makes to
    `acpf.problem_from_raw` so that the very same problem object / branch table is available, and the
    returned V is bit-identical to `acpf.base_case_voltages(...)["V"]`.
    """
    cap: dict = {}
    orig = acpf.problem_from_raw

    def spy(bus, branch, gen, *a, **k):
        p = orig(bus, branch, gen, *a, **k)
        cap["p"], cap["branch"] = p, branch.copy()
        return p

    acpf.problem_from_raw = spy
    try:
        r = acpf.base_case_voltages(bus_ls, branch_ls, gen_ls, base_mva)
    finally:
        acpf.problem_from_raw = orig
    return BaseCase(p=cap["p"], V=r["V"], converged=bool(r["converged"]), iterations=int(r["iterations"]),
                    branch=cap["branch"].reset_index(drop=True), base_mva=base_mva)


def outaged_branches(base: BaseCase, branch_c: pd.DataFrame) -> pd.DataFrame:
    """Base-case in-service branches that are out of service in the contingency scenario (rows of base.branch).

    `branch_c` holds the contingency scenario's raw branch rows (needs `idx`, `br_status`). Raises if the
    contingency restores a branch that is out in the base case (not supported by compensation).
    """
    st = branch_c.set_index("idx").br_status
    b = base.branch
    now = b.idx.map(st)
    if now.isna().any():
        raise ValueError("contingency branch table does not cover every base-case branch")
    if ((b.br_status == 0) & (now == 1)).any():
        raise ValueError("contingency restores a branch that is out of service in the base case")
    return b[(b.br_status == 1) & (now == 0)]


@dataclass
class Outage:
    """Numpy view of the outaged base-case branches (avoids pandas overhead inside the solvers)."""
    f: np.ndarray
    t: np.ndarray
    x: np.ndarray
    Y: dict          # "Yff"/"Yft"/"Ytf"/"Ytt" -> complex per-unit admittance entries of each branch

    def __len__(self):
        return len(self.f)


def as_outage(outaged) -> Outage:
    """Accept the DataFrame returned by `outaged_branches` or an Outage."""
    if isinstance(outaged, Outage):
        return outaged
    Y = {nm: outaged[f"{nm}_r"].to_numpy() + 1j * outaged[f"{nm}_i"].to_numpy() for nm in ("Yff", "Yft", "Ytf", "Ytt")}
    return Outage(outaged.from_bus.to_numpy().astype(int), outaged.to_bus.to_numpy().astype(int), outaged.x.to_numpy(), Y)


def pattern_key(pv: np.ndarray, pq: np.ndarray) -> bytes:
    return np.sort(pv).astype(np.int32).tobytes() + b"|" + np.sort(pq).astype(np.int32).tobytes()


# ---------------------------------------------------------------------------
# Results and small linear-algebra helpers
# ---------------------------------------------------------------------------

@dataclass
class FastResult:
    V: np.ndarray
    converged: bool                 # reached min(tols)
    reason: str                     # "" | "max_it" | "diverged" | "singular"
    hist: list                      # max |F| after the start and after every step / half-step
    hit: dict                       # tol -> {"iters", "solves", "evals"} at first crossing; None if never
    solves: int                     # triangular-solve pairs used inside the iteration
    setup_solves: int = 0           # compensation / bordering solves (once per contingency)
    new_factors: int = 0            # LU factorizations created for this contingency (pattern changes)
    evals: int = 0                  # mismatch evaluations

    def iters(self, tol: float) -> float:
        h = self.hit.get(tol)
        return np.nan if h is None else h["iters"]

    def solves_to(self, tol: float, include_setup: bool = True) -> float:
        h = self.hit.get(tol)
        return np.nan if h is None else h["solves"] + (self.setup_solves if include_setup else 0)


class SingularCompensation(RuntimeError):
    """The low-rank update makes the outaged matrix singular (e.g. the outage islands part of the grid)."""


class LowRank:
    """Solve (A + U C W^T) y = r from a factorization of A (Sherman-Morrison-Woodbury).

    (A + U C W^T)^{-1} = A^{-1} - Z (I + C W^T Z)^{-1} C W^T A^{-1},   Z = A^{-1} U.
    `lu` is a SuperLU (or any object with `.solve`) of A. U is n x k, W is n x k' (default W = U, the
    symmetric case), C is k x k' dense; C need not be invertible. Building the object costs k solves with
    A (`setup_solves`); each `solve` costs one solve with A (`solves_per_call` of the underlying solver).
    """

    def __init__(self, lu, U: sp.spmatrix, C: np.ndarray, W: sp.spmatrix | None = None):
        self.lu = lu
        self.solves_per_call = getattr(lu, "solves_per_call", 1)
        U = U.tocsr()
        self.WT = (U if W is None else W.tocsr()).T.tocsr()
        k = U.shape[1]
        self.setup_solves = k * self.solves_per_call
        self.Z = lu.solve(np.asarray(U.todense())) if k else np.zeros((U.shape[0], 0))
        S = np.eye(k) + C @ (self.WT @ self.Z)
        try:
            if k and np.linalg.cond(S) > 1e12:
                raise np.linalg.LinAlgError
            self.M = np.linalg.solve(S, C) if k else np.zeros((0, 0))
        except np.linalg.LinAlgError as e:
            raise SingularCompensation("capacitance matrix singular") from e

    def solve(self, r: np.ndarray) -> np.ndarray:
        w = self.lu.solve(r)
        if self.Z.shape[1] == 0:
            return w
        return w - self.Z @ (self.M @ (self.WT @ w))


def _lu(M: sp.spmatrix):
    return spla.splu(M.tocsc())


def _hit_update(hit, hist_val, tols, iters, solves, evals):
    for t in tols:
        if hit[t] is None and hist_val <= t:
            hit[t] = {"iters": iters, "solves": solves, "evals": evals}


# ---------------------------------------------------------------------------
# (a) Chord Newton with the base-case Jacobian
# ---------------------------------------------------------------------------

def _jac_from_ds(dSa, dSm, pvpq, pq) -> sp.csc_matrix:
    """Same blocks as acpf.jacobian, from precomputed dS/dVa, dS/dVm (all buses) for any (pvpq, pq) pattern."""
    return sp.bmat([[dSa[pvpq][:, pvpq].real, dSm[pvpq][:, pq].real],
                    [dSa[pq][:, pvpq].imag, dSm[pq][:, pq].imag]], format="csc")


def _jac_derivatives(Ybus: sp.csr_matrix, V: np.ndarray):
    Ibus = Ybus @ V
    diagV, diagI = sp.diags(V).tocsr(), sp.diags(Ibus).tocsr()
    diagVn = sp.diags(V / np.abs(V)).tocsr()
    dSm = diagV @ np.conj(Ybus @ diagVn) + np.conj(diagI) @ diagVn
    dSa = 1j * diagV @ np.conj(diagI - Ybus @ diagV)
    return dSa.tocsr(), dSm.tocsr()


class _Bordered:
    """Solve [[A0, B], [C, D]] x = r with a factorization of A0 only (Schur complement).

    `M` is the full matrix in the NEW ordering; `old` lists, in the order of A0's own rows/columns, the
    positions of A0's unknowns in M, and `new` the remaining positions (buses that changed type). Then
    M[old][:, old] equals A0 exactly, so no new factorization is needed: one setup solve per new unknown
    (Z = A0^{-1} B) and two base solves per call.
    """

    solves_per_call = 2

    def __init__(self, lu0, M: sp.spmatrix, old: np.ndarray, new: np.ndarray):
        Mc = M.tocsr()
        self.lu0, self.old, self.new = lu0, old, new
        self.B = Mc[old][:, new].toarray()
        self.C = Mc[new][:, old].toarray()
        D = Mc[new][:, new].toarray()
        self.Z = lu0.solve(self.B)
        self.Sinv = np.linalg.inv(D - self.C @ self.Z)
        self.setup_solves = len(new)

    def solve(self, r: np.ndarray) -> np.ndarray:
        w1 = self.lu0.solve(r[self.old])
        x2 = self.Sinv @ (r[self.new] - self.C @ w1)
        x1 = self.lu0.solve(r[self.old] - self.B @ x2)
        x = np.empty_like(r)
        x[self.old], x[self.new] = x1, x2
        return x


class ChordContext:
    """Chord Newton for all contingencies of one load scenario (one base-topology Jacobian factorization)."""

    def __init__(self, base: BaseCase, bordered: bool = False):
        self.base = base
        self.bordered = bordered
        p0 = base.p
        self.dSa, self.dSm = _jac_derivatives(p0.Ybus, base.V)
        self._cache: dict = {}
        self.base_factors = 0
        self.t_base_factor = 0.0
        t0 = time.perf_counter()
        self._base_pat = self._factor(p0.pv, p0.pq)
        self.t_base_factor = time.perf_counter() - t0
        self.base_factors = 1
        self._cache[pattern_key(p0.pv, p0.pq)] = self._base_pat
        self.n_patterns_created = 0

    def _factor(self, pv, pq):
        pvpq = np.r_[pv, pq]
        return _lu(_jac_from_ds(self.dSa, self.dSm, pvpq, pq))

    def _pattern(self, p_c: acpf.PFProblem):
        """(solver, new factorizations created, setup solves). Cached per distinct (pv, pq) pattern."""
        key = pattern_key(p_c.pv, p_c.pq)
        if key in self._cache:
            return self._cache[key], 0, 0
        p0 = self.base.p
        if self.bordered and set(p0.pq.tolist()) <= set(p_c.pq.tolist()) and set(p0.pvpq.tolist()) == set(p_c.pvpq.tolist()):
            pvpq, pq = p_c.pvpq, p_c.pq
            Jn = _jac_from_ds(self.dSa, self.dSm, np.r_[p_c.pv, pq], pq)
            pos_a = {int(b): i for i, b in enumerate(pvpq)}
            pos_v = {int(b): len(pvpq) + i for i, b in enumerate(pq)}
            old = np.array([pos_a[int(b)] for b in p0.pvpq] + [pos_v[int(b)] for b in p0.pq])
            base_pq = set(p0.pq.tolist())
            new = np.array([pos_v[int(b)] for b in pq if int(b) not in base_pq])
            pat = _Bordered(self._base_pat, Jn, old, new)
            self._cache[key] = pat
            return pat, 0, pat.setup_solves
        pat = self._factor(p_c.pv, p_c.pq)
        self._cache[key] = pat
        self.n_patterns_created += 1
        return pat, 1, 0

    def _outage_correction(self, p_c: acpf.PFProblem, pat, outaged):
        """LowRank solver for J_pattern(V_base) minus the Jacobian contribution of the outaged branches.

        dS/dV is linear in Ybus for fixed V, so the Jacobian of the outaged base topology at V_base is the
        pattern Jacobian minus the Jacobian built from the outaged branches' own admittance entries; it has
        at most 4 nonzero rows / columns per branch (P, Q at both ends)."""
        K, n = self.base.base_mva, len(self.base.p.Vm_set)
        o = as_outage(outaged)
        f, t = o.f, o.t
        ent = [(a, b, K * o.Y[nm]) for nm, a, b in (("Yff", f, f), ("Yft", f, t), ("Ytf", t, f), ("Ytt", t, t))]
        Ybr = sp.coo_matrix((np.concatenate([e[2] for e in ent]),
                             (np.concatenate([e[0] for e in ent]), np.concatenate([e[1] for e in ent]))), shape=(n, n)).tocsr()
        dSa_b, dSm_b = _jac_derivatives(Ybr, self.base.V)
        Jb = _jac_from_ds(dSa_b, dSm_b, p_c.pvpq, p_c.pq).tocoo()
        keep = Jb.data != 0
        if not keep.any():
            return pat
        R, Cc = np.unique(Jb.row[keep]), np.unique(Jb.col[keep])
        N = Jb.shape[0]
        Jd = Jb.tocsr()[R][:, Cc].toarray()
        U = sp.coo_matrix((np.ones(len(R)), (R, np.arange(len(R)))), shape=(N, len(R))).tocsr()
        W = sp.coo_matrix((np.ones(len(Cc)), (Cc, np.arange(len(Cc)))), shape=(N, len(Cc))).tocsr()
        return LowRank(pat, U, -Jd, W)

    def solve(self, p_c: acpf.PFProblem, V0: np.ndarray, tols=TOLS, max_it: int = MAX_IT,
              outaged=None) -> FastResult:
        """Chord iteration x <- x - J^{-1} F_c(x). With outaged=None J is the base-topology Jacobian (plain chord,
        same J for every contingency of the pattern). With the outaged base branches given, the Jacobian is
        compensated for them (rank <= 4 per branch, `setup_solves`), still evaluated at the base-case voltages."""
        pat, new_f, setup = self._pattern(p_c)
        if outaged is not None and len(outaged):
            try:
                pat = self._outage_correction(p_c, pat, outaged)
                setup += pat.setup_solves if isinstance(pat, LowRank) else 0
            except SingularCompensation:
                return FastResult(V=V0, converged=False, reason="singular", hist=[np.nan], hit={t: None for t in tols},
                                  solves=0, setup_solves=setup, new_factors=new_f, evals=0)
        conv_tol = min(tols)
        V = V0.copy()
        F = acpf.mismatch(p_c, V)
        hist = [float(np.abs(F).max())]
        hit = {t: None for t in tols}
        evals, solves = 1, 0
        _hit_update(hit, hist[-1], tols, 0, 0, 1)
        reason, it = "", 0
        while hist[-1] > conv_tol and it < max_it:
            dx = pat.solve(-F)
            solves += getattr(pat, "solves_per_call", 1)
            V = acpf.apply_dx(p_c, V, dx)
            it += 1
            F = acpf.mismatch(p_c, V)
            evals += 1
            h = float(np.abs(F).max()) if np.all(np.isfinite(F)) else np.inf
            hist.append(h)
            _hit_update(hit, h, tols, it, solves, evals)
            if not np.isfinite(h) or h > DIVERGED:
                reason = "diverged"
                break
        conv = hist[-1] <= conv_tol
        if not conv and not reason:
            reason = "max_it"
        return FastResult(V=V, converged=conv, reason=reason, hist=hist, hit=hit, solves=solves,
                          setup_solves=setup, new_factors=new_f, evals=evals)


# ---------------------------------------------------------------------------
# (b) Fast-decoupled XB with rank-k compensation of branch outages
# ---------------------------------------------------------------------------

def build_Bp(n_bus: int, br: pd.DataFrame, pvpq: np.ndarray, base_mva: float) -> sp.csc_matrix:
    """XB B': weighted Laplacian with weights baseMVA / x of the in-service branches, restricted to PV+PQ buses."""
    f, t = br.from_bus.to_numpy().astype(int), br.to_bus.to_numpy().astype(int)
    b = base_mva / br.x.to_numpy()
    L = sp.coo_matrix((np.r_[b, b, -b, -b], (np.r_[f, t, f, t], np.r_[f, t, t, f])), shape=(n_bus, n_bus)).tocsr()
    return L[pvpq][:, pvpq].tocsc()


def build_Bpp(Ybus: sp.spmatrix, pq: np.ndarray) -> sp.csc_matrix:
    """XB B'': -Im(Ybus) restricted to PQ buses (taps, charging, shunts and r included)."""
    B = sp.csr_matrix(-Ybus.imag)
    return B[pq][:, pq].tocsc()


@dataclass
class FDOperator:
    """Solvers for the (possibly outaged / re-patterned) B' and B'' of one contingency."""
    solve_p: callable
    solve_q: callable
    cost_p: int          # base-factor solves per P half-step (1; 2 when B'' is bordered)
    cost_q: int
    setup_solves: int
    new_factors: int
    perm_p: np.ndarray | None = None   # position of p_c.pvpq[i] in B' ordering (None = identical order)


class FDLFContext:
    """Fast-decoupled XB for all contingencies of one load scenario (B', B'' factorized once, base topology).

    bordered=True: a generator outage (new PQ buses) is handled by a Schur complement on the base B''
    factorization instead of one extra B'' factorization per distinct PQ pattern.
    """

    def __init__(self, base: BaseCase, bordered: bool = False):
        self.base = base
        self.bordered = bordered
        p0 = base.p
        self.n = len(p0.Vm_set)
        self.K = base.base_mva
        self.pvpq0 = p0.pvpq
        t0 = time.perf_counter()
        self.Bp0 = build_Bp(self.n, base.in_service, self.pvpq0, self.K)
        self.lu_p = _lu(self.Bp0)
        self.Bpp_full = sp.csr_matrix(-p0.Ybus.imag)
        self._pq_cache: dict[bytes, tuple] = {}
        self._base_pq_key = np.sort(p0.pq).astype(np.int32).tobytes()
        Bpp0 = self.Bpp_full[p0.pq][:, p0.pq].tocsc()
        self._base_lu_q = _lu(Bpp0)
        self._pq_cache[self._base_pq_key] = (self._base_lu_q, p0.pq, 0, 0)
        self.t_base_factor = time.perf_counter() - t0
        self.base_factors = 2
        self.pos_p = np.full(self.n, -1)
        self.pos_p[self.pvpq0] = np.arange(len(self.pvpq0))
        self.n_patterns_created = 0

    def _get_pq(self, pq: np.ndarray):
        """(solver for B'' on this PQ pattern, pq order used, new factorizations, setup solves)."""
        key = np.sort(pq).astype(np.int32).tobytes()
        if key in self._pq_cache:
            lu, order, _, _ = self._pq_cache[key]
            return lu, order, 0, 0
        p0 = self.base.p
        if self.bordered and set(p0.pq.tolist()) <= set(pq.tolist()):
            pos = {int(b): i for i, b in enumerate(pq)}
            old = np.array([pos[int(b)] for b in p0.pq])
            base_pq = set(p0.pq.tolist())
            new = np.array([pos[int(b)] for b in pq if int(b) not in base_pq])
            bor = _Bordered(self._base_lu_q, self.Bpp_full[pq][:, pq], old, new)
            self._pq_cache[key] = (bor, pq, 0, bor.setup_solves)
            return bor, pq, 0, bor.setup_solves
        lu = _lu(self.Bpp_full[pq][:, pq])
        self._pq_cache[key] = (lu, pq, 1, 0)
        self.n_patterns_created += 1
        return lu, pq, 1, 0

    def prepare(self, p_c: acpf.PFProblem, outaged) -> FDOperator:
        """Operator for contingency p_c whose branches `outaged` (rows of base.branch) are out of service."""
        if set(p_c.pvpq.tolist()) != set(self.pvpq0.tolist()) or set(p_c.ref.tolist()) != set(self.base.p.ref.tolist()):
            raise ValueError("PV+PQ / REF sets differ from the base case: not supported")
        K = self.K
        o = as_outage(outaged)
        f, t, m = o.f, o.t, len(o)
        # --- B': rank-1 per outaged branch, -K/x u u^T with u = e_f - e_t (reference bus dropped)
        rows, cols, vals = [], [], []
        for k in range(m):
            for bus, sgn in ((f[k], 1.0), (t[k], -1.0)):
                if self.pos_p[bus] >= 0:
                    rows.append(self.pos_p[bus]); cols.append(k); vals.append(sgn)
        Up = sp.coo_matrix((vals, (rows, cols)), shape=(len(self.pvpq0), m)).tocsr()
        Cp = np.diag(-K / o.x) if m else np.zeros((0, 0))
        comp_p = LowRank(self.lu_p, Up, Cp) if m else None
        # --- B'': PQ pattern of the contingency (new factorization if it is a new pattern), rank <= 2 per branch
        base_q, pq, new_f, setup_b = self._get_pq(p_c.pq)
        pos_q = np.full(self.n, -1)
        pos_q[pq] = np.arange(len(pq))
        touched = sorted({int(b) for b in np.r_[f, t] if pos_q[b] >= 0})
        comp_q = None
        if m and touched:
            loc = {b: i for i, b in enumerate(touched)}
            C = np.zeros((len(touched), len(touched)))
            for k in range(m):
                ent = ((f[k], f[k], "Yff"), (f[k], t[k], "Yft"), (t[k], f[k], "Ytf"), (t[k], t[k], "Ytt"))
                for a, b, nm in ent:
                    if a in loc and b in loc:
                        C[loc[a], loc[b]] += K * o.Y[nm][k].imag  # Im of the branch Y (MVA units)
            Uq = sp.coo_matrix((np.ones(len(touched)), (pos_q[touched], np.arange(len(touched)))),
                               shape=(len(pq), len(touched))).tocsr()
            comp_q = LowRank(base_q, Uq, C)
        cost_q = getattr(base_q, "solves_per_call", 1)
        setup = (comp_p.setup_solves if comp_p else 0) + (comp_q.setup_solves if comp_q else 0) + setup_b
        sp_ = comp_p.solve if comp_p else self.lu_p.solve
        sq_ = comp_q.solve if comp_q else base_q.solve
        # B' is ordered like the base case's pvpq = [pv; pq]; a generator outage moves a bus from pv to pq
        idx = self.pos_p[p_c.pvpq]
        perm = None if np.array_equal(idx, np.arange(len(idx))) else idx
        return FDOperator(sp_, sq_, 1, cost_q, setup, new_f, perm)

    def solve(self, p_c: acpf.PFProblem, V0: np.ndarray, op: FDOperator, tols=TOLS, max_it: int = MAX_IT) -> FastResult:
        """Alternate P-theta and Q-V half-steps until max |mismatch| <= min(tols). One iteration = two half-steps;
        `iters` is counted in full iterations (a crossing after the P half-step counts k - 0.5)."""
        conv_tol = min(tols)
        pvpq, pq = p_c.pvpq, p_c.pq
        npvpq = len(pvpq)
        Va, Vm = np.angle(V0).copy(), np.abs(V0).copy()
        V = V0.copy()
        F = acpf.mismatch(p_c, V)
        hist = [float(np.abs(F).max())]
        hit = {t: None for t in tols}
        evals, solves = 1, 0
        _hit_update(hit, hist[-1], tols, 0.0, 0, 1)
        reason = ""
        half = 0
        while hist[-1] > conv_tol and half < 2 * max_it:
            if half % 2 == 0:  # P-theta
                rhs = F[:npvpq] / Vm[pvpq]
                if op.perm_p is None:
                    dVa = op.solve_p(rhs)
                else:
                    r0 = np.empty(npvpq)
                    r0[op.perm_p] = rhs
                    dVa = op.solve_p(r0)[op.perm_p]
                Va[pvpq] -= dVa
                solves += op.cost_p
            else:              # Q-V
                Vm[pq] += -op.solve_q(F[npvpq:] / Vm[pq])
                solves += op.cost_q
            half += 1
            V = Vm * np.exp(1j * Va)
            F = acpf.mismatch(p_c, V)
            evals += 1
            h = float(np.abs(F).max()) if np.all(np.isfinite(F)) else np.inf
            hist.append(h)
            _hit_update(hit, h, tols, half / 2.0, solves, evals)
            if not np.isfinite(h) or h > DIVERGED:
                reason = "diverged"
                break
        conv = hist[-1] <= conv_tol
        if not conv and not reason:
            reason = "max_it"
        return FastResult(V=V, converged=conv, reason=reason, hist=hist, hit=hit, solves=solves,
                          setup_solves=op.setup_solves, new_factors=op.new_factors, evals=evals)


def fdlf_solve(ctx: FDLFContext, p_c: acpf.PFProblem, V0: np.ndarray, outaged, tols=TOLS,
               max_it: int = MAX_IT) -> FastResult:
    """prepare + solve; an island / singular compensation is returned as a recorded failure."""
    try:
        op = ctx.prepare(p_c, outaged)
    except SingularCompensation:
        return FastResult(V=V0, converged=False, reason="singular", hist=[np.nan], hit={t: None for t in tols},
                          solves=0, evals=0)
    return ctx.solve(p_c, V0, op, tols, max_it)
