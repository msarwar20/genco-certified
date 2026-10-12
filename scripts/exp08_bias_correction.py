"""Experiment 08: are GENCO's SIGNED voltage errors systematic per bus? A per-bus bias correction test.

The embedding probe (exp04 v2) found that GENCO's |dVm| errors are largely bus-specific. If the signed
error of bus b has a stable mean (or a stable dependence on the load level or on the bus's own injection),
a correction fitted on other operating points can be subtracted from GENCO's physical state
(p.V_from with the true setpoints) and used as a better Newton start.

For every scenario the script computes the signed errors of GENCO's physical state against datakit's solution,
dVm at PQ buses and dVa at non-reference buses, and builds corrected starts

    unc      the uncorrected physical state (V_from, setpoints enforced, as in exp02)
    bias     minus the per-bus mean signed error (shrunk toward zero with a few pseudo-samples)
    lin      per-bus ridge regression of the signed error on [1, total load MW]
    inj      per-bus ridge regression on [1, total load MW, the bus's own net P and Q injection]
    <v>_x    the same corrections fitted on a SEPARATE set (--fit_*, distribution-shift test)
    bias_in  the per-bus mean fitted on the evaluation set itself (optimistic, state error only)

Corrections without suffix are 5-fold cross-fitted, folds grouped by LOAD scenario (a load scenario is never
in the fit and the evaluation fold together). PV/REF setpoints are re-enforced after the correction.

Per scenario and variant: state error (l2, inf-norm, Vm and Va parts), max power mismatch, Newton iterations to
1e-2 / 1e-4 / 1e-6 MW, branch loadings and PQ-bus voltages after 0/1/2 Newton steps (exp02 element tables,
so the equal-cost screening ladder uses analyze_exp02's Screen and decision_stats). The pre-contingency base
case (acpf.base_case_voltages) and base + 1/2 steps are included for reference. All intervals are 95% cluster
bootstraps over load scenarios.

Usage:
    python scripts/exp08_bias_correction.py --grid case118_ieee --size base \\
        --fit calib=data/calib/n1:splits_all \\
        --sets paper=data/pf:splits_paper_test stress100_n5=data/stress100/n5:splits_all
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from gridfm_graphkit.datasets.masking import RemovePFMask

from genco_certified import acpf
from genco_certified.loading import ROOT, load_genco
from genco_certified.tracing import trace_layers

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_exp02 import RawSet, Screen, cluster_bootstrap, decision_stats, fmt_ci, grouped_folds  # noqa: E402

OUT = ROOT / "results" / "exp08"
TOLS = {2: 1e-2, 4: 1e-4, 6: 1e-6}
KINDS = ("bias", "lin", "inj")
LAM_I, LAM_S = 3.0, 5.0  # ridge pseudo-samples: intercept (shrinks the bias toward 0) and slopes (z-scored features)
SD_FLOOR, Z_CLIP = 1.0, 4.0  # feature scale floor (MW) and clip of the z-scored features (guards against extrapolation)


# ---------------------------------------------------------------------------------------------------
# data collection
# ---------------------------------------------------------------------------------------------------

def parse_spec(spec):
    name, rest = spec.split("=", 1)
    root, split = rest.rsplit(":", 1)
    return name, root, split


def collect(grid, size, seed, data_root, split, with_base, keep_p, max_scen=0):
    """Run GENCO over a split. Returns a list of records: p (if keep_p), Vm, Va, signed errors and features,
    and the base-case voltages (if with_base)."""
    Lg = load_genco("pf", grid, size, seed, batch_size=32, overrides={"task": {"task_name": "PowerFlow"}},
                    data_root=data_root, split_dir=(ROOT / (data_root or "data/pf")) / grid / split)
    model = Lg.task.model
    root = ROOT / (data_root or "data/pf") / grid
    test_ids = torch.load(root / split / "test.pt").tolist()
    raw = {t: pd.read_parquet(root / "raw" / f"{t}_data.parquet") for t in ["bus", "branch", "gen"]}
    load_of = raw["bus"].drop_duplicates("scenario").set_index("scenario").load_scenario_idx
    if with_base:
        loads = raw["bus"].loc[raw["bus"].scenario.isin(test_ids), "load_scenario_idx"].unique()
        raw = {t: d[d.load_scenario_idx.isin(loads)] for t, d in raw.items()}
    base_cache = {}

    def base_case(li):
        if li not in base_cache:
            sel = {t: d[d.load_scenario_idx == li] for t, d in raw.items()}
            base_cache[li] = acpf.base_case_voltages(sel["bus"], sel["branch"], sel["gen"])
        return base_cache[li]

    recs, sid = [], 0
    for batch in Lg.test_loader():
        with torch.no_grad(), trace_layers(model) as tr:
            model(batch)
        state = tr["state"][-1].double().numpy()
        Lg.normalizer.inverse_transform(batch)
        RemovePFMask()(batch)
        ptr = batch["bus"].ptr.numpy()
        for g, p in enumerate(acpf.extract_problems(batch)):
            s = state[ptr[g]:ptr[g + 1]]
            li = int(load_of.loc[test_ids[sid]])
            rec = dict(sid=sid, raw_id=test_ids[sid], load_idx=li, Vm=s[:, 0].copy(), Va=s[:, 1].copy(),
                       L=float(p.meta["load_MW"]), P=p.Sbus.real.copy(), Q=p.Sbus.imag.copy())
            rec["eVm"], rec["eVa"] = signed_errors(p, rec["Vm"], rec["Va"])
            if with_base:
                bc = base_case(li)
                rec["Vb"] = p.V_from(np.abs(bc["V"]), np.angle(bc["V"])) if bc["converged"] else p.flat_start()
            if keep_p:
                rec["p"] = p
            recs.append(rec)
            sid += 1
            if max_scen and sid >= max_scen:
                return recs
    return recs


def signed_errors(p, Vm, Va):
    """Signed error of the physical state: dVm at PQ buses, dVa (wrapped) at non-REF buses; NaN elsewhere."""
    V = p.V_from(Vm, Va)
    n = len(Vm)
    eVm, eVa = np.full(n, np.nan), np.full(n, np.nan)
    eVm[p.pq] = np.abs(V[p.pq]) - np.abs(p.V_true[p.pq])
    nr = np.setdiff1d(np.arange(n), p.ref)
    eVa[nr] = np.angle(V[nr] * np.conj(p.V_true[nr]))
    return eVm, eVa


# ---------------------------------------------------------------------------------------------------
# per-bus corrector
# ---------------------------------------------------------------------------------------------------

class Corrector:
    """Per-bus ridge regression of the signed error on global/own features; kind in {bias, lin, inj}."""

    def __init__(self, kind):
        self.kind = kind

    def _feats(self, L, P, Q):
        """L scalar-or-[m], P, Q [m, nb] -> [m, nb, k-1] raw slope features."""
        L = np.broadcast_to(np.asarray(L, float).reshape(-1, 1), P.shape)
        if self.kind == "bias":
            return np.zeros(P.shape + (0,))
        if self.kind == "lin":
            return L[..., None]
        return np.stack([L, P, Q], axis=-1)

    def fit(self, recs):
        E = {"vm": np.array([r["eVm"] for r in recs]), "va": np.array([r["eVa"] for r in recs])}
        L = np.array([r["L"] for r in recs])
        P, Q = np.array([r["P"] for r in recs]), np.array([r["Q"] for r in recs])
        F = self._feats(L, P, Q)
        nb, k = E["vm"].shape[1], F.shape[-1]
        self.par = {}
        for name, e in E.items():
            mu, sd = np.zeros((nb, k)), np.ones((nb, k))
            w = np.zeros((nb, k + 1))
            for b in range(nb):
                m = np.isfinite(e[:, b])
                if m.sum() < 3:
                    continue
                Fb = F[m, b, :]
                if k:
                    mu[b], sd[b] = Fb.mean(0), np.maximum(Fb.std(0), SD_FLOOR)
                X = np.c_[np.ones(m.sum()), (Fb - mu[b]) / sd[b]]
                pen = np.diag([LAM_I] + [LAM_S] * k)
                w[b] = np.linalg.solve(X.T @ X + pen, X.T @ e[m, b])
            self.par[name] = (mu, sd, w)
        return self

    def predict(self, L, P, Q):
        """L scalar, P, Q [nb] (one scenario) -> (dVm, dVa) estimated signed errors per bus."""
        F = self._feats(L, P[None], Q[None])[0]  # [nb, k-1]
        out = []
        for name in ("vm", "va"):
            mu, sd, w = self.par[name]
            z = np.clip((F - mu) / sd, -Z_CLIP, Z_CLIP) if F.shape[-1] else F
            out.append(w[:, 0] + (w[:, 1:] * z).sum(1))
        return out[0], out[1]


def crossfit(recs, kind, K, seed=0):
    """Out-of-fold (dVm, dVa) estimates per record, folds grouped by load scenario."""
    fold = grouped_folds([r["load_idx"] for r in recs], K, seed)
    est = [None] * len(recs)
    for f in range(fold.max() + 1):
        tr_ = [r for r, fo in zip(recs, fold) if fo != f]
        c = Corrector(kind).fit(tr_)
        for i, (r, fo) in enumerate(zip(recs, fold)):
            if fo == f:
                est[i] = c.predict(r["L"], r["P"], r["Q"])
    return est


# ---------------------------------------------------------------------------------------------------
# evaluation of one set
# ---------------------------------------------------------------------------------------------------

def v_corrected(p, rec, est):
    dvm, dva = est
    return p.V_from(rec["Vm"] - np.nan_to_num(dvm), rec["Va"] - np.nan_to_num(dva))


def evaluate_set(name, recs, args, ext=None):
    """recs with p. ext: dict kind -> Corrector fitted on a separate set (or None). Writes exp02-style parquet files."""
    K = args.folds
    t0 = time.time()
    est = {k: crossfit(recs, k, K) for k in KINDS}
    est_in = Corrector("bias").fit(recs)
    est_x = {}
    if ext:
        for k, c in ext.items():
            est_x[k + "_x"] = [c.predict(r["L"], r["P"], r["Q"]) for r in recs]
    variants = ["unc"] + list(KINDS) + list(est_x)
    tag = f"{name}_{args.size}_seed{args.seed}"
    br_rows, bu_rows, sc_rows = [], [], []
    for i, r in enumerate(recs):
        p = r["p"]
        starts = {"unc": p.V_from(r["Vm"], r["Va"])}
        for k in KINDS:
            starts[k] = v_corrected(p, r, est[k][i])
        for k, e in est_x.items():
            starts[k] = v_corrected(p, r, e[i])
        starts["base"] = r["Vb"]
        V = {"true": p.V_true, "phys": starts["unc"]}
        row = dict(scenario=r["sid"], raw_id=r["raw_id"], load_idx=r["load_idx"], load_MW=r["L"])
        for v, V0 in starts.items():
            tj = acpf.newton_trajectory(p, V0, n_steps=2)
            e = acpf.state_error(p, V0)
            nva = len(p.pvpq)
            row[f"err_l2_{v}"] = float(np.linalg.norm(e))
            row[f"err_inf_{v}"] = float(np.abs(e).max())
            row[f"err_va_inf_{v}"] = float(np.abs(e[:nva]).max())
            row[f"err_vm_inf_{v}"] = float(np.abs(e[nva:]).max()) if len(e) > nva else 0.0
            row[f"err_vm_l2_{v}"] = float(np.linalg.norm(e[nva:]))
            row[f"err_va_l2_{v}"] = float(np.linalg.norm(e[:nva]))
            row[f"res_{v}"] = float(np.abs(acpf.mismatch(p, V0)).max())
            row[f"step1_l2_{v}"] = float(np.linalg.norm(tj["dx"][0]))
            for ee, tol in TOLS.items():
                n_it = acpf.iterations_to_tol(tj["history"], tol)
                row[f"nri_{v}_e{ee}"] = float(n_it) if n_it >= 0 else float(tj["iterations"])
            pre = {"unc": "g", "base": "b"}.get(v, v + "_")
            if v != "unc" and v != "base":
                V[f"{v}_0"] = V0
            for s_ in (1, 2):
                V[f"{pre}{s_}"] = tj["V"][s_ - 1]
        e_in = acpf.state_error(p, v_corrected(p, r, est_in.predict(r["L"], r["P"], r["Q"])))
        row["err_l2_bias_in"] = float(np.linalg.norm(e_in))
        row["err_inf_bias_in"] = float(np.abs(e_in).max())
        loads = {k: acpf.branch_loading(p, acpf.branch_flows(p, v)) for k, v in V.items()}
        nb = len(loads["true"])
        br_rows.append(pd.DataFrame({"scenario": r["sid"], "branch": np.arange(nb), **{f"load_{k}": v for k, v in loads.items()}}))
        btype = np.zeros(p.meta["n_bus"], dtype=np.int8)
        btype[p.pv], btype[p.ref] = 1, 2
        bu_rows.append(pd.DataFrame({"scenario": r["sid"], "bus": np.arange(p.meta["n_bus"]), "btype": btype, "vmin": p.vmin, "vmax": p.vmax,
                                     **{f"vm_{k}": np.abs(v) for k, v in V.items()}}))
        sc_rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(br_rows, ignore_index=True).to_parquet(OUT / f"{tag}_branches.parquet")
    pd.concat(bu_rows, ignore_index=True).to_parquet(OUT / f"{tag}_buses.parquet")
    sc = pd.DataFrame(sc_rows)
    sc.to_parquet(OUT / f"{tag}_scenarios.parquet")
    print(f"[{tag}] evaluated {len(sc)} scenarios in {time.time() - t0:.0f}s", flush=True)
    return tag, variants


# ---------------------------------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------------------------------

def boot_stat(fn, x, groups, B=1000, seed=0):
    """Cluster bootstrap of an arbitrary statistic fn(array) over load scenarios -> (point, lo, hi)."""
    x, g = np.asarray(x, float), np.asarray(groups)
    u, code = np.unique(g, return_inverse=True)
    members = [np.flatnonzero(code == j) for j in range(len(u))]
    rng = np.random.default_rng(seed)
    st = []
    for _ in range(B):
        pick = rng.integers(0, len(u), len(u))
        st.append(fn(x[np.concatenate([members[j] for j in pick])]))
    lo, hi = np.quantile(st, [0.025, 0.975])
    return float(fn(x)), float(lo), float(hi)


def report(tag, variants, B, lines):
    P = lines.append
    d = OUT
    sc = pd.read_parquet(d / f"{tag}_scenarios.parquet")
    S = RawSet(tag, d)
    scr = Screen(S, 1.01, 1e-3)
    g = sc.load_idx.to_numpy()
    allv = ["unc"] + [v for v in variants if v != "unc"]
    P(f"\n{'=' * 110}\n{tag}: {len(sc)} scenarios, {sc.load_idx.nunique()} load scenarios")
    # --- 1. state error
    P("\n1. STATE ERROR of the start (x = [Va at PV+PQ, Vm at PQ]; rad / p.u.). SSE reduction = 1 - sum(err_after^2)/sum(err_unc^2) with cluster CI;"
      " 'better' = share of scenarios with smaller l2 than the uncorrected start. bias_in is fitted on the evaluation set itself (optimistic).")
    rows = []
    for v in allv + ["bias_in", "base"]:
        if f"err_l2_{v}" not in sc:
            continue
        l2 = sc[f"err_l2_{v}"].to_numpy()
        m = boot_stat(np.median, l2, g, B)
        r = dict(start=v, median_l2=fmt_ci(*m, nd=5), p95_l2=f"{np.quantile(l2, .95):.5f}", median_inf=f"{np.median(sc[f'err_inf_{v}']):.5f}",
                 max_inf=f"{sc[f'err_inf_{v}'].max():.4f}")
        if v != "unc":
            num, den = l2 ** 2, sc["err_l2_unc"].to_numpy() ** 2
            red = cluster_bootstrap(num, den=den, groups=g, B=B)
            r["SSE_reduction"] = fmt_ci(1 - red[0], 1 - red[2], 1 - red[1], 3)
            r["better"] = f"{np.mean(l2 < sc['err_l2_unc'].to_numpy()):.0%}"
        rows.append(r)
    P(pd.DataFrame(rows).to_string(index=False))
    P("   split by component, median l2 of the Vm part / Va part (p.u., rad) and SSE reduction of each part:")
    rows = []
    for v in allv:
        r = dict(start=v)
        for comp in ("vm", "va"):
            x = sc[f"err_{comp}_l2_{v}"].to_numpy()
            r[f"median_{comp}_l2"] = f"{np.median(x):.5f}"
            if v != "unc":
                red = cluster_bootstrap(x ** 2, den=sc[f"err_{comp}_l2_unc"].to_numpy() ** 2, groups=g, B=B)
                r[f"{comp}_SSE_red"] = fmt_ci(1 - red[0], 1 - red[2], 1 - red[1], 3)
        rows.append(r)
    P(pd.DataFrame(rows).to_string(index=False))
    # --- 2. NR iterations
    P("\n2. NEWTON-RAPHSON ITERATIONS to 1e-2 / 1e-4 / 1e-6 MW, mean [95% cluster CI]; paired difference to the uncorrected GENCO start and to the base-case start")
    rows = []
    for ee, tol in TOLS.items():
        for v in allv + ["base"]:
            x = sc[f"nri_{v}_e{ee}"].to_numpy()
            r = dict(tol=f"{tol:g}", start=v, iters=fmt_ci(*cluster_bootstrap(x, groups=g, B=B), nd=3))
            if v != "unc":
                r["minus_unc"] = fmt_ci(*cluster_bootstrap(x - sc[f"nri_unc_e{ee}"].to_numpy(), groups=g, B=B), nd=3)
            if v != "base":
                r["minus_base"] = fmt_ci(*cluster_bootstrap(x - sc[f"nri_base_e{ee}"].to_numpy(), groups=g, B=B), nd=3)
            rows.append(r)
    P(pd.DataFrame(rows).fillna("").to_string(index=False))
    # --- 3. ladder
    P("\n3. EQUAL-COST SCREENING (tau = 1.01, eps_v = 1 mV; branches + PQ buses): violating elements missed / false alarms after 0, 1, 2 Newton steps;"
      " contingency-level misses; element miss rate with 95% CI and one-sided Clopper-Pearson upper bound")
    spec = [("unc, 0 steps (V_from)", "phys", 0)]
    for v in allv:
        if v == "unc":
            spec += [("unc + 1 step", "g1", 1), ("unc + 2 steps", "g2", 2)]
        else:
            spec += [(f"{v}, 0 steps", f"{v}_0", 0), (f"{v} + 1 step", f"{v}_1", 1), (f"{v} + 2 steps", f"{v}_2", 2)]
    spec += [("base case + 1 step", "b1", 1), ("base case + 2 steps", "b2", 2)]
    rows = []
    for name, m, lu in spec:
        if not S.has(m):
            continue
        st = decision_stats(S, scr, scr.counts(m), B)
        rows.append(dict(method=name, LU=lu, el_miss=f"{st['el_miss']}/{st['el_truth']}", el_fa=st["el_fa"],
                         miss_rate=fmt_ci(st["el_miss_rate"], st["el_miss_rate_lo"], st["el_miss_rate_hi"], 4), ub1=f"{st['el_cp_hi1']:.2%}",
                         cont_miss=f"{st['cont_miss']}/{st['cont_truth']}", cont_fa=st["cont_fa"], load_miss=f"{st['load_miss']}/{st['n_loads']}"))
    P(pd.DataFrame(rows).to_string(index=False))
    pd.DataFrame(rows).to_csv(d / f"{tag}_ladder.csv", index=False)


def diagnostics(recs, K, lines, name):
    """How systematic are the signed errors? Everything is about the signed errors e = V_phys - V_true of the fit set."""
    P = lines.append
    eVm = np.array([r["eVm"] for r in recs])
    eVa = np.array([r["eVa"] for r in recs])
    loads = np.array([r["load_idx"] for r in recs])
    ul = np.unique(loads)
    fold = grouped_folds(loads, K, 0)
    P(f"\nDIAGNOSTICS of the signed errors on {name}: {len(recs)} scenarios, {len(ul)} load scenarios")
    for comp, E, unit in (("Vm (PQ buses)", eVm, "p.u."), ("Va (non-REF buses)", eVa, "rad")):
        ok = np.isfinite(E)
        n_b = ok.sum(0)
        mean_b = np.where(n_b > 0, np.nansum(E, 0) / np.maximum(n_b, 1), 0.0)
        rms_b = np.sqrt(np.where(n_b > 0, np.nansum(E ** 2, 0) / np.maximum(n_b, 1), 0.0))
        sse = np.nansum(E ** 2)
        # (1) share of the total squared signed error that a per-bus constant could remove (in-sample, no shrinkage)
        share_mean = float(np.sum(n_b * mean_b ** 2) / sse)
        # (2) the same at the level of load scenarios: mean error of a bus within a load scenario, t-statistic across load scenarios
        tstat = []
        for b in range(E.shape[1]):
            if n_b[b] < 3 * len(ul) / 5:
                continue
            lm = np.array([np.nanmean(E[loads == l, b]) for l in ul if np.isfinite(E[loads == l, b]).any()])
            if len(lm) > 5 and lm.std(ddof=1) > 0:
                tstat.append(lm.mean() / (lm.std(ddof=1) / np.sqrt(len(lm))))
        tstat = np.array(tstat)
        # (3) sign consistency: mean over buses of |fraction positive - 0.5|
        fpos = np.array([np.mean(E[ok[:, b], b] > 0) for b in range(E.shape[1]) if n_b[b] > 20])
        # (4) magnitude IS bus-specific: cross-fitted R^2 of |e| from the per-bus mean of |e| on other load scenarios
        A = np.abs(E)
        pred = np.full_like(A, np.nan)
        for f in range(fold.max() + 1):
            tr_, te_ = fold != f, fold == f
            mu = np.nanmean(np.where(tr_[:, None], A, np.nan), axis=0)
            pred[te_] = mu[None, :]
        m = np.isfinite(A) & np.isfinite(pred)
        r2_mag = 1 - np.sum((A[m] - pred[m]) ** 2) / np.sum((A[m] - A[m].mean()) ** 2)
        P(f"  {comp}: overall RMS {np.sqrt(sse / ok.sum()):.5f} {unit}; RMS of the per-bus mean signed error {np.sqrt(np.sum(n_b * mean_b ** 2) / ok.sum()):.5f} {unit}")
        P(f"    per-bus constant removes {share_mean:.4f} of the total squared error (in-sample, upper bound for the bias correction)")
        P(f"    |t| of the per-bus mean across load scenarios: median {np.median(np.abs(tstat)):.2f}, share of buses with |t| > 3: {np.mean(np.abs(tstat) > 3):.2f} "
          f"(n={len(tstat)} buses; noise alone gives median 0.67, share 0.003)")
        P(f"    sign consistency: mean |P(e>0) - 0.5| over buses = {np.mean(np.abs(fpos - 0.5)):.3f}; share of buses with P(e>0) outside [0.4, 0.6] = {np.mean((fpos < 0.4) | (fpos > 0.6)):.2f}")
        P(f"    magnitude |e| is bus-specific: cross-fitted R^2 of |e| from the per-bus mean |e| (other load scenarios) = {r2_mag:.3f}")
    # (5) how much of the Va error is a common-mode angle shift of the whole scenario?
    sse_tot = sse_cm = 0.0
    for r in recs:
        e = r["eVa"][np.isfinite(r["eVa"])]
        sse_tot += float((e ** 2).sum())
        sse_cm += float(((e - e.mean()) ** 2).sum())
    P(f"  Va: removing one common-mode angle offset per scenario (needs the truth, not a usable correction) removes {1 - sse_cm / sse_tot:.3f} of the squared Va error")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="case118_ieee")
    ap.add_argument("--size", default="base")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sets", nargs="+", default=["paper=data/pf:splits_paper_test"], help="evaluation sets: name=data_root:split")
    ap.add_argument("--fit", default=None, help="separate set to fit the corrections on (shift test): name=data_root:split")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--max_scen", type=int, default=0, help="debug: only the first N scenarios of every set")
    ap.add_argument("--report_only", action="store_true")
    ap.add_argument("--diag_only", action="store_true", help="only collect the --fit set and print how systematic its signed errors are")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()
    lines = [f"exp08 bias correction: {args.grid} {args.size} seed {args.seed}; {args.folds}-fold cross-fitting grouped by load scenario; "
             f"ridge pseudo-samples intercept {LAM_I}, slopes {LAM_S}; {args.boot} cluster-bootstrap resamples"]
    ext, fit_recs = None, None
    if args.fit and not args.report_only:
        fname, froot, fsplit = parse_spec(args.fit)
        t0 = time.time()
        fit_recs = collect(args.grid, args.size, args.seed, froot, fsplit, with_base=False, keep_p=False, max_scen=args.max_scen)
        ext = {k: Corrector(k).fit(fit_recs) for k in KINDS}
        print(f"[fit set {fname}] {len(fit_recs)} scenarios, {len({r['load_idx'] for r in fit_recs})} load scenarios in {time.time() - t0:.0f}s", flush=True)
        lines.append(f"corrections '_x' fitted on {fname} ({froot}/{fsplit}): {len(fit_recs)} scenarios, {len({r['load_idx'] for r in fit_recs})} load scenarios")
        # how systematic is the error on the fit set itself?
        fold_est = {k: crossfit(fit_recs, k, args.folds) for k in KINDS}
        for k in KINDS:
            sse0 = sse1 = 0.0
            for r, e in zip(fit_recs, fold_est[k]):
                for key, est in (("eVm", e[0]), ("eVa", e[1])):
                    m = np.isfinite(r[key])
                    sse0 += float((r[key][m] ** 2).sum())
                    sse1 += float(((r[key] - est)[m] ** 2).sum())
            lines.append(f"  fit set {fname}: cross-fitted SSE reduction ({k}) over all (scenario, bus) signed errors = {1 - sse1 / sse0:.3f}")
    if fit_recs is not None:
        diagnostics(fit_recs, args.folds, lines, fname)
    if args.diag_only:
        txt = "\n".join(lines)
        print(txt)
        (OUT / f"diagnostics_{args.size}_seed{args.seed}.txt").write_text(txt + "\n")
        return
    tags = []
    for spec in args.sets:
        name, root, split = parse_spec(spec)
        tag = f"{name}_{args.size}_seed{args.seed}"
        if not args.report_only:
            recs = collect(args.grid, args.size, args.seed, root, split, with_base=True, keep_p=True, max_scen=args.max_scen)
            _, variants = evaluate_set(name, recs, args, ext)
            json.dump(variants, open(OUT / f"{tag}_variants.json", "w"))
            del recs
        variants = json.load(open(OUT / f"{tag}_variants.json"))
        tags.append((tag, variants))
    for tag, variants in tags:
        report(tag, variants, args.boot, lines)
    txt = "\n".join(lines)
    print(txt)
    out = Path(args.log) if args.log else OUT / f"report_{args.size}_seed{args.seed}.txt"
    out.write_text(txt + "\n")


if __name__ == "__main__":
    main()
