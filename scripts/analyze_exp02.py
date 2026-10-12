"""Screening-policy analysis on exp02 outputs (no model re-runs).

Policy family (all cost in sparse LU factorizations, one per Newton step):
  tiered    after ONE Newton step from a prior, accept when every element margin exceeds c * ||dx||^2
            (Newton's quadratic-convergence error model); otherwise finish Newton-Raphson (tier 2).
  sequential  same test after every step k = 1, 2, 3; take another step until it passes.
Priors: GENCO (phys, slack setpoint enforced) and the pre-contingency base case ("classical").

Cost accounting (fixed 2026-10-11): an accepted scenario costs the steps it took; an escalated scenario
costs its OWN Newton-Raphson iteration count (the first step is the first NR iteration), never the
all-scenario mean. Tier-2 decisions come from our own converged NR, not from datakit's stored solution.

Calibration of c (the unit of exchangeability is the LOAD SCENARIO, never the single contingency):
  score of a load scenario = max over its scenarios and PQ-bus/branch elements of |error| / ||dx||^2
  rule 'max'      c = largest calibration score                      (the old rule)
  rule 'conf<a>'  split-conformal: c = ceil((n+1)(1-a))-th smallest score of n calibration load scenarios
  scheme 'kfold'  K-fold cross-fitting grouped by load_idx on one set, pooled out-of-fold results
  scheme 'cross'  calibrate on a different set (default: the paper set) and evaluate on all of another

Usage:
    python scripts/analyze_exp02.py --v3                 # full corrected analysis -> results/policy_v3.log + CSVs
    python scripts/analyze_exp02.py --tag stress_n2_base --cal_tag case118_ieee_base_seed0 --tau 1.01
                                                         # legacy single-set table (format used by make_figures)
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from genco_certified.loading import ROOT

D = ROOT / "results" / "exp02"
PAPER = "case118_ieee_base_seed0"
STRESS = ["stress_n1_base", "stress_n2_base", "stress_n3_base", "stress_n5_base"]
GOC = "case2000_goc_small_seed0"
ALL_SETS = [PAPER] + STRESS + [GOC]
LABEL = {PAPER: "paper N-1", "stress_n1_base": "stress N-1", "stress_n2_base": "stress N-2",
         "stress_n3_base": "stress N-3", "stress_n5_base": "stress N-5", GOC: "GOC 2000"}
TAU, EPS_V = 1.01, 1e-3
TAUS = [1.00, 1.005, 1.01, 1.02]
EPSS = [0.0, 0.5e-3, 1e-3, 2e-3]
PRIORS = {"g": "GENCO", "b": "base case"}
SEQ_STEPS = (1, 2, 3)
CNT = ["tp_b", "fp_b", "fn_b", "tp_v", "fp_v", "fn_v"]
ALIAS = {"g1": "step1", "b1": "base1"}  # names used by the pre-v3 parquet files


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested in tests/test_screening.py)
# ---------------------------------------------------------------------------

def conformal_quantile(scores, alpha: float) -> float:
    """Split-conformal threshold: the ceil((n+1)(1-alpha))-th smallest of n calibration scores.

    inf when that rank exceeds n (too few calibration points for the requested alpha, or n == 0);
    the smallest score when the rank would be below 1 (alpha >= 1). Scores must be finite or +inf.
    """
    s = np.sort(np.asarray(scores, dtype=float))
    if np.isnan(s).any():
        raise ValueError("NaN calibration score")
    n = len(s)
    if n == 0:
        return math.inf
    k = math.ceil((n + 1) * (1.0 - alpha) - 1e-12)
    if k > n:
        return math.inf
    return float(s[max(k, 1) - 1])


def choose_c(scores, rule: str) -> float:
    """c from per-load-scenario calibration scores: 'max' or 'conf<alpha>' (e.g. 'conf0.10')."""
    if rule == "max":
        return float(np.max(scores)) if len(scores) else math.inf
    if rule.startswith("conf"):
        return conformal_quantile(scores, float(rule[4:]))
    raise ValueError(rule)


def grouped_folds(groups, k: int, seed: int = 0) -> np.ndarray:
    """Fold id (0..k'-1) per row such that all rows of a group share a fold (k' = min(k, #groups)).

    Groups are shuffled and dealt round-robin, so fold sizes differ by at most one group.
    """
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    k = min(k, len(uniq))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    fold_of = np.empty(len(uniq), dtype=int)
    fold_of[perm] = np.arange(len(uniq)) % k
    return fold_of[np.searchsorted(uniq, groups)]


def lu_per_scenario(accepted_step, nr_iters, last_step=1):
    """LU per scenario. accepted_step[i] = Newton steps taken when scenario i was accepted (0 = escalated).

    Accepted scenarios cost the steps they took. Escalated scenarios cost their OWN NR iteration count
    (steps already taken are the first NR iterations), but never less than the steps already spent
    (`last_step`, the final verification step attempted).
    """
    acc = np.asarray(accepted_step)
    return np.where(acc > 0, acc, np.maximum(np.asarray(nr_iters, dtype=float), last_step))


def tiered_lu(tier1_mask, nr_iters):
    """Mean LU of the one-step tiered policy: 1 if accepted after step 1, else the scenario's own NR count."""
    t1 = np.asarray(tier1_mask, dtype=bool)
    return float(lu_per_scenario(np.where(t1, 1, 0), nr_iters, 1).mean())


def tune_threshold(load, truth, beta: float = 3.0, recall_target: float = 0.99, grid=None) -> dict:
    """Pick a loading threshold on a calibration set: max F-beta, and the largest one with recall >= target."""
    load, truth = np.asarray(load, float), np.asarray(truth, bool)
    th = np.arange(0.30, 1.50, 0.0025) if grid is None else np.asarray(grid)
    pos, neg = np.sort(load[truth]), np.sort(load[~truth])
    tp = len(pos) - np.searchsorted(pos, th, side="right")
    fp = len(neg) - np.searchsorted(neg, th, side="right")
    fn = len(pos) - tp
    b2 = beta ** 2
    fb = (1 + b2) * tp / np.maximum((1 + b2) * tp + b2 * fn + fp, 1e-12)
    rec = tp / max(len(pos), 1)
    i_f = len(th) - 1 - int(np.argmax(fb[::-1]))  # last maximiser (fewest flags)
    ok = np.flatnonzero(rec >= recall_target)
    i_r = int(ok[-1]) if len(ok) else 0
    return {"fbeta": float(th[i_f]), "recall": float(th[i_r])}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class RawSet:
    """One exp02 run: per-scenario table plus element tables (branches; PQ buses only)."""

    def __init__(self, tag, d=None):
        d = Path(d) if d else D
        self.tag = tag
        br = pd.read_parquet(d / f"{tag}_branches.parquet")
        bu = pd.read_parquet(d / f"{tag}_buses.parquet")
        # PV/REF voltages are setpoints the operator already knows: screen PQ buses only.
        self.bu = bu[bu.btype == 0].reset_index(drop=True)
        self.br = br
        sc = pd.read_parquet(d / f"{tag}_scenarios.parquet").set_index("scenario").sort_index()
        self.sc = sc
        self.n = len(sc)
        assert (sc.index.to_numpy() == np.arange(self.n)).all()
        for t in (self.br, self.bu):
            assert (np.diff(t.scenario.to_numpy()) >= 0).all(), "element rows must be sorted by scenario"
        self.cb = np.bincount(self.br.scenario.to_numpy(), minlength=self.n)
        self.cv = np.bincount(self.bu.scenario.to_numpy(), minlength=self.n)
        assert (self.cb > 0).all() and (self.cv > 0).all()
        self.sb = np.r_[0, np.cumsum(self.cb)[:-1]]
        self.sv = np.r_[0, np.cumsum(self.cv)[:-1]]
        self.load_idx = sc.load_idx.to_numpy()
        self.n_loads = len(np.unique(self.load_idx))
        self._ratio_cache = {}
        self.ref = "true"

    # element columns -------------------------------------------------------
    def _col(self, df, kind, m):
        for name in (f"{kind}_{m}", f"{kind}_{ALIAS.get(m, m)}"):
            if name in df.columns:
                return df[name].to_numpy()
        raise KeyError(f"{self.tag}: no {kind}_{m}")

    def has(self, m):
        return any(c in self.br.columns for c in (f"load_{m}", f"load_{ALIAS.get(m, m)}"))

    def L(self, m):
        return self._col(self.br, "load", m)

    def Vm(self, m):
        return self._col(self.bu, "vm", m)

    # scenario columns ------------------------------------------------------
    def dsq(self, prior, k):
        """||dx_k||^2 of the k-th Newton step from the prior (g = GENCO, b = base case)."""
        sc = self.sc
        if f"step_{prior}{k}_l2" in sc.columns:
            return sc[f"step_{prior}{k}_l2"].to_numpy() ** 2
        assert k == 1, "pre-v3 data only has the first step"
        return sc["step_l2" if prior == "g" else "step_base_l2"].to_numpy() ** 2

    def nri(self, start, e=6):
        """NR iterations to 1e-<e> MW from start in {flat, base, genco, dc}."""
        col = f"nri_{start}_e{e}"
        if col in self.sc.columns:
            v = self.sc[col].to_numpy().astype(float)
            if np.isnan(v).any():  # never reached the tolerance: charge the iterations actually taken
                v = np.where(np.isnan(v), self.sc[f"nr_iters_{start}"].to_numpy(), v)
            return v
        assert e == 6
        return self.sc[f"nr_iters_{start}"].to_numpy().astype(float)

    def per_load_max(self, scen_scores):
        """Max of a per-scenario score within each load scenario -> (sorted load ids, scores)."""
        ls = np.unique(self.load_idx)
        out = np.full(len(ls), -np.inf)
        np.maximum.at(out, np.searchsorted(ls, self.load_idx), scen_scores)
        return ls, out

    def ratio_scores(self, prior, k=1, ref=None, floor=1e-12, err_min=0.0):
        ref = ref or self.ref  # 'true' = datakit's stored solution (labels); 'nr' = our converged NR (no label noise)
        key = (prior, k, ref, floor, err_min)
        if key not in self._ratio_cache:
            self._ratio_cache[key] = self._ratio_scores(prior, k, ref, floor, err_min)
        return self._ratio_cache[key]

    def _ratio_scores(self, prior, k, ref, floor, err_min):
        """Per scenario: max over branch and PQ-bus elements of |error| / ||dx_k||^2 (zero-error elements ignored).

        The error is measured against `ref` (default datakit's solution) on the iterate after k steps.
        """
        m = f"{prior}{k}"
        d = np.maximum(self.dsq(prior, k), floor)
        eb = np.abs(self.L(m) - self.L(ref))
        ev = np.abs(self.Vm(m) - self.Vm(ref))
        rb = np.where(eb > err_min, eb / np.repeat(d, self.cb), 0.0)
        rv = np.where(ev > err_min, ev / np.repeat(d, self.cv), 0.0)
        return np.maximum(np.maximum.reduceat(rb, self.sb), np.maximum.reduceat(rv, self.sv))


class Screen:
    """Per-scenario decision counts and verification margins of a RawSet at thresholds (tau, eps_v)."""

    def __init__(self, S: RawSet, tau=TAU, eps_v=EPS_V):
        self.S, self.tau, self.eps = S, tau, eps_v
        self.vmin = S.bu.vmin.to_numpy() - eps_v
        self.vmax = S.bu.vmax.to_numpy() + eps_v
        self.tb = S.L("true") > tau
        vt = S.Vm("true")
        self.tv = (vt < self.vmin) | (vt > self.vmax)
        self.ntruth_b = self._sum_b(self.tb)
        self.ntruth_v = self._sum_v(self.tv)
        self.ntruth = self.ntruth_b + self.ntruth_v
        self.truth_any = self.ntruth > 0
        self._cnt, self._mm = {}, {}

    def _sum_b(self, x):
        return np.add.reduceat(x.astype(np.int64), self.S.sb)

    def _sum_v(self, x):
        return np.add.reduceat(x.astype(np.int64), self.S.sv)

    def pred_counts(self, pred_b, pred_v=None):
        """(n, 6) counts [tp_b, fp_b, fn_b, tp_v, fp_v, fn_v] of boolean element predictions."""
        pv = np.zeros_like(self.tv) if pred_v is None else pred_v
        return np.stack([
            self._sum_b(pred_b & self.tb), self._sum_b(pred_b & ~self.tb), self._sum_b(~pred_b & self.tb),
            self._sum_v(pv & self.tv), self._sum_v(pv & ~self.tv), self._sum_v(~pv & self.tv)], axis=1)

    def counts(self, m):
        if m not in self._cnt:
            vm = self.S.Vm(m)
            self._cnt[m] = self.pred_counts(self.S.L(m) > self.tau, (vm < self.vmin) | (vm > self.vmax))
        return self._cnt[m]

    def mmin(self, m):
        """Smallest |distance to the violation boundary| over all branch and PQ-bus elements, per scenario."""
        if m not in self._mm:
            mb = np.abs(self.tau - self.S.L(m))
            vm = self.S.Vm(m)
            mv = np.abs(np.minimum(vm - self.vmin, self.vmax - vm))
            self._mm[m] = np.minimum(np.minimum.reduceat(mb, self.S.sb), np.minimum.reduceat(mv, self.S.sv))
        return self._mm[m]


@dataclass
class Outcome:
    cnt: np.ndarray      # (n, 6) decision counts of the answer actually used per scenario
    cost: np.ndarray     # LU per scenario
    acc: np.ndarray      # Newton step at which the scenario was accepted (0 = escalated to NR)


def run_policy(scr: Screen, prior: str, ks, c, esc_tol: int = 6) -> Outcome:
    """Verification policy: after each step k in ks accept if min margin > c * ||dx_k||^2; else escalate to NR."""
    S = scr.S
    c = np.broadcast_to(np.asarray(c, dtype=float), (S.n,))
    acc = np.zeros(S.n, dtype=int)
    for k in ks:
        with np.errstate(invalid="ignore"):
            ok = (acc == 0) & (scr.mmin(f"{prior}{k}") > c * S.dsq(prior, k))
        acc[ok] = k
    ref = "nr" if S.has("nr") else "true"
    cnt = scr.counts(ref).copy()
    for k in ks:
        sel = acc == k
        cnt[sel] = scr.counts(f"{prior}{k}")[sel]
    start = "genco" if prior == "g" else "base"
    return Outcome(cnt, lu_per_scenario(acc, S.nri(start, esc_tol), max(ks)), acc)


def summarize(scr: Screen, cnt, cost=None, acc=None, mask=None) -> dict:
    m = np.ones(scr.S.n, bool) if mask is None else mask
    tp = (cnt[:, 0] + cnt[:, 3])[m].sum()
    fp = (cnt[:, 1] + cnt[:, 4])[m].sum()
    fn = (cnt[:, 2] + cnt[:, 5])[m].sum()
    pred_any = (cnt[:, 0] + cnt[:, 1] + cnt[:, 3] + cnt[:, 4]) > 0
    ta = scr.truth_any
    r = dict(n_scen=int(m.sum()), el_truth=int(tp + fn), el_miss=int(fn), el_fa=int(fp),
             recall=float(tp / max(tp + fn, 1)), precision=float(tp / max(tp + fp, 1)),
             cont_truth=int((ta & m).sum()), cont_miss=int((ta & ~pred_any & m).sum()),
             cont_fa=int((~ta & pred_any & m).sum()))
    if cost is not None:
        r["LU"] = float(cost[m].mean())
    if acc is not None:
        r["acc1"] = float((acc[m] == 1).mean())
        r["acc_any"] = float((acc[m] > 0).mean())
    return r


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def cross_c(Scal: RawSet, prior, rule):
    """c from ALL load scenarios of the calibration set (step-1 ratios of the given prior)."""
    _, sc_ls = Scal.per_load_max(Scal.ratio_scores(prior, 1))
    return choose_c(sc_ls, rule), len(sc_ls)


def kfold_c(S: RawSet, prior, rule, K=5, seed=0):
    """Out-of-fold c per scenario (fold-specific), fold id per scenario, and exceedance summaries.

    Folds are grouped by load scenario, so a load scenario is never in calibration and evaluation together.
    """
    scen = S.ratio_scores(prior, 1)
    ls, ls_score = S.per_load_max(scen)
    fold_s = grouped_folds(S.load_idx, K, seed)
    fold_l = np.empty(len(ls), dtype=int)
    fold_l[np.searchsorted(ls, S.load_idx)] = fold_s
    nf = int(fold_s.max()) + 1
    cf = np.array([choose_c(ls_score[fold_l != f], rule) for f in range(nf)])
    c_s = cf[fold_s]
    return c_s, fold_s, dict(c_folds=cf, exc_load=float(np.mean(ls_score > cf[fold_l])),
                             exc_scen=float(np.mean(scen > c_s)), n_loads=len(ls),
                             n_cal=float(np.mean([(fold_l != f).sum() for f in range(nf)])))


def exceedance(S: RawSet, prior, c):
    scen = S.ratio_scores(prior, 1)
    _, ls_score = S.per_load_max(scen)
    return float(np.mean(ls_score > c)), float(np.mean(scen > c))


# ---------------------------------------------------------------------------
# Legacy single-set table (format parsed by make_figures.py)
# ---------------------------------------------------------------------------

def legacy_main(args):
    S = RawSet(args.tag, args.dir)
    scr = Screen(S, args.tau, args.eps_v)
    has_base = S.has("base1")
    if args.cal_tag:
        Sc = RawSet(args.cal_tag, args.dir)
        c1, _ = cross_c(Sc, "g", "max")
        c1b = cross_c(Sc, "b", "max")[0] if has_base else None
        test = np.ones(S.n, bool)
        cal_desc = args.cal_tag
    else:  # random half/half split of LOAD scenarios (the old split was by contingency: leaky)
        ls = np.unique(S.load_idx)
        rng = np.random.default_rng(args.seed)
        cal_loads = rng.permutation(ls)[: len(ls) // 2]
        calm = np.isin(S.load_idx, cal_loads)
        test = ~calm
        sg = S.ratio_scores("g", 1)
        c1 = float(sg[calm].max())
        c1b = float(S.ratio_scores("b", 1)[calm].max()) if has_base else None
        cal_desc = "own half (grouped by load scenario)"
    ts = S.sc[test]
    t1 = run_policy(scr, "g", (1,), c1)
    print(f"{args.tag} (calibrated on {cal_desc}) | tau={args.tau} | test scenarios {test.sum()} | violations in test: "
          f"{scr.ntruth_b[test].sum()} branch + {scr.ntruth_v[test].sum()} bus | c1={c1:.3g}")
    dc_cnt = scr.pred_counts(S.L("dc") > args.tau)
    rows = [("DC power flow", dc_cnt, 0.5),
            ("GENCO (as published)", scr.counts("raw"), 0.0),
            ("GENCO + 1 Newton step", scr.counts("g1"), 1.0)]
    if has_base:
        rows.append(("Base case + 1 Newton step", scr.counts("b1"), 1.0))
    rows.append(("Full NR, flat start", scr.counts("true"), float(S.nri("flat")[test].mean())))
    if has_base:
        rows.append(("Full NR, base-case start", scr.counts("true"), float(S.nri("base")[test].mean())))
    rows.append(("Full NR, GENCO warm start", scr.counts("true"), float(S.nri("genco")[test].mean())))
    out = []
    for name, cnt, lu in rows:
        s = summarize(scr, cnt, mask=test)
        out.append((name, s["recall"], s["precision"], s["el_miss"], lu))
    s = summarize(scr, t1.cnt, t1.cost, t1.acc, test)
    out.append(("Certified tiered policy", s["recall"], s["precision"], s["el_miss"], s["LU"]))
    if has_base:
        tb = run_policy(scr, "b", (1,), c1b)
        sb = summarize(scr, tb.cnt, tb.cost, tb.acc, test)
        out.append(("Classical tiered policy (base case)", sb["recall"], sb["precision"], sb["el_miss"], sb["LU"]))
    df = pd.DataFrame(out, columns=["method", "recall", "precision", "missed", "LU per scenario"])
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"policy tiers on test: tier1 {s['acc1']:.1%} | tier2 {1 - s['acc1']:.1%}   "
          f"(escalated scenarios are charged their own NR iteration count)")
    print("(DC voltage violations not modelled; DC LU counted as 0.5 for the smaller real-valued B matrix)")
    print("(legacy table: calibration on a single set; see policy_v3.log for the grouped/conformal analysis)", file=sys.stderr)


# ---------------------------------------------------------------------------
# v3 analysis
# ---------------------------------------------------------------------------

class Log:
    def __init__(self):
        self.lines = []

    def __call__(self, s=""):
        print(s)
        self.lines.append(s)

    def table(self, df, floatfmt=None, **kw):
        fmt = floatfmt or (lambda x: f"{x:.3f}")
        counts = {}
        for c in df.columns:
            if df[c].dtype.kind == "f" and (c.startswith(("el_", "cont_", "n_", "all_el", "all_cont")) or c in INT_COLS):
                v = df[c].dropna()
                if len(v) and (v == np.round(v)).all():
                    counts[c] = lambda x: "" if pd.isna(x) else f"{int(x)}"
        self("\n".join("  " + l for l in df.to_string(index=False, float_format=fmt, formatters=counts, **kw).splitlines()))


INT_COLS = {"scenarios", "load_scen", "branches", "draws_with_miss"}
POLICIES = [("genco_tiered", "g", (1,)), ("base_tiered", "b", (1,)), ("genco_seq", "g", SEQ_STEPS), ("base_seq", "b", SEQ_STEPS)]
POL_NAME = {"genco_tiered": "GENCO tiered (1 step, then NR)", "base_tiered": "Base case tiered (1 step, then NR)",
            "genco_seq": "GENCO sequential (<=3 steps)", "base_seq": "Base case sequential (<=3 steps)"}
RULES = ["max", "conf0.10", "conf0.05"]


def eval_policies(S, scr, scheme, cal, rules, K, fold_seed, esc_tol=6):
    """Rows (dicts) for every policy x rule on set S. scheme 'cross' (cal = RawSet) or 'kfold'."""
    rows = []
    for rule in rules:
        for name, prior, ks in POLICIES:
            if scheme == "cross":
                c, ncal = cross_c(cal, prior, rule)
                c_s = np.full(S.n, c)
                exc_l, exc_s = exceedance(S, prior, c)
                cdesc = dict(c=c, c_min=c, c_max=c, n_cal=ncal)
            else:
                c_s, _, info = kfold_c(S, prior, rule, K, fold_seed)
                exc_l, exc_s = info["exc_load"], info["exc_scen"]
                cf = info["c_folds"]
                cdesc = dict(c=float(np.median(cf)), c_min=float(cf.min()), c_max=float(cf.max()), n_cal=info["n_cal"])
            o = run_policy(scr, prior, ks, c_s, esc_tol)
            r = dict(policy=name, rule=rule, **cdesc, exc_load=exc_l, exc_scen=exc_s, **summarize(scr, o.cnt, o.cost, o.acc))
            rows.append(r)
    return rows


def baseline_rows(S, scr, dc_thr=None):
    """Uncalibrated reference methods (no escalation rule), element + contingency level."""
    rows = []
    nr = lambda start, e: float(S.nri(start, e).mean())
    spec = [("GENCO (as published)", scr.counts("raw"), 0.0),
            ("GENCO, setpoint enforced", scr.counts("phys"), 0.0)]
    for k in SEQ_STEPS:
        spec.append((f"GENCO + {k} Newton step" + ("s" if k > 1 else ""), scr.counts(f"g{k}"), float(k)))
    for k in SEQ_STEPS:
        spec.append((f"Base case + {k} Newton step" + ("s" if k > 1 else ""), scr.counts(f"b{k}"), float(k)))
    spec += [("NR flat start to 1e-6 MW", scr.counts("nr"), nr("flat", 6)),
             ("NR base-case start to 1e-6 MW", scr.counts("nr"), nr("base", 6)),
             ("NR GENCO start to 1e-6 MW", scr.counts("nr"), nr("genco", 6))]
    for name, cnt, lu in spec:
        rows.append(dict(method=name, LU=lu, **summarize(scr, cnt)))
    return rows


def _dc_row(S, scr, name, th, col="dc"):
    """One DC variant: element-level (branch only) and contingency-level metrics. th: scalar or per-scenario thresholds."""
    thr_el = np.repeat(np.broadcast_to(np.asarray(th, dtype=float), (S.n,)), S.cb)
    cnt = scr.pred_counts(S.L(col) > thr_el)
    tp, fp, fn = cnt[:, 0].sum(), cnt[:, 1].sum(), cnt[:, 2].sum()
    flagged_any = (cnt[:, 0] + cnt[:, 1]) > 0
    truth_b_any = scr.ntruth_b > 0
    allm = summarize(scr, cnt)
    return dict(method=name, threshold=float(np.median(th)), el_recall_b=float(tp / max(tp + fn, 1)),
                el_precision_b=float(tp / max(tp + fp, 1)), flag_rate=float((tp + fp) / len(thr_el)),
                el_miss_b=int(fn), el_fa_b=int(fp),
                cont_recall_b=float((truth_b_any & flagged_any).sum() / max(truth_b_any.sum(), 1)),
                cont_miss_b=int((truth_b_any & ~flagged_any).sum()), cont_fa_b=int((~truth_b_any & flagged_any).sum()),
                flagged_scen_frac=float(flagged_any.mean()),
                all_cont_miss=allm["cont_miss"], all_cont_fa=allm["cont_fa"])


DC_LABEL = {"fbeta": "F3", "recall": "recall>=0.99"}


def dc_rows(S, scr, thr):
    """DC variants with thresholds tuned elsewhere (thr: {'fbeta': x, 'recall': y})."""
    rows = [_dc_row(S, scr, "DC ours (Im Yft), threshold = tau", scr.tau, "dc_ours") if S.has("dc_ours") else None,
            _dc_row(S, scr, "DC datakit, threshold = tau", scr.tau)]
    rows += [_dc_row(S, scr, f"DC datakit, tuned {DC_LABEL[k]}", v) for k, v in thr.items()]
    return [r for r in rows if r]


def dc_rows_kfold(S, scr, th_s, tau):
    """Same with out-of-fold thresholds (th_s: {'fbeta': per-scenario array, 'recall': ...})."""
    rows = [_dc_row(S, scr, "DC ours (Im Yft), threshold = tau", tau, "dc_ours") if S.has("dc_ours") else None,
            _dc_row(S, scr, "DC datakit, threshold = tau", tau)]
    rows += [_dc_row(S, scr, f"DC datakit, tuned {DC_LABEL[k]}", v) for k, v in th_s.items()]
    return [r for r in rows if r]


def tune_on(Scal: RawSet, tau, eps_v, mask=None):
    scr = Screen(Scal, tau, eps_v)
    m = np.ones(Scal.n, bool) if mask is None else mask
    em = np.repeat(m, Scal.cb)
    return tune_threshold(Scal.L("dc")[em], scr.tb[em])


def tune_kfold(S: RawSet, tau, eps_v, K, seed):
    """Out-of-fold DC thresholds per scenario (grouped by load scenario) -> dict name -> per-element threshold array."""
    fold = grouped_folds(S.load_idx, K, seed)
    th = {"fbeta": np.zeros(S.n), "recall": np.zeros(S.n)}
    for f in range(fold.max() + 1):
        t = tune_on(S, tau, eps_v, mask=fold != f)
        for k in th:
            th[k][fold == f] = t[k]
    return th


def main_v3(args):
    log = Log()
    out = Path(args.out) if args.out else D
    out.mkdir(parents=True, exist_ok=True)
    sets = {t: RawSet(t, args.dir) for t in ALL_SETS if (Path(args.dir or D) / f"{t}_scenarios.parquet").exists()}
    paper = sets[PAPER]
    tau, eps = args.tau, args.eps_v
    K, R = args.kfold, args.repeats
    alphas = args.alphas
    rules = ["max"] + [f"conf{a:.2f}" for a in alphas]
    log(f"policy_v3: corrected screening analysis | tau={tau} eps_v={eps} | K={K} folds grouped by load scenario | "
        f"calibration set for cross-set scheme: {args.cal_tag}")
    log("Escalated scenarios are charged their own NR iteration count; tier-2 decisions come from our own converged NR.")
    log("'el_' = element level (branches + PQ buses), 'cont_' = contingency level (scenario has any violation).")
    log("Misses are against datakit's AC solution as labels. FA = false alarms.\n")

    # --- 0. data overview ---------------------------------------------------------------------------
    log("=" * 100)
    log("0. DATA")
    rows = []
    for t, S in sets.items():
        scr = Screen(S, tau, eps)
        rows.append(dict(set=LABEL[t], scenarios=S.n, load_scen=S.n_loads, branches=int(S.cb[0]),
                         el_violations=int(scr.ntruth.sum()), viol_per_scen=float(scr.ntruth.mean()),
                         scen_with_viol=float(scr.truth_any.mean())))
    log.table(pd.DataFrame(rows))
    pd.DataFrame(rows).to_csv(out / "v3_data_overview.csv", index=False)

    # --- 1. legacy LU check -------------------------------------------------------------------------
    log("\n" + "=" * 100)
    log("1. LU ACCOUNTING FIX under the OLD calibration (c calibrated on ALL of stress N-1, max rule; the N-1 row is in-sample)")
    log("   old = escalated scenarios charged the all-scenario mean NR count; new = each charged its own count.")
    rows = []
    if "stress_n1_base" in sets:
        cal1 = sets["stress_n1_base"]
        for prior, start in (("g", "genco"), ("b", "base")):
            c_old, _ = cross_c(cal1, prior, "max")
            for t in STRESS:
                S = sets[t]
                scr = Screen(S, tau, eps)
                o = run_policy(scr, prior, (1,), c_old)
                nr = S.nri(start, 6)
                share1 = (o.acc == 1).mean()
                lu_old = share1 * 1 + (1 - share1) * np.maximum(nr, 1).mean()
                rows.append(dict(policy="GENCO tiered" if prior == "g" else "Base case tiered", set=LABEL[t], c=c_old, tier1=share1, LU_old=lu_old, LU_corrected=float(o.cost.mean()),
                                 el_miss=summarize(scr, o.cnt)["el_miss"]))
        df = pd.DataFrame(rows)
        log.table(df)
        df.to_csv(out / "v3_lu_legacy_check.csv", index=False)

    # --- 2. calibration constants ---------------------------------------------------------------------
    log("\n" + "=" * 100)
    log("2. CALIBRATION CONSTANTS c (step-1 ratio |error| / ||dx||^2, per load scenario)")
    rows = []
    for prior in "gb":
        ls, ls_sc = paper.per_load_max(paper.ratio_scores(prior, 1))
        for rule in rules:
            rows.append(dict(prior=PRIORS[prior], calibration=f"paper set, {len(ls)} load scenarios", rule=rule, c=choose_c(ls_sc, rule)))
    log.table(pd.DataFrame(rows), floatfmt=lambda x: f"{x:.4g}")
    log("   distribution of per-load-scenario scores (percentiles) on every set, GENCO prior / base-case prior:")
    rows = []
    for t, S in sets.items():
        r = dict(set=LABEL[t])
        for prior in "gb":
            _, s_ = S.per_load_max(S.ratio_scores(prior, 1))
            r.update({f"{prior}_p50": np.median(s_), f"{prior}_p90": np.quantile(s_, .9), f"{prior}_max": s_.max()})
        rows.append(r)
    log.table(pd.DataFrame(rows), floatfmt=lambda x: f"{x:.4g}")
    pd.DataFrame(rows).to_csv(out / "v3_score_percentiles.csv", index=False)

    # --- 3. main grouped/conformal results -----------------------------------------------------------------
    def grouped_section(ref, title_head, csv_name):
        for S_ in sets.values():
            S_.ref = ref
        log("\n" + "=" * 100)
        log(title_head)
        out_rows = []
        cols = ["policy", "rule", "c", "acc1", "LU", "el_miss", "el_fa", "cont_miss", "cont_fa", "exc_load", "exc_scen", "el_truth", "cont_truth"]
        schemes = [("cross", t) for t in STRESS if t in sets and t != args.cal_tag] + [("kfold", t) for t in sets]
        for scheme, t in schemes:
            S = sets[t]
            scr = Screen(S, tau, eps)
            if scheme == "cross":
                rr = eval_policies(S, scr, "cross", sets[args.cal_tag], rules, K, 0)
                title = (f"{LABEL[t]}: c calibrated on {LABEL[args.cal_tag]} ({sets[args.cal_tag].n_loads} load scenarios), "
                         f"evaluated on ALL {S.n} scenarios / {S.n_loads} load scenarios")
            else:
                rr = eval_policies(S, scr, "kfold", None, rules, K, 0)
                title = f"{LABEL[t]}: {min(K, S.n_loads)}-fold CV grouped by load scenario ({S.n_loads} load scenarios), pooled out-of-fold, fold seed 0"
            for r in rr:
                out_rows.append(dict(scheme=scheme, set=LABEL[t], ref=ref, **r))
            log("\n" + title)
            df = pd.DataFrame(rr)
            df["policy"] = df.policy.map(POL_NAME)
            log.table(df[cols], floatfmt=lambda x: f"{x:.4g}")
        pd.DataFrame(out_rows).to_csv(out / csv_name, index=False)
        for S_ in sets.values():
            S_.ref = "true"

    grouped_section("true", "3. POLICIES WITH GROUPED CALIBRATION. Error measured against datakit's stored solution. c is calibrated on "
                    "load scenarios not in the evaluation fold/set; exc_load = fraction of evaluation load scenarios whose score "
                    "exceeds c; exc_scen = same per scenario", "v3_policies_grouped.csv")

    # --- 3b. fold-seed variability ----------------------------------------------------------------------
    log("\n" + "=" * 100)
    log(f"3b. VARIABILITY OVER {R} GROUPED K-FOLD DRAWS (fold seeds 0..{R-1}); median [min, max] per policy and rule")
    rows = []
    for t, S in sets.items():
        scr = Screen(S, tau, eps)
        reps = []
        for seed in range(R):
            for r in eval_policies(S, scr, "kfold", None, rules, K, seed):
                reps.append(dict(set=LABEL[t], seed=seed, **r))
        df = pd.DataFrame(reps)
        for (pol, rule), g in df.groupby(["policy", "rule"], sort=False):
            row = dict(set=LABEL[t], policy=pol, rule=rule)
            for k in ("LU", "el_miss", "cont_miss", "exc_load"):
                row[k] = f"{g[k].median():.4g} [{g[k].min():.4g}, {g[k].max():.4g}]"
            row["draws_with_miss"] = int((g.el_miss > 0).sum())
            rows.append(row)
    dfv = pd.DataFrame(rows)
    dfv.to_csv(out / "v3_policies_kfold_repeats.csv", index=False)
    log.table(dfv[dfv.policy.isin(["genco_tiered", "base_tiered"])])

    # --- 3c. robustness: error measured against our own converged NR (no label noise) ----------------------------
    log("\n" + "=" * 100)
    log("3c. LABEL-NOISE CHECK ON THE CALIBRATION. datakit's stored solution is only reproduced to ~1e-6 (graphkit's float32 inputs), which "
        "inflates |error|/||dx||^2 when ||dx|| is tiny (a prior that is already almost exact). Same constants with the error measured "
        "against our own converged NR:")
    rows = []
    for ref in ("true", "nr"):
        for S_ in sets.values():
            S_.ref = ref
        for prior in "gb":
            ls, ls_sc = paper.per_load_max(paper.ratio_scores(prior, 1))
            r = dict(error_vs="datakit solution" if ref == "true" else "our NR", prior=PRIORS[prior])
            for rule in rules:
                r[rule] = choose_c(ls_sc, rule)
            r["paper_p50"] = float(np.median(ls_sc))
            r["paper_p90"] = float(np.quantile(ls_sc, .9))
            rows.append(r)
    log.table(pd.DataFrame(rows), floatfmt=lambda x: f"{x:.4g}")
    pd.DataFrame(rows).to_csv(out / "v3_c_label_noise_check.csv", index=False)
    for S_ in sets.values():
        S_.ref = "true"
    grouped_section("nr", "3d. SAME AS SECTION 3 with the error measured against our own converged NR when calibrating c (decisions are still "
                    "scored against datakit's labels)", "v3_policies_grouped_refnr.csv")

    log("\n" + "=" * 100)
    log("3e. IS THE STEP-1 CONSTANT VALID AT STEP 2? Fraction of load scenarios whose step-k ratio |error vs our NR| / ||dx_k||^2 exceeds "
        "the paper-set step-1 constant (max rule, same reference); zero-error elements ignored. Step 3 is not shown: the iterate is "
        "at the NR noise floor.")
    rows = []
    for S_ in sets.values():
        S_.ref = "nr"
    for t, S in sets.items():
        r = dict(set=LABEL[t])
        for prior in "gb":
            c = cross_c(paper, prior, "max")[0]
            r[f"c_{prior}"] = c
            for k in (1, 2):
                _, ls_sc = S.per_load_max(S.ratio_scores(prior, k, err_min=1e-9))
                r[f"{prior}{k}_exceed"] = float(np.mean(ls_sc > c))
                r[f"{prior}{k}_p90"] = float(np.quantile(ls_sc, .9))
        rows.append(r)
    for S_ in sets.values():
        S_.ref = "true"
    log.table(pd.DataFrame(rows), floatfmt=lambda x: f"{x:.4g}")
    pd.DataFrame(rows).to_csv(out / "v3_step_k_ratio_check.csv", index=False)

    # --- 4. baselines & ladder ------------------------------------------------------------------------------
    log("\n" + "=" * 100)
    log("4. EQUAL-COST LADDER (no calibration): misses / false alarms after k Newton steps from each start, and NR to practical tolerances")
    allb = []
    for t, S in sets.items():
        scr = Screen(S, tau, eps)
        rr = baseline_rows(S, scr)
        for r in rr:
            allb.append(dict(set=LABEL[t], **r))
        log(f"\n{LABEL[t]} ({S.n} scenarios, {scr.ntruth.sum()} violating elements, {int(scr.truth_any.sum())} violating scenarios)")
        log.table(pd.DataFrame(rr)[["method", "LU", "recall", "precision", "el_miss", "el_fa", "cont_truth", "cont_miss", "cont_fa"]])
    pd.DataFrame(allb).to_csv(out / "v3_ladder.csv", index=False)

    log("\nNR iterations by tolerance (mean over scenarios; all runs converged to 1e-6 MW unless noted)")
    rows = []
    for t, S in sets.items():
        r = dict(set=LABEL[t])
        for start in ("flat", "dc", "base", "genco"):
            for e in (2, 4, 6):
                r[f"{start}_1e-{e}"] = float(S.nri(start, e).mean())
        r["unconverged_flat/dc/base/genco"] = "/".join(str(int((~S.sc[f"nr_conv_{s}"]).sum())) for s in ("flat", "dc", "base", "genco"))
        rows.append(r)
    df = pd.DataFrame(rows)
    log.table(df, floatfmt=lambda x: f"{x:.3f}")
    df.to_csv(out / "v3_nr_iterations.csv", index=False)

    # --- 5. DC ---------------------------------------------------------------------------------------------------
    log("\n" + "=" * 100)
    log("5. DC BASELINE (datakit's stored PowerModels DC solution; branch overloads only, DC has no voltages)")
    log("   threshold tuned on the calibration set (F-beta, beta=3, as in the organizers' example) or for element recall >= 0.99")
    thr_cal = tune_on(sets[args.cal_tag], tau, eps)
    log(f"   thresholds tuned on {LABEL[args.cal_tag]} (all {sets[args.cal_tag].n_loads} load scenarios): F3 -> {thr_cal['fbeta']:.4f}, recall>=0.99 -> {thr_cal['recall']:.4f}")
    allrows = []
    for t, S in sets.items():
        scr = Screen(S, tau, eps)
        if t == args.cal_tag or t == GOC:
            th_s = tune_kfold(S, tau, eps, K, 0)
            rr = dc_rows_kfold(S, scr, th_s, tau)
            note = f"{LABEL[t]}: thresholds from {min(K, S.n_loads)}-fold CV grouped by load scenario (median threshold shown)"
        else:
            rr = dc_rows(S, scr, thr_cal)
            note = f"{LABEL[t]}: thresholds tuned on {LABEL[args.cal_tag]}"
        for r in rr:
            allrows.append(dict(set=LABEL[t], **r))
        log("\n" + note)
        log.table(pd.DataFrame(rr)[["method", "threshold", "el_recall_b", "el_precision_b", "flag_rate", "el_miss_b", "el_fa_b",
                                    "cont_recall_b", "cont_miss_b", "cont_fa_b", "flagged_scen_frac", "all_cont_miss", "all_cont_fa"]],
                  floatfmt=lambda x: f"{x:.4g}")
    pd.DataFrame(allrows).to_csv(out / "v3_dc_tuned.csv", index=False)
    log("   (all_cont_*: contingency-level counts against ANY violation, i.e. including voltage violations DC cannot see)")

    # --- 6. sensitivity --------------------------------------------------------------------------------------------
    log("\n" + "=" * 100)
    a0 = alphas[0]
    log(f"6. THRESHOLD SENSITIVITY: tau x eps_v, policies calibrated with conformal alpha={a0:.2f} on the {LABEL[args.cal_tag]} "
        f"(kfold for the paper set itself and GOC 2000 uses the 'max' rule: too few load scenarios for conformal)")
    sens = []
    for t, S in sets.items():
        scheme = "kfold" if (t == args.cal_tag or t == GOC) else "cross"
        rule = "max" if t == GOC else f"conf{a0:.2f}"
        for tau_i in TAUS:
            for eps_i in EPSS:
                scr = Screen(S, tau_i, eps_i)
                for name, prior, ks in POLICIES[:2]:
                    if scheme == "cross":
                        c = np.full(S.n, cross_c(sets[args.cal_tag], prior, rule)[0])
                    else:
                        c = kfold_c(S, prior, rule, K, 0)[0]
                    o = run_policy(scr, prior, ks, c)
                    sens.append(dict(set=LABEL[t], tau=tau_i, eps_v=eps_i, method=POL_NAME[name], rule=rule, **summarize(scr, o.cnt, o.cost, o.acc)))
                for lab, m, lu in (("GENCO + 1 step", "g1", 1.0), ("Base + 1 step", "b1", 1.0), ("exact NR (label-noise floor)", "nr", np.nan)):
                    sens.append(dict(set=LABEL[t], tau=tau_i, eps_v=eps_i, method=lab, rule="-", LU=lu, **summarize(scr, scr.counts(m))))
    dsens = pd.DataFrame(sens)
    dsens.to_csv(out / "v3_sensitivity.csv", index=False)
    show = ["tau", "eps_v", "method", "acc1", "LU", "el_truth", "el_miss", "el_fa", "cont_miss", "cont_fa"]
    for t in sets:
        log(f"\n{LABEL[t]}")
        sub = dsens[dsens.set == LABEL[t]]
        log.table(sub[show].assign(eps_v=lambda d: d.eps_v * 1e3).rename(columns={"eps_v": "eps_mV"}), floatfmt=lambda x: f"{x:.4g}")

    # --- 7. Pareto ----------------------------------------------------------------------------------------------------
    log("\n" + "=" * 100)
    log("7. COST-SAFETY PARETO in c (no calibration; every c of the grid applied to every scenario) -> results/exp02/pareto_<tag>.csv")
    grid = np.r_[0.0, np.logspace(-4, 1, 51), math.inf]
    summ = []
    for t, S in sets.items():
        scr = Screen(S, tau, eps)
        rows = []
        for name, prior, ks in POLICIES:
            scen_sc = S.ratio_scores(prior, 1)
            _, ls_sc = S.per_load_max(scen_sc)
            for c in grid:
                o = run_policy(scr, prior, ks, c)
                s = summarize(scr, o.cnt, o.cost, o.acc)
                rows.append(dict(policy=name, c=c, exc_load=float(np.mean(ls_sc > c)), exc_scen=float(np.mean(scen_sc > c)), **s))
        dfp = pd.DataFrame(rows)
        dfp.to_csv(out / f"pareto_{t}.csv", index=False)
        for name, g in dfp.groupby("policy", sort=False):
            g = g[np.isfinite(g.c)].sort_values("c")

            def stable_zero(col):
                """Smallest grid c such that every grid c' >= c has zero misses (in-sample oracle), and its LU."""
                bad = np.flatnonzero(g[col].to_numpy() > 0)
                i = 0 if len(bad) == 0 else bad[-1] + 1
                return (float(g.c.iloc[i]), float(g.LU.iloc[i])) if i < len(g) else (np.nan, np.nan)
            c_e, lu_e = stable_zero("el_miss")
            c_c, lu_c = stable_zero("cont_miss")
            summ.append(dict(set=LABEL[t], policy=name, c_min_zero_el_miss=c_e, LU_el=lu_e,
                             c_min_zero_cont_miss=c_c, LU_cont=lu_c,
                             LU_c0=float(g[g.c == 0].LU.iloc[0]), el_miss_c0=int(g[g.c == 0].el_miss.iloc[0]),
                             cont_miss_c0=int(g[g.c == 0].cont_miss.iloc[0])))
    log("   smallest (= cheapest) grid c from which every larger c has ZERO element misses on the evaluated set itself (an in-sample oracle,")
    log("   not a usable rule), with its LU; same for zero contingency-level misses; and the c=0 point (accept whatever has positive margin)")
    dsum = pd.DataFrame(summ)
    log.table(dsum, floatfmt=lambda x: f"{x:.4g}")
    dsum.to_csv(out / "v3_pareto_summary.csv", index=False)

    # --- 8. tier-2 from our own NR; recertification ---------------------------------------------------------------------
    log("\n" + "=" * 100)
    log("8. TIER-2 ANSWER FROM OUR OWN CONVERGED NR vs datakit's stored solution; RECERTIFICATION")
    rows, rec = [], {}
    for t, S in sets.items():
        if not S.has("nr"):
            continue
        dl = np.abs(S.L("nr") - S.L("true"))
        dv = np.abs(S.Vm("nr") - S.Vm("true"))
        rows.append(dict(set=LABEL[t], max_dloading=float(dl.max()), p99_dloading=float(np.quantile(dl, .99)),
                         max_dVm=float(dv.max()), p99_dVm=float(np.quantile(dv, .99)),
                         max_state_err=float(S.sc.nr_state_err_max.max())))
        if "recert_raw_max" in S.sc.columns:
            rec[t] = dict(label=LABEL[t], n_scenarios=int(S.n), n_load_scenarios=int(S.n_loads),
                          raw_parquet_problem_max_mismatch_MW=float(S.sc.recert_raw_max.max()),
                          raw_parquet_problem_median_mismatch_MW=float(S.sc.recert_raw_max.median()),
                          raw_parquet_problem_p99_mismatch_MW=float(S.sc.recert_raw_max.quantile(.99)),
                          graphkit_problem_max_mismatch_MW=float(S.sc.recert_gk_max.max()),
                          graphkit_problem_median_mismatch_MW=float(S.sc.recert_gk_max.median()),
                          our_NR_vs_datakit_state_max=float(S.sc.nr_state_err_max.max()),
                          n_scen_raw_mismatch_gt_1em6_MW=int((S.sc.recert_raw_max > 1e-6).sum()),
                          n_scen_raw_mismatch_gt_1em3_MW=int((S.sc.recert_raw_max > 1e-3).sum()),
                          nr_converged_all=bool(S.sc[["nr_conv_flat", "nr_conv_base", "nr_conv_genco"]].all().all()))
    log.table(pd.DataFrame(rows), floatfmt=lambda x: f"{x:.3e}")
    pd.DataFrame(rows).to_csv(out / "v3_tier2_vs_truth.csv", index=False)
    log("\n   re-certification: max |mismatch| (MW/MVAr) of datakit's stored solution on a problem built from the raw parquet, and on graphkit's problem")
    log.table(pd.DataFrame(rec).T.drop(columns=["label"]).infer_objects().reset_index().rename(columns={"index": "set"}), floatfmt=lambda x: f"{x:.3e}")
    if rec:
        vdir = ROOT / "results" / "validation"
        vdir.mkdir(parents=True, exist_ok=True)
        (vdir / "recertify.json").write_text(json.dumps(rec, indent=2))

    log("\nDone.")
    Path(args.log).write_text("\n".join(log.lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3", action="store_true", help="full corrected analysis (grouped/conformal calibration, ladder, DC, sensitivity, Pareto)")
    ap.add_argument("--tag", default=PAPER, help="legacy mode: set to evaluate")
    ap.add_argument("--cal_tag", default=None,
                    help="legacy: calibrate c on ALL scenarios of this run, evaluate on ALL of --tag. v3: calibration set (default paper set)")
    ap.add_argument("--tau", type=float, default=None, help="overload threshold on |S|/rate (legacy default 1.001, v3 default 1.01)")
    ap.add_argument("--eps_v", type=float, default=EPS_V, help="voltage violation tolerance (p.u.)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=20, help="v3: number of grouped K-fold draws for the variability table")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.10, 0.05])
    ap.add_argument("--dir", default=None, help="folder holding the exp02 parquet files (default results/exp02)")
    ap.add_argument("--out", default=None, help="v3: folder for CSV outputs (default results/exp02)")
    ap.add_argument("--log", default=str(ROOT / "results" / "policy_v3.log"))
    args = ap.parse_args()
    if args.v3:
        args.tau = TAU if args.tau is None else args.tau
        args.cal_tag = args.cal_tag or PAPER
        main_v3(args)
    else:
        args.tau = 1.001 if args.tau is None else args.tau
        legacy_main(args)


if __name__ == "__main__":
    main()
