"""Tiered, calibrated screening policy on exp02 outputs (no model re-runs).

Tier 0 (free):   GENCO output. A branch/bus decision is accepted when its margin to the
                 limit exceeds c0 * (GENCO's own power-balance residual), a ground-truth-free
                 signal. c0 is the smallest constant that covers every calibration error.
Tier 1 (1 LU):   one Newton step from GENCO. Accepted when the margin exceeds c1 * ||dx||^2
                 (Newton's quadratic-convergence error model), c1 calibrated likewise.
Tier 2 (~1-2 LU): finish Newton-Raphson from the tier-1 point -> exact AC answer.

A scenario stops at the first tier where *every* branch and bus decision is accepted.
Calibration and evaluation use disjoint halves of the paper-test scenarios.

Usage: python scripts/analyze_exp02.py --tag case118_ieee_base_seed0 --tau 1.001
"""

import argparse

import numpy as np
import pandas as pd

from genco_certified.loading import ROOT

D = ROOT / "results" / "exp02"


def load(tag, eps_v):
    br = pd.read_parquet(D / f"{tag}_branches.parquet")
    bu = pd.read_parquet(D / f"{tag}_buses.parquet")
    # PV/REF voltages are setpoints the operator already knows; screen PQ buses only, and
    # treat a violation as exceeding the band by more than eps_v (many limits are OPF-binding).
    bu = bu[bu.btype == 0].reset_index(drop=True)
    bu["vmin"] = bu.vmin - eps_v
    bu["vmax"] = bu.vmax + eps_v
    sc = pd.read_parquet(D / f"{tag}_scenarios.parquet").set_index("scenario")
    return br, bu, sc


def margins(br, bu, m, tau):
    """Signed distance to the violation boundary (positive = secure side) per element."""
    mb = tau - br[f"load_{m}"]
    lo = bu[f"vm_{m}"] - bu.vmin
    hi = bu.vmax - bu[f"vm_{m}"]
    return mb, np.minimum(lo, hi)


def errors(br, bu, m):
    return (br[f"load_{m}"] - br.load_true).abs(), (bu[f"vm_{m}"] - bu.vm_true).abs()


def calibrate(br, bu, sc, m, feat, cal):
    """Smallest c with |error| <= c * feature(scenario) for every calibration element."""
    eb, ev = errors(br, bu, m)
    fb = sc.loc[br.scenario, feat].to_numpy()
    fv = sc.loc[bu.scenario, feat].to_numpy()
    rb = (eb / np.maximum(fb, 1e-12))[br.scenario.isin(cal).to_numpy()]
    rv = (ev / np.maximum(fv, 1e-12))[bu.scenario.isin(cal).to_numpy()]
    return max(rb.max(), rv.max())


def certain(br, bu, sc, m, feat, c, tau):
    """Per-scenario: are all element decisions outside the error band?"""
    mb, mv = margins(br, bu, m, tau)
    ok_b = mb.abs().to_numpy() > c * sc.loc[br.scenario, feat].to_numpy()
    ok_v = mv.abs().to_numpy() > c * sc.loc[bu.scenario, feat].to_numpy()
    return (pd.Series(ok_b).groupby(br.scenario.to_numpy()).all()
            & pd.Series(ok_v).groupby(bu.scenario.to_numpy()).all())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="case118_ieee_base_seed0")
    ap.add_argument("--tau", type=float, default=1.001, help="overload threshold on |S|/rate")
    ap.add_argument("--eps_v", type=float, default=1e-3, help="voltage violation tolerance (p.u.)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cal_tag", default=None,
                    help="calibrate c0/c1 on ALL scenarios of this run and evaluate on ALL of --tag "
                         "(distribution-shift test); default: random half/half split of --tag")
    args = ap.parse_args()
    br, bu, sc = load(args.tag, args.eps_v)
    sc["step_sq"] = sc.step_l2 ** 2
    has_base = "load_base1" in br.columns
    if has_base:
        sc["step_base_sq"] = sc.step_base_l2 ** 2

    if args.cal_tag:
        brc, buc, scc = load(args.cal_tag, args.eps_v)
        scc["step_sq"] = scc.step_l2 ** 2
        allc = set(scc.index)
        c0 = calibrate(brc, buc, scc, "raw", "res_raw_max", allc)
        c1 = calibrate(brc, buc, scc, "step1", "step_sq", allc)
        if has_base:
            scc["step_base_sq"] = scc.step_base_l2 ** 2
            c1b = calibrate(brc, buc, scc, "base1", "step_base_sq", allc)
        test = set(sc.index)
    else:
        rng = np.random.default_rng(args.seed)
        ids = sc.index.to_numpy().copy()
        rng.shuffle(ids)
        cal, test = set(ids[: len(ids) // 2]), set(ids[len(ids) // 2:])
        c0 = calibrate(br, bu, sc, "raw", "res_raw_max", cal)
        c1 = calibrate(br, bu, sc, "step1", "step_sq", cal)
        if has_base:
            c1b = calibrate(br, bu, sc, "base1", "step_base_sq", cal)
    tier0 = certain(br, bu, sc, "raw", "res_raw_max", c0, args.tau)
    tier1 = certain(br, bu, sc, "step1", "step_sq", c1, args.tau)

    # Decisions: tier 0 -> GENCO raw; tier 1 -> step1; tier 2 -> exact (true)
    tier = pd.Series(2, index=sc.index)
    tier[tier1[tier1].index] = 1
    tier[tier0[tier0].index] = 0
    src = tier.map({0: "raw", 1: "step1", 2: "true"})
    pick_b = np.choose(br.scenario.map(tier).to_numpy(), [br.load_raw, br.load_step1, br.load_true])
    pick_v = np.choose(bu.scenario.map(tier).to_numpy(), [bu.vm_raw, bu.vm_step1, bu.vm_true])

    T = br.scenario.isin(test).to_numpy()
    Tv = bu.scenario.isin(test).to_numpy()
    truth_b = (br.load_true > args.tau).to_numpy()
    truth_v = ((bu.vm_true < bu.vmin) | (bu.vm_true > bu.vmax)).to_numpy()

    def score(pred_b, pred_v):
        tp = (pred_b & truth_b & T).sum() + (pred_v & truth_v & Tv).sum()
        fn = (~pred_b & truth_b & T).sum() + (~pred_v & truth_v & Tv).sum()
        fp = (pred_b & ~truth_b & T).sum() + (pred_v & ~truth_v & Tv).sum()
        return tp / max(tp + fn, 1), tp / max(tp + fp, 1), int(fn)

    ts = sc.loc[sorted(test)]
    print(f"{args.tag} (calibrated on {args.cal_tag or 'own half'}) | tau={args.tau} | test scenarios {len(ts)} | violations in test: "
          f"{(truth_b & T).sum()} branch + {(truth_v & Tv).sum()} bus | c0={c0:.3g} c1={c1:.3g}")
    rows = []
    for name, lb, vm, lu in [
        ("DC power flow", br.load_dc, None, 0.5),
        ("GENCO (as published)", br.load_raw, bu.vm_raw, 0.0),
        ("GENCO + 1 Newton step", br.load_step1, bu.vm_step1, 1.0),
        ("Full NR, flat start", br.load_true, bu.vm_true, ts.nr_iters_flat.mean()),
        ("Full NR, GENCO warm start", br.load_true, bu.vm_true, ts.nr_iters_genco.mean()),
    ]:
        pv = ((vm < bu.vmin) | (vm > bu.vmax)).to_numpy() if vm is not None else np.zeros(len(bu), bool)
        r, p, fn = score((lb > args.tau).to_numpy(), pv)
        rows.append((name, r, p, fn, lu))
    tt = tier[ts.index]
    lu_policy = (tt == 1).mean() * 1 + (tt == 2).mean() * (1 + np.maximum(ts.nr_iters_genco - 1, 0)).mean()
    pv = ((pick_v < bu.vmin) | (pick_v > bu.vmax)).to_numpy()
    r, p, fn = score(pick_b > args.tau, pv)
    rows.append(("Certified tiered policy", r, p, fn, lu_policy))
    if has_base:
        # Classical analogue: base-case warm start + 1 Newton step, same calibrated escalation rule.
        okb = certain(br, bu, sc, "base1", "step_base_sq", c1b, args.tau)
        tierb = pd.Series(2, index=sc.index)
        tierb[okb[okb].index] = 1
        pb = np.choose(br.scenario.map(tierb).to_numpy(), [br.load_base1, br.load_base1, br.load_true])
        pvb = np.choose(bu.scenario.map(tierb).to_numpy(), [bu.vm_base1, bu.vm_base1, bu.vm_true])
        ttb = tierb[ts.index]
        lu_b = (ttb == 1).mean() + (ttb == 2).mean() * (1 + np.maximum(ts.nr_iters_base - 1, 0)).mean()
        r, p, fn = score((br.load_base1 > args.tau).to_numpy(), ((bu.vm_base1 < bu.vmin) | (bu.vm_base1 > bu.vmax)).to_numpy())
        rows.insert(3, ("Base case + 1 Newton step", r, p, fn, 1.0))
        rows.insert(5, ("Full NR, base-case start", 1.0, 1.0, 0, ts.nr_iters_base.mean()))
        r, p, fn = score(pb > args.tau, ((pvb < bu.vmin) | (pvb > bu.vmax)).to_numpy())
        rows.append(("Classical tiered policy (base case)", r, p, fn, lu_b))
    df = pd.DataFrame(rows, columns=["method", "recall", "precision", "missed", "LU per scenario"])
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"policy tiers on test: tier0 {(tt == 0).mean():.1%} | tier1 {(tt == 1).mean():.1%} | tier2 {(tt == 2).mean():.1%}")
    print("(DC voltage violations not modelled; DC LU counted as 0.5 for the smaller real-valued B matrix)")


if __name__ == "__main__":
    main()
