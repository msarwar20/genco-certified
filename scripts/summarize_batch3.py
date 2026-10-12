"""Summaries of exp01 / exp05 / exp06 runs (any grid) with cluster-bootstrap CIs over load scenarios.

Reads results/exp01, exp05, exp06 and the exp02 scenario table of the same set (for the load-scenario ids).

Usage:
    python scripts/summarize_batch3.py --runs goc500_base=case500_goc_base_seed0:case500_goc_base \\
        goc500_small=case500_goc_small_seed0:case500_goc_small ...
    (label=<tag of exp01/exp02/exp06>:<tag of exp05>)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_exp02 import cluster_bootstrap, fmt_ci  # noqa: E402
from genco_certified.loading import ROOT  # noqa: E402

R = ROOT / "results"


def ci(x, g, nd=2, B=2000):
    return fmt_ci(*cluster_bootstrap(np.asarray(x, float), groups=g, B=B), nd=nd)


def exp07_summary(tag, label):
    """Iterations / triangular solves / factorizations of the factor-reuse solvers from base-case vs GENCO starts,
    paired savings, failures (results/exp07/<tag>_scenarios.parquet). Same definitions as exp07's own summary."""
    from exp07_industrial_baselines import COMBOS, solves_of

    df = pd.read_parquet(R / "exp07" / f"{tag}_scenarios.parquet")
    if "is_n0" in df and df.is_n0.any():
        df = df[~df.is_n0]
    df = df[df.base_ok]
    g = df.load_idx.to_numpy()
    print("=" * 100)
    print(f"exp07 {label}: {len(df)} contingencies, {df.load_idx.nunique()} load scenarios; mean outaged branches {df.n_br_out.mean():.2f}; "
          f"with a PV->PQ change {int((df.n_new_pq > 0).sum())}")
    for e, tol in ((2, 1e-2), (4, 1e-4)):
        print(f"  tolerance {tol:g} MW (iterations / triangular-solve pairs incl. setup / factorizations per contingency; not converged within the cap)")
        for meth, st in COMBOS:
            it = df[f"{meth}_{st}_it{e}"]
            ok = it.notna().to_numpy()
            sv, fac = solves_of(df, meth, st, e)
            print(f"    {meth:7s} {st:6s} iters {ci(it[ok], g[ok]):>24s} | solves {ci(sv[ok], g[ok]):>24s} | fact {fac.mean():5.2f} | fail {int((~ok).sum())}/{len(df)}")
        print(f"  paired GENCO start - base start, tolerance {tol:g} MW")
        for meth in ("nr", "chord", "chordC", "fdlf"):
            a, b = df[f"{meth}_genco_it{e}"], df[f"{meth}_base_it{e}"]
            ok = (a.notna() & b.notna()).to_numpy()
            sva, _ = solves_of(df, meth, "genco", e)
            svb, _ = solves_of(df, meth, "base", e)
            d_it = cluster_bootstrap((a - b)[ok], groups=g[ok], B=2000)
            r_sv = cluster_bootstrap(sva[ok], den=svb[ok], groups=g[ok], B=2000)
            d_sv = cluster_bootstrap((sva - svb)[ok], groups=g[ok], B=2000)
            print(f"    {meth:7s} d_iters {fmt_ci(*d_it, nd=2):>24s} | d_solves {fmt_ci(*d_sv, nd=2):>24s} | solve saving {fmt_ci(1 - r_sv[0], 1 - r_sv[2], 1 - r_sv[1], 2):>22s} "
                  f"| GENCO fewer iters in {100 * (a < b)[ok].mean():.0f}%, more in {100 * (a > b)[ok].mean():.0f}% (n={ok.sum()})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp07", nargs=2, metavar=("TAG", "LABEL"), default=None, help="summarize results/exp07/<TAG>_scenarios.parquet")
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--layers", action="store_true", help="also print the per-layer table of exp01")
    args = ap.parse_args()
    if args.exp07:
        exp07_summary(*args.exp07)
    for spec in args.runs:
        label, rest = spec.split("=", 1)
        tag, tag5 = rest.split(":")
        sc2 = pd.read_parquet(R / "exp02" / f"{tag}_scenarios.parquet")
        g = sc2.load_idx.to_numpy()
        print("=" * 100)
        print(f"{label}: {len(sc2)} scenarios, {sc2.load_idx.nunique()} load scenarios (exp02 tag {tag})")
        # ---- exp01
        f1 = R / "exp01" / f"{tag}_scenarios.parquet"
        e1 = pd.read_parquet(f1) if f1.exists() else None
        if e1 is None:
            print("  exp01  (not run yet)")
        if e1 is not None:
            print("  exp01  slack Vm error (p.u.) median %.3g | reported max mismatch (MW) median %.3g | physical %.3g" % (
                e1.ref_vm_err.median(), e1.res_raw_max.median(), e1.res_phys_max.median()))
            print("         NR iterations to 1e-6 MW: flat %s | DC %s | GENCO %s | converged %.3f / %.3f / %.3f" % (
                ci(e1.nr_iters_flat, g), ci(e1.nr_iters_dc, g), ci(e1.nr_iters_genco, g),
                e1.nr_conv_flat.mean(), e1.nr_conv_dc.mean(), e1.nr_conv_genco.mean()))
            print("         state error l2 median %.3g -> after one Newton step %.3g (ratio of medians %.0fx); step-norm estimate vs error Spearman %.3f" % (
                e1.err_l2.median(), e1.err1_l2.median(), e1.err_l2.median() / e1.err1_l2.median(),
                pd.Series(e1.est_l2).corr(pd.Series(e1.err_l2), method="spearman")))
            if args.layers:
                ly = pd.read_parquet(R / "exp01" / f"{tag}_layers.parquet")
                t = ly.groupby("layer").agg(res_raw_med=("res_raw_max", "median"), res_phys_med=("res_phys_max", "median"),
                                            err_vm=("err_vm_inf", "median"), err_va=("err_va_inf", "median"), nr_iters=("nr_iters", "mean"))
                print(t.to_string(float_format=lambda x: f"{x:.3g}"))
        # ---- exp05
        e5 = pd.read_parquet(R / "exp05" / f"{tag5}.parquet")
        assert len(e5) == len(sc2)
        print("  exp05  NR iterations to 1e-6 MW: always-GENCO %s | always-base %s | best-of-two %s (picks GENCO %.0f%%) | oracle %s" % (
            ci(e5.it_genco, g), ci(e5.it_base, g), ci(e5.it_best, g), 100 * e5.pick_genco.mean(), ci(e5.it_oracle, g)))
        print("         paired: GENCO - base %s | best-of-two - base %s | best-of-two - GENCO %s | GENCO fewer/more than base: %.0f%% / %.0f%%" % (
            ci(e5.it_genco - e5.it_base, g), ci(e5.it_best - e5.it_base, g), ci(e5.it_best - e5.it_genco, g),
            100 * (e5.it_genco < e5.it_base).mean(), 100 * (e5.it_genco > e5.it_base).mean()))
        # ---- exp06
        j = json.load(open(R / "exp06" / f"{tag}_summary.json"))
        f = lambda k, nd=3: f"{j[k]:.{nd}g} [{j[k + '_lo']:.{nd}g}, {j[k + '_hi']:.{nd}g}]"
        print("  exp06  slack Vm error median %s | max mismatch reported %s -> physical %s | ratio %s" % (
            f("ref_vm_err_med"), f("res_raw_max_med"), f("res_phys_max_med"), f("res_ratio_med")))
        print("         worst bus adjacent to slack: full %s | constrained %s | share of |dS| in 1-hop (full) %s | constrained %s | enrichment %s" % (
            f("worst_adj_frac"), f("worst_adj_pf_frac"), f("share_hop1_med"), f("share_hop1_pf_med"), f("enrichment_med")))
        print("         re-anchoring worse in %.0f%% (shift, state error) / %.0f%% (scale)" % (100 * j["shift_err_worse_frac"], 100 * j["scale_err_worse_frac"]))


if __name__ == "__main__":
    main()
