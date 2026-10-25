"""Figures for the Hugging Face discussion post (readable at ~800 px width).

    python scripts/make_post_figures.py [--only figB_exchange_rate ...]

Reads existing result files only (no experiments, except the optional one-contingency trace of figG).
Every figure writes results/figures/post/<name>.png (2x), .svg and data/<name>.csv (exactly the plotted values,
with a `source` column naming the result file).
"""
import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = R / "figures" / "post"
DATA = OUT / "data"
OUT.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

# ---- style (same surface/ink as scripts/make_figures.py; one fixed colour per method) -------------------------
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C = {
    "GENCO": "#eb6834",            # released GENCO start (orange)
    "base": "#4a3aa7",             # pre-contingency (base-case) start (purple)
    "flat": "#6f6e6a",             # flat start (neutral grey, distinguished by marker)
    "DC": "#2a78d6",               # DC power-flow start (blue)
    "resid": "#0f8a5f",            # GENCO-Delta residual (green)
    "plain": "#c0407f",            # GENCO-Delta plain (magenta)
}
MARK = {"flat": "s", "DC": "D", "base": "o", "GENCO": "o"}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9.5,
    "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False,
    "axes.axisbelow": True, "svg.fonttype": "none",
})
MINUS = "−"


def rel(p):
    return str(Path(p).resolve().relative_to(ROOT)) if Path(p).is_absolute() else str(p)


def save(fig, name, df):
    """Write PNG (2x), SVG and the plotted-values CSV."""
    fig.savefig(OUT / f"{name}.png", dpi=200)
    fig.savefig(OUT / f"{name}.svg")
    plt.close(fig)
    df.to_csv(DATA / f"{name}.csv", index=False)
    print(f"{name}: wrote {OUT / (name + '.png')} ({len(df)} data rows)")


def fmt(v, nd=2, sign=False):
    s = f"{abs(v):.{nd}f}"
    if sign:
        return (MINUS if v < 0 else "+") + s
    return (MINUS if v < 0 else "") + s


FIGS = {}


def figure(f):
    FIGS[f.__name__] = f
    return f


# =====================================================================================================
# figB: exchange rate
# =====================================================================================================
TOLS = [(1e-2, "1e-2"), (1e-4, "1e-4"), (1e-6, "1e-6")]


def _b_rows():
    """Long table: grid, tol, start, value, lo, hi, n, source."""
    rows = []
    # IEEE 118 Base seed 0, paper-test set (638 contingencies, 29 load scenarios): CIs = cluster bootstrap over loads
    d = pd.read_csv(R / "exp02" / "v3_ci_nr_iterations.csv")
    src = "results/exp02/v3_ci_nr_iterations.csv"
    for seed in (0, 1, 42):
        x = d[(d.model == f"base seed {seed}") & (d.set_key == "paper")]
        for tol, _ in TOLS:
            y = x[np.isclose(x.tol_MW.astype(float), tol)]
            for _, r in y.iterrows():
                rows.append(dict(grid="IEEE 118 Base", seed=seed, tol=tol, start=r.start, value=r.iters,
                                 lo=r.iters_lo, hi=r.iters_hi, n=638 if seed == 0 else np.nan, source=src))
    for label, f, key, n in [("GOC 500 Base", "goc500", "GOC500 Base", 638),
                             ("GOC 2000 Small", "goc2000h", "GOC2000 Small (879)", 879)]:
        d = pd.read_csv(R / "exp02" / f"{f}_nr_iterations.csv")
        x = d[d.set == key]
        for tol, _ in TOLS:
            y = x[np.isclose(x.tol_MW.astype(float), tol)]
            for _, r in y.iterrows():
                rows.append(dict(grid=label, seed=0, tol=tol, start=r.start, value=r.iters, lo=r.iters_lo,
                                 hi=r.iters_hi, n=n, source=f"results/exp02/{f}_nr_iterations.csv"))
    return pd.DataFrame(rows)


@figure
def figB_exchange_rate():
    D = _b_rows()
    grids = ["IEEE 118 Base", "GOC 500 Base", "GOC 2000 Small"]
    nlab = {"IEEE 118 Base": "IEEE 118 Base\n638 contingencies", "GOC 500 Base": "GOC 500 Base\n638 contingencies",
            "GOC 2000 Small": "GOC 2000 Small\n879 contingencies"}
    fig, axs = plt.subplots(2, 3, figsize=(8, 6.3), gridspec_kw=dict(hspace=0.95, wspace=0.12, height_ratios=[1, 1]))
    ys = {g: len(grids) - 1 - i for i, g in enumerate(grids)}
    off = {"flat": 0.27, "DC": 0.09, "base": -0.09, "GENCO": -0.27}
    for j, (tol, tl) in enumerate(TOLS):
        a, b = axs[0, j], axs[1, j]
        for g in grids:
            for s in ("flat", "DC", "base", "GENCO"):
                r = D[(D.grid == g) & (D.seed == 0) & (D.tol == tol) & (D.start == s.lower())].iloc[0]
                y = ys[g] + off[s]
                a.plot([r.lo, r.hi], [y, y], color=C[s], lw=1.2, zorder=2)
                a.plot(r.value, y, MARK[s], color=C[s], ms=4.5, mec=SURFACE, mew=0.6, zorder=3)
            # difference
            r = D[(D.grid == g) & (D.seed == 0) & (D.tol == tol) & (D.start == "genco - base")].iloc[0]
            y = ys[g]
            for sd in (1, 42):
                q = D[(D.grid == g) & (D.seed == sd) & (D.tol == tol) & (D.start == "genco - base")]
                if len(q):
                    b.plot(q.value.iloc[0], y - 0.0 + (0.22 if sd == 1 else -0.22), "o", mfc="none",
                           mec=C["GENCO"], ms=3.5, mew=0.9, zorder=2)
            b.plot([r.lo, r.hi], [y, y], color=C["GENCO"], lw=1.6, zorder=2)
            b.plot(r.value, y, "o", color=C["GENCO"], ms=6, mec=SURFACE, mew=0.8, zorder=3)
            b.text(1.52, y, fmt(r.value, 2, True), ha="right", va="center", fontsize=9, color=INK, fontweight="bold")
        a.set_title(f"to {tl} MW", loc="left", pad=6)
        a.set_xlim(1.4, 7.9)
        b.set_xlim(-1.15, 1.55)
        b.axvline(0, color=INK2, lw=1.0, zorder=1)
        for ax in (a, b):
            ax.set_ylim(-0.6, len(grids) - 0.4)
            ax.set_yticks([ys[g] for g in grids])
            ax.set_yticklabels([nlab[g] for g in grids] if j == 0 else [], fontsize=8.5)
            ax.grid(axis="y", visible=False)
            ax.tick_params(length=0)
        a.set_xlabel("Newton iterations, mean (95% CI)", fontsize=8.5)
        b.set_xlabel("GENCO " + MINUS + " base (iterations)", fontsize=8.5)
        b.text(-0.04, 1.04, "\u2190 GENCO fewer", transform=b.get_xaxis_transform(), ha="right", fontsize=8, color=INK2)
        b.text(0.04, 1.04, "GENCO more \u2192", transform=b.get_xaxis_transform(), ha="left", fontsize=8, color=INK2)
        b.set_xticks([-1, -0.5, 0, 0.5, 1])
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker=MARK[s], color=C[s], ls="", ms=5.5, label=l) for s, l in
         [("flat", "Flat start"), ("DC", "DC start"), ("base", "Pre-contingency state"), ("GENCO", "GENCO prediction")]]
    h.append(Line2D([], [], marker="o", mfc="none", mec=C["GENCO"], ls="", ms=4.5, label="IEEE 118 seeds 1, 42"))
    fig.legend(handles=h, loc="upper center", ncol=5, bbox_to_anchor=(0.55, 0.99), fontsize=8.5, handletextpad=0.2,
               columnspacing=1.2)
    fig.subplots_adjust(left=0.17, right=0.985, top=0.83, bottom=0.08)
    pa = axs[0, 0].get_position(); pb = axs[1, 0].get_position()
    fig.text(0.01, pa.y1 + 0.065, "(a) Newton iterations needed from each start", fontweight="bold", fontsize=9.5, va="center")
    fig.text(0.01, pb.y1 + 0.075, "(b) Paired difference: GENCO prediction minus pre-contingency state", fontweight="bold", fontsize=9.5, va="center")
    out = D.rename(columns={"start": "series", "value": "mean_iterations_or_difference", "lo": "ci95_lo", "hi": "ci95_hi"})
    out["tol_MW"] = out.tol
    out = out.drop(columns="tol")
    out = out[["grid", "seed", "tol_MW", "series", "mean_iterations_or_difference", "ci95_lo", "ci95_hi", "n", "source"]]
    out = out[out.series.isin(["flat", "dc", "base", "genco", "genco - base"])].copy()
    save(fig, "figB_exchange_rate", out)


# =====================================================================================================
# figC: equal cost
# =====================================================================================================
def cp_upper(k, n, conf=0.95):
    """One-sided Clopper-Pearson upper bound."""
    from scipy.stats import beta
    return 1.0 if k >= n else float(beta.ppf(conf, k + 1, n - k))


@figure
def figC_equal_cost():
    sys.path.insert(0, str(ROOT / "scripts"))
    import analyze_exp02 as A
    sets = [("N-1", "stress100_n1_base", "S100 N-1"), ("N-2", "stress100_n2_base", "S100 N-2"),
            ("N-3", "stress100_n3_base", "S100 N-3"), ("N-5", "stress100_n5_base", "S100 N-5")]
    pol = pd.read_csv(R / "exp02" / "big118_policies.csv")
    lad = pd.read_csv(R / "exp02" / "big118_ladder.csv")
    rows = []
    for order, tag, sname in sets:
        S = A.RawSet(tag)
        scr = A.Screen(S)
        truth = int(scr.ntruth.sum())
        # k = 1..3 recomputed from the stored per-element predictions (k = 1, 2 verified against big118_ladder.csv)
        for prior, pl in (("g", "GENCO"), ("b", "base")):
            for k in (1, 2, 3):
                miss = int(scr.counts(f"{prior}{k}")[:, [2, 5]].sum())
                rows.append(dict(order=order, series=f"{pl} + {k} step", prior=pl, kind="ladder", LU=float(k), el_miss=miss,
                                 el_truth=truth, source=f"results/exp02/{tag}_branches|buses.parquet (analyze_exp02.Screen)"))
        for meth, k in (("GENCO + 1 step", 1), ("GENCO + 2 steps", 2), ("Base case + 1 step", 1), ("Base case + 2 steps", 2)):
            v = int(lad[(lad.set == sname) & (lad.method == meth)].el_miss.iloc[0])
            mine = [r for r in rows if r["order"] == order and r["series"] == meth.replace("Base case", "base").replace("steps", "step")][0]
            assert mine["el_miss"] == v, (order, meth, mine["el_miss"], v)
        g0 = lad[(lad.set == sname) & (lad.method == "GENCO as published")].iloc[0]
        rows.append(dict(order=order, series="GENCO as published", prior="GENCO", kind="ladder", LU=0.0, el_miss=int(g0.el_miss),
                         el_truth=int(g0.el_truth), source="results/exp02/big118_ladder.csv"))
        for pri, name, ref in (("genco_seq", "GENCO sequential policy", "datakit"), ("base_seq", "base sequential policy", "own NR")):
            q = pol[(pol.set == sname) & (pol.policy == pri) & (pol.rule == "conf0.10") & (pol.c_ref == ref)].iloc[0]
            rows.append(dict(order=order, series=name, prior="GENCO" if pri.startswith("g") else "base", kind="policy",
                             LU=float(q.LU), el_miss=int(q.el_miss), el_truth=int(q.el_truth),
                             source="results/exp02/big118_policies.csv (conformal 0.10, c from calib_n1_base)"))
    D = pd.DataFrame(rows)
    D["miss_rate_pct"] = 100 * D.el_miss / D.el_truth
    D["ub95_pct"] = [100 * cp_upper(k, n) for k, n in zip(D.el_miss, D.el_truth)]
    D["plotted_pct"] = np.where(D.el_miss == 0, D.ub95_pct, D.miss_rate_pct)
    fig, axs = plt.subplots(2, 2, figsize=(8, 6.2), sharex=True, sharey=True, gridspec_kw=dict(hspace=0.28, wspace=0.08))
    for ax, (order, tag, _) in zip(axs.ravel(), sets):
        d = D[D.order == order]
        for prior in ("GENCO", "base"):
            col = C[prior]
            lad_ = d[(d.prior == prior) & (d.kind == "ladder")].sort_values("LU")
            ax.plot(lad_.LU, lad_.plotted_pct, color=col, lw=1.3, zorder=2)
            for _, r in lad_.iterrows():
                if r.el_miss == 0:
                    ax.plot(r.LU, r.plotted_pct, "v", mfc=SURFACE, mec=col, ms=6, mew=1.3, zorder=3)
                else:
                    ax.plot(r.LU, r.plotted_pct, "o", color=col, ms=5.5, mec=SURFACE, mew=0.6, zorder=3)
                    ax.annotate(f"{int(r.el_miss)}", (r.LU, r.plotted_pct), xytext=(6, 4), textcoords="offset points",
                                fontsize=8, color=INK2)
            p = d[(d.prior == prior) & (d.kind == "policy")].iloc[0]
            ax.plot(p.LU, p.plotted_pct, "*" if p.el_miss else "*", color=col if p.el_miss else SURFACE, mec=col, mew=1.3,
                    ms=11, zorder=4)
        ax.set_yscale("log")
        ax.set_ylim(0.012, 20)
        ax.set_xlim(-0.25, 3.4)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_title(f"{order}  ({int(d.el_truth.iloc[0]):,} violating elements)", loc="left", fontsize=9.5)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}%"))
    for ax in axs[1]:
        ax.set_xlabel("LU factorizations per contingency", fontsize=9)
    for ax in axs[:, 0]:
        ax.set_ylabel("Missed violating elements", fontsize=9)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=C["GENCO"], marker="o", lw=1.3, ms=5, label="GENCO + k Newton steps"),
         Line2D([], [], color=C["base"], marker="o", lw=1.3, ms=5, label="Pre-contingency state + k steps"),
         Line2D([], [], color=INK2, marker="v", mfc=SURFACE, ls="", ms=6, label="0 missed: 95% upper bound shown"),
         Line2D([], [], color=INK2, marker="*", mfc=SURFACE, ls="", ms=10, label="Sequential verification policy")]
    fig.legend(handles=h, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0), fontsize=8.5, columnspacing=1.5)
    fig.subplots_adjust(left=0.11, right=0.985, top=0.88, bottom=0.08)
    save(fig, "figC_equal_cost", D)


# =====================================================================================================
# figD: factor reuse and break-even
# =====================================================================================================
SOLVERS = [("nr", "Full Newton"), ("chordC", "Compensated chord"), ("fdlf", "Fast-decoupled")]


def _goc500_savings():
    """Parse 'solve saving' (1e-4 MW block) of GOC 500 Base from results/notes/batch3_H.md section 2.3."""
    txt = (R / "notes" / "batch3_H.md").read_text()
    i = txt.index("### 2.3 Factor-reuse solvers")
    blk = txt[i:i + 7000]
    j = blk.index("paired GENCO start - base start, tolerance 0.0001 MW")
    blk = blk[j:]
    out = {}
    for m in re.finditer(r"^\s+(nr|chord|chordC|fdlf)\s+d_iters.*?solve saving\s+(-?[\d.]+) \[(-?[\d.]+), (-?[\d.]+)\]", blk, re.M):
        out[m.group(1)] = tuple(float(m.group(k)) for k in (2, 3, 4))
    assert {"nr", "chordC", "fdlf"} <= set(out), out
    return out


def _gpu_rows():
    """GPU inference ms per scenario from results/colab/**/genco_gpu_timing.csv (columns per notebooks/colab_gpu_timing)."""
    out = {}
    for f in sorted((R / "colab").glob("**/genco_gpu_timing.csv")):
        d = pd.read_csv(f)
        if "kind" in d.columns:
            d = d[d.kind.astype(str).str.contains("genco|inference|forward", case=False) | ~d.kind.astype(str).str.contains("newton|precision|lu", case=False)]
        if "layers" in d.columns:
            d = d[pd.to_numeric(d.layers, errors="coerce") == 12]
        if "status" in d.columns:
            d = d[d.status.astype(str).str.lower().isin(["ok", "nan", "success"])]
        if "dtype" in d.columns:
            d = d[d.dtype.astype(str).isin(["fp32", "auto", "tf32"])]
        for (grid, size), g in d.groupby(["grid", "size"]):
            key = {"case118_ieee": "IEEE 118 Base", "case500_goc": "GOC 500 Base"}.get(grid)
            if key is None or (grid == "case118_ieee" and size != "base") or (grid == "case500_goc" and size != "base"):
                continue
            v = g.ms_per_scenario.astype(float)
            out[key] = (float(v.min()), float(v.max()), rel(f))
    return out


@figure
def figD_factor_reuse_breakeven():
    paired = pd.read_csv(R / "exp07" / "summary_paired_genco_vs_base.csv")
    P = paired[(paired.set == "paper N-1") & (paired.group == "all") & np.isclose(paired.tol, 1e-4)].set_index("method")
    sav = {"IEEE 118 Base": {m: (P.loc[m, "saving_solves"], P.loc[m, "saving_solves_lo"], P.loc[m, "saving_solves_hi"]) for m, _ in SOLVERS},
           "GOC 500 Base": _goc500_savings()}
    be = pd.read_csv(R / "exp07" / "breakeven.csv")
    be = be[be.basis == "measured"]
    import json
    t5 = json.load(open(R / "exp07" / "timing_case500_goc_base_seed0.json"))["solve_ms"]
    brk = {"IEEE 118 Base": {}, "GOC 500 Base": {}}
    for m, _ in SOLVERS:
        v = {t: float(be[(be.method == m) & np.isclose(be.tol, t)].breakeven_same_solver_ms.iloc[0]) for t in (1e-2, 1e-4)}
        brk["IEEE 118 Base"][m] = v
        brk["GOC 500 Base"][m] = {t: t5[f"{m}|base|{s}"]["ms_mean"] - t5[f"{m}|genco|{s}"]["ms_mean"] for t, s in ((1e-2, "0.01"), (1e-4, "0.0001"))}
    inf118 = json.load(open(R / "runtime" / "case118_ieee_base.json"))
    inf500 = json.load(open(R / "runtime" / "case500_goc_base_cpu.json"))
    inf = {"IEEE 118 Base": {"CPU": (inf118["genco_cpu_bs16_ms_per_scen"], inf118["genco_cpu_bs1_ms_per_scen"], "results/runtime/case118_ieee_base.json"),
                             "MPS": (inf118["genco_mps_bs16_ms_per_scen"], inf118["genco_mps_bs1_ms_per_scen"], "results/runtime/case118_ieee_base.json")},
           "GOC 500 Base": {"CPU": (inf500["genco_cpu_bs16_ms_per_scen"], inf500["genco_cpu_bs1_ms_per_scen"], "results/runtime/case500_goc_base_cpu.json")}}
    gpu = _gpu_rows()

    rows = []
    for g in sav:
        for m, ml in SOLVERS:
            s = sav[g][m]
            rows.append(dict(panel="a", grid=g, series=ml, value=100 * s[0], lo=100 * s[1], hi=100 * s[2], unit="% of triangular solves saved (1e-4 MW)",
                             source="results/exp07/summary_paired_genco_vs_base.csv (paper N-1, all)" if g.startswith("IEEE") else "results/notes/batch3_H.md section 2.3 (exp07 GOC 500 Base)"))
            b = brk[g][m]
            rows.append(dict(panel="b", grid=g, series=f"break-even: {ml}", value=max(b.values()), lo=min(b.values()), hi=max(b.values()),
                             unit="ms per contingency (value = max over 1e-2, 1e-4 MW; lo/hi = min/max)",
                             source="results/exp07/breakeven.csv" if g.startswith("IEEE") else "results/exp07/timing_case500_goc_base_seed0.json (base_ms - genco_ms)"))
        for dev, (lo, hi, src) in inf[g].items():
            rows.append(dict(panel="b", grid=g, series=f"GENCO inference {dev}", value=lo, lo=lo, hi=hi,
                             unit="ms per scenario (value = batch 16, hi = batch 1)", source=src))
        if g in gpu:
            lo, hi, src = gpu[g]
            rows.append(dict(panel="b", grid=g, series="GENCO inference GPU", value=lo, lo=lo, hi=hi, unit="ms per scenario (min/max over batch)", source=src))
        else:
            rows.append(dict(panel="b", grid=g, series="GENCO inference GPU", value=np.nan, lo=np.nan, hi=np.nan,
                             unit="PLACEHOLDER: fill from results/colab/**/genco_gpu_timing.csv", source="pending"))
    D = pd.DataFrame(rows)

    fig = plt.figure(figsize=(8, 6.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.8, 1.2], hspace=0.7, wspace=0.08, left=0.2, right=0.985, top=0.9, bottom=0.14)
    ax = fig.add_subplot(gs[0, :])
    gcol = {"IEEE 118 Base": "filled", "GOC 500 Base": "hollow"}
    for i, (m, ml) in enumerate(SOLVERS):
        y0 = len(SOLVERS) - 1 - i
        for g, dy in (("IEEE 118 Base", 0.16), ("GOC 500 Base", -0.16)):
            v, lo, hi = (100 * x for x in sav[g][m])
            ax.plot([lo, hi], [y0 + dy] * 2, color=C["GENCO"], lw=1.6, zorder=2)
            ax.plot(v, y0 + dy, "o", color=C["GENCO"] if gcol[g] == "filled" else SURFACE, mec=C["GENCO"], mew=1.5, ms=6.5, zorder=3)
            ax.text(24.5, y0 + dy, fmt(v, 0, True) + "%", ha="right", va="center", fontsize=8.5, fontweight="bold")
    ax.axvline(0, color=INK2, lw=1.0)
    ax.set_xlim(-12, 25)
    ax.set_ylim(-0.6, 2.6)
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels([ml for _, ml in SOLVERS])
    ax.grid(axis="y", visible=False)
    ax.tick_params(length=0)
    ax.set_xlabel("Triangular solves saved by a GENCO start vs the pre-contingency start (%, 95% CI), to 1e-4 MW", fontsize=8.5)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", color=C["GENCO"], ls="", ms=6.5, label="IEEE 118 Base, 638 contingencies"),
                       Line2D([], [], marker="o", mfc=SURFACE, mec=C["GENCO"], mew=1.5, ls="", ms=6.5, label="GOC 500 Base, 638 contingencies")],
              loc="upper left", bbox_to_anchor=(0.0, 1.22), ncol=2, fontsize=8.5, handletextpad=0.2)
    rowsb = [("Break-even:\nfull Newton", "nr"), ("Break-even:\ncompensated chord", "chordC"), ("Break-even:\nfast-decoupled", "fdlf")]
    ylab = ["GENCO on GPU", "GENCO on MPS", "GENCO on CPU"] + [r[0] for r in rowsb]
    axb = []
    for k, g in enumerate(["IEEE 118 Base", "GOC 500 Base"]):
        b = fig.add_subplot(gs[1, k], sharex=axb[0] if axb else None)
        axb.append(b)
        for yi, dev in enumerate(["GPU", "MPS", "CPU"]):
            y = 6 - yi
            if dev == "GPU":
                if g in gpu:
                    lo, hi, _ = gpu[g]
                    b.plot([lo, hi], [y, y], color=C["GENCO"], lw=1.2)
                    b.plot(lo, y, "o", color=C["GENCO"], ms=6.5, mec=SURFACE)
                    b.text(hi * 1.35, y, f"{lo:.3g} ms", va="center", fontsize=8.5)
                else:
                    b.text(1.0, y, "GPU timing pending", ha="center", va="center", fontsize=8.5, color=INK2, style="italic",
                           bbox=dict(boxstyle="round,pad=0.25", fc="none", ec=INK2, lw=0.8, ls=(0, (3, 2))))
            elif dev in inf[g]:
                lo, hi, _ = inf[g][dev]
                b.plot([min(lo, hi), max(lo, hi)], [y, y], color=C["GENCO"], lw=1.2, zorder=2)
                b.plot(lo, y, "o", color=C["GENCO"], ms=6.5, mec=SURFACE, mew=0.6, zorder=3)
                b.plot(hi, y, "o", mfc=SURFACE, mec=C["GENCO"], ms=4.5, mew=1.0, zorder=3)
                if max(lo, hi) > 60:
                    b.text(min(lo, hi) / 1.35, y, f"{lo:.3g} ms", va="center", ha="right", fontsize=8.5, color=INK)
                else:
                    b.text(max(lo, hi) * 1.35, y, f"{lo:.3g} ms", va="center", fontsize=8.5, color=INK)
            else:
                b.text(1.0, y, "not measured", ha="center", va="center", fontsize=8.5, color=INK2, style="italic")
        for yi, (_, m) in enumerate(rowsb):
            y = 2 - yi
            v = max(brk[g][m].values())
            b.plot(v, y, "D", color=INK, ms=5.5, mec=SURFACE, mew=0.6, zorder=3)
            b.text(v * 1.4, y, f"{v:.2g} ms", va="center", fontsize=8.5, color=INK)
        b.axhline(3.0, color=GRID, lw=1.2)
        b.set_xscale("log")
        b.set_xlim(3e-3, 4e2)
        b.set_ylim(-0.6, 6.6)
        b.set_yticks([6, 5, 4, 2, 1, 0])
        b.set_yticklabels(ylab if k == 0 else [], fontsize=8.5)
        b.grid(axis="y", visible=False)
        b.tick_params(length=0, axis="y")
        b.set_title(g, loc="left", fontsize=9.5, pad=4)
        b.set_xlabel("milliseconds (log scale)", fontsize=8.5)
    fig.legend(handles=[Line2D([], [], marker="o", color=C["GENCO"], ls="", ms=6, label="GENCO inference, batch 16"),
                           Line2D([], [], marker="o", mfc=SURFACE, mec=C["GENCO"], ls="", ms=4.5, label="GENCO inference, batch 1"),
                           Line2D([], [], marker="D", color=INK, ls="", ms=5, label="Solver time a GENCO start saves")],
                  loc="lower center", ncol=3, bbox_to_anchor=(0.6, 0.0), fontsize=8, handletextpad=0.2, columnspacing=1.2)
    fig.text(0.01, 0.955, "(a) Triangular-solve savings under factor reuse", fontweight="bold", fontsize=9.5)
    pb = axb[0].get_position()
    fig.text(0.01, pb.y1 + 0.055, "(b) GENCO inference time vs the solver time it saves (break-even)", fontweight="bold", fontsize=9.5)
    save(fig, "figD_factor_reuse_breakeven", D)


# =====================================================================================================
# figE: hidden slack
# =====================================================================================================
@figure
def figE_slack():
    S = pd.read_csv(R / "exp06" / "summary.csv").set_index("tag")
    sel = [("case14_ieee_base_seed0", "IEEE 14"), ("case30_ieee_base_seed0", "IEEE 30"), ("case57_ieee_base_seed0", "IEEE 57"),
           ("case118_ieee_base_seed0", "IEEE 118"), ("case500_goc_base_seed0", "GOC 500"),
           ("case2000_goc_small_seed0", "GOC 2000 Small\n(198 scenarios)"), ("case2000_goc_small_seed0_h879", "GOC 2000 Small\n(879 scenarios)")]
    rows = []
    for tag, name in sel:
        r = S.loc[tag]
        rows.append(dict(grid=name, tag=tag, n_scenarios=int(r.n_scenarios), n_load_scenarios=int(r.n_load_scenarios),
                         reported_MW=r.res_raw_max_med, reported_lo=r.res_raw_max_med_lo, reported_hi=r.res_raw_max_med_hi,
                         physical_MW=r.res_phys_max_med, physical_lo=r.res_phys_max_med_lo, physical_hi=r.res_phys_max_med_hi,
                         ratio_median=r.res_ratio_med, ratio_lo=r.res_ratio_med_lo, ratio_hi=r.res_ratio_med_hi,
                         source="results/exp06/summary.csv"))
    D = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4.3))
    n = len(D)
    for i, r in D.iterrows():
        y = n - 1 - i
        ax.plot([r.reported_MW, r.physical_MW], [y, y], color=GRID, lw=3.5, zorder=1, solid_capstyle="round")
        for v, lo, hi, col in ((r.reported_MW, r.reported_lo, r.reported_hi, C["flat"]), (r.physical_MW, r.physical_lo, r.physical_hi, C["GENCO"])):
            ax.plot([lo, hi], [y, y], color=col, lw=1.2, zorder=2)
            ax.plot(v, y, "o", color=col, ms=7, mec=SURFACE, mew=1.2, zorder=3)
        ax.text(1.4e3, y, f"{r.ratio_median:.0f}×", ha="right", va="center", fontsize=10, fontweight="bold")
    ax.text(1.4e3, n - 0.35, "physical / reported", ha="right", va="bottom", fontsize=8, color=INK2)
    ax.set_xscale("log")
    ax.set_xlim(2e-3, 1.5e3)
    ax.set_ylim(-0.6, n - 0.1)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{g}" for g in D.grid][::-1])
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Median over test contingencies of the max power mismatch (MW, log scale)")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", color=C["flat"], ls="", ms=7, label="Reported: GENCO's own slack voltage"),
                       Line2D([], [], marker="o", color=C["GENCO"], ls="", ms=7, label="Physical: true slack setpoint")],
              loc="upper left", bbox_to_anchor=(0.0, 1.14), ncol=2, fontsize=8.5, handletextpad=0.2)
    fig.subplots_adjust(left=0.2, right=0.985, top=0.86, bottom=0.15)
    save(fig, "figE_slack", D)


# =====================================================================================================
# figF: GENCO-Delta
# =====================================================================================================
DIVERGED_SEED = 42   # both plain-Delta arms diverged for init seed 42 (RESEARCH_LOG 2026-10-21); excluded from mean/SD, shown as x
F_SETS = [("paper", "paper"), ("n1", "N-1"), ("n2", "N-2"), ("n3", "N-3"), ("n5", "N-5")]
F_ARMS = [  # key, label, colour, marker, filled, plain?
    ("released", "Released GENCO", C["GENCO"], "o", True),
    ("control", "Fine-tuned control", C["GENCO"], "o", False),
    ("slack_aware", "Slack-aware fine-tune", C["GENCO"], "s", True),
    ("delta_vm_va", "Plain Δ (vm, va)", C["plain"], "^", True),
    ("delta_vm_va_flows", "Plain Δ (vm, va, flows)", C["plain"], "^", False),
    ("residual_vm_va_flows", "Residual Δ, lr 1e-4", C["resid"], "o", True),
    ("residual_vm_va_flows_lr3e-4", "Residual Δ, lr 3e-4", C["resid"], "s", True),
    ("residual_nk_vm_va_flows_lr3e-4", "Residual Δ, lr 3e-4, N-k trained", C["resid"], "*", True),
]


@figure
def figF_genco_delta():
    f = R / "genco_delta" / "residual_summary_runs.csv"
    src = rel(f)
    d = pd.read_csv(f)
    present = set(d.arm)
    arms = [a for a in F_ARMS if a[0] in present]
    # any other N-k-trained residual arm that appears later is plotted with the same style as the lr3e-4 N-k entry
    for extra in sorted(a for a in present if a.startswith("residual_nk") and a not in [x[0] for x in arms]):
        arms.append((extra, extra.replace("_", " "), C["resid"], "*", True))
    rows, off_rows = [], []
    for sk, sl in F_SETS:
        b = d[(d.set == sk) & (d.arm == "released")]  # base-case start is arm-independent per set/seed split
        base_err, base_it = b.err_l2_median_b.mean(), b.iters_b_e2.mean()
        for ak, al, *_ in arms:
            x = d[(d.set == sk) & (d.arm == ak)]
            plain = ak in ("delta_vm_va", "delta_vm_va_flows")
            keep = x[x.seed != DIVERGED_SEED] if plain else x
            div = x[x.seed == DIVERGED_SEED] if plain else x.iloc[0:0]
            if not len(keep):
                continue
            rows.append(dict(set=sl, arm=ak, label=al, n_seeds=len(keep), seeds=",".join(map(str, keep.seed)),
                             state_err_mean=keep.err_l2_median_g.mean(), state_err_sd=keep.err_l2_median_g.std(ddof=1) if len(keep) > 1 else np.nan,
                             iters_1e2_mean=keep.iters_g_e2.mean(), iters_1e2_sd=keep.iters_g_e2.std(ddof=1) if len(keep) > 1 else np.nan,
                             base_state_err=base_err, base_iters_1e2=base_it,
                             diverged_seed42_state_err=div.err_l2_median_g.iloc[0] if len(div) else np.nan,
                             diverged_seed42_iters_1e2=div.iters_g_e2.iloc[0] if len(div) else np.nan,
                             seed42_marked_diverged=bool(plain and (d[(d.arm == ak) & (d.seed == DIVERGED_SEED)].shape[0] > 0)),
                             seed42_evaluation=("evaluated (off scale)" if len(div) else ("evaluation crashed" if plain else "")), source=src))
    D = pd.DataFrame(rows)
    fig, axs = plt.subplots(2, 1, figsize=(8, 6.8), sharex=True, gridspec_kw=dict(hspace=0.12))
    k = len(arms)
    offs = np.linspace(-0.36, 0.36, k)
    TOP = {"err": 0.62, "it": 2.85}
    for pi, (ax, col, sdcol, top) in enumerate(((axs[0], "state_err_mean", "state_err_sd", TOP["err"]),
                                                (axs[1], "iters_1e2_mean", "iters_1e2_sd", TOP["it"]))):
        for si, (sk, sl) in enumerate(F_SETS):
            dd = D[D.set == sl]
            if not len(dd):
                continue
            bv = dd["base_state_err" if pi == 0 else "base_iters_1e2"].iloc[0]
            ax.plot([si - 0.42, si + 0.42], [bv, bv], color=C["base"], lw=2.2, zorder=1, solid_capstyle="butt")
            for (ak, al, colr, mk, filled), off in zip(arms, offs):
                r = dd[dd.arm == ak]
                if not len(r):
                    continue
                r = r.iloc[0]
                x = si + off
                if not np.isnan(r[sdcol]):
                    ax.plot([x, x], [r[col] - r[sdcol], r[col] + r[sdcol]], color=colr, lw=0.9, zorder=2) if pi == 1 else \
                        ax.plot([x, x], [max(r[col] - r[sdcol], 1e-3), r[col] + r[sdcol]], color=colr, lw=0.9, zorder=2)
                ax.plot(x, r[col], mk, mfc=colr if filled else SURFACE, mec=colr, mew=1.1, ms=5.2 if mk != "*" else 8, zorder=3)
                if r["seed42_marked_diverged"]:
                    ax.plot(x, top, "x", color=colr, ms=6, mew=1.6, zorder=3, clip_on=False)
        ax.set_ylim(top=top * 1.06)
    axs[0].set_yscale("log")
    axs[0].set_ylim(0.011, 0.62 * 1.06)
    axs[0].set_yticks([0.01, 0.03, 0.1, 0.3])
    axs[0].set_yticklabels(["0.01", "0.03", "0.1", "0.3"])
    axs[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axs[0].set_ylabel("Median state error\n(L2 of [Va, Vm] error)", fontsize=9)
    axs[1].set_ylim(1.15, 2.85 * 1.04)
    axs[1].set_ylabel("Newton iterations to 1e-2 MW\n(mean over seeds)", fontsize=9)
    axs[1].set_xticks(range(len(F_SETS)))
    axs[1].set_xticklabels([sl for _, sl in F_SETS], fontsize=9.5)
    axs[1].set_xlim(-0.55, len(F_SETS) - 0.45)
    axs[1].set_xlabel("Test set: released paper-test set (N-1), then stress sets by outage order", fontsize=9)
    for ax in axs:
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", length=0)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=C["base"], lw=2.2, label="Pre-contingency state (no model)")]
    for ak, al, colr, mk, filled in arms:
        h.append(Line2D([], [], marker=mk, mfc=colr if filled else SURFACE, mec=colr, ls="", ms=6 if mk != "*" else 9, label=al))
    h.append(Line2D([], [], marker="x", color=INK2, ls="", ms=6, mew=1.6, label="Plain Δ, seed 42 diverged (off scale or crashed)"))
    fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0), fontsize=8, handletextpad=0.2, columnspacing=1.2, labelspacing=0.35)
    fig.subplots_adjust(left=0.13, right=0.985, top=0.97, bottom=0.2)
    axs[0].text(0.005, 0.96, "Lower is better", transform=axs[0].transAxes, fontsize=8, color=INK2, va="top")
    save(fig, "figF_genco_delta", D)


# =====================================================================================================
# figA: workflow schematic
# =====================================================================================================
@figure
def figA_workflow():
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(8, 3.9))
    ax.set_xlim(0, 102.5)
    ax.set_ylim(0, 50)
    ax.axis("off")
    ax.grid(False)
    nodes = []

    def box(id_, cx, cy, w, h, title, sub="", edge=INK2, face=SURFACE, lw=1.3, tcol=INK):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.0,rounding_size=1.2",
                                    fc=face, ec=edge, lw=lw, zorder=2))
        if sub:
            ax.text(cx, cy + h * 0.17, title, ha="center", va="center", fontsize=9, fontweight="bold", color=tcol, zorder=3, linespacing=1.15)
            ax.text(cx, cy - h * 0.22, sub, ha="center", va="center", fontsize=7.6, color=INK2, zorder=3, linespacing=1.2)
        else:
            ax.text(cx, cy, title, ha="center", va="center", fontsize=9, fontweight="bold", color=tcol, zorder=3, linespacing=1.15)
        nodes.append(dict(id=id_, title=title.replace("\n", " "), detail=sub.replace("\n", " "), x=cx, y=cy))

    def arrow(x0, y0, x1, y1, col=INK2, rad=0.0, lw=1.3):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10, color=col, lw=lw,
                                     connectionstyle=f"arc3,rad={rad}", zorder=1, shrinkA=0, shrinkB=0))
    W = 17.5
    box("outage", 9.5, 25, W - 1, 15, "Outage", "branch or generator\nout of service")
    box("base", 31, 37, W + 1, 15, "Pre-contingency\nstate", "known base-case solve", edge=C["base"], lw=1.8)
    box("genco", 31, 13, W + 1, 15, "GENCO\nprediction", "12 layers,\none forward pass", edge=C["GENCO"], lw=1.8)
    box("newton", 55, 25, W + 2, 21, "k Newton steps", "float64, exact admittances\nk LU factorizations")
    box("bound", 78, 25, W + 1.5, 21, "Calibrated bound", "c · ‖dx‖² on every\nmonitored element\n(branch loading, PQ |V|)")
    box("accept", 96, 38.5, 12, 15, "Accept", "verified\nanswer", edge=C["resid"], lw=1.8)
    box("escalate", 96, 11.5, 12, 15, "Escalate", "full Newton\nto convergence", edge=C["flat"], lw=1.8)
    # arrows
    arrow(18.3, 28, 22.2, 36.5)
    arrow(18.3, 22, 22.2, 13.5)
    ax.text(21.0, 25, "either\nstart", fontsize=7.5, color=INK2, ha="center", va="center")
    arrow(39.7, 36.5, 45.2, 29.5)
    arrow(39.7, 13.5, 45.2, 20.5)
    arrow(64.8, 25, 68.4, 25)
    arrow(87.8, 29.5, 90.0, 36.5, col=C["resid"])
    arrow(87.8, 20.5, 90.0, 13.5, col=C["flat"])
    # loop back: not accepted -> another step
    ax.add_patch(FancyArrowPatch((78, 14.2), (55, 14.2), arrowstyle="-|>", mutation_scale=9, color=INK2, lw=1.0, ls=(0, (3, 2)),
                                 connectionstyle="arc3,rad=0.0", zorder=1))
    ax.text(66.5, 11.2, "margin ≤ bound: take step k+1", fontsize=7.6, color=INK2, ha="center", va="center")
    ax.text(80, 43.5, "every margin > bound:\nverified screening", fontsize=7.6, color=C["resid"], ha="center", va="center", linespacing=1.2)
    ax.text(77, 3.5, "step limit reached, still inside\nthe bound: full Newton", fontsize=7.6, color=INK2, ha="center", va="center", linespacing=1.2)
    ax.text(0.5, 49.5, "Prior = pre-contingency state or GENCO; only the cost of reaching a verified answer differs.", fontsize=8, color=INK2, va="top")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    D = pd.DataFrame(nodes)
    D["source"] = "schematic (no data); Newton/bound design: src/genco_certified/acpf.py, scripts/analyze_exp02.py"
    save(fig, "figA_workflow", D)


# =====================================================================================================
# figG: hero trace (one IEEE 118 N-2 contingency)
# =====================================================================================================
G_CACHE = DATA / "figG_hero.csv"


def _pick_scenario():
    """Representative N-2 contingency: modal iteration counts (GENCO 2, base 3, flat 4 to 1e-6 MW) and median GENCO physical mismatch."""
    S = pd.read_parquet(R / "exp02" / "stress_n2_base_scenarios.parquet")
    m = S[(S.nri_genco_e6 == 2) & (S.nri_base_e6 == 3) & (S.nri_flat_e6 == 4)]
    med = m.res_phys_max.median()
    return int(m.iloc[(m.res_phys_max - med).abs().argmin()].scenario)


def _g_compute():
    """One-contingency trace: GENCO layer mismatches, then Newton iterates from GENCO / base case / flat start (float64)."""
    import torch
    torch.set_num_threads(2)
    sys.path.insert(0, str(ROOT / "src"))
    from genco_certified import acpf
    from genco_certified.loading import load_genco
    from genco_certified.tracing import trace_layers
    sid = _pick_scenario()
    data_root, split = "data/stress/n2", "splits_all"
    Lg = load_genco("pf", "case118_ieee", "base", 0, batch_size=32, overrides={"task": {"task_name": "PowerFlow"}},
                    data_root=data_root, split_dir=ROOT / data_root / "case118_ieee" / split)
    model = Lg.task.model
    root = ROOT / data_root / "case118_ieee"
    test_ids = torch.load(root / split / "test.pt").tolist()
    raw_id = test_ids[sid]
    raw = {t: pd.read_parquet(root / "raw" / f"{t}_data.parquet") for t in ["bus", "branch", "gen"]}
    load_idx = int(raw["bus"].loc[raw["bus"].scenario == raw_id, "load_scenario_idx"].iloc[0])
    sel = {t: d[d.load_scenario_idx == load_idx] for t, d in raw.items()}
    bc = acpf.base_case_voltages(sel["bus"], sel["branch"], sel["gen"])
    sid_now = 0
    rows = []
    for batch in Lg.test_loader():
        n = int(batch["bus"].ptr.numel() - 1)
        if sid_now + n <= sid:
            sid_now += n
            continue
        with torch.no_grad(), trace_layers(model) as tr:
            model(batch)
        states = [s.double().numpy() for s in tr["state"]]
        Lg.normalizer.inverse_transform(batch)
        ptr = batch["bus"].ptr.numpy()
        probs = acpf.extract_problems(batch)
        g = sid - sid_now
        p = probs[g]
        sl = slice(ptr[g], ptr[g + 1])
        for k, s in enumerate(states):
            V = p.V_from(s[sl][:, 0], s[sl][:, 1])
            rows.append(dict(stage="GENCO layer", series="GENCO", step=k + 1,
                             max_mismatch_MW=float(np.abs(acpf.mismatch(p, V)).max())))
        Vg = p.V_from(states[-1][sl][:, 0], states[-1][sl][:, 1])
        Vb = p.V_from(np.abs(bc["V"]), np.angle(bc["V"]))
        for name, v0 in (("GENCO", Vg), ("base", Vb), ("flat", p.flat_start())):
            tr_ = acpf.newton_trajectory(p, v0, n_steps=0, tol_mw=1e-8, max_it=12)
            for i, h in enumerate(tr_["history"]):
                rows.append(dict(stage="Newton step", series=name, step=i, max_mismatch_MW=float(h)))
        break
    D = pd.DataFrame(rows)
    D["scenario_id_in_stress_n2_test"] = sid
    D["source"] = ("recomputed for one contingency: results/exp02/stress_n2_base_scenarios.parquet (selection), "
                   "data/stress/n2/case118_ieee (GENCO Base seed 0 forward pass, acpf.newton_trajectory)")
    return D


@figure
def figG_hero():
    import sys as _s
    D = pd.read_csv(G_CACHE) if (G_CACHE.exists() and "--recompute" not in _s.argv) else _g_compute()
    L = D[D.stage == "GENCO layer"]
    N = D[D.stage == "Newton step"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(8, 3.9), sharey=True, gridspec_kw=dict(width_ratios=[1.0, 1.1], wspace=0.06))
    a.plot(L.step, L.max_mismatch_MW, "-o", color=C["GENCO"], lw=1.5, ms=4.5, mec=SURFACE, mew=0.6)
    a.set_xticks([1, 3, 6, 9, 12])
    a.set_xlabel("GENCO layer (one forward pass)")
    a.set_title("1. GENCO's 12 correction layers", loc="left", fontsize=9.5)
    a.set_ylabel("Max power mismatch (MW)")
    for name, col, mk in (("flat", C["flat"], "s"), ("base", C["base"], "o"), ("GENCO", C["GENCO"], "o")):
        x = N[N.series == name].sort_values("step")
        y = x.max_mismatch_MW.clip(lower=1e-9)
        b.plot(x.step, y, "-" + mk, color=col, lw=1.5, ms=4.8, mec=SURFACE, mew=0.6)
    b.set_xlabel("Newton step (1 LU factorization each)")
    b.set_title("2. Newton steps from each start", loc="left", fontsize=9.5)
    b.set_xticks(range(0, 6))
    b.set_xlim(-0.3, 5.3)
    b.axhline(1e-2, color=INK2, lw=0.9, ls=(0, (4, 3)))
    b.text(5.25, 1.3e-2, "1e-2 MW", ha="right", va="bottom", fontsize=8, color=INK2)
    lab = {"flat": ("Flat start", C["flat"]), "base": ("Pre-contingency state", C["base"]), "GENCO": ("From GENCO output", C["GENCO"])}
    last = {}
    for name in lab:
        x = N[N.series == name].sort_values("step")
        last[name] = x
    from matplotlib.lines import Line2D
    b.legend(handles=[Line2D([], [], marker=mk, color=lab[n][1], lw=1.5, ms=5, label=lab[n][0])
                      for n, mk in (("flat", "s"), ("base", "o"), ("GENCO", "o"))], loc="upper right", fontsize=8.5, handletextpad=0.3)
    for ax in (a, b):
        ax.set_yscale("log")
    a.set_ylim(1e-9, 6e3)
    b.tick_params(labelleft=False)
    sid = int(D.scenario_id_in_stress_n2_test.iloc[0])
    fig.subplots_adjust(left=0.1, right=0.985, top=0.9, bottom=0.17)
    save(fig, "figG_hero", D)


# =====================================================================================================
# figH: hero. GENCO's saving by solver and grid size (10^-4 MW, triangular solves, paired vs pre-contingency start)
# =====================================================================================================
HERO = [  # grid, buses, solver, saving %, lo, hi, source
    ("IEEE 118", 118, "Full Newton", 20, 18, 21, "results/exp07/summary_paired_genco_vs_base.csv (notes/ws1_F_baselines_stats.md s.3)"),
    ("IEEE 118", 118, "Chord + compensation", 21, 20, 21, "same"),
    ("IEEE 118", 118, "Fast-decoupled + compensation", 5, 2, 8, "same"),
    ("GOC 500", 500, "Full Newton", 13, 7, 18, "results/notes/batch3_H.md s.2.3"),
    ("GOC 500", 500, "Chord + compensation", 11, 10, 12, "same"),
    ("GOC 500", 500, "Fast-decoupled + compensation", -7, -9, -4, "same"),
    ("GOC 2000", 2000, "Full Newton", -26, -29, -24, "results/notes/ws1_F2_goc2000_reuse.md (results/exp07/*_goc2000.*)"),
    ("GOC 2000", 2000, "Chord + compensation", 2, 1, 2, "same"),
    ("GOC 2000", 2000, "Fast-decoupled + compensation", -18, -20, -16, "same"),
]


@figure
def figH_hero():
    df = pd.DataFrame(HERO, columns=["grid", "buses", "solver", "saving_pct", "lo", "hi", "source"])
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    x = {"IEEE 118": 0, "GOC 500": 1, "GOC 2000": 2}
    style = {"Full Newton": ("o", "-"), "Chord + compensation": ("s", "--"), "Fast-decoupled + compensation": ("D", ":")}
    off = {"Full Newton": -0.06, "Chord + compensation": 0.0, "Fast-decoupled + compensation": 0.06}
    ax.axhspan(-35, 0, color=C["base"], alpha=0.06, lw=0)
    ax.axhline(0, color=INK2, lw=1)
    for solver, g in df.groupby("solver", sort=False):
        m, ls = style[solver]
        xs = np.array([x[k] for k in g.grid]) + off[solver]
        ax.errorbar(xs, g.saving_pct, yerr=[g.saving_pct - g.lo, g.hi - g.saving_pct], fmt=m + ls, color=C["GENCO"],
                    ms=7, lw=1.6, capsize=0, elinewidth=1.4, mfc=C["GENCO"] if m == "o" else SURFACE, mew=1.6)
        last = g.iloc[-1]
        ax.annotate(solver, (xs[-1], last.saving_pct), xytext=(10, 0), textcoords="offset points", va="center",
                    fontsize=9, color=INK)
    ax.set_xticks([0, 1, 2], ["IEEE 118\n118 buses", "GOC 500\n500 buses", "GOC 2000\n2,000 buses"])
    ax.set_xlim(-0.3, 3.05)
    ax.set_ylim(-33, 28)
    ax.set_ylabel("Triangular solves saved by a GENCO start\nvs the pre-contingency start (%)")
    ax.text(-0.25, 25, "GENCO start cheaper", color=C["GENCO"], fontsize=9, va="top")
    ax.text(-0.25, -30, "pre-contingency start cheaper", color=C["base"], fontsize=9, va="bottom")
    ax.set_title("What a GENCO prediction saves, by solver and grid size (to 10⁻⁴ MW, 95% CI)", loc="left")
    fig.tight_layout()
    save(fig, "figH_hero", df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    names = args.only or list(FIGS)
    for n in names:
        if n not in FIGS:
            sys.exit(f"unknown figure {n}; choose from {list(FIGS)}")
        try:
            FIGS[n]()
        except Exception as e:  # keep going so one missing input does not block the others
            import traceback
            traceback.print_exc()
            print(f"!! {n} failed: {e}")


if __name__ == "__main__":
    main()
