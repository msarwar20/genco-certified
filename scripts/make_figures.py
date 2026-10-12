"""Static report figures from results/exp01 and results/exp02 (PNG, light theme)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from genco_certified.loading import ROOT

R = ROOT / "results"
FIG = R / "figures"
FIG.mkdir(parents=True, exist_ok=True)

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
# One colour per method, fixed across every figure (reference palette slots 1-8).
METHOD_COLOR = {
    "DC power flow": "#2a78d6", "GENCO (as published)": "#eb6834", "GENCO + 1 Newton step": "#1baf7a",
    "Certified tiered policy": "#eda100", "Full NR, flat start": "#e87ba4", "Full NR, GENCO warm start": "#008300",
    "Base case + 1 Newton step": "#4a3aa7", "Full NR, base-case start": "#e34948",
    "Classical tiered policy (base case)": "#e34948",
}
OURS = ("GENCO + 1 Newton step", "Certified tiered policy", "Full NR, GENCO warm start")
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
})


def fig_layers():
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    for size, c in [("base", BLUE), ("small", ORANGE), ("tiny", AQUA)]:
        L = pd.read_parquet(R / "exp01" / f"case118_ieee_{size}_seed0_layers.parquet")
        g = L.groupby("layer")
        med, lo, hi = g.res_raw_mean.median(), g.res_raw_mean.quantile(.1), g.res_raw_mean.quantile(.9)
        x = med.index + 1
        a.fill_between(x, lo, hi, color=c, alpha=0.12, linewidth=0)
        a.plot(x, med, color=c, lw=2, marker="o", ms=4, label=f"GENCO {size.capitalize()}")
        it = g.nr_iters.mean()[1:]
        b.plot(it.index + 1, it, color=c, lw=2, marker="o", ms=4, label=f"GENCO {size.capitalize()}")
    a.set_yscale("log")
    a.set(xlabel="Correction layer", ylabel="Mean |power mismatch| (MW)", xticks=range(1, 13),
          title="GENCO halves its own power mismatch every layer")
    a.legend(loc="upper right")
    S = pd.read_parquet(R / "exp01" / "case118_ieee_base_seed0_scenarios.parquet")
    b.axhline(S.nr_iters_flat.mean(), color=INK2, lw=1.2, ls="--")
    b.text(12, S.nr_iters_flat.mean() + 0.08, "flat start", ha="right", va="bottom", color=INK2, fontsize=9)
    b.set(xlabel="Warm-start Newton-Raphson from layer", ylabel="Newton iterations to 1e-6 MW",
          xticks=range(2, 13), ylim=(1.5, 4.4), title="Exiting after layer 9 costs < 0.1 extra iteration")
    fig.text(0.01, 0.01, "IEEE 118, 638 paper-test scenarios; band = 10-90th percentile. Layer 1 warm starts diverge (not shown).",
             color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "fig1_layer_dynamics.png", dpi=200)


SLACK_GRIDS = [("case14_ieee_base_seed0", "IEEE 14"), ("case30_ieee_base_seed0", "IEEE 30"),
               ("case57_ieee_base_seed0", "IEEE 57"), ("case118_ieee_base_seed0", "IEEE 118"),
               ("case2000_goc_small_seed0", "GOC 2000 (Small)")]


def _slack_rows():
    """[(label, reported, physical, n_scenarios)] from results/exp06/summary.csv, else from the exp01 parquets."""
    summ = R / "exp06" / "summary.csv"
    S = pd.read_csv(summ).set_index("tag") if summ.exists() else None
    rows = []
    for tag, name in SLACK_GRIDS:
        if S is not None and tag in S.index:
            r = S.loc[tag]
            rows.append((name, float(r.res_raw_max_med), float(r.res_phys_max_med), int(r.n_scenarios)))
            continue
        f = R / "exp01" / f"{tag}_scenarios.parquet"
        if f.exists():
            d = pd.read_parquet(f)
            rows.append((name, float(d.res_raw_max.median()), float(d.res_phys_max.median()), len(d)))
    return rows


def fig_slack():
    rows = _slack_rows()
    fig, ax = plt.subplots(figsize=(7.5, 3.9))
    for i, (name, rep, phy, _) in enumerate(rows):
        ax.plot([rep, phy], [i, i], color=GRID, lw=3, zorder=1)
        ax.scatter([rep], [i], color=BLUE, s=60, zorder=2, edgecolor=SURFACE, linewidth=2,
                   label="Reported (GENCO's own slack voltage)" if i == 0 else None)
        ax.scatter([phy], [i], color=ORANGE, s=60, zorder=2, edgecolor=SURFACE, linewidth=2,
                   label="Physical (true slack setpoint)" if i == 0 else None)
        ax.text(phy * 1.25, i, f"{phy / rep:.0f}x", va="center", color=INK2, fontsize=9)
    ax.set_xscale("log")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_ylim(-0.5, len(rows) - 0.5)
    ax.set_xlim(2e-3, 1200)
    ax.set(xlabel="Median max |power mismatch| per scenario (MW, log scale)")
    fig.suptitle("GENCO's power balance holds only at its own guess of the slack voltage", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", visible=False)
    n = {r[0]: r[3] for r in rows}
    ieee = [v for k, v in n.items() if k.startswith("IEEE")]
    fig.text(0.01, 0.01, "Medians over all test scenarios (IEEE: paper-test N-1 subset, %d-%d each; GOC 2000 Small: %d held-out).\n"
             "Ratio = physical / reported. 95%% cluster-bootstrap CIs over load scenarios: results/exp06/summary.csv." % (
                 min(ieee), max(ieee), n.get("GOC 2000 (Small)", 0)), color=INK2, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIG / "fig2_hidden_slack.png", dpi=200)


def fig_slack_locality():
    """Mean |dS| per bus vs hop distance from the slack (median and 25-75% band over test scenarios)."""
    colors = [BLUE, AQUA, "#4a3aa7", ORANGE, "#e87ba4"]
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    drawn = 0
    for (tag, name), c in zip(SLACK_GRIDS, colors):
        f = R / "exp06" / f"{tag}_hops.parquet"
        if not f.exists():
            continue
        H = pd.read_parquet(f)
        n_scen = H.scenario.nunique()
        W = H.pivot(index="scenario", columns="hop", values="mean_dS")
        W = W.loc[:, W.notna().sum() >= 0.5 * n_scen]  # hop bins present in at least half of the scenarios
        med, lo, hi = W.median(), W.quantile(.25), W.quantile(.75)
        ax.fill_between(med.index, lo, hi, color=c, alpha=0.14, linewidth=0)
        ax.plot(med.index, med, color=c, lw=2, marker="o", ms=5, markeredgecolor=SURFACE, markeredgewidth=1.2, label=name)
        drawn += 1
    ax.set_yscale("log")
    ax.set_xticks(range(1, 7), ["1", "2", "3", "4", "5", "6+"])
    ax.set(xlabel="Hop distance from the slack bus", ylabel="Mean |power mismatch| per bus (MW, log scale)")
    fig.suptitle("The slack inconsistency is concentrated within a hop or two of the slack", fontsize=11, fontweight="bold")
    if drawn > 1:
        ax.legend(loc="upper right", fontsize=9, ncol=1)
    fig.text(0.01, 0.01, "Physical state (true slack setpoint enforced), all test scenarios; line = median, band = 25-75th percentile over scenarios.\n"
             "|dS| = |V conj(Ybus V) - S| per bus, slack bus excluded; hop 6+ pools buses six or more hops away.\n"
             "Hop bins reached in fewer than half of a grid's scenarios are omitted (IEEE 14: hops 5 and 6+).", color=INK2, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(FIG / "fig2b_slack_locality.png", dpi=200)


def _parse_table(text):
    import re
    rows = []
    for line in text.splitlines():
        m = re.match(r"\s*(.+?)\s+([01]\.\d+)\s+([01]\.\d+)\s+(\d+)\s+([\d.]+)$", line)
        if m:
            rows.append((m.group(1), float(m.group(2)), float(m.group(3)), int(m.group(4)), float(m.group(5))))
    return rows


def fig_screening(tag="case118_ieee_base_seed0", tau=1.01):
    import subprocess, sys
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "analyze_exp02.py"), "--tag", tag, "--tau", str(tau)],
                         capture_output=True, text=True).stdout
    rows = sorted(_parse_table(out), key=lambda r: r[4])
    fig, ax = plt.subplots(figsize=(9, 4.3))
    for i, (name, rec, prec, missed, lu) in enumerate(rows):
        exact = missed == 0
        ax.barh(i, max(lu, 0.04), height=0.62, color=METHOD_COLOR.get(name, BLUE),
                alpha=1.0 if exact else 0.35, edgecolor=SURFACE, linewidth=2)
        note = "finds every violation" if exact else f"misses {missed} violations"
        ax.text(max(lu, 0.04) + 0.06, i, f"{lu:.2f}  ·  {note}", va="center", fontsize=8.5,
                color=INK if name in OURS else INK2, fontweight="bold" if name in OURS else "normal")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    for lab, r in zip(ax.get_yticklabels(), rows):
        if r[0] in OURS:
            lab.set_fontweight("bold"); lab.set_color(INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 6.3)
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="Sparse LU factorizations per contingency (hardware-independent cost)",
           title="Exact screening at about half the industry-standard cost")
    fig.text(0.01, 0.01, f"IEEE 118 N-1 paper-test scenarios (319 held out). Violation: loading > {tau:.0%} or PQ-bus |V| outside limits ± 0.001 p.u. "
             "Faded bars miss violations.", color=INK2, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "fig3_screening.png", dpi=200)


def _parse_policy_log(path):
    """{k: {method: (recall, lu)}} from analyze_exp02 output of the stress sets."""
    import re
    out, k, buf = {}, None, []
    for line in open(path):
        m = re.match(r"stress_n(\d+)_", line)
        if m:
            k = int(m.group(1)); out[k] = {}
            continue
        if k is not None:
            for name, rec, prec, missed, lu in _parse_table(line):
                out[k][name] = (rec, lu)
    return out


def fig_stress(log=R / "policy_with_basecase.log"):
    d = _parse_policy_log(log)
    ks = sorted(d)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.1))
    left = ["DC power flow", "GENCO (as published)", "Base case + 1 Newton step", "GENCO + 1 Newton step", "Certified tiered policy"]
    dy = {"GENCO + 1 Newton step": -9, "Certified tiered policy": 8, "Base case + 1 Newton step": 3, "GENCO (as published)": -6}
    for name in left:
        y = [d[k][name][0] for k in ks]
        a.plot(ks, y, color=METHOD_COLOR[name], lw=2, marker="o", ms=6, markeredgecolor=SURFACE, markeredgewidth=1.5,
               ls="--" if name == "GENCO + 1 Newton step" else "-")
        a.annotate(name.replace(" (as published)", ""), (ks[-1], y[-1]), xytext=(8, dy.get(name, 0)),
                   textcoords="offset points", va="center", fontsize=8.5, color=INK2)
    a.set(xticks=ks, xticklabels=[f"N-{k}" for k in ks], ylim=(0.4, 1.04), xlim=(0.7, 8.2), ylabel="Violation recall",
          title="Calibrated on N-1: zero misses up to N-5")
    a.set_xlabel("Contingency order (N-1 = training distribution)")
    right = ["Full NR, flat start", "Full NR, base-case start", "Full NR, GENCO warm start", "Certified tiered policy"]
    for name in right:
        y = [d[k][name][1] for k in ks]
        b.plot(ks, y, color=METHOD_COLOR[name], lw=2, marker="o", ms=6, markeredgecolor=SURFACE, markeredgewidth=1.5)
        b.annotate(name.replace("Full NR, ", "NR, "), (ks[-1], y[-1]), xytext=(8, 0), textcoords="offset points",
                   va="center", fontsize=8.5, color=INK2)
    b.set(xticks=ks, xticklabels=[f"N-{k}" for k in ks], ylim=(0, 4.6), xlim=(0.7, 8.2),
          ylabel="LU factorizations per contingency", title="40-51% fewer LUs than base-case start")
    b.set_xlabel("Contingency order")
    fig.text(0.01, 0.01, "IEEE 118, GENCO Base, ~660 datakit scenarios per order. Violation: loading > 101% or PQ-bus |V| outside "
             "limits ± 0.001 p.u.\nAll full-NR rows and the tiered policy have recall = precision = 1.000.", color=INK2, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "fig4_stress.png", dpi=200)


if __name__ == "__main__":
    fig_stress()
    fig_layers()
    fig_slack()
    fig_slack_locality()
    fig_screening()
    print("figures in", FIG)
