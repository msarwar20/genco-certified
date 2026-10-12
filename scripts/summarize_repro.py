"""Render results/repro/summary_{pf,opf}.csv into a markdown comparison table."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEY_METRICS = {
    "pf": [
        "Avg. active res. (MW)",
        "Avg. reactive res. (MVar)",
        "PBE Mean",
        "PBE Max",
        "RMSE-PQ Vm (p.u.)",
        "RMSE-PQ Va (radians)",
        "DC Avg. active res. (MW)",
    ],
    "opf": [
        "Mean optimality gap (%)",
        "Avg. active res. (MW)",
        "Avg. reactive res. (MVar)",
        "RMSE PG generators (MW)",
        "Mean branch thermal violation from (MVA)",
        "Mean Qg violation",
        "DC Mean optimality gap (%)",
        "DC Avg. active res. (MW)",
    ],
}


def fmt(x: float) -> str:
    if pd.isna(x):
        return "–"
    return f"{x:.3g}" if abs(x) >= 1e-3 or x == 0 else f"{x:.2e}"


def table(task: str) -> str:
    f = ROOT / "results" / "repro" / f"summary_{task}.csv"
    if not f.exists():
        return ""
    df = pd.read_csv(f)
    df = df[df["metric"].isin(KEY_METRICS[task])]
    lines = [f"### {task.upper()}", "", "| run | metric | ours | paper | Δ % |", "|---|---|---|---|---|"]
    for run, g in df.groupby("run", sort=False):
        g = g.set_index("metric").reindex([m for m in KEY_METRICS[task] if m in set(g["metric"])])
        for m, r in g.iterrows():
            d = "–" if pd.isna(r["rel_diff_%"]) else f"{r['rel_diff_%']:+.1f}"
            lines.append(f"| {run.replace(f'{task}_', '')} | {m} | {fmt(r['ours'])} | {fmt(r['paper'])} | {d} |")
    return "\n".join(lines)


if __name__ == "__main__":
    md = "\n\n".join(t for t in (table("pf"), table("opf")) if t)
    out = ROOT / "results" / "repro" / "summary.md"
    out.write_text(md + "\n")
    print(md)
