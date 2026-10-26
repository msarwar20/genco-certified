"""Precompute the GENCO Operator Console for the static Hugging Face Space.

Runs the live console code (space/engine.py and the panel builders in space/app.py) on every
bundled scenario at N-1, N-2, N-3 and N-5 and writes what the Inspect and Screen tabs show to
space_static/data/*.json. Nothing here recomputes physics: every number comes from the engine.

    python scripts/build_static_space.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import re

import numpy as np
from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
SPACE = ROOT / "space"
OUT = ROOT / "space_static"
sys.path.insert(0, str(SPACE))      # as on the Space: engine.py, app.py and the vendored package
os.chdir(SPACE)

import app  # noqa: E402  (builds the Engine; gradio is imported but nothing is launched)
from engine import EPS_V, TAU  # noqa: E402
from genco_certified.acpf import branch_keys  # noqa: E402

E = app.E
ORDERS = app.ORDERS
MD = MarkdownIt("commonmark", {"html": True}).enable("table")


def findings_html():
    """app.FINDINGS rendered once, with each [[fig:...]] marker turned into an image."""
    parts = []
    for chunk in re.split(r"\[\[fig:([\w.]+)\]\]", app.FINDINGS):
        if chunk.endswith(".png"):
            parts.append(f'<figure><img src="assets/{chunk}" alt="{chunk[:-4]}" loading="lazy"></figure>')
        elif chunk.strip():
            parts.append(MD.render(chunk))
    return "\n".join(parts)


def r(x, sig=4):
    """Round to `sig` significant digits (keeps tiny Newton mismatches readable on a log axis)."""
    x = float(x)
    if x == 0 or not np.isfinite(x):
        return 0.0 if x == 0 else None
    return float(f"{x:.{sig}g}")


def branches(s):
    """Unique branches as [from, to, true loading], same de-duplication as app.grid_map."""
    p, load = s.p, s.load["true"]
    seen, out = set(), []
    for k, f, t in zip(branch_keys(p), p.f, p.t):
        if k in seen:
            continue
        seen.add(k)
        out.append([int(f), int(t), round(float(load[k]), 5)])
    return out


def scenario_record(s):
    p = s.p
    vm = s.vm["true"]
    bad = [int(b) for b in p.pq if vm[b] < p.vmin[b] - EPS_V or vm[b] > p.vmax[b] + EPS_V]
    pq = p.pq
    vt = s.vm["true"][pq]
    real = int((s.load["true"] > TAU).sum() + ((vt < p.vmin[pq] - EPS_V) | (vt > p.vmax[pq] + EPS_V)).sum())
    vt_df = app.violation_table(s)
    return {
        "idx": s.idx,
        "label": s.label,
        "outages": s.outages,
        "ref": int(p.ref[0]),
        "branches": branches(s),
        "bus_mis": [r(x, 3) for x in s.bus_mis],
        "vm_true": [round(float(x), 4) for x in vm],
        "v_viol_buses": bad,
        "layer_raw": [r(x) for x in s.layer_raw],
        "layer_phys": [r(x) for x in s.layer_phys],
        "hist_genco": [r(x) for x in s.hist_genco],
        "hist_base": [r(x) for x in s.hist_base],
        "hist_flat": [r(x) for x in s.hist_flat],
        "card": {
            "markdown": app.decision_card(s),
            "html": MD.render(app.decision_card(s)),
            "accepted_step": int(s.accepted_step),
            "accepted_step_base": int(s.accepted_step_base),
            "lu_policy": int(s.lu_policy),
            "lu_policy_base": int(s.lu_policy_base),
            "lu_base_start": len(s.hist_base) - 1,
            "lu_flat_start": len(s.hist_flat) - 1,
            "lu_genco_start": len(s.hist_genco) - 1,
            "mismatch_own_slack_mw": r(s.layer_raw[-1]),
            "mismatch_true_setpoint_mw": r(s.layer_phys[-1]),
            "step_l2": r(s.step_l2),
            "violations_true": real,
        },
        "violations": {"columns": list(vt_df.columns), "rows": vt_df.astype(str).values.tolist()},
    }


def main():
    data = OUT / "data"
    data.mkdir(parents=True, exist_ok=True)
    meta = {
        "tau": TAU, "eps_v": EPS_V, "rule": E.rule, "c_genco": E.c_genco, "c_base": E.c_base,
        "orders": ORDERS,
        "layout": {str(k): [round(v[0], 4), round(v[1], 4)] for k, v in E.layout.items()},
        "screen": {},
        "built": time.strftime("%Y-%m-%d %H:%M"),
        "findings_html": findings_html(),
    }
    for order in ORDERS:
        t0 = time.time()
        S = E.scenarios(order)
        recs = [scenario_record(s) for s in S]
        (data / f"{order}.json").write_text(json.dumps({"order": order, "scenarios": recs}, separators=(",", ":"),
                                                       ensure_ascii=False))
        df, note = app.batch(order)
        meta["screen"][order] = {"columns": list(df.columns), "rows": df.astype(str).values.tolist(), "note": note}
        print(f"{order}: {len(recs)} scenarios in {time.time() - t0:.1f} s")
    (data / "meta.json").write_text(json.dumps(meta, separators=(",", ":"), ensure_ascii=False))
    assets = OUT / "assets"
    assets.mkdir(exist_ok=True)
    for png in sorted((SPACE / "assets").glob("*.png")):
        shutil.copy2(png, assets / png.name)
    total = sum(f.stat().st_size for f in data.glob("*.json"))
    print(f"data: {total / 1e6:.2f} MB in {data}")


if __name__ == "__main__":
    main()
