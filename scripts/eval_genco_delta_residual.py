"""Evaluate the RESIDUAL GENCO-Delta model as a Newton warm start / contingency screen.

Thin wrapper around scripts/eval_genco_delta.py (imported, not modified). For `--task_name
PowerFlowContingencyResidual` (the default here) it builds the `GNS_heterogeneous_residual` model, and

  * without `--weights`: loads the RELEASED weights and zeroes the delta head, i.e. the model at initialisation
    (0 training steps). Its output is the base case, so its Newton iteration counts must equal the base-case
    columns (a sanity check, tag it e.g. residual_init_s0__paper);
  * with `--weights checkpoints/finetuned/<tag>/best_model_state_dict.pt`: the fine-tuned residual model.

All other flags and outputs are those of eval_genco_delta.py (`--task_name PowerFlow` without weights evaluates the
released model as before).

    python scripts/eval_genco_delta_residual.py --delta_inputs vm_va_flows --seed 0 \\
        --weights checkpoints/finetuned/residual_vm_va_flows_s0/best_model_state_dict.pt --tag residual_vm_va_flows_s0__paper

Aggregation over runs, residual arms included (written next to the eval folder as residual_summary_*.csv/.md, the
files of the main matrix are not touched):
    python scripts/eval_genco_delta_residual.py --aggregate --out_dir results/genco_delta/eval [--prefix residual_]
"""

import argparse
import json
import re
import sys

sys.dont_write_bytecode = True  # do not touch scripts/__pycache__ (other jobs import eval_genco_delta.py)
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

import eval_genco_delta as ed  # noqa: E402  scripts/eval_genco_delta.py, unmodified
from genco_certified import genco_residual as gr  # noqa: E402  registers model / task

ARM_RE = re.compile(r"^(?:smoke_|dry_)?(released|control|slack_aware|delta_vm_va_flows|delta_vm_va|"
                    r"residual_init|residual_nk_vm_va_flows_lr3e-4|residual_vm_va_flows_lr3e-4|residual_vm_va_flows|residual_vm_va)_s(\d+)$")
ARM_ORDER = {"released": 0, "control": 1, "slack_aware": 2, "delta_vm_va": 3, "delta_vm_va_flows": 4,
             "residual_init": 5, "residual_vm_va": 6, "residual_vm_va_flows": 7,
             "residual_vm_va_flows_lr3e-4": 8, "residual_nk_vm_va_flows_lr3e-4": 9}


def aggregate(out_dir: Path, prefix: str = "") -> None:
    """Like eval_genco_delta.aggregate, but knows the residual arms and writes residual_{prefix}summary_*."""
    rows = []
    for f in sorted(out_dir.glob("*.json")):
        run, _, set_ = f.stem.partition("__")
        if (prefix and not run.startswith(prefix)) or (not prefix and run.startswith("smoke_")):
            continue
        m = ARM_RE.match(run)
        if not m:
            continue
        rows.append(dict(run=run, arm=m.group(1), seed=int(m.group(2)), set=set_, **ed._flatten(json.loads(f.read_text()))))
    if not rows:
        print("nothing to aggregate in", out_dir)
        return
    df = pd.DataFrame(rows)
    stem = f"residual_{prefix}summary_"
    df.to_csv(out_dir.parent / f"{stem}runs.csv", index=False)
    num = [c for c in df.columns if c not in ("run", "arm", "seed", "set")]
    g = df.groupby(["set", "arm"])
    mean, sd, cnt = g[num].mean(), g[num].std(ddof=1), g.size().rename("n_seeds")
    by = pd.concat([cnt, mean.add_suffix("_mean"), sd.add_suffix("_sd")], axis=1).reset_index()
    by.to_csv(out_dir.parent / f"{stem}by_arm.csv", index=False)
    key = ["iters_g_e2", "iters_g_e4", "iters_g_e6", "iters_b_e6", "iters_g_minus_b_e6", "err_l2_median_g", "g1_el_miss", "b1_el_miss", "g2_el_miss"]
    lines = ["| set | arm | seeds | " + " | ".join(key) + " |", "|" + "---|" * (len(key) + 3)]
    by["_o"] = by.arm.map(ARM_ORDER)
    for _, r in by.sort_values(["set", "_o"]).iterrows():
        cells = [f"{r[k + '_mean']:.3g} ± {0 if pd.isna(r[k + '_sd']) else r[k + '_sd']:.2g}" for k in key]
        lines.append(f"| {r['set']} | {r['arm']} | {int(r['n_seeds'])} | " + " | ".join(cells) + " |")
    (out_dir.parent / f"{stem}by_arm.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {out_dir.parent}/{stem}runs.csv, {stem}by_arm.csv, {stem}by_arm.md")


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--aggregate", action="store_true")
    pre.add_argument("--out_dir", default=None)
    pre.add_argument("--prefix", default="")
    pre.add_argument("--task_name", default=gr.RESIDUAL_TASK_NAME)
    known = pre.parse_known_args()[0]
    if known.aggregate:
        aggregate(Path(known.out_dir) if known.out_dir else ed.ROOT / "results" / "genco_delta" / "eval", known.prefix)
        return
    if not any(a == "--task_name" or a.startswith("--task_name=") for a in sys.argv[1:]):
        sys.argv += ["--task_name", known.task_name]
    # basecase_store / delta_inputs overrides and the meta record apply to the residual task as to the delta task
    ed.DELTA = known.task_name if known.task_name in (gr.RESIDUAL_TASK_NAME, ed.DELTA) else ed.DELTA
    ed.load_genco = gr.load_genco_residual  # model.type override; no --weights -> released weights with a zeroed delta head
    ed.main()


if __name__ == "__main__":
    main()
