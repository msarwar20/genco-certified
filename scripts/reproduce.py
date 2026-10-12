"""Reproduce published GENCO metrics on the paper's own held-out test scenarios.

For each (task, grid, size, seed) this downloads the released checkpoint, rewrites its
YAML to evaluate only the paper-test scenarios fetched by `fetch_data.py`, runs the
unmodified `gridfm_graphkit evaluate`, and compares every metric against the
authors' published CSVs.

Usage:
    python scripts/reproduce.py --task pf --grids case118_ieee --sizes base small tiny
"""

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import pandas as pd
import yaml
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "repro"
WEIGHTS = {"pf": "best_model_state_dict.pt", "opf": "last.pt"}
# PF numbers come from a dedicated eval run; OPF numbers from the test pass after training.
REF_STAGE = {"pf": "eval", "opf": "train"}


def checkpoint_files(task: str, grid: str, size: str, seed: int) -> dict:
    repo = f"gridfm/genco-{task}-datakit"
    local = ROOT / "checkpoints" / f"genco-{task}-datakit"
    sub = f"{grid}/{size}/seed{seed}"
    get = lambda name: Path(hf_hub_download(repo, f"{sub}/{name}", local_dir=local))
    return {
        "weights": get(WEIGHTS[task]),
        "stats": get("normalizer_stats.pt"),
        "config": get(f"{grid}_{size}_seed{seed}.yaml"),
    }


def write_eval_config(task: str, grid: str, src: Path, dst: Path, accelerator: str) -> None:
    cfg = yaml.safe_load(src.read_text())
    cfg["data"]["split_by_load_scenario_idx"] = False
    cfg["data"]["split_from_existing_files"] = str(ROOT / "data" / task / grid / "splits_paper_test")
    cfg["data"]["workers"] = 0
    cfg["training"]["accelerator"] = accelerator
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(yaml.safe_dump(cfg, sort_keys=False))


def read_metrics(test_dir: Path, grid: str) -> pd.Series:
    """Flatten the Metric,Value and RMSE CSVs graphkit writes into one Series."""
    out = {}
    for name in ["metrics", "ac_dc_metrics", "opf_ac_dc_metrics"]:
        f = test_dir / f"{grid}_{name}.csv"
        if f.exists():
            for _, r in pd.read_csv(f).iterrows():
                out[r["Metric"]] = float(r["Value"])
    f = test_dir / f"{grid}_RMSE.csv"
    if f.exists():
        rmse = pd.read_csv(f).set_index("Metric")
        for row in rmse.index:
            for col in rmse.columns:
                out[f"{row} {col}"] = float(rmse.loc[row, col])
    return pd.Series(out)


def reference_dir(task: str, grid: str, size: str, seed: int) -> Path | None:
    pattern = str(
        ROOT / "checkpoints" / f"genco-{task}-datakit" / "mlflow" / REF_STAGE[task]
        / grid / size / f"seed{seed}" / "*" / "artifacts" / "test"
    )
    hits = glob.glob(pattern)
    return Path(hits[0]) if hits else None


def run_one(task: str, grid: str, size: str, seed: int, accelerator: str) -> pd.DataFrame:
    name = f"{task}_{grid}_{size}_seed{seed}"
    files = checkpoint_files(task, grid, size, seed)
    cfg_path = OUT / "configs" / f"{name}.yaml"
    write_eval_config(task, grid, files["config"], cfg_path, accelerator)
    log_dir = OUT / "mlruns"
    cmd = [
        "gridfm_graphkit", "evaluate",
        "--config_conversion", "auto_inline",
        "--config", str(cfg_path),
        "--data_path", str(ROOT / "data" / task),
        "--model_path", str(files["weights"]),
        "--normalizer_stats", str(files["stats"]),
        "--log_dir", str(log_dir),
        "--exp_name", "repro",
        "--run_name", name,
        "--compute_dc_ac_metrics",
        "--num_workers", "0",
    ]
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    log_file = OUT / "logs" / f"{name}.log"
    env = {**os.environ, "MLFLOW_ALLOW_FILE_STORE": "true"}
    with open(log_file, "w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
    if proc.returncode != 0:
        print(f"[FAIL] {name} (see {log_file})")
        print("".join(open(log_file).readlines()[-15:]))
        return pd.DataFrame()

    runs = sorted(log_dir.glob("*/*/artifacts/test"), key=lambda p: p.stat().st_mtime)
    ours = read_metrics(runs[-1], grid)
    ref_dir = reference_dir(task, grid, size, seed)
    ref = read_metrics(ref_dir, grid) if ref_dir else pd.Series(dtype=float)
    df = pd.DataFrame({"ours": ours, "paper": ref})
    df["rel_diff_%"] = 100 * (df["ours"] - df["paper"]) / df["paper"].abs().where(df["paper"].abs() > 1e-12)
    df.insert(0, "run", name)
    print(f"[ok] {name}")
    return df.reset_index(names="metric")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["pf", "opf"], required=True)
    ap.add_argument("--grids", nargs="+", default=["case14_ieee", "case30_ieee", "case57_ieee", "case118_ieee"])
    ap.add_argument("--sizes", nargs="+", default=["base"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--accelerator", default="cpu")
    args = ap.parse_args()

    frames = [run_one(args.task, g, s, args.seed, args.accelerator) for g in args.grids for s in args.sizes]
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    out = OUT / f"summary_{args.task}.csv"
    if out.exists():
        old = pd.read_csv(out)
        df = pd.concat([old[~old["run"].isin(df["run"])], df], ignore_index=True)
    df.to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
