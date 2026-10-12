"""Download the first K scenario partitions of GridFM datakit datasets and build
evaluation splits that contain only the paper's own held-out test scenarios.

graphkit's streaming loader needs scenario ids 0..N-1, so we always fetch the
contiguous prefix of partitions (200 scenarios each). Dataset positions then equal
the paper's scenario ids, and the published `*_scenario_splits.json` can be
intersected with the downloaded range.

Usage:
    python scripts/fetch_data.py --task pf --grids case14_ieee case118_ieee --partitions 30
"""

import argparse
import glob
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
TABLES = ["bus_data", "gen_data", "branch_data", "runtime_data", "y_bus_data"]
SCENARIOS_PER_PARTITION = 200


def find_split_file(task: str, grid: str) -> Path:
    """Locate the paper's published train/val/test scenario ids for this grid."""
    repo_dir = ROOT / "checkpoints" / f"genco-{task}-datakit"
    # Splits are identical across model sizes (same seed); use whichever size was released.
    pattern = f"mlflow/*/{grid}/*/seed0/*/artifacts/stats/{grid}_scenario_splits.json"
    hits = sorted(repo_dir.glob(pattern))
    if not hits:
        snapshot_download(
            f"gridfm/genco-{task}-datakit",
            local_dir=repo_dir,
            allow_patterns=[pattern],
        )
        hits = sorted(repo_dir.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"No published split for {task}/{grid}")
    return hits[0]


def fetch(task: str, grid: str, n_partitions: int) -> None:
    raw_dir = ROOT / "data" / task / grid / "raw"
    patterns = [
        f"{table}.parquet/scenario_partition={i}/*"
        for table in TABLES
        for i in range(n_partitions)
    ]
    snapshot_download(
        f"gridfm/{task}_small_{grid}",
        repo_type="dataset",
        local_dir=raw_dir,
        allow_patterns=patterns + ["n_scenarios.txt", "args.log"],
    )

    # Sanity check: scenario ids must be exactly 0..N-1 for the streaming loader.
    import pandas as pd

    ids = pd.read_parquet(raw_dir / "bus_data.parquet", columns=["scenario"])["scenario"].unique()
    n = n_partitions * SCENARIOS_PER_PARTITION
    assert ids.min() == 0 and ids.max() == len(ids) - 1, (ids.min(), ids.max(), len(ids))
    if len(ids) != n:
        print(f"  note: {len(ids)} scenarios downloaded (expected {n})")
    n = len(ids)

    split = json.loads(find_split_file(task, grid).read_text())
    split_dir = ROOT / "data" / task / grid / "splits_paper_test"
    split_dir.mkdir(parents=True, exist_ok=True)
    in_range = {k: sorted(i for i in v if i < n) for k, v in split.items()}
    # train/val are unused by `evaluate` (normalizer stats come from the checkpoint)
    # but must be non-empty for the datamodule.
    for k in ["train", "val", "test"]:
        torch.save(torch.tensor(in_range[k], dtype=torch.long), split_dir / f"{k}.pt")
    print(
        f"{task}/{grid}: {n} scenarios on disk, paper test scenarios in range: "
        f"{len(in_range['test'])} (of {len(split['test'])} in full test set)"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["pf", "opf"], required=True)
    ap.add_argument("--grids", nargs="+", default=["case14_ieee", "case30_ieee", "case57_ieee", "case118_ieee"])
    ap.add_argument("--partitions", type=int, default=30)
    args = ap.parse_args()
    for grid in args.grids:
        fetch(args.task, grid, args.partitions)


if __name__ == "__main__":
    main()
