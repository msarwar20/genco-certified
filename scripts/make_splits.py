"""Create evaluation splits that are not published by the authors.

* data/{stress,stress100,calib,exhaustive}/<set>/case118_ieee/splits_all - test = every generated
  scenario (train/val = the first 2 / next 2 ids; unused by evaluation but must be non-empty).
  Scenario ids are checked to be contiguous 0..N-1, which graphkit's streaming loader requires.
* data/pf/case2000_goc/splits_heldout{,_seed42,_allseeds} - leak-free held-out sets: scenarios whose
  load scenario never appears in the train/val split of the model seed(s) concerned. Each seed trains
  on its own random 20k-scenario subset of the 218k, so a held-out set for seed 0 contains scenarios
  that seed 42 trained on; `_allseeds` is the (much smaller) set that is clean for every seed. The
  last load scenario on disk is dropped because it may continue in the next (not downloaded) partition.

Usage: python scripts/make_splits.py
"""

import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
GENERATED = ["stress", "stress100", "calib", "exhaustive"]  # data/<root>/<set>/case118_ieee/raw
HELDOUT_GRIDS = ["case2000_goc"]


def save(d: Path, train, val, test):
    d.mkdir(parents=True, exist_ok=True)
    for name, ids in [("train", train), ("val", val), ("test", test)]:
        torch.save(torch.tensor(sorted(ids), dtype=torch.long), d / f"{name}.pt")


def scenario_table(raw: Path) -> pd.DataFrame:
    """One row per scenario (scenario, load_scenario_idx), after asserting ids are 0..N-1."""
    b = pd.read_parquet(raw / "bus_data.parquet", columns=["scenario", "load_scenario_idx"])
    b = b.drop_duplicates("scenario").sort_values("scenario").reset_index(drop=True)
    assert np.array_equal(b.scenario.to_numpy(), np.arange(len(b))), f"{raw}: scenario ids are not contiguous 0..N-1"
    return b


def published_splits(grid: str) -> dict[int, dict]:
    """{model seed: published split}; sizes (small/tiny/...) share one split per seed, seeds differ."""
    out = {}
    pattern = f"checkpoints/genco-pf-datakit/mlflow/*/{grid}/*/seed*/*/artifacts/stats/{grid}_scenario_splits.json"
    for f in sorted(glob.glob(str(ROOT / pattern))):
        seed = int(re.search(r"/seed(\d+)/", f).group(1))
        sp = json.load(open(f))
        assert out.setdefault(seed, sp) == sp, f"{f}: split differs between model sizes of seed {seed}"
    assert out, f"no published splits for {grid}; run scripts/fetch_data.py first"
    return out


for root in GENERATED:
    for raw in sorted((DATA / root).glob("*/case118_ieee/raw")):
        if not (raw / "bus_data.parquet").exists():
            print(f"skip {raw.parent.parent}: no bus_data.parquet (generation unfinished or failed)")
            continue
        b = scenario_table(raw)
        ids = b.scenario.tolist()
        save(raw.parent / "splits_all", ids[:2], ids[2:4], ids)
        print(f"{root}/{raw.parent.parent.name}: {len(ids)} scenarios, {b.load_scenario_idx.nunique()} load scenarios, contiguous")

for grid in HELDOUT_GRIDS:
    g = DATA / "pf" / grid
    if not (g / "raw").exists():
        continue
    b = scenario_table(g / "raw")
    by_seed = published_splits(grid)
    last = b.load_scenario_idx.max()  # may be cut by the partition boundary
    # splits_heldout          -> seed-0 models (default; what the experiments use)
    # splits_heldout_seed<S>  -> seed-S models
    # splits_heldout_allseeds -> leak-free for every released seed at once (much smaller)
    variants = {"splits_heldout" if s == 0 else f"splits_heldout_seed{s}": [s] for s in sorted(by_seed)}
    variants["splits_heldout_allseeds"] = sorted(by_seed)
    for name, seeds in variants.items():
        seen = set().union(*[set(by_seed[s]["train"]) | set(by_seed[s]["val"]) for s in seeds])
        ls_seen = set(b[b.scenario.isin(seen)].load_scenario_idx)
        held = b[~b.load_scenario_idx.isin(ls_seen) & (b.load_scenario_idx != last)]
        tv = sorted(b[b.scenario.isin(seen)].scenario.tolist())
        save(g / name, tv[: len(tv) // 2], tv[len(tv) // 2:], held.scenario.tolist())
        print(
            f"{grid}/{name} (seeds {seeds}): {len(held)} scenarios from {held.load_scenario_idx.nunique()} load scenarios "
            f"({len(b)} on disk, {b.load_scenario_idx.nunique()} load scenarios, {len(ls_seen)} touched by train/val; "
            f"last load scenario {int(last)} dropped)"
        )
