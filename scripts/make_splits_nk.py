"""Train/val split of the mixed N-k training set (data/train_nk), grouped by load scenario.

Writes data/train_nk/case118_ieee/splits_nk/{train,val,test}.pt: ~90% / ~10% of the LOAD SCENARIOS (all variants of a
load scenario go to the same side), test = copy of val (graphkit needs a non-empty test file; it is never used).
Usage: python scripts/make_splits_nk.py [--data_root data/train_nk] [--seed 0] [--val_frac 0.1]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--data_root", default="data/train_nk")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--val_frac", type=float, default=0.1)
a = ap.parse_args()
gd = Path(a.data_root) / "case118_ieee"
b = pd.read_parquet(gd / "raw" / "bus_data.parquet", columns=["scenario", "load_scenario_idx"])
b = b.drop_duplicates("scenario").sort_values("scenario").reset_index(drop=True)
assert np.array_equal(b.scenario.to_numpy(), np.arange(len(b))), "scenario ids not contiguous 0..N-1"
ls = np.sort(b.load_scenario_idx.unique())
rng = np.random.default_rng(a.seed)
val_ls = set(rng.choice(ls, size=int(round(a.val_frac * len(ls))), replace=False).tolist())
isval = b.load_scenario_idx.isin(val_ls).to_numpy()
train, val = b.scenario[~isval].tolist(), b.scenario[isval].tolist()
out = gd / "splits_nk"
out.mkdir(parents=True, exist_ok=True)
for name, ids in [("train", train), ("val", val), ("test", val)]:
    torch.save(torch.tensor(sorted(ids), dtype=torch.long), out / f"{name}.pt")
assert not set(b.load_scenario_idx[~isval]) & set(b.load_scenario_idx[isval])
print(f"{out}: train {len(train)} ({len(ls) - len(val_ls)} load scenarios), val {len(val)} ({len(val_ls)}), test = val")
