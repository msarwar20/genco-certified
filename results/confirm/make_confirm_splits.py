"""splits_all for data/confirm/n*: same convention as scripts/make_splits.py (test = all ids, train/val = first 2 / next 2), asserts contiguous ids.
Standalone so that the other (already published) splits are not rewritten."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT = Path(__file__).resolve().parents[2]
for raw in sorted((ROOT / "data" / "confirm").glob("*/case118_ieee/raw")):
    b = pd.read_parquet(raw / "bus_data.parquet", columns=["scenario", "load_scenario_idx"]).drop_duplicates("scenario").sort_values("scenario").reset_index(drop=True)
    assert np.array_equal(b.scenario.to_numpy(), np.arange(len(b))), f"{raw}: ids not contiguous"
    ids = b.scenario.tolist()
    d = raw.parent / "splits_all"; d.mkdir(parents=True, exist_ok=True)
    for name, x in [("train", ids[:2]), ("val", ids[2:4]), ("test", ids)]:
        torch.save(torch.tensor(sorted(x), dtype=torch.long), d / f"{name}.pt")
    print(f"{raw.parent.parent.name}: {len(ids)} scenarios, {b.load_scenario_idx.nunique()} load scenarios, contiguous")
