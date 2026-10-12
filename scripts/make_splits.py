"""Create evaluation splits that are not published by the authors.

* data/stress/n{k}/case118_ieee/splits_all  - test = every generated scenario
* data/pf/case2000_goc/splits_heldout         - leak-free held-out set: scenarios outside the
  model's 20k training subset that share no load scenario with any train/val scenario
"""

import glob
import json
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]


def save(d: Path, train, val, test):
    d.mkdir(parents=True, exist_ok=True)
    for name, ids in [("train", train), ("val", val), ("test", test)]:
        torch.save(torch.tensor(sorted(ids), dtype=torch.long), d / f"{name}.pt")


for raw in sorted((ROOT / "data" / "stress").glob("n*/case118_ieee/raw")):
    ids = sorted(pd.read_parquet(raw / "bus_data.parquet", columns=["scenario"]).scenario.unique().tolist())
    save(raw.parent / "splits_all", ids[:2], ids[2:4], ids)
    print(raw.parent.parent.name, len(ids))

g = ROOT / "data" / "pf" / "case2000_goc"
if g.exists():
    sp = json.load(open(glob.glob(str(ROOT / "checkpoints/genco-pf-datakit/mlflow/*/case2000_goc/*/seed0/*/artifacts/stats/*splits.json"))[0]))
    b = pd.read_parquet(g / "raw" / "bus_data.parquet", columns=["scenario", "load_scenario_idx"]).drop_duplicates("scenario")
    seen = set(sp["train"]) | set(sp["val"])
    ls_seen = set(b[b.scenario.isin(seen)].load_scenario_idx)
    held = b[~b.load_scenario_idx.isin(ls_seen) & (b.load_scenario_idx != b.load_scenario_idx.max())].scenario.tolist()
    tv = sorted(b[b.scenario.isin(seen)].scenario.tolist())
    save(g / "splits_heldout", tv[: len(tv) // 2], tv[len(tv) // 2:], held)
    print("case2000_goc held-out", len(held))
