"""Build the pre-contingency (base-case) store GENCO-Delta conditions on.

For a data root `<data_root>/<grid>/raw` (datakit parquet) this solves, per load scenario, the
intact-topology AC power flow from all of the load scenario's variants
(`acpf.base_case_voltages`) and writes

    <data_root>/<grid>/basecase_store.pt      {scenario_id -> (Vm [p.u.], Va [rad])}

covering EVERY scenario id of those load scenarios, hence every id of every split.

Usage:
    python scripts/build_basecase_store.py --data_root data/pf --grid case118_ieee
    python scripts/build_basecase_store.py --data_root data/stress/n3 --grid case118_ieee
    python scripts/build_basecase_store.py --all          # IEEE 118 paper set + stress n1,n2,n3,n5 + per-seed paper splits

Per-seed paper splits: the released checkpoints of seeds 1 and 42 were trained on DIFFERENT scenario splits than
seed 0 (about 70-80% of seed 0's test scenarios are in their training sets), so a study over init seeds must
evaluate seed s on seed s's own published test ids. `--seed_splits 1 42` writes
`<data_root>/<grid>/splits_paper_test_seed<s>/{train,val,test}.pt` (published ids restricted to the downloaded range).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from genco_certified.basecase_store import BaseCaseStore, build_store, split_ids
from genco_certified.loading import ROOT

DEFAULT_ROOTS = ["data/pf", "data/stress/n1", "data/stress/n2", "data/stress/n3", "data/stress/n5"]


def check_store(store: BaseCaseStore, grid_dir: Path, n_probe: int = 200, seed: int = 0) -> dict:
    """Coverage of every split id, plus how far the base case is from the post-contingency labels (probe)."""
    out = {"splits": {}}
    for name, ids in split_ids(grid_dir).items():
        miss = store.missing(ids)
        out["splits"][name] = {"n": len(ids), "missing": len(miss)}
        if miss:
            print(f"  WARNING {name}: {len(miss)} of {len(ids)} ids not covered, e.g. {miss[:5]}")
    # difference to the stored (post-contingency) solution of a random sample of scenarios
    import pandas as pd
    bus = grid_dir / "raw" / "bus_data.parquet"
    part = sorted(bus.glob("scenario_partition=*"))[0] if bus.is_dir() else bus
    b = pd.read_parquet(part, columns=["scenario", "bus", "Vm", "Va", "PQ", "PV", "REF"])
    sids = np.array(sorted(set(b.scenario.unique()) & set(torch.nonzero(store.row_of >= 0).flatten().tolist())))
    pick = np.random.default_rng(seed).choice(sids, size=min(n_probe, len(sids)), replace=False)
    dvm, dva, dvm_pv, dvm_ref = [], [], [], []
    for s in pick:
        g = b[b.scenario == s].sort_values("bus")
        Vm, Va = store.lookup(int(s))
        pq, pv, ref = g.PQ.to_numpy() == 1, g.PV.to_numpy() == 1, g.REF.to_numpy() == 1
        dvm.append(np.abs(Vm.numpy()[pq] - g.Vm.to_numpy()[pq]).max())
        dva.append(np.abs(np.angle(np.exp(1j * (Va.numpy() - np.deg2rad(g.Va.to_numpy()))))[~ref]).max())
        dvm_pv.append(np.abs(Vm.numpy()[pv] - g.Vm.to_numpy()[pv]).max() if pv.any() else 0.0)
        dvm_ref.append(np.abs(Vm.numpy()[ref] - g.Vm.to_numpy()[ref]).max())
    out["probe"] = {"n": len(pick), "median_max_dVm_pq": float(np.median(dvm)), "median_max_dVa_rad": float(np.median(dva)),
                    "max_dVm_pv_setpoint": float(np.max(dvm_pv)), "max_dVm_ref_setpoint": float(np.max(dvm_ref))}
    p = out["probe"]
    print(f"  probe ({p['n']} scenarios): base vs post-contingency label | median of per-scenario max |dVm| at PQ "
          f"{p['median_max_dVm_pq']:.2e} p.u., |dVa| {p['median_max_dVa_rad']:.2e} rad | max |dVm| at PV setpoints "
          f"{p['max_dVm_pv_setpoint']:.1e}, at REF {p['max_dVm_ref_setpoint']:.1e}")
    return out


def make_seed_splits(grid_dir: Path, grid: str, size: str, seed: int, task: str = "pf") -> Path:
    """Published train/val/test ids of the released `<size>/seed<seed>` checkpoint, restricted to the downloaded scenarios."""
    import glob
    import pandas as pd
    from huggingface_hub import snapshot_download
    repo_dir = ROOT / "checkpoints" / f"genco-{task}-datakit"
    pattern = f"mlflow/*/{grid}/{size}/seed{seed}/*/artifacts/stats/{grid}_scenario_splits.json"
    hits = sorted(glob.glob(str(repo_dir / pattern)))
    if not hits:
        snapshot_download(f"gridfm/genco-{task}-datakit", local_dir=repo_dir, allow_patterns=[pattern])
        hits = sorted(glob.glob(str(repo_dir / pattern)))
    if not hits:
        raise FileNotFoundError(f"no published split for {grid}/{size}/seed{seed}")
    split = json.loads(Path(hits[0]).read_text())
    bus = grid_dir / "raw" / "bus_data.parquet"
    n = int(pd.read_parquet(bus, columns=["scenario"]).scenario.max()) + 1
    out = grid_dir / (f"splits_paper_test_seed{seed}" if seed else "splits_paper_test")
    out.mkdir(parents=True, exist_ok=True)
    for k in ("train", "val", "test"):
        torch.save(torch.tensor(sorted(i for i in split[k] if i < n), dtype=torch.long), out / f"{k}.pt")
    print(f"  {out}: " + ", ".join(f"{k} {len(torch.load(out / f'{k}.pt'))}" for k in ("train", "val", "test")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/pf", help="folder holding <grid>/raw (default data/pf)")
    ap.add_argument("--grid", default="case118_ieee")
    ap.add_argument("--out", default=None, help="default <data_root>/<grid>/basecase_store.pt")
    ap.add_argument("--all", action="store_true", help=f"build for {DEFAULT_ROOTS} (grid case118_ieee)")
    ap.add_argument("--seed_splits", type=int, nargs="*", default=None,
                    help="also write splits_paper_test_seed<s> for these released Base seeds (published ids, downloaded range)")
    ap.add_argument("--size", default="base", help="released model size for --seed_splits")
    ap.add_argument("--no_check", action="store_true")
    ap.add_argument("--force", action="store_true", help="rebuild even if the store file exists")
    args = ap.parse_args()

    jobs = [(r, "case118_ieee") for r in DEFAULT_ROOTS] if args.all else [(args.data_root, args.grid)]
    seeds = [1, 42] if (args.all and args.seed_splits is None) else (args.seed_splits or [])
    for root, grid in jobs:
        data_root = Path(root) if Path(root).is_absolute() else ROOT / root
        grid_dir = data_root / grid
        if root == "data/pf" or not args.all:
            for sd in seeds:
                make_seed_splits(grid_dir, grid, args.size, sd)
        out = Path(args.out) if (args.out and not args.all) else grid_dir / "basecase_store.pt"
        if out.exists() and not args.force:
            print(f"{out} exists (use --force to rebuild)")
            store = BaseCaseStore.load(out)
        else:
            print(f"building {out}")
            t0 = time.time()
            store = build_store(grid_dir / "raw")
            store.save(out)
            print(f"  wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.0f}s)")
        if not args.no_check:
            res = check_store(store, grid_dir)
            if any(v["missing"] for v in res["splits"].values()):
                sys.exit(f"{out}: some split ids are not covered")


if __name__ == "__main__":
    main()
