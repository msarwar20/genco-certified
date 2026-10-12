"""Build the small data bundle the Hugging Face Space runs GENCO on, live.

For each contingency order (N-1 paper test, stress N-2/N-3/N-5 on IEEE 118) we pick
scenarios (the hardest ones by GENCO residual plus a random sample), copy their raw
datakit rows renumbered to contiguous ids (graphkit's streaming loader requires
0..K-1), precompute each load scenario's base-case voltages (classical warm start),
and store a fixed grid layout.

Output: space/bundle/<set>/case118_ieee/raw/*.parquet, splits_all/, base_cases.npz,
        space/bundle/layout_case118.json, space/bundle/manifest.json
"""

import json
import shutil

import networkx as nx
import numpy as np
import pandas as pd
import torch

from genco_certified import acpf
from genco_certified.loading import ROOT

SETS = {
    "N-1": ("data/pf", "splits_paper_test", "case118_ieee_base_seed0"),
    "N-2": ("data/stress/n2", "splits_all", "stress_n2_base"),
    "N-3": ("data/stress/n3", "splits_all", "stress_n3_base"),
    "N-5": ("data/stress/n5", "splits_all", "stress_n5_base"),
}
GRID = "case118_ieee"
OUT = ROOT / "space" / "bundle"
N_HARD, N_RANDOM = 8, 22


def pick(set_tag: str, test_ids: list, rng) -> list:
    sc = pd.read_parquet(ROOT / "results" / "exp02" / f"{set_tag}_scenarios.parquet")
    sc["raw_id"] = sc.get("raw_id", pd.Series(np.array(test_ids)[sc.scenario.to_numpy()]))
    hard = sc.nlargest(N_HARD, "res_raw_max").raw_id.tolist()
    rest = sc[~sc.raw_id.isin(hard)].raw_id.to_numpy()
    return sorted(set(hard) | set(rng.choice(rest, N_RANDOM, replace=False).tolist()))


def main():
    rng = np.random.default_rng(0)
    if OUT.exists():
        shutil.rmtree(OUT)
    manifest = {}
    for name, (root, split, set_tag) in SETS.items():
        src = ROOT / root / GRID
        test_ids = torch.load(src / split / "test.pt").tolist()
        chosen = pick(set_tag, test_ids, rng)
        remap = {old: new for new, old in enumerate(chosen)}
        dst = OUT / name / GRID / "raw"
        dst.mkdir(parents=True, exist_ok=True)
        tables = {}
        for t in ["bus", "gen", "branch", "runtime", "y_bus"]:
            d = pd.read_parquet(src / "raw" / f"{t}_data.parquet")
            if t == "bus":
                load_of = d.drop_duplicates("scenario").set_index("scenario").load_scenario_idx
            tables[t] = d
            sel = d[d.scenario.isin(chosen)].copy()
            sel["scenario"] = sel.scenario.map(remap)
            sel["scenario_partition"] = 0
            sel.drop(columns=["scenario_partition"]).to_parquet(dst / f"{t}_data.parquet", partition_cols=None)
        # base-case voltages per load scenario (needs all variants of that load scenario)
        loads = sorted({int(load_of[s]) for s in chosen})
        base = {}
        for li in loads:
            sel = {t: tables[t][tables[t].load_scenario_idx == li] for t in ["bus", "branch", "gen"]}
            bc = acpf.base_case_voltages(sel["bus"], sel["branch"], sel["gen"])
            if bc["converged"]:
                base[str(li)] = np.stack([np.abs(bc["V"]), np.angle(bc["V"])])
        np.savez_compressed(OUT / name / "base_cases.npz", **base)
        sd = OUT / name / GRID / "splits_all"
        sd.mkdir(parents=True, exist_ok=True)
        ids = torch.arange(len(chosen))
        torch.save(ids[:2], sd / "train.pt"); torch.save(ids[2:4], sd / "val.pt"); torch.save(ids, sd / "test.pt")
        manifest[name] = {"source": root, "original_ids": chosen, "n": len(chosen),
                          "load_scenarios": [int(load_of[s]) for s in chosen]}
        print(f"{name}: {len(chosen)} scenarios, {len(base)} base cases")

    # Fixed layout of the intact IEEE 118 topology (Kamada-Kawai), shared by all views.
    br = tables["branch"]
    edges = br[br.scenario == br.scenario.min()][["from_bus", "to_bus"]].astype(int).to_numpy()
    G = nx.Graph(); G.add_edges_from(map(tuple, edges))
    pos = nx.kamada_kawai_layout(G)
    (OUT / "layout_case118.json").write_text(json.dumps({int(k): [float(v[0]), float(v[1])] for k, v in pos.items()}))
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("bundle size:", sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e6, "MB")


if __name__ == "__main__":
    main()
