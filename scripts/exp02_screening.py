"""Experiment 02: contingency screening with GENCO vs DC-PF vs AC, per branch and bus.

For every paper-test scenario (random N-k outages, in GENCO's training distribution)
records branch loadings |S|/rate and bus voltages under:
  true   - datakit AC power-flow solution
  dc     - classical DC power flow
  raw    - GENCO output as published (its own predicted slack Vm)
  phys   - GENCO with the slack Vm setpoint enforced
  step1  - GENCO + one Newton step (1 sparse factorization)
plus scenario-level certificate features (GENCO's own residual, Newton step norm).

Usage:
    python scripts/exp02_screening.py --grid case118_ieee --size base
"""

import argparse
import time

import numpy as np
import pandas as pd
import torch
from gridfm_graphkit.datasets.masking import RemovePFMask

from genco_certified import acpf
from genco_certified.loading import ROOT, load_genco
from genco_certified.tracing import trace_layers

OUT = ROOT / "results" / "exp02"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="case118_ieee")
    ap.add_argument("--size", default="base")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task_name", default="PowerFlow")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--data_root", default=None, help="folder holding <grid>/raw (default data/pf)")
    ap.add_argument("--split", default="splits_paper_test", help="split folder under <data_root>/<grid>/")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    t0 = time.time()
    Lg = load_genco("pf", args.grid, args.size, args.seed, batch_size=32,
                    overrides={"task": {"task_name": args.task_name}}, weights=args.weights,
                    data_root=args.data_root,
                    split_dir=(ROOT / (args.data_root or "data/pf")) / args.grid / args.split)
    model = Lg.task.model
    root = ROOT / (args.data_root or "data/pf") / args.grid
    test_ids = torch.load(root / args.split / "test.pt").tolist()  # loader order = sorted test ids
    raw = {t: pd.read_parquet(root / "raw" / f"{t}_data.parquet") for t in ["bus", "branch", "gen"]}
    raw = {t: d[d.scenario.isin(test_ids) | d.load_scenario_idx.isin(
        raw["bus"].loc[raw["bus"].scenario.isin(test_ids), "load_scenario_idx"].unique())] for t, d in raw.items()}
    base_cache = {}

    def base_case(load_idx):
        if load_idx not in base_cache:
            sel = {t: d[d.load_scenario_idx == load_idx] for t, d in raw.items()}
            base_cache[load_idx] = acpf.base_case_voltages(sel["bus"], sel["branch"], sel["gen"])
        return base_cache[load_idx]

    branch_rows, bus_rows, scen_rows = [], [], []
    sid = 0
    for batch in Lg.test_loader():
        with torch.no_grad(), trace_layers(model) as tr:
            model(batch)
        state = tr["state"][-1].double().numpy()
        Lg.normalizer.inverse_transform(batch)
        RemovePFMask()(batch)
        ptr = batch["bus"].ptr.numpy()
        for g, p in enumerate(acpf.extract_problems(batch)):
            s = state[ptr[g]:ptr[g + 1]]
            V = {
                "true": p.V_true,
                "raw": s[:, 0] * np.exp(1j * s[:, 1]),
                "phys": p.V_from(s[:, 0], s[:, 1]),
            }
            V["step1"], dx, F = acpf.newton_step(p, V["phys"])
            raw_id = test_ids[sid]
            load_idx = int(raw["bus"].loc[raw["bus"].scenario == raw_id, "load_scenario_idx"].iloc[0])
            bc = base_case(load_idx)
            assert np.allclose(np.abs(p.V_true), raw["bus"].loc[raw["bus"].scenario == raw_id].sort_values("bus").Vm, atol=1e-5)
            Vb0 = p.V_from(np.abs(bc["V"]), np.angle(bc["V"])) if bc["converged"] else p.flat_start()
            V["base1"], dxb, _ = acpf.newton_step(p, Vb0)
            Vdc = p.dc_start()
            loads = {k: acpf.branch_loading(p, acpf.branch_flows(p, v)) for k, v in V.items()}
            loads["dc"] = acpf.branch_loading(p, acpf.dc_branch_flows(p, np.angle(Vdc)))
            nb = len(loads["true"])
            branch_rows.append(pd.DataFrame({"scenario": sid, "branch": np.arange(nb),
                                             **{f"load_{k}": v for k, v in loads.items()}}))
            btype = np.zeros(p.meta["n_bus"], dtype=np.int8)  # 0 PQ, 1 PV, 2 REF
            btype[p.pv], btype[p.ref] = 1, 2
            bus_rows.append(pd.DataFrame({"scenario": sid, "bus": np.arange(p.meta["n_bus"]), "btype": btype,
                                          "vmin": p.vmin, "vmax": p.vmax,
                                          **{f"vm_{k}": np.abs(v) for k, v in V.items()}}))
            nr_g = acpf.newton_pf(p, V["phys"])
            nr_f = acpf.newton_pf(p, p.flat_start())
            nr_d = acpf.newton_pf(p, Vdc)
            nr_b = acpf.newton_pf(p, Vb0)
            Fraw = acpf.mismatch(p, V["raw"])
            scen_rows.append(dict(
                scenario=sid, n_branch=nb, load_MW=p.meta["load_MW"],
                res_raw_max=float(np.abs(Fraw).max()), res_raw_mean=float(np.abs(Fraw).mean()),
                res_phys_max=float(np.abs(F).max()), step_l2=float(np.linalg.norm(dx)),
                step_inf=float(np.abs(dx).max()), step_base_l2=float(np.linalg.norm(dxb)),
                raw_id=raw_id, load_idx=load_idx, base_ok=bool(bc["converged"]),
                nr_iters_base=nr_b["iterations"], nr_conv_base=nr_b["converged"],
                nr_iters_genco=nr_g["iterations"], nr_iters_flat=nr_f["iterations"], nr_iters_dc=nr_d["iterations"],
                nr_conv_genco=nr_g["converged"], nr_conv_flat=nr_f["converged"], nr_conv_dc=nr_d["converged"],
            ))
            sid += 1

    OUT.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{args.grid}_{args.size}_seed{args.seed}"
    br = pd.concat(branch_rows, ignore_index=True)
    bu = pd.concat(bus_rows, ignore_index=True)
    sc = pd.DataFrame(scen_rows)
    br.to_parquet(OUT / f"{tag}_branches.parquet")
    bu.to_parquet(OUT / f"{tag}_buses.parquet")
    sc.to_parquet(OUT / f"{tag}_scenarios.parquet")

    print(f"{tag}: {len(sc)} scenarios, {len(br)} branch-rows in {time.time() - t0:.0f}s")
    truth = br.load_true > 1.0
    print(f"  overloaded branches: {truth.sum()} ({truth.mean():.2%}); scenarios with >=1 overload: "
          f"{br.groupby('scenario').load_true.max().gt(1).mean():.1%}")
    for m in ["dc", "raw", "phys", "step1", "base1"]:
        pred = br[f"load_{m}"] > 1.0
        tp, fp, fn = (pred & truth).sum(), (pred & ~truth).sum(), (~pred & truth).sum()
        mae = (br[f"load_{m}"] - br.load_true).abs()
        print(f"  {m:6s} recall {tp / max(tp + fn, 1):.4f} precision {tp / max(tp + fp, 1):.4f} "
              f"missed {fn:4d} | loading abs err: median {mae.median():.2e} p99 {mae.quantile(.99):.2e} max {mae.max():.2e}")


if __name__ == "__main__":
    main()
