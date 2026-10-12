"""Experiment 02: contingency screening with GENCO vs DC-PF vs AC, per branch and bus.

For every test scenario (random N-k outages) records branch loadings |S|/rate and bus voltages under:
  true      - datakit AC power-flow solution
  dc        - datakit's stored PowerModels DC solution: max(|pf_dc|, |pt_dc|)/rate_a per in-service
              branch, merged per undirected bus pair by max (same keys/merge as acpf.branch_loading)
  dc_ours   - our own DC power flow (Im(Y_ft) susceptances), kept for reference only
  raw       - GENCO output as published (its own predicted slack Vm)
  phys      - GENCO with the slack Vm setpoint enforced
  g1,g2,g3  - GENCO (phys) + k Newton steps, k = 1..3     (step1 = g1 alias)
  b1,b2,b3  - pre-contingency base case + k Newton steps  (base1 = b1 alias)
  nr        - our own converged Newton-Raphson from the GENCO start (used for escalated decisions)
plus scenario-level features: GENCO's own residual, Newton step norms ||dx_k|| of every stored step,
NR iteration counts to 1e-2 / 1e-4 / 1e-6 MW from flat / base / GENCO / DC starts, and a
re-certification of datakit's stored solution (max |mismatch| of the raw-parquet problem and of the
graphkit problem at V_true).

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
N_STEPS = 3
TOLS = {2: 1e-2, 4: 1e-4, 6: 1e-6}  # MW


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
    # Per-scenario row groups of the evaluated scenarios (re-certification, datakit DC).
    rows = {t: {s: g for s, g in d[d.scenario.isin(test_ids)].groupby("scenario")} for t, d in raw.items()}
    load_of = raw["bus"].drop_duplicates("scenario").set_index("scenario").load_scenario_idx
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
            raw_id = test_ids[sid]
            load_idx = int(load_of.loc[raw_id])
            bus_s = rows["bus"][raw_id].sort_values("bus")
            assert np.allclose(np.abs(p.V_true), bus_s.Vm.to_numpy(), atol=1e-5)
            V = {
                "true": p.V_true,
                "raw": s[:, 0] * np.exp(1j * s[:, 1]),
                "phys": p.V_from(s[:, 0], s[:, 1]),
            }
            F = acpf.mismatch(p, V["phys"])
            bc = base_case(load_idx)
            Vb0 = p.V_from(np.abs(bc["V"]), np.angle(bc["V"])) if bc["converged"] else p.flat_start()
            Vdc = p.dc_start()
            # Newton trajectories (>= N_STEPS steps) from GENCO and from the base case; NR from flat/DC.
            starts = {"genco": V["phys"], "base": Vb0, "flat": p.flat_start(), "dc": Vdc}
            traj = {k: acpf.newton_trajectory(p, v0, n_steps=(N_STEPS if k in ("genco", "base") else 0))
                    for k, v0 in starts.items()}
            for pref, k in (("g", "genco"), ("b", "base")):
                assert len(traj[k]["V"]) >= N_STEPS, f"{k} trajectory shorter than {N_STEPS} steps (scenario {sid})"
                for i in range(N_STEPS):
                    V[f"{pref}{i + 1}"] = traj[k]["V"][i]
            V["step1"], V["base1"] = V["g1"], V["b1"]
            it_g = traj["genco"]["iterations"]
            V["nr"] = V["phys"] if it_g == 0 else traj["genco"]["V"][it_g - 1]

            loads = {k: acpf.branch_loading(p, acpf.branch_flows(p, v)) for k, v in V.items()}
            loads["dc_ours"] = acpf.branch_loading(p, acpf.dc_branch_flows(p, np.angle(Vdc)))
            loads["dc"] = acpf.dc_loading_from_raw(p, rows["branch"][raw_id])  # asserts key-set equality
            nb = len(loads["true"])
            assert all(len(v) == nb for v in loads.values())
            branch_rows.append(pd.DataFrame({"scenario": sid, "branch": np.arange(nb),
                                             **{f"load_{k}": v for k, v in loads.items()}}))
            btype = np.zeros(p.meta["n_bus"], dtype=np.int8)  # 0 PQ, 1 PV, 2 REF
            btype[p.pv], btype[p.ref] = 1, 2
            bus_rows.append(pd.DataFrame({"scenario": sid, "bus": np.arange(p.meta["n_bus"]), "btype": btype,
                                          "vmin": p.vmin, "vmax": p.vmax,
                                          **{f"vm_{k}": np.abs(v) for k, v in V.items()}}))

            # Re-certification of datakit's stored solution on a problem built from the raw parquet.
            p_raw = acpf.problem_from_raw(bus_s, rows["branch"][raw_id], rows["gen"][raw_id])
            rec_raw = float(np.abs(acpf.mismatch(p_raw, p_raw.V_true)).max())
            rec_gk = float(np.abs(acpf.mismatch(p, p.V_true)).max())

            Fraw = acpf.mismatch(p, V["raw"])
            row = dict(
                scenario=sid, n_branch=nb, load_MW=p.meta["load_MW"],
                res_raw_max=float(np.abs(Fraw).max()), res_raw_mean=float(np.abs(Fraw).mean()),
                res_phys_max=float(np.abs(F).max()),
                step_l2=float(np.linalg.norm(traj["genco"]["dx"][0])), step_inf=float(np.abs(traj["genco"]["dx"][0]).max()),
                step_base_l2=float(np.linalg.norm(traj["base"]["dx"][0])),
                raw_id=raw_id, load_idx=load_idx, base_ok=bool(bc["converged"]),
                recert_raw_max=rec_raw, recert_gk_max=rec_gk,
                nr_state_err_max=float(np.abs(acpf.state_error(p, V["nr"])).max()),
            )
            for pref, k in (("g", "genco"), ("b", "base")):
                for i in range(N_STEPS):
                    row[f"step_{pref}{i + 1}_l2"] = float(np.linalg.norm(traj[k]["dx"][i]))
            for k, tr_ in traj.items():
                row[f"nr_iters_{k}"] = tr_["iterations"]
                row[f"nr_conv_{k}"] = tr_["converged"]
                for e, tol in TOLS.items():
                    n_it = acpf.iterations_to_tol(tr_["history"], tol)
                    row[f"nri_{k}_e{e}"] = float(n_it) if n_it >= 0 else np.nan
            scen_rows.append(row)
            sid += 1

    OUT.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{args.grid}_{args.size}_seed{args.seed}"
    br = pd.concat(branch_rows, ignore_index=True)
    bu = pd.concat(bus_rows, ignore_index=True)
    sc = pd.DataFrame(scen_rows)
    br.to_parquet(OUT / f"{tag}_branches.parquet")
    bu.to_parquet(OUT / f"{tag}_buses.parquet")
    sc.to_parquet(OUT / f"{tag}_scenarios.parquet")

    print(f"{tag}: {len(sc)} scenarios ({sc.load_idx.nunique()} load scenarios), {len(br)} branch-rows in {time.time() - t0:.0f}s")
    truth = br.load_true > 1.0
    print(f"  overloaded branches: {truth.sum()} ({truth.mean():.2%}); scenarios with >=1 overload: "
          f"{br.groupby('scenario').load_true.max().gt(1).mean():.1%}")
    print(f"  recertification max |mismatch| at datakit solution: raw-parquet problem {sc.recert_raw_max.max():.2e}, "
          f"graphkit problem {sc.recert_gk_max.max():.2e}; our NR vs datakit state: {sc.nr_state_err_max.max():.2e}")
    for m in ["dc", "dc_ours", "raw", "phys", "g1", "g2", "g3", "b1", "b2", "b3"]:
        pred = br[f"load_{m}"] > 1.0
        tp, fp, fn = (pred & truth).sum(), (pred & ~truth).sum(), (~pred & truth).sum()
        mae = (br[f"load_{m}"] - br.load_true).abs()
        print(f"  {m:8s} recall {tp / max(tp + fn, 1):.4f} precision {tp / max(tp + fp, 1):.4f} "
              f"missed {fn:4d} | loading abs err: median {mae.median():.2e} p99 {mae.quantile(.99):.2e} max {mae.max():.2e}")
    print("  mean NR iterations (1e-2 / 1e-4 / 1e-6 MW): " + "; ".join(
        f"{k} " + "/".join(f"{sc[f'nri_{k}_e{e}'].mean():.2f}" for e in TOLS) for k in ["flat", "base", "genco", "dc"]))


if __name__ == "__main__":
    main()
