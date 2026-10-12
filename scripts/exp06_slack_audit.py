"""Experiment 06: scripted audit of the slack-voltage convention of GENCO's PF task.

graphkit's PF mask treats the REF (slack) bus Vm as unknown, although in datakit data it is a
dispatch setpoint. For every test scenario of a PF checkpoint this script records

  (1) slack Vm error; "reported" mismatch (V_raw = GENCO output as is) vs "physical" mismatch
      (V_phys = same state with REF/PV Vm and REF Va reset to their setpoints);
  (2) locality: hop distance of every bus to the slack, whether the worst physical-mismatch bus is
      a slack neighbour, share of the total |dS| held by 1-hop neighbours, and the mean |dS| per hop
      distance (1..6+). |dS| = |V conj(Ybus V) - Sbus| per bus, slack excluded ("full"). Because P is
      free at the slack and Q is free at PV buses, a second version keeps only the components that
      the power-flow equations constrain (P at PV/PQ, Q at PQ), suffix "_pf";
  (3) uniform re-anchoring: shift (or scale) all PQ-bus Vm by the slack-Vm error;
  (4) mask surgery (IEEE 118 only): feed the true REF Vm to the model ('visible') or only through the
      decoder clamp ('clamp_only').

Outputs (results/exp06): <tag>_scenarios.parquet, <tag>_hops.parquet, <tag>_summary.json and an
aggregate summary.csv (rebuilt from every *_summary.json). CIs are 95% cluster bootstraps over
LOAD scenarios (the ~22 topology variants of one load scenario are not independent).

Usage:
    python scripts/exp06_slack_audit.py --grid case118_ieee --size base
    python scripts/exp06_slack_audit.py --grid case2000_goc --size small --split splits_heldout --batch_size 8
    python scripts/exp06_slack_audit.py --aggregate_only
"""

import argparse
import copy
import json
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.sparse import csgraph

from gridfm_graphkit.datasets.globals import VM_H
from genco_certified import acpf
from genco_certified.loading import ROOT, load_genco
from genco_certified.tracing import trace_layers

OUT = ROOT / "results" / "exp06"
MAX_HOP = 6          # hop profile bins 1..5 and "6+"
N_BOOT = 2000


# ---------------------------------------------------------------------------------------------
# per-scenario analysis
# ---------------------------------------------------------------------------------------------

def hop_distance(p):
    """Hop distance of every bus to the nearest REF bus (-1 if unreachable)."""
    n = len(p.Vm_set)
    A = sp.coo_matrix((np.ones(len(p.f)), (p.f, p.t)), shape=(n, n)).tocsr()
    d = csgraph.dijkstra(A, directed=False, unweighted=True, indices=p.ref, min_only=True)
    return np.where(np.isfinite(d), d, -1).astype(int)


def bus_mismatch(p, V):
    """Per-bus |dS| in MW/MVA: full complex ('full') and constrained components only ('pf')."""
    mis = V * np.conj(p.Ybus @ V) - p.Sbus
    full = np.abs(mis)
    pf2 = np.zeros(len(mis))
    pf2[p.pvpq] += mis.real[p.pvpq] ** 2
    pf2[p.pq] += mis.imag[p.pq] ** 2
    return full, np.sqrt(pf2)


def locality(p, V):
    hop = hop_distance(p)
    full, pf = bus_mismatch(p, V)
    nonref = np.ones(len(hop), bool)
    nonref[p.ref] = False
    ok = nonref & (hop >= 1)
    out = dict(max_hop=int(hop.max()), n_hop1=int(((hop == 1) & ok).sum()),
               frac_hop1=float(((hop == 1) & ok).sum() / max(ok.sum(), 1)))
    rows = []
    for name, d in (("", full), ("_pf", pf)):
        w = int(np.argmax(np.where(ok, d, -1.0)))
        tot = d[ok].sum()
        out[f"worst_bus{name}"] = w
        out[f"worst_hop{name}"] = int(hop[w])
        out[f"worst_adj{name}"] = bool(hop[w] == 1)
        out[f"share_hop1{name}"] = float(d[ok & (hop == 1)].sum() / tot) if tot > 0 else np.nan
        out[f"dS_total{name}"] = float(tot)
    for h in range(1, MAX_HOP + 1):
        sel = ok & ((hop == h) if h < MAX_HOP else (hop >= h))
        if sel.any():
            rows.append(dict(hop=h, n_bus=int(sel.sum()), mean_dS=float(full[sel].mean()), sum_dS=float(full[sel].sum()),
                             mean_dS_pf=float(pf[sel].mean()), sum_dS_pf=float(pf[sel].sum())))
    return out, rows


def max_and_mean(p, V):
    a = np.abs(acpf.mismatch(p, V))
    return float(a.max()), float(a.mean())


def err_l2(p, V):
    return float(np.linalg.norm(acpf.state_error(p, V)))


def analyse_scenario(p, s, surgery=None):
    """s: [n_bus, 4] final decoded state (Vm p.u., Va rad). surgery: {variant: state} or None."""
    ref, pq = p.ref, p.pq
    Vm, Va = s[:, 0], s[:, 1]
    V_raw = Vm * np.exp(1j * Va)
    V_phys = p.V_from(Vm, Va)
    vm_true = np.abs(p.V_true[ref])
    d = float(np.mean(vm_true - Vm[ref]))
    r = float(np.mean(vm_true / Vm[ref]))

    row = dict(n_bus=p.meta["n_bus"], n_edges=p.meta["n_edges"], load_MW=p.meta["load_MW"], n_ref=len(ref),
               ref_vm_true=float(vm_true.mean()), ref_vm_pred=float(Vm[ref].mean()),
               ref_vm_err=float(np.max(np.abs(Vm[ref] - vm_true))), ref_vm_signed=float(-d))
    row["res_raw_max"], row["res_raw_mean"] = max_and_mean(p, V_raw)
    row["res_phys_max"], row["res_phys_mean"] = max_and_mean(p, V_phys)
    row["err_l2"] = err_l2(p, V_phys)
    row["pq_vm_err_max"] = float(np.max(np.abs(Vm[pq] - np.abs(p.V_true[pq])))) if len(pq) else np.nan

    loc, hop_rows = locality(p, V_phys)
    row.update(loc)

    # (3) uniform re-anchoring of PQ-bus Vm
    for name, Vm2 in (("shift", np.where(np.isin(np.arange(len(Vm)), pq), Vm + d, Vm)),
                      ("scale", np.where(np.isin(np.arange(len(Vm)), pq), Vm * r, Vm))):
        V2 = p.V_from(Vm2, Va)
        row[f"{name}_res_phys_max"], _ = max_and_mean(p, V2)
        row[f"{name}_err_l2"] = err_l2(p, V2)

    # (4) mask surgery
    for v in ("vis", "clamp"):
        row[f"{v}_res_phys_max"] = row[f"{v}_err_l2"] = row[f"{v}_nr_iters"] = np.nan
    row["plain_nr_iters"] = np.nan
    if surgery is not None:
        row["plain_nr_iters"] = acpf.newton_pf(p, V_phys)["iterations"]
        for v, sv in surgery.items():
            Vv = p.V_from(sv[:, 0], sv[:, 1])
            row[f"{v}_res_phys_max"], _ = max_and_mean(p, Vv)
            row[f"{v}_err_l2"] = err_l2(p, Vv)
            row[f"{v}_nr_iters"] = acpf.newton_pf(p, Vv)["iterations"]
    return row, hop_rows


def surgery_states(model, batch, mask_value):
    """Final decoded state for the two mask-surgery variants. Returns {variant: [N_bus, 4] float64}."""
    out = {}
    for variant, name in (("visible", "vis"), ("clamp_only", "clamp")):
        b = copy.deepcopy(batch)
        ref = b.mask_dict["REF"]
        b.x_dict["bus"][ref, VM_H] = b.y_dict["bus"][ref, VM_H]
        b.mask_dict["bus"][ref, VM_H] = False
        handle = None
        if variant == "clamp_only":
            def pre(_m, inp, ref=ref):
                x = inp[0].clone()
                x[ref, VM_H] = mask_value
                return (x,) + tuple(inp[1:])
            handle = model.input_proj_bus.register_forward_pre_hook(pre)
        try:
            with torch.no_grad(), trace_layers(model) as tr:
                model(b)
        finally:
            if handle is not None:
                handle.remove()
        out[name] = tr["state"][-1].double().numpy()
    return out


# ---------------------------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------------------------

def cluster_bootstrap(groups, stats, B=N_BOOT, seed=0):
    """95% percentile CIs of `stats` (name -> fn(row_index_array)) resampling load-scenario groups."""
    rng = np.random.default_rng(seed)
    G = len(groups)
    draws = {k: np.empty(B) for k in stats}
    for b in range(B):
        idx = np.concatenate([groups[i] for i in rng.integers(0, G, G)])
        for k, fn in stats.items():
            draws[k][b] = fn(idx)
    return {k: tuple(np.nanpercentile(v, [2.5, 97.5])) for k, v in draws.items()}


def summarise(scen, args, runtime):
    cols = {c: scen[c].to_numpy() for c in scen.columns if scen[c].dtype != object}
    med = lambda c: (lambda i: np.nanmedian(cols[c][i]))
    mean = lambda c: (lambda i: np.nanmean(cols[c][i]))
    stats = {}
    for c in ("ref_vm_err", "res_raw_max", "res_phys_max", "res_raw_mean", "res_phys_mean", "err_l2", "pq_vm_err_max",
              "shift_res_phys_max", "shift_err_l2", "scale_res_phys_max", "scale_err_l2",
              "share_hop1", "share_hop1_pf", "frac_hop1"):
        stats[f"{c}_med"] = med(c)
    stats["res_ratio_med"] = lambda i: np.nanmedian(cols["res_phys_max"][i]) / np.nanmedian(cols["res_raw_max"][i])
    stats["share_hop1_mean"] = mean("share_hop1")
    stats["enrichment_med"] = lambda i: np.nanmedian(cols["share_hop1"][i] / cols["frac_hop1"][i])
    stats["worst_adj_frac"] = mean("worst_adj")
    stats["worst_adj_pf_frac"] = mean("worst_adj_pf")
    stats["shift_res_worse_frac"] = lambda i: np.mean(cols["shift_res_phys_max"][i] > cols["res_phys_max"][i])
    stats["shift_err_worse_frac"] = lambda i: np.mean(cols["shift_err_l2"][i] > cols["err_l2"][i])
    stats["scale_res_worse_frac"] = lambda i: np.mean(cols["scale_res_phys_max"][i] > cols["res_phys_max"][i])
    stats["scale_err_worse_frac"] = lambda i: np.mean(cols["scale_err_l2"][i] > cols["err_l2"][i])
    if scen["vis_res_phys_max"].notna().any():
        for v in ("vis", "clamp"):
            stats[f"{v}_res_phys_max_med"] = med(f"{v}_res_phys_max")
            stats[f"{v}_err_l2_med"] = med(f"{v}_err_l2")
            stats[f"{v}_nr_iters_mean"] = mean(f"{v}_nr_iters")
            stats[f"{v}_err_ratio_med"] = lambda i, v=v: np.nanmedian(cols[f"{v}_err_l2"][i]) / np.nanmedian(cols["err_l2"][i])
        stats["plain_nr_iters_mean"] = mean("plain_nr_iters")

    groups = [np.flatnonzero(scen.load_scenario_idx.to_numpy() == g) for g in np.unique(scen.load_scenario_idx)]
    est = {k: float(fn(np.arange(len(scen)))) for k, fn in stats.items()}
    ci = cluster_bootstrap(groups, stats)
    out = dict(tag=args.tag, grid=args.grid, size=args.size, split=args.split, seed=args.seed,
               task_name=args.task_name, n_scenarios=len(scen), n_load_scenarios=len(groups),
               n_bus=int(scen.n_bus.iloc[0]), runtime_s=runtime, n_boot=N_BOOT)
    for k in stats:
        out[k] = est[k]
        out[k + "_lo"], out[k + "_hi"] = float(ci[k][0]), float(ci[k][1])
    for c in ("ref_vm_err", "res_raw_max", "res_phys_max", "err_l2"):
        out[f"{c}_p95"] = float(scen[c].quantile(0.95))
    out["ref_vm_err_max"] = float(scen.ref_vm_err.max())
    out["res_phys_max_max"] = float(scen.res_phys_max.max())
    out["max_hop_med"] = float(scen.max_hop.median())
    out["n_hop1_med"] = float(scen.n_hop1.median())
    out["worst_hop_med"] = float(scen.worst_hop.median())
    return out


def aggregate():
    """Rebuild summary.csv from every *_summary.json."""
    rows = [json.load(open(f)) for f in sorted(OUT.glob("*_summary.json"))]
    if not rows:
        return
    df = pd.DataFrame(rows)
    order = {"tiny": 0, "small": 1, "base": 2}
    df["_o"] = list(zip(df.n_bus, df["size"].map(order), df.tag))
    df = df.sort_values("_o").drop(columns="_o")
    df.to_csv(OUT / "summary.csv", index=False)
    print("wrote", OUT / "summary.csv", f"({len(df)} checkpoints)")


# ---------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="case118_ieee")
    ap.add_argument("--size", default="base")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--task_name", default="PowerFlow", help="PowerFlowSlackAware for slack-aware models")
    ap.add_argument("--weights", default=None, help="fine-tuned state dict (default: released checkpoint)")
    ap.add_argument("--data_root", default=None, help="folder holding <grid>/raw (default data/pf)")
    ap.add_argument("--split", default="splits_paper_test", help="split folder under <data_root>/<grid>/")
    ap.add_argument("--tag", default=None, help="output name (default: grid_size_seed)")
    ap.add_argument("--surgery", default="auto", choices=["auto", "on", "off"],
                    help="mask surgery variants; auto = only for case118_ieee")
    ap.add_argument("--aggregate_only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        return aggregate()
    args.tag = args.tag or f"{args.grid}_{args.size}_seed{args.seed}"
    do_surgery = args.surgery == "on" or (args.surgery == "auto" and args.grid == "case118_ieee")

    t0 = time.time()
    root = ROOT / (args.data_root or "data/pf") / args.grid
    Lg = load_genco("pf", args.grid, args.size, args.seed, batch_size=args.batch_size,
                    overrides={"task": {"task_name": args.task_name}}, weights=args.weights,
                    data_root=args.data_root, split_dir=root / args.split)
    model = Lg.task.model
    mask_value = float(Lg.config.data.mask_value)

    # load scenario of every test scenario (test.pt ids are sorted = loader order)
    test_ids = torch.load(root / args.split / "test.pt").tolist()
    braw = pd.read_parquet(root / "raw" / "bus_data.parquet", columns=["scenario", "load_scenario_idx", "Pd"])
    braw = braw[braw.scenario.isin(test_ids)]
    load_of = braw.drop_duplicates("scenario").set_index("scenario").load_scenario_idx
    pd_of = braw.groupby("scenario").Pd.sum()

    scen_rows, hop_rows = [], []
    sid = 0
    for batch in Lg.test_loader():
        with torch.no_grad(), trace_layers(model) as tr:
            model(batch)
        state = tr["state"][-1].double().numpy()
        sur = surgery_states(model, batch, mask_value) if do_surgery else None
        Lg.normalizer.inverse_transform(batch)
        ptr = batch["bus"].ptr.numpy()
        for g, p in enumerate(acpf.extract_problems(batch)):
            sl = slice(ptr[g], ptr[g + 1])
            raw_id = test_ids[sid]
            assert abs(p.meta["load_MW"] - pd_of[raw_id]) < 1e-3 * max(pd_of[raw_id], 1.0), \
                f"scenario order mismatch at {sid}: {p.meta['load_MW']} vs {pd_of[raw_id]}"
            row, hrows = analyse_scenario(p, state[sl], None if sur is None else {k: v[sl] for k, v in sur.items()})
            scen_rows.append(dict(scenario=sid, raw_id=int(raw_id), load_scenario_idx=int(load_of[raw_id]), **row))
            hop_rows += [dict(scenario=sid, load_scenario_idx=int(load_of[raw_id]), **h) for h in hrows]
            sid += 1
        if sid % 160 < args.batch_size:
            print(f"  {sid}/{len(test_ids)} scenarios, {time.time() - t0:.0f}s", flush=True)
    assert sid == len(test_ids), (sid, len(test_ids))

    scen = pd.DataFrame(scen_rows)
    scen.to_parquet(OUT / f"{args.tag}_scenarios.parquet")
    pd.DataFrame(hop_rows).to_parquet(OUT / f"{args.tag}_hops.parquet")
    summ = summarise(scen, args, time.time() - t0)
    json.dump(summ, open(OUT / f"{args.tag}_summary.json", "w"), indent=1)

    ci = lambda k, f="{:.3g}": f"{f.format(summ[k])} [{f.format(summ[k + '_lo'])}, {f.format(summ[k + '_hi'])}]"
    print(f"{args.tag}: {len(scen)} scenarios, {summ['n_load_scenarios']} load scenarios, {summ['runtime_s']:.0f}s")
    print("  slack Vm error median", ci("ref_vm_err_med"), "| p95 %.3g | max %.3g" % (summ["ref_vm_err_p95"], summ["ref_vm_err_max"]))
    print("  max mismatch reported", ci("res_raw_max_med"), "-> physical", ci("res_phys_max_med"), "| ratio", ci("res_ratio_med"))
    print("  worst bus adjacent to slack: full", ci("worst_adj_frac", "{:.3f}"), "| pf", ci("worst_adj_pf_frac", "{:.3f}"))
    print("  share of |dS| in 1-hop neighbours (median): full", ci("share_hop1_med", "{:.3f}"), "| pf", ci("share_hop1_pf_med", "{:.3f}"),
          "| uniform expectation %.3f" % summ["frac_hop1_med"])
    print("  re-anchoring: physical max mismatch plain %.3g | shift %.3g | scale %.3g" % (
        summ["res_phys_max_med"], summ["shift_res_phys_max_med"], summ["scale_res_phys_max_med"]),
        "| err_l2 plain %.3g | shift %.3g | scale %.3g" % (summ["err_l2_med"], summ["shift_err_l2_med"], summ["scale_err_l2_med"]))
    if do_surgery:
        print("  mask surgery (median max mismatch / err_l2 / NR iters mean): plain %.3g / %.3g / %.2f" % (
            summ["res_phys_max_med"], summ["err_l2_med"], summ["plain_nr_iters_mean"]))
        for v in ("vis", "clamp"):
            print(f"    {v:6s} %.3g / %.3g / %.2f" % (summ[f"{v}_res_phys_max_med"], summ[f"{v}_err_l2_med"], summ[f"{v}_nr_iters_mean"]))
    aggregate()


if __name__ == "__main__":
    main()
