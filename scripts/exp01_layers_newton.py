"""Experiment 01: GENCO layer dynamics, slack-voltage formulation, Newton certificates.

For every paper-test scenario of a PF checkpoint, records:
  * per-layer: power-balance residual (GENCO-consistent and physical) and state error
  * slack-bus Vm error (GENCO's PF mask treats REF Vm as unknown; physically it is a setpoint)
  * one-Newton-step error estimate ||dx|| vs true state error, and error after that step
  * Newton-Raphson iterations from flat start vs warm-started from every GENCO layer

Two voltage vectors are distinguished:
  V_raw  = GENCO output as-is (its own predicted slack Vm) -> matches graphkit's reported metrics
  V_phys = V_raw with the slack Vm reset to its setpoint   -> the operator's AC power-flow problem

Usage:
    python scripts/exp01_layers_newton.py --grid case118_ieee --size base
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from genco_certified import acpf
from genco_certified.loading import ROOT, load_genco
from genco_certified.tracing import trace_layers

OUT = ROOT / "results" / "exp01"


def residual_stats(p, V):
    F = acpf.mismatch(p, V)
    return float(np.max(np.abs(F))), float(np.mean(np.abs(F)))


def err_stats(p, V):
    e = acpf.state_error(p, V)
    nva = len(p.pvpq)
    return float(np.max(np.abs(e[:nva]))), float(np.max(np.abs(e[nva:]))) if len(e) > nva else 0.0


def analyse_scenario(p, states):
    """states: list of [n_bus, 4] arrays (one per layer), normalized units (Vm p.u., Va rad)."""
    L = len(states)
    ref = p.ref
    raw = [s[:, 0] * np.exp(1j * s[:, 1]) for s in states]
    phys = [p.V_from(s[:, 0], s[:, 1]) for s in states]

    layer_rows = []
    for k in range(L):
        rr_max, rr_mean = residual_stats(p, raw[k])
        rp_max, rp_mean = residual_stats(p, phys[k])
        ea, em = err_stats(p, raw[k])
        nr = acpf.newton_pf(p, phys[k])
        layer_rows.append(
            dict(layer=k, res_raw_max=rr_max, res_raw_mean=rr_mean, res_phys_max=rp_max,
                 res_phys_mean=rp_mean, err_va_inf=ea, err_vm_inf=em,
                 ref_vm_err=float(np.max(np.abs(states[k][ref, 0] - np.abs(p.V_true[ref])))),
                 nr_iters=nr["iterations"], nr_conv=nr["converged"])
        )

    Vf = phys[-1]
    e0 = acpf.state_error(p, Vf)
    V1, dx, F = acpf.newton_step(p, Vf)
    e1 = acpf.state_error(p, V1)
    nva = len(p.pvpq)
    flat = acpf.newton_pf(p, p.flat_start())
    dc = acpf.newton_pf(p, p.dc_start())
    J = acpf.jacobian(p, p.V_true)
    sig = np.linalg.svd(J.toarray(), compute_uv=False) if J.shape[0] <= 1500 else np.array([np.nan])
    scen = dict(
        n_bus=p.meta["n_bus"], n_edges=p.meta["n_edges"], load_MW=p.meta["load_MW"],
        ref_vm_err=layer_rows[-1]["ref_vm_err"],
        res_raw_max=layer_rows[-1]["res_raw_max"], res_raw_mean=layer_rows[-1]["res_raw_mean"],
        res_phys_max=layer_rows[-1]["res_phys_max"], res_phys_mean=layer_rows[-1]["res_phys_mean"],
        err_va_inf=float(np.max(np.abs(e0[:nva]))), err_vm_inf=float(np.max(np.abs(e0[nva:]))),
        err_l2=float(np.linalg.norm(e0)),
        est_va_inf=float(np.max(np.abs(dx[:nva]))), est_vm_inf=float(np.max(np.abs(dx[nva:]))),
        est_l2=float(np.linalg.norm(dx)),
        err1_va_inf=float(np.max(np.abs(e1[:nva]))), err1_vm_inf=float(np.max(np.abs(e1[nva:]))),
        err1_l2=float(np.linalg.norm(e1)),
        nr_iters_flat=flat["iterations"], nr_conv_flat=flat["converged"],
        nr_iters_dc=dc["iterations"], nr_conv_dc=dc["converged"],
        nr_iters_genco=layer_rows[-1]["nr_iters"], nr_conv_genco=layer_rows[-1]["nr_conv"],
        jac_sigma_min=float(sig.min()), jac_cond=float(sig.max() / sig.min()),
    )
    return scen, layer_rows


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
    args = ap.parse_args()

    t0 = time.time()
    Lg = load_genco("pf", args.grid, args.size, args.seed, batch_size=args.batch_size,
                    overrides={"task": {"task_name": args.task_name}}, weights=args.weights,
                    data_root=args.data_root,
                    split_dir=(ROOT / (args.data_root or "data/pf")) / args.grid / args.split)
    model = Lg.task.model
    scen_rows, layer_rows = [], []
    sid = 0
    for batch in Lg.test_loader():
        with torch.no_grad(), trace_layers(model) as tr:
            model(batch)
        states = [s.double().numpy() for s in tr["state"]]
        Lg.normalizer.inverse_transform(batch)
        ptr = batch["bus"].ptr.numpy()
        for g, p in enumerate(acpf.extract_problems(batch)):
            sl = slice(ptr[g], ptr[g + 1])
            scen, layers = analyse_scenario(p, [s[sl] for s in states])
            scen_rows.append(dict(scenario=sid, **scen))
            layer_rows += [dict(scenario=sid, **r) for r in layers]
            sid += 1

    OUT.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{args.grid}_{args.size}_seed{args.seed}"
    scen = pd.DataFrame(scen_rows)
    scen.to_parquet(OUT / f"{tag}_scenarios.parquet")
    pd.DataFrame(layer_rows).to_parquet(OUT / f"{tag}_layers.parquet")
    print(f"{tag}: {len(scen)} scenarios in {time.time() - t0:.0f}s")
    q = lambda c: f"median {scen[c].median():.3g} | p95 {scen[c].quantile(.95):.3g} | max {scen[c].max():.3g}"
    print("  slack Vm error (p.u.)      ", q("ref_vm_err"))
    print("  residual reported (MW, max)", q("res_raw_max"))
    print("  residual physical (MW, max)", q("res_phys_max"))
    print("  NR iters flat / DC / GENCO  ", scen.nr_iters_flat.mean().round(2), "/", scen.nr_iters_dc.mean().round(2), "/",
          scen.nr_iters_genco.mean().round(2), "| conv", scen.nr_conv_flat.mean().round(3), "/", scen.nr_conv_dc.mean().round(3),
          "/", scen.nr_conv_genco.mean().round(3))
    from scipy.stats import spearmanr
    print("  Spearman(est, err) l2 %.3f | Va-inf %.3f | Vm-inf %.3f" % (
        spearmanr(scen.est_l2, scen.err_l2)[0], spearmanr(scen.est_va_inf, scen.err_va_inf)[0],
        spearmanr(scen.est_vm_inf, scen.err_vm_inf)[0]))
    print("  error after 1 Newton step (l2): median %.2e (before %.2e)" % (scen.err1_l2.median(), scen.err_l2.median()))


if __name__ == "__main__":
    main()
