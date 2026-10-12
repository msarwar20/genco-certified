"""Wall-clock benchmark: GENCO inference vs Newton-Raphson on the same machine.

Measures, per scenario (median over repeats, warm caches, data already in memory):
  * GENCO forward pass at several batch sizes, CPU and (if available) MPS/CUDA
  * one sparse LU solve of the PF Jacobian (scipy/SuperLU) and full NR from flat / DC / GENCO
Run on an otherwise idle machine. LU counts (exp01/exp02) stay the primary,
hardware-independent cost measure; these timings make them concrete.

Usage: python scripts/bench_runtime.py --grid case118_ieee --size base
"""

import argparse
import copy
import json
import time

import numpy as np
import torch

from genco_certified import acpf
from genco_certified.loading import ROOT, load_genco


def timeit(fn, repeats=5, sync=lambda: None):
    fn(); sync()  # warm-up
    ts = []
    for _ in range(repeats):
        t = time.perf_counter(); fn(); sync(); ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def early_exit(model, n_layers):
    """Copy of GENCO that stops after `n_layers` correction steps (decoder heads are shared)."""
    m = copy.deepcopy(model)
    m.layers = m.layers[:n_layers]
    m.num_layers = n_layers
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="case118_ieee")
    ap.add_argument("--size", default="base")
    ap.add_argument("--split", default="splits_paper_test")
    ap.add_argument("--n_scen", type=int, default=128)
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    out = {"grid": args.grid, "size": args.size, "torch_threads": torch.get_num_threads()}
    Lg = load_genco("pf", args.grid, args.size, batch_size=args.n_scen,
                    split_dir=ROOT / "data" / "pf" / args.grid / args.split)
    batch = next(iter(Lg.test_loader()))
    n = batch.num_graphs
    model = Lg.task.model.eval()

    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []) + (["cuda"] if torch.cuda.is_available() else [])
    for dev in devices:
        sync = {"mps": torch.mps.synchronize, "cuda": torch.cuda.synchronize}.get(dev, lambda: None)
        m = copy.deepcopy(model).to(dev)
        for bs in [1, 16, n]:
            sub = batch.__class__.from_data_list(batch.to_data_list()[:bs]).to(dev)
            with torch.no_grad():
                t = timeit(lambda: m(sub), sync=sync)
            out[f"genco_{dev}_bs{bs}_ms_per_scen"] = 1e3 * t / bs
        # Early exit: keep only the first 9 correction layers (exp01: +0.1 NR iteration).
        m9 = early_exit(m, 9)
        sub = batch.clone().to(dev)  # .to() is in-place for PyG batches
        with torch.no_grad():
            out[f"genco_{dev}_bs{n}_9layers_ms_per_scen"] = 1e3 * timeit(lambda: m9(sub), sync=sync) / n

    Lg.normalizer.inverse_transform(batch)
    probs = acpf.extract_problems(batch)[: min(n, 64)]
    import scipy.sparse.linalg as spla
    lu_t, it = [], {"flat": [], "dc": []}
    for p in probs:
        V = p.flat_start()
        J, F = acpf.jacobian(p, V), acpf.mismatch(p, V)
        lu_t.append(timeit(lambda: spla.spsolve(J, -F), repeats=3))
        for name, V0 in [("flat", p.flat_start()), ("dc", p.dc_start())]:
            it[name].append(timeit(lambda: acpf.newton_pf(p, V0), repeats=3))
    out["lu_solve_ms"] = 1e3 * float(np.median(lu_t))
    out["nr_flat_ms"] = 1e3 * float(np.median(it["flat"]))
    out["nr_dc_ms"] = 1e3 * float(np.median(it["dc"]))
    out["n_unknowns"] = int(len(probs[0].pvpq) + len(probs[0].pq))
    print(json.dumps(out, indent=2))
    dst = ROOT / "results" / "runtime"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / f"{args.grid}_{args.size}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
