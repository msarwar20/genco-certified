"""CPU-only GENCO inference time per scenario (batch sizes 1 and 16), torch threads fixed; no MPS/CUDA.

A light companion of bench_runtime.py for grids where only the CPU number is wanted (e.g. while a GPU job is running).
Usage: python scripts/bench_genco_cpu.py --grid case500_goc --size base [--split splits_paper_test] [--threads 4]
Writes results/runtime/<grid>_<size>_cpu.json.
"""
import argparse
import json
import time

import numpy as np
import torch

from genco_certified.loading import ROOT, load_genco


def timeit(fn, repeats=7):
    fn()
    ts = []
    for _ in range(repeats):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="case500_goc")
    ap.add_argument("--size", default="base")
    ap.add_argument("--split", default="splits_paper_test")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    Lg = load_genco("pf", args.grid, args.size, batch_size=16, split_dir=ROOT / "data" / "pf" / args.grid / args.split)
    batch = next(iter(Lg.test_loader()))
    model = Lg.task.model.eval()
    out = {"grid": args.grid, "size": args.size, "torch_threads": torch.get_num_threads(), "loadavg": list(__import__("os").getloadavg())}
    for bs in (1, 16):
        sub = batch.__class__.from_data_list(batch.to_data_list()[:bs])
        with torch.no_grad():
            out[f"genco_cpu_bs{bs}_ms_per_scen"] = 1e3 * timeit(lambda: model(sub)) / bs
    print(json.dumps(out, indent=2))
    (ROOT / "results" / "runtime" / f"{args.grid}_{args.size}_cpu.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
