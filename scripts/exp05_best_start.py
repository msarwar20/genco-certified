"""Experiment 05: best-of-two warm start chosen without any factorization.

For each contingency, evaluate the power mismatch ||F||_inf at GENCO's output and at the
pre-contingency base case (two cheap function evaluations, no LU), start Newton from the
smaller one, and count iterations. Compared with always-GENCO and always-base-case.

Usage: python scripts/exp05_best_start.py --grid case118_ieee --size base
       python scripts/exp05_best_start.py --grid case2000_goc --size small --split splits_heldout
"""

import argparse

import numpy as np
import pandas as pd
import torch

from genco_certified import acpf
from genco_certified.loading import ROOT, load_genco
from genco_certified.tracing import trace_layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="case118_ieee")
    ap.add_argument("--size", default="base")
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--split", default="splits_paper_test")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    root = ROOT / (args.data_root or "data/pf") / args.grid
    Lg = load_genco("pf", args.grid, args.size, batch_size=16, data_root=args.data_root, split_dir=root / args.split)
    test_ids = torch.load(root / args.split / "test.pt").tolist()
    raw = {t: pd.read_parquet(root / "raw" / f"{t}_data.parquet") for t in ["bus", "branch", "gen"]}
    load_of = raw["bus"].drop_duplicates("scenario").set_index("scenario").load_scenario_idx
    base_cache, rows, sid = {}, [], 0
    for batch in Lg.test_loader():
        with torch.no_grad(), trace_layers(Lg.task.model) as tr:
            Lg.task.model(batch)
        st = tr["state"][-1].double().numpy()
        Lg.normalizer.inverse_transform(batch)
        ptr = batch["bus"].ptr.numpy()
        for g, p in enumerate(acpf.extract_problems(batch)):
            li = int(load_of[test_ids[sid]])
            if li not in base_cache:
                sel = {t: d[d.load_scenario_idx == li] for t, d in raw.items()}
                base_cache[li] = acpf.base_case_voltages(sel["bus"], sel["branch"], sel["gen"])
            bc = base_cache[li]
            s = st[ptr[g]:ptr[g + 1]]
            Vg = p.V_from(s[:, 0], s[:, 1])
            Vb = p.V_from(np.abs(bc["V"]), np.angle(bc["V"]))
            fg, fb = np.abs(acpf.mismatch(p, Vg)).max(), np.abs(acpf.mismatch(p, Vb)).max()
            ig, ib = acpf.newton_pf(p, Vg)["iterations"], acpf.newton_pf(p, Vb)["iterations"]
            rows.append(dict(scenario=sid, F_genco=fg, F_base=fb, it_genco=ig, it_base=ib,
                             it_best=ig if fg <= fb else ib, it_oracle=min(ig, ib), pick_genco=fg <= fb))
            sid += 1
    df = pd.DataFrame(rows)
    tag = args.tag or f"{args.grid}_{args.size}"
    out = ROOT / "results" / "exp05"
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / f"{tag}.parquet")
    print(f"{tag}: {len(df)} scenarios | NR iterations: always-GENCO {df.it_genco.mean():.2f} | always-base {df.it_base.mean():.2f} "
          f"| best-of-two by mismatch {df.it_best.mean():.2f} (picks GENCO {df.pick_genco.mean():.0%}) | oracle {df.it_oracle.mean():.2f}")


if __name__ == "__main__":
    main()
