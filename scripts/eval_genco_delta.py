"""Evaluate a GENCO variant as a Newton warm start and as a contingency screen (works for released GENCO too).

For every scenario of a split (per-scenario rows -> parquet, cluster-bootstrap summary -> JSON):
  * Newton-Raphson iterations to 1e-2 / 1e-4 / 1e-6 MW from (g) the model output with the slack/PV
    setpoints enforced (`PFProblem.V_from`), (b) the pre-contingency base case (from the store), (f) a flat
    start; with the acpf Newton history, so iteration counts are comparable with exp02's `nri_*` columns.
  * state error of the model output and of the base case against datakit's solution (unknowns Va[pv,pq], Vm[pq]),
    and the physical power mismatch of both.
  * element decisions after k Newton steps (model: k = 0, 1, 2; base case: k = 1, 2): branch overloads
    |S|/rate > tau and PQ-bus voltage violations (outside [vmin - eps, vmax + eps]) against the truth:
    true / false positives and misses per scenario.
Summary: means and medians with 95% cluster-bootstrap CIs, clusters = load scenarios (all variants of a load
scenario are resampled together), ratio-of-sums for rates.

Usage (released GENCO, apples to apples with the fine-tuned arms):
    python scripts/eval_genco_delta.py --task_name PowerFlow --tag released_base_s0__paper
    python scripts/eval_genco_delta.py --task_name PowerFlowContingencyAware --delta_inputs vm_va_flows \\
        --weights checkpoints/finetuned/<tag>/best_model_state_dict.pt --tag <tag>__paper
    python scripts/eval_genco_delta.py ... --data_root data/stress/n3 --split splits_all --tag <tag>__n3
"""

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from gridfm_graphkit.datasets.masking import RemovePFMask

from genco_certified import acpf
from genco_certified.basecase_store import BaseCaseStore
from genco_certified.loading import ROOT, load_genco

DELTA = "PowerFlowContingencyAware"
TOLS = {2: 1e-2, 4: 1e-4, 6: 1e-6}  # MW
N_STEPS = 2
STARTS = {"g": "model output (setpoints enforced)", "b": "base case", "f": "flat start"}
DECISIONS = ["g0", "g1", "g2", "b1", "b2"]  # start letter + number of Newton steps


# ----------------------------------------------------------------------------- per-scenario analysis
def _decisions(p, V, tau, eps_v, L_true, tb, tv):
    L = acpf.branch_loading(p, acpf.branch_flows(p, V))
    vm = np.abs(V[p.pq])
    pb = L > tau
    pv = (vm < p.vmin[p.pq] - eps_v) | (vm > p.vmax[p.pq] + eps_v)
    return dict(tpb=int((pb & tb).sum()), fpb=int((pb & ~tb).sum()), fnb=int((~pb & tb).sum()),
                tpv=int((pv & tv).sum()), fpv=int((pv & ~tv).sum()), fnv=int((~pv & tv).sum()),
                max_dload=float(np.abs(L - L_true).max()), max_dvm_pq=float(np.abs(vm - np.abs(p.V_true[p.pq])).max()))


def analyse(p, pred, V_base, base_ok, tau, eps_v):
    """One scenario. pred: [n,2] model (Vm, Va). Returns a dict of scalar columns."""
    Vg = p.V_from(pred[:, 0], pred[:, 1])
    Vraw = pred[:, 0] * np.exp(1j * pred[:, 1])
    Vb = p.V_from(np.abs(V_base), np.angle(V_base))
    starts = {"g": Vg, "b": Vb, "f": p.flat_start()}
    traj = {k: acpf.newton_trajectory(p, v0, n_steps=(N_STEPS if k in ("g", "b") else 0)) for k, v0 in starts.items()}
    row = dict(n_bus=p.meta["n_bus"], n_branch=int(len(acpf.branch_key_values(p))), load_MW=p.meta["load_MW"], base_ok=bool(base_ok),
               res_raw_max=float(np.abs(acpf.mismatch(p, Vraw)).max()), res_phys_max=float(np.abs(acpf.mismatch(p, Vg)).max()),
               res_base_max=float(np.abs(acpf.mismatch(p, Vb)).max()),
               ref_vm_err=float(np.abs(pred[p.ref, 0] - np.abs(p.V_true[p.ref])).max()))
    nva = len(p.pvpq)
    for k, V in (("g", Vg), ("b", Vb)):
        e = acpf.state_error(p, V)
        row.update({f"err_l2_{k}": float(np.linalg.norm(e)), f"err_va_inf_{k}": float(np.abs(e[:nva]).max()),
                    f"err_vm_inf_{k}": float(np.abs(e[nva:]).max()) if len(e) > nva else 0.0})
    for k, t in traj.items():
        row[f"nr_iters_{k}"] = int(t["iterations"])
        row[f"nr_conv_{k}"] = bool(t["converged"])
        for e, tol in TOLS.items():
            n_it = acpf.iterations_to_tol(t["history"], tol)
            row[f"nri_{k}_e{e}"] = float(n_it) if n_it >= 0 else np.nan
    L_true = acpf.branch_loading(p, acpf.branch_flows(p, p.V_true))
    tb = L_true > tau
    vm_t = np.abs(p.V_true[p.pq])
    tv = (vm_t < p.vmin[p.pq] - eps_v) | (vm_t > p.vmax[p.pq] + eps_v)
    row.update(truth_b=int(tb.sum()), truth_v=int(tv.sum()))
    V_of = {"g0": Vg, "g1": traj["g"]["V"][0], "g2": traj["g"]["V"][1], "b1": traj["b"]["V"][0], "b2": traj["b"]["V"][1]}
    for m in DECISIONS:
        row.update({f"{m}_{k}": v for k, v in _decisions(p, V_of[m], tau, eps_v, L_true, tb, tv).items()})
    return row


# ----------------------------------------------------------------------------- cluster bootstrap
class Boot:
    def __init__(self, df, n_boot=2000, seed=0):
        self.df, self.B = df, n_boot
        self.rng = np.random.default_rng(seed)
        keys = df.load_idx.to_numpy()
        self.groups = [np.flatnonzero(keys == k) for k in np.unique(keys)]
        self.G = len(self.groups)
        self.draws = self.rng.integers(0, self.G, size=(n_boot, self.G))

    def _ci(self, vals):
        v = np.asarray(vals, float)
        v = v[np.isfinite(v)]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [np.nan, np.nan]

    def mean(self, x, label=None):
        x = np.asarray(x, float)
        s = np.array([x[g].sum() for g in self.groups]); n = np.array([len(g) for g in self.groups], float)
        est = float(s.sum() / n.sum())
        return {"mean": est, "ci95": self._ci((s[self.draws].sum(1)) / n[self.draws].sum(1))}

    def ratio(self, num, den):
        num, den = np.asarray(num, float), np.asarray(den, float)
        a = np.array([num[g].sum() for g in self.groups]); b = np.array([den[g].sum() for g in self.groups])
        est = float(a.sum() / b.sum()) if b.sum() > 0 else float("nan")
        with np.errstate(invalid="ignore", divide="ignore"):
            boot = a[self.draws].sum(1) / b[self.draws].sum(1)
        return {"value": est, "ci95": self._ci(boot), "num": float(a.sum()), "den": float(b.sum())}

    def median(self, x):
        x = np.asarray(x, float)
        meds = [np.median(x[np.concatenate([self.groups[i] for i in d])]) for d in self.draws[: min(self.B, 1000)]]
        return {"median": float(np.median(x)), "ci95": self._ci(meds)}


def summarize(df, tau, eps_v, n_boot=2000):
    bt = Boot(df, n_boot)
    out = {"n_scenarios": int(len(df)), "n_load_scenarios": int(bt.G), "n_bootstrap": n_boot}
    it = {}
    for e in TOLS:
        for k in "gbf":
            x = df[f"nri_{k}_e{e}"].to_numpy(float)
            x = np.where(np.isnan(x), df[f"nr_iters_{k}"].to_numpy(float), x)  # unconverged: charge the iterations taken
            it[(k, e)] = x
            out[f"nr_iters_{k}_e{e}"] = bt.mean(x)
            out[f"nr_unconverged_{k}_e{e}"] = int(df[f"nri_{k}_e{e}"].isna().sum())
        out[f"nr_iters_g_minus_b_e{e}"] = bt.mean(it[("g", e)] - it[("b", e)])
        out[f"nr_iters_g_minus_f_e{e}"] = bt.mean(it[("g", e)] - it[("f", e)])
        out[f"frac_fewer_iters_than_base_e{e}"] = bt.mean((it[("g", e)] < it[("b", e)]).astype(float))
        out[f"frac_more_iters_than_base_e{e}"] = bt.mean((it[("g", e)] > it[("b", e)]).astype(float))
    for k in "gb":
        for c in ("err_l2", "err_va_inf", "err_vm_inf"):
            out[f"{c}_{k}"] = {**bt.mean(df[f"{c}_{k}"]), **{"median": bt.median(df[f"{c}_{k}"])}}
    out["err_l2_g_over_b"] = bt.ratio(df.err_l2_g, df.err_l2_b)
    out["res_phys_max_g"] = bt.median(df.res_phys_max)
    out["res_base_max"] = bt.median(df.res_base_max)
    out["res_raw_max_g"] = bt.median(df.res_raw_max)
    out["ref_vm_err_g"] = bt.median(df.ref_vm_err)
    tr = df.truth_b + df.truth_v
    out["truth_elements"] = int(tr.sum())
    out["truth_contingencies"] = int((tr > 0).sum())
    for m in DECISIONS:
        tp, fp, fn = df[f"{m}_tpb"] + df[f"{m}_tpv"], df[f"{m}_fpb"] + df[f"{m}_fpv"], df[f"{m}_fnb"] + df[f"{m}_fnv"]
        cont_miss = ((tr > 0) & ((tp + fp) == 0)).astype(float)
        out[f"{m}_el_miss_count"] = int(fn.sum())
        out[f"{m}_el_fa_count"] = int(fp.sum())
        out[f"{m}_el_miss_rate"] = bt.ratio(fn, tr)
        out[f"{m}_el_recall"] = bt.ratio(tp, tr)
        out[f"{m}_el_precision"] = bt.ratio(tp, tp + fp)
        out[f"{m}_cont_miss_count"] = int(cont_miss.sum())
        out[f"{m}_cont_miss_rate"] = bt.ratio(cont_miss, (tr > 0).astype(float))
        out[f"{m}_max_dload"] = bt.median(df[f"{m}_max_dload"])
        out[f"{m}_max_dvm_pq"] = bt.median(df[f"{m}_max_dvm_pq"])
    out["tau"], out["eps_v"] = tau, eps_v
    return out


# ----------------------------------------------------------------------------- aggregation over runs
ARM_RE = re.compile(r"^(?:smoke_|dry_)?(released|control|delta_vm_va_flows|delta_vm_va)_s(\d+)$")


def _flatten(s):
    row = dict(n=s["n_scenarios"], n_load=s["n_load_scenarios"], truth_elements=s["truth_elements"],
               truth_contingencies=s["truth_contingencies"])
    for e in TOLS:
        for k in "gbf":
            row[f"iters_{k}_e{e}"] = s[f"nr_iters_{k}_e{e}"]["mean"]
        row[f"iters_g_minus_b_e{e}"] = s[f"nr_iters_g_minus_b_e{e}"]["mean"]
        row[f"iters_g_minus_b_e{e}_lo"], row[f"iters_g_minus_b_e{e}_hi"] = s[f"nr_iters_g_minus_b_e{e}"]["ci95"]
        row[f"iters_g_e{e}_lo"], row[f"iters_g_e{e}_hi"] = s[f"nr_iters_g_e{e}"]["ci95"]
    for k in "gb":
        row[f"err_l2_median_{k}"] = s[f"err_l2_{k}"]["median"]["median"]
    row["err_l2_ratio_g_over_b"] = s["err_l2_g_over_b"]["value"]
    row["res_phys_max_median_g"] = s["res_phys_max_g"]["median"]
    for m in DECISIONS:
        row[f"{m}_el_miss"] = s[f"{m}_el_miss_count"]
        row[f"{m}_el_fa"] = s[f"{m}_el_fa_count"]
        row[f"{m}_cont_miss"] = s[f"{m}_cont_miss_count"]
        row[f"{m}_el_miss_rate"] = s[f"{m}_el_miss_rate"]["value"]
    return row


def aggregate(out_dir, prefix=""):
    """results/genco_delta/eval/*.json -> {prefix}summary_runs.csv (one row per run x set) and
    {prefix}summary_by_arm.csv/.md (mean and SD over init seeds per arm x set)."""
    out_dir = Path(out_dir)
    rows = []
    for f in sorted(out_dir.glob("*.json")):
        run, _, set_ = f.stem.partition("__")
        if (prefix and not run.startswith(prefix)) or (not prefix and run.startswith("smoke_")):
            continue
        m = ARM_RE.match(run)
        if not m:
            continue
        rows.append(dict(run=run, arm=m.group(1), seed=int(m.group(2)), set=set_, **_flatten(json.loads(f.read_text()))))
    if not rows:
        print("nothing to aggregate in", out_dir)
        return
    df = pd.DataFrame(rows)
    df.to_csv(out_dir.parent / f"{prefix}summary_runs.csv", index=False)
    num = [c for c in df.columns if c not in ("run", "arm", "seed", "set")]
    g = df.groupby(["set", "arm"])
    mean, sd, cnt = g[num].mean(), g[num].std(ddof=1), g.size().rename("n_seeds")
    by = pd.concat([cnt, mean.add_suffix("_mean"), sd.add_suffix("_sd")], axis=1).reset_index()
    by.to_csv(out_dir.parent / f"{prefix}summary_by_arm.csv", index=False)
    key = ["iters_g_e2", "iters_g_e4", "iters_g_e6", "iters_b_e6", "iters_g_minus_b_e6", "err_l2_median_g", "g1_el_miss", "b1_el_miss", "g2_el_miss"]
    lines = ["| set | arm | seeds | " + " | ".join(key) + " |", "|" + "---|" * (len(key) + 3)]
    order = {"released": 0, "control": 1, "delta_vm_va": 2, "delta_vm_va_flows": 3}
    for _, r in by.sort_values(["set", "arm"], key=lambda c: c.map(order) if c.name == "arm" else c).iterrows():
        cells = [f"{r[k + '_mean']:.3g} ± {0 if pd.isna(r[k + '_sd']) else r[k + '_sd']:.2g}" for k in key]
        lines.append(f"| {r['set']} | {r['arm']} | {int(r['n_seeds'])} | " + " | ".join(cells) + " |")
    (out_dir.parent / f"{prefix}summary_by_arm.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {out_dir.parent}/{prefix}summary_runs.csv, {prefix}summary_by_arm.csv, {prefix}summary_by_arm.md")


# ----------------------------------------------------------------------------- main
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 22):
            h.update(b)
    return h.hexdigest()


def print_summary(s):
    f = lambda d, k="mean": f"{d[k]:.3g} [{d['ci95'][0]:.3g}, {d['ci95'][1]:.3g}]"
    print(f"  {s['n_scenarios']} scenarios / {s['n_load_scenarios']} load scenarios | "
          f"truth: {s['truth_elements']} violating elements in {s['truth_contingencies']} contingencies")
    for e in TOLS:
        print(f"  NR iters to 1e-{e} MW: model {f(s[f'nr_iters_g_e{e}'])} | base {f(s[f'nr_iters_b_e{e}'])} | "
              f"flat {f(s[f'nr_iters_f_e{e}'])} | model-base {f(s[f'nr_iters_g_minus_b_e{e}'])}")
    print(f"  state error l2 (median): model {s['err_l2_g']['median']['median']:.3g} | base {s['err_l2_b']['median']['median']:.3g} "
          f"| mean ratio model/base {f(s['err_l2_g_over_b'], 'value')}")
    for m in DECISIONS:
        print(f"  {m}: element misses {s[f'{m}_el_miss_count']} (rate {f(s[f'{m}_el_miss_rate'], 'value')}), false alarms "
              f"{s[f'{m}_el_fa_count']}, contingency misses {s[f'{m}_cont_miss_count']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", default="case118_ieee")
    ap.add_argument("--size", default="base")
    ap.add_argument("--seed", type=int, default=0, help="released checkpoint seed (config / normalizer)")
    ap.add_argument("--task_name", default="PowerFlow")
    ap.add_argument("--weights", default=None, help="fine-tuned state dict (default: the released weights)")
    ap.add_argument("--data_root", default=None, help="folder holding <grid>/raw (default data/pf)")
    ap.add_argument("--split", default="splits_paper_test")
    ap.add_argument("--basecase_store", default=None, help="default <data_root>/<grid>/basecase_store.pt (always needed: base-case baseline)")
    ap.add_argument("--delta_inputs", default="vm_va_flows", choices=["vm_va", "vm_va_flows"])
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--tau", type=float, default=1.01)
    ap.add_argument("--eps_v", type=float, default=1e-3)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--max_batches", type=int, default=None, help="debug: stop after this many batches")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out_dir", default=None, help="default results/genco_delta/eval")
    ap.add_argument("--aggregate", action="store_true", help="only tabulate the JSONs in --out_dir (mean/SD over init seeds)")
    ap.add_argument("--prefix", default="", help="--aggregate: run-name prefix (e.g. smoke_)")
    args = ap.parse_args()
    if args.aggregate:
        aggregate(Path(args.out_dir) if args.out_dir else ROOT / "results" / "genco_delta" / "eval", args.prefix)
        return

    t_start = time.time()
    data_root = Path(args.data_root) if args.data_root else Path("data/pf")
    data_root = data_root if data_root.is_absolute() else ROOT / data_root
    grid_dir = data_root / args.grid
    store_path = Path(args.basecase_store) if args.basecase_store else grid_dir / "basecase_store.pt"
    store_path = store_path if store_path.is_absolute() else ROOT / store_path
    if not store_path.is_file():
        raise SystemExit(f"{store_path} not found: run scripts/build_basecase_store.py --data_root {args.data_root or 'data/pf'} --grid {args.grid}")
    store = BaseCaseStore.load(store_path)
    over = {"task": {"task_name": args.task_name}}
    if args.task_name == DELTA:
        over["data"] = {"basecase_store": str(store_path), "delta_inputs": args.delta_inputs}
    Lg = load_genco("pf", args.grid, args.size, args.seed, batch_size=args.batch_size, overrides=over,
                    weights=args.weights, data_root=data_root, split_dir=grid_dir / args.split)
    model = Lg.task.model.eval()
    tag = args.tag or f"{args.task_name}_{Path(args.weights).parent.name if args.weights else 'released'}_{data_root.name}_{args.split}"

    rows, infer_s, n_done = [], 0.0, 0
    for bi, batch in enumerate(Lg.test_loader()):
        if args.max_batches and bi >= args.max_batches:
            break
        t0 = time.time()
        with torch.no_grad():
            out = model(batch)
        infer_s += time.time() - t0
        pred = out["bus"][:, :2].double().numpy()  # Vm [p.u.], Va [rad]
        sids = batch["scenario_id"].numpy()
        Lg.normalizer.inverse_transform(batch)
        RemovePFMask()(batch)
        ptr = batch["bus"].ptr.numpy()
        for g, p in enumerate(acpf.extract_problems(batch)):
            sid = int(sids[g])
            Vm_b, Va_b = store.lookup(sid)
            V_base = Vm_b.double().numpy() * np.exp(1j * Va_b.double().numpy())
            ls = store.load_scenario(sid)
            row = analyse(p, pred[ptr[g]:ptr[g + 1]], V_base, bool(store.converged[store.row(sid)]), args.tau, args.eps_v)
            rows.append(dict(scenario=n_done, raw_id=sid, load_idx=ls, **row))
            n_done += 1
    df = pd.DataFrame(rows)
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "results" / "genco_delta" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / f"{tag}.parquet")
    summ = summarize(df, args.tau, args.eps_v, args.n_boot)
    summ["meta"] = {
        "tag": tag, "task_name": args.task_name, "delta_inputs": args.delta_inputs if args.task_name == DELTA else None,
        "grid": args.grid, "size": args.size, "released_seed": args.seed,
        "weights": args.weights or "released", "weights_sha256": sha256_file(args.weights) if args.weights else None,
        "data_root": str(data_root), "split": args.split, "basecase_store": str(store_path),
        "basecase_store_sha256": sha256_file(store_path), "store_nonconverged": int((~torch.as_tensor(store.converged)).sum()),
        "inference_ms_per_scenario_cpu": round(1e3 * infer_s / max(n_done, 1), 2),
        "wall_seconds": round(time.time() - t_start, 1), "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "starts": STARTS, "decisions": "g<k>: model output + k Newton steps; b<k>: base case + k steps",
        "torch": torch.__version__,
    }
    (out_dir / f"{tag}.json").write_text(json.dumps(summ, indent=1))
    print(f"{tag}: {len(df)} scenarios in {time.time() - t_start:.0f}s -> {out_dir / (tag + '.json')}")
    print_summary(summ)


if __name__ == "__main__":
    main()
