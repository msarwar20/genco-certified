"""Experiment 07: does a GENCO start still save work when the solver reuses its factorization?

Industry contingency analysis factorizes once per operating point (load scenario) and re-uses that
factorization for every contingency. This script prices base-case and GENCO starts under three such
solvers (src/genco_certified/fastpf.py) next to full Newton-Raphson (refactorize every iteration):

  nr      full Newton-Raphson, Jacobian rebuilt and refactorized at every iteration (flat / base / GENCO start)
  chord   chord Newton: base-case Jacobian factorized once per load scenario, unchanged for every contingency
  chordC  chord Newton whose base-case Jacobian is compensated (Woodbury, rank <= 4 per branch) for the outaged
          branches; still evaluated at the base-case voltages, still no refactorization
  fdlf    fast-decoupled XB with B', B'' factorized once and rank-k compensation of branch outages

Generator outages that turn a PV bus into PQ change the unknown set: one extra factorization per distinct
pattern and load scenario (counted as `new_factors`). `--timing` measures end-to-end wall-clock in a
dedicated, single-threaded benchmark (run it on an otherwise idle machine); `--summarize` aggregates the
per-scenario tables into cluster-bootstrap tables and the cost / break-even table.

Usage:
    python scripts/exp07_industrial_baselines.py --grid case118_ieee --size base                       # paper set
    python scripts/exp07_industrial_baselines.py --grid case118_ieee --size base \\
        --data_root data/stress/n2 --split splits_all --tag stress_n2_base                             # stress N-2
    python scripts/exp07_industrial_baselines.py --timing                                              # benchmark
    python scripts/exp07_industrial_baselines.py --summarize                                           # tables
"""

import os
import sys

if "--timing" in sys.argv:  # single-threaded BLAS for the benchmark (must be set before numpy is imported)
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_v] = "1"

import argparse
import gc
import json
import time

import numpy as np
import pandas as pd
import scipy.sparse.linalg as spla

from genco_certified import acpf, fastpf
from genco_certified.loading import ROOT

sys.path.insert(0, str(os.path.dirname(os.path.abspath(__file__))))
from analyze_exp02 import cluster_bootstrap, fmt_ci  # noqa: E402

OUT = ROOT / "results" / "exp07"
TOLS = (1e-2, 1e-4)
EXH = {"case118_ieee": 240}  # exhaustive N-1 workload: 186 branches + 54 generators
PAPER = "case118_ieee_base_seed0"
SETS = [PAPER, "stress_n1_base", "stress_n2_base", "stress_n3_base", "stress_n5_base"]
LABEL = {PAPER: "paper N-1", "stress_n1_base": "stress N-1", "stress_n2_base": "stress N-2",
         "stress_n3_base": "stress N-3", "stress_n5_base": "stress N-5"}
EXH_SETS = ["exh_n1_base", "exh_n1gen_base"]
LABEL.update({"exh_n1_base": "exhaustive N-1, branches", "exh_n1gen_base": "exhaustive N-1, generators",
              "exh_n1_all": "exhaustive N-1, branches + generators"})
COMBOS = [("nr", "flat"), ("nr", "base"), ("nr", "genco"), ("chord", "base"), ("chord", "genco"),
          ("chordC", "base"), ("chordC", "genco"), ("fdlf", "base"), ("fdlf", "genco")]
NAME = {"nr": "Full NR (refactorize each iteration)", "chord": "Chord Newton, base Jacobian",
        "chordC": "Chord Newton, outage-compensated Jacobian", "fdlf": "Fast-decoupled XB, compensated"}


# ---------------------------------------------------------------------------
# data access shared by the run and the timing benchmark
# ---------------------------------------------------------------------------

def load_raw(root, split):
    import torch

    test_ids = torch.load(root / split / "test.pt").tolist()  # loader order = sorted test ids
    raw = {t: pd.read_parquet(root / "raw" / f"{t}_data.parquet") for t in ["bus", "branch", "gen"]}
    load_idx_test = raw["bus"].loc[raw["bus"].scenario.isin(test_ids), "load_scenario_idx"].unique()
    raw = {t: d[d.scenario.isin(test_ids) | d.load_scenario_idx.isin(load_idx_test)] for t, d in raw.items()}
    rows = {t: {s: g for s, g in d[d.scenario.isin(test_ids)].groupby("scenario")} for t, d in raw.items()}
    load_of = raw["bus"].drop_duplicates("scenario").set_index("scenario").load_scenario_idx
    return test_ids, raw, rows, load_of


class LoadScenario:
    """Base case and the factor-reuse contexts of one load scenario (built once, shared by its contingencies)."""

    def __init__(self, raw, load_idx):
        sel = {t: d[d.load_scenario_idx == load_idx] for t, d in raw.items()}
        self.base = fastpf.solve_base_case(sel["bus"], sel["branch"], sel["gen"])
        self.ok = self.base.converged
        # one context (own pattern cache) per method, so that no method inherits another's cached factorizations
        self.chord = fastpf.ChordContext(self.base) if self.ok else None
        self.chordC = fastpf.ChordContext(self.base) if self.ok else None
        self.fd = fastpf.FDLFContext(self.base) if self.ok else None


# ---------------------------------------------------------------------------
# main run
# ---------------------------------------------------------------------------

def rec_fast(row, key, res):
    row[f"{key}_it2"], row[f"{key}_it4"] = res.iters(1e-2), res.iters(1e-4)
    row[f"{key}_sv2"] = res.solves_to(1e-2, include_setup=False)
    row[f"{key}_sv4"] = res.solves_to(1e-4, include_setup=False)
    row[f"{key}_su"], row[f"{key}_nf"] = res.setup_solves, res.new_factors
    h = res.hit.get(1e-4)
    row[f"{key}_ev4"] = np.nan if h is None else h["evals"]
    row[f"{key}_why"] = res.reason
    row[f"{key}_final"] = res.hist[-1]


def run(args):
    import torch
    from gridfm_graphkit.datasets.masking import RemovePFMask

    from genco_certified.loading import load_genco
    from genco_certified.tracing import trace_layers

    t0 = time.time()
    Lg = load_genco("pf", args.grid, args.size, args.seed, batch_size=32,
                    overrides={"task": {"task_name": args.task_name}}, weights=args.weights, data_root=args.data_root,
                    split_dir=(ROOT / (args.data_root or "data/pf")) / args.grid / args.split)
    model = Lg.task.model
    root = ROOT / (args.data_root or "data/pf") / args.grid
    test_ids, raw, rows, load_of = load_raw(root, args.split)
    ctx_cache = {}
    scen_rows, v_genco = [], []
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
            if load_idx not in ctx_cache:
                ctx_cache[load_idx] = LoadScenario(raw, load_idx)
            ls = ctx_cache[load_idx]
            Vg = p.V_from(s[:, 0], s[:, 1])
            v_genco.append((np.abs(Vg), np.angle(Vg)))
            row = dict(scenario=sid, raw_id=raw_id, load_idx=load_idx, base_ok=bool(ls.ok),
                       genco_res_max=float(np.abs(acpf.mismatch(p, Vg)).max()))
            br_s = rows["branch"][raw_id]
            row["n_gen_out_raw"] = int((rows["gen"][raw_id].in_service == 0).sum())
            starts = {"flat": p.flat_start(), "genco": Vg}
            if ls.ok:
                starts["base"] = ls.base.start_for(p)
                out = fastpf.outaged_branches(ls.base, br_s)
                oa = fastpf.as_outage(out)
                row["n_br_out"] = len(out)
                row["n_new_pq"] = int(len(set(p.pq.tolist()) - set(ls.base.p.pq.tolist())))
            else:
                row["n_br_out"], row["n_new_pq"] = -1, -1
            row["is_n0"] = bool(row["n_br_out"] == 0 and row["n_gen_out_raw"] == 0)  # exhaustive sets contain the intact N-0 variant
            for st, V0 in starts.items():
                tr_ = acpf.newton_trajectory(p, V0, n_steps=0, tol_mw=TOLS[-1])
                for e, tol in ((2, 1e-2), (4, 1e-4)):
                    n_it = acpf.iterations_to_tol(tr_["history"], tol)
                    row[f"nr_{st}_it{e}"] = float(n_it) if n_it >= 0 else np.nan
            if ls.ok:
                for st in ("base", "genco"):
                    V0 = starts[st]
                    rec_fast(row, f"chord_{st}", ls.chord.solve(p, V0))
                    rec_fast(row, f"chordC_{st}", ls.chordC.solve(p, V0, outaged=oa))
                    rec_fast(row, f"fdlf_{st}", fastpf.fdlf_solve(ls.fd, p, V0, oa))
                # the pattern factorization belongs to the contingency, not to the start: the GENCO-start run found it
                # already cached by the base-start run, so charge it identically
                for m in ("chord", "chordC", "fdlf"):
                    row[f"{m}_genco_nf"] = row[f"{m}_base_nf"]
            scen_rows.append(row)
            sid += 1
    OUT.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{args.grid}_{args.size}_seed{args.seed}"
    sc = pd.DataFrame(scen_rows)
    sc.to_parquet(OUT / f"{tag}_scenarios.parquet")
    np.savez(OUT / f"{tag}_starts.npz", raw_id=np.array(test_ids[:sid]), Vm=np.array([v[0] for v in v_genco]),
             Va=np.array([v[1] for v in v_genco]))
    nlf = {k: (c.chord.n_patterns_created if c.ok else 0) for k, c in ctx_cache.items()}  # per method; chord shown
    print(f"{tag}: {len(sc)} scenarios ({sc.load_idx.nunique()} load scenarios) in {time.time() - t0:.0f}s; "
          f"base case failed for {int((~sc.base_ok).sum())}; extra Jacobian patterns created: {sum(nlf.values())}")
    for meth, st in COMBOS:
        if st == "base" and not sc.base_ok.any():
            continue
        c2 = sc[f"{meth}_{st}_it2"].mean()
        c4 = sc[f"{meth}_{st}_it4"].mean()
        nfail = int(sc[f"{meth}_{st}_it4"].isna().sum())
        print(f"  {meth:7s} {st:6s} iterations to 1e-2 / 1e-4 MW: {c2:6.2f} / {c4:6.2f}   not converged to 1e-4: {nfail}")


# ---------------------------------------------------------------------------
# dedicated timing benchmark
# ---------------------------------------------------------------------------

def nr_timed(p, V, tol, max_it=30):
    """Lean sparse Newton-Raphson, one mismatch + one Jacobian assembly + one factorize-and-solve per iteration."""
    F = acpf.mismatch(p, V)
    it = 0
    while np.abs(F).max() > tol and it < max_it:
        dx = spla.spsolve(acpf.jacobian(p, V), -F)
        V = acpf.apply_dx(p, V, dx)
        F = acpf.mismatch(p, V)
        it += 1
    return it


def build_timing_set(args):
    root = ROOT / (args.data_root or "data/pf") / args.grid
    tag = args.tag or f"{args.grid}_{args.size}_seed{args.seed}"
    z = np.load(OUT / f"{tag}_starts.npz")
    test_ids, raw, rows, load_of = load_raw(root, args.split)
    assert list(z["raw_id"]) == test_ids[:len(z["raw_id"])]
    sc = pd.read_parquet(OUT / f"{tag}_scenarios.parquet")
    skip = set(sc.index[sc.is_n0]) if "is_n0" in sc else set()
    loads = {}
    for i, rid in enumerate(z["raw_id"]):
        if i not in skip:
            loads.setdefault(int(load_of.loc[rid]), []).append(i)
    return tag, z, raw, rows, load_of, loads


def _t(fn):
    t0 = time.perf_counter()
    r = fn()
    return time.perf_counter() - t0, r


def primitives(base, p, V, reps=300):
    """Median / min wall-clock of the building blocks on one problem (ms): what a factorization costs vs a solve."""
    import scipy.sparse as sp

    J = acpf.jacobian(p, V)
    lu = spla.splu(J.tocsc())
    F = acpf.mismatch(p, V)
    Bp = fastpf.build_Bp(len(p.Vm_set), base.in_service, p.pvpq, base.base_mva)
    Bpp = fastpf.build_Bpp(p.Ybus, p.pq)
    lup, lupp = spla.splu(Bp), spla.splu(Bpp)
    ops = {
        "mismatch eval": lambda: acpf.mismatch(p, V),
        "Jacobian assembly (acpf.jacobian)": lambda: acpf.jacobian(p, V),
        "LU factorization of J (splu)": lambda: spla.splu(J.tocsc()),
        "triangular-solve pair with J factors": lambda: lu.solve(-F),
        "spsolve(J, -F) (factorize+solve, as in NR)": lambda: spla.spsolve(J, -F),
        "LU factorization of B' (splu)": lambda: spla.splu(Bp),
        "LU factorization of B'' (splu)": lambda: spla.splu(Bpp),
        "triangular-solve pair with B'": lambda: lup.solve(F[:len(p.pvpq)]),
        "triangular-solve pair with B''": lambda: lupp.solve(F[len(p.pvpq):]),
        "apply_dx (voltage update)": lambda: acpf.apply_dx(p, V, -F),
    }
    out = {}
    for k, fn in ops.items():
        ts = np.array([_t(fn)[0] for _ in range(reps)]) * 1e3
        out[k] = dict(median_ms=float(np.median(ts)), min_ms=float(ts.min()))
    return out


def timing(args):
    tag, z, raw, rows, load_of, loads = build_timing_set(args)
    R = args.repeats
    sel_loads = sorted(loads)[: args.max_loads] if args.max_loads else sorted(loads)
    load_before = os.getloadavg()
    t_start = time.time()
    # --- per load scenario: problems, starts, outages (outside the timed regions)
    data = {}
    for li in sel_loads:
        base = fastpf.solve_base_case(*(d[d.load_scenario_idx == li] for d in (raw["bus"], raw["branch"], raw["gen"])))
        items = []
        for i in loads[li]:
            rid = int(z["raw_id"][i])
            p = acpf.problem_from_raw(rows["bus"][rid], rows["branch"][rid], rows["gen"][rid])
            out = fastpf.as_outage(fastpf.outaged_branches(base, rows["branch"][rid]))
            starts = {"flat": p.flat_start(), "base": base.start_for(p), "genco": p.V_from(z["Vm"][i], z["Va"][i])}
            items.append((i, p, out, starts))
        data[li] = (base, items)
    n_total = sum(len(v[1]) for v in data.values())
    methods = ["nr", "chord", "chordC", "chordC_bd", "fdlf", "fdlf_bd"]
    keys = [(m, st, tol) for m in methods for st in (["flat", "base", "genco"] if m == "nr" else ["base", "genco"]) for tol in TOLS]
    best = {k: np.full(n_total, np.inf) for k in keys}
    iters = {k: np.full(n_total, np.nan) for k in keys}
    pat_ms = {m: np.full(n_total, np.inf) for m in methods if m != "nr"}   # cold pattern creation (new factorization), min over repeats
    ctor = {"chord": lambda b: fastpf.ChordContext(b), "chordC": lambda b: fastpf.ChordContext(b),
            "chordC_bd": lambda b: fastpf.ChordContext(b, bordered=True),
            "fdlf": lambda b: fastpf.FDLFContext(b), "fdlf_bd": lambda b: fastpf.FDLFContext(b, bordered=True)}
    base_t = {m: {li: np.inf for li in data} for m in ctor}
    order = [i for li in data for i, *_ in data[li][1]]
    pos = {i: n for n, i in enumerate(order)}
    # primitives on the first load scenario's first contingency
    li0 = next(iter(data))
    b0, items0 = data[li0]
    prim = primitives(b0, items0[0][1], items0[0][3]["base"])
    gc.collect()
    gc.disable()
    for rep in range(R):
        for li, (base, items) in data.items():
            ctxs = {}
            for name, mk in ctor.items():     # one context per method: separate pattern caches, separate base factorizations
                dt, ctxs[name] = _t(lambda: mk(base))
                base_t[name][li] = min(base_t[name][li], dt)
            for i, p, out, starts in items:
                n = pos[i]
                # cold pattern creation (a PV->PQ change needs a new factorization / bordering), charged to every start and tolerance
                tp = {}
                for m in ("chord", "chordC", "chordC_bd"):
                    tp[m] = _t(lambda: ctxs[m]._pattern(p))[0]
                for m in ("fdlf", "fdlf_bd"):
                    tp[m] = _t(lambda: ctxs[m]._get_pq(p.pq))[0]
                for m, v in tp.items():
                    pat_ms[m][n] = min(pat_ms[m][n], v)
                for tol in TOLS:
                    for st in ("flat", "base", "genco"):
                        dt, it = _t(lambda: nr_timed(p, starts[st], tol))
                        k = ("nr", st, tol)
                        best[k][n] = min(best[k][n], dt)
                        iters[k][n] = it
                    for st in ("base", "genco"):
                        V0 = starts[st]
                        calls = {
                            "chord": lambda: ctxs["chord"].solve(p, V0, tols=(tol,)),
                            "chordC": lambda: ctxs["chordC"].solve(p, V0, tols=(tol,), outaged=out),
                            "chordC_bd": lambda: ctxs["chordC_bd"].solve(p, V0, tols=(tol,), outaged=out),
                            "fdlf": lambda: fastpf.fdlf_solve(ctxs["fdlf"], p, V0, out, tols=(tol,)),
                            "fdlf_bd": lambda: fastpf.fdlf_solve(ctxs["fdlf_bd"], p, V0, out, tols=(tol,)),
                        }
                        for m, fn in calls.items():
                            dt, res = _t(fn)           # pattern already cached: warm solve (+ outage compensation setup)
                            k = (m, st, tol)
                            best[k][n] = min(best[k][n], dt + tp[m])
                            iters[k][n] = res.iters(tol)
        gc.collect()
    gc.enable()
    load_after = os.getloadavg()
    grp = np.array([li for li in data for _ in data[li][1]])
    nc = {li: len(data[li][1]) for li in data}
    out = dict(tag=tag, grid=args.grid, repeats=R, n_contingencies=int(n_total), n_load_scenarios=len(data),
               contingencies_per_load=float(np.mean(list(nc.values()))), exhaustive_n1_contingencies=EXH.get(args.grid),
               loadavg_before=list(load_before), loadavg_after=list(load_after), ncpu=os.cpu_count(),
               quiet=bool(max(load_before[0], load_after[0]) < 2.0), single_thread=True, wall_s=time.time() - t_start,
               note="per contingency: min over repeats of (warm solve + cold pattern creation); fresh contexts per repeat; python/scipy implementation; "
                    "chordC/fdlf include the outage-compensation setup in the solve; the base factorization is reported separately")
    cost = {}
    for (m, st, tol), v in best.items():
        ok = np.isfinite(iters[(m, st, tol)])
        pt, lo, hi = cluster_bootstrap(v * 1e3, groups=grp, B=2000)
        cost[f"{m}|{st}|{tol:g}"] = dict(ms_mean=pt, ms_lo=lo, ms_hi=hi, ms_mean_converged=float(np.mean(v[ok]) * 1e3) if ok.any() else np.nan,
                                         n_not_converged=int((~ok).sum()), mean_iters=float(np.nanmean(iters[(m, st, tol)])) if ok.any() else np.nan)
    out["solve_ms"] = cost
    out["pattern_creation_ms_mean"] = {m: float(np.mean(v) * 1e3) for m, v in pat_ms.items()}
    bf = {}
    for name, d in base_t.items():
        v = np.array([d[li] * 1e3 for li in data])
        bf[name] = dict(ms_per_load_scenario=float(v.mean()), ms_sd=float(v.std(ddof=1)) if len(v) > 1 else np.nan)
    out["base_factor_ms"] = bf
    out["primitives_ms"] = prim
    (OUT / f"timing_{tag}.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"timing {tag}: {n_total} contingencies / {len(data)} load scenarios, {R} repeats, loadavg {load_before[0]:.2f} -> {load_after[0]:.2f} "
          f"({'quiet' if out['quiet'] else 'LOADED: provisional'}), {out['wall_s']:.0f}s")
    for k, v in cost.items():
        if k.endswith("0.0001"):
            print(f"  {k:24s} {v['ms_mean']:7.3f} ms  [{v['ms_lo']:.3f}, {v['ms_hi']:.3f}]  iters {v['mean_iters']:.2f}")
    print("  base factorization ms per load scenario:", {k: round(v["ms_per_load_scenario"], 3) for k, v in bf.items()})
    print("  primitives (median ms):", {k: round(v["median_ms"], 4) for k, v in prim.items()})


# ---------------------------------------------------------------------------
# summaries: iterations / solves with cluster-bootstrap CIs, paired GENCO-vs-base, cost and break-even
# ---------------------------------------------------------------------------

def load_sets(tags):
    sets = {t: pd.read_parquet(OUT / f"{t}_scenarios.parquet") for t in tags if (OUT / f"{t}_scenarios.parquet").exists()}
    if all(t in sets for t in EXH_SETS):  # same 30 load scenarios in both: together the full per-operating-point N-1 workload
        a, b = sets[EXH_SETS[0]], sets[EXH_SETS[1]]
        sets["exh_n1_all"] = pd.concat([a, b.assign(scenario=b.scenario + a.scenario.max() + 1)], ignore_index=True)
    return sets


def solves_of(df, meth, st, e):
    """Total triangular-solve pairs to tolerance 1e-<e> (iteration solves + setup), factorizations per contingency."""
    if meth == "nr":
        it = df[f"nr_{st}_it{e}"]
        return it, it  # one factorization and one solve per iteration
    return df[f"{meth}_{st}_sv{e}"] + df[f"{meth}_{st}_su"], df[f"{meth}_{st}_nf"]


def summarize(args):
    tags = args.tags or SETS + EXH_SETS
    sets = load_sets(tags)
    log = []

    def P(s=""):
        print(s)
        log.append(s)

    P("exp07 summary: industry-style solvers (factorization reuse) from base-case vs GENCO starts; cluster bootstrap (2000) over load scenarios")
    P("iters = Newton/chord iterations (FDLF: full P+Q iterations); solves = triangular-solve pairs incl. compensation setup; fact = LU factorizations per contingency")
    P("(NR: one Jacobian factorization and one solve per iteration; reuse methods: the per-load-scenario base factorization is NOT included, it is amortised in the cost table)")
    P("'mean' is over scenarios that reached the tolerance within 50 iterations; failures are counted separately. NaN-iteration scenarios of both starts are dropped in the paired rows.\n")
    tidy, paired = [], []
    for tag, df in sets.items():
        use = df[df.base_ok] if not df.base_ok.all() else df
        if "is_n0" in use:
            if use.is_n0.any():
                P(f"[{tag}: dropping {int(use.is_n0.sum())} intact N-0 variants]")
            use = use[~use.is_n0]
        groups = {"all": np.ones(len(use), bool), "branch-only": (use.n_new_pq == 0).to_numpy(), "gen-outage (PV->PQ)": (use.n_new_pq > 0).to_numpy()}
        P("=" * 110)
        P(f"{LABEL.get(tag, tag)}: {len(use)} contingencies, {use.load_idx.nunique()} load scenarios; with a PV->PQ change: {int((use.n_new_pq > 0).sum())}; "
          f"mean outaged branches {use.n_br_out.mean():.2f}")
        for gname, gmask in groups.items():
            d = use[gmask]
            if len(d) == 0:
                continue
            rows = []
            for meth, st in COMBOS:
                if st == "flat" and gname != "all":
                    pass
                for e in (2, 4):
                    it = d[f"{meth}_{st}_it{e}"]
                    ok = it.notna()
                    sv, fac = solves_of(d, meth, st, e)
                    m_it = cluster_bootstrap(it[ok], groups=d.load_idx[ok])
                    m_sv = cluster_bootstrap(sv[ok], groups=d.load_idx[ok])
                    m_fac = cluster_bootstrap(fac, groups=d.load_idx)  # a property of the contingency: all scenarios
                    rows.append(dict(set=LABEL.get(tag, tag), tag=tag, group=gname, method=meth, start=st, tol=TOLS[e // 2 - 1],
                                     n=len(d), n_fail=int((~ok).sum()), fail_rate=float((~ok).mean()),
                                     iters=m_it[0], iters_lo=m_it[1], iters_hi=m_it[2], solves=m_sv[0], solves_lo=m_sv[1], solves_hi=m_sv[2],
                                     fact=m_fac[0], fact_lo=m_fac[1], fact_hi=m_fac[2],
                                     iters_max=float(it.max()) if ok.any() else np.nan))
            tidy += rows
            if gname == "all":
                P(f"\n  [{gname}]  iterations / solves / factorizations to tolerance, mean [95% CI]; fail = not reached in 50 iterations")
                tb = pd.DataFrame(rows)
                for e_tol in TOLS:
                    P(f"  tolerance {e_tol:g} MW")
                    for _, r in tb[tb.tol == e_tol].iterrows():
                        P(f"    {r.method:7s} {r.start:6s} iters {fmt_ci(r.iters, r.iters_lo, r.iters_hi, 2):>22s}  solves {fmt_ci(r.solves, r.solves_lo, r.solves_hi, 2):>22s}  "
                          f"fact {r.fact:5.2f}  fail {int(r.n_fail):3d}/{int(r.n)}")
            # paired GENCO vs base start
            for meth in ("nr", "chord", "chordC", "fdlf"):
                for e in (2, 4):
                    a, b = d[f"{meth}_genco_it{e}"], d[f"{meth}_base_it{e}"]
                    ok = a.notna() & b.notna()
                    sva, fa = solves_of(d, meth, "genco", e)
                    svb, fb = solves_of(d, meth, "base", e)
                    diff = cluster_bootstrap((a - b)[ok], groups=d.load_idx[ok])
                    ratio = cluster_bootstrap(a[ok], den=b[ok], groups=d.load_idx[ok])
                    sdiff = cluster_bootstrap((sva - svb)[ok], groups=d.load_idx[ok])
                    sratio = cluster_bootstrap(sva[ok], den=svb[ok], groups=d.load_idx[ok])
                    fdiff = cluster_bootstrap((fa - fb)[ok], groups=d.load_idx[ok])
                    paired.append(dict(set=LABEL.get(tag, tag), tag=tag, group=gname, method=meth, tol=TOLS[e // 2 - 1], n=int(ok.sum()),
                                       n_excl=int((~ok).sum()), d_iters=diff[0], d_iters_lo=diff[1], d_iters_hi=diff[2],
                                       saving_iters=1 - ratio[0], saving_iters_lo=1 - ratio[2], saving_iters_hi=1 - ratio[1],
                                       d_solves=sdiff[0], d_solves_lo=sdiff[1], d_solves_hi=sdiff[2],
                                       saving_solves=1 - sratio[0], saving_solves_lo=1 - sratio[2], saving_solves_hi=1 - sratio[1],
                                       d_fact=fdiff[0], d_fact_lo=fdiff[1], d_fact_hi=fdiff[2],
                                       frac_genco_fewer=float((a < b)[ok].mean()), frac_genco_more=float((a > b)[ok].mean())))
        pdf = pd.DataFrame([r for r in paired if r["tag"] == tag and r["group"] == "all"])
        P(f"\n  paired GENCO-start minus base-start (all contingencies), mean difference [95% CI] and relative saving in iterations / solves:")
        for _, r in pdf.iterrows():
            P(f"    {r.method:7s} tol {r.tol:g}  d_iters {fmt_ci(r.d_iters, r.d_iters_lo, r.d_iters_hi, 2):>24s}  saving_iters {fmt_ci(r.saving_iters, r.saving_iters_lo, r.saving_iters_hi, 2):>22s}  "
              f"saving_solves {fmt_ci(r.saving_solves, r.saving_solves_lo, r.saving_solves_hi, 2):>22s}  GENCO fewer in {r.frac_genco_fewer:.0%}, more in {r.frac_genco_more:.0%}  (n={r.n}, excl {r.n_excl})")
    pd.DataFrame(tidy).to_csv(OUT / "summary_iterations.csv", index=False)
    pd.DataFrame(paired).to_csv(OUT / "summary_paired_genco_vs_base.csv", index=False)

    # failures table
    P("\n" + "=" * 110)
    P("FAILURES (not converged to the tolerance within the iteration cap: 30 for NR, 50 for the reuse solvers)")
    fr = pd.DataFrame(tidy)
    ff = fr[(fr.group == "all") & (fr.tol == 1e-4)][["set", "method", "start", "n", "n_fail", "iters_max"]]
    P(ff.to_string(index=False))
    ff.to_csv(OUT / "summary_failures.csv", index=False)
    why = []
    for tag, df in sets.items():
        for meth, st in COMBOS[3:]:
            if f"{meth}_{st}_why" in df:
                vc = df[f"{meth}_{st}_why"].value_counts()
                for k, v in vc.items():
                    if k:
                        why.append(dict(set=LABEL.get(tag, tag), method=meth, start=st, reason=k, count=int(v)))
    P("\nfailure reasons: " + (str(pd.DataFrame(why).to_string(index=False)) if why else "none"))
    pd.DataFrame(why).to_csv(OUT / "summary_failure_reasons.csv", index=False)

    cost_tables(args, sets, P)
    (OUT / "exp07.log").write_text("\n".join(log) + "\n")


def model_ms(df, prim, method, start, e, ideal_assembly=True, cap=50):
    """Count-based cost model, ms per contingency: mean over scenarios of  counts x median building-block times.

    Counts come from the per-scenario table (mismatch evaluations, voltage updates, triangular solves incl. compensation
    setup, new factorizations; runs that never reached the tolerance are charged the iteration cap). Building blocks are
    the medians measured by --timing. ideal_assembly=True charges a Jacobian assembly at 2 mismatch evaluations, which is
    what compiled code does, instead of the ~100x slower python/scipy sparse-matrix assembly; it removes the overhead that
    makes this implementation's full Newton look slower than it is, i.e. it favours the classical baseline.
    """
    g = lambda k: prim[k]["median_ms"]
    t_mis, t_app, t_asm = g("mismatch eval"), g("apply_dx (voltage update)"), g("Jacobian assembly (acpf.jacobian)")
    if ideal_assembly:
        t_asm = 2 * t_mis
    t_facJ, t_slvJ = g("LU factorization of J (splu)"), g("triangular-solve pair with J factors")
    t_facBpp = g("LU factorization of B'' (splu)")
    t_slvB = 0.5 * (g("triangular-solve pair with B'") + g("triangular-solve pair with B''"))
    if method == "nr":
        it = df[f"nr_{start}_it{e}"].fillna(30)
        return float((t_mis + it * (t_mis + t_app + t_asm + t_facJ + t_slvJ)).mean())
    it = df[f"{method}_{start}_it{e}"].fillna(cap)
    su, nf = df[f"{method}_{start}_su"], df[f"{method}_{start}_nf"]
    if method == "fdlf":  # one iteration = one P and one Q half-step
        return float(((2 * it + 1) * t_mis + 2 * it * t_app + 2 * it * t_slvB + su * t_slvB + nf * t_facBpp).mean())
    return float(((it + 1) * t_mis + it * (t_app + t_slvJ) + su * t_slvJ + nf * (t_facJ + t_asm)).mean())


def cost_tables(args, sets, P):
    files = [f for f in sorted(OUT.glob("timing_*.json")) if not f.stem.endswith("_run1")]
    if not files:
        P("\n(no timing file: run --timing on an idle machine for the cost table)")
        return
    primary = None
    for f in files:
        T = json.loads(f.read_text())
        if T["tag"] == args.timing_tag:
            primary = T
        cost_one(T, sets, P, primary=(T["tag"] == args.timing_tag))
    if primary is not None:
        model_by_set(primary, sets, P)


def prep_df(df):
    d = df[df.base_ok] if not df.base_ok.all() else df
    if "is_n0" in d:
        d = d[~d.is_n0]
    return d


def model_by_set(T, sets, P):
    """Count-based cost model and break-even on every set, building blocks from the primary timing run."""
    prim = T["primitives_ms"]
    pm = lambda k: prim[k]["median_ms"]
    nex = T.get("exhaustive_n1_contingencies") or 240
    base_fac = {"chord": pm("LU factorization of J (splu)") + 2 * pm("mismatch eval"), "chordC": pm("LU factorization of J (splu)") + 2 * pm("mismatch eval"),
                "fdlf": pm("LU factorization of B' (splu)") + pm("LU factorization of B'' (splu)")}
    P("\n" + "=" * 110)
    P("COUNT-BASED COST MODEL AND BREAK-EVEN ON EVERY SET (counts from the per-scenario tables x building-block times of the primary timing run, Jacobian assembly "
      f"at 2 mismatch evaluations, base factorization amortised over {nex} contingencies per load scenario; failed runs charged the iteration cap)")
    P("break-even = model cost from the base-case start minus from the GENCO start, same solver (ms of GENCO inference at which a GENCO start pays off)")
    rows = []
    for tag, df in sets.items():
        d = prep_df(df)
        for e, tol in ((2, 1e-2), (4, 1e-4)):
            for m in ["nr", "chord", "chordC", "fdlf"]:
                tb = model_ms(d, prim, m, "base", e) + base_fac.get(m, 0.0) / nex
                tg = model_ms(d, prim, m, "genco", e) + base_fac.get(m, 0.0) / nex
                rows.append(dict(set=LABEL.get(tag, tag), tag=tag, n=len(d), tol=tol, solver=m, base_model_ms=tb, genco_model_ms=tg, breakeven_ms=tb - tg))
    df_ = pd.DataFrame(rows)
    df_.to_csv(OUT / "breakeven_model_by_set.csv", index=False)
    for tol in TOLS:
        P(f"  tolerance {tol:g} MW: break-even ms (same solver) [base-start model ms -> GENCO-start model ms]")
        for tag in df_.tag.unique():
            sub = df_[(df_.tag == tag) & (df_.tol == tol)]
            P(f"    {LABEL.get(tag, tag):38s} n={int(sub.n.iloc[0]):5d}  " + "   ".join(
                f"{r.solver} {r.breakeven_ms:.3f} [{r.base_model_ms:.3f}->{r.genco_model_ms:.3f}]" for _, r in sub.iterrows()))
    P(f"  largest model break-even over all sets and solvers: {df_.breakeven_ms.max():.3f} ms")


def cost_one(T, sets, P, primary=True):
    sfx = "" if primary else f"_{T['tag']}"
    rt = json.loads((ROOT / "results" / "runtime" / "case118_ieee_base.json").read_text())
    df = sets.get(T["tag"])
    df = prep_df(df) if df is not None else None
    prim = T["primitives_ms"]
    pm = lambda k: prim[k]["median_ms"]
    P("\n" + "=" * 110)
    prov = "" if T["quiet"] else "  ** PROVISIONAL: the machine was loaded during the benchmark **"
    P(f"END-TO-END COST per contingency, {T['tag']}: {T['n_contingencies']} contingencies, {T['n_load_scenarios']} load scenarios, single thread, "
      f"min of {T['repeats']} repeats; load average {T['loadavg_before'][0]:.2f} -> {T['loadavg_after'][0]:.2f} on {T['ncpu']} cores{prov}")
    P("pure python/scipy implementations. 'measured' = wall-clock of these implementations; 'model' = counts x building-block times with the Jacobian assembly "
      "charged at 2 mismatch evaluations (compiled-code-like), which strips python overhead from full Newton and favours the classical baseline.")
    bf = T["base_factor_ms"]
    ncl = T["contingencies_per_load"]
    nex = T.get("exhaustive_n1_contingencies") or ncl
    P(f"base factorization per load scenario, measured (Jacobian derivatives + LU for chord; B', B'' LUs for fdlf): chord {bf['chord']['ms_per_load_scenario']:.3f} ms, "
      f"fdlf {bf['fdlf']['ms_per_load_scenario']:.3f} ms; amortised over {ncl:.1f} contingencies per load scenario in this set, or over {nex} for an exhaustive N-1 sweep")
    P("building blocks, median ms on one IEEE 118 problem (single thread): " + "; ".join(f"{k} {v['median_ms']:.4f}" for k, v in prim.items()))
    P(f"  => one LU of the Jacobian costs {pm('LU factorization of J (splu)') / pm('triangular-solve pair with J factors'):.1f} triangular-solve pairs; "
      f"assembling the Jacobian in python costs {pm('Jacobian assembly (acpf.jacobian)') / pm('triangular-solve pair with J factors'):.0f}; "
      f"a mismatch evaluation {pm('mismatch eval') / pm('triangular-solve pair with J factors'):.1f}")
    if "pattern_creation_ms_mean" in T:
        P("mean cold pattern-creation time per contingency (new B''/Jacobian factorization for PV->PQ changes; already included in the measured solve times below): "
          + ", ".join(f"{k} {v:.3f}" for k, v in T["pattern_creation_ms_mean"].items()) + " ms")
    infer = {"GENCO CPU bs=1 (4 threads)": rt["genco_cpu_bs1_ms_per_scen"], "GENCO CPU bs=16": rt["genco_cpu_bs16_ms_per_scen"],
             "GENCO MPS bs=16": rt["genco_mps_bs16_ms_per_scen"], "GENCO MPS bs=128, 9 layers": rt["genco_mps_bs128_9layers_ms_per_scen"]}
    P("GENCO inference ms per scenario (results/runtime/case118_ieee_base.json, measured earlier, 4 torch threads): " + ", ".join(f"{k} {v:.1f}" for k, v in infer.items()))
    base_fac_model = {"chord": pm("LU factorization of J (splu)") + 2 * pm("mismatch eval"), "chordC": pm("LU factorization of J (splu)") + 2 * pm("mismatch eval"),
                      "fdlf": pm("LU factorization of B' (splu)") + pm("LU factorization of B'' (splu)")}
    rows = []
    for tol in TOLS:
        e = 2 if tol == 1e-2 else 4
        for m in ["nr", "chord", "chordC", "chordC_bd", "fdlf", "fdlf_bd"]:
            for st in (["flat", "base", "genco"] if m == "nr" else ["base", "genco"]):
                c = T["solve_ms"][f"{m}|{st}|{tol:g}"]
                am_s = bf[m]["ms_per_load_scenario"] / ncl if m in bf else 0.0
                am_e = bf[m]["ms_per_load_scenario"] / nex if m in bf else 0.0
                r = dict(tol=tol, method=m, start=st, measured_solve_ms=c["ms_mean"], solve_lo=c["ms_lo"], solve_hi=c["ms_hi"],
                         amort_ms_set=am_s, amort_ms_exhaustive=am_e, measured_total_ms=c["ms_mean"] + am_e, measured_total_ms_set=c["ms_mean"] + am_s,
                         mean_iters=c["mean_iters"], n_not_converged=c["n_not_converged"])
                if df is not None and m in ("nr", "chord", "chordC", "fdlf"):
                    mb = base_fac_model.get(m, 0.0) / nex
                    r["model_solve_ms"] = model_ms(df, prim, m, st, e)
                    r["model_total_ms"] = r["model_solve_ms"] + mb
                rows.append(r)
    ct = pd.DataFrame(rows)
    ct.to_csv(OUT / f"cost_table{sfx}.csv", index=False)
    for tol in TOLS:
        P(f"\n  tolerance {tol:g} MW; ms per contingency: measured solve [95% CI over load scenarios] + base factorization amortised over {nex} (exhaustive N-1) | model | iters | not converged")
        for _, r in ct[ct.tol == tol].iterrows():
            mod = f"model {r.model_total_ms:6.3f}" if "model_total_ms" in r and pd.notna(r.get("model_total_ms")) else "model     n/a"
            P(f"    {r.method:10s} {r.start:6s} {r.measured_solve_ms:7.3f} [{r.solve_lo:.3f}, {r.solve_hi:.3f}] + {r.amort_ms_exhaustive:.3f} = {r.measured_total_ms:7.3f}   {mod}   "
              f"iters {r.mean_iters:5.2f}   not converged {int(r.n_not_converged)}")
    # break-even
    P("\nBREAK-EVEN GENCO INFERENCE TIME per contingency (ms) = what the solver saves by starting from GENCO instead of the base case (negative: no saving)")
    P("  'same solver': both starts use the same solver. 'vs cheapest': GENCO start + solver against the cheapest base-start solver the operator could run instead.")
    be = []
    for basis in ("measured", "model"):
        col = f"{basis}_total_ms"
        if col not in ct or ct[col].isna().all():
            continue
        for tol in TOLS:
            sub = ct[(ct.tol == tol) & ct[col].notna()].set_index(["method", "start"])
            cheapest_base = sub.xs("base", level="start")[col]
            cb_name = cheapest_base.idxmin()
            for m in ["nr", "chord", "chordC", "chordC_bd", "fdlf", "fdlf_bd"]:
                if (m, "base") not in sub.index:
                    continue
                tb, tg = sub.loc[(m, "base"), col], sub.loc[(m, "genco"), col]
                be.append(dict(basis=basis, tol=tol, method=m, base_ms=tb, genco_ms=tg, breakeven_same_solver_ms=tb - tg,
                               breakeven_vs_cheapest_base_ms=cheapest_base.min() - tg, cheapest_base_solver=cb_name,
                               cheapest_base_ms=cheapest_base.min(), amortisation=f"exhaustive N-1 ({nex} contingencies per load scenario)"))
    be = pd.DataFrame(be)
    be.to_csv(OUT / f"breakeven{sfx}.csv", index=False)
    for basis in be.basis.unique():
        for tol in TOLS:
            b = be[(be.basis == basis) & (be.tol == tol)]
            P(f"  [{basis}] tolerance {tol:g}: cheapest base-start solver = {b.cheapest_base_solver.iloc[0]} at {b.cheapest_base_ms.iloc[0]:.3f} ms")
            for _, r in b.iterrows():
                P(f"      {r.method:10s} base {r.base_ms:7.3f} ms, GENCO {r.genco_ms:7.3f} ms -> break-even {r.breakeven_same_solver_ms:7.3f} ms (same solver), "
                  f"{r.breakeven_vs_cheapest_base_ms:7.3f} ms (vs cheapest base-start solver)")
    top = be.breakeven_same_solver_ms.max()
    P(f"\n  largest same-solver break-even over all rows: {top:.3f} ms; measured GENCO inference is " + ", ".join(f"{v / max(top, 1e-9):.0f}x ({k})" for k, v in infer.items()) + " of that")
    out = dict(timing=T, genco_inference_ms=infer, breakeven=be.to_dict(orient="records"), cost_table=ct.to_dict(orient="records"))
    (OUT / f"cost_breakeven{sfx}.json").write_text(json.dumps(out, indent=2, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="case118_ieee")
    ap.add_argument("--size", default="base")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task_name", default="PowerFlow")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--data_root", default=None, help="folder holding <grid>/raw (default data/pf)")
    ap.add_argument("--split", default="splits_paper_test")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--timing", action="store_true", help="dedicated single-thread wall-clock benchmark (needs a finished run for --tag)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max_loads", type=int, default=0, help="timing: use only the first N load scenarios (0 = all)")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--tags", nargs="*", default=None, help="summarize: result tags (default paper set + stress N-1/2/3/5)")
    ap.add_argument("--timing_tag", default=PAPER, help="summarize: which timing_<tag>.json feeds the cost table")
    args = ap.parse_args()
    if args.timing:
        timing(args)
    elif args.summarize:
        summarize(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
