"""Experiment 04 (v2): does GENCO's embedding know where GENCO is wrong, beyond bus identity?

Per bus, predict GENCO's true state error (log10 |dVm| at PQ buses, log10 |dVa| at non-slack
buses; GENCO physical state with true setpoints vs the datakit AC solution) from:
  - local mismatch |dS_i| of GENCO's own output (ground-truth-free),
  - structure (bus type, outage type, hops to slack, hops to the true outage location),
  - bus-ID one-hot (118-d), alone / + mismatch / + structure + mismatch,
  - local raw inputs (Pd, Qd, GS, BS, Vm setpoint, type),
  - the final-layer bus embedding (384-d), alone / + mismatch / + all the above,
  - the same embedding from a RANDOMLY INITIALISED GENCO of the same architecture,
  - PCA of the real / random embedding, k in {5, 20, 118} (fit on train folds only).

Methodology (fixes of the v1 probe, see RESEARCH_LOG "External reviews" point 10):
  * grouped K-fold (K=5) by LOAD SCENARIO (22 topology variants of one operating point never
    straddle train/test); the same folds are used for every feature set, model and seed;
  * ridge on standardised features fit on train folds only; alpha is chosen by an INNER
    grouped CV (equivalent to RidgeCV, but with load-scenario folds; `--alpha_mode loo` gives
    sklearn's leave-one-out RidgeCV that v1 used, as a sensitivity check);
  * R^2 and Spearman on the pooled out-of-fold predictions; 95% CIs by a cluster bootstrap over
    load scenarios (paired across feature sets, so contrasts get proper CIs);
  * outage location from the raw datakit tables (out-of-service branch endpoints / generator bus).

Usage: python scripts/exp04_embedding_probe.py [--models base:0,base:1,base:42,small:0]
Writes results/exp04/probe_v2.csv, probe_v2_contrasts.csv, probe_v2_summary.json.
"""

import argparse
import copy
import json
import time

import networkx as nx
import numpy as np
import pandas as pd
import torch
from scipy.linalg import eigh
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from genco_certified import acpf
from genco_certified.loading import ROOT, load_genco
from genco_certified.tracing import trace_layers
from gridfm_graphkit.datasets.globals import BS, GS, PD_H, QD_H, VM_H

OUT = ROOT / "results" / "exp04"
PCA_KS = (5, 20, 118)
ALPHAS = np.logspace(-2, 6, 17)
K_OUTER = 5
K_INNER = 4


# --------------------------------------------------------------------------------------
# data collection
# --------------------------------------------------------------------------------------
def hops_from(n, edges, sources):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    d = nx.multi_source_dijkstra_path_length(G, set(int(s) for s in sources)) if len(sources) else {}
    return np.array([d.get(i, 99) for i in range(n)], dtype=float)


def raw_outages(grid, data_root, test_ids):
    """Contingency location per scenario from the raw datakit tables.

    Elements out of service in the *base case* of a load scenario (majority status over its
    topology variants) are not contingencies. Returns {scenario: dict(kind, buses)} and
    {scenario: load_scenario_idx}.
    """
    raw = data_root / grid / "raw"
    bus = pd.read_parquet(raw / "bus_data.parquet", columns=["scenario", "load_scenario_idx"]).drop_duplicates("scenario")
    load_of = bus.set_index("scenario").load_scenario_idx
    ls_of_test = {int(s): int(load_of[s]) for s in test_ids}
    ls_ids = sorted(set(ls_of_test.values()))
    cols_b = ["scenario", "load_scenario_idx", "idx", "from_bus", "to_bus", "br_status"]
    cols_g = ["scenario", "load_scenario_idx", "idx", "bus", "in_service"]
    br = pd.read_parquet(raw / "branch_data.parquet", columns=cols_b)
    gn = pd.read_parquet(raw / "gen_data.parquet", columns=cols_g)
    br, gn = br[br.load_scenario_idx.isin(ls_ids)], gn[gn.load_scenario_idx.isin(ls_ids)]
    # base-case status = majority over the load scenario's variants
    brm = br.groupby(["load_scenario_idx", "idx"]).br_status.mean().rename("m").reset_index()
    gnm = gn.groupby(["load_scenario_idx", "idx"]).in_service.mean().rename("m").reset_index()
    br = br.merge(brm, on=["load_scenario_idx", "idx"])
    gn = gn.merge(gnm, on=["load_scenario_idx", "idx"])
    br_out = br[(br.br_status == 0) & (br.m > 0.5)]
    gn_out = gn[(gn.in_service == 0) & (gn.m > 0.5)]
    out = {}
    for s in test_ids:
        s = int(s)
        b, g = br_out[br_out.scenario == s], gn_out[gn_out.scenario == s]
        buses = sorted(set(b.from_bus.astype(int)) | set(b.to_bus.astype(int)) | set(g.bus.astype(int)))
        out[s] = {"kind": "branch" if len(b) else ("gen" if len(g) else "none"), "buses": buses,
                  "n_el": int(len(b) + len(g))}
    return out, ls_of_test


def init_random(model, seed):
    """Fresh random weights, same architecture (every submodule's own reset_parameters)."""
    torch.manual_seed(seed)
    for m in model.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
    return model


def collect(size, seed, grid, rand_seed, cache=None):
    tag = f"{grid}_{size}_seed{seed}"
    if cache is not None and (cache / f"{tag}.npz").exists():
        z = np.load(cache / f"{tag}.npz", allow_pickle=True)
        return pd.read_parquet(cache / f"{tag}.parquet"), z["E_real"], z["E_rand"], z["X_raw"]
    Lg = load_genco("pf", grid, size, seed=seed, batch_size=32)
    data_root = ROOT / "data" / "pf"
    test_ids = sorted(torch.load(data_root / grid / "splits_paper_test" / "test.pt").tolist())
    outages, ls_of = raw_outages(grid, data_root, test_ids)
    real = Lg.task.model
    rand = init_random(copy.deepcopy(real), rand_seed).eval()
    rows, E_real, E_rand, X_raw = [], [], [], []
    sid = 0
    for batch in Lg.test_loader():
        with torch.no_grad(), trace_layers(real, capture_embeddings=True) as tr:
            real(batch)
        with torch.no_grad(), trace_layers(rand, capture_embeddings=True) as trr:
            rand(batch)
        st = tr["state"][-1].double().numpy()
        er, en = tr["embedding"][-1].numpy(), trr["embedding"][-1].numpy()
        assert np.isfinite(en).all(), "random-init embedding has non-finite values"
        Lg.normalizer.inverse_transform(batch)
        xb = batch.x_dict["bus"].double().numpy()
        ptr = batch["bus"].ptr.numpy()
        sids = batch["scenario_id"].numpy() if "scenario_id" in batch.keys() else None
        for g, p in enumerate(acpf.extract_problems(batch)):
            raw_id = test_ids[sid]
            if sids is not None:
                assert int(sids[g]) == raw_id, (sids[g], raw_id)
            sl = slice(ptr[g], ptr[g + 1])
            n = ptr[g + 1] - ptr[g]
            s = st[sl]
            V = p.V_from(s[:, 0], s[:, 1])
            mis = np.abs(V * np.conj(p.Ybus @ V) - p.Sbus)
            mis[p.ref] = 0.0
            dva = np.abs(np.angle(V * np.conj(p.V_true)))
            dvm = np.abs(np.abs(V) - np.abs(p.V_true))
            btype = np.zeros(n); btype[p.pv] = 1; btype[p.ref] = 2
            edges = [tuple(e) for e in np.c_[p.f, p.t].tolist()]
            o = outages[raw_id]
            assert o["kind"] != "none" and o["n_el"] == 1, (raw_id, o)  # N-1 data: one contingency each
            rows.append(pd.DataFrame({
                "scenario": raw_id, "load_scenario": ls_of[raw_id], "bus": np.arange(n), "btype": btype,
                "outage_kind": 0.0 if o["kind"] == "branch" else 1.0, "mis": mis, "dva": dva, "dvm": dvm,
                "hops_slack": hops_from(n, edges, p.ref), "hops_outage": hops_from(n, edges, o["buses"])}))
            E_real.append(er[sl]); E_rand.append(en[sl])
            X_raw.append(np.c_[xb[sl, PD_H], xb[sl, QD_H], xb[sl, GS], xb[sl, BS], xb[sl, VM_H]])
            sid += 1
    assert sid == len(test_ids)
    df = pd.concat(rows, ignore_index=True)
    E_real, E_rand, X_raw = np.concatenate(E_real), np.concatenate(E_rand), np.concatenate(X_raw)
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache / f"{tag}.parquet")
        np.savez(cache / f"{tag}.npz", E_real=E_real, E_rand=E_rand, X_raw=X_raw)
    return df, E_real, E_rand, X_raw


# --------------------------------------------------------------------------------------
# ridge with grouped alpha selection (all work from Gram statistics)
# --------------------------------------------------------------------------------------
def _stats(X, y):
    return X.T @ X, X.T @ y, len(y), X.sum(0), y.sum()


def ridge_fit_predict(Xtr, ytr, gtr, Xte, alpha_mode, rng):
    """Centered ridge; alpha by inner grouped K-fold CV on the training rows."""
    if alpha_mode == "loo":  # v1 behaviour: sklearn RidgeCV (generalised leave-one-out, rows exchangeable)
        m = RidgeCV(alphas=ALPHAS).fit(Xtr, ytr)
        return m.predict(Xte), float(m.alpha_)
    ug = rng.permutation(np.unique(gtr))
    inner = {int(g): i % K_INNER for i, g in enumerate(ug)}
    fold = np.array([inner[int(g)] for g in gtr])
    G, b, n, sx, sy = _stats(Xtr, ytr)
    sse = np.zeros(len(ALPHAS))
    for k in range(K_INNER):
        h = fold == k
        Gh, bh, nh, sxh, syh = _stats(Xtr[h], ytr[h])
        nt, mx, my = n - nh, (sx - sxh) / (n - nh), (sy - syh) / (n - nh)
        Gc = (G - Gh) - nt * np.outer(mx, mx)
        bc = (b - bh) - nt * mx * my
        lam, V = eigh(Gc)
        lam = np.maximum(lam, 0)
        Wm = V @ ((V.T @ bc)[:, None] / (lam[:, None] + ALPHAS[None, :]))
        pred = (Xtr[h] - mx) @ Wm + my
        sse += ((ytr[h][:, None] - pred) ** 2).sum(0)
    a = ALPHAS[int(np.argmin(sse))]
    mx, my = sx / n, sy / n
    w = np.linalg.solve(G - n * np.outer(mx, mx) + a * np.eye(G.shape[0]), b - n * mx * my)
    return (Xte - mx) @ w + my, float(a)


def fold_assignment(groups, seed=0):
    ug = np.random.default_rng(seed).permutation(np.unique(groups))
    return {int(g): i % K_OUTER for i, g in enumerate(ug)}


# --------------------------------------------------------------------------------------
# feature blocks and sets
# --------------------------------------------------------------------------------------
def build_blocks(df, E_real, E_rand, X_raw, fit_rows):
    """Per-fold preprocessing: every block is standardised (and PCA'd) using train rows only."""
    nb = df.bus.max() + 1
    mis = np.log10(df.mis.to_numpy() + 1e-6)[:, None]
    struct = np.c_[np.eye(3)[df.btype.astype(int)], df.outage_kind, np.minimum(df.hops_slack, 20),
                   np.minimum(df.hops_outage, 20)]
    raw_in = np.c_[X_raw, np.eye(3)[df.btype.astype(int)]]
    blocks = {"mis": mis, "struct": struct, "bus": np.eye(nb)[df.bus.to_numpy()], "raw": raw_in,
              "emb": E_real, "remb": E_rand}
    out = {}
    for name, X in blocks.items():
        X = X.astype(np.float64)
        sc = StandardScaler().fit(X[fit_rows])
        out[name] = sc.transform(X)
    for src, pre in (("emb", "pca"), ("remb", "rpca")):
        pc = PCA(max(PCA_KS), svd_solver="full").fit(out[src][fit_rows])
        Z = pc.transform(out[src])
        for k in PCA_KS:
            Zk = Z[:, :k]
            sc = StandardScaler().fit(Zk[fit_rows])
            out[f"{pre}{k}"] = sc.transform(Zk)
    return out


def feature_sets():
    s = {
        "local mismatch only": ["mis"],
        "structure (type, outage kind, hops slack/outage)": ["struct"],
        "bus-ID one-hot (118)": ["bus"],
        "bus-ID + mismatch": ["bus", "mis"],
        "bus-ID + structure + mismatch": ["bus", "struct", "mis"],
        "raw local inputs (Pd, Qd, GS, BS, Vm-set, type)": ["raw"],
        "embedding (384)": ["emb"],
        "embedding + mismatch": ["emb", "mis"],
        "embedding + bus-ID + structure + mismatch": ["emb", "bus", "struct", "mis"],
        "random-init embedding (384)": ["remb"],
        "random-init embedding + mismatch": ["remb", "mis"],
    }
    for k in PCA_KS:
        s[f"PCA-{k} of embedding"] = [f"pca{k}"]
        s[f"PCA-{k} of embedding + mismatch"] = [f"pca{k}", "mis"]
        s[f"PCA-{k} of random-init embedding"] = [f"rpca{k}"]
        s[f"PCA-{k} of random-init embedding + mismatch"] = [f"rpca{k}", "mis"]
    return s


CONTRASTS = [  # (label, a, b): R2(a) - R2(b)
    ("embedding - bus-ID", "embedding (384)", "bus-ID one-hot (118)"),
    ("embedding - random-init embedding", "embedding (384)", "random-init embedding (384)"),
    ("embedding - raw local inputs", "embedding (384)", "raw local inputs (Pd, Qd, GS, BS, Vm-set, type)"),
    ("embedding+mis - bus-ID+mis", "embedding + mismatch", "bus-ID + mismatch"),
    ("embedding+mis - random embedding+mis", "embedding + mismatch", "random-init embedding + mismatch"),
    ("embedding+mis - bus-ID+struct+mis", "embedding + mismatch", "bus-ID + structure + mismatch"),
    ("embedding+all - bus-ID+struct+mis", "embedding + bus-ID + structure + mismatch", "bus-ID + structure + mismatch"),
    ("embedding+mis - mismatch only", "embedding + mismatch", "local mismatch only"),
    ("PCA-118 emb - bus-ID", "PCA-118 of embedding", "bus-ID one-hot (118)"),
    ("PCA-118 emb - PCA-118 random", "PCA-118 of embedding", "PCA-118 of random-init embedding"),
    ("PCA-20 emb - PCA-20 random", "PCA-20 of embedding", "PCA-20 of random-init embedding"),
    ("PCA-5 emb - PCA-5 random", "PCA-5 of embedding", "PCA-5 of random-init embedding"),
    ("PCA-20 emb - bus-ID", "PCA-20 of embedding", "bus-ID one-hot (118)"),
]


# --------------------------------------------------------------------------------------
# evaluation: out-of-fold predictions + cluster bootstrap
# --------------------------------------------------------------------------------------
def wranks(v, order, w):
    """Weighted mid-ranks (row i counted w_i times) given the sort order of v."""
    ws = w[order]
    cw = np.cumsum(ws)
    r = np.empty_like(cw)
    r[order] = cw - (ws - 1) / 2.0
    return r


def wpearson(a, b, w):
    W = w.sum()
    ma, mb = (w * a).sum() / W, (w * b).sum() / W
    da, db = a - ma, b - mb
    return (w * da * db).sum() / np.sqrt((w * da * da).sum() * (w * db * db).sum())


def evaluate(y, preds, base, gidx, n_groups, B, rng):
    """R2 / Spearman point estimates and paired cluster-bootstrap replicates over groups.

    `base` is the per-bus-mean predictor (train-fold mean of y at that bus): skill = 1 - SSE/SSE(base)
    measures what a feature set explains beyond a bus's average error level (bus identity).
    """
    names = list(preds)
    P = np.stack([preds[k] for k in names], 1)
    # point estimates
    r2 = 1 - ((P - y[:, None]) ** 2).sum(0) / ((y - y.mean()) ** 2).sum()
    order_y = np.argsort(y, kind="stable")
    order_p = [np.argsort(P[:, j], kind="stable") for j in range(P.shape[1])]
    one = np.ones(len(y))
    ry = wranks(y, order_y, one)
    sp = np.array([wpearson(ry, wranks(P[:, j], order_p[j], one), one) for j in range(P.shape[1])])
    # per-group aggregates for the R2 bootstrap
    cnt = np.bincount(gidx, minlength=n_groups).astype(float)
    sy = np.bincount(gidx, weights=y, minlength=n_groups)
    syy = np.bincount(gidx, weights=y * y, minlength=n_groups)
    sse = np.stack([np.bincount(gidx, weights=(P[:, j] - y) ** 2, minlength=n_groups) for j in range(P.shape[1])], 1)
    sse_base = np.bincount(gidx, weights=(base - y) ** 2, minlength=n_groups)
    skill = 1 - sse.sum(0) / sse_base.sum()
    C = rng.multinomial(n_groups, np.full(n_groups, 1.0 / n_groups), size=B).astype(float)  # B x G
    N = C @ cnt
    sst = C @ syy - (C @ sy) ** 2 / N
    r2_b = 1 - (C @ sse) / sst[:, None]
    skill_b = 1 - (C @ sse) / (C @ sse_base)[:, None]
    sp_b = np.empty((B, P.shape[1]))
    for b in range(B):
        w = C[b][gidx]
        r_y = wranks(y, order_y, w)
        for j in range(P.shape[1]):
            sp_b[b, j] = wpearson(r_y, wranks(P[:, j], order_p[j], w), w)
    return names, r2, sp, r2_b, sp_b, skill, skill_b


def ci(x):
    return np.percentile(x, [2.5, 97.5], axis=0)


def run_model(size, seed, args, cache):
    t0 = time.time()
    df, E_real, E_rand, X_raw = collect(size, seed, args.grid, rand_seed=1000 + seed, cache=cache)
    print(f"[{size}:{seed}] collected {len(df)} rows in {time.time() - t0:.0f}s; "
          f"random-init embedding std {E_rand.std():.3g} (real {E_real.std():.3g})", flush=True)
    groups = df.load_scenario.to_numpy().astype(int)
    fmap = fold_assignment(groups)
    fold = np.array([fmap[g] for g in groups])
    btype = df.btype.to_numpy()
    nonslack = btype != 2
    targets = {
        "log|dVm| (PQ buses)": (np.log10(df.dvm.to_numpy() + 1e-9), nonslack & (btype == 0) & (df.dvm.to_numpy() > 0)),
        "log|dVa| (non-slack)": (np.log10(df.dva.to_numpy() + 1e-9), nonslack),
    }
    sets = feature_sets()
    oof = {t: {name: np.full(len(df), np.nan) for name in sets} for t in targets}
    alphas = {t: {name: [] for name in sets} for t in targets}
    busmean = {t: np.full(len(df), np.nan) for t in targets}
    bus_ix = df.bus.to_numpy()
    rng = np.random.default_rng(seed + 17)
    for k in range(K_OUTER):
        te, tr = fold == k, fold != k
        blocks = build_blocks(df, E_real, E_rand, X_raw, fit_rows=tr & nonslack)
        for tname, (y, mask) in targets.items():
            rtr, rte = tr & mask, te & mask
            nbus = bus_ix.max() + 1
            cnt_b = np.bincount(bus_ix[rtr], minlength=nbus)
            mean_b = np.bincount(bus_ix[rtr], weights=y[rtr], minlength=nbus) / np.maximum(cnt_b, 1)
            busmean[tname][rte] = mean_b[bus_ix[rte]]
            for name, parts in sets.items():
                X = np.concatenate([blocks[p] for p in parts], 1)
                pred, a = ridge_fit_predict(X[rtr], y[rtr], groups[rtr], X[rte], args.alpha_mode, rng)
                oof[tname][name][rte] = pred
                alphas[tname][name].append(a)
        print(f"[{size}:{seed}] fold {k + 1}/{K_OUTER} done ({time.time() - t0:.0f}s)", flush=True)
    rows, contrasts = [], []
    ug = np.unique(groups)
    for tname, (y, mask) in targets.items():
        gidx = np.searchsorted(ug, groups[mask])
        names, r2, sp, r2_b, sp_b, sk, sk_b = evaluate(
            y[mask], {n: oof[tname][n][mask] for n in sets}, busmean[tname][mask], gidx, len(ug), args.boot,
            np.random.default_rng(1234))
        lo, hi = ci(r2_b); slo, shi = ci(sp_b); klo, khi = ci(sk_b)
        for j, n in enumerate(names):
            rows.append({"model": f"{size}:{seed}", "size": size, "seed": seed, "feature_set": n, "target": tname,
                         "R2": r2[j], "R2_lo": lo[j], "R2_hi": hi[j], "Spearman": sp[j], "Spearman_lo": slo[j],
                         "Spearman_hi": shi[j], "skill_vs_busmean": sk[j], "skill_lo": klo[j],
                         "skill_hi": khi[j], "n_rows": int(mask.sum()), "n_groups": len(ug),
                         "alpha_median": float(np.median(alphas[tname][n]))})
        idx = {n: j for j, n in enumerate(names)}
        for lab, a, b in CONTRASTS:
            d, db = r2[idx[a]] - r2[idx[b]], r2_b[:, idx[a]] - r2_b[:, idx[b]]
            ds, dsb = sp[idx[a]] - sp[idx[b]], sp_b[:, idx[a]] - sp_b[:, idx[b]]
            contrasts.append({"model": f"{size}:{seed}", "size": size, "seed": seed, "contrast": lab, "target": tname,
                              "dR2": d, "dR2_lo": np.percentile(db, 2.5), "dR2_hi": np.percentile(db, 97.5),
                              "dSpearman": ds, "dSpearman_lo": np.percentile(dsb, 2.5),
                              "dSpearman_hi": np.percentile(dsb, 97.5)})
    print(f"[{size}:{seed}] done in {time.time() - t0:.0f}s", flush=True)
    return rows, contrasts


def summarize(rows, contrasts):
    d = pd.DataFrame(rows)
    c = pd.DataFrame(contrasts)
    base = d[d["size"] == "base"]
    summ = {"metrics_base_mean_sd_over_seeds": {}, "metrics_small_seed0": {}, "contrasts_base_mean_sd_over_seeds": {},
            "contrasts_small_seed0": {}, "base_seeds": sorted(base.seed.unique().tolist())}
    for (t, f), g in base.groupby(["target", "feature_set"]):
        summ["metrics_base_mean_sd_over_seeds"].setdefault(t, {})[f] = {
            "R2_mean": g.R2.mean(), "R2_sd": g.R2.std(ddof=1), "Spearman_mean": g.Spearman.mean(),
            "Spearman_sd": g.Spearman.std(ddof=1), "skill_vs_busmean_mean": g.skill_vs_busmean.mean(), "skill_vs_busmean_sd": g.skill_vs_busmean.std(ddof=1),
            "R2_per_seed": g.R2.round(4).tolist(),
            "R2_CI_per_seed": [[round(a, 3), round(b, 3)] for a, b in zip(g.R2_lo, g.R2_hi)]}
    for _, r in d[d["size"] == "small"].iterrows():
        summ["metrics_small_seed0"].setdefault(r.target, {})[r.feature_set] = {
            "R2": r.R2, "R2_lo": r.R2_lo, "R2_hi": r.R2_hi, "Spearman": r.Spearman,
            "skill_vs_busmean": r.skill_vs_busmean}
    cb = c[c["size"] == "base"]
    for (t, k), g in cb.groupby(["target", "contrast"]):
        summ["contrasts_base_mean_sd_over_seeds"].setdefault(t, {})[k] = {
            "dR2_mean": g.dR2.mean(), "dR2_sd": g.dR2.std(ddof=1), "dSpearman_mean": g.dSpearman.mean(),
            "dSpearman_sd": g.dSpearman.std(ddof=1), "dR2_per_seed": g.dR2.round(4).tolist(),
            "dR2_CI_per_seed": [[round(a, 3), round(b, 3)] for a, b in zip(g.dR2_lo, g.dR2_hi)],
            "seeds_with_CI_excluding_0": int(((g.dR2_lo > 0) | (g.dR2_hi < 0)).sum())}
    for _, r in c[c["size"] == "small"].iterrows():
        summ["contrasts_small_seed0"].setdefault(r.target, {})[r.contrast] = {
            "dR2": r.dR2, "dR2_lo": r.dR2_lo, "dR2_hi": r.dR2_hi}
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="case118_ieee")
    ap.add_argument("--models", default="base:0,base:1,base:42,small:0")
    ap.add_argument("--alpha_mode", default="group", choices=["group", "loo"])
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--cache_dir", default=None, help="cache embeddings/targets (large; not for results/)")
    ap.add_argument("--tag", default="", help="suffix for output files (e.g. _loo)")
    args = ap.parse_args()
    from pathlib import Path
    cache = Path(args.cache_dir) if args.cache_dir else None
    rows, contrasts = [], []
    for m in args.models.split(","):
        size, seed = m.split(":")
        r, c = run_model(size, int(seed), args, cache)
        rows += r; contrasts += c
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).round(5).to_csv(OUT / f"probe_v2{args.tag}.csv", index=False)
    pd.DataFrame(contrasts).round(5).to_csv(OUT / f"probe_v2_contrasts{args.tag}.csv", index=False)
    summ = summarize(rows, contrasts)
    summ["config"] = {"K_outer": K_OUTER, "K_inner": K_INNER, "alpha_mode": args.alpha_mode, "bootstrap": args.boot,
                      "pca_ks": list(PCA_KS), "models": args.models, "random_init_seed": "1000 + model seed"}
    (OUT / f"probe_v2_summary{args.tag}.json").write_text(json.dumps(summ, indent=1, default=float))


if __name__ == "__main__":
    main()
