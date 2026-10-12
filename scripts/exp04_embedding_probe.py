"""Experiment 04: does GENCO's embedding space know where GENCO is wrong?

Per bus, predict GENCO's true state error (|ΔVa|, |ΔVm| vs the AC solution, log10) from:
  (a) GENCO's own local power mismatch at its output (ground-truth-free),
  (b) structural features (bus type, hops to slack, hops to nearest outaged element),
  (c) the final-layer bus embedding (the decoder-head input, 384-d for Base),
  (d) embedding + local mismatch.
Ridge regression, scenarios split 50/50. Also saves a 2-D PCA of embeddings.

Usage: python scripts/exp04_embedding_probe.py --grid case118_ieee --size base
"""

import argparse
import json

import networkx as nx
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from genco_certified import acpf
from genco_certified.loading import ROOT, load_genco
from genco_certified.tracing import trace_layers

OUT = ROOT / "results" / "exp04"


def hops(p, sources):
    G = nx.Graph()
    G.add_nodes_from(range(len(p.Vm_set)))
    G.add_edges_from(zip(p.f.tolist(), p.t.tolist()))
    d = nx.multi_source_dijkstra_path_length(G, set(int(s) for s in sources)) if len(sources) else {}
    return np.array([d.get(i, 99) for i in range(len(p.Vm_set))], dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="case118_ieee")
    ap.add_argument("--size", default="base")
    args = ap.parse_args()
    Lg = load_genco("pf", args.grid, args.size, batch_size=32)
    rows, embs = [], []
    sid = 0
    full_edges = None
    for batch in Lg.test_loader():
        with torch.no_grad(), trace_layers(Lg.task.model, capture_embeddings=True) as tr:
            Lg.task.model(batch)
        st = tr["state"][-1].double().numpy()
        emb = tr["embedding"][-1].numpy()
        Lg.normalizer.inverse_transform(batch)
        ptr = batch["bus"].ptr.numpy()
        for g, p in enumerate(acpf.extract_problems(batch)):
            sl = slice(ptr[g], ptr[g + 1])
            s = st[sl]
            V = p.V_from(s[:, 0], s[:, 1])
            mis = np.abs(V * np.conj(p.Ybus @ V) - p.Sbus)
            mis[p.ref] = 0.0
            dva = np.abs(np.angle(V * np.conj(p.V_true)))
            dvm = np.abs(np.abs(V) - np.abs(p.V_true))
            # outaged elements = branches of the intact graph missing in this scenario (bus endpoints)
            edges = set(map(tuple, np.sort(np.c_[p.f, p.t], axis=1).tolist()))
            full_edges = edges if full_edges is None or len(edges) > len(full_edges) else full_edges
            n = len(s)
            btype = np.zeros(n); btype[p.pv] = 1; btype[p.ref] = 2
            rows.append(pd.DataFrame({"scenario": sid, "bus": np.arange(n), "btype": btype, "mis": mis,
                                      "dva": dva, "dvm": dvm, "hops_slack": hops(p, p.ref), "_edges": [edges] * n}))
            embs.append(emb[sl])
            sid += 1
    df = pd.concat(rows, ignore_index=True)
    E = np.concatenate(embs)
    # hops to the outage: endpoints of intact-grid branches absent from this scenario
    out_h = []
    for sid_, d in df.groupby("scenario", sort=True):
        missing = full_edges - d["_edges"].iloc[0]
        ends = {u for e in missing for u in e}
        G = nx.Graph(); G.add_nodes_from(range(len(d))); G.add_edges_from(d["_edges"].iloc[0])
        dist = nx.multi_source_dijkstra_path_length(G, ends) if ends else {}
        out_h.append(np.array([dist.get(i, 99) for i in range(len(d))], dtype=float))
    df["hops_outage"] = np.concatenate(out_h)
    df = df.drop(columns="_edges")
    mask = df.btype.to_numpy() != 2  # slack bus has no unknowns
    rng = np.random.default_rng(0)
    scen = df.scenario.unique(); rng.shuffle(scen)
    tr_s = set(scen[: len(scen) // 2])
    train = df.scenario.isin(tr_s).to_numpy() & mask
    test = ~df.scenario.isin(tr_s).to_numpy() & mask

    feats = {
        "local mismatch only": np.c_[np.log10(df.mis + 1e-6)],
        "structure (type, hops to slack/outage)": np.c_[np.eye(3)[df.btype.astype(int)], np.minimum(df.hops_slack, 20), np.minimum(df.hops_outage, 20)],
        "embedding": E,
        "embedding + local mismatch": np.c_[E, np.log10(df.mis + 1e-6)],
    }
    res = {}
    for target in ["dva", "dvm"]:
        y = np.log10(df[target].to_numpy() + 1e-9)
        tmask_tr = train & (df[target].to_numpy() > 0) if target == "dvm" else train
        tmask_te = test & (df[target].to_numpy() > 0) if target == "dvm" else test
        if target == "dvm":  # PV buses have Vm fixed (zero error); probe PQ buses only
            tmask_tr &= df.btype.to_numpy() == 0
            tmask_te &= df.btype.to_numpy() == 0
        for name, X in feats.items():
            sc = StandardScaler().fit(X[tmask_tr])
            m = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(sc.transform(X[tmask_tr]), y[tmask_tr])
            pred = m.predict(sc.transform(X[tmask_te]))
            r2 = 1 - np.mean((pred - y[tmask_te]) ** 2) / np.var(y[tmask_te])
            rho = spearmanr(pred, y[tmask_te])[0]
            res[f"{target} | {name}"] = {"R2": round(float(r2), 3), "spearman": round(float(rho), 3)}
            print(f"{target:3s} | {name:40s} R2 {r2:6.3f}  Spearman {rho:6.3f}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.grid}_{args.size}_probe.json").write_text(json.dumps(res, indent=1))
    pca = PCA(2).fit(E[mask])
    Z = pca.transform(E)
    pd.DataFrame({"pc1": Z[:, 0], "pc2": Z[:, 1], **{c: df[c] for c in ["scenario", "bus", "btype", "hops_slack", "hops_outage", "dva", "dvm", "mis"]}}
                 ).to_parquet(OUT / f"{args.grid}_{args.size}_pca.parquet")
    print("PCA explained variance:", pca.explained_variance_ratio_.round(3))
    print("corr(PC1, hops_slack) spearman:", round(spearmanr(Z[mask, 0], df.hops_slack[mask])[0], 3),
          "| corr(PC2, hops_slack):", round(spearmanr(Z[mask, 1], df.hops_slack[mask])[0], 3))


if __name__ == "__main__":
    main()
