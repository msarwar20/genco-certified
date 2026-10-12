"""Tests for the corrected screening pipeline (scripts/analyze_exp02.py, acpf additions).

Data-dependent tests need data/pf/case118_ieee (scripts/fetch_data.py) and skip otherwise.
"""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from genco_certified import acpf

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "pf" / "case118_ieee" / "raw"
need_data = pytest.mark.skipif(not RAW.exists(), reason="run scripts/fetch_data.py first")

_spec = importlib.util.spec_from_file_location("analyze_exp02", ROOT / "scripts" / "analyze_exp02.py")
an = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(an)


# ------------------------------------------------------------------ LU accounting
def test_lu_accounting_charges_own_nr_count():
    df = pd.DataFrame({"tier1": [True, False, False, True], "nr_iters": [2, 5, 3, 2]})
    # accepted scenarios cost 1; escalated ones cost their OWN NR iterations (5 and 3), not the mean (3)
    assert an.tiered_lu(df.tier1, df.nr_iters) == pytest.approx((1 + 5 + 3 + 1) / 4)
    # the old bug: escalated share x all-scenario mean NR count
    share = (~df.tier1).mean()
    old = (1 - share) * 1 + share * df.nr_iters.mean()
    assert old == pytest.approx((1 + 3 + 3 + 1) / 4)
    assert an.tiered_lu(df.tier1, df.nr_iters) > old  # escalated scenarios are the hard ones


def test_lu_accounting_sequential_and_floor():
    acc = np.array([1, 2, 0, 3, 0])
    nr = np.array([2, 2, 6, 3, 1])
    # accepted at step k costs k; escalated costs its own NR count but never less than the steps already spent
    assert an.lu_per_scenario(acc, nr, last_step=3).tolist() == [1, 2, 6, 3, 3]
    assert an.lu_per_scenario(np.array([0]), np.array([0]), last_step=1).tolist() == [1]  # zero-iteration NR: the step was taken


# ------------------------------------------------------------------ grouped folds
def test_grouped_folds_disjoint_and_balanced():
    rng = np.random.default_rng(0)
    groups = rng.integers(0, 29, size=638)          # 29 load scenarios, 638 contingencies
    for K in (2, 5, 29, 50):
        for seed in range(5):
            fold = an.grouped_folds(groups, K, seed)
            k_eff = min(K, 29)
            assert fold.min() == 0 and fold.max() == k_eff - 1
            # no load scenario in two folds -> calibration and evaluation never share one
            for g in np.unique(groups):
                assert len(np.unique(fold[groups == g])) == 1
            sizes = np.bincount(np.unique(np.c_[groups, fold], axis=0)[:, 1])
            assert sizes.max() - sizes.min() <= 1
    assert (an.grouped_folds(groups, 5, 3) == an.grouped_folds(groups, 5, 3)).all()
    assert (an.grouped_folds(groups, 5, 3) != an.grouped_folds(groups, 5, 4)).any()


def _toy_set(tmp_path, n_loads=6, per_load=3, nb=2, nv=2, seed=0):
    """A synthetic exp02 run (g1/b1 only) with known errors, for pipeline tests."""
    rng = np.random.default_rng(seed)
    n = n_loads * per_load
    sc = pd.DataFrame({"scenario": np.arange(n), "load_idx": np.repeat(np.arange(n_loads), per_load),
                       "step_l2": rng.uniform(0.01, 0.1, n), "step_base_l2": rng.uniform(0.05, 0.3, n),
                       "nr_iters_genco": rng.integers(2, 4, n), "nr_iters_base": rng.integers(2, 6, n),
                       "nr_iters_flat": 4, "nr_iters_dc": 4})
    br = pd.DataFrame({"scenario": np.repeat(np.arange(n), nb), "branch": np.tile(np.arange(nb), n)})
    bu = pd.DataFrame({"scenario": np.repeat(np.arange(n), nv), "bus": np.tile(np.arange(nv), n), "btype": 0,
                       "vmin": 0.94, "vmax": 1.06})
    br["load_true"] = rng.uniform(0.2, 0.9, len(br))
    bu["vm_true"] = rng.uniform(0.98, 1.02, len(bu))
    for m, step in (("step1", "step_l2"), ("base1", "step_base_l2")):
        d = np.repeat(sc[step].to_numpy() ** 2, nb)
        br[f"load_{m}"] = br.load_true + 0.5 * d
        bu[f"vm_{m}"] = bu.vm_true + 0.25 * np.repeat(sc[step].to_numpy() ** 2, nv)
    br["load_raw"], bu["vm_raw"] = br.load_true, bu.vm_true
    br["load_dc"] = br.load_true
    br.to_parquet(tmp_path / "toy_branches.parquet")
    bu.to_parquet(tmp_path / "toy_buses.parquet")
    sc.to_parquet(tmp_path / "toy_scenarios.parquet")
    return an.RawSet("toy", tmp_path)


def test_kfold_calibration_never_shares_a_load_scenario(tmp_path):
    S = _toy_set(tmp_path)
    c_s, fold_s, info = an.kfold_c(S, "g", "max", K=3, seed=1)
    ls, ls_score = S.per_load_max(S.ratio_scores("g", 1))
    assert len(ls) == 6 and fold_s.max() == 2
    for f in range(3):
        cal_loads = np.unique(S.load_idx[fold_s != f])
        ev_loads = np.unique(S.load_idx[fold_s == f])
        assert not set(cal_loads) & set(ev_loads)
        # the fold constant is exactly the max score over the OTHER folds' load scenarios
        expect = ls_score[np.isin(ls, cal_loads)].max()
        assert info["c_folds"][f] == pytest.approx(expect)
        assert (c_s[fold_s == f] == info["c_folds"][f]).all()


def test_ratio_scores_and_policy_on_toy_set(tmp_path):
    S = _toy_set(tmp_path)
    # error of step1 = 0.5 d (branch) and 0.25 d (bus) with d = ||dx||^2 -> ratio 0.5
    assert np.allclose(S.ratio_scores("g", 1), 0.5)
    scr = an.Screen(S, tau=1.01, eps_v=0.0)
    assert scr.ntruth.sum() == 0  # nothing violates by construction
    # c just above the true ratio, margins are large -> everything accepted after one step, cost 1
    o = an.run_policy(scr, "g", (1,), 0.6)
    assert (o.acc == 1).all() and (o.cost == 1).all()
    # c enormous -> everything escalated, each charged its OWN NR iteration count
    o = an.run_policy(scr, "g", (1,), 1e9)
    assert (o.acc == 0).all() and np.allclose(o.cost, np.maximum(S.sc.nr_iters_genco.to_numpy(), 1))
    # infinite c (too few calibration scenarios for conformal) escalates everything without NaN problems
    o = an.run_policy(scr, "b", (1,), math.inf)
    assert (o.acc == 0).all()


# ------------------------------------------------------------------ conformal quantile
def test_conformal_quantile_edge_cases():
    s = np.arange(1, 30, dtype=float)[::-1]          # 29 scores, shuffled order must not matter
    assert an.conformal_quantile(s, 0.10) == 27.0     # ceil(30 * 0.9) = 27th smallest (float-safe)
    assert an.conformal_quantile(s, 0.05) == 29.0     # ceil(30 * 0.95) = 29 -> the max
    assert an.conformal_quantile(s, 0.0) == math.inf  # rank 30 > n
    assert an.conformal_quantile(s, 1.0) == 1.0       # rank clipped to the smallest score
    assert an.conformal_quantile([], 0.1) == math.inf
    assert an.conformal_quantile([5.0], 0.5) == 5.0   # rank ceil(2 * 0.5) = 1
    assert an.conformal_quantile([5.0], 0.4) == math.inf
    assert an.conformal_quantile(np.arange(9.0), 0.10) == 8.0          # n = 9: rank 9 exists
    assert an.conformal_quantile(np.arange(8.0), 0.10) == math.inf     # n = 8: needs rank 9 > 8
    assert an.conformal_quantile([1.0, 2.0, math.inf], 0.1) == math.inf
    with pytest.raises(ValueError):
        an.conformal_quantile([1.0, float("nan")], 0.1)
    assert an.choose_c([1.0, 3.0, 2.0], "max") == 3.0
    assert an.choose_c([], "max") == math.inf
    assert an.choose_c(np.arange(1.0, 30.0), "conf0.10") == 27.0


def test_conformal_coverage_is_at_least_nominal():
    rng = np.random.default_rng(1)
    n, alpha, trials = 29, 0.10, 6000
    cov = [rng.random() <= an.conformal_quantile(rng.random(n), alpha) for _ in range(trials)]
    assert 0.88 < np.mean(cov) < 0.93  # exact coverage is ceil(30 * 0.9) / 30 = 0.9


def test_tune_threshold_fbeta_and_recall():
    rng = np.random.default_rng(0)
    truth = rng.random(20000) < 0.02
    load = np.where(truth, rng.normal(1.05, 0.1, truth.size), rng.normal(0.55, 0.2, truth.size))
    t = an.tune_threshold(load, truth, beta=3.0, recall_target=0.99)
    rec = lambda th: (load[truth] > th).mean()
    # largest grid threshold that still reaches the recall target
    assert rec(t["recall"]) >= 0.99 and rec(t["recall"] + 0.0025) < 0.99

    def f3(th):
        tp = ((load > th) & truth).sum(); fp = ((load > th) & ~truth).sum(); fn = ((load <= th) & truth).sum()
        return 10 * tp / (10 * tp + 9 * fn + fp)
    grid = np.arange(0.30, 1.50, 0.0025)
    assert f3(t["fbeta"]) >= max(f3(g) for g in grid) - 1e-12  # global maximiser over the grid
    assert f3(t["fbeta"]) >= f3(t["recall"]) - 1e-12


# ------------------------------------------------------------------ datakit DC loading
def test_dc_loading_merges_parallel_circuits_and_drops_outages():
    br = pd.DataFrame({"from_bus": [0, 1, 1, 2], "to_bus": [1, 0, 2, 0], "br_status": [1, 1, 1, 0],
                       "pf_dc": [50.0, -30.0, 20.0, 999.0], "pt_dc": [-50.0, 30.0, -25.0, -999.0],
                       "rate_a": [100.0, 100.0, 0.0, 10.0]})
    out = acpf.dc_loading_from_raw_rows(br, 3)
    # keys: (0,1) -> merged parallel circuits max(0.5, 0.3); (1,2) has rate 0 -> 0; (0,2) is out of service and dropped
    assert out.tolist() == [0.5, 0.0]
    with pytest.raises(AssertionError):
        acpf.dc_loading_from_raw_rows(br, 3, expected_keys=np.array([1, 5, 2]))


@need_data
def test_dc_loading_from_raw_matches_definition():
    part = "scenario_partition=0"
    bus, br = (pd.read_parquet(RAW / f"{t}_data.parquet" / part) for t in ("bus", "branch"))
    gen = pd.read_parquet(RAW / "gen_data.parquet" / part)
    out_of_service = br[br.br_status == 0].scenario.unique()
    s = int(out_of_service[0]) if len(out_of_service) else 0   # prefer a contingency scenario
    b, r, g = (d[d.scenario == s] for d in (bus, br, gen))
    p = acpf.problem_from_raw(b, r, g)
    got = acpf.dc_loading_from_raw(p, r)  # also asserts the key sets match the problem's
    ins = r[r.br_status == 1]
    expect = {}
    for row in ins.itertuples():
        k = (min(row.from_bus, row.to_bus), max(row.from_bus, row.to_bus))
        v = max(abs(row.pf_dc), abs(row.pt_dc)) / row.rate_a if row.rate_a > 0 else 0.0
        expect[k] = max(expect.get(k, 0.0), v)
    assert len(got) == len(expect) == len(acpf.branch_key_values(p))
    n = len(b)
    keys = acpf.branch_key_values(p)
    for key, val in zip(keys, got):
        assert val == pytest.approx(expect[(key // (n + 1), key % (n + 1))])
    if len(out_of_service):
        assert (r.br_status == 0).any() and len(got) < len(r)  # the outaged branch is not in the output


@need_data
def test_newton_trajectory_matches_newton_pf():
    part = "scenario_partition=0"
    bus, br, gen = (pd.read_parquet(RAW / f"{t}_data.parquet" / part) for t in ("bus", "branch", "gen"))
    s = 4
    p = acpf.problem_from_raw(*(d[d.scenario == s] for d in (bus, br, gen)))
    ref = acpf.newton_pf(p, p.flat_start())
    tr = acpf.newton_trajectory(p, p.flat_start(), n_steps=3)
    assert tr["iterations"] == ref["iterations"] and tr["converged"] == ref["converged"]
    assert np.allclose(tr["V"][ref["iterations"] - 1], ref["V"])
    assert len(tr["V"]) >= 3 and np.allclose(tr["history"][: ref["iterations"] + 1], ref["history"])
    assert acpf.iterations_to_tol(ref["history"], 1e-6) == ref["iterations"]
    assert acpf.iterations_to_tol(ref["history"], 1e-2) <= acpf.iterations_to_tol(ref["history"], 1e-6)
    assert acpf.iterations_to_tol([5.0, 1.0], 1e-6) == -1


def test_sequential_policy_accepts_at_the_first_step_that_clears_the_bound(tmp_path):
    """Three scenarios with margins 0.5 / 5e-4 / 1e-12 to the loading limit, c = 1, ||dx_k||^2 = 1e-2 / 1e-6 / 1e-10 at steps 1 / 2 / 3."""
    tau, margins = 1.01, np.array([0.5, 5e-4, 1e-12])
    n = 3
    sc = pd.DataFrame({"scenario": np.arange(n), "load_idx": np.arange(n),
                       "step_g1_l2": np.full(n, 0.1), "step_g2_l2": np.full(n, 1e-3), "step_g3_l2": np.full(n, 1e-5),
                       "step_b1_l2": np.full(n, 0.1), "step_b2_l2": np.full(n, 1e-3), "step_b3_l2": np.full(n, 1e-5),
                       "nr_iters_genco": [2, 2, 5], "nr_iters_base": [2, 2, 5]})
    br = pd.DataFrame({"scenario": np.arange(n), "branch": 0})
    bu = pd.DataFrame({"scenario": np.arange(n), "bus": 0, "btype": 0, "vmin": 0.9, "vmax": 1.1, "vm_true": 1.0})
    br["load_true"] = tau - margins
    for m in ("g1", "g2", "g3", "b1", "b2", "b3", "nr", "raw", "dc"):
        br[f"load_{m}"] = br.load_true
        bu[f"vm_{m}"] = bu.vm_true
    br.to_parquet(tmp_path / "seq_branches.parquet")
    bu.to_parquet(tmp_path / "seq_buses.parquet")
    sc.to_parquet(tmp_path / "seq_scenarios.parquet")
    S = an.RawSet("seq", tmp_path)
    scr = an.Screen(S, tau=tau, eps_v=0.0)
    o = an.run_policy(scr, "g", (1, 2, 3), 1.0)
    assert o.acc.tolist() == [1, 2, 0]          # scenario 3 never clears c * ||dx||^2 -> escalates
    assert o.cost.tolist() == [1, 2, 5]          # accepted at step k costs k; escalated costs its own NR count (5 >= 3)
    t = an.run_policy(scr, "g", (1,), 1.0)       # one-step tiered rule escalates scenarios 2 and 3
    assert t.acc.tolist() == [1, 0, 0] and t.cost.tolist() == [1, 2, 5]
    # an escalated scenario whose NR needed fewer iterations than the steps already spent is charged the steps spent
    S.sc.loc[2, "nr_iters_genco"] = 1
    assert an.run_policy(scr, "g", (1, 2, 3), 1.0).cost.tolist() == [1, 2, 3]
