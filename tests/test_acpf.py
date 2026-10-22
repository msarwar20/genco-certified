"""Unit tests for the AC power-flow machinery (run: pytest -q). Needs data/pf/case118_ieee (scripts/fetch_data.py)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from genco_certified import acpf

RAW = Path(__file__).resolve().parents[1] / "data" / "pf" / "case118_ieee" / "raw"
pytestmark = [pytest.mark.data, pytest.mark.skipif(not RAW.exists(), reason="run scripts/fetch_data.py first")]


@pytest.fixture(scope="module")
def raw():
    part = "scenario_partition=0"
    return {t: pd.read_parquet(RAW / f"{t}_data.parquet" / part) for t in ["bus", "branch", "gen"]}


def scenario(raw, s):
    return [raw[t][raw[t].scenario == s] for t in ["bus", "branch", "gen"]]


def test_ground_truth_satisfies_power_balance(raw):
    p = acpf.problem_from_raw(*scenario(raw, 0))
    assert np.abs(acpf.mismatch(p, p.V_true)).max() < 1e-6  # MW


def test_flat_start_newton_recovers_datakit_solution(raw):
    p = acpf.problem_from_raw(*scenario(raw, 3))
    r = acpf.newton_pf(p, p.flat_start())
    assert r["converged"] and r["iterations"] <= 6
    assert np.abs(acpf.state_error(p, r["V"])).max() < 1e-8


def test_newton_converges_quadratically(raw):
    p = acpf.problem_from_raw(*scenario(raw, 5))
    h = acpf.newton_pf(p, p.flat_start(), tol_mw=1e-9)["history"]
    # once in the quadratic regime the residual roughly squares each iteration
    assert h[-1] < 1e-9 and h[-2] < 1e-2


def test_jacobian_matches_finite_differences(raw):
    p = acpf.problem_from_raw(*scenario(raw, 0))
    V = p.V_true * np.exp(1j * 1e-3)  # off the solution
    J = acpf.jacobian(p, V).toarray()
    x = np.r_[np.angle(V)[p.pvpq], np.abs(V)[p.pq]]
    def F(x):
        Va, Vm = np.angle(V).copy(), np.abs(V).copy()
        Va[p.pvpq] = x[: len(p.pvpq)]; Vm[p.pq] = x[len(p.pvpq):]
        return acpf.mismatch(p, Vm * np.exp(1j * Va))
    k = 7
    e = np.zeros_like(x); e[k] = 1e-6
    fd = (F(x + e) - F(x - e)) / 2e-6
    assert np.allclose(fd, J[:, k], rtol=1e-4, atol=1e-3)


def test_base_case_warm_start_beats_flat_start(raw):
    ls = raw["bus"].load_scenario_idx.min()
    sel = [raw[t][raw[t].load_scenario_idx == ls] for t in ["bus", "branch", "gen"]]
    bc = acpf.base_case_voltages(*sel)
    assert bc["converged"]
    its = []
    for s in sorted(sel[0].scenario.unique())[:6]:
        p = acpf.problem_from_raw(*scenario(raw, s))
        its.append(acpf.newton_pf(p, p.V_from(np.abs(bc["V"]), np.angle(bc["V"])))["iterations"])
    assert np.mean(its) < 4


def test_branch_loading_uses_both_directions(raw):
    p = acpf.problem_from_raw(*scenario(raw, 0))
    S = acpf.branch_flows(p, p.V_true)
    L = acpf.branch_loading(p, S)
    assert L.shape[0] == len(np.unique(acpf.branch_keys(p)))
    assert (L >= 0).all() and L.max() < 2
