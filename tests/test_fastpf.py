"""Tests for the factor-reuse contingency solvers (run: pytest -q). Needs data/pf/case118_ieee (scripts/fetch_data.py)."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse.linalg as spla

from genco_certified import acpf, fastpf

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "pf" / "case118_ieee" / "raw"
need_data = pytest.mark.skipif(not RAW.exists(), reason="run scripts/fetch_data.py first")
TOL = 1e-4  # MW

_spec = importlib.util.spec_from_file_location("analyze_exp02", ROOT / "scripts" / "analyze_exp02.py")
an = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(an)


@pytest.fixture(scope="module")
def load_scenario():
    part = "scenario_partition=0"
    raw = {t: pd.read_parquet(RAW / f"{t}_data.parquet" / part) for t in ["bus", "branch", "gen"]}
    ls = int(sorted(raw["bus"].load_scenario_idx.unique())[0])
    return {t: d[d.load_scenario_idx == ls] for t, d in raw.items()}


@pytest.fixture(scope="module")
def base(load_scenario):
    b = fastpf.solve_base_case(load_scenario["bus"], load_scenario["branch"], load_scenario["gen"])
    assert b.converged
    return b


def contingencies(ls):
    """(scenario id, problem, branch rows, n branches out, n gens out) for every variant of the load scenario."""
    out = []
    for s in sorted(ls["bus"].scenario.unique()):
        b, br, g = (ls[t][ls[t].scenario == s] for t in ["bus", "branch", "gen"])
        out.append((int(s), acpf.problem_from_raw(b, br, g), br, int((br.br_status == 0).sum()), int((g.in_service == 0).sum())))
    return out


def safe_outages(br, k, seed):
    """k in-service branches whose removal keeps the network connected (deterministic)."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    rng = np.random.default_rng(seed)
    ins = br[br.br_status == 1]
    for _ in range(1000):
        kill = ins.idx.to_numpy()[rng.choice(len(ins), k, replace=False)]
        rest = ins[~ins.idx.isin(kill)]
        n = int(max(br.from_bus.max(), br.to_bus.max())) + 1
        A = sp.coo_matrix((np.ones(len(rest)), (rest.from_bus.astype(int), rest.to_bus.astype(int))), shape=(n, n))
        if connected_components(A, directed=False)[0] == 1:
            return list(kill)
    raise AssertionError("no safe outage set")


def pick(cons, branch=True, k=2):
    sel = [c for c in cons if (c[3] > 0 and c[4] == 0) == branch and (c[3] > 0 or c[4] > 0)]
    assert len(sel) >= k
    return sel[:k]


@need_data
def test_base_case_wrapper_is_identical_to_acpf(load_scenario, base):
    ref = acpf.base_case_voltages(load_scenario["bus"], load_scenario["branch"], load_scenario["gen"])
    assert np.array_equal(base.V, ref["V"]) and base.iterations == ref["iterations"]
    assert acpf.problem_from_raw is not None and "spy" not in acpf.problem_from_raw.__name__  # patch undone
    assert np.abs(acpf.mismatch(base.p, base.V)).max() < 1e-6


@need_data
def test_pattern_jacobian_slices_equal_acpf_jacobian(base):
    ch = fastpf.ChordContext(base)
    J = acpf.jacobian(base.p, base.V)
    assert abs(J - fastpf._jac_from_ds(ch.dSa, ch.dSm, base.p.pvpq, base.p.pq)).max() == 0.0
    assert ch.base_factors == 1


@pytest.mark.parametrize("branch", [True, False])
@need_data
def test_chord_and_fdlf_converge_to_datakit_solution(load_scenario, base, branch):
    cons = contingencies(load_scenario)
    chord, fd = fastpf.ChordContext(base), fastpf.FDLFContext(base)
    for _, p, br, nb, ng in pick(cons, branch, 3):
        V0 = base.start_for(p)
        out = fastpf.outaged_branches(base, br)
        assert len(out) == nb
        runs = {
            "chordC": chord.solve(p, V0, outaged=out),
            "fdlf": fastpf.fdlf_solve(fd, p, V0, out),
            "chord": chord.solve(p, V0),
        }
        for name, r in runs.items():
            if name == "chord" and not r.converged:  # plain chord may be too slow on a critical branch: it must say so
                assert r.reason == "max_it" and np.isnan(r.iters(TOL))
                continue
            assert r.converged and r.reason == "" and r.hist[-1] <= TOL, name
            assert np.abs(acpf.mismatch(p, r.V)).max() <= TOL
            assert np.abs(acpf.state_error(p, r.V)).max() < 1e-5, name  # vs datakit's stored solution
            assert r.iters(1e-2) <= r.iters(TOL) and r.solves_to(1e-2) <= r.solves_to(TOL)
        if not branch:  # generator outage: PV -> PQ changes the pattern; plain chord converges in a few iterations
            assert runs["chord"].converged and runs["chord"].iters(TOL) <= 6


@need_data
def test_genco_like_start_beats_base_start(load_scenario, base):
    """A start closer to the solution needs no more iterations (and strictly fewer from the exact solution)."""
    chord, fd = fastpf.ChordContext(base), fastpf.FDLFContext(base)
    for _, p, br, *_ in pick(contingencies(load_scenario), True, 2):
        out = fastpf.outaged_branches(base, br)
        near = p.V_from(np.abs(p.V_true), np.angle(p.V_true))
        for solver in (lambda V: chord.solve(p, V, outaged=out), lambda V: fastpf.fdlf_solve(fd, p, V, out)):
            assert solver(near).iters(TOL) < solver(base.start_for(p)).iters(TOL)


def _fresh_matrices(base, p_raw, br_table):
    Bp = fastpf.build_Bp(len(p_raw.Vm_set), br_table[br_table.br_status == 1], p_raw.pvpq, base.base_mva)
    Bpp = fastpf.build_Bpp(p_raw.Ybus, p_raw.pq)
    return Bp, Bpp


@need_data
def test_compensation_equals_fresh_factorization_of_outaged_matrices(load_scenario, base):
    """Sherman-Morrison-Woodbury solves on the BASE factors == a fresh solve of the outaged B' and B'' (also k > 1)."""
    fd = fastpf.FDLFContext(base)
    rng = np.random.default_rng(0)
    s0 = int(load_scenario["bus"].scenario.min())
    b, br, g = (load_scenario[t][load_scenario[t].scenario == s0] for t in ["bus", "branch", "gen"])
    for kill in (safe_outages(br, 1, 0), safe_outages(br, 3, 1), safe_outages(br, 5, 2)):
        brm = br.copy()
        brm.loc[brm.idx.isin(kill), "br_status"] = 0
        p = acpf.problem_from_raw(b, brm, g)
        out = fastpf.outaged_branches(base, brm)
        assert set(out.idx) >= set(kill)
        op = fd.prepare(p, out)
        Bp, Bpp = _fresh_matrices(base, p, brm)
        rp, rq = rng.normal(size=Bp.shape[0]), rng.normal(size=Bpp.shape[0])
        assert np.allclose(op.solve_p(rp), spla.spsolve(Bp, rp), rtol=1e-8, atol=1e-10)
        assert np.allclose(op.solve_q(rq), spla.spsolve(Bpp, rq), rtol=1e-8, atol=1e-10)
        assert op.new_factors == 0 and op.setup_solves >= len(out)


@need_data
def test_compensation_with_generator_outage_pattern_and_extra_branches(load_scenario, base):
    cons = contingencies(load_scenario)
    _, p0, br0, _, _ = pick(cons, False, 1)[0]
    g_out_scen = pick(cons, False, 1)[0][0]
    b, g = (load_scenario[t][load_scenario[t].scenario == g_out_scen] for t in ["bus", "gen"])
    brm = br0.copy()
    brm.loc[brm.idx.isin(safe_outages(brm, 2, 3)), "br_status"] = 0
    p = acpf.problem_from_raw(b, brm, g)
    assert len(p.pq) == len(base.p.pq) + 1
    fd = fastpf.FDLFContext(base)
    out = fastpf.outaged_branches(base, brm)
    op = fd.prepare(p, out)
    assert op.new_factors == 1  # the PQ pattern is new: one extra B'' factorization
    Bp, Bpp = _fresh_matrices(base, p, brm)
    rng = np.random.default_rng(1)
    rp, rq = rng.normal(size=Bp.shape[0]), rng.normal(size=Bpp.shape[0])
    # B' is stored in the base pvpq order; permute exactly as FDLFContext.solve does
    r0 = np.empty_like(rp)
    r0[op.perm_p] = rp
    assert np.allclose(op.solve_p(r0)[op.perm_p], spla.spsolve(Bp, rp), rtol=1e-8, atol=1e-10)
    assert np.allclose(op.solve_q(rq), spla.spsolve(Bpp, rq), rtol=1e-8, atol=1e-10)
    assert fd.prepare(p, out).new_factors == 0  # the pattern is cached for the rest of the load scenario


@need_data
def test_compensated_chord_jacobian_equals_fresh_jacobian_of_outaged_topology(load_scenario, base):
    chord = fastpf.ChordContext(base)
    rng = np.random.default_rng(2)
    s0 = int(load_scenario["bus"].scenario.min())
    b, br, g = (load_scenario[t][load_scenario[t].scenario == s0] for t in ["bus", "branch", "gen"])
    for kill in (safe_outages(br, 1, 4), safe_outages(br, 3, 5)):
        brm = br.copy()
        brm.loc[brm.idx.isin(kill), "br_status"] = 0
        p = acpf.problem_from_raw(b, brm, g)
        out = fastpf.outaged_branches(base, brm)
        pat, new_f, _ = chord._pattern(p)
        comp = chord._outage_correction(p, pat, out)
        J_fresh = acpf.jacobian(p, base.V).tocsc()  # outaged topology, evaluated at the base-case voltages
        r = rng.normal(size=J_fresh.shape[0])
        assert np.allclose(comp.solve(r), spla.spsolve(J_fresh, r), rtol=1e-7, atol=1e-9)
        assert new_f == 0


@need_data
def test_bordered_generator_outage_equals_refactorized_pattern(load_scenario, base):
    cons = contingencies(load_scenario)
    plain, bord = fastpf.ChordContext(base), fastpf.ChordContext(base, bordered=True)
    fd_plain, fd_bord = fastpf.FDLFContext(base), fastpf.FDLFContext(base, bordered=True)
    for _, p, br, *_ in pick(cons, False, 3):
        V0 = base.start_for(p)
        out = fastpf.outaged_branches(base, br)
        a, b2 = plain.solve(p, V0), bord.solve(p, V0)
        assert np.allclose(a.V, b2.V, atol=1e-12) and a.iters(TOL) == b2.iters(TOL)
        assert b2.new_factors == 0 and b2.setup_solves in (0, 1)  # one new PQ bus -> one setup solve (0 when cached)
        assert b2.solves == 2 * a.solves  # Schur complement: two base solves per iteration
        fa = fastpf.fdlf_solve(fd_plain, p, V0, out)
        fb = fastpf.fdlf_solve(fd_bord, p, V0, out)
        assert np.allclose(fa.V, fb.V, atol=1e-10) and fa.iters(TOL) == fb.iters(TOL) and fb.new_factors == 0
    assert plain.n_patterns_created >= 1 and bord.n_patterns_created == 0


@need_data
def test_branch_outages_never_create_factorizations_and_gen_outages_count_once_per_pattern(load_scenario, base):
    cons = contingencies(load_scenario)
    chord = fastpf.ChordContext(base)
    seen = set()
    for _, p, br, nb, ng in cons:
        r = chord.solve(p, base.start_for(p), outaged=fastpf.outaged_branches(base, br))
        key = fastpf.pattern_key(p.pv, p.pq)
        expect = 0 if (key in seen or key == fastpf.pattern_key(base.p.pv, base.p.pq)) else 1
        assert r.new_factors == expect
        seen.add(key)
        if ng == 0:
            assert key == fastpf.pattern_key(base.p.pv, base.p.pq)


@need_data
def test_islanding_outage_is_reported_as_singular_not_crashed(load_scenario, base):
    s0 = int(load_scenario["bus"].scenario.min())
    b, br, g = (load_scenario[t][load_scenario[t].scenario == s0] for t in ["bus", "branch", "gen"])
    # take out every branch of one PQ bus -> the bus is islanded, B' is singular
    bus = int(base.p.pq[3])
    brm = br.copy()
    brm.loc[(brm.from_bus == bus) | (brm.to_bus == bus), "br_status"] = 0
    p = acpf.problem_from_raw(b, brm, g)
    out = fastpf.outaged_branches(base, brm)
    r = fastpf.fdlf_solve(fastpf.FDLFContext(base), p, base.start_for(p), out)
    assert not r.converged and r.reason == "singular"
    rc = fastpf.ChordContext(base).solve(p, base.start_for(p), outaged=out)
    assert not rc.converged


@need_data
def test_plain_chord_rate_is_the_dc_self_ptdf_of_the_outaged_branch(load_scenario, base):
    """Why plain chord is slow: rho(I - J0^-1 Jc) ~ (baseMVA/x) u' B'^-1 u, the branch's share of its own flow, up to ~0.9."""
    fd = fastpf.FDLFContext(base)
    J0inv = np.linalg.inv(acpf.jacobian(base.p, base.V).toarray())
    Bpinv = np.linalg.inv(fd.Bp0.toarray())
    for _, p, br, *_ in pick(contingencies(load_scenario), True, 4):
        out = fastpf.as_outage(fastpf.outaged_branches(base, br))
        assert len(out) == 1
        Jc = acpf.jacobian(p, base.V).toarray()  # outaged topology at the base-case voltages
        rho = max(abs(np.linalg.eigvals(np.eye(len(Jc)) - J0inv @ Jc)))
        u = np.zeros(len(fd.pvpq0))
        for bus, sgn in ((out.f[0], 1.0), (out.t[0], -1.0)):
            if fd.pos_p[bus] >= 0:
                u[fd.pos_p[bus]] += sgn
        pred = base.base_mva / out.x[0] * (u @ Bpinv @ u)
        assert abs(rho - pred) < 0.05 and rho < 1


# ------------------------------------------------------------------ statistics helpers (scripts/analyze_exp02.py)

def test_clopper_pearson_known_values():
    lo, hi = an.clopper_pearson(0, 100)
    assert lo == 0.0 and abs(hi - (1 - 0.025 ** (1 / 100))) < 1e-9          # two-sided 95%
    assert abs(an.clopper_pearson(0, 30, two_sided=False)[1] - (1 - 0.05 ** (1 / 30))) < 1e-9  # one-sided 95% ("rule of three")
    from scipy.stats import binomtest

    ref = binomtest(5, 50).proportion_ci(confidence_level=0.95, method="exact")   # independent implementation
    lo, hi = an.clopper_pearson(5, 50)
    assert abs(lo - ref.low) < 1e-9 and abs(hi - ref.high) < 1e-9
    assert an.clopper_pearson(10, 10)[1] == 1.0


def test_cluster_bootstrap_resamples_clusters_not_rows():
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(40), 25)
    x = rng.normal(size=40)[g] + 0.01 * rng.normal(size=len(g))              # almost no within-cluster variation
    pt, lo, hi = an.cluster_bootstrap(x, groups=g, B=2000)
    assert abs(pt - x.mean()) < 1e-12 and lo < pt < hi
    se_cluster = (hi - lo) / 3.92
    se_naive = x.std() / np.sqrt(len(x))
    assert se_cluster > 3 * se_naive                                          # the row-level SE would be far too small
    # ratio estimator: sum(num) / sum(den)
    num, den = np.array([1.0, 2.0, 3.0, 4.0]), np.array([2.0, 2.0, 2.0, 2.0])
    assert an.cluster_bootstrap(num, den, groups=np.array([0, 0, 1, 1]))[0] == 10 / 8
    assert np.isnan(an.cluster_bootstrap([np.nan], groups=[0])[0])


def test_cluster_bootstrap_covers_the_mean():
    rng = np.random.default_rng(1)
    hit = 0
    for rep in range(200):
        g = np.repeat(np.arange(30), 5)
        x = rng.normal(size=30)[g] + rng.normal(size=len(g)) * 0.3
        _, lo, hi = an.cluster_bootstrap(x, groups=g, B=400, seed=rep)
        hit += lo <= 0.0 <= hi
    assert 0.88 <= hit / 200 <= 0.99


def test_ci_tags_follow_the_run_naming():
    assert an.ci_tags("base", 0)["n3"] == "stress_n3_base" and an.ci_tags("base", 42)["n2"] == "stress_n2_base_seed42"
    assert an.ci_tags("small", 0)["paper"] == "case118_ieee_small_seed0" and an.ci_tags("tiny", 0)["n5"] == "stress_n5_tiny"
