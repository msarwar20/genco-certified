"""Tests for the GENCO-Delta plugin (PowerFlowContingencyAware) and the base-case store.

Needs data/pf/case118_ieee (scripts/fetch_data.py) and the released Base normalizer
(checkpoints/genco-pf-datakit/case118_ieee/base/seed0/normalizer_stats.pt). No network.
"""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Batch

import genco_certified.graphkit_plugin as gp
from genco_certified import acpf
from genco_certified.basecase_store import BaseCaseStore, build_store
from gridfm_graphkit.datasets.globals import (
    ANG_MAX, ANG_MIN, B_ON, P_E, Q_E, RATE_A, REF_H, VA_H, VM_H, YFF_TT_I, YFF_TT_R,
)
from gridfm_graphkit.datasets.powergrid_hetero_dataset import HeteroGridDatasetDisk
from gridfm_graphkit.datasets.transforms import RemoveInactiveBranches
from gridfm_graphkit.io.param_handler import NestedNamespace, get_task_transforms, load_normalizer
from gridfm_graphkit.io.registries import PHYSICS_DECODER_REGISTRY, TASK_REGISTRY, TRANSFORM_REGISTRY

REPO = Path(__file__).resolve().parents[1]
GRID = REPO / "data" / "pf" / "case118_ieee"
STATS = REPO / "checkpoints" / "genco-pf-datakit" / "case118_ieee" / "base" / "seed0" / "normalizer_stats.pt"
pytestmark = pytest.mark.skipif(not (GRID / "processed").exists() or not STATS.exists(),
                                reason="needs data/pf/case118_ieee and the released Base normalizer")

EDGE = ("bus", "connects", "bus")
PROBE = [40, 199, 3000]  # 199 sits in a load scenario whose variants straddle the partition 0/1 boundary


def _args(task, store=None, mode=None):
    data = dict(baseMVA=100, mask_value=0.0, normalization="HeteroDataMVANormalizer")
    if store is not None:
        data["basecase_store"] = str(store)
    if mode is not None:
        data["delta_inputs"] = mode
    return NestedNamespace(data=data, task=dict(task_name=task))


@pytest.fixture(scope="module")
def store_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("store") / "basecase_store.pt"
    build_store(GRID / "raw", scenario_ids=PROBE, log=lambda *_: None).save(path)
    return path


@pytest.fixture(scope="module")
def store(store_path):
    return BaseCaseStore.load(store_path)


@pytest.fixture(scope="module")
def normalizer():
    n = load_normalizer(_args("PowerFlow"))
    n.fit_from_dict(torch.load(STATS, weights_only=True)["case118_ieee"])
    return n


@pytest.fixture(scope="module")
def datasets(store_path, normalizer):
    mk = lambda a: HeteroGridDatasetDisk(str(GRID), normalizer, transform=get_task_transforms(a))
    return {
        "raw": HeteroGridDatasetDisk(str(GRID), normalizer),
        "pf": mk(_args("PowerFlow")),
        "vm_va": mk(_args("PowerFlowContingencyAware", store_path, "vm_va")),
        "flows": mk(_args("PowerFlowContingencyAware", store_path, "vm_va_flows")),
    }


# --------------------------------------------------------------------------- registration / config
def test_task_is_registered_with_pf_task_and_decoder():
    from gridfm_graphkit.models.utils import PhysicsDecoderPF
    from gridfm_graphkit.tasks.pf_task import PowerFlowTask
    assert gp.DELTA_TASK_NAME == "PowerFlowContingencyAware"
    assert TASK_REGISTRY.get(gp.DELTA_TASK_NAME) is PowerFlowTask
    assert PHYSICS_DECODER_REGISTRY.get(gp.DELTA_TASK_NAME) is PhysicsDecoderPF
    assert TRANSFORM_REGISTRY.get(gp.DELTA_TASK_NAME) is gp.PowerFlowContingencyAwareTransforms
    # the slack-aware task is still registered, unchanged
    assert TRANSFORM_REGISTRY.get("PowerFlowSlackAware") is gp.PowerFlowSlackAwareTransforms


def test_config_errors_are_explicit(store_path):
    with pytest.raises(ValueError, match="basecase_store"):
        get_task_transforms(_args("PowerFlowContingencyAware"))
    with pytest.raises(ValueError, match="delta_inputs"):
        get_task_transforms(_args("PowerFlowContingencyAware", store_path, "everything"))
    a = _args("PowerFlowContingencyAware", store_path)
    a.data.mask_type = "rnd"
    with pytest.raises(ValueError, match="deterministic PF mask"):
        get_task_transforms(a)
    # default mode is the full variant
    assert get_task_transforms(_args("PowerFlowContingencyAware", store_path)).transforms[-1].mode == "vm_va_flows"


# --------------------------------------------------------------------------- masks and what changes
@pytest.mark.parametrize("mode", ["vm_va", "flows"])
@pytest.mark.parametrize("i", [40, 199, 3000])
def test_masks_identical_to_powerflow(datasets, mode, i):
    a, b = datasets["pf"][i], datasets[mode][i]
    assert set(a.mask_dict) == set(b.mask_dict)
    for k in a.mask_dict:
        assert torch.equal(a.mask_dict[k], b.mask_dict[k]), k
    # the graph itself is the same graph
    assert torch.equal(a[EDGE].edge_index, b[EDGE].edge_index)
    assert torch.equal(a["gen"].x, b["gen"].x)
    assert torch.equal(a["scenario_id"], b["scenario_id"])


@pytest.mark.parametrize("mode", ["vm_va", "flows"])
def test_only_masked_voltage_and_flow_values_change(datasets, mode):
    i = 40
    a, b = datasets["pf"][i], datasets[mode][i]
    xa, xb = a.x_dict["bus"], b.x_dict["bus"]
    mb = a.mask_dict["bus"]
    changed = xa != xb
    assert changed.any(0).nonzero().flatten().tolist() == [VM_H, VA_H]  # no other bus column moves
    assert not (changed & ~mb).any()  # nothing outside the mask moves
    assert (xb[:, [c for c in range(xb.size(1)) if c not in (VM_H, VA_H)]]
            == xa[:, [c for c in range(xb.size(1)) if c not in (VM_H, VA_H)]]).all()
    # every other masked entry is still the PF mask value (0)
    other = mb.clone(); other[:, [VM_H, VA_H]] = False
    assert (xb[other] == 0).all()
    ea, eb = a.edge_attr_dict[EDGE], b.edge_attr_dict[EDGE]
    if mode == "vm_va":
        assert torch.equal(ea, eb)
    else:
        assert (ea != eb).any(0).nonzero().flatten().tolist() == [P_E, Q_E]
        assert (eb[:, [ANG_MIN, ANG_MAX, RATE_A]] == 0).all()  # masked limits stay masked
    # gen features untouched
    assert torch.equal(a.x_dict["gen"], b.x_dict["gen"])


# --------------------------------------------------------------------------- values and units
def test_filled_bus_values_equal_the_base_case_in_model_units(datasets, store):
    """Va in radians, Vm in p.u., equal to an independently solved base case; checked against the raw parquet."""
    parts = sorted({q for i in PROBE for q in (i // 200 - 1, i // 200, i // 200 + 1) if q >= 0})
    raw = {t: pd.concat([pd.read_parquet(GRID / "raw" / f"{t}_data.parquet" / f"scenario_partition={q}") for q in parts])
           for t in ("bus", "branch", "gen")}
    for i in PROBE:
        d = datasets["flows"][i]
        ls = int(raw["bus"][raw["bus"].scenario == i].load_scenario_idx.iloc[0])
        sel = {t: df[df.load_scenario_idx == ls] for t, df in raw.items()}
        V = acpf.base_case_voltages(sel["bus"], sel["branch"], sel["gen"])["V"]
        x, m = d.x_dict["bus"], d.mask_dict["bus"]
        assert np.allclose(x[m[:, VM_H], VM_H].numpy(), np.abs(V)[m[:, VM_H].numpy()], atol=2e-6)
        assert np.allclose(x[m[:, VA_H], VA_H].numpy(), np.angle(V)[m[:, VA_H].numpy()], atol=2e-6)
        assert store.load_scenario(i) == ls
    # the unit of bus Va in x at transform time is radians: the (unmasked) REF angle and the label y are on the same scale
    d = datasets["raw"][40]
    ref = d.x_dict["bus"][:, REF_H] == 1
    assert torch.equal(d.x_dict["bus"][ref, VA_H], d.y_dict["bus"][ref, VA_H])
    va_deg = pd.read_parquet(GRID / "raw" / "bus_data.parquet" / "scenario_partition=0", columns=["scenario", "bus", "Va"])
    va_deg = va_deg[va_deg.scenario == 40].sort_values("bus").Va.to_numpy()
    assert np.allclose(d.y_dict["bus"][:, VA_H].numpy(), np.deg2rad(va_deg), atol=1e-6)


def test_every_variant_of_a_load_scenario_gets_the_same_base_case(datasets, store):
    ls = store.load_scenario(199)
    sibs = [s for s in range(150, 260) if int(store.load_of[s]) == ls]
    assert len(sibs) >= 16 and 199 in sibs
    assert min(sibs) < 200 <= max(sibs)  # variants straddle the partition boundary and were merged before solving
    ref = datasets["flows"][sibs[0]].x_dict["bus"][:, [VM_H, VA_H]]
    for s in sibs[1:4]:
        d = datasets["flows"][s]
        m = datasets["pf"][s].mask_dict["bus"][:, [VM_H, VA_H]]
        assert torch.equal(d.x_dict["bus"][:, [VM_H, VA_H]][m], ref[m])


def test_edge_flow_formula_reproduces_datakit_flows_in_normalised_units(datasets, normalizer):
    """S_ft = V_f conj(Yff V_f + Yft V_t) on normalised Y and p.u. V equals datakit's stored pf/qf divided by baseMVA."""
    for i in PROBE:
        d = RemoveInactiveBranches()(datasets["raw"][i])
        ea, y = d[EDGE].edge_attr, d["bus"].y
        S = gp.sending_end_flows(y[:, VM_H], y[:, VA_H], d[EDGE].edge_index, ea)
        assert torch.allclose(S.real.float(), ea[:, P_E], rtol=2e-4, atol=2e-4)
        assert torch.allclose(S.imag.float(), ea[:, Q_E], rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize("i", PROBE)
def test_filled_edge_flows_match_acpf_branch_flows_on_the_base_state(datasets, store, normalizer, i):
    d = datasets["flows"][i]
    b = Batch.from_data_list([d])
    Vm, Va = store.lookup(i)
    normalizer.inverse_transform(b)  # MW / MVAr
    ea = b.edge_attr_dict[EDGE]
    (p,) = acpf.extract_problems(b)
    S = acpf.branch_flows(p, Vm.double().numpy() * np.exp(1j * Va.double().numpy()))
    assert np.allclose(ea[:, P_E].double().numpy(), S.real, rtol=1e-4, atol=2e-3)
    assert np.allclose(ea[:, Q_E].double().numpy(), S.imag, rtol=1e-4, atol=2e-3)
    # sanity: a real flow pattern, not zeros, and different from the post-contingency labels
    assert np.abs(S).max() > 10
    y_p = b.y_dict[EDGE][:, 0].double().numpy()
    assert np.abs(ea[:, P_E].double().numpy() - y_p).max() > 1e-3


# --------------------------------------------------------------------------- no leakage of labels
def test_no_post_contingency_label_reaches_the_inputs(store_path, datasets):
    t = get_task_transforms(_args("PowerFlowContingencyAware", store_path, "vm_va_flows"))
    raw = datasets["raw"].get(40)  # normalised, untransformed sample
    g = torch.Generator().manual_seed(0)
    garbled = copy.deepcopy(raw)
    garbled["bus"].y = torch.randn(garbled["bus"].y.shape, generator=g) * 7
    garbled["gen"].y = torch.randn(garbled["gen"].y.shape, generator=g) * 7
    garbled[EDGE].y = torch.randn(garbled[EDGE].y.shape, generator=g) * 7
    a, b = t(copy.deepcopy(raw)), t(garbled)
    assert torch.equal(a.x_dict["bus"], b.x_dict["bus"])
    assert torch.equal(a.x_dict["gen"], b.x_dict["gen"])
    assert torch.equal(a.edge_attr_dict[EDGE], b.edge_attr_dict[EDGE])
    # and the filled voltages are not the labels (the base case is the PRE-contingency state)
    m = a.mask_dict["bus"][:, VA_H]
    assert (a.x_dict["bus"][m, VA_H] - raw["bus"].y[m, VA_H]).abs().max() > 1e-4


def test_store_does_not_depend_on_which_variant_asks(store):
    ls = store.load_scenario(199)
    rows = {int(store.row_of[s]) for s in range(len(store.row_of)) if int(store.load_of[s]) == ls}
    assert len(rows) == 1  # one base case per load scenario, shared by all its contingencies


# --------------------------------------------------------------------------- store
def test_store_roundtrip_and_errors(store, tmp_path):
    p = store.save(tmp_path / "s.pt")
    s2 = BaseCaseStore.load(p)
    assert torch.equal(s2.Vm, store.Vm) and torch.equal(s2.row_of, store.row_of) and s2.meta == store.meta
    assert store.covers([40, 199]) and not store.covers([40, 5])
    assert store.missing([5, 40, 10**6, -1]) == [5, 10**6, -1]
    with pytest.raises(KeyError, match="not in the base-case store"):
        store.lookup(5)
    assert store.meta["n_nonconverged"] == 0 and store.n_bus == 118
    assert store.Vm.dtype == torch.float32 and store.Va.dtype == torch.float32


def test_full_store_covers_every_split_id_when_built():
    path = GRID / "basecase_store.pt"
    if not path.exists():
        pytest.skip("run scripts/build_basecase_store.py")
    s = BaseCaseStore.load(path)
    for name in ("train", "val", "test"):
        ids = torch.load(GRID / "splits_paper_test" / f"{name}.pt", weights_only=True).tolist()
        assert s.covers(ids), name


# --------------------------------------------------------------------------- base-case solve
def test_solve_base_case_equals_acpf_when_the_flat_start_converges():
    from genco_certified.basecase_store import solve_base_case
    raw = {t: pd.read_parquet(GRID / "raw" / f"{t}_data.parquet" / "scenario_partition=3") for t in ("bus", "branch", "gen")}
    ls = int(raw["bus"].load_scenario_idx.iloc[len(raw["bus"]) // 2])
    sel = {t: d[d.load_scenario_idx == ls] for t, d in raw.items()}
    ref = acpf.base_case_voltages(sel["bus"], sel["branch"], sel["gen"])
    got = solve_base_case(sel["bus"], sel["branch"], sel["gen"])
    assert ref["converged"] and got["converged"] and got["start"] == 0
    assert np.allclose(got["V"], ref["V"], atol=0, rtol=0)  # same flat-start solve, bit for bit
    # the base case solves the intact network: zero mismatch on it
    Vn = np.abs(got["V"]); assert 0.9 < Vn.min() and Vn.max() < 1.15


GOC = REPO / "data" / "pf" / "case2000_goc" / "raw"


@pytest.mark.skipif(not (GOC / "bus_data.parquet" / "scenario_partition=11").exists(), reason="needs 12+ GOC 2000 partitions")
def test_retry_recovers_goc2000_load_scenarios_where_the_flat_start_diverges():
    """Load scenario 96 of GOC 2000 diverges from a flat start although every variant is solvable."""
    from genco_certified.basecase_store import solve_base_case
    parts = range(8, 12)
    raw = {t: pd.concat([pd.read_parquet(GOC / f"{t}_data.parquet" / f"scenario_partition={q}") for q in parts])
           for t in ("bus", "branch", "gen")}
    sel = {t: d[d.load_scenario_idx == 96] for t, d in raw.items()}
    if sel["bus"].empty:
        pytest.skip("load scenario 96 not in the loaded partitions")
    if acpf.base_case_voltages(sel["bus"], sel["branch"], sel["gen"])["converged"]:
        pytest.skip("flat start converges on this data")
    got = solve_base_case(sel["bus"], sel["branch"], sel["gen"])
    assert got["converged"] and got["start"] in (1, 2)


# --------------------------------------------------------------------------- evaluation script
def _eval_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("eval_genco_delta", REPO / "scripts" / "eval_genco_delta.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_eval_analysis_is_exact_when_the_prediction_is_the_truth(datasets, store, normalizer):
    from gridfm_graphkit.datasets.masking import RemovePFMask
    E = _eval_module()
    b = Batch.from_data_list([datasets["pf"][40]])
    normalizer.inverse_transform(b)
    RemovePFMask()(b)
    (p,) = acpf.extract_problems(b)
    Vm_b, Va_b = store.lookup(40)
    V_base = Vm_b.double().numpy() * np.exp(1j * Va_b.double().numpy())
    truth = np.c_[np.abs(p.V_true), np.angle(p.V_true)]
    row = E.analyse(p, truth, V_base, True, tau=1.01, eps_v=1e-3)
    # (graphkit's float32 labels leave a ~2e-3 MW mismatch floor: converged at 1e-2 MW with no step, at 1e-6 MW after one)
    assert row["err_l2_g"] < 1e-9 and row["nri_g_e2"] == 0 and row["nri_g_e6"] <= 1 and row["nr_conv_g"]
    assert row["g0_fnb"] == row["g0_fpb"] == row["g0_fnv"] == row["g0_fpv"] == 0  # no decision error from the truth itself
    assert row["g0_tpb"] == row["truth_b"] and row["g0_tpv"] == row["truth_v"]
    assert row["err_l2_b"] > 1e-3 and row["nri_b_e6"] >= 1  # the base case is a different state than the post-contingency truth
    assert row["nri_f_e6"] > row["nri_b_e6"] >= 1 or row["nri_f_e6"] >= 3  # flat start is the slowest start


def test_cluster_bootstrap_resamples_load_scenarios_together():
    E = _eval_module()
    df = pd.DataFrame({"load_idx": np.repeat(np.arange(10), 5), "x": np.repeat(np.arange(10.0), 5)})
    bt = E.Boot(df, n_boot=500, seed=1)
    m = bt.mean(df.x)
    assert abs(m["mean"] - 4.5) < 1e-12 and m["ci95"][0] < 4.5 < m["ci95"][1]
    # clusters are indivisible: every bootstrap mean is a mean of whole-cluster values, so its spread equals the
    # between-cluster spread (an iid-per-row bootstrap would give a CI about sqrt(5) narrower)
    assert m["ci95"][1] - m["ci95"][0] > 1.8
    r = bt.ratio(df.x, np.ones(len(df)))
    assert abs(r["value"] - 4.5) < 1e-12 and r["num"] == df.x.sum()
