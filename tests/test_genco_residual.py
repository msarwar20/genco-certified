"""Tests for the residual GENCO-Delta model (src/genco_certified/genco_residual.py).

Needs data/pf/case118_ieee (scripts/fetch_data.py) and the released Base seed-0 checkpoint files
(checkpoints/genco-pf-datakit/case118_ieee/base/seed0/{normalizer_stats.pt,best_model_state_dict.pt,*.yaml}). No network.
"""

import copy
from pathlib import Path

import pytest
import torch
import yaml
from torch_geometric.data import Batch

import genco_certified.genco_residual as gr
from genco_certified.basecase_store import BaseCaseStore, build_store
from gridfm_graphkit.datasets.globals import VA_H, VM_H
from gridfm_graphkit.datasets.powergrid_hetero_dataset import HeteroGridDatasetDisk
from gridfm_graphkit.io.param_handler import NestedNamespace, get_task_transforms, load_model, load_normalizer
from gridfm_graphkit.io.registries import MODELS_REGISTRY, PHYSICS_DECODER_REGISTRY, TASK_REGISTRY, TRANSFORM_REGISTRY
from gridfm_graphkit.models.gnn_heterogeneous_gns import GNS_heterogeneous

REPO = Path(__file__).resolve().parents[1]
GRID = REPO / "data" / "pf" / "case118_ieee"
CKPT = REPO / "checkpoints" / "genco-pf-datakit" / "case118_ieee" / "base" / "seed0"
STATS, WEIGHTS, CFG = CKPT / "normalizer_stats.pt", CKPT / "best_model_state_dict.pt", CKPT / "case118_ieee_base_seed0.yaml"
pytestmark = pytest.mark.skipif(
    not ((GRID / "processed").exists() and STATS.exists() and WEIGHTS.exists() and CFG.exists()),
    reason="needs data/pf/case118_ieee and the released Base seed-0 checkpoint",
)

PROBE = [40, 199, 3000]
BASE_COLS = slice(VM_H, VA_H + 1)


def _config(task=gr.RESIDUAL_TASK_NAME, model_type=gr.RESIDUAL_MODEL_TYPE, store=None, mode="vm_va_flows"):
    cfg = yaml.safe_load(CFG.read_text())
    cfg["task"]["task_name"] = task
    cfg["model"]["type"] = model_type
    if store is not None:
        cfg["data"].update(basecase_store=str(store), delta_inputs=mode)
    return NestedNamespace(**cfg)


@pytest.fixture(scope="module")
def store_path(tmp_path_factory):
    real = GRID / "basecase_store.pt"
    if real.is_file():
        return real
    path = tmp_path_factory.mktemp("store") / "basecase_store.pt"
    build_store(GRID / "raw", scenario_ids=PROBE, log=lambda *_: None).save(path)
    return path


@pytest.fixture(scope="module")
def released():
    return torch.load(WEIGHTS, map_location="cpu", weights_only=False)


@pytest.fixture(scope="module")
def batch(store_path):
    """Three real scenarios through the residual task's transforms (PF mask + base-case fill), batched."""
    norm = load_normalizer(_config())
    norm.fit_from_dict(torch.load(STATS, weights_only=True)["case118_ieee"])
    ds = HeteroGridDatasetDisk(str(GRID), norm, transform=get_task_transforms(_config(store=store_path)))
    return Batch.from_data_list([ds[i] for i in PROBE])


def _model(released=None, zero_head=True):
    m = load_model(_config())
    if released is not None:
        gr.load_released_into_residual(m, released, zero_head=zero_head)
    return m.eval()


def _masked_state(out, batch):
    return out["bus"][:, :2], batch.mask_dict["bus"][:, BASE_COLS]


# --------------------------------------------------------------------------- registration / structure
def test_registration_and_state_dict_layout(released):
    assert MODELS_REGISTRY.get(gr.RESIDUAL_MODEL_TYPE) is gr.GNS_heterogeneous_residual
    assert MODELS_REGISTRY.get("GNS_heterogeneous") is GNS_heterogeneous  # the plain model is untouched
    assert issubclass(gr.GNS_heterogeneous_residual, GNS_heterogeneous)
    from gridfm_graphkit.models.utils import PhysicsDecoderPF
    from gridfm_graphkit.tasks.pf_task import PowerFlowTask
    import genco_certified.graphkit_plugin as gp
    assert TRANSFORM_REGISTRY.get(gr.RESIDUAL_TASK_NAME) is gp.PowerFlowContingencyAwareTransforms
    assert TASK_REGISTRY.get(gr.RESIDUAL_TASK_NAME) is PowerFlowTask
    assert PHYSICS_DECODER_REGISTRY.get(gr.RESIDUAL_TASK_NAME) is PhysicsDecoderPF
    assert TRANSFORM_REGISTRY.get("PowerFlowContingencyAware") is gp.PowerFlowContingencyAwareTransforms  # still there
    # same parameter names and shapes as the plain model, so released and fine-tuned checkpoints load with strict=True
    plain = load_model(_config(task="PowerFlow", model_type="GNS_heterogeneous"))
    res = load_model(_config())
    assert {k: v.shape for k, v in plain.state_dict().items()} == {k: v.shape for k, v in res.state_dict().items()}
    assert isinstance(res.mlp_bus, gr.ResidualBusHead) and res.delta_head is res.mlp_bus
    assert set(released) == {"model." + k for k in res.state_dict()}


def test_loading_released_weights_zeroes_only_the_last_head_layer(released):
    m = _model(released)
    lin = gr.last_linear(m.delta_head)
    assert lin.weight.abs().max() == 0 and lin.bias.abs().max() == 0
    plain = load_model(_config(task="PowerFlow", model_type="GNS_heterogeneous"))
    plain.load_state_dict({k[len("model."):]: v for k, v in released.items()})
    sd_r, sd_p = m.state_dict(), plain.state_dict()
    for k, v in sd_p.items():
        if k in ("mlp_bus.3.weight", "mlp_bus.3.bias"):
            assert sd_r[k].abs().max() == 0 and v.abs().max() > 0
        else:
            assert torch.equal(sd_r[k], v), k  # everything else is the released weight, bit for bit
    # a mismatching state dict is an error, not a silent partial load
    bad = {k: v for k, v in released.items() if "mlp_gen" not in k}
    with pytest.raises(RuntimeError, match="missing"):
        gr.load_released_into_residual(load_model(_config()), bad)


# --------------------------------------------------------------------------- (1) initialisation == base case
def test_at_init_masked_vm_va_equal_the_base_case_exactly(batch, released, store_path):
    m = _model(released)
    with torch.no_grad():
        out = m(batch)
    pred, mask = _masked_state(out, batch)
    base_in = batch.x_dict["bus"][:, BASE_COLS]
    assert mask.any() and mask[:, 0].any() and mask[:, 1].any()
    assert torch.equal(pred[mask], base_in[mask])  # bit for bit, all 12 layers' decode collapses to the base case
    # and that input really is the base case of the store
    store = BaseCaseStore.load(store_path)
    ptr = batch["bus"].ptr
    for g, sid in enumerate(batch["scenario_id"].tolist()):
        Vm, Va = store.lookup(int(sid))
        sl = slice(int(ptr[g]), int(ptr[g + 1]))
        mg = mask[sl]
        assert torch.equal(pred[sl][:, 0][mg[:, 0]], Vm[mg[:, 0]])
        assert torch.equal(pred[sl][:, 1][mg[:, 1]], Va[mg[:, 1]])
    # the plain released model is NOT at the base case on the same inputs (the test has teeth)
    plain = load_model(_config(task="PowerFlow", model_type="GNS_heterogeneous"))
    plain.load_state_dict({k[len("model."):]: v for k, v in released.items()})
    with torch.no_grad():
        p_out = plain.eval()(batch)
    assert (p_out["bus"][:, :2][mask] - base_in[mask]).abs().max() > 1e-3


def test_at_init_is_the_base_case_for_any_backbone_weights(batch):
    torch.manual_seed(0)
    m = _model()  # random backbone, zero_init_head default
    with torch.no_grad():
        for p in m.parameters():
            if p is not gr.last_linear(m.delta_head).weight and p is not gr.last_linear(m.delta_head).bias:
                p.add_(0.1 * torch.randn_like(p))
        out = m(batch)
    pred, mask = _masked_state(out, batch)
    assert torch.equal(pred[mask], batch.x_dict["bus"][:, BASE_COLS][mask])


def test_residual_decode_is_base_plus_delta_of_the_last_layer(batch, released):
    torch.manual_seed(1)
    m = _model(released)
    lin = gr.last_linear(m.delta_head)
    with torch.no_grad():
        lin.weight.normal_(0, 0.05)
        lin.bias.normal_(0, 0.01)
        out, emb = m(batch, return_embeddings=True)
        delta = m.delta_head.delta(emb["bus"])  # head input of the last layer
    pred, mask = _masked_state(out, batch)
    base_in = batch.x_dict["bus"][:, BASE_COLS]
    assert torch.allclose(pred[mask], (base_in + delta)[mask], atol=1e-6)
    assert (pred[mask] - base_in[mask]).abs().max() > 1e-4  # a nonzero head does move the state


# --------------------------------------------------------------------------- (4) unmasked entries unchanged
def test_unmasked_entries_are_unchanged_even_with_a_large_head(batch, released):
    torch.manual_seed(2)
    m = _model(released)
    base_in = batch.x_dict["bus"][:, BASE_COLS]
    mask = batch.mask_dict["bus"][:, BASE_COLS]
    assert (~mask).any()
    for scale in (0.0, 5.0):
        with torch.no_grad():
            lin = gr.last_linear(m.delta_head)
            lin.weight.normal_(0, scale + 1e-12)
            lin.bias.normal_(0, scale + 1e-12)
            out = m(batch)
        pred = out["bus"][:, :2]
        assert torch.equal(pred[~mask], base_in[~mask])  # known quantities (PV Vm, REF Va, ...) pass straight through
    # the input itself was not modified by the forward pass (the base tensor is a view of the input)
    assert torch.equal(batch.x_dict["bus"][:, BASE_COLS], base_in)


def test_base_is_cleared_after_forward_and_head_needs_a_base(batch, released):
    m = _model(released)
    with torch.no_grad():
        m(batch)
    assert m.delta_head._base is None
    with pytest.raises(RuntimeError, match="no base state"):
        m.delta_head(torch.zeros(3, m.hidden_dim * m.heads))


# --------------------------------------------------------------------------- (2) gradients
def _vm_va_loss(model, batch):
    out = model(batch)
    mask = batch.mask_dict["bus"][:, BASE_COLS]
    return torch.nn.functional.mse_loss(out["bus"][:, :2][mask], batch.y_dict["bus"][:, BASE_COLS][mask])


def test_gradients_flow_to_the_delta_head(batch, released):
    m = _model(released).train()
    lin = gr.last_linear(m.delta_head)
    _vm_va_loss(m, batch).backward()
    # at the zero initialisation the loss reaches the last head layer ...
    assert lin.weight.grad is not None and lin.weight.grad.abs().max() > 0
    assert lin.bias.grad is not None and lin.bias.grad.abs().max() > 0
    # ... and (W = 0) nothing upstream of it through the head
    assert m.delta_head[0].weight.grad.abs().max() == 0
    assert m.input_proj_bus[0].weight.grad.abs().max() == 0
    # one small step on the last layer opens the rest of the head and the backbone
    with torch.no_grad():
        lin.weight.add_(1e-2 * lin.weight.grad.sign())
    m.zero_grad()
    _vm_va_loss(m, batch).backward()
    assert m.delta_head[0].weight.grad.abs().max() > 0
    assert m.layers[0].convs[("bus", "connects", "bus")].lin_query.weight.grad.abs().max() > 0
    assert m.input_proj_bus[0].weight.grad.abs().max() > 0


# --------------------------------------------------------------------------- (3) save / reload round trip
def _finetune_a_bit(model, batch, steps=4):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(steps):
        opt.zero_grad()
        _vm_va_loss(model, batch).backward()
        opt.step()
    return model.eval()


def test_finetuned_state_dict_roundtrip_gives_identical_outputs(batch, released, tmp_path):
    m = _finetune_a_bit(_model(released), batch)
    lin = gr.last_linear(m.delta_head)
    assert lin.weight.abs().max() > 0  # the head moved away from zero
    with torch.no_grad():
        ref = m(batch)
    path = tmp_path / "best_model_state_dict.pt"
    torch.save({"model." + k: v for k, v in m.state_dict().items()}, path)  # Lightning-task layout, like finetune.py saves

    # (a) fresh model, state dict loaded as-is (the head must NOT be re-zeroed)
    m2 = _model()
    gr.load_released_into_residual(m2, torch.load(path, weights_only=False), zero_head=False)
    with torch.no_grad():
        out = m2(batch)
    for k in ("bus", "gen"):
        assert torch.equal(ref[k], out[k]), k
    # (b) same, plain load_state_dict (identical keys)
    m3 = _model()
    m3.load_state_dict({k[len("model."):]: v for k, v in torch.load(path, weights_only=False).items()})
    with torch.no_grad():
        out3 = m3.eval()(batch)
    assert torch.equal(ref["bus"], out3["bus"])
    # (c) re-zeroing (the released-weights path) would NOT reproduce it: the two code paths are really different
    m4 = _model()
    gr.load_released_into_residual(m4, torch.load(path, weights_only=False), zero_head=True)
    with torch.no_grad():
        out4 = m4(batch)
    mask = batch.mask_dict["bus"][:, BASE_COLS]
    assert torch.equal(out4["bus"][:, :2][mask], batch.x_dict["bus"][:, BASE_COLS][mask])
    assert not torch.equal(ref["bus"][:, :2][mask], out4["bus"][:, :2][mask])


@pytest.fixture(scope="module")
def loaded_init(store_path):
    """End to end through the project loader (config, normalizer, dataset, Lightning task) on the paper test split."""
    try:
        return gr.load_genco_residual(
            "pf", "case118_ieee", "base", 0, batch_size=4, data_root=REPO / "data" / "pf",
            overrides={"task": {"task_name": gr.RESIDUAL_TASK_NAME},
                       "data": {"basecase_store": str(store_path), "delta_inputs": "vm_va_flows"}})
    except Exception as e:  # e.g. released config / stats not obtainable offline
        pytest.skip(f"loader unavailable here: {type(e).__name__}: {e}")


def test_loader_builds_the_residual_model_at_the_base_case(loaded_init):
    Lg = loaded_init
    assert isinstance(Lg.task.model, gr.GNS_heterogeneous_residual)
    assert Lg.config.model.type == gr.RESIDUAL_MODEL_TYPE and Lg.config.task.task_name == gr.RESIDUAL_TASK_NAME
    batch = next(iter(Lg.test_loader()))
    with torch.no_grad():
        out = Lg.task.model.eval()(batch)
    pred, mask = _masked_state(out, batch)
    assert torch.equal(pred[mask], batch.x_dict["bus"][:, BASE_COLS][mask])


def test_loader_reloads_a_finetuned_checkpoint_identically(loaded_init, store_path, tmp_path):
    Lg = loaded_init
    batch = next(iter(Lg.test_loader()))
    model = copy.deepcopy(Lg.task.model)
    _finetune_a_bit(model, batch, steps=3)
    path = tmp_path / "best_model_state_dict.pt"
    torch.save({"model." + k: v for k, v in model.state_dict().items()}, path)
    Lg2 = gr.load_genco_residual(
        "pf", "case118_ieee", "base", 0, batch_size=4, data_root=REPO / "data" / "pf", weights=str(path),
        overrides={"task": {"task_name": gr.RESIDUAL_TASK_NAME},
                   "data": {"basecase_store": str(store_path), "delta_inputs": "vm_va_flows"}})
    with torch.no_grad():
        a, b = model.eval()(batch), Lg2.task.model.eval()(batch)
    assert torch.equal(a["bus"], b["bus"]) and torch.equal(a["gen"], b["gen"])
    assert gr.last_linear(Lg2.task.model.delta_head).weight.abs().max() > 0  # not re-zeroed when weights are given
