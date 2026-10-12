"""GENCO-Delta, residual variant: predict a correction to the pre-contingency state.

Plain GENCO-Delta (`PowerFlowContingencyAware`) puts the base-case Vm/Va at the masked input
positions and lets the released network read them as features. The released weights never used
those columns, so the network has to re-learn to exploit them and, after a few epochs, ends up
worse than the base case as a Newton start. The residual parameterisation removes that problem:

    bus_temp(layer i) = base + delta_head(h_bus(layer i))            (masked Vm, Va)

where `base` is the filled base-case Vm (p.u.) / Va (rad) taken from the model input
`x_dict["bus"][:, VM_H:VA_H+1]`, and `delta_head` is the released `mlp_bus` with its last linear
layer zero-initialised. At initialisation the decoded state IS the base case, bit for bit, so
Newton from the model costs exactly what Newton from the base case costs, and training only has to
learn a correction. Everything else in `GNS_heterogeneous.forward` is unchanged: the
`torch.where(bus_mask, bus_temp, bus_fixed)` that keeps known (unmasked) quantities, the generator
head, the physics decoder and the residual feedback.

Implementation. `delta_head` keeps the attribute name `mlp_bus` and the Sequential layout of the
released model, so the state-dict keys are identical to a plain GNS_heterogeneous: released
checkpoints load into the residual model with `strict=True` (but then the head is NOT zero: use
`load_released_into_residual` / `load_genco_residual`, which re-zero it), and fine-tuned residual
checkpoints reload with a plain `load_state_dict`. The base tensor is handed to the head by the
model's `forward` (set before, cleared after), it is not a parameter or buffer.

Use:
    model.type: GNS_heterogeneous_residual       (graphkit config)
    task.task_name: PowerFlowContingencyResidual (same transforms as PowerFlowContingencyAware)
    data.basecase_store / data.delta_inputs      (as for GENCO-Delta)
See scripts/finetune_residual.py and scripts/eval_genco_delta_residual.py.
"""

from __future__ import annotations

import copy

import torch
from torch import nn

from gridfm_graphkit.datasets.globals import VA_H, VM_H
from gridfm_graphkit.io.registries import MODELS_REGISTRY, PHYSICS_DECODER_REGISTRY, TASK_REGISTRY, TRANSFORM_REGISTRY
from gridfm_graphkit.models.gnn_heterogeneous_gns import GNS_heterogeneous
from gridfm_graphkit.models.utils import PhysicsDecoderPF
from gridfm_graphkit.tasks.pf_task import PowerFlowTask

from genco_certified.graphkit_plugin import PowerFlowContingencyAwareTransforms
from genco_certified.loading import load_genco

RESIDUAL_MODEL_TYPE = "GNS_heterogeneous_residual"
RESIDUAL_TASK_NAME = "PowerFlowContingencyResidual"
BUS_STATE = slice(VM_H, VA_H + 1)  # columns [Vm, Va] of the bus input; VM_OUT, VA_OUT = 0, 1 of the head output


class ResidualBusHead(nn.Sequential):
    """`mlp_bus` with a base-case offset: forward(h) = base + mlp(h), base of shape [num_bus, 2].

    A Sequential subclass on purpose: parameters keep the names `mlp_bus.0.*, 1.*, 3.*` of the released model.
    """

    def __init__(self, *layers):
        super().__init__(*layers)
        self._base = None

    def set_base(self, base: torch.Tensor | None) -> None:
        self._base = base

    def delta(self, h: torch.Tensor) -> torch.Tensor:
        """The correction alone (what the plain Sequential computes)."""
        return super().forward(h)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self._base is None:
            raise RuntimeError("ResidualBusHead has no base state: call it through GNS_heterogeneous_residual.forward")
        return self.delta(h) + self._base


def last_linear(seq: nn.Sequential) -> nn.Linear:
    lins = [m for m in seq if isinstance(m, nn.Linear)]
    if not lins:
        raise ValueError("no Linear layer in the head")
    return lins[-1]


def zero_init_delta_head(model: nn.Module) -> nn.Module:
    """Zero the last linear layer (weight and bias) of the residual head, in place: decoded state == base case."""
    head = model.mlp_bus
    if not isinstance(head, ResidualBusHead):
        raise TypeError(f"expected a ResidualBusHead, got {type(head).__name__}: not a residual GENCO model")
    lin = last_linear(head)
    with torch.no_grad():
        lin.weight.zero_()
        lin.bias.zero_()
    return model


class GNS_heterogeneous_residual(GNS_heterogeneous):
    """GENCO (`GNS_heterogeneous`) whose bus decode is base case + learned correction at every layer."""

    def __init__(self, args) -> None:
        super().__init__(args)
        self.mlp_bus = ResidualBusHead(*list(self.mlp_bus))  # same Linear/LayerNorm/LeakyReLU modules, same state-dict keys
        if getattr(args.model, "zero_init_head", True):
            zero_init_delta_head(self)

    @property
    def delta_head(self) -> ResidualBusHead:
        return self.mlp_bus

    def forward(self, batch, return_embeddings: bool = False):
        # Filled base-case Vm/Va at masked positions (known values at unmasked ones; those are overwritten
        # by `torch.where(bus_mask, bus_temp, bus_fixed)` in the parent forward anyway).
        self.mlp_bus.set_base(batch.x_dict["bus"][:, BUS_STATE])
        try:
            return super().forward(batch, return_embeddings)
        finally:
            self.mlp_bus.set_base(None)


# --------------------------------------------------------------------------- loading released / fine-tuned weights
def _strip_model_prefix(state: dict) -> dict:
    return {k[len("model."):]: v for k, v in state.items()} if state and all(k.startswith("model.") for k in state) else state


def load_released_into_residual(model: GNS_heterogeneous_residual, state: dict, zero_head: bool = True) -> tuple[list, list]:
    """Load a released GENCO state dict (plain GNS_heterogeneous keys, with or without the Lightning `model.` prefix)
    into the residual model with strict=False semantics, check nothing was left out, then re-zero the head.

    Returns (missing, unexpected), both empty on success. With `zero_head=False` (a fine-tuned residual
    checkpoint) the head is left as loaded.
    """
    res = model.load_state_dict(_strip_model_prefix(dict(state)), strict=False)
    if res.missing_keys or res.unexpected_keys:
        raise RuntimeError(f"state dict does not match the residual model: missing {res.missing_keys[:5]}, "
                           f"unexpected {res.unexpected_keys[:5]}")
    if zero_head:
        zero_init_delta_head(model)
    return list(res.missing_keys), list(res.unexpected_keys)


def load_genco_residual(task: str, grid: str, size: str = "base", seed: int = 0, *, overrides: dict | None = None,
                        weights=None, **kw):
    """`loading.load_genco` that also understands the residual task.

    If `overrides["task"]["task_name"]` is PowerFlowContingencyResidual the model type is set to
    GNS_heterogeneous_residual. With `weights=None` the RELEASED weights are loaded and the delta head is zeroed
    (the model starts at the base case); with `weights=<fine-tuned residual state dict>` they are loaded as they are.
    Any other task is passed through to `load_genco` unchanged.
    """
    over = copy.deepcopy(overrides or {})
    residual = (over.get("task") or {}).get("task_name") == RESIDUAL_TASK_NAME
    if residual:
        over.setdefault("model", {})["type"] = RESIDUAL_MODEL_TYPE
    Lg = load_genco(task, grid, size, seed, overrides=over, weights=weights, **kw)
    if residual and weights is None:
        zero_init_delta_head(Lg.task.model)
    return Lg


# --------------------------------------------------------------------------- registration
def _register(registry, name, obj):
    try:
        registry.register(name)(obj)
    except (KeyError, ValueError):  # already registered (module re-imported)
        pass


_register(MODELS_REGISTRY, RESIDUAL_MODEL_TYPE, GNS_heterogeneous_residual)
_register(TRANSFORM_REGISTRY, RESIDUAL_TASK_NAME, PowerFlowContingencyAwareTransforms)  # same PF mask + base-case fill
_register(TASK_REGISTRY, RESIDUAL_TASK_NAME, PowerFlowTask)
_register(PHYSICS_DECODER_REGISTRY, RESIDUAL_TASK_NAME, PhysicsDecoderPF)
