"""Capture GENCO's intermediate state at every correction layer via forward hooks.

GENCO (`GNS_heterogeneous`) decodes a full bus state [Vm, Va, Pg, Qg] after each of
its message-passing layers by calling `model.physics_decoder`, then computes nodal
power-balance residuals with `model.node_residuals_layer` and feeds them back. We
hook both modules instead of re-implementing the forward pass, so the trace is
exactly what the model computes.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch


@contextmanager
def trace_layers(model, capture_embeddings: bool = False):
    """Yield a dict that fills with per-layer tensors during `model(batch)`.

    Keys:
      "state":     list[L] of [N_bus, 4] decoded states (normalized units: Vm p.u.,
                   Va rad, Pg/Qg in units of the normalizer's baseMVA)
      "residual":  list[L] of [N_bus, 2] (P, Q) power-balance residuals (normalized)
      "embedding": list[L] of [N_bus, H*heads] bus latents fed to the decoder head
    """
    trace = {"state": [], "residual": [], "embedding": []}

    def on_decoder(_m, _inp, out):
        trace["state"].append(out.detach().clone())

    def on_residuals(_m, _inp, out):
        res_p, res_q = out
        trace["residual"].append(torch.stack([res_p, res_q], dim=-1).detach().clone())

    def on_head(_m, inp):
        trace["embedding"].append(inp[0].detach().clone())

    handles = [
        model.physics_decoder.register_forward_hook(on_decoder),
        model.node_residuals_layer.register_forward_hook(on_residuals),
    ]
    if capture_embeddings:
        handles.append(model.mlp_bus.register_forward_pre_hook(on_head))
    try:
        yield trace
    finally:
        for h in handles:
            h.remove()
