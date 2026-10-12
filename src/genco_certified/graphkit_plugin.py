"""gridfm-graphkit plugin: slack-aware power flow.

Graphkit's PF mask (`AddPFHeteroMask`) treats the slack (REF) bus voltage magnitude
as an unknown. In the physical AC power-flow problem it is a known setpoint, so a
GENCO prediction can be power-balanced at the *wrong* slack voltage. This plugin
registers a `PowerFlowSlackAware` task that is identical to `PowerFlow` except that
REF-bus Vm is passed to the model as a known input.

Use with the graphkit CLI:
    gridfm_graphkit finetune ... --plugins genco_certified.graphkit_plugin
and set `task.task_name: PowerFlowSlackAware` in the YAML.
"""

import torch
from torch_geometric.transforms import BaseTransform, Compose

from gridfm_graphkit.datasets.globals import P_E, Q_E, VA_H, VM_H, YFF_TT_I, YFF_TT_R, YFT_TF_I, YFT_TF_R
from gridfm_graphkit.datasets.masking import AddPFHeteroMask, AddRandomHeteroMask
from gridfm_graphkit.datasets.transforms import ApplyMasking, RemoveInactiveBranches, RemoveInactiveGenerators
from gridfm_graphkit.io.registries import PHYSICS_DECODER_REGISTRY, TASK_REGISTRY, TRANSFORM_REGISTRY
from gridfm_graphkit.models.utils import PhysicsDecoderPF
from gridfm_graphkit.tasks.pf_task import PowerFlowTask

from genco_certified.basecase_store import load_store_cached

TASK_NAME = "PowerFlowSlackAware"


class AddPFSlackAwareMask(AddPFHeteroMask):
    """PF mask with the slack-bus voltage magnitude known (unmasked)."""

    def forward(self, data):
        data = super().forward(data)
        data.mask_dict["bus"][data.mask_dict["REF"], VM_H] = False
        return data


class PowerFlowSlackAwareTransforms(Compose):
    def __init__(self, args):
        mask_type = getattr(args.data, "mask_type", None)
        mask = AddRandomHeteroMask(mask_ratio=args.data.mask_ratio) if mask_type == "rnd" else AddPFSlackAwareMask()
        super().__init__([RemoveInactiveBranches(), RemoveInactiveGenerators(), mask, ApplyMasking(args=args)])


def _register(registry, name, obj):
    try:
        registry.register(name)(obj)
    except (KeyError, ValueError):  # already registered (module re-imported)
        pass


_register(TRANSFORM_REGISTRY, TASK_NAME, PowerFlowSlackAwareTransforms)
_register(TASK_REGISTRY, TASK_NAME, PowerFlowTask)
_register(PHYSICS_DECODER_REGISTRY, TASK_NAME, PhysicsDecoderPF)


# ---------------------------------------------------------------------------
# GENCO-Delta: PowerFlowContingencyAware
# ---------------------------------------------------------------------------
# At contingency time the operator already knows the solved pre-contingency state. The PF
# task zeroes every masked input; this task fills the *values* at the masked positions with
# base-case information and leaves every mask boolean exactly as in PowerFlow, so the model
# still has to predict the same unknowns:
#   bus  VM_H, VA_H  (masked at PQ/REF and PQ/PV)  <- base-case Vm [p.u.], Va [rad]
#   edge P_E, Q_E    (masked on every branch)      <- pre-contingency sending-end flow of THIS
#                                                     edge, from the base-case voltages and this
#                                                     edge's own Yff/Yft   (--delta_inputs vm_va_flows)
# Units: graphkit normalises (HeteroGridDatasetDisk.get -> data_normalizer.transform) *before*
# the task transforms run, so at fill time bus Va is already in radians, Vm in p.u. and edge
# admittances are scaled by baseMVA_orig/baseMVA. A sending-end flow computed from p.u. voltages
# and those scaled admittances is S/baseMVA, which is exactly how P_E/Q_E are normalised, i.e.
# the same quantity the model's own ComputeBranchFlow produces from normalised inputs. No
# post-contingency label (y) is read.
#
# Config (YAML `data:` section): basecase_store: <path to basecase_store.pt>
#                                delta_inputs:   vm_va | vm_va_flows   (default vm_va_flows)

DELTA_TASK_NAME = "PowerFlowContingencyAware"
DELTA_MODES = ("vm_va", "vm_va_flows")
_BRANCH = ("bus", "connects", "bus")


def sending_end_flows(Vm, Va, edge_index, edge_attr):
    """S_ft = V_f conj(Yff V_f + Yft V_t) per directed edge, from the edge's own admittances (float64 complex)."""
    V = torch.polar(Vm.double(), Va.double())
    f, t = edge_index
    yff = torch.complex(edge_attr[:, YFF_TT_R].double(), edge_attr[:, YFF_TT_I].double())
    yft = torch.complex(edge_attr[:, YFT_TF_R].double(), edge_attr[:, YFT_TF_I].double())
    return V[f] * torch.conj(yff * V[f] + yft * V[t])


class FillBaseCaseInputs(BaseTransform):
    """Overwrite the values (never the masks) at masked VM_H/VA_H (and optionally P_E/Q_E) with base-case information.

    Must run after ApplyMasking and before batching. The base case is looked up by `data["scenario_id"]`.
    """

    def __init__(self, store_path, mode: str = "vm_va_flows"):
        super().__init__()
        if mode not in DELTA_MODES:
            raise ValueError(f"delta_inputs must be one of {DELTA_MODES}, got {mode!r}")
        if not store_path:
            raise ValueError("PowerFlowContingencyAware needs data.basecase_store (path to basecase_store.pt; "
                             "build it with scripts/build_basecase_store.py)")
        self.mode = mode
        self.store_path = str(store_path)
        self.store = load_store_cached(store_path)

    def forward(self, data):
        sid = int(data["scenario_id"].reshape(-1)[0])
        Vm, Va = self.store.lookup(sid)
        bus_x = data["bus"].x
        if Vm.numel() != bus_x.size(0):
            raise ValueError(f"scenario {sid}: base case has {Vm.numel()} buses, sample has {bus_x.size(0)}")
        mb = data.mask_dict["bus"]
        bus_x[:, VM_H] = torch.where(mb[:, VM_H], Vm.to(bus_x.dtype), bus_x[:, VM_H])
        bus_x[:, VA_H] = torch.where(mb[:, VA_H], Va.to(bus_x.dtype), bus_x[:, VA_H])
        if self.mode == "vm_va_flows":
            ea = data[_BRANCH].edge_attr
            S = sending_end_flows(Vm, Va, data[_BRANCH].edge_index, ea)
            me = data.mask_dict["branch"]
            ea[:, P_E] = torch.where(me[:, P_E], S.real.to(ea.dtype), ea[:, P_E])
            ea[:, Q_E] = torch.where(me[:, Q_E], S.imag.to(ea.dtype), ea[:, Q_E])
        return data

    def __repr__(self):
        return f"{self.__class__.__name__}(mode={self.mode!r}, store={self.store_path!r})"


class PowerFlowContingencyAwareTransforms(Compose):
    """PowerFlow transforms (same PF mask) followed by the base-case fill."""

    def __init__(self, args):
        if getattr(args.data, "mask_type", None) == "rnd":
            raise ValueError("PowerFlowContingencyAware is defined for the deterministic PF mask only")
        super().__init__([
            RemoveInactiveBranches(), RemoveInactiveGenerators(), AddPFHeteroMask(), ApplyMasking(args=args),
            FillBaseCaseInputs(getattr(args.data, "basecase_store", None), getattr(args.data, "delta_inputs", "vm_va_flows")),
        ])


_register(TRANSFORM_REGISTRY, DELTA_TASK_NAME, PowerFlowContingencyAwareTransforms)
_register(TASK_REGISTRY, DELTA_TASK_NAME, PowerFlowTask)
_register(PHYSICS_DECODER_REGISTRY, DELTA_TASK_NAME, PhysicsDecoderPF)
