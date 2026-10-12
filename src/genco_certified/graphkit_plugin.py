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

from torch_geometric.transforms import Compose

from gridfm_graphkit.datasets.globals import VM_H
from gridfm_graphkit.datasets.masking import AddPFHeteroMask, AddRandomHeteroMask
from gridfm_graphkit.datasets.transforms import ApplyMasking, RemoveInactiveBranches, RemoveInactiveGenerators
from gridfm_graphkit.io.registries import PHYSICS_DECODER_REGISTRY, TASK_REGISTRY, TRANSFORM_REGISTRY
from gridfm_graphkit.models.utils import PhysicsDecoderPF
from gridfm_graphkit.tasks.pf_task import PowerFlowTask

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
