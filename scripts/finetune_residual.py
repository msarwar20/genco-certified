"""Fine-tune the RESIDUAL GENCO-Delta model: base-case state + learned correction (see src/genco_certified/genco_residual.py).

Thin wrapper around scripts/finetune.py (imported, not modified): it swaps in the loader that builds the
`GNS_heterogeneous_residual` model, initialises it from the RELEASED weights with the delta head's last layer
zeroed (so the model starts exactly at the base case), and treats `PowerFlowContingencyResidual` like the
delta task (base-case store check, `--delta_inputs`, tag). Every finetune.py flag works; `--task_name` defaults
to PowerFlowContingencyResidual.

    python scripts/finetune_residual.py --delta_inputs vm_va_flows --seed 0 --seed_train 0 --epochs 12 --accelerator mps \\
        --tag residual_vm_va_flows_s0 --log_dir results/genco_delta/train_logs/residual_vm_va_flows_s0

Outputs as finetune.py: checkpoints/finetuned/<tag>/best_model_state_dict.pt (a residual-model state dict; load it with
scripts/eval_genco_delta_residual.py --weights ...) and results/genco_delta/runs/<tag>.json.
"""

import argparse
import sys

sys.dont_write_bytecode = True  # do not touch scripts/__pycache__ (other jobs import finetune.py)
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import finetune  # noqa: E402  scripts/finetune.py, unmodified
from genco_certified import genco_residual as gr  # noqa: E402  registers model / task


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--task_name", default=gr.RESIDUAL_TASK_NAME)
    task = pre.parse_known_args()[0].task_name
    if not any(a == "--task_name" or a.startswith("--task_name=") for a in sys.argv[1:]):
        sys.argv += ["--task_name", task]
    # store coverage check, data overrides and tag suffix apply to the residual task as to the delta task
    finetune.DELTA = task if task in (gr.RESIDUAL_TASK_NAME, finetune.DELTA) else finetune.DELTA
    finetune.load_genco = gr.load_genco_residual  # model.type override + zero-initialised delta head
    finetune.main()


if __name__ == "__main__":
    main()
