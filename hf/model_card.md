---
license: apache-2.0
library_name: gridfm-graphkit
pipeline_tag: tabular-regression
base_model: gridfm/genco-pf-datakit
tags:
  - power-systems
  - graph-neural-network
  - power-flow
  - genco
  - contingency-analysis
---

# GENCO fine-tunes on IEEE 118 (slack-aware, GENCO-Δ, controls)

This repository holds twelve fine-tuned checkpoints of the released GENCO power-flow model (IEEE 118, Base, `gridfm/genco-pf-datakit`, `case118_ieee/base/seed{0,1,42}`), which are the research artifacts behind the post "What Is GENCO Worth to a Newton Solver? Savings, Safety and Break-Even Against the Pre-Contingency Start" (GridFM GENCO Challenge 2026). Code, data configs and the full numbers are available at these links: [GitHub](https://github.com/msarwar20/genco-certified), [checkpoints](https://huggingface.co/sarwar21255/genco-ieee118-finetunes), [data](https://huggingface.co/datasets/sarwar21255/genco-contingency-ieee118), [demo](https://huggingface.co/spaces/sarwar21255/genco-operator-console).

These checkpoints are not improved general-purpose power-flow models: two of the four families were trained to test whether giving GENCO the pre-contingency state helps a Newton solver, and the answer in the post is mostly "no". They are meant for reproducing the post or as a starting point for the same experiments.

## Checkpoints

Each family has three seeds (0, 1, 42), and seed s is initialised from the released seed-s checkpoint and then trained and validated on that seed's published train/val split. Checkpoint selection uses validation loss only.

| Directory | Task / model | What it is |
|---|---|---|
| `residual_vm_va_flows_lr3e-4_s{0,1,42}` | `PowerFlowContingencyResidual`, `GNS_heterogeneous_residual` | GENCO-Δ, residual decoder: at every layer the bus state is the pre-contingency state plus a learned delta (head initialised to zero). Trained on N-1. Takes the pre-contingency Vm, Va and branch flows as extra inputs. |
| `residual_nk_vm_va_flows_lr3e-4_s{0,1,42}` | same | Same model, trained on 5,056 fresh scenarios with one to five elements out. |
| `slack_aware_s{0,1,42}` | `PowerFlowSlackAware`, `GNS_heterogeneous` | Same architecture as the released model. The slack-bus voltage magnitude is a known input instead of a masked unknown. |
| `control_s{0,1,42}` | `PowerFlow`, `GNS_heterogeneous` | Matched control: the original task, same recipe as `slack_aware`. |

The post also discusses plain GENCO-Δ (`delta_vm_va*`), but those checkpoints are left out of this repository because they ended at about base-case quality and one seed diverged.

## Training recipe

All runs use the IEEE 118 Base architecture (hidden size 48, 12 layers, 8 heads), the released normalizer statistics, AdamW, ReduceLROnPlateau (factor 0.7, patience 5), batch size 32, 12 epochs and the loss `LayeredWeightedPhysics` (0.1) + `MaskedBusMSE` (0.9). Training ran on Apple MPS with graphkit commit `8a8defb` and datakit commit `d8a092d`, and each run took about 1,700 to 1,800 optimiser steps and 28 to 42 minutes. The exception is N-k seed 0, which took 2.7 hours because the machine slept (see `run.json`).

| Family | lr | Training data | Train / val scenarios (s0 / s1 / s42) | Best epoch (s0 / s1 / s42) | Best val loss (s0 / s1 / s42) |
|---|---|---|---|---|---|
| `residual_vm_va_flows_lr3e-4` | 3e-4 | paper N-1 split | 4,724 / 4,416 / 4,790 train; 638 / 770 / 594 val | 10 / 10 / 12 | 3.64e-4 / 5.58e-4 / 6.10e-4 |
| `residual_nk_vm_va_flows_lr3e-4` | 3e-4 | `data/train_nk` (seed 4001, N-1 to N-5) | 4,550 train; 506 val | 10 / 12 / 11 | 7.22e-4 / 6.34e-4 / 9.44e-4 |
| `slack_aware` | 1e-4 | paper N-1 split | as the first row | 10 / 12 / 10 | 3.95e-4 / 3.38e-4 / 6.84e-4 |
| `control` | 1e-4 | paper N-1 split | as the first row | 7 / 12 / 7 | 2.76e-4 / 2.37e-4 / 5.58e-4 |

The data hashes below are also stored in each `run.json`:
- paper N-1 raw data (first 30 partitions of `gridfm/pf_small_case118_ieee`): sha256 `da277aa1e851…`; split sha256 `165593de47c6…` (s0), `e1cd451f9bb9…` (s1), `d93c5708ad97…` (s42).
- N-k set: raw sha256 `22b0283a162a…`, split sha256 `eb016de1ebc0…`. It was generated from `configs/datakit/case118_train_nk.yaml`, and 23.7% of its outages are generators while 58.2% of its scenarios contain a violation.
- Released initial weights (sha256 prefix): `3ea2f552b190…` (s0), `290783871551…` (s1), `2706a1403301…` (s42).

The fine-tuning used about 2% of the data of the original 250,000-scenario training run, which is a small budget, so none of the conclusions below concern full-scale training.

## Evaluation

Results are for IEEE 118 Base, averaged over seeds 0/1/42, with each seed evaluated on its own published test split (616 to 814 scenarios for "paper" and 660 for the 30-load-scenario N-5 stress set). Iterations are Newton–Raphson iterations counted from the model's output and averaged over scenarios, and the numbers come from `docs/NUMBERS.md` sections 3.5 and 9, `results/genco_delta/residual_summary_by_arm.md` and the 2026-10-25 entry of `RESEARCH_LOG.md`.

| Model | State error, paper | State error, N-5 | Iterations to 10⁻² MW, paper | To 10⁻⁴ MW, paper | To 10⁻⁶ MW, paper | To 10⁻⁶ MW, N-5 | One-step misses, N-5 (of 1,763) |
|---|---|---|---|---|---|---|---|
| released | 0.054 | 0.112 | 1.74 | 2.01 | 2.15 | 2.50 | 6.3 |
| control | 0.067 | 0.119 | 1.76 | 2.01 | 2.19 | 2.52 | 6.7 |
| slack-aware | 0.086 | 0.133 | 1.74 | 2.00 | 2.19 | 2.49 | 7.0 |
| residual GENCO-Δ (N-1) | 0.022 | 0.155 | 1.33 | 2.05 | 2.19 | 2.69 | 17.7 |
| residual GENCO-Δ (mixed N-k) | 0.062 | 0.116 | 1.79 | 2.03 | 2.14 | 2.57 | 9.0 |
| pre-contingency start (no model) | | | | | 2.91 | 3.32 | 129 |

State error is the median L2 error of the predicted (Vm, Va) against the datakit solution, and "one-step misses" counts the violating elements still missed after the model plus one Newton step.

The numbers support the following readings:
- Residual GENCO-Δ trained on N-1 cuts N-1 state error about 2.5× and lowers the iterations to 10⁻² MW from 1.74 to 1.33, but it saves nothing at 10⁻⁴ MW and regresses at N-5.
- Mixed N-k training removes the N-5 regression (error 0.116 against 0.155) but also removes the N-1 gain (0.056 against 0.022 at N-1), so it ends at about released quality everywhere and does not beat the released model.
- Slack-aware fine-tuning lowers the physical largest nodal mismatch from 17.2 to 1.0 MW at the slack setpoint on the paper set (29% more state error than the control), and does not change Newton iterations.
- The N-k rows carry a confound, because the training scenarios are new as well as differently mixed, and a matched N-1-only set of 4,550 fresh scenarios was not run.

## How to load

The checkpoints are state dicts for graphkit's `GNS_heterogeneous` (slack-aware, control) and for `GNS_heterogeneous_residual` (GENCO-Δ). They need three things beyond the weights: graphkit at commit `8a8defb`, the plugin package `genco_certified` from the code repository (it registers the tasks and the residual model), and the released normalizer statistics, which `load_genco` downloads from `gridfm/genco-pf-datakit` on first use. The GENCO-Δ checkpoints also need the pre-contingency state of every scenario (`basecase_store.pt`, built by `scripts/build_basecase_store.py` from the same datakit data).

```python
from pathlib import Path
from huggingface_hub import hf_hub_download
from genco_certified.loading import load_genco
from genco_certified.genco_residual import load_genco_residual

repo = "sarwar21255/genco-ieee118-finetunes"
data_root = Path("data/pf")  # <data_root>/case118_ieee/raw/*.parquet, from scripts/fetch_data.py
split = data_root / "case118_ieee" / "splits_paper_test"  # seed 1 and 42: splits_paper_test_seed1 / _seed42

# slack-aware or control: a plain GENCO state dict
w = hf_hub_download(repo, "slack_aware_s0/best_model_state_dict.pt")
lg = load_genco("pf", "case118_ieee", "base", seed=0, weights=w,
                overrides={"task": {"task_name": "PowerFlowSlackAware"}},  # omit for control
                data_root=data_root, split_dir=split)

# residual GENCO-Δ
w = hf_hub_download(repo, "residual_vm_va_flows_lr3e-4_s0/best_model_state_dict.pt")
lg = load_genco_residual(
    "pf", "case118_ieee", "base", seed=0, weights=w,
    overrides={"task": {"task_name": "PowerFlowContingencyResidual"},
               "data": {"basecase_store": str(data_root / "case118_ieee" / "basecase_store.pt"),
                        "delta_inputs": "vm_va_flows"}},
    data_root=data_root, split_dir=split)

model = lg.task.model.eval()           # call as model(batch); lg.test_loader() yields batches
```

Load the seed-s checkpoint with `seed=s` and that seed's split, since the config and normalizer come from the released seed-s run. All twelve checkpoints were loaded with this recipe and gave finite outputs on a test batch, and `scripts/eval_genco_delta.py` and `scripts/eval_genco_delta_residual.py` reproduce the evaluation table.

## Limitations

- These models cover IEEE 118 only, and GENCO-Δ was not run on the 500-bus or 2,000-bus grids, where the pre-contingency start is the stronger prior.
- The weights are not standalone: they cannot be used without graphkit, the plugin code and the normalizer statistics, and the GENCO-Δ models also need the pre-contingency state as input.
- The fine-tuning budget is small (about 2% of the original data, 12 epochs), so the slack-aware and GENCO-Δ conclusions apply to this budget and not to full-scale training.
- Residual GENCO-Δ trained on N-1 regresses at N-5; the mixed-N-k version trades that for the N-1 gain.
- Evaluation used datakit scenarios only, and the N-k and N-1 test sets share a global load trajectory with some training sets (datakit takes the first n hours of the load profile), although outages, noise and dispatch are independent.
- The slack-aware models fix the reported-versus-physical power balance gap at the cost of state error, and the other repairs we tried (setpoint as zero-shot input, output clamp), which the post lists, are not released as checkpoints.
- These models are not for operational use, because the post's safety results are empirical, were obtained on datakit scenarios, and apply to the screening policy with its calibration and not to these weights alone.

## Files

The repository holds the state dicts only (`best_model_state_dict.pt`, 80,445,203 bytes each) and the per-run training record `run.json` (copied from `results/genco_delta/runs/<tag>.json`, with paths written as `<repo>`). The Lightning `best.ckpt` files (241 MB each) are not needed to load or evaluate the models and are not uploaded.

## Citation

```
@misc{sarwar2026genco_worth,
  author = {Sarwar, Muhammad and Rizwan, Muhammad},
  title  = {What Is GENCO Worth to a Newton Solver? Savings, Safety and Break-Even Against the Pre-Contingency Start},
  year   = {2026},
  note   = {GridFM GENCO Challenge 2026 submission},
  url    = {https://github.com/msarwar20/genco-certified}
}
```

The base model is GENCO (A. Puech, M. Mazzonelli, T. R. Govindasamy et al., "GENCO: a unified neural solver embedded in a development framework for steady-state grid analysis", arXiv:2608.09921, 2026), and this repository is licensed Apache-2.0, like the base checkpoints.

## Authors

Muhammad Sarwar (Iowa State University) and Muhammad Rizwan (Pakistan Institute of Engineering & Applied Sciences). Contact: msarwar@iastate.edu.
