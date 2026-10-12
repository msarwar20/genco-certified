# WS4-G: residual GENCO-Delta (base case + learned correction)

Status 2026-10-12: implemented, tested, smoke-run. No full-length arm has been trained. Everything is in new files; nothing that the running matrix imports or re-reads was touched (graphkit_plugin.py, loading.py, basecase_store.py, finetune.py, eval_genco_delta.py and run_genco_delta.sh keep their old mtimes; the wrappers import the scripts and patch module globals in memory).

## 1. Why

Plain GENCO-Delta (`PowerFlowContingencyAware`) fills the masked Vm/Va inputs with the base case and lets the network read them as features. The released network never used those columns (ws4_E section 5.4: state error 7.1 with filled inputs before training), so 12 epochs have to re-learn the whole readout, and the delta model ended worse than released GENCO as a Newton start (2.93 vs 2.15 iterations to 1e-6 MW on the IEEE 118 paper set; base case 2.88). The residual variant starts exactly at the base case and only learns a correction.

## 2. Design

Files (all new):

| file | content |
|---|---|
| `src/genco_certified/genco_residual.py` | model, loaders, registration |
| `scripts/finetune_residual.py` | wrapper around `scripts/finetune.py` |
| `scripts/eval_genco_delta_residual.py` | wrapper around `scripts/eval_genco_delta.py` + aggregation with the residual arms |
| `scripts/run_genco_delta_residual.sh` | the 2 arms x 3 seeds study (not launched) |
| `tests/test_genco_residual.py` | 11 tests |

Model `GNS_heterogeneous_residual` (MODELS_REGISTRY; subclass of `GNS_heterogeneous`, config `model.type`). At every layer

    bus_temp = base + delta_head(h_bus)        base = x_dict["bus"][:, VM_H:VA_H+1]  (Vm in p.u., Va in rad)

and everything else in the parent forward is unchanged: `torch.where(bus_mask, bus_temp, bus_fixed)` (known entries pass through), generator head, branch flows / node injection, physics decoder, residual feedback `h_bus += physics_mlp(residuals)`. With the contingency-aware transform, `x_dict["bus"]` holds the base-case Vm/Va at masked entries and the known values elsewhere, so `base` is exactly what the plugin put there.

Implementation choices:
- `delta_head` is the released `mlp_bus` itself, re-typed as `ResidualBusHead(nn.Sequential)` with `forward(h) = Sequential(h) + base`. The attribute name stays `mlp_bus`, so state-dict keys and shapes are identical to the plain model (test): released checkpoints load with `strict=True`, fine-tuned residual checkpoints reload with a plain `load_state_dict`, and `model.delta_head` is an alias. `base` is handed to the head by `forward` (set before the parent forward, cleared in a `finally`), not a parameter or buffer.
- Zero-init: last Linear of the head (weight and bias) set to 0. Done in `__init__` (config `model.zero_init_head`, default true) and again after loading released weights, because that load overwrites it. At init the decode is `base + 0.0`, bit for bit the base case at masked entries.
- `load_released_into_residual(model, state, zero_head=True)`: loads with `strict=False`, raises if anything is missing or unexpected, then zeroes the head (`zero_head=False` for fine-tuned residual checkpoints).
- `load_genco_residual(...)` wraps `loading.load_genco`: if `task.task_name == PowerFlowContingencyResidual` it sets `model.type`, and with `weights=None` it zeroes the head (released weights + zero head = the model at init); with `weights=<fine-tuned>` the head is left as trained. Other tasks pass through unchanged.
- Task `PowerFlowContingencyResidual`: TRANSFORM_REGISTRY -> `PowerFlowContingencyAwareTransforms` (same PF mask, same `FillBaseCaseInputs`, so same `data.basecase_store` / `data.delta_inputs` keys), TASK_REGISTRY -> `PowerFlowTask`, PHYSICS_DECODER_REGISTRY -> `PhysicsDecoderPF`. The model type is chosen by `model.type`, not by the task name; the wrappers set both.
- Wrappers: `finetune_residual.py` / `eval_genco_delta_residual.py` import the original script as a module (`sys.dont_write_bytecode`), set `module.DELTA = PowerFlowContingencyResidual` (so the store checks, `data` overrides, tag and meta treat it like the delta task), replace `module.load_genco` by `load_genco_residual`, default `--task_name` to the residual task, and call the original `main()`. All original flags and outputs apply. `--task_name PowerFlow` without weights in the eval wrapper still evaluates released GENCO.

Gradient structure (tested): at init only the last head layer (and the generator / physics-feedback path) receives gradient from the Vm/Va loss, since W = 0 blocks the rest of the head and the backbone. After the first optimiser step the earlier head layers and the backbone get gradient. This is the usual zero-init behaviour; if learning is too slow at lr 1e-4, `LR=3e-4` or `1e-3` is the first thing to try.

Known property: the physics feedback (`h_bus += physics_mlp(residuals)`) is computed from the decoded state, which at init is the base case at every layer, so the backbone sees the base-case mismatch (median max 31 MW on the paper set) instead of the geometric 261 -> 0.055 MW sequence it was pre-trained on. The released backbone features are therefore somewhat off at init as well, but they cannot hurt the output until the head is non-zero. A possible follow-up (not implemented): also zero the input-projection columns of the filled Va / P_E / Q_E features so the backbone behaves like the released one at init.

## 3. Tests (`tests/test_genco_residual.py`, 11 tests; `nice -n 15 python -m pytest -q tests` -> 69 passed in 14 s after a warm run)

1. Registration (model, transforms, task, decoder; plain model and aware task unchanged), identical state-dict keys/shapes as the plain model, released key set matches, released weights load with only `mlp_bus.3.*` zeroed and every other tensor bit-identical, a mismatching state dict raises.
2. (1) At init, on 3 real scenarios (released Base seed 0 weights): the final masked Vm/Va equal the filled input `torch.equal` and equal the store's base case per scenario; the same holds for random backbone weights (head zero); the plain released model on the same inputs is far from the base case (the test can fail).
3. Decode formula: with a random head, masked output = base + `delta_head.delta(h_last)`.
4. (4) Unmasked entries (known PV Vm, REF Va, ...) equal the input with a zero and with a large random head; the input is not modified by forward; `base` is cleared after forward and calling the head without a base raises.
5. (2) Gradients: last head layer weight and bias get nonzero gradient at init, earlier head layer and `input_proj_bus` get exactly zero; after one step on the last layer, the head's first layer, a transformer conv and `input_proj_bus` get nonzero gradient.
6. (3) Round trip: a 4-step fine-tuned state dict (Lightning `model.` prefix, as finetune.py saves) reloaded into a fresh model by (a) `load_released_into_residual(zero_head=False)` and (b) plain `load_state_dict` gives identical outputs (`torch.equal`, bus and gen); with `zero_head=True` it does not (the two paths are distinct), and then masked output = base.
7. End to end through the project loader (`load_genco_residual`, config + normalizer + dataset + Lightning task on the paper test split): model type, at-init equals base on a real test batch, a fine-tuned state dict passed as `weights=` reloads identically and the head is not re-zeroed.

## 4. Smoke results (IEEE 118 Base, init seed 0, paper test split, 638 scenarios / 29 load scenarios; CPU, niced, `lr 1e-4`, batch 32, vm_va_flows)

Mean NR iterations to 1e-2 / 1e-4 / 1e-6 MW (95% cluster-bootstrap CI), median state error (l2). Artifacts in `results/genco_delta/residual_smoke/` (eval JSONs, run records, logs, table); state dicts `checkpoints/finetuned/smoke_residual_*/best_model_state_dict.pt` (the Lightning `best.ckpt` files were deleted to save disk).

| model | iters 1e-2 | iters 1e-4 | iters 1e-6 | model - base (1e-6) | state err | elem. misses / false alarms after 1 Newton step |
|---|---|---|---|---|---|---|
| released GENCO (reference, ws4_E) | 1.72 | 2.00 | 2.15 | -0.73 | 0.049 | 1 / 0 |
| base case (reference) | 2.04 | 2.49 | 2.88 | | 0.0502 | 33 / 6 |
| residual at init (0 steps) | 2.04 [2.00, 2.07] | 2.49 [2.45, 2.53] | 2.88 [2.83, 2.93] | 0 [0, 0] | 0.0502 (ratio to base 1.0000) | 33 / 6 |
| residual, 14 steps (0.1 epoch, 160 s, val loss 2.04e-3) | 2.04 | 2.50 [2.46, 2.54] | 2.88 [2.83, 2.93] | -0.0016 [-0.0063, 0.0031] | 0.0499 | 33 / 6 |
| residual, 74 steps (0.5 epoch, 226 s, val loss 2.09e-3) | 2.03 [2.00, 2.06] | 2.49 [2.45, 2.53] | 2.87 [2.82, 2.92] | -0.0094 [-0.0172, -0.0016] | 0.0496 | 31 / 5 |
| for comparison, plain Delta vm_va_flows, 14 steps (ws4_E) | 3.01 | 3.24 | 3.94 | +1.06 | 1.82 | 203 / 641 |

Reading: at init the residual model reproduces the base case exactly on every statistic (iteration counts, state error, physical mismatch, decisions), as designed. After 14 or 74 steps it is still within noise of the base case (slightly better state error, ratio 0.997 to base after 74 steps); nothing is destroyed, in contrast with the plain Delta model (state error 1.82 after the same 14 steps). That the model does not yet beat the base case after 0.5 epoch is expected at lr 1e-4 with a zero-initialised head; whether 12 epochs reach or beat the released GENCO (2.15 iterations) is the open experiment. Plumbing is exercised end to end: fine-tune wrapper -> run record (config shows `GNS_heterogeneous_residual`, `PowerFlowContingencyResidual`) -> checkpoint -> eval wrapper reload -> aggregation (`--aggregate --prefix smoke_`).

Not run: GPU/MPS, other seeds, stress sets, 12 epochs. `results/configs/PowerFlowContingencyResidual_case118_ieee_base_seed0.yaml` is written by `load_genco` (separate file from the matrix' tasks).

## 5. How to launch (after the current matrix has finished)

```
nohup ./scripts/run_genco_delta_residual.sh > results/genco_delta/run_residual.log 2>&1 &
```
Protocol identical to `run_genco_delta.sh`: arms `residual_vm_va`, `residual_vm_va_flows` x seeds {0,1,42}; released Base seed s as init and training seed s; per-seed splits (`splits_paper_test`, `splits_paper_test_seed1`, `..._seed42`); 12 epochs, lr 1e-4, batch 32, `--accelerator mps`; model selection on paper-val loss; each fine-tuned model evaluated on the seed's paper test split and stress n1, n2, n3, n5 into `results/genco_delta/eval/<arm>_s<seed>__<set>.{json,parquet}` (tags distinct from the matrix' `control_*`, `delta_*`, `slack_aware_*`). Released checkpoints are evaluated with the matrix' tags `released_s<seed>__<set>` and skipped when those JSONs exist. Resumable (JSON exists -> skip), `FORCE=1` redoes. Runs: logs `results/genco_delta/logs/`, run records `results/genco_delta/runs/`, checkpoints `checkpoints/finetuned/<arm>_s<seed>/`.

Knobs: `SEEDS`, `ARMS`, `EPOCHS`, `LR` (e.g. `LR=3e-4`; note that the tags do not encode lr, so use a different `ARMS`/tag or a fresh checkout of the outputs for a second lr), `BS`, `ACCEL`, `SETS`, `SMOKE=1` (tag prefix `smoke_`), `NO_TRAIN`, `NO_EVAL`, `INIT_EVAL=1` (also evaluates the 0-step model as `residual_init_s<seed>__<set>`; must equal the base case), `NO_RELEASED=1`, `NICE=15`. Stores and per-seed splits are only rebuilt when missing (the matrix creates them). Do not run two evaluations of the residual task in parallel (shared `results/configs` YAML).

Table: the script ends with `eval_genco_delta_residual.py --aggregate`, which writes `results/genco_delta/residual_summary_runs.csv`, `residual_summary_by_arm.csv`, `residual_summary_by_arm.md` (mean and SD over init seeds per arm x set, released / control / slack_aware / delta / residual arms together when their evaluations are in `results/genco_delta/eval`). The matrix' `summary_*` files are not touched, and its own aggregator ignores the residual runs.

Single run by hand:
```
python scripts/finetune_residual.py --delta_inputs vm_va_flows --seed 0 --seed_train 0 --split splits_paper_test --epochs 12 --lr 1e-4 --accelerator mps --tag residual_vm_va_flows_s0 --log_dir results/genco_delta/train_logs/residual_vm_va_flows_s0
python scripts/eval_genco_delta_residual.py --delta_inputs vm_va_flows --seed 0 --weights checkpoints/finetuned/residual_vm_va_flows_s0/best_model_state_dict.pt --tag residual_vm_va_flows_s0__paper
python scripts/eval_genco_delta_residual.py --delta_inputs vm_va_flows --seed 0 --tag residual_init_s0__paper     # no --weights: released + zero head = base case
```

Caveats: (i) `meta.weights` in an at-init eval JSON says "released" (it is the released backbone with a zeroed head, `task_name` shows the residual task); (ii) the main aggregator `eval_genco_delta.py --aggregate` does not know the `slack_aware` arm or the residual arms, use the residual aggregator for tables that should contain them; (iii) as in the matrix, the delta / residual arms receive the true slack Vm through the base case (ws4_E 5.2).
