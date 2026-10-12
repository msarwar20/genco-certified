# WS4-E: GENCO-Δ implementation (plugin, store, fine-tune, evaluation, matrix, Colab)

Status 2026-10-11: everything below is implemented and exercised end to end on a smoke run. No full-length arm has been trained yet. The numbers in the smoke section are plumbing checks, not results.

## 1. Design

Idea: at contingency time the operator already holds the solved pre-contingency state. GENCO's PF task zeroes every masked input. GENCO-Δ keeps the PF mask exactly as it is (same booleans, same unknowns, same loss) and changes only the *values* sitting at masked positions:

| masked position | PowerFlow | PowerFlowContingencyAware |
|---|---|---|
| bus `VM_H` (PQ and REF) | 0 | base-case Vm (p.u.) |
| bus `VA_H` (PQ and PV) | 0 | base-case Va (rad) |
| edge `P_E`, `Q_E` (all branches) | 0 | pre-contingency sending-end flow of this edge (`--delta_inputs vm_va_flows` only) |
| everything else masked (limits, ratings, ANG_MIN/MAX, Qg, gen limits/costs) | 0 | 0, untouched |

Code:
- `src/genco_certified/graphkit_plugin.py`: `FillBaseCaseInputs` transform and the `PowerFlowContingencyAware` task (TRANSFORM_REGISTRY: RemoveInactiveBranches, RemoveInactiveGenerators, AddPFHeteroMask, ApplyMasking, FillBaseCaseInputs; TASK_REGISTRY: PowerFlowTask; PHYSICS_DECODER_REGISTRY: PhysicsDecoderPF). `PowerFlowSlackAware` is untouched. Config keys: `data.basecase_store` (path), `data.delta_inputs` (`vm_va` | `vm_va_flows`, default `vm_va_flows`).
- `src/genco_certified/basecase_store.py`: `BaseCaseStore` (scenario id → Vm, Va; dense rows per load scenario plus an id map), `build_store`, `solve_base_case`.
- `scripts/build_basecase_store.py`: builds `<data_root>/<grid>/basecase_store.pt` and checks that it covers every id of every `splits*` folder. Also writes per-seed paper splits (section 5).

Where the base case comes from: `acpf.base_case_voltages` on the raw parquet of all variants of the load scenario (intact topology = majority status per element, then Newton from a flat start). If that diverges (11 of 182 GOC 2000 load scenarios; none on IEEE 118) `solve_base_case` retries from the stored solution of a few variants, then from a DC start. All 182 then converge. Raw parquet is read one Hive partition at a time and a load scenario is solved after its last partition has been read (load scenarios straddle partition boundaries, and a test covers it).

Lookup: `data["scenario_id"]` is set by `graph_builder.build_hetero_data` (line 184), is a global store attribute, survives `RemoveInactive*`, masking and the dataset transform, and is concatenated by PyG batching. The transform runs per sample before batching.

## 2. Units decision, with code evidence

The question was whether masked features are filled in normalized or physical units. Answer: normalized, because the order is `Dataset.get` (normalise) then `transform` (task transforms).

- `datasets/powergrid_hetero_dataset.py:352`: `HeteroGridDatasetDisk.get` calls `self.data_normalizer.transform(data=data)` and returns the normalised sample.
- `torch_geometric/data/dataset.py:296`: `__getitem__` then applies `self.transform` to what `get` returned. So `ApplyMasking` (`datasets/transforms.py:91`, `x[mask] = mask_value`) and our fill both see normalised tensors.
- Normaliser (`datasets/normalizers.py`): bus `VA_H` is multiplied by π/180 (line 196), so Va is in **radians** at fill time; `VM_H` is not touched (p.u.). Edge `P_E`, `Q_E` are divided by baseMVA (225) and the four admittance columns multiplied by `baseMVA_orig / baseMVA` (227–228).
- Consequence for flows: with p.u. voltages and those scaled admittances, `S = V_f · conj(Yff V_f + Yft V_t)` equals MW / baseMVA, which is exactly the scale of `P_E`/`Q_E`, and is what the model's own `ComputeBranchFlow` computes from normalised inputs. So the transform needs no normaliser object, works for both `HeteroDataMVANormalizer` and the per-sample normaliser, and uses this edge's own `yff`, `yft` (no branch-index mapping; the reverse edge carries Ytt/Ytf in the same slots).
- The model never reads the filled values as "fixed" quantities: `models/gnn_heterogeneous_gns.py:255` is `torch.where(bus_mask, bus_temp, bus_fixed)`, so at masked positions the prediction is used. The filled values only enter `input_proj_bus` / `input_proj_edge` as features. The physics decoder and residual layers read Pd, Qd, GS, BS and predictions, not the masked Vm/Va.

Evidence in tests (all pass): the formula reproduces datakit's stored `pf`/`qf` from the true voltages in normalised units; the filled `P_E`/`Q_E`, de-normalised, equal `acpf.branch_flows` on the base state (MW / MVAr); the filled Va/Vm equal an independently solved `acpf.base_case_voltages`; the unmasked REF angle in `x` equals the label `y` in radians.

## 3. Tests (`tests/test_genco_delta.py`, 24 tests; `python -m pytest -q tests` → 57 passed)

1. Registration (task, transform, decoder; SlackAware still registered) and explicit config errors (no store, bad mode, `mask_type: rnd`).
2. `test_masks_identical_to_powerflow`: every mask tensor equal to PowerFlow for 3 scenarios × 2 modes; same edge index, gen features and scenario id.
3. `test_only_masked_voltage_and_flow_values_change`: only bus columns VM/VA (only where masked) and edge columns P_E/Q_E change; all other masked entries stay 0; ANG/RATE limits stay 0; `vm_va` leaves edges bit-identical.
4. Units / values: filled Vm, Va equal the independently solved base case (including a load scenario whose variants straddle a partition boundary); Va unit check against the label; all variants of a load scenario get the same base case.
5. Flow formula versus datakit's stored flows; filled flows versus `acpf.branch_flows` on the base state for 3 scenarios.
6. No leakage: garbling `y` (bus, gen, edge) leaves every input tensor bit-identical; filled voltages differ from the labels.
7. Store: round trip, coverage, unknown scenario raises `KeyError`, full store covers every id of train/val/test.
8. Base-case solver: equals `acpf.base_case_voltages` bit for bit when the flat start converges; the retry converges on GOC 2000 load scenario 96, where the flat start diverges.
9. Evaluation: with the truth as prediction the analysis gives zero state error and zero decision errors; cluster bootstrap treats load scenarios as indivisible.

## 4. Scripts

- `scripts/finetune.py` (backward compatible: old calls give the old tags and outputs). New flags: `--data_root`, `--split`, `--basecase_store`, `--delta_inputs`, `--seed_train` (Lightning `seed_everything`), `--tag`, `--max_time`, `--run_dir`. Always the released normaliser. Best checkpoint by **validation** loss only. Writes `results/genco_delta/runs/<tag>.json`: args, git rev, versions, sha256 of the raw parquet files, of the split ids, of the store, of the normaliser stats and of the initial weights, epochs run, best epoch, full validation curve, wall time, effective config.
- `scripts/eval_genco_delta.py`: per scenario NR iterations to 1e-2/1e-4/1e-6 MW from the model output (setpoints via `V_from`), from the base case (store) and from a flat start; state error and physical mismatch of model and base case; element decisions (branch |S|/rate > 1.01; PQ-bus |V| outside limits by more than 1 mV) after 0/1/2 Newton steps from the model and 1/2 from the base case, counted against the datakit truth. Writes `<tag>.parquet` and `<tag>.json` (means/medians, ratio-of-sums rates, 95% cluster bootstrap over load scenarios, 2000 resamples; unconverged runs are charged the iterations taken). `--aggregate` tabulates all runs into `summary_runs.csv`, `summary_by_arm.csv/.md` (mean ± SD over init seeds). Works for released GENCO (`--task_name PowerFlow`, no `--weights`). **Cross-check:** on the 638 paper-test scenarios it reproduces exp02's per-scenario iteration counts exactly (1.72 / 2.00 / 2.15 for GENCO, 2.04 / 2.49 / 2.88 for the base case, 3.01 / 4.00 / 4.00 flat).
- `scripts/run_genco_delta.sh`: the full matrix (section 6).
- `notebooks/colab_genco_delta.ipynb`: section 7.

## 5. Findings that affect the study design

1. **Released seeds 1 and 42 were trained on different splits than seed 0.** Of seed 0's 638 paper-test ids in the first 6,000 scenarios, 462 (72%) are in seed 1's training set and 506 (79%) in seed 42's. Evaluating released seed-1/42 checkpoints, or fine-tunes starting from them, on seed 0's test split is not a held-out test. Any earlier table that does this for seeds 1/42 (WS2.1) has the same issue. `scripts/build_basecase_store.py --seed_splits 1 42` writes `data/pf/case118_ieee/splits_paper_test_seed{1,42}` from the published split files (train/val/test = 4416/770/814 and 4790/594/616 ids in range); the matrix fine-tunes and tests seed s on its own split. Stress sets are generated fresh, so they are clean for every seed.
2. **The base-case Vm at PV and REF buses equals the post-contingency setpoint to 6e-8 p.u.** So the delta arms receive the true slack Vm as an input, although REF Vm stays "masked" (the model still predicts it, and evaluation enforces the setpoint anyway). The control does not get this. If a clean attribution is needed, the PowerFlowSlackAware fine-tune is the comparator for that part, or the REF Vm fill can be made optional.
3. **How far the base case is from the post-contingency labels** (median over scenarios of the per-scenario max): |ΔVm| at PQ buses 4.3e-3 p.u. and |ΔVa| 2.4e-2 rad on the paper set; 4.5e-3 / 3.3e-2 (N-1), 9.6e-3 / 5.2e-2 (N-2), 1.4e-2 / 9.5e-2 (N-3), 1.8e-2 / 1.3e-1 (N-5) on the stress sets.
4. **Released weights with filled inputs are badly broken before any training**: median state error 7.1 (vs 0.049 unfilled), both modes. The Va input column of the first layer has norm 0.26, like the always-masked MIN_VM/MAX_VM/MIN_QG/MAX_QG columns (0.26–0.31) and unlike inputs the model does use (Pd, Qd, Vm, the bus-type flags, BS: 0.4–0.9); that is consistent with the Va column never having received a gradient in PF pre-training (Va is 0 at REF and masked elsewhere), so the arms start from a destroyed state and must re-learn. This is inference, not tested.
5. **GOC 2000 load scenarios are almost all partially seen by the released model.** It trained on a random 20k-scenario subset, about 8% of the variants of a typical load scenario, so in partitions 0–19 only 40 of 182 load scenarios (18 published-test plus 22 never sampled) have no variant in its train/val split. The Colab notebook uses these 40 for validation and test and everything else (partial overlap with pre-training, identical for all arms) for training.

## 6. How to run the IEEE 118 matrix

```
nohup ./scripts/run_genco_delta.sh > results/genco_delta/run_matrix.log 2>&1 &
```
Arms control / delta_vm_va / delta_vm_va_flows × init seeds {0,1,42}; released Base seed s as init and training seed s; 12 epochs, lr 1e-4, batch 32, MPS; paper-train ids of seed s for training, paper-val ids for selection. After each arm: evaluation on the seed's paper test split and stress n1, n2, n3, n5; the released checkpoints are evaluated the same way (`released_s<seed>__<set>`). Resumable (finished runs and evaluations are skipped; `FORCE=1` redoes), logs in `results/genco_delta/logs/`, run records in `results/genco_delta/runs/`, evaluations in `results/genco_delta/eval/`, tables in `results/genco_delta/summary_*`. Estimated 6.5 h on an M4 (about 3.3 min per epoch, 9 runs, plus about 25 s per evaluation). Knobs by environment variable: `SEEDS`, `ARMS`, `SETS`, `EPOCHS`, `LR`, `BS`, `ACCEL`, `SMOKE=1`, `NO_TRAIN`, `NO_EVAL`, `FORCE`. `finetune.py --log_dir` keeps graphkit's `stats/` side files (normaliser stats, split ids) out of the git-tracked `results/finetune/stats`; the matrix passes `results/genco_delta/train_logs/<tag>`. Do not run the evaluations of two arms of the same task name in parallel: `load_genco` writes a shared `results/configs/<task>_<grid>_<size>_seed<s>.yaml` and reads it back immediately.

Stores for IEEE 118 (6000 scenarios / 273 load scenarios) and the four stress sets (660, 659, 660, 660 scenarios / 30 load scenarios each) are built (3 s each, none non-converged).

## 7. Colab notebook (GOC 2000, Small, released seed 0)

`notebooks/colab_genco_delta.ipynb`, 24 cells. Parameters in cell 1 (`DRY_RUN`, `EPOCHS`, `BATCH_SIZE`, `N_TEST_LOAD`, `MAX_TRAIN_MINUTES`, ...). Steps: upload a tarball of `src scripts pyproject.toml` (`tar czf genco_repo.tgz src scripts pyproject.toml`); install (torch 2.10 to 2.13, torch-scatter wheel or a pure-torch stand-in, pinned graphkit `8a8defb`, pandas<3); download partitions 0–19 and the released checkpoint's published split; split by load scenario (section 5.5); base-case store; fine-tune the three arms (`--accelerator gpu`, `--max_time` cap); evaluate released + arms on the test load scenarios; table, plot, zip. Expected 1.5 to 3 h on a T4 with the defaults (6 epochs per arm).

Tested locally (everything except the Colab-only upload/install cells and CUDA): in `DRY_RUN` mode on 5 GOC 2000 partitions with the real scripts, on MPS (split, store, 3 fine-tunes, 4 evaluations, table, zip). The torch-scatter stand-in was checked against the real `torch_scatter` (add, mean, max, N-D, autograd). Not tested: the pip steps on Colab itself (torch upgrade, `gridfm-datakit==1.1.0` and its dependencies such as `juliacall`, which graphkit imports at module level through `opf_ac_dc_baseline`), so expect to look at the install cell's output first.

## 8. Smoke results (IEEE 118 Base, init seed 0, paper test split, 638 scenarios / 29 load scenarios; plumbing only)

Smoke fine-tune: `PowerFlowContingencyAware`, `vm_va_flows`, 1 epoch with `limit_train_batches 0.1` (14 steps), MPS, 48 s, validation loss 0.056 (`results/genco_delta/runs/smoke_delta_vm_va_flows_s0.json`). Mean NR iterations to 1e-2 / 1e-4 / 1e-6 MW (95% cluster-bootstrap CI), median state error (l2), element decisions after 1 Newton step (484 violating elements in the truth):

| model | iters 1e-2 | iters 1e-4 | iters 1e-6 | model − base (1e-6) | state err | misses g1 / false alarms |
|---|---|---|---|---|---|---|
| released Base s0 | 1.72 [1.58, 1.85] | 2.00 [2.00, 2.01] | 2.15 [2.06, 2.25] | −0.73 [−0.83, −0.63] | 0.049 | 1 / 0 |
| control, 0.1 epoch | 1.95 [1.88, 2.00] | 2.01 [2.00, 2.02] | 2.30 [2.17, 2.45] | −0.58 [−0.72, −0.44] | 0.107 | 1 / 0 |
| delta_vm_va, 0.1 epoch | 3.01 [3.00, 3.01] | 3.08 [3.03, 3.15] | 3.58 [3.42, 3.74] | +0.70 [0.54, 0.85] | 1.53 | 146 / 487 |
| delta_vm_va_flows, 0.1 epoch | 3.01 [3.00, 3.01] | 3.24 [3.12, 3.38] | 3.94 [3.85, 4.00] | +1.06 [0.96, 1.15] | 1.82 | 203 / 641 |
| delta_vm_va_flows, 2 epochs | 2.06 [2.04, 2.08] | 2.84 [2.72, 2.95] | 3.04 [3.03, 3.05] | +0.16 [0.11, 0.20] | 0.308 | 20 / 2 |
| base case (reference) | 2.04 | 2.49 | 2.88 | | 0.050 | 33 / 6 |
| flat start (reference) | 3.01 | 4.00 | 4.00 | | | |

Reading: after 14 steps the delta arms are worse than a flat start, as expected from section 5.4. After 2 full epochs (3.2 min each; validation loss 7.6e-3 → 5.2e-3, against about 2.7e-4 for the control after one epoch) the delta model is close to the base case on iterations but its state error is 6× the base case's. Whether 12 epochs at lr 1e-4 closes that gap is the open experiment; if it does not, candidate changes are more epochs or a higher learning rate for the first-layer input weights, and predicting the correction to the base case instead of the absolute state. The smoke matrix (`SMOKE=1`, seeds 0 and 1, sets paper and n3, `results/genco_delta/smoke_*`) also exercised the per-seed splits, the stress path and `--aggregate`.

Smoke artifacts are all prefixed `smoke_`; the aggregator skips them unless `--prefix smoke_`. `inference_ms_per_scenario_cpu` in the eval JSONs was measured on a busy machine and is not a benchmark.
