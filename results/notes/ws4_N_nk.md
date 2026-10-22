# WS4-N: residual GENCO-Delta trained on a mixed N-k set

Status 2026-10-22: data, splits, store, smoke done; three-seed run launched, seed 0 training.

## Data
`configs/datakit/case118_train_nk.yaml` = `case118_n2.yaml` with 230 load scenarios, `data_dir: data/train_nk`, seed 4001 (new), `k: 5`, `outage_count_probabilities: [0, .35, .2, .15, .15, .15]` (datakit mixes k in one run). 5056 scenarios (4 PF failures), ids contiguous 0..5055. Outages per scenario k=1..5: 1860 / 1036 / 752 / 719 / 689; 23.7% of outages are generators. Violating scenarios 58.2% (36 / 56 / 70 / 81 / 84% by k). Checks in `results/genco_delta/train_nk_checks.txt`.
Disjoint from the test sets: closest load vectors differ by at least 14 MW at some bus, no exact match. Caveat (data_D section 2): the first 30 load scenarios share the global load level with stress n*; noise, dispatch and outages are independent.

## Splits and store
`scripts/make_splits_nk.py` -> `splits_nk`: train 4550 (207 load scenarios), val 506 (23), test = val copy. Base-case store: 230/230 load scenarios, no split id missing.

## Smoke
1 epoch at 10% of batches: 80 s, val loss 5.9e-3; stress n3 eval 3.15 iterations to 1e-6 MW vs base 3.17.

## Run
`scripts/run_genco_delta_nk.sh`, PID 7102 (training child 7118), log `results/genco_delta/run_nk.log`, per-run logs in `results/genco_delta/logs/`. Tags `residual_nk_vm_va_flows_lr3e-4_s{0,1,42}`, sets paper n1 n2 n3 n5 with the residual study's roots and per-seed splits. Resumable.
Incident: the Mac went into deep idle sleep on battery, training crawled (19 steps in about 2 h) and seed 0 epoch 1 restarted slowly. `caffeinate -dimsu -w 7102` was attached; epoch 1 then finished and speed is normal. Not restarted; seed 0 `wall_seconds` includes the stall. Expected about 50 min per seed.

## Re-aggregate and compare
`source .venv/bin/activate; python -W ignore scripts/eval_genco_delta_residual.py --aggregate --out_dir results/genco_delta/eval --prefix ""`
Rows `released`, `residual_vm_va_flows_lr3e-4`, `residual_nk_vm_va_flows_lr3e-4` in `results/genco_delta/residual_summary_by_arm.md`: n3 and n5 first (regression to fix), then paper and n1 (gain to keep).

## Confound
The N-1 model trained on the seed's published split (4724 / 4416 / 4790 scenarios for seeds 0 / 1 / 42), the N-k model on 4550, so sizes are similar but the scenarios are new and 63% multi-outage; validation (checkpoint selection) is N-k as well. A gain mixes outage order with different data; a matched control would be an N-1-only set of 4550 fresh scenarios.
