# Confirmatory test protocol (written before any confirmatory data was generated)

Purpose: test the frozen screening rule on IEEE 118 data that did not exist when the rule was frozen. Nothing below
may be re-tuned after the data are seen. Results are reported whatever they are, including any miss, any false alarm,
any exceedance above nominal and any loss of the iteration advantage; they are not to be dropped, re-run with other
settings, or replaced by a favourable subset.

## 1. Frozen rule

Source of the constants: `results/exp02/big118_policies.csv` (sha256 781aada01f3717d5ebbd3b509890309c0e96404f33975c48a47c75b73dad3e3f),
calibrated on ALL 100 load scenarios of `data/calib/n1` (exp02 run tag `calib_n1_base`, parquet sha256 in section 6).
Score of a load scenario = max over its scenarios and PQ-bus/branch elements of |error| / ||dx||^2 after step 1.
Escalation constant c (screening accepts after step k when every element margin exceeds c * ||dx||^2):

| Prior | rule | c (exact) |
|---|---|---|
| GENCO (setpoint enforced) | max | 2.1783875271941735 |
| GENCO | conf0.10 | 0.235791782658666 |
| GENCO | conf0.05 | 0.3796815414586852 |
| base case (reference = our own converged NR) | max | 39.81159460297254 |
| base case | conf0.10 | 20.92768418226285 |
| base case | conf0.05 | 34.437769088921264 |

Other frozen settings: overload threshold tau = 1.01 (max(|S_f|,|S_t|)/rate_a); voltage tolerance eps_v = 1 mV (1e-3 p.u.);
sequential policy takes up to 3 Newton steps (one LU each) and escalates to full NR if the test has not passed; tiered
policy (1 step, then NR) is also reported. GENCO = released IEEE 118 Base checkpoint, seed 0, `best_model_state_dict.pt`
(sha256 in section 6). Analysis code: `scripts/analyze_exp02.py --xcal`, `scripts/exp02_screening.py`, `src/genco_certified/*`
(hashes in section 6); none is modified for this test. The confirmatory run must reproduce the six constants above to
the printed precision (they are recomputed from `calib_n1_base` by the same code); the confirm sets are NOT used for calibration.

## 2. Sets

Four new datakit sets, IEEE 118 (`case118_ieee`), 100 load scenarios x 22 topology variants each, `elements: [branch, gen]`, random N-k,
k = 1, 2, 3, 5. Configs are copies of `configs/datakit/case118_stress100_n{k}.yaml` (diff: seed, data_dir, agg_profile,
num_processes 6 -> 4; everything else identical).

| Set | config | seed | data_dir |
|---|---|---|---|
| confirm N-1 | `configs/datakit/case118_confirm_n1.yaml` | 5101 | `data/confirm/n1` |
| confirm N-2 | `..._n2.yaml` | 5102 | `data/confirm/n2` |
| confirm N-3 | `..._n3.yaml` | 5103 | `data/confirm/n3` |
| confirm N-5 | `..._n5.yaml` | 5105 | `data/confirm/n5` |

Seeds never used before (used: 2027-2031, 3101-3103, 3105, 3111, 3201, 4001). Load profile: calib and stress100 take the first 100 hours of the
packaged `default` aggregate profile (min-max scaled to [0.6u, u]); the confirm sets use the packaged ERCOT 2024 East profile
(`agg_profile: ercot_load_act_hr_2024_east`, first 100 hours), a different trajectory (correlation -0.49 with the default's first 100 hours after min-max
scaling, selected from the packaged profiles by its range 0.23-0.61 of [l,u], similar to the default's 0.10-0.59, and its mean 0.40 vs 0.30; chosen before
any generation or model evaluation, no results were consulted). Per-bus +-20% noise, dispatch (cost permutation), admittance perturbation and outage draws are
independent through the new seed. Splits: `splits_all` (test = every scenario), as in `scripts/make_splits.py`.
Evaluation on each set uses the released Base seed 0 checkpoint, `scripts/exp02_screening.py --data_root data/confirm/n{k} --split splits_all --tag confirm_n{k}_base`.

## 3. Analysis (fixed)

`python scripts/analyze_exp02.py --xcal --cal_tag calib_n1_base --eval_tags confirm_n1_base confirm_n2_base confirm_n3_base confirm_n5_base --labels "C N-1" "C N-2" "C N-3" "C N-5" --prefix confirm --out results/confirm --log results/confirm/policy_confirm.log`
(2000 cluster-bootstrap resamples over load scenarios). The calibration set is not among the evaluation sets, so the frozen c is applied to every confirm set.

## 4. Metrics reported (every cell, all three rules, tiered and sequential, both priors)

1. Element and contingency misses and false alarms (a miss = a violating element/contingency that the screen accepts as safe; a false alarm = a non-violating
   one flagged), per set and pooled over the four sets, with Clopper-Pearson upper bounds (one-sided and two-sided 95%) at element, contingency and load-scenario level.
2. LU factorisations per contingency, both priors (GENCO sequential, base-case sequential), 95% cluster CIs, tier-1 acceptance share.
3. Exceedance rate: share of confirm load scenarios whose step-1 score exceeds c, against the nominal alpha (0.10, 0.05, ~1/(n+1) for max), with Clopper-Pearson CIs.
4. Newton iterations to 1e-2, 1e-4 and 1e-6 MW from flat, DC, base-case and GENCO starts; GENCO minus base-case paired difference with cluster-bootstrap CI and fewer/more shares.
5. Equal-cost ladder (no calibration) as in NUMBERS 5.6.
6. Properties of the new data: violating share of scenarios, counts of violating elements/contingencies, dropped scenarios, max load/rate, and total-demand comparison with calib/stress100.

Primary endpoints: pooled element and contingency misses of GENCO sequential under conf0.10, conf0.05 and max, and the GENCO-minus-base paired difference at 1e-6 MW.
The load-scenario level count is a conservative cluster-level view; the exchangeability argument does not extend to a new load profile, so exceedance above nominal is a finding, not a failure to hide.

## 5. Out of scope

No re-tuning of c, tau, eps_v, steps, priors or rules; no change of checkpoint; no selection among profiles or seeds after seeing results. Deviations, if any, are listed in `results/notes/confirm_Z.md`.

## 6. Hashes of frozen artifacts

```
32f7d4858cfe900a497419ce56ef920893786af14141864935ca6c096afcee8f  results/exp02/calib_n1_base_branches.parquet
57d121b2852c1fca87e6c82bac7585f50b9311c41f0a5859ac2354a39950f70b  results/exp02/calib_n1_base_buses.parquet
24546a0b2b49cf813985a9c01af5f64199d0e06dab886f5a2b64f745f004dd26  results/exp02/calib_n1_base_scenarios.parquet
781aada01f3717d5ebbd3b509890309c0e96404f33975c48a47c75b73dad3e3f  results/exp02/big118_policies.csv
03a0081efe39b9a7fcf9c63abe7cbbd0045a10ddc5a56984b52d1983c3ba1421  scripts/analyze_exp02.py
79389dc2b3749ef26cb1e56254a48ddbf3a1f9d2a5f492e89eee9dc3fdb230c3  scripts/exp02_screening.py
77ee3ce27ec7ac8688f0271a44f53a09ff237b22bd9a363aade5105a0e5d53d2  scripts/make_splits.py
77355ba0f1690ae3b078d81dc7e91ba01af2e0bcd0eca79e3249d0c7c636671f  src/genco_certified/__init__.py
ccf3ba30bc6a67a9fdf50fe919b7afd92a000e7e7d4e31fd6a3a9f9c749c4f20  src/genco_certified/acpf.py
8a87559ff204c9c6851a8146977fb7dd1d9bdc05fa1fe430d54a3c6bc9d6a0d6  src/genco_certified/basecase_store.py
7c986f4f437e28fef2e6050ecc1ff360fb09dc2e9490e0f091eee0a9b9b72ffe  src/genco_certified/fastpf.py
2dfb54291a2f047b10947ebe31684535a03e5f4e00f6c8105bb63c4e2ab3b178  src/genco_certified/genco_residual.py
ad78b730ff6618507b6c2aa333106cbac105ff842a4aba397ae42805daa55f91  src/genco_certified/graphkit_plugin.py
0c47cca780037e2714083331e141928ebe2fb7c9b8cbb229223dca46eececc2e  src/genco_certified/loading.py
eb12586750e3314d5aa38bff11c4bd7e379453025b2502d7525445b832ebb685  src/genco_certified/tracing.py
3ea2f552b190541fe08e0e27604daa5f122232af411f8bfd9bf355a557cab065  checkpoints/genco-pf-datakit/case118_ieee/base/seed0/best_model_state_dict.pt
01c93104fc47fbe4f7492d2926ad60fe7395f10f77f7788b2d71504ff6a68411  checkpoints/genco-pf-datakit/case118_ieee/base/seed0/normalizer_stats.pt
f8cc885cf63ba40d6fb580d6f16d57b34114666e4d1b0ac2d4a5a32c487c26dc  configs/datakit/case118_confirm_n1.yaml
c5de094566e27063978667a7107a2f331e6dd889da0b81570207cc650546c374  configs/datakit/case118_confirm_n2.yaml
9477e826953de4ecc9fbfd971914b9ada58640273c0c52a5a009506b621e932e  configs/datakit/case118_confirm_n3.yaml
11b88253b037970692fd800aea65ffbe59622654122f0835f2e85d9f162dace3  configs/datakit/case118_confirm_n5.yaml
04fe7efa344b23eb884ffc44747034ad4387e089c29930dde7d1c45b0ae91214  external/gridfm-datakit/gridfm_datakit/load_profiles/ercot_load_act_hr_2024_east.csv
```

---
Footer: sha256 of the text above this line (excluding this footer, from the first character through the closing code fence and newline): 027cf172823fdb3e3b0d8eac592d3b80018cbb50b558352b04d0a1803551aa42
Written: 2026-10-25 00:15:08 EDT (before any file under data/confirm existed)
