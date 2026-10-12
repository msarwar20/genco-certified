# WS0-A: screening pipeline corrections (exp02 v3)

Date: 2026-10-11. Scope: PLAN WS0.1, 0.2, 0.6 (DC part), 0.7, WS1.1, 1.3, 2.3 (conformal, using the paper set as calibration set), 2.4.
Outputs: `results/policy_v3.log` (all tables), `results/exp02/v3_*.csv`, `results/exp02/pareto_<tag>.csv`, `results/validation/recertify.json`, `results/exp02/run_v3.log`.
Reproduce: the four `exp02_screening.py` commands in `scripts/run_all.sh` (paper, stress n1/n2/n3/n5, GOC 2000 small/heldout), then `python scripts/analyze_exp02.py --v3`, then `python -m pytest -q tests` (18 passed).

## 1. What changed

`src/genco_certified/acpf.py` (functions added only; existing ones untouched)
- `newton_trajectory`: NR that records every iterate and dx, takes at least `n_steps` steps; its iteration count equals `newton_pf`'s.
- `iterations_to_tol`: steps until max mismatch first falls to a tolerance (-1 if never).
- `branch_key_values`, `dc_loading_from_raw`, `dc_loading_from_raw_rows`: datakit's stored PowerModels DC loading, `max(|pf_dc|,|pt_dc|)/rate_a` over in-service branches, merged per undirected (min bus, max bus) key by max. It asserts that the key set equals the graphkit-derived problem's key set. The assertion held for every scenario of every set, GOC 2000 included.

`scripts/exp02_screening.py` (rewritten; every old column is bit-identical to the old parquet except `load_dc`, checked on the paper set)
- New element columns, branches `load_*` and PQ-bus `vm_*`: `g1,g2,g3` (GENCO phys + k steps), `b1,b2,b3` (base case + k steps), `nr` (our converged NR from the GENCO start), `dc` (datakit DC), `dc_ours` (old DC). `step1`/`base1` are kept as aliases of `g1`/`b1`.
- New scenario columns: `step_g{1,2,3}_l2`, `step_b{1,2,3}_l2`; `nri_{flat,base,genco,dc}_e{2,4,6}` (NR iterations to 1e-2, 1e-4, 1e-6 MW); `recert_raw_max`, `recert_gk_max`, `nr_state_err_max`.

`scripts/analyze_exp02.py` (rewritten)
- `lu_per_scenario` / `tiered_lu`: LU accounting fix.
- `conformal_quantile`, `choose_c`, `grouped_folds`, `cross_c`, `kfold_c`: grouped calibration.
- `run_policy` implements the tiered rule (`ks=(1,)`) and the sequential rule (`ks=(1,2,3)`) for both priors. `Screen` and `RawSet` hold counts and margins per scenario. `tune_threshold` does the DC F-beta tuning.
- `--v3` writes the full analysis. The old CLI (`--tag/--tau/--cal_tag/--eps_v/--seed`) still works, because `make_figures.py` parses its table (9 rows parsed). It now charges own NR counts, and its default half/half split is by load scenario. Its c0/tier-0 was dropped (0% accepted in every old log).

`tests/test_screening.py` (new, 12 tests): LU accounting on a toy table; sequential accept/escalate logic; disjoint grouped folds (also through `kfold_c`); conformal quantile edge cases and coverage; DC-threshold tuning; DC-from-raw equals `max(|pf_dc|,|pt_dc|)/rate_a` for a contingency scenario of `data/pf/case118_ieee` (outaged branch dropped, parallel circuits merged); `newton_trajectory` equals `newton_pf`.

## 2. Bugs and caveats found

1. The LU bug is as reported (section 3).
2. Label-noise floor inflates c for the base-case prior. graphkit stores features as float32, so datakit's solution seen through graphkit's problem has a median mismatch of 1.9e-3 MW (max 7.5e-3 MW; the 118-bus raw-parquet problem is exact to 1e-6 MW). When the base case is already almost exact (||dx||^2 ~ 1e-9), |error vs datakit|/||dx||^2 explodes: c_base(max) = 3575 against 42.1 when the error is measured against our own NR. GENCO's c is unaffected (0.3465 vs 0.3506). This handicaps the classical policy, so v3 reports both (log sections 3 and 3d, `v3_policies_grouped_refnr.csv`).
3. At tau = 1.00, or at eps_v = 0, the labels are noise. Exact converged NR "misses" 21 to 83 elements at tau = 1.00 and 25 elements with ~419 false alarms at eps_v = 0, because OPF-binding limits sit within 1e-6 of the threshold. Sensitivity rows must be read net of this floor (included as the "exact NR" rows).
4. Conformal alpha = 0.05 is identical to the max rule whenever 19 <= n <= 38 calibration load scenarios (29 here). With 9 GOC load scenarios and K = 5, conformal needs rank > n, so c = inf and everything escalates.
5. Bookkeeping: one batch of exp02 runs was started twice by mistake and the two overlapped. Both write identical deterministic outputs, and `run_v3.log` was de-duplicated by hand. The final parquets were verified against the old files.

## 3. Key numbers

Defaults: tau = 1.01, eps_v = 1 mV (PQ buses only), K = 5 folds grouped by `load_idx`, element level = branches + PQ buses, "cont" = scenario-level. Load scenarios per set: paper 29, stress 30 each, GOC 9. Scenarios: 638 / 660 / 659 / 660 / 660 / 198.

### 3.1 Corrected LU under the OLD calibration (c = 0.261 from all of stress N-1, in-sample at N-1)

| set | tier-1 share | LU old | LU corrected | reviewer | classical old -> corrected |
|---|---|---|---|---|---|
| N-1 | 61.4% | 1.429 | 1.494 | 1.49 | 2.858 -> 2.868 |
| N-2 | 56.3% | 1.498 | 1.569 | 1.57 | 3.076 -> 3.076 |
| N-3 | 47.0% | 1.633 | 1.718 | 1.72 | 3.173 -> 3.173 |
| N-5 | 30.3% | 1.974 | 2.076 | 2.08 | 3.317 -> 3.317 |

### 3.2 Cross-set: c from the paper set (29 load scenarios), evaluated on ALL scenarios (stress N-1 included)

Paper-set constants: GENCO c = 0.3465 (max, also conformal 0.05) and 0.1257 (conformal 0.10); base case c = 3575 (max) and 88.7 (conformal 0.10).
LU is per scenario. Element misses, false alarms and contingency misses are 0 in every cell of this table. NR to 1e-6 MW costs 2.11/2.14/2.19/2.40 from GENCO and 2.88/3.08/3.17/3.32 from the base case (N-1/2/3/5).

| set | rule | GENCO tiered: tier-1 / LU | GENCO sequential LU | base tiered: tier-1 / LU | base sequential LU | exceed, load scen. (GENCO) | exceed, scen. (GENCO) |
|---|---|---|---|---|---|---|---|
| N-1 | max | 52.3% / 1.585 | 1.477 | 0.9% / 2.871 | 2.692 | 0/30 | 0 |
| N-2 | max | 51.6% / 1.619 | 1.487 | 0% / 3.076 | 2.974 | 4/30 = 13.3% | 0.61% |
| N-3 | max | 41.2% / 1.777 | 1.588 | 0% / 3.173 | 3.083 | 0/30 | 0 |
| N-5 | max | 25.8% / 2.124 | 1.755 | 0% / 3.317 | 3.217 | 1/30 = 3.3% | 0.15% |
| N-1 | conf 0.10 | 77.6% / 1.318 | 1.224 | 7.1% / 2.808 | 2.338 | 2/30 = 6.7% | 0.61% |
| N-2 | conf 0.10 | 69.2% / 1.428 | 1.311 | 0.6% / 3.070 | 2.654 | 10/30 = 33.3% | 2.9% |
| N-3 | conf 0.10 | 63.5% / 1.538 | 1.365 | 0% / 3.173 | 2.836 | 7/30 = 23.3% | 1.1% |
| N-5 | conf 0.10 | 44.9% / 1.912 | 1.559 | 0% / 3.317 | 3.061 | 4/30 = 13.3% | 0.76% |

The base-case rows use the label-noise-inflated c (see section 2, item 2). With the error measured against our own NR (c_base = 42.1, conformal 26.2), base sequential LU is 2.22/2.54/2.75/3.00 (max rule) and 2.14/2.48/2.68/2.95 (conf 0.10) for N-1/2/3/5. GENCO is unchanged to within 0.01. Misses stay 0. The conclusion does not change.

The in-distribution exceedance (stress N-1 at alpha = 0.10, 6.7%) is within nominal. Out of distribution (N-2, N-3, N-5) it is 13 to 33% against a nominal 10%. The exchangeability assumption fails across contingency order. There were still no misses, because the accepted margins are far larger than the realised errors.

### 3.3 Single-set 5-fold CV grouped by load scenario (pooled out-of-fold, fold seed 0; the 20-draw median is in brackets)

| set | GENCO tiered LU (max) | GENCO seq LU (max) | GENCO tiered LU (conf .10) | GENCO seq LU (conf .10) | base seq LU (max) | exceed load scen. at conf .10, 20-draw median [min,max] |
|---|---|---|---|---|---|---|
| paper | 1.545 (1.56) | 1.403 (1.42) | 1.342 (1.36) | 1.215 (1.23) | 2.611 | 0.069 [0.069, 0.172] |
| stress N-1 | 1.427 (1.47) | 1.342 (1.36) | 1.292 (1.31) | 1.211 (1.22) | 2.626 | 0.067 [0.067, 0.100] |
| stress N-2 | 1.777 | 1.640 | 1.678 | 1.545 | 2.407 | 0.067 [0.067, 0.100] |
| stress N-3 | 1.736 | 1.549 | 1.676 | 1.492 | 2.367 | 0.067 [0.067, 0.100] |
| stress N-5 | 2.205 | 1.833 | 2.026 | 1.661 | 2.592 | 0.067 [0.067, 0.100] |
| GOC 2000 | 3.227 (0% accepted) | 2.005 | inf, all escalate (3.227) | 3.227 | 2.071 | max rule 0.111 |

Misses: 0 in all 72 (set x policy x rule) cells over 20 fold draws, and in all 120 rows of the grouped table. Single-split c is fragile: a random half of the paper load scenarios gave c = 0.167 (legacy CLI) against 0.3465 for all 29.

### 3.4 Equal-cost ladder (tau = 1.01, eps_v = 1 mV; no calibration; element misses / false alarms (contingency misses / false alarms))

| method (LU) | paper | stress N-1 | N-2 | N-3 | N-5 | GOC 2000 |
|---|---|---|---|---|---|---|
| violating elements | 484 | 435 | 714 | 950 | 1763 | 61 |
| GENCO, no step (0) | 31/178 (6/59) | 20/129 (12/82) | 49/92 (15/51) | 52/97 (14/40) | 182/106 (13/16) | 46/237 (4/162) |
| GENCO + 1 (1) | 1/0 (0/0) | 1/0 (0/0) | 0/0 (0/0) | 1/0 (0/0) | 6/0 (0/0) | 4/130 (0/60) |
| GENCO + 2 (2) | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| base + 1 (1) | 33/6 (4/0) | 21/12 (3/0) | 37/20 (3/3) | 60/32 (6/4) | 129/46 (5/5) | 16/0 (0/0) |
| base + 2 (2) | 0/0 | 1/0 | 0/1 | 2/0 | 6/0 | 0/0 |
| base + 3 (3) | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |

The reviewer's base + 2 numbers reproduce exactly (1/435 at N-1, 6/1763 at N-5). At equal element misses, GENCO + 1 (1 LU) matches base + 2 (2 LU). GENCO + 2 is exact everywhere. At 2000 buses GENCO + 1 has fewer misses than base + 1 (4 vs 16) but 130 false alarms (60 at scenario level) against 0; both are exact at 2 steps.

### 3.5 NR iterations by tolerance (mean; all runs converged)

| set | flat 1e-2/1e-4/1e-6 | DC (ours) | base | GENCO |
|---|---|---|---|---|
| paper | 3.01/4.00/4.00 | 3.00/3.72/3.97 | 2.04/2.49/2.88 | 1.72/2.01/2.15 |
| N-1 | 3.01/4.00/4.00 | 3.00/3.80/4.00 | 2.03/2.50/2.88 | 1.69/2.00/2.11 |
| N-2 | 3.03/4.00/4.01 | 3.00/3.90/4.00 | 2.19/2.74/3.08 | 1.60/2.01/2.14 |
| N-3 | 3.03/4.00/4.01 | 3.00/3.88/4.00 | 2.29/2.90/3.17 | 1.73/2.01/2.19 |
| N-5 | 3.08/4.01/4.03 | 3.01/3.76/4.00 | 2.49/3.08/3.32 | 1.89/2.08/2.40 |
| GOC 2000 | 5.00/5.22/6.00 | 5.00/5.22/5.89 | 2.08/2.40/2.95 | 2.56/3.00/3.23 |

### 3.6 DC baseline (datakit PowerModels DC; branch overloads only; thresholds tuned on the paper set, 5-fold CV for the paper set and GOC)

- Datakit's DC differs only slightly from our old DC: element recall 0.440 vs 0.427 on the paper set. The old DC was not the problem. Tuning and metric level are.
- Tuned thresholds on the paper set: F3 (beta = 3) gives 0.875, recall >= 0.99 gives 0.75.
- Element level, F3 threshold: recall 0.87 / 0.95 / 0.92 / 0.97 / 0.91 (paper, N-1, N-2, N-3, N-5), precision 0.15 to 0.46, flag rate 1.9 to 2.8% of branches.
- Element level, recall >= 0.99 threshold: recall 0.98 / 1.00 / 1.00 / 1.00 / 0.99, precision 0.09 to 0.30, flag rate 3.6 to 4.6%.
- Contingency level, untuned (threshold = tau): recall of violating scenarios 0.81 / 0.87 / 0.74 / 0.89 / 0.87, flagging 58 to 80% of scenarios.
- Contingency level, tuned: recall 1.00, but the tuned DC flags 97 to 100% of scenarios, so it carries no contingency-level information on this data.
- GOC 2000 (own 5-fold CV): F3 gives recall 0.55 at precision 1.0; recall >= 0.99 gives recall 0.91 with 201 false alarms. Contingency recall is 0.75 for all variants.

### 3.7 Sensitivity (GENCO tiered, c from conformal 0.10 on the paper set; full grid in `v3_sensitivity.csv`)

Stress N-1 tier-1 share / LU, by tau (rows) and eps_v (columns):

| tau | eps_v = 0 | 0.5 mV | 1 mV | 2 mV |
|---|---|---|---|---|
| 1.000 | 3.6% / 2.07 | 23% / 1.88 | 26% / 1.85 | 28% / 1.83 |
| 1.005 | 8.8% / 2.02 | 62% / 1.48 | 75% / 1.35 | 84% / 1.25 |
| 1.010 | 9.5% / 2.01 | 64% / 1.47 | 78% / 1.32 | 87% / 1.20 |
| 1.020 | 9.8% / 2.01 | 65% / 1.46 | 79% / 1.30 | 89% / 1.18 |

- For stress N-5 the same grid gives 9% to 62%, with LU from 2.28 down to 1.71.
- The classical tiered rule accepts at most 9% of scenarios at N-1 and 0% at N-5.
- Every miss at eps_v = 0 or tau = 1.00 equals the exact-NR floor (stress N-1: 25 elements at eps_v = 0; 26 at tau = 1.00 with eps_v > 0). With eps_v >= 0.5 mV and tau >= 1.005 the floor is 0 and all policies have 0 misses.
- GENCO + 1 step at tau = 1.01, eps_v = 1 mV: element misses 1 (N-1), 6 (N-5). Base + 1 step: 21 (N-1), 129 (N-5).
- The tier-1 share collapses at eps_v = 0 because many PQ buses sit exactly on their limits (zero margin), so no verification can pass. This is a property of the OPF data, not of the method.

### 3.8 Pareto in-sample oracle (`pareto_<tag>.csv`, 53 values of c from 0 to inf)

The smallest c from which every larger c has zero element misses:

| prior | c at zero misses | LU there |
|---|---|---|
| GENCO tiered | 0.004 to 0.0126 (0 at N-2) | 1.00 to 1.29 |
| base tiered | 0.13 to 0.32 | 1.76 to 3.25 |

This is 27 to 87 times below the calibrated max-rule c = 0.3465. The calibrated rule is safe but conservative. The oracle is in-sample and is not a usable rule. GOC 2000: GENCO needs c = 1e-4 (LU 2.99 tiered, 1.88 sequential); base needs c = 0.016 (LU 1.24 tiered, 1.11 sequential).

### 3.9 Is the step-1 constant valid at step 2? (error vs our NR, c from the paper set, `v3_step_k_ratio_check.csv`)

The fraction of load scenarios exceeding the step-1 constant at step 2 is 0 to 6.7% (GENCO: N-3 3.3%, N-5 6.7%; base: stress N-1 6.7%). Step 3 is at the NR noise floor and is not assessable.

### 3.10 Tier-2 answer from our own NR vs datakit's stored solution

| set | max abs diff in loading | max abs diff in Vm |
|---|---|---|
| 118-bus sets | 0.9e-5 to 1.4e-5 | 1.2e-6 to 2.6e-6 |
| GOC 2000 | 3.5e-5 | 1.9e-6 |

All NR runs converged to 1e-6 MW from flat, DC, base and GENCO starts.

### 3.11 Recertification (`results/validation/recertify.json`)

Max |mismatch| at datakit's stored solution on a problem built from the raw parquet (MW/MVAr):

| set | max | median |
|---|---|---|
| paper | 3.8e-7 | 1.4e-9 |
| stress N-1 | 9.9e-7 | 4.3e-10 |
| stress N-2 | 8.8e-7 | 1.8e-9 |
| stress N-3 | 9.7e-7 | 4.3e-9 |
| stress N-5 | 9.8e-7 | 5.7e-9 |
| GOC 2000 | 1.5e-5 | 4.1e-7 |

No scenario exceeds 1e-3 MW. The same solution on graphkit's float32-derived problem has median mismatch 1.8e-3 to 2.2e-3 MW and max 5.5e-3 to 7.5e-3 MW (GOC: median 1.8e-2, max 3.1e-2). This is a data-pipeline precision floor.

## 4. Not done / limits

- No bootstrap CIs (WS2.2) and no separate calibration set beyond the 29-load-scenario paper set (WS2.3). Quantiles at n = 29 are coarse.
- Chord / fast-decoupled with factor reuse (WS1.2) is not implemented. LU counts remain a weak cost proxy; inference cost is excluded.
- DC LU cost is not reported in v3 (the old 0.5 LU convention is kept only in the legacy CLI).
- `scripts/run_all.sh` and `make_figures.py` still call the legacy CLI. They should switch to `analyze_exp02.py --v3` and the CSVs (not my files).
