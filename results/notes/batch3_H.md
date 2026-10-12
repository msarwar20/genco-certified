# Batch 3 (agent H): big IEEE 118 sets, GOC 500, GOC 2000 held-out, per-bus bias correction

Written 2026-10-12. Scope: PLAN WS2.3, WS2.5, WS3.1, WS3.2 and the open per-bus bias-correction question from the embedding probe.
All heavy jobs ran with `nice -n 15`, one at a time (drivers `scripts/run_batch3.sh`, `run_batch3b.sh`, `run_batch3c.sh`; per-job logs in `results/batch3_logs/`). Nothing under `src/genco_certified/`, `scripts/finetune*`, `scripts/eval_genco_delta*`, `scripts/run_genco_delta*.sh` was modified. No commits; RESEARCH_LOG and docs untouched.

Conventions are those of `ws0_A_screening.md` and `ws1_F_baselines_stats.md`: tau = 1.01, eps_v = 1 mV, element = branch or PQ bus, "LU" = factorizations per contingency (an escalated scenario is charged its own NR count), c is the constant in "|error| <= c ||dx||^2" calibrated over LOAD scenarios (rule `max`, or split-conformal `conf0.10` / `conf0.05`), base-case policies use c measured against our own NR (the stronger baseline). CIs are 95% cluster bootstraps over load scenarios (2000 resamples). "UB" is a Clopper-Pearson 95% upper bound (two-sided unless marked 1-sided). Released checkpoints, seed 0.

## 0. Verdicts

1. **Big IEEE 118 sets (task 1): the safety result survives a 3x larger calibration set and 3x larger test sets, and the exceedance problem of the 29-load-scenario calibration goes away.**
   Calibrated on `calib/n1` (100 load scenarios, 2,200 scenarios) and tested on `stress100/n{1,2,3,5}` (100 load scenarios each) plus the paper set, all three rules (max, conformal 0.10, 0.05) and both priors have **0 element misses, 0 element false alarms, 0 contingency misses and 0 contingency false alarms in every cell**. Pooled over the six evaluation sets: 0 / 18,030 violating elements (95% UB 0.02%), 0 / 6,531 violating contingencies (UB 0.05%), 0 / 529 load scenarios (UB 0.56%; the load levels are shared between sets, see the caveat in 1.1, so treat that last number as optimistic).
   GENCO's sequential policy costs fewer factorizations than the base case's: with conformal 0.10, 1.37 / 1.47 / 1.53 / 1.65 LU against 2.14 / 2.51 / 2.73 / 2.96 on N-1 / N-2 / N-3 / N-5 (35 to 44% fewer; CIs far apart); with the max rule 1.84 to 1.96 against 2.24 to 3.02. Honest calibration costs roughly 0.1 to 0.15 LU more than the 29-load-scenario calibration did (1.22 / 1.31 / 1.37 / 1.56 on the older 30-load-scenario stress sets; different test sets, so indicative), and the max-rule constant is 6x larger (2.18 instead of 0.35) because 100 load scenarios sample the heavy tail.
   Bound exceedance (share of evaluation load scenarios whose step-1 score exceeds c) at conformal alpha = 0.10, GENCO: 1% (N-1), 7% (N-2), 2% (N-3), 5% (N-5), 3% (paper); every value is at or below the nominal 10%. The earlier calibration on the 29-load-scenario paper set exceeded in 6.7 / 33 / 23 / 13%. With the max rule there are 0 exceedances on all four stress100 sets. The guarantee is therefore *not* violated here, but it is also not tight (N-1: 1% against 10%): the calibration set has a heavier score tail than the test sets (p90 0.235 against 0.14 to 0.18).
   Equal-cost ladder on the large sets: GENCO + 1 step misses 0 / 2 / 11 / 18 elements (N-1 / N-2 / N-3 / N-5) against 2 / 3 / 8 / 19 for base + 2 steps (twice the cost) and 114 / 213 / 371 / 519 for base + 1 step. GENCO + 2 steps is exact on all five sets. So "a GENCO prediction is worth about one Newton iteration" holds on the 3x larger sets. NR iterations to 1e-4 MW: GENCO 2.00 to 2.07, base 2.51 to 3.08, flat 4.0.
2. **GOC 500 (task 2): GENCO is only marginally better than the base case as a Newton prior, and loses clearly on wall-clock.**
   Full-NR iterations GENCO minus base: -0.10 [-0.14, -0.07] to 1e-2 MW, -0.33 [-0.46, -0.18] to 1e-4, -0.04 [-0.13, 0.04] to 1e-6 (a tie); Small and Tiny: -0.11 / -0.27 / +0.01 and -0.11 / -0.30 / +0.01. The base-case start needs 2.11 / 2.60 / 2.96 iterations, flat 4.00 / 4.79 / 5.00. The sequential screening policy is cheaper from GENCO (1.81 to 1.98 LU with conformal 0.10 and 1.85 to 1.98 with max, against 2.18 and 2.25 from the base case; 0 misses, 0 false alarms; but 29 load scenarios only and tier-1 acceptance of just 2 to 19%). The equal-cost ladder does not separate the starts (GENCO + 1: 0 / 133 elements missed; base + 1: 1 / 133, with 83 / 30 / 34 (Base / Small / Tiny) against 24 false alarms; 92 violating contingencies). With factor reuse the GENCO start saves 11 to 12% of triangular solves for the compensated chord and 5 to 13% for full Newton, and is *worse* than the base case for fast-decoupled (+0.44 iterations to 1e-4 MW, 7% more solves). GENCO inference on the CPU takes 166 to 179 ms per scenario, 20x a complete full-NR solve from the base case (8.6 ms) and at least 590x the largest break-even. The ratio of physical to reported slack mismatch is 19x [12, 29] (Base), 13x (Small) and 8.6x (Tiny), smaller than on IEEE 118 (35 to 60x); locality holds (worst bus adjacent to the slack in 94 to 99%).
3. **GOC 2000 held-out, 879 scenarios / 40 load scenarios (task 3): the old 198-scenario conclusion is confirmed with 4.4x the scenarios and 4.4x the load scenarios.** The base case remains the better Newton prior: GENCO needs +0.57 [0.40, 0.72] / +0.63 [0.59, 0.67] / +0.22 [0.12, 0.35] more iterations to 1e-2 / 1e-4 / 1e-6 MW (Small; Tiny is the same within 0.03), 4% of scenarios fewer and 25 to 64% more. The 681 scenarios that are *not* in the old set (31 new load scenarios) give the same answer (+0.59 / +0.64 / +0.21), so this is a replication, not just a larger sample of the same 9 operating points. The best-of-two rule is -0.01 [-0.03, 0.00] against the base case (picks GENCO in 11%). The sequential screening policy is nominally cheaper from GENCO (2.00 against 2.16 / 2.26 LU) but only because GENCO + 2 steps always passes verification (tier-1 acceptance 0%), and the equal-cost ladder shows what the 1-step answer looks like: 4 elements missed against 18 for the base case, at the price of 941 against 9 false alarms (310 against 0 contingencies). Zero misses and zero false alarms for every calibrated policy.
4. **Per-bus bias correction (task 4): it does nothing. GENCO's signed errors are not systematic per bus.** Fitted on 2,200 N-1 scenarios and tested on the paper set and on N-5, a per-bus mean (or per-bus linear in load level, or in the bus's own injection) leaves the state error, the Newton iteration counts and the equal-cost 1-step misses unchanged within noise; the injection-based variant is worse. In-sample, a per-bus constant could remove only 0.3% (Base) of the squared error, 0.6% / 3.5% (Small, Vm / Va) and 2.1% / 1.5% (Tiny). What the embedding probe found (|dVm| is bus-specific, cross-fitted R^2 = 0.45 for |e| from the per-bus mean |e|) is a property of the error *magnitude*; the sign is not consistent across operating points. 63 to 75% of the squared angle error is a per-scenario common-mode offset, which no per-bus correction can remove.

## 1. Big IEEE 118 sets (task 1)

### 1.1 What was run

`scripts/exp02_screening.py` (v3 columns), Base seed 0, on `data/calib/n1` (tag `calib_n1_base`, 2,200 scenarios, 100 load scenarios) and `data/stress100/n{1,2,3,5}` (tags `stress100_n{k}_base`, 2,200 / 2,199 / 2,199 / 2,194 scenarios, 100 load scenarios each), split `splits_all`. Wall time 2 to 4 minutes per set. Analysis: new mode `scripts/analyze_exp02.py --xcal` (see section 5):

```
python scripts/analyze_exp02.py --xcal --cal_tag calib_n1_base \
  --eval_tags calib_n1_base stress100_n1_base stress100_n2_base stress100_n3_base stress100_n5_base case118_ieee_base_seed0 \
  --labels "calib N-1 (CV)" "S100 N-1" "S100 N-2" "S100 N-3" "S100 N-5" "paper N-1" --prefix big118
```

c is calibrated on ALL 100 load scenarios of `calib_n1_base` and applied to every other set; the calibration set itself is evaluated by 5-fold cross-fitting grouped by load scenario (the in-distribution check; pooled out-of-fold). Outputs: `results/policy_big118.log`, `results/exp02/big118_*.csv`.

**Caveat (data_D): calibration and test sets share the global load trajectory.** datakit takes the first n hours of the aggregate load profile and uses the seed only for the per-bus noise, the dispatch and the outage draws. Load scenario i therefore has nearly the same total demand in every set (calib against stress100/n1: correlation 0.95, mean absolute relative difference 1.8%). Outages, noise and dispatch are independent, but the exchangeability argument for split-conformal holds only conditionally on this one 100-hour trajectory, and the 529 "load scenarios" pooled over the six sets are about 100 distinct operating levels, not 529 independent draws. I did not check whether the published paper set shares the trajectory. The right fix is a calibration set from a different `agg_profile`.

### 1.2 Calibration constants and score distributions

Calibration on `calib_n1_base` (100 load scenarios), step-1 score = max over a load scenario's elements of |error| / ||dx||^2:

| prior | max | conf 0.10 | conf 0.05 |
|---|---|---|---|
| GENCO | 2.178 | 0.2358 | 0.3797 |
| base case (error against own NR) | 39.81 | 20.93 | 34.44 |

(The 29-load-scenario paper-set constants were 0.3465 / 0.1257 / 0.3465 for GENCO and 42.1 / 26.2 / 42.1 for the base case.) The score distribution per set (percentiles over load scenarios; GENCO error against datakit, base case against own NR):

| set | load scen. | GENCO p50 | GENCO p90 | GENCO max | base p50 | base p90 | base max |
|---|---|---|---|---|---|---|---|
| calib N-1 (CV) | 100 | 0.0413 | 0.235 | 2.18 | 6.8 | 18.9 | 39.8 |
| S100 N-1 | 100 | 0.0432 | 0.144 | 0.329 | 6.89 | 20.3 | 44 |
| S100 N-2 | 100 | 0.0484 | 0.176 | 0.464 | 2.63 | 6.59 | 12.4 |
| S100 N-3 | 100 | 0.0581 | 0.179 | 0.387 | 1.52 | 2.96 | 6.39 |
| S100 N-5 | 100 | 0.0617 | 0.179 | 0.328 | 0.714 | 1.52 | 3.31 |
| paper N-1 | 29 | 0.0342 | 0.112 | 0.347 | 8.85 | 24.9 | 42.1 |

### 1.3 Cost: sequential policy, LU per contingency [95% CI]

Misses and false alarms are all zero in these rows (next table). "tier-1 G/B" = share accepted after one step, GENCO / base case. The calib row is 5-fold cross-fitted, all other rows use the full calibration set.

| set | c GENCO (max) | GENCO seq LU (max) | base seq LU (max) | tier-1 G/B (max) | c GENCO (conf 0.10) | GENCO seq LU (conf 0.10) | base seq LU (conf 0.10) | tier-1 G/B (conf 0.10) |
|---|---|---|---|---|---|---|---|---|
| S100 N-1 | 2.18 | 1.842 [1.795, 1.886] | 2.235 [2.200, 2.268] | 0.16 / 0.10 | 0.236 | 1.369 [1.299, 1.439] | 2.135 [2.101, 2.168] | 0.63 / 0.15 |
| S100 N-2 | 2.18 | 1.864 [1.826, 1.902] | 2.601 [2.577, 2.626] | 0.14 / 0.01 | 0.236 | 1.465 [1.400, 1.531] | 2.509 [2.483, 2.535] | 0.54 / 0.01 |
| S100 N-3 | 2.18 | 1.915 [1.889, 1.940] | 2.815 [2.788, 2.843] | 0.10 / 0.00 | 0.236 | 1.531 [1.477, 1.586] | 2.726 [2.697, 2.755] | 0.47 / 0.00 |
| S100 N-5 | 2.18 | 1.964 [1.946, 1.982] | 3.016 [2.996, 3.036] | 0.06 / 0.00 | 0.236 | 1.651 [1.612, 1.689] | 2.961 [2.939, 2.983] | 0.36 / 0.00 |
| paper N-1 | 2.18 | 1.771 [1.671, 1.870] | 2.219 [2.163, 2.274] | 0.23 / 0.10 | 0.236 | 1.367 [1.234, 1.514] | 2.089 [2.027, 2.149] | 0.63 / 0.17 |
| calib N-1 (CV) | 2.18 | 1.763 [1.699, 1.820] | 2.219 [2.187, 2.248] | 0.24 / 0.10 | 0.236 | 1.345 [1.274, 1.417] | 2.126 [2.095, 2.153] | 0.66 / 0.14 |

### 1.4 Safety: misses, false alarms, bounds

Element misses, element false alarms, contingency misses and contingency false alarms are **0 in every one of the 3 rules x 3 policies (GENCO tiered, GENCO sequential, base-case sequential) x 6 sets = 54 cells.** The misses are measured against datakit's AC solution as labels, escalated scenarios are decided by our own converged NR. Upper bounds (identical for all policies because the counts are all zero):

| set | violating elements / contingencies | element UB | contingency UB | load scenarios with a miss | load-scenario UB |
|---|---|---|---|---|---|
| calib N-1 (CV) | 1581 / 746 | 0.23% | 0.49% | 0/100 | 3.6% |
| S100 N-1 | 1624 / 786 | 0.23% | 0.47% | 0/100 | 3.6% |
| S100 N-2 | 3210 / 1303 | 0.11% | 0.28% | 0/100 | 3.6% |
| S100 N-3 | 4514 / 1555 | 0.08% | 0.24% | 0/100 | 3.6% |
| S100 N-5 | 6617 / 1916 | 0.06% | 0.19% | 0/100 | 3.6% |
| paper N-1 | 484 / 225 | 0.76% | 1.63% | 0/29 | 11.9% |

Pooled over the six sets (and compared with the uncalibrated ladder; 1-sided 95% Clopper-Pearson; the load-scenario row is optimistic, see the caveat in 1.1):

| rule / method | el misses | el 1-sided UB | cont misses | cont UB | load scen. with miss | load UB |
|---|---|---|---|---|---|---|
| GENCO sequential (max rule; conf rules identical) | 0/18030 | 0.017% | 0/6531 | 0.046% | 0/529 | 0.56% |
| base case sequential (max rule; conf rules identical) | 0/18030 | 0.017% | 0/6531 | 0.046% | 0/529 | 0.56% |
| GENCO + 1 step | 32/18030 | 0.238% | 3/6531 | 0.119% | 24/529 | 6.32% |
| GENCO + 2 steps | 0/18030 | 0.017% | 0/6531 | 0.046% | 0/529 | 0.56% |
| Base case + 1 step | 1369/18030 | 7.925% | 119/6531 | 2.119% | 389/529 | 76.68% |
| Base case + 2 steps | 33/18030 | 0.245% | 1/6531 | 0.073% | 31/529 | 7.83% |

### 1.5 Bound exceedance rate per set

Share of evaluation load scenarios whose step-1 score exceeds the calibrated c [95% Clopper-Pearson CI]. For the conformal rules the nominal level is alpha (0.10 or 0.05); for max it is about 1/(n+1). The calibration set itself is cross-fitted. Under exchangeability the conformal exceedance is at most alpha in expectation; N-2 to N-5 are not exchangeable with the N-1 calibration set, so a value at or below nominal there is empirical evidence, not a guarantee.

| set | prior | max (nominal ~1/(n+1)) | conf 0.10 | conf 0.05 |
|---|---|---|---|---|
| calib N-1 (CV) | GENCO | 1/100 = 1.0% [0.0%, 5.4%] | 10/100 = 10.0% [4.9%, 17.6%] | 6/100 = 6.0% [2.2%, 12.6%] |
| calib N-1 (CV) | base case | 1/100 = 1.0% [0.0%, 5.4%] | 10/100 = 10.0% [4.9%, 17.6%] | 4/100 = 4.0% [1.1%, 9.9%] |
| S100 N-1 | GENCO | 0/100 = 0.0% [0.0%, 3.6%] | 1/100 = 1.0% [0.0%, 5.4%] | 0/100 = 0.0% [0.0%, 3.6%] |
| S100 N-1 | base case | 2/100 = 2.0% [0.2%, 7.0%] | 10/100 = 10.0% [4.9%, 17.6%] | 2/100 = 2.0% [0.2%, 7.0%] |
| S100 N-2 | GENCO | 0/100 = 0.0% [0.0%, 3.6%] | 7/100 = 7.0% [2.9%, 13.9%] | 3/100 = 3.0% [0.6%, 8.5%] |
| S100 N-2 | base case | 0/100 = 0.0% [0.0%, 3.6%] | 0/100 = 0.0% [0.0%, 3.6%] | 0/100 = 0.0% [0.0%, 3.6%] |
| S100 N-3 | GENCO | 0/100 = 0.0% [0.0%, 3.6%] | 2/100 = 2.0% [0.2%, 7.0%] | 1/100 = 1.0% [0.0%, 5.4%] |
| S100 N-3 | base case | 0/100 = 0.0% [0.0%, 3.6%] | 0/100 = 0.0% [0.0%, 3.6%] | 0/100 = 0.0% [0.0%, 3.6%] |
| S100 N-5 | GENCO | 0/100 = 0.0% [0.0%, 3.6%] | 5/100 = 5.0% [1.6%, 11.3%] | 0/100 = 0.0% [0.0%, 3.6%] |
| S100 N-5 | base case | 0/100 = 0.0% [0.0%, 3.6%] | 0/100 = 0.0% [0.0%, 3.6%] | 0/100 = 0.0% [0.0%, 3.6%] |
| paper N-1 | GENCO | 0/29 = 0.0% [0.0%, 11.9%] | 1/29 = 3.4% [0.1%, 17.8%] | 0/29 = 0.0% [0.0%, 11.9%] |
| paper N-1 | base case | 1/29 = 3.4% [0.1%, 17.8%] | 4/29 = 13.8% [3.9%, 31.7%] | 1/29 = 3.4% [0.1%, 17.8%] |

Reading: no set exceeds its nominal level beyond sampling noise. The N-2 value for GENCO at alpha = 0.10 (7/100, CI up to 13.9%) is the closest to the limit. Base-case policies are at 10% on N-1 (the calibration is in-distribution there) and at 0% on N-2 to N-5 (the base-case score *falls* with outage order: median 6.9, 2.6, 1.5, 0.7). The paper set, calibrated on `calib/n1`, has 1/29 (GENCO) and 4/29 (base case, 13.8% [3.9, 31.7]) at alpha = 0.10.

### 1.6 Equal-cost ladder (no calibration)

| set | method | LU | el misses | miss rate 95% CP UB (2-sided) | el FA | cont misses | cont FA | load scen. with miss | load UB |
|---|---|---|---|---|---|---|---|---|---|
| S100 N-1 | GENCO + 1 step | 1 | 0/1624 | 0.23% | 0 | 0/786 | 0 | 0/100 | 3.6% |
| S100 N-1 | GENCO + 2 steps | 2 | 0/1624 | 0.23% | 0 | 0/786 | 0 | 0/100 | 3.6% |
| S100 N-1 | Base case + 1 step | 1 | 114/1624 | 8.37% | 46 | 12/786 | 8 | 53/100 | 63.1% |
| S100 N-1 | Base case + 2 steps | 2 | 2/1624 | 0.44% | 0 | 0/786 | 0 | 2/100 | 7.0% |
| S100 N-2 | GENCO + 1 step | 1 | 2/3210 | 0.22% | 0 | 0/1303 | 0 | 2/100 | 7.0% |
| S100 N-2 | GENCO + 2 steps | 2 | 0/3210 | 0.11% | 0 | 0/1303 | 0 | 0/100 | 3.6% |
| S100 N-2 | Base case + 1 step | 1 | 213/3210 | 7.55% | 51 | 32/1303 | 5 | 77/100 | 84.8% |
| S100 N-2 | Base case + 2 steps | 2 | 3/3210 | 0.27% | 0 | 0/1303 | 0 | 3/100 | 8.5% |
| S100 N-3 | GENCO + 1 step | 1 | 11/4514 | 0.44% | 2 | 2/1555 | 1 | 8/100 | 15.2% |
| S100 N-3 | GENCO + 2 steps | 2 | 0/4514 | 0.08% | 0 | 0/1555 | 0 | 0/100 | 3.6% |
| S100 N-3 | Base case + 1 step | 1 | 371/4514 | 9.06% | 117 | 29/1555 | 5 | 88/100 | 93.6% |
| S100 N-3 | Base case + 2 steps | 2 | 8/4514 | 0.35% | 0 | 0/1555 | 0 | 8/100 | 15.2% |
| S100 N-5 | GENCO + 1 step | 1 | 18/6617 | 0.43% | 2 | 1/1916 | 0 | 13/100 | 21.2% |
| S100 N-5 | GENCO + 2 steps | 2 | 0/6617 | 0.06% | 0 | 0/1916 | 0 | 0/100 | 3.6% |
| S100 N-5 | Base case + 1 step | 1 | 519/6617 | 8.52% | 205 | 21/1916 | 2 | 97/100 | 99.4% |
| S100 N-5 | Base case + 2 steps | 2 | 19/6617 | 0.45% | 3 | 1/1916 | 0 | 17/100 | 25.8% |
| paper N-1 | GENCO + 1 step | 1 | 1/484 | 1.15% | 0 | 0/225 | 0 | 1/29 | 17.8% |
| paper N-1 | GENCO + 2 steps | 2 | 0/484 | 0.76% | 0 | 0/225 | 0 | 0/29 | 11.9% |
| paper N-1 | Base case + 1 step | 1 | 33/484 | 9.44% | 6 | 4/225 | 0 | 21/29 | 87.3% |
| paper N-1 | Base case + 2 steps | 2 | 0/484 | 0.76% | 0 | 0/225 | 0 | 0/29 | 11.9% |

### 1.7 Newton iterations by tolerance (mean [95% CI]; paired difference GENCO minus base-case start)

| set | tol (MW) | flat | base case | GENCO | GENCO - base (paired) | GENCO fewer/more |
|---|---|---|---|---|---|---|
| S100 N-1 | 1e-2 | 3.01 | 2.05 [2.03, 2.07] | 1.69 [1.62, 1.76] | -0.37 [-0.43, -0.30] | 38% / 4% |
| S100 N-1 | 1e-4 | 4.00 | 2.51 [2.48, 2.53] | 2.00 [2.00, 2.00] | -0.50 [-0.53, -0.48] | 49% / 0% |
| S100 N-1 | 1e-6 | 4.00 | 2.89 [2.87, 2.91] | 2.10 [2.06, 2.14] | -0.79 [-0.83, -0.75] | 75% / 1% |
| S100 N-2 | 1e-2 | 3.03 | 2.22 [2.20, 2.24] | 1.72 [1.66, 1.77] | -0.50 [-0.56, -0.44] | 46% / 0% |
| S100 N-2 | 1e-4 | 4.00 | 2.79 [2.77, 2.81] | 2.01 [2.01, 2.02] | -0.78 [-0.80, -0.76] | 75% / 0% |
| S100 N-2 | 1e-6 | 4.01 | 3.10 [3.09, 3.12] | 2.17 [2.13, 2.22] | -0.93 [-0.97, -0.88] | 85% / 0% |
| S100 N-3 | 1e-2 | 3.05 | 2.33 [2.31, 2.35] | 1.75 [1.71, 1.80] | -0.58 [-0.63, -0.53] | 52% / 0% |
| S100 N-3 | 1e-4 | 4.00 | 2.95 [2.93, 2.97] | 2.03 [2.02, 2.04] | -0.92 [-0.94, -0.90] | 86% / 0% |
| S100 N-3 | 1e-6 | 4.02 | 3.20 [3.18, 3.21] | 2.21 [2.18, 2.24] | -0.99 [-1.02, -0.95] | 88% / 0% |
| S100 N-5 | 1e-2 | 3.08 | 2.51 [2.48, 2.53] | 1.85 [1.82, 1.88] | -0.65 [-0.69, -0.62] | 59% / 0% |
| S100 N-5 | 1e-4 | 4.01 | 3.08 [3.07, 3.10] | 2.07 [2.06, 2.09] | -1.01 [-1.02, -0.99] | 94% / 0% |
| S100 N-5 | 1e-6 | 4.03 | 3.34 [3.31, 3.36] | 2.38 [2.34, 2.41] | -0.96 [-1.00, -0.92] | 83% / 0% |
| paper N-1 | 1e-2 | 3.01 | 2.04 [2.00, 2.07] | 1.72 [1.58, 1.85] | -0.32 [-0.45, -0.19] | 35% / 4% |
| paper N-1 | 1e-4 | 4.00 | 2.49 [2.45, 2.53] | 2.00 [2.00, 2.01] | -0.49 [-0.53, -0.45] | 48% / 1% |
| paper N-1 | 1e-6 | 4.00 | 2.88 [2.83, 2.93] | 2.15 [2.06, 2.25] | -0.73 [-0.83, -0.63] | 71% / 2% |

### 1.8 Reading and limits

- The LU advantage comes from the verification test: GENCO's calibrated constant is 18x (max rule) to 90x (conformal 0.10) smaller than the base case's, which lets the margin test pass after one or two steps, while the base-case prior mostly needs three or the NR count. I did not decompose this further. At N-5 the gap is largest (1.65 against 2.96) because the base-case start degrades with outage order and GENCO's does not.
- The max-rule constant depends on the calibration sample (heavy tail): 0.35 from 29 load scenarios, 2.18 from 100. Expect it to keep growing slowly with more load scenarios; the conformal constant is far more stable (0.126 to 0.236).
- The calibration set is the only source of independence for the conformal statement and it shares the load trajectory with the test sets (1.1).
- Everything here is Base seed 0 on its own paper split; seeds 1/42 and Small/Tiny were not rerun on these sets.

## 2. GOC 500, organizers' example grid (task 2)

Data `data/pf/case500_goc`, split `splits_paper_test` (seed 0): 638 scenarios, 29 load scenarios. Released Base / Small / Tiny seed 0. Jobs: exp01, exp02 (v3), exp05, exp06 for all three sizes; exp07 for Base (88 s). Analysis: `--xcal --cal_tag none` (5-fold cross-fitting grouped by load scenario inside each model's paper set; there is no separate calibration set for GOC 500). 133 violating elements (branches + PQ buses) in 92 of 638 contingencies, so every count below is small.

```
python scripts/analyze_exp02.py --xcal --cal_tag none --eval_tags case500_goc_base_seed0 case500_goc_small_seed0 case500_goc_tiny_seed0 \
  --labels "GOC500 Base" "GOC500 Small" "GOC500 Tiny" --prefix goc500
```

### 2.1 Does GENCO beat the base case as a Newton prior?

Marginally, and only at loose tolerances. NR iterations (mean [95% CI]; GENCO minus base case is paired):

| set | tol (MW) | flat | base case | GENCO | GENCO - base (paired) | GENCO fewer/more |
|---|---|---|---|---|---|---|
| GOC500 Base | 1e-2 | 4.00 | 2.11 [2.07, 2.15] | 2.00 [2.00, 2.01] | -0.10 [-0.14, -0.07] | 16% / 6% |
| GOC500 Base | 1e-4 | 4.79 | 2.60 [2.56, 2.64] | 2.27 [2.15, 2.42] | -0.33 [-0.46, -0.18] | 42% / 10% |
| GOC500 Base | 1e-6 | 5.00 | 2.96 [2.92, 3.00] | 2.92 [2.82, 3.00] | -0.04 [-0.13, 0.04] | 16% / 13% |
| GOC500 Small | 1e-2 | 4.00 | 2.11 [2.07, 2.15] | 2.00 [2.00, 2.00] | -0.11 [-0.15, -0.07] | 16% / 6% |
| GOC500 Small | 1e-4 | 4.79 | 2.60 [2.56, 2.64] | 2.33 [2.20, 2.47] | -0.27 [-0.41, -0.13] | 39% / 13% |
| GOC500 Small | 1e-6 | 5.00 | 2.96 [2.92, 3.00] | 2.97 [2.92, 3.00] | 0.01 [-0.06, 0.07] | 13% / 14% |
| GOC500 Tiny | 1e-2 | 4.00 | 2.11 [2.07, 2.15] | 2.00 [2.00, 2.00] | -0.11 [-0.15, -0.07] | 16% / 6% |
| GOC500 Tiny | 1e-4 | 4.79 | 2.60 [2.56, 2.64] | 2.30 [2.16, 2.44] | -0.30 [-0.45, -0.15] | 41% / 11% |
| GOC500 Tiny | 1e-6 | 5.00 | 2.96 [2.92, 3.00] | 2.97 [2.93, 3.00] | 0.01 [-0.05, 0.06] | 12% / 13% |

The best-of-two rule (exp05: choose the start with the smaller mismatch, no extra factorization) is 2.87 [2.80, 2.92] for Base against 2.96 for the base case alone (-0.10 [-0.14, -0.06]); GENCO is picked in 40% (Base) and 52% (Small, Tiny). Full output of exp01 / exp05 / exp06 for the three sizes:

```
====================================================================================================
GOC 500 Base: 638 scenarios, 29 load scenarios (exp02 tag case500_goc_base_seed0)
  exp01  slack Vm error (p.u.) median 0.00777 | reported max mismatch (MW) median 4.53 | physical 86.3
         NR iterations to 1e-6 MW: flat 5.00 [5.00, 5.00] | DC 5.00 [5.00, 5.00] | GENCO 2.92 [2.82, 3.00] | converged 1.000 / 1.000 / 1.000
         state error l2 median 0.527 -> after one Newton step 0.00297 (ratio of medians 177x); step-norm estimate vs error Spearman 1.000
       res_raw_med  res_phys_med  err_vm  err_va  nr_iters
layer                                                     
0         1.88e+03       2.2e+03  0.0854   0.282         4
1              514           600  0.0491   0.257         4
2              136           578  0.0195   0.224      3.14
3              158           409  0.0123   0.185      3.13
4             47.6           171 0.00702   0.135      3.06
5             84.3           140 0.00417   0.123      3.05
6             30.3          92.9 0.00313   0.111      3.05
7             17.3          80.6 0.00271  0.0976      3.04
8               12          78.6 0.00266  0.0932      3.04
9             8.03          84.6 0.00256  0.0701         3
10            4.61          86.1 0.00214   0.046      2.94
11            4.53          86.3 0.00187  0.0425      2.92
  exp05  NR iterations to 1e-6 MW: always-GENCO 2.92 [2.82, 3.00] | always-base 2.96 [2.92, 3.00] | best-of-two 2.87 [2.80, 2.92] (picks GENCO 40%) | oracle 2.80 [2.71, 2.87]
         paired: GENCO - base -0.04 [-0.13, 0.04] | best-of-two - base -0.10 [-0.14, -0.06] | best-of-two - GENCO -0.06 [-0.10, -0.00] | GENCO fewer/more than base: 16% / 13%
  exp06  slack Vm error median 0.00777 [0.00565, 0.00958] | max mismatch reported 4.53 [3.72, 6] -> physical 86.3 [62.8, 121] | ratio 19 [12, 29.3]
         worst bus adjacent to slack: full 0.991 [0.981, 0.997] | constrained 0.994 [0.986, 1] | share of |dS| in 1-hop (full) 0.2 [0.128, 0.22] | constrained 0.248 [0.151, 0.278] | enrichment 99.6 [64, 110]
         re-anchoring worse in 100% (shift, state error) / 100% (scale)
====================================================================================================
GOC 500 Small: 638 scenarios, 29 load scenarios (exp02 tag case500_goc_small_seed0)
  exp01  slack Vm error (p.u.) median 0.00552 | reported max mismatch (MW) median 4.68 | physical 61
         NR iterations to 1e-6 MW: flat 5.00 [5.00, 5.00] | DC 5.00 [5.00, 5.00] | GENCO 2.97 [2.92, 3.00] | converged 1.000 / 1.000 / 1.000
         state error l2 median 0.584 -> after one Newton step 0.0039 (ratio of medians 150x); step-norm estimate vs error Spearman 1.000
  exp05  NR iterations to 1e-6 MW: always-GENCO 2.97 [2.92, 3.00] | always-base 2.96 [2.92, 3.00] | best-of-two 2.87 [2.82, 2.92] (picks GENCO 52%) | oracle 2.83 [2.78, 2.87]
         paired: GENCO - base 0.01 [-0.06, 0.07] | best-of-two - base -0.09 [-0.14, -0.05] | best-of-two - GENCO -0.10 [-0.13, -0.06] | GENCO fewer/more than base: 13% / 14%
  exp06  slack Vm error median 0.00552 [0.00429, 0.00773] | max mismatch reported 4.68 [4.04, 5.3] -> physical 61 [45.5, 80.7] | ratio 13 [9.34, 18.8]
         worst bus adjacent to slack: full 0.959 [0.9, 0.995] | constrained 0.969 [0.915, 0.998] | share of |dS| in 1-hop (full) 0.146 [0.126, 0.187] | constrained 0.175 [0.15, 0.224] | enrichment 73 [63, 93.3]
         re-anchoring worse in 100% (shift, state error) / 100% (scale)
====================================================================================================
GOC 500 Tiny: 638 scenarios, 29 load scenarios (exp02 tag case500_goc_tiny_seed0)
  exp01  slack Vm error (p.u.) median 0.00519 | reported max mismatch (MW) median 6.77 | physical 58.4
         NR iterations to 1e-6 MW: flat 5.00 [5.00, 5.00] | DC 5.00 [5.00, 5.00] | GENCO 2.97 [2.93, 3.00] | converged 1.000 / 1.000 / 1.000
         state error l2 median 0.406 -> after one Newton step 0.0023 (ratio of medians 177x); step-norm estimate vs error Spearman 1.000
  exp05  NR iterations to 1e-6 MW: always-GENCO 2.97 [2.93, 3.00] | always-base 2.96 [2.92, 3.00] | best-of-two 2.88 [2.85, 2.92] (picks GENCO 52%) | oracle 2.84 [2.80, 2.87]
         paired: GENCO - base 0.01 [-0.05, 0.06] | best-of-two - base -0.08 [-0.11, -0.05] | best-of-two - GENCO -0.09 [-0.12, -0.06] | GENCO fewer/more than base: 12% / 13%
  exp06  slack Vm error median 0.00519 [0.00414, 0.00809] | max mismatch reported 6.77 [5.68, 7.68] -> physical 58.4 [41.7, 93.2] | ratio 8.63 [6.26, 14.2]
         worst bus adjacent to slack: full 0.939 [0.857, 0.992] | constrained 0.942 [0.864, 0.994] | share of |dS| in 1-hop (full) 0.124 [0.0869, 0.158] | constrained 0.145 [0.1, 0.187] | enrichment 61.7 [43.3, 78.6]
         re-anchoring worse in 97% (shift, state error) / 97% (scale)

```

### 2.2 Screening policy and equal-cost ladder

Sequential policy cost (5-fold cross-fitted c; zero misses and zero false alarms for every rule, policy and size; element UB 2.74%, contingency UB 3.93%, load-scenario UB 11.9%):

| set | c GENCO (max) | GENCO seq LU (max) | base seq LU (max) | tier-1 G/B (max) | c GENCO (conf 0.10) | GENCO seq LU (conf 0.10) | base seq LU (conf 0.10) | tier-1 G/B (conf 0.10) |
|---|---|---|---|---|---|---|---|---|
| GOC500 Base | 0.032 | 1.846 [1.724, 1.953] | 2.245 [2.205, 2.284] | 0.16 / 0.03 | 0.0137 | 1.812 [1.676, 1.926] | 2.179 [2.143, 2.218] | 0.19 / 0.06 |
| GOC500 Small | 0.0354 | 1.944 [1.865, 1.997] | 2.245 [2.205, 2.284] | 0.06 / 0.03 | 0.0125 | 1.851 [1.743, 1.939] | 2.179 [2.143, 2.218] | 0.15 / 0.06 |
| GOC500 Tiny | 0.12 | 1.981 [1.947, 2.003] | 2.245 [2.205, 2.284] | 0.02 / 0.03 | 0.0759 | 1.980 [1.942, 2.003] | 2.179 [2.143, 2.218] | 0.02 / 0.06 |

Exceedance of the cross-fitted c:

| set | prior | max (nominal ~1/(n+1)) | conf 0.10 | conf 0.05 |
|---|---|---|---|---|
| GOC500 Base | GENCO | 1/29 = 3.4% [0.1%, 17.8%] | 2/29 = 6.9% [0.8%, 22.8%] | 1/29 = 3.4% [0.1%, 17.8%] |
| GOC500 Base | base case | 1/29 = 3.4% [0.1%, 17.8%] | 2/29 = 6.9% [0.8%, 22.8%] | 1/29 = 3.4% [0.1%, 17.8%] |
| GOC500 Small | GENCO | 1/29 = 3.4% [0.1%, 17.8%] | 3/29 = 10.3% [2.2%, 27.4%] | 1/29 = 3.4% [0.1%, 17.8%] |
| GOC500 Small | base case | 1/29 = 3.4% [0.1%, 17.8%] | 2/29 = 6.9% [0.8%, 22.8%] | 1/29 = 3.4% [0.1%, 17.8%] |
| GOC500 Tiny | GENCO | 2/29 = 6.9% [0.8%, 22.8%] | 2/29 = 6.9% [0.8%, 22.8%] | 2/29 = 6.9% [0.8%, 22.8%] |
| GOC500 Tiny | base case | 1/29 = 3.4% [0.1%, 17.8%] | 2/29 = 6.9% [0.8%, 22.8%] | 1/29 = 3.4% [0.1%, 17.8%] |

Equal-cost ladder:

| set | method | LU | el misses | miss rate 95% CP UB (2-sided) | el FA | cont misses | cont FA | load scen. with miss | load UB |
|---|---|---|---|---|---|---|---|---|---|
| GOC500 Base | GENCO + 1 step | 1 | 0/133 | 2.74% | 83 | 0/92 | 72 | 0/29 | 11.9% |
| GOC500 Base | GENCO + 2 steps | 2 | 0/133 | 2.74% | 0 | 0/92 | 0 | 0/29 | 11.9% |
| GOC500 Base | Base case + 1 step | 1 | 1/133 | 4.12% | 24 | 1/92 | 11 | 1/29 | 17.8% |
| GOC500 Base | Base case + 2 steps | 2 | 0/133 | 2.74% | 0 | 0/92 | 0 | 0/29 | 11.9% |
| GOC500 Base | GENCO as published | 0 | 19/133 | 21.41% | 313 | 6/92 | 173 | 14/29 | 67.5% |
| GOC500 Small | GENCO + 1 step | 1 | 0/133 | 2.74% | 30 | 0/92 | 26 | 0/29 | 11.9% |
| GOC500 Small | GENCO + 2 steps | 2 | 0/133 | 2.74% | 0 | 0/92 | 0 | 0/29 | 11.9% |
| GOC500 Small | Base case + 1 step | 1 | 1/133 | 4.12% | 24 | 1/92 | 11 | 1/29 | 17.8% |
| GOC500 Small | Base case + 2 steps | 2 | 0/133 | 2.74% | 0 | 0/92 | 0 | 0/29 | 11.9% |
| GOC500 Small | GENCO as published | 0 | 17/133 | 19.67% | 260 | 6/92 | 133 | 12/29 | 61.1% |
| GOC500 Tiny | GENCO + 1 step | 1 | 0/133 | 2.74% | 34 | 0/92 | 29 | 0/29 | 11.9% |
| GOC500 Tiny | GENCO + 2 steps | 2 | 0/133 | 2.74% | 0 | 0/92 | 0 | 0/29 | 11.9% |
| GOC500 Tiny | Base case + 1 step | 1 | 1/133 | 4.12% | 24 | 1/92 | 11 | 1/29 | 17.8% |
| GOC500 Tiny | Base case + 2 steps | 2 | 0/133 | 2.74% | 0 | 0/92 | 0 | 0/29 | 11.9% |
| GOC500 Tiny | GENCO as published | 0 | 19/133 | 21.41% | 408 | 4/92 | 211 | 13/29 | 64.3% |

Pooled over the three sizes (same scenarios, so this is a count of tests, not independent evidence):

| rule / method | el misses | el 1-sided UB | cont misses | cont UB | load scen. with miss | load UB |
|---|---|---|---|---|---|---|
| GENCO sequential (max rule; conf rules identical) | 0/399 | 0.748% | 0/276 | 1.080% | 0/87 | 3.38% |
| base case sequential (max rule; conf rules identical) | 0/399 | 0.748% | 0/276 | 1.080% | 0/87 | 3.38% |
| GENCO + 1 step | 0/399 | 0.748% | 0/276 | 1.080% | 0/87 | 3.38% |
| GENCO + 2 steps | 0/399 | 0.748% | 0/276 | 1.080% | 0/87 | 3.38% |
| Base case + 1 step | 3/399 | 1.932% | 3/276 | 2.785% | 3/87 | 8.67% |
| Base case + 2 steps | 0/399 | 0.748% | 0/276 | 1.080% | 0/87 | 3.38% |

### 2.3 Factor-reuse solvers (exp07, Base seed 0)

```
====================================================================================================
exp07 GOC 500 Base seed 0: 638 contingencies, 29 load scenarios; mean outaged branches 0.77; with a PV->PQ change 66
  tolerance 0.01 MW (iterations / triangular-solve pairs incl. setup / factorizations per contingency; not converged within the cap)
    nr      flat   iters        4.00 [4.00, 4.00] | solves        4.00 [4.00, 4.00] | fact  4.00 | fail 0/638
    nr      base   iters        2.11 [2.07, 2.15] | solves        2.11 [2.07, 2.15] | fact  2.11 | fail 0/638
    nr      genco  iters        2.00 [2.00, 2.01] | solves        2.00 [2.00, 2.01] | fact  2.00 | fail 0/638
    chord   base   iters     15.10 [14.26, 15.99] | solves     15.10 [14.26, 15.99] | fact  0.10 | fail 81/638
    chord   genco  iters     11.75 [10.98, 12.55] | solves     11.75 [10.98, 12.55] | fact  0.10 | fail 36/638
    chordC  base   iters        3.11 [2.99, 3.24] | solves        6.07 [5.92, 6.23] | fact  0.10 | fail 0/638
    chordC  genco  iters        2.36 [2.28, 2.45] | solves        5.32 [5.20, 5.44] | fact  0.10 | fail 0/638
    fdlf    base   iters        3.49 [3.44, 3.54] | solves        9.17 [9.05, 9.29] | fact  0.10 | fail 0/638
    fdlf    genco  iters        3.78 [3.64, 3.93] | solves       9.76 [9.47, 10.05] | fact  0.10 | fail 0/638
  paired GENCO start - base start, tolerance 0.01 MW
    nr      d_iters     -0.10 [-0.14, -0.07] | d_solves     -0.10 [-0.14, -0.07] | solve saving      0.05 [0.03, 0.07] | GENCO fewer iters in 16%, more in 6% (n=638)
    chord   d_iters     -5.45 [-6.04, -4.92] | d_solves     -5.45 [-6.04, -4.92] | solve saving      0.36 [0.34, 0.38] | GENCO fewer iters in 81%, more in 7% (n=557)
    chordC  d_iters     -0.75 [-0.83, -0.68] | d_solves     -0.75 [-0.83, -0.68] | solve saving      0.12 [0.11, 0.13] | GENCO fewer iters in 57%, more in 7% (n=638)
    fdlf    d_iters        0.29 [0.15, 0.44] | d_solves        0.59 [0.31, 0.88] | solve saving   -0.06 [-0.10, -0.03] | GENCO fewer iters in 16%, more in 41% (n=638)
  tolerance 0.0001 MW (iterations / triangular-solve pairs incl. setup / factorizations per contingency; not converged within the cap)
    nr      flat   iters        4.79 [4.62, 4.93] | solves        4.79 [4.62, 4.93] | fact  4.79 | fail 0/638
    nr      base   iters        2.60 [2.56, 2.64] | solves        2.60 [2.56, 2.64] | fact  2.60 | fail 0/638
    nr      genco  iters        2.27 [2.15, 2.42] | solves        2.27 [2.15, 2.42] | fact  2.27 | fail 0/638
    chord   base   iters     17.49 [16.36, 18.65] | solves     17.49 [16.36, 18.65] | fact  0.10 | fail 149/638
    chord   genco  iters     16.10 [15.22, 16.99] | solves     16.10 [15.22, 16.99] | fact  0.10 | fail 97/638
    chordC  base   iters        4.57 [4.40, 4.75] | solves        7.53 [7.35, 7.72] | fact  0.10 | fail 0/638
    chordC  genco  iters        3.71 [3.57, 3.86] | solves        6.67 [6.50, 6.84] | fact  0.10 | fail 0/638
    fdlf    base   iters        5.29 [5.23, 5.37] | solves     12.78 [12.63, 12.93] | fact  0.10 | fail 0/638
    fdlf    genco  iters        5.73 [5.58, 5.88] | solves     13.65 [13.35, 13.95] | fact  0.10 | fail 0/638
  paired GENCO start - base start, tolerance 0.0001 MW
    nr      d_iters     -0.33 [-0.46, -0.18] | d_solves     -0.33 [-0.46, -0.18] | solve saving      0.13 [0.07, 0.18] | GENCO fewer iters in 42%, more in 10% (n=638)
    chord   d_iters     -4.12 [-4.61, -3.65] | d_solves     -4.12 [-4.61, -3.65] | solve saving      0.24 [0.22, 0.25] | GENCO fewer iters in 80%, more in 9% (n=489)
    chordC  d_iters     -0.86 [-0.95, -0.78] | d_solves     -0.86 [-0.95, -0.78] | solve saving      0.11 [0.10, 0.12] | GENCO fewer iters in 62%, more in 7% (n=638)
    fdlf    d_iters        0.44 [0.29, 0.59] | d_solves        0.87 [0.58, 1.18] | solve saving   -0.07 [-0.09, -0.04] | GENCO fewer iters in 10%, more in 48% (n=638)

```

Timing (provisional: `--timing --repeats 3 --max_loads 8`, 176 contingencies, load average 4.3 to 3.0 on 10 cores while a GPU job and another batch ran, python/scipy, `results/exp07/timing_case500_goc_base_seed0.json`). Per contingency to 1e-4 MW, base start against GENCO start (ms): full NR 8.59 against 8.67; compensated chord 2.70 against 2.52; fast-decoupled 1.51 against 1.51 (plain chord 3.45 against 3.03, with 41 and 30 of 176 not converged). Largest same-solver break-even: 0.28 ms (full NR, 1e-2), 0.20 ms (compensated chord), 0.02 ms (fast-decoupled); for full NR at 1e-4 MW it is negative (-0.09 ms). GENCO CPU inference on this grid is 179 ms per scenario at batch size 1 and 166 ms at batch size 16 (`scripts/bench_genco_cpu.py`, 4 threads, same loaded machine; `results/runtime/case500_goc_base_cpu.json`), which is at least 590x the largest same-solver break-even and about 9,000x the fast-decoupled one. No wall-clock benefit on CPU.

### 2.4 Verdict for GOC 500

- Newton prior: GENCO saves 0.1 iteration to 1e-2 and 0.3 to 1e-4 (4 to 13%), nothing to 1e-6. On IEEE 118 the saving is 0.5 to 0.7 iterations (20 to 25%). The advantage shrinks with grid size: 118 buses -0.73 [-0.83, -0.63] to 1e-6, 500 buses -0.04 [-0.13, 0.04], 2000 buses +0.22 [+0.12, +0.35].
- Equal-cost ladder: GENCO + 1 and base + 1 are both nearly exact on this grid (0 and 1 of 133 elements missed), so the ladder carries no information about the prior; GENCO + 1 produces more false alarms (83 / 30 / 34 against 24). Both are exact after 2 steps.
- Screening LU: GENCO is cheaper than the base case (1.81 to 1.98 against 2.18 to 2.25) with zero misses; but with 29 load scenarios the CI on the GENCO side is wide (Base: 1.68 to 1.93) and tier-1 acceptance is small.
- Slack gap: 19x [12, 29] (Base), 13x [9.3, 19] (Small), 8.6x [6.3, 14] (Tiny). Smaller than IEEE 118 (35 to 60x) and GOC 2000 (20 to 26x), the same locality (worst physical-mismatch bus adjacent to the slack in 99% / 96% / 94%, enrichment about 100x / 73x / 62x the uniform share). Reported mismatch (Base) is 4.5 MW against 86 MW physical. The layer dynamics are not geometric here: the median reported mismatch goes 1880, 514, 136, 158, 48, 84, 30, 17, 12, 8.0, 4.6, 4.5 MW over the 12 layers (two increases), and the state error after layer 0 is 0.28 rad (Va). One Newton step cuts the state error 177x (median l2 0.527 to 0.0030). Spearman(estimated step, error) is 1.000, as expected (dx = -e to first order).

## 3. GOC 2000 held-out (task 3)

`data/pf/case2000_goc`, split `splits_heldout` (seed 0; the new 879 scenarios / 40 load scenarios), released Small and Tiny seed 0. exp01 (batch 8), exp02 (v3), exp05, exp06 (batch 8). Tags `case2000_goc_{small,tiny}_seed0_h879` (the old 198-scenario run `case2000_goc_small_seed0` is untouched). The old 198 scenarios are a strict subset of the new 879 (9 of the 40 load scenarios). Analysis as for GOC 500 (`--cal_tag none`, 5-fold cross-fitting grouped by load scenario) with the old set added for comparison:

```
python scripts/analyze_exp02.py --xcal --cal_tag none --eval_tags case2000_goc_small_seed0_h879 case2000_goc_tiny_seed0_h879 case2000_goc_small_seed0 \
  --labels "GOC2000 Small (879)" "GOC2000 Tiny (879)" "GOC2000 Small (old 198)" --prefix goc2000h
```

Seed-0 Small and Tiny are the only released GOC 2000 sizes tested; `splits_heldout` leaks for the seed-42 checkpoint (84 of 839 outside-subset scenarios), so seed 42 was not run.

### 3.1 Newton iterations

| set | tol (MW) | flat | base case | GENCO | GENCO - base (paired) | GENCO fewer/more |
|---|---|---|---|---|---|---|
| GOC2000 Small (879) | 1e-2 | 6.20 | 2.06 [2.04, 2.09] | 2.63 [2.48, 2.77] | 0.57 [0.40, 0.72] | 4% / 59% |
| GOC2000 Small (879) | 1e-4 | 6.55 | 2.37 [2.33, 2.41] | 3.00 [3.00, 3.00] | 0.63 [0.59, 0.67] | 1% / 64% |
| GOC2000 Small (879) | 1e-6 | 7.06 | 2.93 [2.90, 2.96] | 3.15 [3.05, 3.28] | 0.22 [0.12, 0.35] | 4% / 25% |
| GOC2000 Tiny (879) | 1e-2 | 6.20 | 2.06 [2.04, 2.09] | 2.64 [2.49, 2.77] | 0.57 [0.41, 0.72] | 4% / 59% |
| GOC2000 Tiny (879) | 1e-4 | 6.55 | 2.37 [2.33, 2.41] | 3.00 [3.00, 3.00] | 0.63 [0.59, 0.67] | 1% / 64% |
| GOC2000 Tiny (879) | 1e-6 | 7.06 | 2.93 [2.90, 2.96] | 3.18 [3.08, 3.30] | 0.25 [0.13, 0.38] | 4% / 27% |
| GOC2000 Small (old 198) | 1e-2 | 5.01 | 2.08 [2.04, 2.12] | 2.56 [2.22, 2.89] | 0.48 [0.12, 0.84] | 5% / 53% |
| GOC2000 Small (old 198) | 1e-4 | 5.22 | 2.40 [2.32, 2.48] | 3.00 [3.00, 3.00] | 0.60 [0.52, 0.68] | 1% / 60% |
| GOC2000 Small (old 198) | 1e-6 | 6.00 | 2.95 [2.88, 3.01] | 3.23 [3.01, 3.56] | 0.28 [0.03, 0.57] | 3% / 28% |

Comparison with the old 198-scenario numbers in RESEARCH_LOG ("F8"): GENCO 3.23 against base 2.95 NR iterations to 1e-6 (+0.28 [0.03, 0.57]); on 879: 3.15 against 2.93 (+0.22 [0.12, 0.35]). The CI is now 2x to 4x tighter and the point estimate is unchanged. Restricted to the 681 scenarios outside the old set (31 new load scenarios): +0.59 [0.42, 0.76] / +0.64 [0.59, 0.68] / +0.21 [0.10, 0.35] iterations to 1e-2 / 1e-4 / 1e-6 MW.

exp01 / exp05 / exp06 output:

```
====================================================================================================
GOC 2000 Small (879): 879 scenarios, 40 load scenarios (exp02 tag case2000_goc_small_seed0_h879)
  exp01  slack Vm error (p.u.) median 0.0173 | reported max mismatch (MW) median 11 | physical 217
         NR iterations to 1e-6 MW: flat 7.06 [6.79, 7.34] | DC 6.81 [6.55, 7.06] | GENCO 3.15 [3.05, 3.28] | converged 0.999 / 0.999 / 1.000
         state error l2 median 4.77 -> after one Newton step 0.114 (ratio of medians 42x); step-norm estimate vs error Spearman 0.999
       res_raw_med  res_phys_med  err_vm  err_va  nr_iters
layer                                                     
0         3.91e+04      3.91e+04    1.09     2.3         1
1              794           794   0.139   0.494      4.11
2              415           429   0.136   0.439         4
3              331           375   0.128   0.418         4
4              234           292   0.114   0.383         4
5              126           273  0.0646   0.313         4
6             65.7           252  0.0342   0.272      3.73
7             41.5           247   0.016   0.227      3.36
8               23           234  0.0106   0.191      3.23
9             15.2           231  0.0103   0.179      3.23
10            11.7           224 0.00877   0.177       3.2
11              11           217 0.00826   0.182      3.15
  exp05  NR iterations to 1e-6 MW: always-GENCO 3.15 [3.05, 3.28] | always-base 2.93 [2.90, 2.96] | best-of-two 2.92 [2.89, 2.94] (picks GENCO 11%) | oracle 2.89 [2.86, 2.91]
         paired: GENCO - base 0.22 [0.12, 0.35] | best-of-two - base -0.01 [-0.03, -0.00] | best-of-two - GENCO -0.24 [-0.36, -0.14] | GENCO fewer/more than base: 4% / 25%
  exp06  slack Vm error median 0.0173 [0.0152, 0.0201] | max mismatch reported 11 [9.83, 12.9] -> physical 217 [188, 271] | ratio 19.7 [15.8, 25.7]
         worst bus adjacent to slack: full 0.95 [0.875, 1] | constrained 1 [1, 1] | share of |dS| in 1-hop (full) 0.0728 [0.0638, 0.0912] | constrained 0.0893 [0.0796, 0.107] | enrichment 146 [127, 182]
         re-anchoring worse in 100% (shift, state error) / 100% (scale)
====================================================================================================
GOC 2000 Tiny (879): 879 scenarios, 40 load scenarios (exp02 tag case2000_goc_tiny_seed0_h879)
  exp01  slack Vm error (p.u.) median 0.0227 | reported max mismatch (MW) median 11.1 | physical 289
         NR iterations to 1e-6 MW: flat 7.06 [6.79, 7.34] | DC 6.81 [6.55, 7.06] | GENCO 3.18 [3.08, 3.30] | converged 0.999 / 0.999 / 1.000
         state error l2 median 5.09 -> after one Newton step 0.117 (ratio of medians 44x); step-norm estimate vs error Spearman 0.996
       res_raw_med  res_phys_med  err_vm  err_va  nr_iters
layer                                                     
0         6.77e+03      6.77e+03   0.154   0.512         5
1         1.23e+03      1.23e+03   0.147    0.49      4.14
2              620           620   0.153   0.454         4
3              415           453   0.145   0.431         4
4              311           366   0.123   0.368         4
5              130           315  0.0611   0.265         4
6             71.4           305  0.0194    0.22      3.38
7             33.4           289  0.0126   0.207      3.25
8             21.6           290  0.0109   0.194      3.21
9             15.1           289  0.0101   0.187      3.18
10            12.8           290    0.01   0.187      3.18
11            11.1           289 0.00976   0.187      3.18
  exp05  NR iterations to 1e-6 MW: always-GENCO 3.18 [3.08, 3.30] | always-base 2.93 [2.90, 2.96] | best-of-two 2.92 [2.89, 2.94] (picks GENCO 11%) | oracle 2.89 [2.86, 2.91]
         paired: GENCO - base 0.25 [0.13, 0.38] | best-of-two - base -0.01 [-0.02, -0.01] | best-of-two - GENCO -0.26 [-0.39, -0.15] | GENCO fewer/more than base: 4% / 27%
  exp06  slack Vm error median 0.0227 [0.0179, 0.0255] | max mismatch reported 11.1 [10.2, 13.1] -> physical 289 [214, 355] | ratio 26 [18.5, 32]
         worst bus adjacent to slack: full 0.95 [0.875, 1] | constrained 0.95 [0.875, 1] | share of |dS| in 1-hop (full) 0.0958 [0.0823, 0.112] | constrained 0.12 [0.106, 0.138] | enrichment 191 [165, 225]
         re-anchoring worse in 97% (shift, state error) / 97% (scale)

```

### 3.2 Screening policy, exceedance, ladder

| set | c GENCO (max) | GENCO seq LU (max) | base seq LU (max) | tier-1 G/B (max) | c GENCO (conf 0.10) | GENCO seq LU (conf 0.10) | base seq LU (conf 0.10) | tier-1 G/B (conf 0.10) |
|---|---|---|---|---|---|---|---|---|
| GOC2000 Small (879) | 0.00515 | 2.032 [2.002, 2.086] | 2.261 [2.227, 2.293] | 0.00 / 0.04 | 0.00175 | 2.003 [2.000, 2.008] | 2.156 [2.122, 2.188] | 0.00 / 0.08 |
| GOC2000 Tiny (879) | 0.00239 | 2.008 [2.001, 2.017] | 2.261 [2.227, 2.293] | 0.00 / 0.04 | 0.0016 | 2.006 [2.001, 2.010] | 2.156 [2.122, 2.188] | 0.00 / 0.08 |
| GOC2000 Small (old 198) | 0.00161 | 2.005 [2.000, 2.015] | 2.071 [1.995, 2.136] | 0.00 / 0.10 | inf | 3.227 [3.005, 3.556] | 3.061 [3.035, 3.081] | 0.00 / 0.00 |

The old-198 conformal rows are c = inf (9 load scenarios, K = 5: the conformal rank exceeds n), i.e. every scenario escalates to NR; the new 40-load-scenario set no longer has that problem. Zero misses and zero false alarms for every calibrated policy on every set.

| set | prior | max (nominal ~1/(n+1)) | conf 0.10 | conf 0.05 |
|---|---|---|---|---|
| GOC2000 Small (879) | GENCO | 2/40 = 5.0% [0.6%, 16.9%] | 3/40 = 7.5% [1.6%, 20.4%] | 2/40 = 5.0% [0.6%, 16.9%] |
| GOC2000 Small (879) | base case | 1/40 = 2.5% [0.1%, 13.2%] | 5/40 = 12.5% [4.2%, 26.8%] | 1/40 = 2.5% [0.1%, 13.2%] |
| GOC2000 Tiny (879) | GENCO | 1/40 = 2.5% [0.1%, 13.2%] | 3/40 = 7.5% [1.6%, 20.4%] | 1/40 = 2.5% [0.1%, 13.2%] |
| GOC2000 Tiny (879) | base case | 1/40 = 2.5% [0.1%, 13.2%] | 5/40 = 12.5% [4.2%, 26.8%] | 1/40 = 2.5% [0.1%, 13.2%] |
| GOC2000 Small (old 198) | GENCO | 1/9 = 11.1% [0.3%, 48.2%] | 0/9 = 0.0% [0.0%, 33.6%] | 0/9 = 0.0% [0.0%, 33.6%] |
| GOC2000 Small (old 198) | base case | 1/9 = 11.1% [0.3%, 48.2%] | 0/9 = 0.0% [0.0%, 33.6%] | 0/9 = 0.0% [0.0%, 33.6%] |

| set | method | LU | el misses | miss rate 95% CP UB (2-sided) | el FA | cont misses | cont FA | load scen. with miss | load UB |
|---|---|---|---|---|---|---|---|---|---|
| GOC2000 Small (879) | GENCO + 1 step | 1 | 4/173 | 5.81% | 941 | 0/64 | 310 | 2/40 | 16.9% |
| GOC2000 Small (879) | GENCO + 2 steps | 2 | 0/173 | 2.11% | 0 | 0/64 | 0 | 0/40 | 8.8% |
| GOC2000 Small (879) | Base case + 1 step | 1 | 18/173 | 15.94% | 9 | 0/64 | 0 | 6/40 | 29.8% |
| GOC2000 Small (879) | Base case + 2 steps | 2 | 0/173 | 2.11% | 0 | 0/64 | 0 | 0/40 | 8.8% |
| GOC2000 Small (879) | GENCO as published | 0 | 105/173 | 68.02% | 720 | 17/64 | 425 | 27/40 | 81.4% |
| GOC2000 Tiny (879) | GENCO + 1 step | 1 | 3/173 | 4.98% | 1046 | 0/64 | 387 | 2/40 | 16.9% |
| GOC2000 Tiny (879) | GENCO + 2 steps | 2 | 0/173 | 2.11% | 0 | 0/64 | 0 | 0/40 | 8.8% |
| GOC2000 Tiny (879) | Base case + 1 step | 1 | 18/173 | 15.94% | 9 | 0/64 | 0 | 6/40 | 29.8% |
| GOC2000 Tiny (879) | Base case + 2 steps | 2 | 0/173 | 2.11% | 0 | 0/64 | 0 | 0/40 | 8.8% |
| GOC2000 Tiny (879) | GENCO as published | 0 | 118/173 | 75.07% | 781 | 12/64 | 441 | 27/40 | 81.4% |
| GOC2000 Small (old 198) | GENCO + 1 step | 1 | 4/61 | 15.95% | 130 | 0/16 | 60 | 2/9 | 60.0% |
| GOC2000 Small (old 198) | GENCO + 2 steps | 2 | 0/61 | 5.87% | 0 | 0/16 | 0 | 0/9 | 33.6% |
| GOC2000 Small (old 198) | Base case + 1 step | 1 | 16/61 | 39.07% | 0 | 0/16 | 0 | 4/9 | 78.8% |
| GOC2000 Small (old 198) | Base case + 2 steps | 2 | 0/61 | 5.87% | 0 | 0/16 | 0 | 0/9 | 33.6% |
| GOC2000 Small (old 198) | GENCO as published | 0 | 46/61 | 85.54% | 237 | 4/16 | 162 | 6/9 | 92.5% |

Pooled bounds are not tabulated for GOC 2000 because Small and Tiny use the same 879 scenarios and the old 198 are a subset of them. Per model on the 879 set, all calibrated policies (both priors, all rules): 0 / 173 violating elements missed (95% UB 2.11%), 0 / 64 violating contingencies missed (5.60%), 0 / 40 load scenarios with a miss (8.8%); zero element and contingency false alarms.

### 3.3 Verdict for GOC 2000

- As a Newton prior, GENCO is still worse than the base case by 0.2 to 0.6 iterations (an 8 to 28% penalty in iterations). The base case wins in 59 to 64% of scenarios at 1e-2 / 1e-4 and GENCO in 1 to 4%. Small and Tiny are indistinguishable.
- As a screening verifier, GENCO + 2 steps is exact (0 / 173 missed, 0 false alarms) and the sequential policy costs 2.00 LU (CI 2.00 to 2.01), against 2.16 [2.12, 2.19] (conformal 0.10) or 2.26 [2.23, 2.29] (max) for the base case. This is 7 to 11% cheaper, but it is the floor of a two-step policy (GENCO never passes after one step), not a sign that GENCO's state is good; at one step GENCO misses 4 elements against 18 for the base case and raises 941 against 9 element false alarms (310 against 0 contingency false alarms).
- Slack gap: 19.7x [15.8, 25.7] (Small) and 26x [18.5, 32] (Tiny) on 879 scenarios; the old 198-scenario estimate was 26x [17, 44]. The worst bus is adjacent to the slack in 95%; the share of |dS| held by the slack's 1-hop neighbours is 7 to 10% (enrichment 146x to 191x the uniform share).

## 4. Per-bus bias correction (task 4, `scripts/exp08_bias_correction.py`)

### 4.1 Method

For every scenario the signed errors of GENCO's physical state (`p.V_from(Vm, Va)` with true setpoints; exactly the exp02 "phys" state) against datakit's solution: dVm at PQ buses, dVa (wrapped) at non-reference buses. Corrections subtracted from the state before the setpoints are re-enforced:

- `bias`: per-bus mean signed error (ridge-shrunk to 0 with 3 pseudo-samples);
- `lin`: per-bus ridge regression on [1, total load MW];
- `inj`: per-bus ridge regression on [1, total load, the bus's own net P and Q injection] (features z-scored, scale floor 1 MW, clipped at +-4 sd);
- suffix `_x`: fitted on a *separate* set (`calib/n1`, 2,200 scenarios, 100 load scenarios; the same model's errors), which is the distribution-shift test;
- no suffix: 5-fold cross-fitting grouped by load scenario inside the evaluation set (a load scenario is never in the fit and the test fold together);
- `bias_in`: per-bus mean fitted on the evaluation set itself (optimistic reference, state error only).

Metrics per scenario and start: state error l2 / inf-norm (x = [Va at PV+PQ, Vm at PQ]), NR iterations to 1e-2 / 1e-4 / 1e-6 MW (`acpf.newton_trajectory`), branch loadings and PQ-bus voltages after 0 / 1 / 2 Newton steps fed to analyze_exp02's `Screen` and `decision_stats` (same equal-cost screening as the ladder), and the pre-contingency base case for reference. Sets: paper set (Base, Small, Tiny; 638 scenarios, 29 load scenarios) and stress100 N-5 (Base; 2,194 scenarios, 100 load scenarios). Reports: `results/exp08/report_{base,small,tiny}_seed0.txt`; per-scenario tables `results/exp08/*_scenarios.parquet`.

```
python scripts/exp08_bias_correction.py --size base --fit calib=data/calib/n1:splits_all \
   --sets paper=data/pf:splits_paper_test stress100_n5=data/stress100/n5:splits_all
python scripts/exp08_bias_correction.py --size base --diag_only --fit calib=data/calib/n1:splits_all
```

### 4.2 How systematic are the signed errors? (Base, `calib/n1`, 2,200 scenarios / 100 load scenarios)

```
DIAGNOSTICS of the signed errors on calib: 2200 scenarios, 100 load scenarios
  Vm (PQ buses): overall RMS 0.00027 p.u.; RMS of the per-bus mean signed error 0.00001 p.u.
    per-bus constant removes 0.0029 of the total squared error (in-sample, upper bound for the bias correction)
    |t| of the per-bus mean across load scenarios: median 0.77, share of buses with |t| > 3: 0.14 (n=64 buses; noise alone gives median 0.67, share 0.003)
    sign consistency: mean |P(e>0) - 0.5| over buses = 0.069; share of buses with P(e>0) outside [0.4, 0.6] = 0.22
    magnitude |e| is bus-specific: cross-fitted R^2 of |e| from the per-bus mean |e| (other load scenarios) = 0.447
  Va (non-REF buses): overall RMS 0.00872 rad; RMS of the per-bus mean signed error 0.00051 rad
    per-bus constant removes 0.0034 of the total squared error (in-sample, upper bound for the bias correction)
    |t| of the per-bus mean across load scenarios: median 0.93, share of buses with |t| > 3: 0.00 (n=117 buses; noise alone gives median 0.67, share 0.003)
    sign consistency: mean |P(e>0) - 0.5| over buses = 0.011; share of buses with P(e>0) outside [0.4, 0.6] = 0.00
    magnitude |e| is bus-specific: cross-fitted R^2 of |e| from the per-bus mean |e| (other load scenarios) = 0.095
  Va: removing one common-mode angle offset per scenario (needs the truth, not a usable correction) removes 0.751 of the squared Va error

```

Small:

```
DIAGNOSTICS of the signed errors on calib: 2200 scenarios, 100 load scenarios
  Vm (PQ buses): overall RMS 0.00032 p.u.; RMS of the per-bus mean signed error 0.00002 p.u.
    per-bus constant removes 0.0060 of the total squared error (in-sample, upper bound for the bias correction)
    |t| of the per-bus mean across load scenarios: median 1.31, share of buses with |t| > 3: 0.11 (n=64 buses; noise alone gives median 0.67, share 0.003)
    sign consistency: mean |P(e>0) - 0.5| over buses = 0.092; share of buses with P(e>0) outside [0.4, 0.6] = 0.38
    magnitude |e| is bus-specific: cross-fitted R^2 of |e| from the per-bus mean |e| (other load scenarios) = 0.491
  Va (non-REF buses): overall RMS 0.00937 rad; RMS of the per-bus mean signed error 0.00176 rad
    per-bus constant removes 0.0353 of the total squared error (in-sample, upper bound for the bias correction)
    |t| of the per-bus mean across load scenarios: median 2.40, share of buses with |t| > 3: 0.00 (n=117 buses; noise alone gives median 0.67, share 0.003)
    sign consistency: mean |P(e>0) - 0.5| over buses = 0.076; share of buses with P(e>0) outside [0.4, 0.6] = 0.05
    magnitude |e| is bus-specific: cross-fitted R^2 of |e| from the per-bus mean |e| (other load scenarios) = 0.120
  Va: removing one common-mode angle offset per scenario (needs the truth, not a usable correction) removes 0.726 of the squared Va error

```

Tiny:

```
DIAGNOSTICS of the signed errors on calib: 2200 scenarios, 100 load scenarios
  Vm (PQ buses): overall RMS 0.00032 p.u.; RMS of the per-bus mean signed error 0.00005 p.u.
    per-bus constant removes 0.0213 of the total squared error (in-sample, upper bound for the bias correction)
    |t| of the per-bus mean across load scenarios: median 1.53, share of buses with |t| > 3: 0.06 (n=64 buses; noise alone gives median 0.67, share 0.003)
    sign consistency: mean |P(e>0) - 0.5| over buses = 0.062; share of buses with P(e>0) outside [0.4, 0.6] = 0.16
    magnitude |e| is bus-specific: cross-fitted R^2 of |e| from the per-bus mean |e| (other load scenarios) = 0.397
  Va (non-REF buses): overall RMS 0.01118 rad; RMS of the per-bus mean signed error 0.00137 rad
    per-bus constant removes 0.0151 of the total squared error (in-sample, upper bound for the bias correction)
    |t| of the per-bus mean across load scenarios: median 1.61, share of buses with |t| > 3: 0.00 (n=117 buses; noise alone gives median 0.67, share 0.003)
    sign consistency: mean |P(e>0) - 0.5| over buses = 0.055; share of buses with P(e>0) outside [0.4, 0.6] = 0.10
    magnitude |e| is bus-specific: cross-fitted R^2 of |e| from the per-bus mean |e| (other load scenarios) = 0.142
  Va: removing one common-mode angle offset per scenario (needs the truth, not a usable correction) removes 0.631 of the squared Va error

```

The per-bus mean signed error is tiny against the error itself (Vm: 1e-5 against an RMS of 2.7e-4 p.u.; Va: 5e-4 against 8.7e-3 rad), so a per-bus constant removes only 0.3% of the squared error even in-sample for Base (Small: 0.6% of the Vm and 3.5% of the Va squared error; Tiny: 2.1% and 1.5%). The *magnitude* of the error is bus-specific (a per-bus mean of |e| from other operating points explains 45% of the variance of |dVm|, in line with the probe's 0.55 for bus-ID one-hot, and 10% of |dVa|), but the sign is mostly not stable: for Vm 22% of the buses (Base) have P(e>0) outside [0.4, 0.6] and 14% have |t| > 3 across load scenarios, yet those per-bus means are only 4% of the RMS (1e-5 against 2.7e-4 p.u.); for Va no bus has |t| > 3. About 75% (Base; 73% Small, 63% Tiny) of the squared Va error is a per-scenario common-mode angle shift (all non-reference buses displaced together), which a per-bus correction cannot address; whether this is the slack-neighbourhood effect of exp06 was not tested.

### 4.3 Paper set, Base seed 0 (638 scenarios, 29 load scenarios)

State error (rad / p.u.; SSE reduction = 1 - sum(after^2) / sum(before^2), negative = worse; "better" = share of scenarios whose l2 is smaller than uncorrected):

```
1. STATE ERROR of the start (x = [Va at PV+PQ, Vm at PQ]; rad / p.u.). SSE reduction = 1 - sum(err_after^2)/sum(err_unc^2) with cluster CI; 'better' = share of scenarios with smaller l2 than the uncorrected start. bias_in is fitted on the evaluation set itself (optimistic).
  start                  median_l2  p95_l2 median_inf max_inf              SSE_reduction better
    unc 0.04940 [0.03488, 0.06888] 0.18474    0.00708  0.0968                        NaN    NaN
   bias 0.05302 [0.03638, 0.06981] 0.19319    0.00724  0.0972     -0.039 [-0.079, 0.002]    34%
    lin 0.05912 [0.04456, 0.07278] 0.19313    0.00854  0.0944     -0.066 [-0.209, 0.036]    46%
    inj 0.06332 [0.05256, 0.07750] 0.19491    0.01707  0.1063    -0.130 [-0.318, -0.005]    24%
 bias_x 0.04899 [0.03470, 0.06775] 0.18209    0.00699  0.0961     -0.002 [-0.035, 0.031]    53%
  lin_x 0.05238 [0.03483, 0.06889] 0.18882    0.00696  0.0963     -0.057 [-0.160, 0.039]    46%
  inj_x 0.05235 [0.03654, 0.06918] 0.18937    0.00926  0.0978     -0.063 [-0.173, 0.035]    39%
bias_in 0.04906 [0.03470, 0.06884] 0.18422    0.00707  0.0968      0.000 [-0.003, 0.003]    47%
   base 0.05022 [0.04274, 0.05851] 0.87444    0.02564  1.1613 -30.731 [-57.332, -17.139]    47%
```

NR iterations (mean [95% cluster CI]; `minus_unc`: paired difference to the uncorrected GENCO start; `minus_base`: to the base-case start; `inj` rows dropped for brevity, they are worse):

```
2. NEWTON-RAPHSON ITERATIONS to 1e-2 / 1e-4 / 1e-6 MW, mean [95% cluster CI]; paired difference to the uncorrected GENCO start and to the base-case start
   tol  start                iters              minus_base              minus_unc
  0.01    unc 1.719 [1.585, 1.845] -0.318 [-0.453, -0.193]                       
  0.01   bias 1.745 [1.599, 1.865] -0.293 [-0.428, -0.165]  0.025 [-0.019, 0.085]
  0.01    lin 1.809 [1.687, 1.912] -0.229 [-0.353, -0.122]   0.089 [0.024, 0.185]
  0.01 bias_x 1.732 [1.592, 1.857] -0.306 [-0.439, -0.179]  0.013 [-0.031, 0.064]
  0.01  lin_x 1.741 [1.597, 1.864] -0.296 [-0.430, -0.172]  0.022 [-0.024, 0.075]
  0.01   base 2.038 [2.005, 2.067]                           0.318 [0.193, 0.453]
0.0001    unc 2.005 [2.000, 2.009] -0.489 [-0.527, -0.450]                       
0.0001   bias 2.005 [2.000, 2.009] -0.489 [-0.527, -0.450]   0.000 [0.000, 0.000]
0.0001    lin 2.005 [2.000, 2.009] -0.489 [-0.527, -0.450]   0.000 [0.000, 0.000]
0.0001 bias_x 2.005 [2.000, 2.009] -0.489 [-0.527, -0.450]   0.000 [0.000, 0.000]
0.0001  lin_x 2.005 [2.000, 2.009] -0.489 [-0.527, -0.450]   0.000 [0.000, 0.000]
0.0001   base 2.494 [2.456, 2.533]                           0.489 [0.450, 0.527]
 1e-06    unc 2.147 [2.060, 2.248] -0.734 [-0.829, -0.629]                       
 1e-06   bias 2.133 [2.052, 2.234] -0.748 [-0.840, -0.646] -0.014 [-0.044, 0.005]
 1e-06    lin 2.127 [2.049, 2.229] -0.754 [-0.840, -0.655] -0.020 [-0.058, 0.006]
 1e-06 bias_x 2.149 [2.060, 2.251] -0.732 [-0.831, -0.627]  0.002 [-0.006, 0.011]
 1e-06  lin_x 2.169 [2.067, 2.288] -0.712 [-0.821, -0.592]  0.022 [-0.002, 0.053]
 1e-06   base 2.881 [2.832, 2.926]                           0.734 [0.629, 0.829]

```

Equal-cost screening (violating elements missed / false alarms after 0, 1, 2 Newton steps; 484 violating elements in 225 contingencies):

```
3. EQUAL-COST SCREENING (tau = 1.01, eps_v = 1 mV; branches + PQ buses): violating elements missed / false alarms after 0, 1, 2 Newton steps; contingency-level misses; element miss rate with 95% CI and one-sided Clopper-Pearson upper bound
               method  LU el_miss  el_fa               miss_rate    ub1 cont_miss  cont_fa load_miss
unc, 0 steps (V_from)   0  47/484    285 0.0971 [0.0561, 0.1483] 12.22%    14/225      111     19/29
         unc + 1 step   1   1/484      0 0.0021 [0.0000, 0.0068]  0.98%     0/225        0      1/29
        unc + 2 steps   2   0/484      0 0.0000 [0.0000, 0.0000]  0.62%     0/225        0      0/29
        bias, 0 steps   0  47/484    300 0.0971 [0.0561, 0.1483] 12.22%    13/225      117     19/29
        bias + 1 step   1   1/484      0 0.0021 [0.0000, 0.0068]  0.98%     0/225        0      1/29
       bias + 2 steps   2   0/484      0 0.0000 [0.0000, 0.0000]  0.62%     0/225        0      0/29
         lin, 0 steps   0  43/484    359 0.0888 [0.0505, 0.1392] 11.31%    10/225      150     17/29
         lin + 1 step   1   1/484      0 0.0021 [0.0000, 0.0068]  0.98%     0/225        0      1/29
        lin + 2 steps   2   0/484      0 0.0000 [0.0000, 0.0000]  0.62%     0/225        0      0/29
         inj, 0 steps   0  54/484    840 0.1116 [0.0671, 0.1679] 13.79%    11/225      310     22/29
         inj + 1 step   1   7/484      0 0.0145 [0.0048, 0.0236]  2.70%     0/225        0      7/29
        inj + 2 steps   2   0/484      0 0.0000 [0.0000, 0.0000]  0.62%     0/225        0      0/29
      bias_x, 0 steps   0  44/484    306 0.0909 [0.0507, 0.1426] 11.53%    11/225      120     17/29
      bias_x + 1 step   1   1/484      0 0.0021 [0.0000, 0.0068]  0.98%     0/225        0      1/29
     bias_x + 2 steps   2   0/484      0 0.0000 [0.0000, 0.0000]  0.62%     0/225        0      0/29
       lin_x, 0 steps   0  44/484    319 0.0909 [0.0507, 0.1426] 11.53%    11/225      126     17/29
       lin_x + 1 step   1   1/484      0 0.0021 [0.0000, 0.0068]  0.98%     0/225        0      1/29
      lin_x + 2 steps   2   0/484      0 0.0000 [0.0000, 0.0000]  0.62%     0/225        0      0/29
       inj_x, 0 steps   0  48/484    445 0.0992 [0.0597, 0.1473] 12.44%    14/225      174     19/29
       inj_x + 1 step   1   1/484      0 0.0021 [0.0000, 0.0068]  0.98%     0/225        0      1/29
      inj_x + 2 steps   2   0/484      0 0.0000 [0.0000, 0.0000]  0.62%     0/225        0      0/29
   base case + 1 step   1  33/484      6 0.0682 [0.0477, 0.0922]  9.01%     4/225        0     21/29
  base case + 2 steps   2   0/484      0 0.0000 [0.0000, 0.0000]  0.62%     0/225        0      0/29

==============================================================================================================
```

### 4.4 Stress100 N-5 with corrections fitted on `calib/n1` (distribution shift) and within N-5 (Base seed 0; 2,194 scenarios, 100 load scenarios)

```
1. STATE ERROR of the start (x = [Va at PV+PQ, Vm at PQ]; rad / p.u.). SSE reduction = 1 - sum(err_after^2)/sum(err_unc^2) with cluster CI; 'better' = share of scenarios with smaller l2 than the uncorrected start. bias_in is fitted on the evaluation set itself (optimistic).
  start                  median_l2  p95_l2 median_inf max_inf              SSE_reduction better
    unc 0.08639 [0.07815, 0.09883] 0.48665    0.01332  0.6142                        NaN    NaN
   bias 0.10634 [0.09810, 0.11456] 0.44733    0.01646  0.6085       0.069 [0.045, 0.095]    48%
    lin 0.10546 [0.09674, 0.11342] 0.45589    0.01618  0.6089       0.069 [0.043, 0.096]    49%
    inj 0.10850 [0.10047, 0.11665] 0.44662    0.02453  0.6085       0.074 [0.047, 0.102]    47%
 bias_x 0.08636 [0.07805, 0.09535] 0.48889    0.01343  0.6138       0.008 [0.006, 0.010]    65%
  lin_x 0.08514 [0.07813, 0.09555] 0.49223    0.01315  0.6137       0.009 [0.005, 0.013]    57%
  inj_x 0.08605 [0.07868, 0.09546] 0.49365    0.01535  0.6137       0.010 [0.006, 0.015]    53%
bias_in 0.10564 [0.09735, 0.11307] 0.44400    0.01638  0.6082       0.071 [0.047, 0.098]    49%
   base 0.38744 [0.36220, 0.41255] 3.60657    0.13664  1.6719 -16.057 [-20.656, -13.087]     5%
```

```
2. NEWTON-RAPHSON ITERATIONS to 1e-2 / 1e-4 / 1e-6 MW, mean [95% cluster CI]; paired difference to the uncorrected GENCO start and to the base-case start
   tol  start                iters              minus_base               minus_unc
  0.01    unc 1.852 [1.824, 1.882] -0.655 [-0.689, -0.618]                        
  0.01   bias 1.938 [1.919, 1.956] -0.568 [-0.598, -0.538]    0.087 [0.050, 0.123]
  0.01    lin 1.939 [1.919, 1.957] -0.567 [-0.599, -0.536]    0.088 [0.050, 0.123]
  0.01    inj 2.062 [2.050, 2.073] -0.445 [-0.470, -0.420]    0.210 [0.181, 0.237]
  0.01 bias_x 1.851 [1.825, 1.880] -0.655 [-0.689, -0.617]  -0.000 [-0.008, 0.007]
  0.01  lin_x 1.854 [1.826, 1.881] -0.653 [-0.688, -0.617]   0.002 [-0.007, 0.011]
  0.01  inj_x 2.018 [2.012, 2.024] -0.489 [-0.512, -0.465]    0.166 [0.137, 0.194]
  0.01   base 2.507 [2.481, 2.530]                            0.655 [0.618, 0.689]
0.0001    unc 2.074 [2.060, 2.088] -1.009 [-1.024, -0.993]                        
0.0001   bias 2.067 [2.055, 2.078] -1.016 [-1.031, -1.002] -0.008 [-0.013, -0.003]
0.0001    lin 2.067 [2.055, 2.078] -1.016 [-1.031, -1.002] -0.008 [-0.013, -0.003]
0.0001    inj 2.723 [2.675, 2.774] -0.360 [-0.407, -0.309]    0.649 [0.603, 0.700]
0.0001 bias_x 2.074 [2.060, 2.088] -1.009 [-1.024, -0.994]  -0.000 [-0.001, 0.000]
0.0001  lin_x 2.073 [2.060, 2.087] -1.010 [-1.025, -0.994]  -0.001 [-0.002, 0.000]
0.0001  inj_x 2.082 [2.068, 2.096] -1.000 [-1.015, -0.985]    0.008 [0.005, 0.012]
0.0001   base 3.083 [3.066, 3.099]                            1.009 [0.993, 1.024]
 1e-06    unc 2.377 [2.341, 2.414] -0.960 [-0.997, -0.923]                        
 1e-06   bias 2.408 [2.374, 2.444] -0.929 [-0.968, -0.888]   0.031 [-0.010, 0.068]
 1e-06    lin 2.415 [2.380, 2.450] -0.922 [-0.961, -0.880]   0.038 [-0.002, 0.076]
 1e-06    inj 3.028 [3.021, 3.036] -0.309 [-0.333, -0.287]    0.651 [0.616, 0.685]
 1e-06 bias_x 2.374 [2.337, 2.410] -0.964 [-0.998, -0.927]  -0.003 [-0.009, 0.002]
 1e-06  lin_x 2.377 [2.341, 2.412] -0.960 [-0.995, -0.924]   0.000 [-0.006, 0.008]
 1e-06  inj_x 2.887 [2.854, 2.919] -0.450 [-0.485, -0.412]    0.510 [0.472, 0.549]
 1e-06   base 3.337 [3.315, 3.361]                            0.960 [0.923, 0.997]

```

```
3. EQUAL-COST SCREENING (tau = 1.01, eps_v = 1 mV; branches + PQ buses): violating elements missed / false alarms after 0, 1, 2 Newton steps; contingency-level misses; element miss rate with 95% CI and one-sided Clopper-Pearson upper bound
               method  LU  el_miss  el_fa               miss_rate    ub1 cont_miss  cont_fa load_miss
unc, 0 steps (V_from)   0 772/6617    334 0.1167 [0.1004, 0.1353] 12.34%   58/1916       39    96/100
         unc + 1 step   1  18/6617      2 0.0027 [0.0013, 0.0043]  0.40%    1/1916        0    13/100
        unc + 2 steps   2   0/6617      0 0.0000 [0.0000, 0.0000]  0.05%    0/1916        0     0/100
        bias, 0 steps   0 497/6617    790 0.0751 [0.0638, 0.0875]  8.07%   21/1916       97    92/100
        bias + 1 step   1  15/6617      3 0.0023 [0.0010, 0.0036]  0.35%    1/1916        0    11/100
       bias + 2 steps   2   0/6617      0 0.0000 [0.0000, 0.0000]  0.05%    0/1916        0     0/100
         lin, 0 steps   0 487/6617    784 0.0736 [0.0622, 0.0861]  7.91%   20/1916       99    92/100
         lin + 1 step   1  14/6617      3 0.0021 [0.0009, 0.0035]  0.33%    1/1916        0    10/100
        lin + 2 steps   2   0/6617      0 0.0000 [0.0000, 0.0000]  0.05%    0/1916        0     0/100
         inj, 0 steps   0 802/6617   2743 0.1212 [0.1089, 0.1334] 12.80%   27/1916      214    97/100
         inj + 1 step   1 139/6617     19 0.0210 [0.0171, 0.0256]  2.41%   12/1916        2    64/100
        inj + 2 steps   2   1/6617      0 0.0002 [0.0000, 0.0005]  0.07%    0/1916        0     1/100
      bias_x, 0 steps   0 747/6617    358 0.1129 [0.0968, 0.1307] 11.95%   56/1916       43    96/100
      bias_x + 1 step   1  18/6617      2 0.0027 [0.0013, 0.0043]  0.40%    1/1916        0    13/100
     bias_x + 2 steps   2   0/6617      0 0.0000 [0.0000, 0.0000]  0.05%    0/1916        0     0/100
       lin_x, 0 steps   0 747/6617    364 0.1129 [0.0969, 0.1304] 11.95%   57/1916       43    96/100
       lin_x + 1 step   1  18/6617      2 0.0027 [0.0013, 0.0043]  0.40%    1/1916        0    13/100
      lin_x + 2 steps   2   0/6617      0 0.0000 [0.0000, 0.0000]  0.05%    0/1916        0     0/100
       inj_x, 0 steps   0 811/6617    718 0.1226 [0.1063, 0.1395] 12.94%   57/1916       89    99/100
       inj_x + 1 step   1  28/6617      4 0.0042 [0.0026, 0.0060]  0.58%    1/1916        0    20/100
      inj_x + 2 steps   2   0/6617      0 0.0000 [0.0000, 0.0000]  0.05%    0/1916        0     0/100
   base case + 1 step   1 519/6617    205 0.0784 [0.0696, 0.0878]  8.41%   21/1916        2    97/100
  base case + 2 steps   2  19/6617      3 0.0029 [0.0016, 0.0042]  0.42%    1/1916        0    17/100
```

### 4.5 Small and Tiny (paper set)

State error, Small:

```
1. STATE ERROR of the start (x = [Va at PV+PQ, Vm at PQ]; rad / p.u.). SSE reduction = 1 - sum(err_after^2)/sum(err_unc^2) with cluster CI; 'better' = share of scenarios with smaller l2 than the uncorrected start. bias_in is fitted on the evaluation set itself (optimistic).
  start                  median_l2  p95_l2 median_inf max_inf              SSE_reduction better
    unc 0.06088 [0.04068, 0.09220] 0.18893    0.00924  0.1131                        NaN    NaN
   bias 0.06260 [0.04726, 0.09149] 0.18094    0.00979  0.1157     -0.001 [-0.083, 0.067]    47%
    lin 0.07389 [0.05449, 0.09642] 0.17853    0.01141  0.1150     -0.034 [-0.165, 0.051]    36%
    inj 0.08037 [0.05866, 0.10112] 0.18038    0.01961  0.1229     -0.099 [-0.273, 0.011]    23%
 bias_x 0.05752 [0.04158, 0.10177] 0.20179    0.00878  0.1103     -0.073 [-0.173, 0.029]    47%
  lin_x 0.05496 [0.03934, 0.10473] 0.20925    0.00843  0.1104     -0.113 [-0.239, 0.020]    46%
  inj_x 0.05591 [0.04066, 0.10502] 0.20652    0.01116  0.1123     -0.116 [-0.238, 0.009]    43%
bias_in 0.06216 [0.04724, 0.09071] 0.17864    0.00971  0.1145      0.017 [-0.057, 0.079]    48%
   base 0.05022 [0.04274, 0.05851] 0.87444    0.02564  1.1613 -24.863 [-42.947, -14.493]    51%
```

State error, Tiny:

```
1. STATE ERROR of the start (x = [Va at PV+PQ, Vm at PQ]; rad / p.u.). SSE reduction = 1 - sum(err_after^2)/sum(err_unc^2) with cluster CI; 'better' = share of scenarios with smaller l2 than the uncorrected start. bias_in is fitted on the evaluation set itself (optimistic).
  start                  median_l2  p95_l2 median_inf max_inf              SSE_reduction better
    unc 0.07217 [0.05367, 0.08696] 0.18199    0.01030  0.2002                        NaN    NaN
   bias 0.07131 [0.05490, 0.08919] 0.18279    0.01041  0.2022     -0.013 [-0.053, 0.035]    48%
    lin 0.07045 [0.05962, 0.08652] 0.17400    0.01014  0.2012     -0.001 [-0.057, 0.064]    43%
    inj 0.07257 [0.06287, 0.08939] 0.18044    0.01650  0.2031     -0.032 [-0.096, 0.031]    35%
 bias_x 0.07052 [0.05207, 0.08553] 0.18111    0.01023  0.2025      0.007 [-0.035, 0.064]    56%
  lin_x 0.06807 [0.05146, 0.08718] 0.18991    0.01023  0.2039     -0.016 [-0.089, 0.074]    55%
  inj_x 0.06828 [0.05259, 0.08832] 0.18738    0.01277  0.2084     -0.018 [-0.089, 0.067]    53%
bias_in 0.07189 [0.05167, 0.08578] 0.17870    0.01019  0.2022      0.009 [-0.023, 0.051]    57%
   base 0.05022 [0.04274, 0.05851] 0.87444    0.02564  1.1613 -19.817 [-31.775, -12.896]    57%
```

NR iterations, Small:

```
2. NEWTON-RAPHSON ITERATIONS to 1e-2 / 1e-4 / 1e-6 MW, mean [95% cluster CI]; paired difference to the uncorrected GENCO start and to the base-case start
   tol  start                iters              minus_base              minus_unc
  0.01    unc 1.683 [1.566, 1.820] -0.354 [-0.480, -0.218]                       
  0.01   bias 1.729 [1.603, 1.845] -0.309 [-0.431, -0.191]  0.045 [-0.072, 0.160]
  0.01    lin 1.779 [1.661, 1.884] -0.259 [-0.373, -0.152]  0.096 [-0.017, 0.201]
  0.01    inj 2.000 [2.000, 2.000] -0.038 [-0.067, -0.005]   0.317 [0.180, 0.434]
  0.01 bias_x 1.721 [1.597, 1.850] -0.317 [-0.440, -0.182]  0.038 [-0.006, 0.089]
  0.01  lin_x 1.687 [1.566, 1.823] -0.351 [-0.478, -0.215]  0.003 [-0.055, 0.064]
  0.01  inj_x 2.000 [2.000, 2.000] -0.038 [-0.067, -0.005]   0.317 [0.180, 0.434]
  0.01   base 2.038 [2.005, 2.067]                           0.354 [0.218, 0.480]
0.0001    unc 2.005 [2.000, 2.009] -0.489 [-0.527, -0.450]                       
0.0001   bias 2.005 [2.000, 2.009] -0.489 [-0.527, -0.450]   0.000 [0.000, 0.000]
0.0001    lin 2.005 [2.000, 2.009] -0.489 [-0.527, -0.450]   0.000 [0.000, 0.000]
0.0001    inj 2.749 [2.599, 2.897]    0.255 [0.099, 0.400]   0.745 [0.594, 0.893]
0.0001 bias_x 2.005 [2.000, 2.009] -0.489 [-0.527, -0.450]   0.000 [0.000, 0.000]
0.0001  lin_x 2.005 [2.000, 2.009] -0.489 [-0.527, -0.450]   0.000 [0.000, 0.000]
0.0001  inj_x 2.011 [2.003, 2.022] -0.483 [-0.522, -0.445]   0.006 [0.000, 0.016]
0.0001   base 2.494 [2.456, 2.533]                           0.489 [0.450, 0.527]
 1e-06    unc 2.218 [2.107, 2.340] -0.663 [-0.779, -0.541]                       
 1e-06   bias 2.185 [2.080, 2.307] -0.696 [-0.809, -0.572] -0.033 [-0.096, 0.022]
 1e-06    lin 2.207 [2.097, 2.328] -0.674 [-0.787, -0.549] -0.011 [-0.071, 0.039]
 1e-06    inj 3.000 [3.000, 3.000]    0.119 [0.074, 0.168]   0.782 [0.660, 0.893]
 1e-06 bias_x 2.213 [2.102, 2.342] -0.668 [-0.778, -0.544] -0.005 [-0.045, 0.028]
 1e-06  lin_x 2.226 [2.114, 2.361] -0.655 [-0.774, -0.525]  0.008 [-0.041, 0.055]
 1e-06  inj_x 2.918 [2.826, 2.995]   0.038 [-0.039, 0.105]   0.701 [0.563, 0.828]
 1e-06   base 2.881 [2.832, 2.926]                           0.663 [0.541, 0.779]

```

NR iterations, Tiny:

```
2. NEWTON-RAPHSON ITERATIONS to 1e-2 / 1e-4 / 1e-6 MW, mean [95% cluster CI]; paired difference to the uncorrected GENCO start and to the base-case start
   tol  start                iters              minus_base               minus_unc
  0.01    unc 1.828 [1.729, 1.915] -0.210 [-0.315, -0.108]                        
  0.01   bias 1.771 [1.679, 1.862] -0.266 [-0.359, -0.168]  -0.056 [-0.129, 0.008]
  0.01    lin 1.817 [1.727, 1.904] -0.221 [-0.317, -0.125]  -0.011 [-0.075, 0.042]
  0.01    inj 2.000 [2.000, 2.000] -0.038 [-0.067, -0.005]    0.172 [0.085, 0.271]
  0.01 bias_x 1.781 [1.688, 1.871] -0.257 [-0.359, -0.158] -0.047 [-0.099, -0.006]
  0.01  lin_x 1.766 [1.674, 1.857] -0.271 [-0.373, -0.174] -0.061 [-0.119, -0.014]
  0.01  inj_x 2.000 [2.000, 2.000] -0.038 [-0.067, -0.005]    0.172 [0.085, 0.271]
  0.01   base 2.038 [2.005, 2.067]                            0.210 [0.108, 0.315]
0.0001    unc 2.011 [2.003, 2.020] -0.483 [-0.520, -0.445]                        
0.0001   bias 2.013 [2.005, 2.022] -0.481 [-0.520, -0.444]    0.002 [0.000, 0.005]
0.0001    lin 2.013 [2.005, 2.022] -0.481 [-0.520, -0.444]    0.002 [0.000, 0.005]
0.0001    inj 2.420 [2.254, 2.586]  -0.074 [-0.240, 0.092]    0.409 [0.246, 0.575]
0.0001 bias_x 2.011 [2.003, 2.019] -0.483 [-0.522, -0.445]   0.000 [-0.005, 0.005]
0.0001  lin_x 2.011 [2.003, 2.019] -0.483 [-0.522, -0.445]   0.000 [-0.005, 0.005]
0.0001  inj_x 2.024 [2.011, 2.038] -0.470 [-0.509, -0.431]    0.013 [0.003, 0.024]
0.0001   base 2.494 [2.456, 2.533]                            0.483 [0.445, 0.520]
 1e-06    unc 2.176 [2.102, 2.252] -0.705 [-0.796, -0.613]                        
 1e-06   bias 2.182 [2.103, 2.262] -0.699 [-0.790, -0.608]   0.006 [-0.022, 0.036]
 1e-06    lin 2.174 [2.096, 2.254] -0.707 [-0.796, -0.618]  -0.002 [-0.027, 0.025]
 1e-06    inj 3.000 [3.000, 3.000]    0.119 [0.074, 0.168]    0.824 [0.748, 0.898]
 1e-06 bias_x 2.174 [2.096, 2.252] -0.707 [-0.799, -0.614]  -0.002 [-0.030, 0.025]
 1e-06  lin_x 2.194 [2.108, 2.285] -0.687 [-0.784, -0.588]   0.019 [-0.019, 0.061]
 1e-06  inj_x 2.981 [2.944, 3.000]    0.100 [0.052, 0.146]    0.806 [0.727, 0.884]
 1e-06   base 2.881 [2.832, 2.926]                            0.705 [0.613, 0.796]

```

Equal-cost 1-step misses (484 violating elements; reports have the full ladder): Small 0 / 484 for uncorrected, `bias`, `lin`, `bias_x`, `lin_x`; `inj` 16 / 484, `inj_x` 1 / 484. Tiny 0 / 484 for the same five; `inj` 9 / 484, `inj_x` 2 / 484. Base case + 1 step: 33 / 484 for all three checkpoints.

### 4.6 Verdict

The correction does nothing useful.

- Fitted on N-1 and applied to the paper set (same distribution) and to N-5 (shifted), `bias_x` changes the squared state error by -0.2% [-3.5, +3.1] on the paper set (`lin_x` -5.7% [-16, +3.9]) and by +0.8% [0.6, 1.0] on N-5 (`lin_x` +0.9%), which is statistically positive but negligible. NR iterations to 1e-4 MW stay where they were (2.005 on the paper set, 2.074 on N-5), and the 1-step equal-cost misses are unchanged (1 / 484 and 18 / 6617).
- Cross-fitted inside the evaluation set the per-bus mean makes the paper-set state error *worse* (SSE -3.9% [-7.9, 0.2]; median l2 0.049 to 0.053), and on N-5 it helps the sum of squares (+6.9% [4.5, 9.5]) while making the typical scenario worse (median l2 0.086 to 0.106, better in only 48% of scenarios; NR to 1e-2 MW +0.09 [0.05, 0.12] iterations). That gain comes from the heavy tail of N-5, not from a stable bias.
- The injection-dependent variant (`inj`) is worse everywhere: more NR iterations (3.00 against 2.15 to 1e-6 MW on the paper set) and 7 against 1 missed elements at one step.
- Small and Tiny behave the same (paper-set SSE reductions between -12% and +1%, CIs through 0 for bias / lin; no iteration or miss benefit). A per-bus constant is slightly more systematic for these two (cross-fitted SSE reduction on `calib/n1`: 2.5% Small, 1.2% Tiny, against 0.1% Base), but it does not transfer to the paper set. At one Newton step every variant misses 0 / 484 elements except `inj` (Small 16 / 484, Tiny 9 / 484; `inj_x` 1 and 2).
- The base-case comparison is unchanged by any correction: GENCO's NR advantage over the base case on IEEE 118 is -0.49 [-0.53, -0.45] iterations to 1e-4 MW with or without it.

So the embedding-probe lead ("a per-bus bias correction may help") is closed negatively: bus-specific error magnitude is not a correctable bias.

## 5. Code, files, reproduction

New or changed (no commits):

- `scripts/analyze_exp02.py`: new mode `--xcal` with `--cal_tag` (calibration run tag, or `none` for cross-fitting inside each evaluation set), `--eval_tags`, `--labels`, `--prefix`, `--boot` (functions `xcal_model`, `main_xcal`; reuses `RawSet`, `Screen`, `run_policy`, `decision_stats`, `pooled_bounds`). The legacy CLI, `--v3` and `--ci` are unchanged (checked: legacy table prints, `tests/test_screening.py` 12 passed). It reproduces the existing paper / stress numbers exactly (e.g. paper-set c = 0.3465, LU 1.403).
- `scripts/exp08_bias_correction.py`, `scripts/summarize_batch3.py`, `scripts/bench_genco_cpu.py`, drivers `scripts/run_batch3{,b,c}.sh`.
- Outputs: `results/exp02/{calib_n1_base,stress100_n*_base,case500_goc_*_seed0,case2000_goc_*_seed0_h879}_{branches,buses,scenarios}.parquet`; `results/exp02/{big118,goc500,goc2000h}_{policies,ladder,nr_iterations,exceedance,pooled_bounds_*,score_percentiles}.csv`; `results/policy_{big118,goc500,goc2000h}.log`; `results/exp01|exp05|exp06|exp07/` entries for the new tags; `results/exp08/`; `results/runtime/case500_goc_base_cpu.json`; `results/batch3_logs/`.
- `results/exp06/summary.csv` was rebuilt by exp06's own aggregation and now has rows for GOC 500 (3 sizes) and the new GOC 2000 runs (2 sizes).

Not done / limits:

- Seeds 1 and 42 and the other sizes were not run on the big IEEE 118 sets; exp07 only for GOC 500 Base; GOC 500 seeds 1 / 42 (splits exist) not run.
- The GOC 500 and GOC 2000 conformal statements use 29 and 40 load scenarios with cross-fitting, so they are weaker than the IEEE 118 statements.
- The exp07 timings and the GPU-free GENCO inference time are provisional (loaded machine).
- exp08 fits linear/ridge corrections only; an outage-aware correction (the error depends on where the outage is) was not tried, and the common-mode angle offset was only measured, not modelled.
