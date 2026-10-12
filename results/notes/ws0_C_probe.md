# WS0.5 (C): embedding probe redone (v2)

Date: 2026-10-11. Script: `scripts/exp04_embedding_probe.py`. Outputs: `results/exp04/probe_v2.csv`
(per model x feature set x target, with 95% CIs), `probe_v2_contrasts.csv` (paired differences),
`probe_v2_summary.json` (mean +- SD over seeds), and `*_loo_seed0.*` (sensitivity: sklearn's leave-one-out RidgeCV).

## What changed relative to the v1 probe (F9)

1. **Grouped split.** Grouped 5-fold CV by load scenario (29 groups x 22 topology variants = 638 test scenarios). v1 split by scenario, so siblings leaked. The same folds are used for every feature set, model and seed. Ridge alpha is chosen by an inner grouped CV (same idea as RidgeCV, but with load-scenario folds). With sklearn's leave-one-out RidgeCV instead, the embedding rows get slightly worse (seed 0: 0.557 -> 0.518 for |dVm|), and the bus-ID rows are unchanged. So the alpha rule does not tilt the comparison against the embedding.
2. **Outage location from the raw tables.** Branch outage = rows with `br_status == 0` (both endpoints); generator outage = bus of the row with `in_service == 0`. Elements that are out in the load scenario's base case (majority status) are excluded, though none exist in IEEE 118. Every test scenario has exactly one contingency (466 branch, 172 generator). No hops value is 99 any more (max 14).
3. **Controls.**
   - Bus-ID one-hot (118-d), alone, + mismatch, + structure + mismatch.
   - Embedding of a randomly initialised GENCO, same architecture, same pipeline. All `reset_parameters` are re-run, init seed 1000 + model seed, so three different random networks over the three Base seeds. The targets are still the trained model's errors.
   - PCA of the real and the random embedding at k = 5, 20, 118, fit on train folds only.
   - Structure (bus type, outage kind, hops to slack, hops to the true outage).
   - Extra: raw local inputs (Pd, Qd, GS, BS, Vm setpoint, type).
   - "Skill vs bus mean" is the new metric that isolates what a feature set explains beyond a bus's average error: 1 - MSE / MSE(train-fold mean of that bus).
4. **Uncertainty.** R^2 and Spearman are computed on pooled out-of-fold predictions. The 95% CI is a cluster bootstrap over the 29 load scenarios (2000 resamples, paired across feature sets). The CI does not include retraining variability. Across models we report mean +- SD (n = 3 Base seeds: 0, 1, 42), and IEEE 118 Small seed 0 separately.

## Results (IEEE 118 Base, mean +- SD over seeds 0, 1, 42; targets are log10 error; Small seed 0 in the last column)

R^2 for log10 |dVm| at PQ buses (about 41k bus-rows per model):

| Features | R^2 | Spearman | Skill vs bus mean | R^2, Small s0 |
|---|---|---|---|---|
| Local mismatch only | 0.224 +- 0.030 | 0.369 | -0.71 | 0.196 |
| Structure (type, outage kind, hops) | 0.196 +- 0.013 | 0.417 | -0.77 | 0.163 |
| Raw local inputs | 0.030 +- 0.011 | 0.154 | -1.13 | 0.022 |
| **Bus-ID one-hot (118-d)** | **0.553 +- 0.029** | 0.719 | 0.020 | 0.519 |
| Bus-ID + mismatch | 0.652 +- 0.008 | 0.797 | 0.233 | 0.642 |
| Bus-ID + structure + mismatch | 0.653 +- 0.007 | 0.797 | 0.236 | 0.643 |
| Random-init embedding (384-d) | 0.520 +- 0.027 | 0.700 | -0.053 | 0.420 |
| Random-init embedding + mismatch | 0.631 +- 0.010 | 0.784 | 0.188 | 0.586 |
| **GENCO embedding (384-d)** | **0.580 +- 0.036** | 0.743 | 0.080 | 0.539 |
| GENCO embedding + mismatch | 0.667 +- 0.004 | 0.807 | 0.266 | 0.642 |
| GENCO embedding + bus-ID + structure + mismatch | 0.670 +- 0.005 | 0.809 | 0.274 | 0.659 |
| PCA-5 of GENCO embedding | 0.238 +- 0.009 | 0.435 | -0.68 | 0.193 |
| PCA-20 of GENCO embedding | 0.492 +- 0.035 | 0.684 | -0.12 | 0.378 |
| PCA-118 of GENCO embedding | 0.570 +- 0.037 | 0.736 | 0.056 | 0.528 |
| PCA-5 / 20 / 118 of random-init embedding | 0.129 / 0.262 / 0.468 | 0.36 / 0.53 / 0.66 | -0.92 / -0.63 / -0.17 | 0.081 / 0.196 / 0.377 |

Per-seed 95% cluster-bootstrap CIs on R^2 are about +-0.03 to +-0.04 wide on each side (e.g. GENCO embedding seed 0: [0.516, 0.592]; bus-ID seed 0: [0.494, 0.563]). The full set is in the CSV.

R^2 for log10 |dVa| at non-slack buses (about 75k bus-rows per model):

| Features | R^2 | Spearman | Skill vs bus mean | R^2, Small s0 |
|---|---|---|---|---|
| Local mismatch only | 0.046 +- 0.041 | 0.287 | 0.03 | 0.067 |
| Bus-ID one-hot | 0.021 +- 0.002 | 0.163 | 0.00 | 0.085 |
| Bus-ID + mismatch | 0.246 +- 0.103 | 0.534 | 0.230 | 0.338 |
| Random-init embedding | 0.003 +- 0.008 | 0.118 | -0.02 | 0.054 |
| Random-init embedding + mismatch | 0.189 +- 0.097 | 0.463 | 0.172 | 0.245 |
| GENCO embedding | 0.014 +- 0.017 | 0.148 | -0.01 | 0.106 |
| GENCO embedding + mismatch | 0.206 +- 0.073 | 0.485 | 0.189 | 0.298 |
| GENCO embedding + bus-ID + structure + mismatch | 0.202 +- 0.079 | 0.477 | 0.185 | 0.314 |

The across-seed SD for angles is large because seed 42 is much worse than seeds 0 and 1 (e.g. bus-ID + mismatch 0.13 vs 0.32 and 0.29).

## Paired contrasts (difference in R^2; mean +- SD over the 3 Base seeds; "CI>0" = seeds whose paired cluster-bootstrap 95% CI excludes 0)

| Contrast | dVm: dR^2 | CI excludes 0 | dVa: dR^2 | CI excludes 0 |
|---|---|---|---|---|
| Embedding - bus-ID | +0.027 +- 0.009 | 1 of 3 | -0.007 +- 0.016 | 0 of 3 |
| Embedding - random-init embedding | +0.060 +- 0.010 | 3 of 3 | +0.011 +- 0.022 | 0 of 3 |
| Embedding + mis - bus-ID + mis | +0.015 +- 0.004 | 2 of 3 (marginal) | -0.040 +- 0.032 | 2 of 3 (negative) |
| Embedding + mis - random emb + mis | +0.035 +- 0.007 | 3 of 3 | +0.017 +- 0.024 | 0 of 3 |
| Embedding + all - (bus-ID + struct + mis) | +0.017 +- 0.002 | 3 of 3 | -0.039 +- 0.026 | 2 of 3 (negative) |
| PCA-118 emb - bus-ID (matched dim.) | +0.016 +- 0.011 | 1 of 3 | -0.018 +- 0.022 | 0 of 3 |
| PCA-118 emb - PCA-118 random | +0.101 +- 0.018 | 3 of 3 | +0.002 +- 0.025 | 0 of 3 |
| PCA-20 emb - bus-ID | -0.062 +- 0.022 | 3 of 3 (negative) | -0.009 +- 0.018 | 0 of 3 |

## Verdict (blunt)

**For voltage magnitude, most of the earlier "the embedding knows where GENCO is wrong" result was bus identity.**
- Bus-ID alone, a lookup table with 118 numbers and no model, reaches R^2 0.553 on |dVm|. The 384-d embedding reaches 0.580, so identity accounts for about 95% of it. The v1 number (0.63, 0.71 with mismatch) was inflated by sibling leakage and had no such control.
- The embedding adds a small amount beyond identity for |dVm|: +0.027 R^2 alone, and +0.015 with mismatch. These are not significant in every seed, and the largest is +0.017 in the "everything" comparison (embedding + bus-ID + structure + mismatch vs the same without the embedding). The skill score says the same: the embedding explains about 8% of the within-bus error variance (random-init network: negative). With local mismatch the figure is about 27% (bus-ID + mismatch: 23%).
- A trained GENCO is better than a random-weights GENCO of the same architecture by a consistent +0.06 R^2 (3 of 3 seeds, CI excluding 0). But the random network already reaches 0.52, i.e. 90% of the trained value. So most of the signal is the architecture prior (message passing on the fixed 118-bus topology gives every bus a distinctive code), not learned physics. PCA-118 at matched dimensionality shows the same picture: trained +0.10 over random, only +0.016 over the bus-ID one-hot (CI excludes 0 in one seed). PCA with 5 or 20 components does worse than bus-ID: the bus-ID information is spread over many directions rather than a few dominant axes.
- Best practical predictor: local mismatch plus bus identity (R^2 0.65). The embedding adds +0.015 on top (about 0.67). It does not carry error information in a way that a per-bus calibration table plus the ground-truth-free mismatch cannot already supply.

**For voltage angle, there is nothing in the embedding beyond the mismatch.**
- The embedding alone has R^2 0.01 and the random one 0.00; both are indistinguishable from bus-ID (0.02) and from zero.
- Embedding + mismatch (0.21) is no better than bus-ID + mismatch (0.25). The paired difference is negative (-0.04) and its CI excludes 0 in 2 of 3 seeds. The embedding also matches the random-network figure (0.19): the +0.017 gap to random is not significant.
- The v1 claim that embedding + mismatch gets R^2 0.36 / Spearman 0.67 on angles was a leakage artefact (v2: 0.21 / 0.49).

**What we can still say.** (a) GENCO's per-bus |dVm| error is strongly bus-specific (a bus table gets 0.55). (b) On top of that, local mismatch is the useful ground-truth-free signal, together giving R^2 about 0.65. (c) The trained embedding gives a small, seed-dependent extra for |dVm| (+0.015 to +0.03 R^2) and none for |dVa|. We should not claim that the latents encode "where the model is wrong" beyond bus identity and the architecture prior. The "learned per-element error bound from embeddings" idea in F9 is not supported by this probe; a per-bus table calibrated on held-out load scenarios is the baseline any such method must beat.

## Caveats

- Only 29 load scenarios of the paper-test set (N-1 only), 5 folds of 5-6 groups each. The cluster-bootstrap CI treats out-of-fold predictions as fixed, and does not capture retraining or fold-assignment variability. Three Base seeds is a small n for the SD; seed 42 behaves differently for angles.
- One random-init network per model seed (init seed 1000 + seed); the spread of the random control across seeds is included in its SD, but we did not sweep more draws per seed.
- A bus-ID table is only available for a fixed grid. It cannot transfer to an unseen topology, where an embedding might (not tested here). On a fixed grid the table is the right baseline.
- The ridge is linear. A nonlinear head (small MLP or boosted trees) on the embedding could extract more. The conclusion is about linear decodability.
- Targets: log10 with +1e-9 floor; |dVm| excludes exact zeros (PQ buses never hit it). GENCO's state is the last layer's decoded state with true setpoints, as in v1.
- `scripts/exp04_embedding_probe.py` no longer writes `case118_ieee_base_pca.parquet` or `case118_ieee_base_probe.json`. The old v1 files are still in `results/exp04/` and now stale: `case118_ieee_base_probe.json` should not be cited. `scripts/run_all.sh` calls the script with no arguments, which now runs all four models (about 10 minutes) and writes the `probe_v2*` files.
- RESEARCH_LOG F9 and the report's Table II need rewriting from this note (not edited here per file ownership).
