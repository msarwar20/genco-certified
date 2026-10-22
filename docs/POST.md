<!--
DRAFT for the Community-tab post. Do not post: the user posts it. Every number is from docs/NUMBERS.md (section in brackets in the source comments).
Placeholders: {{NK}} N-k retraining (agent N, running), {{GPU}} Colab GPU timing (user, 2026-10-22), {{AFFIL}} affiliation, {{LINKS}} repo/HF links (repo is private until the user says otherwise).
Figures: results/figures/post/ (agent V). Upload to the discussion or an HF dataset repo before posting.
-->

# What is GENCO worth to a Newton solver? Pricing a grid foundation model against the pre-contingency state

**Muhammad Sarwar** ({{AFFIL}}), contact: msarwar@iastate.edu
Code: {{LINKS}} · Apache-2.0 · every number below regenerates from one script per table

## Summary

GENCO predicts an AC power-flow state in a single forward pass. In contingency analysis, however, an operator rarely solves from scratch: the solved pre-contingency case is already on the screen and is the warm start every production tool uses. So we asked a narrow, practical question: **if GENCO's prediction is handed to a float64 Newton solver, how much solver work does it save compared with that pre-contingency start, and can the saving be taken without missing an overload?**

The short answer, on the released checkpoints:

1. On IEEE 118, a GENCO prediction is worth **0.73 Newton iterations** [0.63, 0.83] to 10⁻⁶ MW over the pre-contingency start, and about **one iteration at equal cost** for screening: GENCO plus one Newton step misses 32 of 18,030 violating elements, the same as the pre-contingency start plus *two* steps (33).
2. A sequential accept-or-escalate rule, calibrated on a separate 100-operating-point set, gave **0 missed violations out of 18,030** (95% upper bound 0.02%) with **35–44% fewer factorizations** than the same rule started from the pre-contingency state.
3. The advantage **does not survive scale**: it is a tie at 500 buses and the pre-contingency start wins at 2,000 buses (+0.22 iterations). Under industrial factor reuse it shrinks to 5–21% of triangular solves, and pays off in wall-clock time only if inference costs under **0.015–0.6 ms** per contingency. GENCO takes 8–24 ms on a laptop {{GPU}}.
4. An audit finding: the PF mask treats the **slack-bus voltage magnitude as an unknown** although it is a setpoint, so GENCO's reported power balance looks **35–60× better** than the physical one on IEEE 14–118.
5. An extension, **GENCO-Δ**, that conditions GENCO on the pre-contingency state. Feeding the base case into the masked inputs fails (one of three seeds diverged); a residual decoder that starts exactly at the base case cuts N-1 state error **2.5×** and Newton iterations at a practical tolerance by about 25%, but regresses at N-5 {{NK}}.

Negative results are reported in the main table (Table V), not a footnote.

![Fig. 1](figA_workflow.png)
*Fig. 1. Verified screening. After an outage, either the pre-contingency state or GENCO's prediction starts float64 Newton steps on the exact admittances. A calibrated bound c·‖dx‖² on every monitored branch loading and PQ voltage decides, after each step, whether to accept the screen or take another step; after three steps the case goes to a full Newton solve.*

## I. Why price GENCO in Newton iterations?

Most learned power-flow work, and GENCO's own paper [1], compares the network against a flat start or a DC approximation. Neither is what a control room uses for contingency analysis. There, the base-case solution is computed once per operating point, and each outage is solved by Newton (or fast-decoupled) iterations started from it, often reusing the base-case factorization. A learned model is only useful in that loop if it beats this start, in solver work, at the same level of safety.

This makes the question complementary to two other entries. "another entry" (#7) checks GENCO with Newton for N-k screening and finds roughly one iteration saved and no speedup on the Texas grid; another entry (#2) audits sensitivities on fixed topologies. We add the pre-contingency baseline, the factor-reuse solvers operators actually run, a scale trend from 118 to 2,000 buses, and an attempt to make GENCO itself use the base case.

## II. Setup

**Models and data.** Released GENCO PF checkpoints (Base, Small, Tiny; seeds 0, 1, 42) on IEEE 14–118, GOC 500 and GOC 2000, run through gridfm-graphkit (commit 8a8defb) unchanged. Test sets: the paper's held-out N-1 scenarios (638, 29 load scenarios), plus new gridfm-datakit sets for IEEE 118 with 100 load scenarios each at N-1, N-2, N-3 and N-5 (8,792 scenarios), a separate N-1 calibration set (2,200), an exhaustive N-1 set (every branch and generator, 6,900 contingencies), and 879 held-out GOC 2000 scenarios (40 load scenarios) not seen in training.

**Solver.** A float64 Newton–Raphson on exactly the admittance graph GENCO sees, with PV/slack setpoints enforced. The pre-contingency start is the solved base case of the same operating point. We count Newton iterations to 10⁻², 10⁻⁴ and 10⁻⁶ MW, sparse LU factorizations, and, for chord and fast-decoupled solvers, triangular solves. Counts are hardware independent; wall-clock times are reported separately.

**Statistics.** Contingencies that share an operating point are correlated, so every interval is a 95% cluster bootstrap over load scenarios, calibration is split by load scenario, and miss rates carry Clopper–Pearson bounds. Base seeds are reported as mean ± SD.

## III. What GENCO is doing

**Layer dynamics.** Tracing the physics decoder after each of the 12 correction layers, GENCO's own mismatch on IEEE 118 falls from 261 MW after layer 1 to 0.054 MW after layer 12, roughly geometrically after layer 2 (about 0.6 per layer). Stopping after layer 9 cuts inference by 26% and costs 0.12 Newton iterations on Base (0.01 on Small and Tiny). Iterations from layer *k* plateau near 3 until layer 7 and then drop, so the last few layers are what make the prediction worth something to Newton.

**The slack convention.** `AddPFHeteroMask` masks the slack-bus voltage magnitude, while in the data it is a dispatch setpoint (constant across all outages of an operating point). GENCO therefore predicts it, with a median error of 0.006 p.u. on IEEE 118, and its power balance is computed at that predicted voltage. With the setpoint restored, the largest nodal mismatch rises from 0.29 MW to 17.4 MW (TABLE I). The error is local: the worst bus is adjacent to the slack in 97–99% of scenarios. This may well be intended, but it is undocumented, and it makes reported residuals look far better than the physical state they describe.

TABLE I. Largest nodal mismatch, reported (GENCO's own slack Vm) vs physical (setpoint restored); median over scenarios, 95% CI over load scenarios.

| Grid, model | Reported (MW) | Physical (MW) | Ratio |
|---|---|---|---|
| IEEE 14 Base | 0.0048 | 0.173 | 36× [31, 46] |
| IEEE 118 Base | 0.288 | 17.4 | 60× [45, 79] |
| IEEE 118 Small / Tiny | 0.454 / 0.583 | 16.8 / 20.2 | 37× / 35× |
| GOC 500 Base | 4.53 | 86.3 | 19× [12, 29] |
| GOC 2000 Small (879 held-out) | 11.0 | 217 | 20× [16, 26] |

Two zero-shot repairs fail on Base. Feeding the setpoint in as an input multiplies the state error by 17×, and re-anchoring all voltages makes things worse in 95–100% of scenarios. Clamping only the output works on Small and Tiny. A short slack-aware fine-tune fixes the balance (physical mismatch 17.4 → 1.0 MW) but costs about 29% state error against a matched control over three seeds, and saves no Newton iterations.

## IV. The exchange rate

TABLE II. Mean Newton iterations to 10⁻⁶ MW, IEEE 118 Base seed 0, and the paired GENCO − pre-contingency difference with 95% CI.

| Set | Flat | DC | Pre-contingency | GENCO | GENCO − pre-cont. |
|---|---|---|---|---|---|
| paper N-1 | 4.00 | 3.97 | 2.88 | 2.15 | −0.73 [−0.83, −0.63] |
| N-1 (100 load scen.) | 4.00 | | 2.89 | 2.10 | −0.79 [−0.83, −0.75] |
| N-2 | 4.01 | | 3.10 | 2.17 | −0.93 [−0.97, −0.88] |
| N-3 | 4.02 | | 3.20 | 2.21 | −0.99 [−1.02, −0.95] |
| N-5 | 4.03 | | 3.34 | 2.38 | −0.96 [−1.00, −0.92] |

On IEEE 118 the saving grows with outage order: the further the post-contingency state drifts from the base case, the more GENCO's prediction is worth. It holds over three seeds (2.15 ± 0.09) and for Small (2.22) and Tiny (2.18). At looser tolerances it is smaller (−0.32 at 10⁻² MW), because both starts converge quickly there.

Fig. 2 shows the scale trend. At 10⁻⁶ MW the difference is −0.73 on IEEE 118, −0.04 [−0.13, 0.04] on GOC 500, and **+0.22 [0.12, 0.35] on GOC 2000**, where the pre-contingency start is the better prior in about 60% of scenarios at 10⁻² and 10⁻⁴ MW. A best-of-two rule (start from whichever has the smaller mismatch, which costs no factorization) is never worse than either start: 2.25 on IEEE 118 and 2.92 on GOC 2000.

![Fig. 2](figB_exchange_rate.png)
*Fig. 2. (a) Newton iterations from flat, DC, pre-contingency and GENCO starts; (b) paired GENCO − pre-contingency difference. 95% cluster-bootstrap CIs over load scenarios. IEEE 118 Base (n = 638, 29 load scenarios; open dots: seeds 1 and 42 on their own splits), GOC 500 Base (n = 638), GOC 2000 Small (held-out, n = 879, 40 load scenarios).*

**Factor reuse.** Production contingency analysis seldom refactorizes. With a chord Newton that reuses the base-case factorization and compensates the outage with a Woodbury update, GENCO saves 21% [20, 21] of triangular solves on IEEE 118; with fast-decoupled XB, only 5% [2, 8], and the GENCO start is worse in 23% of contingencies (TABLE III). On GOC 500 the chord saving is 11% and fast-decoupled loses 7%. So "one Newton iteration" is not a transferable price: it depends on the solver it is paid in (Fig. 3a).

TABLE III. Paired saving of a GENCO start over the pre-contingency start, IEEE 118 paper set, 10⁻⁴ MW.

| Solver | Triangular-solve saving | Break-even inference (ms/contingency) |
|---|---|---|
| Full Newton | 20% (18–21) | 0.31 (0.63 at N-5) |
| Chord + Woodbury compensation | 21% (20–21) | 0.04 |
| Fast-decoupled XB + compensation | 5% (2–8) | 0.015 |

![Fig. 3](figD_factor_reuse_breakeven.png)
*Fig. 3. (a) Triangular solves saved by a GENCO start over the pre-contingency start, to 10⁻⁴ MW, IEEE 118 and GOC 500. (b) Break-even inference time per contingency against measured GENCO inference on an Apple M4 (filled: batch 16; open: batch 1). {{GPU row}}*

## V. Taking the saving safely

The Newton step gives a free error estimate. Near the solution the step ‖dx‖ approximates the error, so a branch loading computed after *k* steps is off by at most c·‖dx‖² for a constant c that we calibrate. The sequential policy (Fig. 1) checks every monitored branch and PQ voltage after each step, accepts the contingency as safe or violated once every element clears its margin, and otherwise continues, falling back to a full solve after three steps. The same policy runs from either start, so the comparison isolates the prior.

We calibrate c on a separate N-1 set (100 load scenarios) with split-conformal quantiles over load scenarios, and test on everything else: 100-load-scenario N-1, N-2, N-3 and N-5 sets plus the paper set.

- **Misses.** 0 of 18,030 violating elements and 0 of 6,531 violating contingencies are missed, with no false alarms, for every rule and both starts (one-sided Clopper–Pearson upper bound 0.017% and 0.046%).
- **Cost.** GENCO needs 1.37 / 1.47 / 1.53 / 1.65 factorizations per contingency at N-1/2/3/5, against 2.14 / 2.51 / 2.73 / 2.96 from the pre-contingency start: 35–44% fewer (TABLE IV, Fig. 4).
- **Calibration.** The bound is exceeded in 1, 7, 2 and 5% of test operating points at nominal 10%. That is consistent with exchangeability under outage-order shift, but it is not a guarantee of it.

An earlier version, calibrated on 29 operating points, exceeded its nominal rate by up to 33% at N-2. That is why the separate calibration set exists.

TABLE IV. Sequential policy, conformal α = 0.10, calibrated on N-1 (100 load scenarios). Mean factorizations per contingency [95% CI]; misses are elements.

| Test set | GENCO start | Pre-contingency start | Misses (both) |
|---|---|---|---|
| N-1 | 1.37 [1.30, 1.44] | 2.14 [2.10, 2.17] | 0 / 1,624 |
| N-2 | 1.47 [1.40, 1.53] | 2.51 | 0 / 3,210 |
| N-3 | 1.53 [1.48, 1.59] | 2.73 | 0 / 4,514 |
| N-5 | 1.65 [1.61, 1.69] | 2.96 [2.94, 2.98] | 0 / 6,617 |
| paper N-1 | 1.37 [1.23, 1.51] | 2.09 | 0 / 484 |

![Fig. 4](figC_equal_cost.png)
*Fig. 4. Missed violating elements against factorizations per contingency, IEEE 118 Base seed 0, four 100-load-scenario test sets. Lines: GENCO + k and pre-contingency + k Newton steps; stars: sequential policies (conformal α = 0.10, calibrated on a separate N-1 set). Hollow markers are zero-miss cells drawn at their one-sided 95% Clopper–Pearson bound.*

On GOC 2000 both starts also reach zero misses (0/173), but GENCO's predictions never clear the margin after one step, so its policy costs 2.00 factorizations against 2.16. That saving comes from the two-step floor, not from prediction quality.

## VI. Runtime

Measured on an Apple M4: GENCO Base takes 8.7 ms per IEEE 118 contingency at the best batch size (MPS) and 24 ms unbatched on CPU; GOC 2000 Small takes 67–108 ms. The break-even values in TABLE III are 12× to 1,600× smaller. On this hardware, GENCO does not pay in wall-clock time under any classical solver we tried, even though it pays in iterations. {{GPU: one sentence with the T4/A100 numbers from notebooks/colab_gpu_timing.ipynb and whether any break-even is crossed.}} Where no base case exists at all (a flat start costs 2.6 ms here), a GENCO start would be worth up to about 1.3 ms.

## VII. GENCO-Δ: teaching GENCO to use the base case

If the pre-contingency state is the stronger prior at scale, the obvious extension is to give it to GENCO. We tried two ways, on IEEE 118 with three seeds, a matched control, and checkpoints selected on validation loss only.

- **Plain Δ** fills GENCO's masked inputs with the base-case voltages (and, in one arm, pre-contingency branch flows) and fine-tunes. It ends at about base-case quality (2.85–2.95 iterations to 10⁻⁶ MW against the released model's 2.15), and seed 42 diverged in both arms. The released weights read the filled inputs as a large distribution shift and do not recover reliably.
- **Residual Δ** keeps the inputs and changes the decoder. Each layer decodes the base case plus the output of a delta head initialised to zero, so at step 0 the model reproduces the base case bit for bit, and training only has to learn the outage's effect. On N-1 it cuts median state error from 0.054 to 0.022 and iterations to 10⁻² MW from 1.74 to 1.33 (Fig. 5). At 10⁻⁶ MW it ties (2.19 vs 2.15), and it gets worse as outage order rises: at N-5 the error is 0.155 against 0.112.

The contrast between the two is the useful lesson. The same information is useless or helpful depending on whether the architecture lets the network start from it. The N-5 regression is plausibly a training-distribution effect, since the released data are N-1 only. {{NK: result of residual Δ retrained on a fresh mixed N-1..N-5 set (5,056 scenarios, seed 4001), three seeds; one or two sentences and a row in Fig. 5. Note the confound: different (new) training scenarios.}}

![Fig. 5](figF_genco_delta.png)
*Fig. 5. GENCO-Δ on IEEE 118 Base: (a) median state error and (b) Newton iterations to 10⁻² MW, mean ± SD over init seeds 0/1/42, on the paper set and stress N-1 to N-5. Purple bars: the pre-contingency state itself. Plain Δ seed 42 diverged and is shown as a cross, excluded from the mean.*

## VIII. What did not work

TABLE V. Negative and null results.

| Idea | Outcome |
|---|---|
| GENCO beats the pre-contingency start at scale | Tie at 500 buses, loses at 2,000 (+0.22 iterations) |
| Wall-clock gain on a laptop | None; inference is 12–1,600× above break-even |
| Embedding probe ("latents know where GENCO is wrong") | Withdrawn: bus identity alone gives R² 0.55 vs 0.58 for embeddings, random weights 0.52 |
| Per-bus bias correction | Removes 0.3% of squared error in-sample, nothing out of sample |
| Slack fix by mask surgery (Base) | Feeding the setpoint in: 17× state error |
| Slack-aware fine-tune | Honest balance; about 29% worse state error, no iteration gain |
| Plain GENCO-Δ | Base-case quality; one seed diverged |
| Residual GENCO-Δ at N-5 | Worse than released {{NK}} |
| Accept GENCO with no Newton step | Its own residual never certifies a whole contingency |
| Convergence rescue | Flat-start Newton never failed up to N-5, so nothing to rescue |

## IX. Limitations

- **Empirical safety only.** The zero-miss result is empirical, on datakit scenarios. Calibration and test sets share datakit's daily load trajectory, so the operating-point bounds are optimistic. A Newton–Kantorovich certificate would give a proof, but it does not hold at GENCO's raw output.
- **Narrow coverage.** Factor-reuse results cover IEEE 118 and GOC 500 at seed 0. GENCO-Δ was run on IEEE 118 only; GOC 2000 is prepared but not run.
- **Reproduction subset.** The reproduction uses about 3% of each paper test set. Mean metrics agree within about 6% (worst 13%); maxima and RMSE deviate more.
- **Python timings.** Timings are from Python/SciPy, and a compiled solver would push the break-even lower still.

## X. Open science

{{LINKS}}. The repository contains:
- the gridfm-graphkit plugins (slack-aware task, GENCO-Δ, residual decoder);
- the float64 Newton, chord and fast-decoupled solvers, with tests;
- the datakit configs for every new set;
- one script per table;
- a SHA-256 manifest with a verifier;
- the fine-tuned checkpoints (Hugging Face);
- a small operator console that walks one contingency through the policy.

`docs/REPRODUCE.md` lists the commands from a fresh clone to every table.

## References

[1] "GENCO: a grid foundation model architecture for power flow, optimal power flow and state estimation," arXiv:2608.09921, 2026. {{author list}}
[2] gridfm-graphkit and gridfm-datakit, OpenGridFM, commits 8a8defb and d8a092d.
[3] B. Stott and O. Alsaç, "Fast decoupled load flow," *IEEE Trans. PAS*, vol. 93, no. 3, pp. 859–869, 1974.
[4] A. N. Angelopoulos and S. Bates, "A gentle introduction to conformal prediction and distribution-free uncertainty quantification," arXiv:2107.07511, 2021.
[5] Challenge other entries (another entry) and #7 ("another entry"), GENCO Challenge 2026 Community tab.
