<!--
DRAFT v2 (2026-10-22) for the Community-tab post. Do not post: the user posts it.
Numbers: docs/NUMBERS.md; v1 review and its fixes: results/notes/review_post_v1.md.
Placeholders: {{GPU}} Colab GPU timing (user), {{AFFIL}} affiliation, {{LINKS}} repo/HF links (repo private until the user says otherwise).
Figures: results/figures/post/ (upload to the discussion before posting; file names below).
-->

# What is GENCO worth to a Newton solver? Pricing a grid foundation model against the pre-contingency state

**Muhammad Sarwar** ({{AFFIL}}) · msarwar@iastate.edu · code, data and checkpoints: {{LINKS}}

## Summary

In contingency analysis an operator never solves from scratch. The solved pre-contingency case is already there, and production tools start every outage's Newton (or fast-decoupled) solve from it. So we price GENCO's power-flow prediction against that start, in solver work, at equal safety.

**Answer.** On IEEE 118, starting Newton from GENCO instead of the pre-contingency state saves 0.49 [0.45, 0.53] iterations to 10⁻⁴ MW at N-1 and about one iteration at N-5. A screening rule that verifies each contingency with Newton steps and a calibrated error bound needs 18–44% fewer factorizations from GENCO's start. It missed no overload in 429 held-out operating points (one-sided 95% bound 0.7%). The advantage shrinks with grid size: 0.33 iterations at 500 buses, and at 2,000 buses the pre-contingency start is better by 0.63. On a laptop, inference costs 14× to 580× more than the solver work it saves {{GPU}}.

Two further findings. First, GENCO's PF task treats the slack-bus voltage magnitude as an unknown although it is a setpoint, which makes the reported power balance 32–60× better than the physical one on IEEE 14–118. Second, a residual decoder that starts exactly at the pre-contingency state (GENCO-Δ) cuts N-1 state error 2.5×, but saves Newton work only at a loose tolerance and regresses at N-5. Retraining on mixed outages trades the N-1 gain for the N-5 fix.

![Fig. 1](figA_workflow.png)
*Fig. 1. Verified screening. Either the pre-contingency state or GENCO's prediction starts float64 Newton steps on the exact admittances. After each step a calibrated bound c·‖dx‖² on every monitored branch loading and PQ-bus voltage decides whether to accept the screen; after three steps the case goes to a full solve.*

## I. The question

GENCO's paper [1] reports speed and accuracy against PowerModels from a cold start. A control room does not solve contingencies from a cold start. It has the pre-contingency solution, and it usually reuses that solution's factorization in a chord or fast-decoupled solver. So we price GENCO against that baseline. We then follow the answer from 118 to 2,000 buses, and try to make GENCO use the base case itself.

## II. Setup

**Models.** We use the released GENCO PF checkpoints (Base, Small, Tiny; seeds 0, 1, 42) through gridfm-graphkit [2], unmodified.

**Data.**
- IEEE 118 is tested on the paper's held-out N-1 scenarios (638, in 29 load scenarios).
- It is also tested on four new gridfm-datakit sets at N-1, N-2, N-3 and N-5, with 100 load scenarios each (8,792 scenarios).
- A separate N-1 set (2,200 scenarios) is used only for calibration.
- GOC 500 uses the paper split; GOC 2000 uses 879 scenarios held out from training.

**Solver.** A float64 Newton–Raphson runs on exactly the admittance graph GENCO sees, with PV and slack setpoints enforced. The pre-contingency start is the solved base case of the same operating point, and is treated as free.

**Statistics.** Contingencies that share an operating point are correlated. Intervals are therefore 95% cluster bootstraps over load scenarios, and calibration is split by load scenario.

## III. The exchange rate

TABLE II. GENCO minus pre-contingency start, mean Newton iterations (paired) [95% CI]. Released Base seed 0, except GOC 2000 (Small).

| Grid, set | n (load scen.) | at 10⁻⁴ MW: pre-cont. → GENCO | Δ at 10⁻² MW | Δ at 10⁻⁴ MW | Δ at 10⁻⁶ MW |
|---|---|---|---|---|---|
| IEEE 118, paper N-1 | 638 (29) | 2.49 → 2.01 | −0.32 [−0.45, −0.19] | −0.49 [−0.53, −0.45] | −0.73 [−0.83, −0.63] |
| IEEE 118, N-1 | 2,200 (100) | 2.51 → 2.00 | −0.37 [−0.43, −0.30] | −0.50 [−0.53, −0.48] | −0.79 [−0.83, −0.75] |
| IEEE 118, N-5 | 2,194 (100) | 3.08 → 2.07 | −0.65 [−0.69, −0.62] | −1.01 [−1.02, −0.99] | −0.96 [−1.00, −0.92] |
| GOC 500, paper | 638 (29) | 2.60 → 2.27 | −0.10 [−0.14, −0.07] | −0.33 [−0.46, −0.18] | −0.04 [−0.13, 0.04] |
| GOC 2000, held-out | 879 (40) | 2.37 → 3.00 | +0.57 [0.40, 0.72] | +0.63 [0.59, 0.67] | +0.22 [0.12, 0.35] |

**IEEE 118.** The saving holds for all three Base seeds, and for Small and Tiny. It also grows with outage order, from 0.5 iterations at N-1 to 1.0 at N-5 (at 10⁻⁴ MW). The further the outage moves the state, the more the prediction is worth.

**GOC 2000.** The pre-contingency start wins in about 60% of scenarios at 10⁻² and 10⁻⁴ MW (Fig. 2). We tested three grids with one model each, so this is a trend, not a law.

**Best-of-two.** A rule that starts from whichever of the two has the smaller mismatch, at no extra factorization, comes within 0.1 iteration of the better start on IEEE 118 (2.25 vs 2.15). On GOC 2000 it ties the pre-contingency start (2.92 vs 2.93).

![Fig. 2](figB_exchange_rate.png)
*Fig. 2. (a) Newton iterations from flat, DC, pre-contingency and GENCO starts; (b) paired GENCO minus pre-contingency difference, with 95% cluster-bootstrap CIs. Open dots: IEEE 118 seeds 1 and 42 on their own splits.*

## IV. Factor reuse and wall-clock time

Production contingency analysis seldom refactorizes. TABLE III repeats the comparison for two solvers that reuse the base-case factorization and compensate the outage with a low-rank Woodbury update: chord Newton, and fast-decoupled XB [3].

The saving depends a great deal on the solver:
- **IEEE 118:** 21% of triangular solves with chord Newton, but 5% with fast-decoupled. With fast-decoupled, the GENCO start is worse than the pre-contingency start in 23% of contingencies.
- **GOC 500:** 11% with chord Newton, and −7% with fast-decoupled.

TABLE III. Saving of a GENCO start over the pre-contingency start, IEEE 118 paper set, 10⁻⁴ MW. Break-even = solve time saved per contingency, i.e. what GENCO's inference must beat (Python/SciPy, Apple M4, shared machine).

| Solver | Triangular-solve saving [95% CI] | Break-even (ms) |
|---|---|---|
| Full Newton | 20% [18, 21] | 0.31 (0.63 at N-5) |
| Chord + compensation | 21% [20, 21] | 0.04 |
| Fast-decoupled + compensation | 5% [2, 8] | 0.015 |

We measured GENCO Base inference at 8.7 ms per contingency at the best batch size on the M4's GPU (MPS), and 12–24 ms on CPU. That is 14× the largest break-even and 580× the fast-decoupled one.

The cross-solver comparison is starker still. Even with free inference, GENCO plus full Newton (1.25 ms) is slower than fast-decoupled from the pre-contingency state (0.35 ms). Against the cheapest solver an operator already runs, the break-even is negative. {{GPU: T4/A100 inference per contingency from notebooks/colab_gpu_timing.ipynb, and whether any same-solver break-even is crossed.}}

![Fig. 3](figD_factor_reuse_breakeven.png)
*Fig. 3. (a) Triangular solves saved by a GENCO start, IEEE 118 and GOC 500, 10⁻⁴ MW. (b) Break-even time per contingency against measured GENCO inference (filled: batch 16; open: batch 1).*

## V. Taking the saving safely

Near the solution, the Newton step dx approximates the error. So a branch loading computed after *k* steps is within c·‖dx‖² of the truth, for a constant c that we calibrate.

The sequential policy (Fig. 1) works as follows:
- After each Newton step, it checks every monitored element.
- It stops once every element is clearly above or below its limit.
- Otherwise it takes another step, and after three steps it falls back to a full solve.

The same policy runs from either start, so the comparison isolates the prior. c is a split-conformal quantile over the load scenarios of the calibration set [4]. Everything in TABLE IV is held out.

TABLE IV. Sequential policy, conformal α = 0.10. Mean factorizations per contingency [95% CI].

| Test set | GENCO start | Pre-contingency start | Violating elements missed |
|---|---|---|---|
| N-1 | 1.37 [1.30, 1.44] | 2.14 [2.10, 2.17] | 0 / 1,624 |
| N-2 | 1.47 [1.40, 1.53] | 2.51 | 0 / 3,210 |
| N-3 | 1.53 [1.48, 1.59] | 2.73 | 0 / 4,514 |
| N-5 | 1.65 [1.61, 1.69] | 2.96 [2.94, 2.98] | 0 / 6,617 |
| paper N-1 | 1.37 [1.23, 1.51] | 2.09 | 0 / 484 |

Neither start misses a violation or raises a false alarm, over 16,449 violating elements in 5,785 violating contingencies. Errors are correlated within an operating point, so the operating point is the honest unit for a bound: 0 misses in 429 load scenarios gives a one-sided 95% bound of 0.7%. Three caveats:
- **The saving depends on the rule.** It is 35–44% with α = 0.10, 33–42% with α = 0.05, and 18–35% with the conservative maximum rule.
- **The bound is conservative.** It was exceeded in 1, 7, 2 and 5% of test operating points at N-1/2/3/5, against a nominal 10%. In hindsight, the largest constant with zero misses is 27–87× smaller than the calibrated one. The costs above are therefore an upper bound on what this kind of rule needs.
- **This is empirical, not proven.** Nothing guarantees the bound under outage-order shift. An earlier calibration on only 29 operating points had a 33% exceedance at N-2.

At equal cost the gap is wide. After one factorization, GENCO plus one step misses 32 of 16,449 violating elements, while the pre-contingency start misses 1,250. The pre-contingency start needs a second factorization to match GENCO's 32.

On GOC 2000 both starts reach zero misses. GENCO, however, never clears the bound after one step, so its 2.00 factorizations (against 2.16) come from the two-step floor rather than from prediction quality.

![Fig. 4](figC_equal_cost.png)
*Fig. 4. Missed violating elements against factorizations per contingency, IEEE 118, four 100-load-scenario test sets. Lines: GENCO + k and pre-contingency + k Newton steps; stars: sequential policies. Hollow markers are zero-miss cells drawn at their one-sided 95% Clopper–Pearson bound.*

## VI. What GENCO computes

**The slack convention.** Graphkit's PF mask (`AddPFHeteroMask`) hides the slack-bus voltage magnitude from GENCO and asks it to predict it. In the data, though, that voltage is a dispatch setpoint, constant across all outages of an operating point. GENCO's predicted slack voltage is off by a median 0.006 p.u. on IEEE 118, and its power balance is evaluated at that voltage.

Restoring the setpoint raises the largest nodal mismatch by 32–60× (TABLE I). The worst bus is adjacent to the slack in 95–100% of scenarios, while the PQ voltages themselves stay accurate. The convention may be intended, but it is undocumented, and it makes reported residuals look far better than the physical state.

TABLE I. Largest nodal mismatch with GENCO's slack voltage (reported) and with the setpoint (physical). Median over scenarios; CI over load scenarios.

| Grid, model | Reported (MW) | Physical (MW) | Ratio |
|---|---|---|---|
| IEEE 14 / 57 Base | 0.0048 / 0.033 | 0.17 / 1.04 | 36× / 32× |
| IEEE 118 Base | 0.29 | 17.4 | 60× [45, 79] |
| IEEE 118 Small / Tiny | 0.45 / 0.58 | 16.8 / 20.2 | 37× / 35× |
| GOC 500 Base | 4.5 | 86 | 19× [12, 29] |
| GOC 2000 Small | 11.0 | 217 | 20× [16, 26] |

Three repairs, none of them free:
- **Feeding the setpoint in.** Giving the Base model the setpoint zero-shot multiplies its state error by 17×.
- **Clamping the output.** Clamping only the output to the setpoint works on Small and Tiny, but degrades Base by 5.7×.
- **Slack-aware fine-tune.** A short fine-tune brings the physical mismatch from 17.2 to 1.0 MW (three seeds). It costs 29% more state error than a matched control and doesn't change Newton iterations.

**Layers.** Tracing the decoder after each of the 12 correction layers, GENCO's mean mismatch on IEEE 118 falls from 261 MW to 0.054 MW. Newton iterations started from intermediate layers stay near 3 until layer 7 and then drop, so the last layers carry the Newton value. Stopping after layer 9 cuts inference by 26% for 0.12 extra iterations.

## VII. Conditioning GENCO on the base case

If the pre-contingency state is the better prior on large grids, GENCO should be given it. We tried two designs on IEEE 118, each with three seeds and a matched control, and selected checkpoints on validation loss only.
- **Plain Δ** writes the base-case voltages (and optionally branch flows) into GENCO's masked inputs and fine-tunes. The two seeds that trained ended at about base-case quality: 2.85–2.95 iterations to 10⁻⁶ MW, against 2.15 for the released model. The third seed diverged.
- **Residual Δ** leaves the inputs alone. Each layer decodes the base case plus a delta head initialised to zero, so before training the model reproduces the base case exactly. With learning rate 3e-4 (the best of three residual variants by validation loss):
  - The paper-split state error falls from 0.054 to 0.022.
  - Iterations to 10⁻² MW fall from 1.74 to 1.33. At 10⁻⁴ MW there is no gain (2.05 vs 2.01).
  - At N-5 it is worse than released GENCO (error 0.155 vs 0.112; Fig. 5).

The same information is useless as an input and useful as a starting point. The released training data are N-1 only, so the N-5 regression may be a training-distribution effect. We tested this by retraining the residual model on 5,056 fresh scenarios with one to five elements out (three seeds). That removes the N-5 regression (error 0.116 vs 0.155; one-step misses 9 vs 18 of 1,763), but it also removes the N-1 gain (0.056 vs 0.022). With this training budget, the model ends at about released-GENCO quality at every outage order. The training scenarios are new as well as differently mixed, so the comparison is not perfectly controlled.

The base case also carries the true slack voltage. The slack-aware arm alone gains nothing, though, so that is not where the improvement comes from.

![Fig. 5](figF_genco_delta.png)
*Fig. 5. GENCO-Δ on IEEE 118 Base: (a) median state error, (b) Newton iterations to 10⁻² MW; mean ± SD over seeds 0/1/42. Purple bars: the pre-contingency state itself. Stars: residual Δ retrained on mixed N-1…N-5 outages. Crosses: the diverged plain-Δ seed, excluded from the mean.*

## VIII. What did not work

TABLE V. Negative and null results.

| Idea | Outcome |
|---|---|
| GENCO error warnings from its embeddings | A 118-entry bus lookup explains R² 0.55 of the per-bus voltage error, GENCO's embeddings 0.58, a randomly initialised GENCO 0.52 |
| Per-bus bias correction | 0.3% of squared error in-sample, nothing out of sample |
| Accepting GENCO with no Newton step | Its own residual never certifies a whole contingency |
| Convergence rescue | Flat-start Newton converged on every datakit scenario up to N-5 |
| Plain GENCO-Δ | Base-case quality; one seed diverged |

## IX. Limitations

- **Empirical safety.** All safety evidence is empirical and comes from datakit scenarios. Calibration and test sets follow the same daily load trajectory, so the operating-point bound is optimistic.
- **Coverage.** Factor-reuse results cover IEEE 118 and GOC 500 at seed 0 only.
- **GENCO-Δ scope.** It was run on IEEE 118 only, although its motivation is the large grids. The GOC 2000 run is prepared but not done.
- **Timings.** They come from Python on a shared laptop. A compiled solver would lower every break-even further.
- **Reproduction.** Our reproduction of the released metrics uses about 3% of each paper test set. Mean metrics agree within 6% for most runs (worst 13%).

## X. Open science

Everything is at {{LINKS}} under Apache-2.0:
- the graphkit plugins (slack-aware task, GENCO-Δ, residual decoder);
- the float64 Newton, chord and fast-decoupled solvers, with tests;
- the datakit configs for every new set, and one script per table;
- a SHA-256 manifest with a verifier, and the fine-tuned checkpoints;
- a small console that walks one contingency through the policy.

Iteration and miss counts regenerate exactly. Timings depend on the machine and are given with the hardware. Every number in this post is traced to a result file in `docs/NUMBERS.md`.

## References

[1] "GENCO: a grid foundation model architecture for power flow, optimal power flow and state estimation," arXiv:2608.09921, 2026. {{author list}}
[2] OpenGridFM, gridfm-graphkit (commit 8a8defb) and gridfm-datakit (commit d8a092d).
[3] B. Stott and O. Alsaç, "Fast decoupled load flow," *IEEE Trans. Power App. Syst.*, vol. PAS-93, no. 3, pp. 859–869, 1974.
[4] A. N. Angelopoulos and S. Bates, "A gentle introduction to conformal prediction and distribution-free uncertainty quantification," arXiv:2107.07511, 2021.
