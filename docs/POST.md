<!--
DRAFT v3 (2026-10-25) for the Community-tab post. Do not post: the user posts it.
Numbers: docs/NUMBERS.md, results/notes/ws1_F2_goc2000_reuse.md (GOC 2000 factor reuse), results/notes/confirm_Z.md (confirmation run).
Placeholders: {{GPU}} Colab GPU timing (user), {{AFFIL}}, {{LINKS}} (repo private until the user says otherwise), {{author list}}.
Figures (upload with the post): figH_hero.png, figB_exchange_rate.png, figA_workflow.png, figC_equal_cost.png, figF_genco_delta.png.
-->

# What is GENCO worth to a Newton solver? Pricing a grid foundation model against the pre-contingency state

**Muhammad Sarwar** ({{AFFIL}}) · msarwar@iastate.edu · code, data, checkpoints and demo: {{LINKS}}

## Summary

A learned power-flow model is worth exactly what it saves the solver it feeds. In contingency analysis that solver doesn't start from scratch. It starts from the solved pre-contingency case and usually reuses that case's factorization in a chord or fast-decoupled method. We put GENCO's prediction in front of each of these solvers on three grids and measured what it saves at equal safety.

The short version is Fig. 1. On IEEE 118 a GENCO start saves a fifth of the work for Newton and a twentieth for fast-decoupled. At 2,000 buses it saves 2% for chord Newton and costs extra for the other two.

TABLE 0. Results at a glance (IEEE 118, released checkpoints, unless stated).

| Question | Answer |
|---|---|
| Newton iterations saved over the pre-contingency start, to 10⁻⁴ MW | 0.49 [0.45, 0.53] at N-1, 1.0 at N-5; 0.33 on GOC 500; −0.63 on GOC 2000 |
| Work saved under factor reuse | chord: 21% of triangular solves; fast-decoupled: 5%; at 2,000 buses −18% to +2% |
| Wall-clock break-even | inference must cost under 0.015–0.6 ms per contingency; GENCO takes 8.7 ms on a laptop GPU {{GPU}}. Even with free inference, GENCO + Newton is slower than fast-decoupled from the pre-contingency state |
| Screening with a calibrated Newton check | 0 missed violations in 16,449 held out, and 0 in 17,095 more generated after the rule was frozen, from a different load profile. 37–44% fewer factorizations than the same check from the pre-contingency state |
| What GENCO reports vs what it computes | its power balance looks 32–60× better than the physical one, because the slack-bus voltage setpoint is masked and predicted |
| Teaching GENCO the base case | a residual decoder (GENCO-Δ) cuts N-1 state error 2.5× but saves no iterations at 10⁻⁴ MW; mixed N-k training trades the N-1 gain for an N-5 fix |

![Fig. 1](figH_hero.png)
*Fig. 1. Triangular solves saved by starting from GENCO's prediction instead of the pre-contingency state, to 10⁻⁴ MW, paired over contingencies, 95% cluster-bootstrap CIs over load scenarios. Released checkpoints: IEEE 118 Base (638 contingencies), GOC 500 Base (638), GOC 2000 Small (879 held out). Shaded: the pre-contingency start is cheaper.*

## I. Setup

**Models.** The released GENCO PF checkpoints (Base, Small, Tiny; seeds 0, 1, 42), run through gridfm-graphkit [2] unmodified.

**Data.** IEEE 118 uses the paper's held-out N-1 scenarios (638 in 29 load scenarios) and four new gridfm-datakit sets at N-1, N-2, N-3 and N-5 with 100 load scenarios each (8,792 scenarios). A separate N-1 set (2,200) is used only for calibration. GOC 500 uses the paper split, and GOC 2000 uses 879 scenarios held out from training.

**Solvers.** A float64 Newton–Raphson runs on exactly the admittance graph GENCO sees, with PV and slack setpoints enforced. Chord Newton and fast-decoupled XB [3] reuse one factorization per operating point and handle the outage with a low-rank Woodbury compensation. The pre-contingency start is the solved base case of the same operating point, treated as free.

**Statistics.** Contingencies that share an operating point are correlated. Every interval is therefore a 95% cluster bootstrap over load scenarios, and calibration splits by load scenario.

## II. The price in Newton iterations

TABLE I. GENCO minus pre-contingency start, mean Newton iterations (paired) [95% CI].

| Grid, set | n (load scen.) | at 10⁻⁴ MW: pre-cont. → GENCO | Δ at 10⁻² MW | Δ at 10⁻⁴ MW | Δ at 10⁻⁶ MW |
|---|---|---|---|---|---|
| IEEE 118, paper N-1 | 638 (29) | 2.49 → 2.01 | −0.32 [−0.45, −0.19] | −0.49 [−0.53, −0.45] | −0.73 [−0.83, −0.63] |
| IEEE 118, N-1 | 2,200 (100) | 2.51 → 2.00 | −0.37 [−0.43, −0.30] | −0.50 [−0.53, −0.48] | −0.79 [−0.83, −0.75] |
| IEEE 118, N-5 | 2,194 (100) | 3.08 → 2.07 | −0.65 [−0.69, −0.62] | −1.01 [−1.02, −0.99] | −0.96 [−1.00, −0.92] |
| GOC 500 (Base), paper | 638 (29) | 2.60 → 2.27 | −0.10 [−0.14, −0.07] | −0.33 [−0.46, −0.18] | −0.04 [−0.13, 0.04] |
| GOC 2000 (Small), held-out | 879 (40) | 2.37 → 3.00 | +0.57 [0.40, 0.72] | +0.63 [0.59, 0.67] | +0.22 [0.12, 0.35] |

**IEEE 118.** The saving holds for all three Base seeds and for Small and Tiny. It also grows with outage order: the further the outage moves the state, the more the prediction is worth.

**GOC 2000.** Here the pre-contingency start wins in about 60% of scenarios. A plausible reason, which we have not tested, is that GENCO's prediction error grows with the grid while an outage's footprint stays local. The pre-contingency state would then already be close at most buses.

**Best-of-two.** A rule that starts from whichever has the smaller mismatch, at no extra factorization, comes within 0.1 iteration of the better start on both grids.

![Fig. 2](figB_exchange_rate.png)
*Fig. 2. (a) Newton iterations from flat, DC, pre-contingency and GENCO starts; (b) paired GENCO minus pre-contingency difference, 95% CIs. Open dots: IEEE 118 seeds 1 and 42 on their own splits.*

## III. The price in the solvers operators run

TABLE II. Triangular solves saved by a GENCO start, to 10⁻⁴ MW [95% CI]. Break-even = solve time saved per contingency, i.e. what inference must beat (Python/SciPy, Apple M4, shared machine).

| Solver | IEEE 118 | GOC 500 | GOC 2000 | Break-even, IEEE 118 |
|---|---|---|---|---|
| Full Newton | 20% [18, 21] | 13% [7, 18] | −26% [−29, −24] | 0.31 ms (0.63 at N-5) |
| Chord + compensation | 21% [20, 21] | 11% [10, 12] | 2% [1, 2] | 0.04 ms |
| Fast-decoupled + compensation | 5% [2, 8] | −7% [−9, −4] | −18% [−20, −16] | 0.015 ms |

The saving falls with grid size under every solver, and it falls with solver quality. Fast-decoupled converges linearly from almost any nearby start, so a better initial state buys little. On IEEE 118 the GENCO start is worse than the pre-contingency start in 23% of contingencies.

**Wall-clock time.** GENCO Base inference is 8.7 ms per contingency at the best batch size on the M4's GPU, and 12–24 ms on CPU. That is 14× the largest IEEE 118 break-even and 580× the fast-decoupled one. On GOC 2000, the only positive break-even among these three solvers is 0.11 ms (chord), against 67–108 ms of inference.

**Free inference.** Even if inference cost nothing, GENCO plus full Newton (1.25 ms per contingency) would still be slower than fast-decoupled from the pre-contingency state (0.35 ms).

{{GPU: T4/A100 inference per contingency and whether any same-solver break-even is crossed.}}

## IV. Taking the saving safely

Near the solution, the Newton step dx approximates the error. A branch loading computed after *k* steps is therefore within c·‖dx‖² of the truth, for a constant c that we calibrate.

The sequential policy (Fig. 3) works as follows:
- After each Newton step it checks every monitored branch loading and PQ voltage.
- It stops once every element is clearly above or below its limit.
- Otherwise it takes another step, and after three steps it falls back to a full solve.

The policy is identical from either start, so the comparison isolates the prior. c is a split-conformal quantile [4] over the load scenarios of the calibration set. Everything in TABLE III is held out.

![Fig. 3](figA_workflow.png)
*Fig. 3. The sequential policy. Either start feeds float64 Newton steps; after each step a calibrated bound on every monitored element decides accept or continue; after three steps the case goes to a full solve.*

TABLE III. Sequential policy, conformal α = 0.10, IEEE 118. Mean factorizations per contingency [95% CI].

| Test set | GENCO start | Pre-contingency start | Violating elements missed |
|---|---|---|---|
| N-1 | 1.37 [1.30, 1.44] | 2.14 [2.10, 2.17] | 0 / 1,624 |
| N-2 | 1.47 [1.40, 1.53] | 2.51 | 0 / 3,210 |
| N-3 | 1.53 [1.48, 1.59] | 2.73 | 0 / 4,514 |
| N-5 | 1.65 [1.61, 1.69] | 2.96 [2.94, 2.98] | 0 / 6,617 |
| paper N-1 | 1.37 [1.23, 1.51] | 2.09 | 0 / 484 |

**Misses.** Neither start misses a violation or raises a false alarm, over 16,449 violating elements in 5,785 contingencies. Errors are correlated within an operating point, so we bound per operating point: 0 misses in 429 gives a one-sided 95% bound of 0.7%.

**At equal cost.** At one factorization, GENCO plus one step misses 32 violating elements and the pre-contingency start misses 1,250. The pre-contingency start needs a second factorization to get down to 32 (Fig. 4).

**Caveats.**
- The saving depends on the rule: 35–44% with α = 0.10, 33–42% with α = 0.05, and 18–35% with the conservative maximum rule.
- The bound is conservative. It was exceeded in 1–7% of test operating points against a nominal 10%. The largest constant with zero misses in hindsight is 27–87× smaller than the calibrated one.
- An earlier calibration on only 29 operating points reached 33% exceedance at N-2, which is why calibration now has its own 100-operating-point set.
- On GOC 2000 both starts reach zero misses, but GENCO never clears the bound after one step. Its 2.00 factorizations (vs 2.16) come from the two-step floor, not from prediction quality.

**Confirmation run.** With the rule frozen, we wrote the evaluation protocol and recorded its SHA-256 hash (027cf172…) before generating any data. We then generated four fresh IEEE 118 sets at N-1, N-2, N-3 and N-5 (8,796 scenarios, 100 load scenarios each, new seeds). They use a different load profile, ERCOT East 2024, whose demand trajectory is negatively correlated with the calibration set's.

Nothing was re-tuned. Every one of the 36 rule × policy × set cells had zero misses and zero false alarms: 0 of 17,095 violating elements, 0 of 5,713 violating contingencies, and 0 of 400 operating points (one-sided 95% bound 0.75%). GENCO's policy needed 1.38 / 1.42 / 1.57 / 1.71 factorizations, against 2.19 / 2.53 / 2.77 / 2.98 from the pre-contingency start, i.e. 37–44% fewer. The bound was exceeded in 5–10% of operating points at the nominal 10%, and the Newton saving to 10⁻⁶ MW replicated at 0.83–0.98 iterations.

![Fig. 4](figC_equal_cost.png)
*Fig. 4. Missed violating elements against factorizations per contingency, IEEE 118, four 100-load-scenario test sets. Lines: GENCO + k and pre-contingency + k Newton steps; stars: sequential policies. Hollow markers: zero-miss cells at their one-sided 95% Clopper–Pearson bound.*

## V. What GENCO reports and what it computes

**The slack convention.** Graphkit's PF mask (`AddPFHeteroMask`) hides the slack-bus voltage magnitude from GENCO and asks it to predict it. In the data, though, that voltage is a dispatch setpoint, constant across all outages of an operating point. On IEEE 118 GENCO's predicted slack voltage is off by a median 0.006 p.u., and its reported power balance is evaluated at that predicted voltage.

Restoring the setpoint raises the largest nodal mismatch by 32–60× (TABLE IV), and the worst bus is adjacent to the slack in 95–100% of scenarios. The PQ voltages themselves stay accurate. The convention may be intended, but it is undocumented, and it makes residual metrics look far better than the physical state.

TABLE IV. Largest nodal mismatch with GENCO's slack voltage (reported) and with the setpoint (physical), median; CI over load scenarios.

| Grid, model | Reported (MW) | Physical (MW) | Ratio |
|---|---|---|---|
| IEEE 14 / 57 Base | 0.0048 / 0.033 | 0.17 / 1.04 | 36× / 32× |
| IEEE 118 Base | 0.29 | 17.4 | 60× [45, 79] |
| IEEE 118 Small / Tiny | 0.45 / 0.58 | 16.8 / 20.2 | 37× / 35× |
| GOC 500 Base | 4.5 | 86 | 19× [12, 29] |
| GOC 2000 Small | 11.0 | 217 | 20× [16, 26] |

We tried three repairs, and none is free:
- **Setpoint as input.** Feeding the setpoint into Base zero-shot multiplies its state error by 17×.
- **Output clamp.** Clamping the output works on Small and Tiny but degrades Base by 5.7×.
- **Slack-aware fine-tune.** A short fine-tune brings the physical mismatch from 17.2 to 1.0 MW (three seeds). It costs 29% more state error than a matched control, and it doesn't change Newton iterations.

**Layers.** Tracing the decoder after each of the 12 correction layers, GENCO's mean mismatch on IEEE 118 falls from 261 MW to 0.054 MW. Newton iterations from intermediate layers stay near 3 until layer 7 and then drop, so the last layers carry the Newton value. Stopping after layer 9 cuts inference by 26% for 0.12 extra iterations.

## VI. Teaching GENCO the base case

If the pre-contingency state is the better prior on large grids, GENCO should be given it. We tried two designs on IEEE 118. Each had three seeds and a matched control, and checkpoints were selected on validation loss only.

**Plain Δ** writes the base-case voltages (and optionally branch flows) into GENCO's masked inputs and fine-tunes. The two seeds that trained ended at about base-case quality: 2.85–2.95 iterations to 10⁻⁶ MW, against 2.15 for the released model. The third seed diverged.

**Residual Δ** leaves the inputs alone. Each layer decodes the base case plus a delta head initialised to zero, so before training the model reproduces the base case exactly. For the best of three variants by validation loss:
- N-1 state error falls from 0.054 to 0.022.
- Iterations to 10⁻² MW fall from 1.74 to 1.33.
- At 10⁻⁴ MW there is no gain (2.05 vs 2.01).
- At N-5 it regresses (error 0.155 vs 0.112).

**Mixed N-k retraining.** The released training data are N-1 only, so we retrained on 5,056 fresh scenarios with one to five elements out.
- The N-5 regression goes away: error 0.116 vs 0.155, and one-step misses 9 vs 18 of 1,763.
- The N-1 gain goes away too: error 0.056 vs 0.022.

With this training budget the model ends at about released quality at every outage order (Fig. 5). The same information is useless as an input and useful as a starting point, and where its benefit lands depends on the training distribution.

![Fig. 5](figF_genco_delta.png)
*Fig. 5. GENCO-Δ on IEEE 118 Base: (a) median state error, (b) Newton iterations to 10⁻² MW; mean ± SD over seeds 0/1/42. Purple bars: the pre-contingency state. Stars: residual Δ retrained on mixed N-1…N-5 outages. Crosses: the diverged plain-Δ seed, excluded from the mean.*

## VII. What did not work

TABLE V. Negative and null results.

| Idea | Outcome |
|---|---|
| Error warnings from GENCO's embeddings | A 118-entry bus lookup explains R² 0.55 of the per-bus voltage error, the embeddings 0.58, a randomly initialised GENCO 0.52 |
| Per-bus bias correction | 0.3% of squared error in-sample, nothing out of sample |
| Accepting GENCO with no Newton step | Its own residual never certifies a whole contingency |
| Convergence rescue | Flat-start Newton converged on every datakit scenario up to N-5 |
| Plain GENCO-Δ | Base-case quality; one seed diverged |
| Mixed N-k residual GENCO-Δ | Fixes N-5, loses the N-1 gain |

## VIII. Limitations

- **Empirical safety.** All safety evidence is empirical, on datakit scenarios. The 400 confirmation operating points share one load trajectory, so they are not independent draws.
- **Grid-size trend.** It compares one model size per grid (Base, Base, Small), and the factor-reuse results use seed 0.
- **GENCO-Δ coverage.** GENCO-Δ was run on IEEE 118 only.
- **Timings.** Timings come from Python on a shared laptop. A compiled solver would lower every break-even further.
- **Reproduction subset.** Our reproduction of the released metrics uses about 3% of each paper test set. Mean metrics agree within 6% for most runs (worst 13%).

## IX. Open science

Everything is at {{LINKS}} under Apache-2.0:
- the graphkit plugins (slack-aware task, GENCO-Δ, residual decoder);
- the float64 Newton, chord and fast-decoupled solvers, with tests;
- the datakit configs for every new set, and one script per table and figure, with the figure data as CSV;
- the hashed confirmation protocol;
- a SHA-256 manifest with a verifier, and offline CI;
- the fine-tuned checkpoints;
- an operator console that walks one contingency through the policy.

Iteration and miss counts regenerate exactly. Timings depend on the machine and are reported with the hardware. Every number in this post is traced to a result file in `docs/NUMBERS.md`.

## References

[1] "GENCO: a grid foundation model architecture for power flow, optimal power flow and state estimation," arXiv:2608.09921, 2026. {{author list}}
[2] OpenGridFM, gridfm-graphkit (commit 8a8defb) and gridfm-datakit (commit d8a092d).
[3] B. Stott and O. Alsaç, "Fast decoupled load flow," *IEEE Trans. Power App. Syst.*, vol. PAS-93, no. 3, pp. 859–869, 1974.
[4] A. N. Angelopoulos and S. Bates, "A gentle introduction to conformal prediction and distribution-free uncertainty quantification," arXiv:2107.07511, 2021.
