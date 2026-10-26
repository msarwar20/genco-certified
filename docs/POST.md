<!--
DRAFT v4 (2026-10-26) for the Community-tab post. Do not post: the user posts it.
Numbers: docs/NUMBERS.md; moved material: docs/SUPPLEMENT.md.
Placeholders: {{VIDEO}} (video link, user), title still under discussion.
Figures (upload with the post): figH_hero.png, figB_exchange_rate.png, figA_workflow.png, figC_equal_cost.png, figF_genco_delta.png.
-->

# What is GENCO worth to a Newton solver? Pricing a grid foundation model against the pre-contingency state

**Muhammad Sarwar**, Iowa State University, Ames, IA, USA · **Muhammad Rizwan**, Pakistan Institute of Engineering & Applied Sciences, Islamabad, Pakistan · contact: msarwar@iastate.edu

**Abstract.** We price GENCO's [1] power-flow prediction in the solver work it saves during contingency analysis, measured against the start that operators already have, the solved pre-contingency case, under Newton, chord and fast-decoupled solvers on IEEE 118, GOC 500 and GOC 2000. On IEEE 118 a GENCO start saves 20% of the triangular solves for Newton and 5% for fast-decoupled. A calibrated screening rule built on it needs 37–44% fewer factorizations with no missed overload, which we confirmed once on fresh data generated after the rule was frozen. The saving shrinks with grid size and turns negative at 2,000 buses, and on neither a laptop nor a T4 GPU does inference pay back in wall-clock time against the solvers that operators run.

**Resources.** [Code](https://github.com/msarwar20/genco-certified) · [checkpoints](https://huggingface.co/sarwar21255/genco-ieee118-finetunes) · [data](https://huggingface.co/datasets/sarwar21255/genco-contingency-ieee118) · [interactive demo](https://huggingface.co/spaces/sarwar21255/genco-operator-console) · [video, 3:46]({{VIDEO}}) · [supplement](https://github.com/msarwar20/genco-certified/blob/main/docs/SUPPLEMENT.md). Every number in this post is traced to a result file in [`docs/NUMBERS.md`](https://github.com/msarwar20/genco-certified/blob/main/docs/NUMBERS.md), and the following commands set up the environment and check every stored result against its checksum, with the per-table commands in `docs/REPRODUCE.md`:

```bash
git clone https://github.com/msarwar20/genco-certified && cd genco-certified
bash scripts/setup_env.sh            # Python 3.12 environment with the pinned toolkits
python scripts/verify_artifacts.py   # every result file against SHA256SUMS
```

![Fig. 1](figH_hero.png)

*Fig. 1. Triangular solves saved by starting from GENCO's prediction instead of the pre-contingency state, to 10⁻⁴ MW, paired over contingencies, with 95% cluster-bootstrap CIs over load scenarios. Released checkpoints are IEEE 118 Base (638 contingencies), GOC 500 Base (638) and GOC 2000 Small (879 held out); shaded regions mark where the pre-contingency start is cheaper.*

TABLE 0. Questions and answers (IEEE 118, released checkpoints, unless stated).

| Question | Answer |
|---|---|
| How many Newton iterations does a GENCO start save over the pre-contingency start, to 10⁻⁴ MW? | 0.49 [0.45, 0.53] at N-1 and 1.0 at N-5; 0.33 on GOC 500; −0.63 on GOC 2000 |
| How much work does it save when the factorization is reused? | 21% of triangular solves with chord Newton, 5% with fast-decoupled, and −18% to +2% at 2,000 buses |
| Does it pay back in wall-clock time? | No. Inference must cost under 0.015–0.6 ms per contingency, while GENCO Base takes 2.5 ms on a T4 GPU and 8.7 ms on a laptop GPU; only the smallest model in fp16 crosses one threshold (full Newton, N-5). Even with free inference, GENCO + Newton is slower than fast-decoupled from the pre-contingency state |
| Can the saving be taken without missing overloads? | Yes, empirically: 0 missed violations in 16,449 held out and 0 in 17,095 more generated after the rule was frozen, from a different load profile, with 37–44% fewer factorizations than the same check from the pre-contingency state |
| Is GENCO's reported power balance physical? | Not at the slack bus: the balance looks 32–60× better than the physical one on IEEE 14–118 (about 20× on GOC 500 and 2000), because the slack-bus voltage setpoint is masked and predicted |
| Does conditioning GENCO on the base case help? | Partly: a residual decoder (GENCO-Δ) cuts N-1 state error 2.5× but saves no iterations at 10⁻⁴ MW, and mixed N-k training trades the N-1 gain for an N-5 fix |
| Where does it not help? | On large grids (GOC 2000), under fast-decoupled reuse, and in wall-clock time on every machine we measured |

## I. Setup

**Models and data.** We evaluate the released GENCO PF checkpoints (Base, Small and Tiny, with seeds 0, 1 and 42) through gridfm-graphkit [2] without modification. IEEE 118 is tested on the paper's held-out N-1 scenarios (638 in 29 load scenarios) and on four new gridfm-datakit sets at N-1, N-2, N-3 and N-5 with 100 load scenarios each (8,792 scenarios). A separate N-1 set of 2,200 scenarios is used only for calibration. GOC 500 uses the paper split, and GOC 2000 uses 879 scenarios held out from training.

**Solvers.** A float64 Newton–Raphson solver runs on exactly the admittance graph that GENCO sees, with the PV and slack setpoints enforced. Chord Newton and fast-decoupled XB [3] reuse one factorization per operating point and handle the outage with a low-rank Woodbury compensation. The pre-contingency start is the solved base case of the same operating point, and we treat it as free.

**Statistics.** Contingencies that share an operating point are correlated, so the intervals are 95% cluster bootstraps over load scenarios, the miss-rate bounds are Clopper–Pearson, and calibration splits by load scenario.

## II. The price in Newton iterations

TABLE I. GENCO minus pre-contingency start, mean Newton iterations (paired) [95% CI].

| Grid, set | n (load scen.) | at 10⁻⁴ MW: pre-cont. → GENCO | Δ at 10⁻² MW | Δ at 10⁻⁴ MW | Δ at 10⁻⁶ MW |
|---|---|---|---|---|---|
| IEEE 118, paper N-1 | 638 (29) | 2.49 → 2.00 | −0.32 [−0.45, −0.19] | −0.49 [−0.53, −0.45] | −0.73 [−0.83, −0.63] |
| IEEE 118, N-1 | 2,200 (100) | 2.51 → 2.00 | −0.37 [−0.43, −0.30] | −0.50 [−0.53, −0.48] | −0.79 [−0.83, −0.75] |
| IEEE 118, N-5 | 2,194 (100) | 3.08 → 2.07 | −0.65 [−0.69, −0.62] | −1.01 [−1.02, −0.99] | −0.96 [−1.00, −0.92] |
| GOC 500 (Base), paper | 638 (29) | 2.60 → 2.27 | −0.10 [−0.14, −0.07] | −0.33 [−0.46, −0.18] | −0.04 [−0.13, 0.04] |
| GOC 2000 (Small), held-out | 879 (40) | 2.37 → 3.00 | +0.57 [0.40, 0.72] | +0.63 [0.59, 0.67] | +0.22 [0.12, 0.35] |

On IEEE 118 the saving holds for all three Base seeds and for Small and Tiny, and it grows with outage order, because the further an outage moves the state, the more the prediction is worth. On GOC 2000 the pre-contingency start wins in about 60% of scenarios. A plausible reason, which we have not tested, is that GENCO's prediction error grows with the grid while an outage's footprint stays local, so the pre-contingency state is already close at most buses.

![Fig. 2](figB_exchange_rate.png)

*Fig. 2. (a) Newton iterations from flat, DC, pre-contingency and GENCO starts; (b) the paired GENCO minus pre-contingency difference with 95% CIs. Open dots mark IEEE 118 seeds 1 and 42 evaluated on their own splits.*

## III. The price in the solvers operators run

TABLE II. Triangular solves saved by a GENCO start, to 10⁻⁴ MW [95% CI]. Break-even = solve time saved per contingency, i.e. what inference must beat (Python/SciPy, Apple M4, shared machine).

| Solver | IEEE 118 | GOC 500 | GOC 2000 | Break-even, IEEE 118 |
|---|---|---|---|---|
| Full Newton | 20% [18, 21] | 13% [7, 18] | −26% [−29, −24] | 0.31 ms (0.63 at N-5) |
| Chord + compensation | 21% [20, 21] | 11% [10, 12] | 2% [1, 2] | 0.04 ms |
| Fast-decoupled + compensation | 5% [2, 8] | −7% [−9, −4] | −18% [−20, −16] | 0.015 ms |

The saving falls with grid size under every solver and is smallest for fast-decoupled, plausibly because that method converges linearly at a rate set by its fixed approximate Jacobian, so a better initial state buys only a little. Under fast-decoupled on IEEE 118, the GENCO start is worse than the pre-contingency start in 23% of contingencies.

GENCO Base inference takes 8.7 ms per contingency at the best batch size on the GPU of an Apple M4 and 12–24 ms on its CPU, which is 14× the largest IEEE 118 break-even and 580× the fast-decoupled one. On a Tesla T4 it needs 2.5 ms in fp32 and 1.6 ms in fp16, which is 3.5× faster than the M4 but still 8×, 69× and 168× the full-Newton, chord and fast-decoupled break-evens. The only configuration that crosses any threshold is GENCO Tiny in fp16 against full Newton at N-5, and only under generous assumptions (supplement S2). On GOC 2000 the only positive break-even is 0.11 ms (chord), against 13–108 ms of inference across these devices. Even if inference cost nothing, GENCO plus full Newton (1.25 ms per contingency) would still be slower than fast-decoupled from the pre-contingency state (0.35 ms).

## IV. Taking the saving safely

Near the solution the Newton step dx approximates the error, so a branch loading computed after *k* steps lies within c·‖dx‖² of the truth for a constant c that we calibrate. The sequential policy (Fig. 3) checks every monitored branch loading and PQ voltage after each Newton step, stops once every element is clearly above or below its limit, and otherwise takes another step, falling back to a full solve after three. The rule is identical from either start, although c is calibrated separately for each, and each Newton step costs one factorization. We take c as a split-conformal quantile [4] over the load scenarios of the calibration set, and everything in TABLE III is held out.

![Fig. 3](figA_workflow.png)

*Fig. 3. The sequential policy, in which either start feeds float64 Newton steps and, after each step, a calibrated bound on every monitored element decides whether to accept or continue. After three steps the case goes to a full solve.*

TABLE III. Sequential policy, conformal α = 0.10, IEEE 118. Mean factorizations per contingency [95% CI].

| Test set | GENCO start | Pre-contingency start | Violating elements missed |
|---|---|---|---|
| N-1 | 1.37 [1.30, 1.44] | 2.14 [2.10, 2.17] | 0 / 1,624 |
| N-2 | 1.47 [1.40, 1.53] | 2.51 | 0 / 3,210 |
| N-3 | 1.53 [1.48, 1.59] | 2.73 | 0 / 4,514 |
| N-5 | 1.65 [1.61, 1.69] | 2.96 [2.94, 2.98] | 0 / 6,617 |
| paper N-1 | 1.37 [1.23, 1.51] | 2.09 | 0 / 484 |

Neither start misses a violation or raises a false alarm over 16,449 violating elements in 5,785 contingencies. Because errors are correlated within an operating point, we bound the miss rate per operating point, and 0 misses in 429 gives a one-sided 95% bound of 0.7%. At equal cost the gap is wide: after one factorization, GENCO plus one step misses 32 violating elements over the five held-out sets, whereas the pre-contingency start misses 1,250 and needs a second factorization to get down to 32 (Fig. 4).

The saving depends on the decision rule, ranging over 35–44% with α = 0.10, 33–42% with α = 0.05 and 18–35% with the conservative maximum rule. The bound is also conservative, since it was exceeded in only 1–7% of test operating points against a nominal 10%. In hindsight, the largest constant that gives zero misses is 19–59× smaller than the calibrated one (supplement S3 covers the calibration history and the GOC 2000 case).

**Confirmation run.** The screening rule and its constants were fixed on 12 October 2026 (commit `02a6f6b`). On 25 October we wrote the evaluation protocol and recorded its SHA-256 hash (`027cf172…`) before any data existed. We then generated four fresh IEEE 118 sets for the released Base seed-0 model at N-1, N-2, N-3 and N-5 (8,796 scenarios, 100 load scenarios each, new seeds) and ran the protocol once. These sets use a different load profile, ERCOT East 2024, whose demand trajectory is negatively correlated with that of the calibration set. Nothing was re-tuned, and every one of the 36 cells (3 rules × 3 policies × 4 sets) had zero misses and zero false alarms. That covers 0 of 17,095 violating elements, 0 of 5,713 violating contingencies and 0 of 400 operating points (one-sided 95% bound 0.75%). GENCO's policy needed 1.38 / 1.42 / 1.57 / 1.71 factorizations against 2.19 / 2.53 / 2.77 / 2.98 from the pre-contingency start, i.e. 37–44% fewer; the three operational deviations from the protocol are listed in supplement S4.

![Fig. 4](figC_equal_cost.png)

*Fig. 4. Missed violating elements against factorizations per contingency on IEEE 118, for four 100-load-scenario test sets. Lines show GENCO + k and pre-contingency + k Newton steps, and stars show the sequential policies; hollow markers place zero-miss cells at their one-sided 95% Clopper–Pearson bound.*

## V. What GENCO reports and what it computes

Graphkit's PF mask (`AddPFHeteroMask`) hides the slack-bus voltage magnitude from GENCO and asks it to predict that value, although in the data this voltage is a dispatch setpoint that stays constant across all outages of an operating point. On IEEE 118 GENCO's predicted slack voltage is off by a median 0.006 p.u., and its reported power balance is evaluated at that predicted voltage. Restoring the setpoint raises the largest nodal mismatch by 32–60× on IEEE 14–118 and by about 20× on GOC 500 and 2000 (TABLE IV). The worst bus is adjacent to the slack in 95–100% of scenarios, while the PQ voltages themselves stay accurate. The convention may well be intended, but it is undocumented, and it makes residual metrics look far better than the physical state; none of the three repairs we tried comes without a cost (supplement S5).

TABLE IV. Largest nodal mismatch with GENCO's slack voltage (reported) and with the setpoint (physical), median; CI over load scenarios.

| Grid, model | Reported (MW) | Physical (MW) | Ratio |
|---|---|---|---|
| IEEE 14 / 57 Base | 0.0048 / 0.033 | 0.17 / 1.04 | 36× / 32× |
| IEEE 118 Base | 0.29 | 17.4 | 60× [45, 79] |
| IEEE 118 Small / Tiny | 0.45 / 0.58 | 16.8 / 20.2 | 37× / 35× |
| GOC 500 Base | 4.5 | 86 | 19× [12, 29] |
| GOC 2000 Small | 11.0 | 217 | 20× [16, 26] |

## VI. Conditioning GENCO on the base case

If the pre-contingency state is the better prior on large grids, GENCO should be given it, so we tried two designs on IEEE 118, each with three seeds and a matched control, and selected checkpoints on validation loss only. The plain Δ design writes the base-case voltages (and optionally the branch flows) into GENCO's masked inputs and fine-tunes the model; it ends at about base-case quality, and one of its three seeds diverged. The residual Δ design receives the same filled inputs but decodes, at every layer, the base case plus a delta head initialised to zero, so that before training it reproduces the base case exactly. Its paper-set N-1 state error falls from 0.054 to 0.022 and its iterations to 10⁻² MW from 1.74 to 1.33, although it gains nothing at 10⁻⁴ MW and regresses at N-5, with an error of 0.155 against 0.112.

Because the released training data are N-1 only, we retrained the residual model on 5,056 fresh scenarios with one to five elements out. This removes the N-5 regression (error 0.116 against 0.155) but also the N-1 gain, so with this training budget the model ends at about released quality at every outage order (Fig. 5). With identical inputs, then, the plain design gave nothing while decoding a correction to the base case helped, which suggests that what mattered was how the network used the base case rather than whether it saw it (details in supplement S7).

![Fig. 5](figF_genco_delta.png)

*Fig. 5. GENCO-Δ on IEEE 118 Base: median state error (top) and Newton iterations to 10⁻² MW (bottom), as mean ± SD over seeds 0/1/42. Purple bars show the pre-contingency state, stars show residual Δ retrained on mixed N-1…N-5 outages, and crosses show the diverged plain-Δ seed, which is excluded from the mean.*

## VII. What did not work

TABLE V. Negative and null results (more in supplement S8).

| Idea | Outcome |
|---|---|
| Error warnings from GENCO's embeddings | A 118-entry bus lookup explains R² 0.55 of the per-bus voltage error, the embeddings 0.58, a randomly initialised GENCO 0.52 |
| Per-bus bias correction | 0.3% of squared error in-sample, nothing out of sample |
| Accepting GENCO with no Newton step | Its own residual never certifies a whole contingency |
| Convergence rescue | Flat-start Newton converged on every IEEE 118 datakit scenario that solved, up to N-5 |

## VIII. Limitations

All of our safety evidence is empirical and comes from datakit scenarios, and the 400 confirmation operating points share one load trajectory, so they are not independent draws. The grid-size trend compares one model size per grid (Base, Base, Small), the factor-reuse results use seed 0, and GENCO-Δ was run on IEEE 118 only. Timings come from Python on a shared laptop and a single Colab session, and a compiled solver would lower every break-even further.

## IX. Open science and AI use

The code, checkpoints, data and demo are released under Apache-2.0 (links above). The release contains the graphkit plugins for the slack-aware task and both GENCO-Δ designs, and the float64 Newton, chord and fast-decoupled solvers with their tests. It also holds the datakit configs for every new set, the hashed confirmation protocol, the figure data as CSV, and a SHA-256 manifest with a verifier and offline CI. Iteration and miss counts are deterministic and should reproduce on the same software stack, whereas timings depend on the machine and are reported with the hardware. We used an AI coding assistant for parts of the implementation and for editing, and every reported number was checked against the result files in the repository.

## References

[1] A. Puech, M. Mazzonelli, T. R. Govindasamy *et al.*, "GENCO: a unified neural solver embedded in a development framework for steady-state grid analysis," arXiv:2608.09921, 2026.

[2] OpenGridFM, gridfm-graphkit (commit 8a8defb) and gridfm-datakit (commit d8a092d).

[3] B. Stott and O. Alsaç, "Fast decoupled load flow," *IEEE Trans. Power App. Syst.*, vol. PAS-93, no. 3, pp. 859–869, 1974.

[4] A. N. Angelopoulos and S. Bates, "A gentle introduction to conformal prediction and distribution-free uncertainty quantification," arXiv:2107.07511, 2021.
