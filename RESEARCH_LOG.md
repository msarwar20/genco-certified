# Research log: GENCO, Certified

This is a chronological record of findings, assumptions, decisions and open questions, kept for review. Each entry gives numbers and the files they come from. The plan and the summary of findings live in `gridfmchallenge.md`.

Conventions:
- Layers are numbered 1–12.
- "Residual" or "mismatch" means nodal power-balance error in MW/MVAr, max or mean over the Newton equations (P at PV+PQ buses, Q at PQ buses).
- "Reported" means computed with GENCO's own predicted slack Vm, which is what graphkit and the paper evaluate.
- "Physical" means computed with the true slack Vm setpoint enforced.
- "LU" means one sparse factorization of the power-flow Jacobian, i.e. one Newton iteration.

---

## 2026-10-02: Environment and reproduction

**Done**
- `scripts/setup_env.sh` pins graphkit `8a8defb` and datakit `d8a092d`, and uses Python 3.12 and torch 2.12.1 (CPU/MPS). It also pins `pandas<3`.
- 20 released checkpoints were re-evaluated with the unmodified `gridfm_graphkit evaluate` on the paper's own held-out test scenarios:
  - PF: IEEE 14/30/57/118 × Base/Small/Tiny.
  - OPF: IEEE 14/30/57/118 × Base/Small.
- Results are in `results/repro/summary.md`.

**Results**
- PF mean metrics are within ±6% of the paper on most runs (worst −13%).
- OPF on IEEE 57/118 is within ±8%.
- On IEEE 14/30 OPF the relative deviations are larger, but the absolute values are tiny, and the GENCO-independent DC baseline drifts the same way. That points to subset sampling.

**Assumptions**
- The first 30 partitions (6,000 scenarios, about 600 paper-test scenarios per grid) are representative of the full test set (~22k). This is supported by the DC baseline matching within ±10%.
- Max-type metrics (PBE Max, RMSE) are expected to come out lower on a subset. This is not a discrepancy.

**Issue found**
- graphkit's OPF DC baseline crashes under pandas 3: `drop(columns=..., axis=1)` at `tasks/opf_ac_dc_baseline.py:56`.
- Workaround: pin pandas<3. Upstream report: **not filed** (outward-facing; needs the user's OK).

---

## 2026-10-02: Experiment 01: layer dynamics, slack voltage, Newton certificate

Code:
- `src/genco_certified/tracing.py`: forward hooks on `physics_decoder` and `node_residuals_layer`, which capture the exact per-layer model state.
- `src/genco_certified/acpf.py`: float64 Newton-Raphson.
- `scripts/exp01_layers_newton.py`.

Data: `results/exp01/*.parquet`, `results/exp01_run.log`.

**Validation of our AC machinery**
- The Ybus is rebuilt from the same directed-edge admittances and shunts graphkit uses.
- Mismatch at the datakit ground truth is 4e-3 MW, which is float32 storage noise.
- Flat-start Newton-Raphson converges to the datakit solution with 2e-7 state error.
- With a single scenario per batch, our residual is identical to graphkit's `ComputeNodeResiduals` (corr 1.000 under perturbations).

**F1. GENCO behaves like a learned, linearly converging solver**
- Median mean-|F| per layer (IEEE 118 Base): 261 → 11 → 5.9 → 3.7 → 2.5 → 1.6 → 0.81 → 0.37 → 0.17 → 0.097 → 0.066 → 0.055 MW, about a 0.5 contraction per layer.
- Warm-start Newton iterations from layer 9 are 2.26 vs 2.15 from layer 12. Layers 10–12 are nearly redundant for a hybrid solver.
- Warm-starting from layer 1 always diverges.
- The same shape holds for Small and Tiny (`results/figures/fig1_layer_dynamics.png`).

**F2. Hidden slack-voltage setpoint**
- graphkit's `AddPFHeteroMask` (`datasets/masking.py:129-131`) masks REF-bus **Vm** (the model predicts it) and keeps REF Va fixed.
- In datakit PF data the REF Vm is a dispatch setpoint. Within every load scenario (~22 topology variants) the REF Vm standard deviation is exactly 0, identical to PV setpoints. Across load scenarios it ranges 0.988–1.060. Source: `data/pf/case118_ieee/raw/bus_data.parquet`.
- So the operator knows it, but GENCO is asked to guess it.
- Consequence: GENCO's states are power-balanced at the wrong slack voltage.

| Grid / model | Slack Vm error, median / max (p.u.) | Max mismatch, reported → physical (median, MW) |
|---|---|---|
| IEEE 14 Base | 0.0003 / 0.0014 | 0.005 → 0.17 (36×) |
| IEEE 30 Base | 0.0007 / 0.004 | 0.013 → 0.49 (39×) |
| IEEE 57 Base | 0.0010 / 0.008 | 0.033 → 1.04 (32×) |
| IEEE 118 Base | 0.0063 / 0.032 | 0.29 → 17.4 (60×) |
| GOC 2000 Small (held-out) | 0.0186 / 0.032 | 9.1 → 238 (26×) |

- The paper's own RMSE table hints at this: IEEE 118 REF Vm RMSE is 8.2e-3 vs 2.75e-4 for PQ buses.
- The paper text says unknowns are "zeroed out" but never states that REF Vm is treated as unknown (genco.txt line ~403).
- Zero-shot "mask surgery" (supplying the setpoint at inference) fails:
  - Setpoint visible to the input layer: 512 MW median max mismatch.
  - Setpoint only clamped in the correction loop: mismatch 8 MW, but 4.5× the state error.
- Hence the fine-tune (below).

**F3. One Newton step = exact refinement + error certificate**
- From GENCO's state (setpoint enforced), one Newton step reduces state error about 900× on IEEE 118 (4.9e-2 → 5.6e-5) and about 1000–2500× on IEEE 14–57.
- The step norm ‖dx‖ ranks true error with Spearman 1.000 on all 6 IEEE checkpoints.
- Newton iterations to 1e-6 MW (100% convergence for flat and GENCO starts):

| Grid | GENCO warm start | Flat start | DC start |
|---|---|---|---|
| IEEE 14–118 | 1.9–2.2 | 4.0 | 3.0–4.1 |
| GOC 2000 | 3.0 | 6.0 | 5.9 |

**F4. GENCO's own residual is a free error screen**
- Spearman 0.84 between the reported max residual and true state error (IEEE 118 Base).
- Not strong enough alone to certify (see F5).

**Assumptions / caveats (Exp 01)**
- "True" state = datakit's PowerModels AC-PF solution, stored in float32.
- The operator's problem is the standard AC-PF: slack Vm and Va are known, PV Vm and Pg are known, and Q limits are not enforced (same as datakit's `compute_ac_pf`).
- Flat start uses Vm=1 at PQ and Va = slack angle. The DC start uses series susceptances only (`PFProblem.dc_start`).
  - The first IEEE-118 runs used an older DC start that included line charging (4.13 iterations, 98.4% convergence).
  - **The IEEE 118 DC numbers should be re-run with the fixed DC start.** TODO.
- Error l2 is over unknowns [Va(PV+PQ) in rad, Vm(PQ) in p.u.], mixed units. Max-norms are reported per component.

---

## 2026-10-02: Experiment 02: N-k violation screening

Code: `scripts/exp02_screening.py`, `scripts/analyze_exp02.py`. Data: `results/exp02/`.

**Definitions (decisions)**
- Branch violation: max over both directions of |S_ft|/rate_a > τ. The primary threshold is τ = 1.01; τ = 1.001 is also reported.
- Voltage violation is screened only at PQ buses, because PV and REF voltages are known setpoints. A violation is Vm outside [vmin − 0.001, vmax + 0.001].
- Rationale: about 850 branch-scenario pairs sit within ±0.01% of their rating because the OPF dispatch makes those limits binding. A strict 100% threshold just labels solver noise (at τ = 1.0 even a 1e-7-accurate state "misses" 170).
- Parallel circuits are merged into one branch (max loading). Ratings come from graphkit's `.static` (`RemovePFMask`).
- Tiered policy:
  - Tier 0 accepts GENCO when its margin exceeds c0·(GENCO max residual).
  - Tier 1 accepts after one Newton step when its margin exceeds c1·‖dx‖² (quadratic-convergence model).
  - Tier 2 finishes Newton-Raphson.
  - c0 and c1 are calibrated as the minimal constants covering every element on a random half of the scenarios (seed 0) and evaluated on the other half.
  - **This is an empirical calibrated bound, not a mathematical certificate.** A Newton–Kantorovich bound would make it rigorous (open item).

**F5. Results (IEEE 118 Base, 319 test scenarios, τ = 1.01; 216 branch + 17 bus violations)**

| Method | Recall | Precision | Missed | LU / scenario |
|---|---|---|---|---|
| DC power flow | 0.38 | 0.29 | 144 | 0.5 |
| GENCO as published | 0.93 | 0.74 | 17 | 0 |
| GENCO + 1 Newton step | 1.000 | 1.000 | 0 | 1 |
| Full NR, flat start | 1.000 | 1.000 | 0 | 4 |
| Full NR, GENCO warm start | 1.000 | 1.000 | 0 | 2.13 |
| Tiered policy | 1.000 | 1.000 | 0 | 1.53 (53% certified at tier 1) |

- At τ = 1.001: GENCO + 1 step has recall 0.997 (1 miss); the tiered policy has recall 1.000 at 1.65 LU.
- Tier 0 never certified a whole scenario, because a single global constant c0 is too loose (c0 = 0.61 loading units per MW).
- Idea: element-local error bounds, or conformal bounds per element type.
- LU counts are hardware-independent. GENCO inference cost is **not** included in LU, so a wall-clock comparison is still TODO.

---

## 2026-10-02: Slack-aware fine-tune (in progress)

**Setup**
- `src/genco_certified/graphkit_plugin.py` registers a `PowerFlowSlackAware` task: same as PowerFlow, but REF Vm is unmasked.
- `scripts/finetune.py` keeps the released `normalizer_stats.pt`. graphkit's `finetune` CLI would refit the normalizer on the fine-tune data and silently rescale the inputs.
- Data: IEEE 118, paper train ids within the first 6,000 scenarios (4,724 train / 638 val).
- Training: Base, lr 1e-4, batch 32, 12 epochs, MPS (~3.3 min/epoch).
- **Control:** the same fine-tune with the original PowerFlow task. This separates "more training on a subset" from "slack-aware".

**Interim (epoch 3 checkpoint)**
- Physical max mismatch: median 0.64 MW, vs 17.4 for the original (27× lower).
- But state error (l2) is 7.7e-2 vs 4.9e-2 for the original. Under investigation.
- Hypothesis: early fine-tuning favours the physics loss. Also, a small residual does not imply a small state error when the Jacobian is ill-conditioned (σ_min ≈ 8–22 on IEEE 118).

---

**Final slack-aware model** (best val loss 3.92e-4 at epoch 9 of 12; 2,227 s on MPS)

Evaluated on the same 638 paper-test scenarios. Outputs:
- `results/exp01/slackaware_case118_base_*`
- `results/exp02/slackaware_case118_base_*`

| Metric (median) | Original Base | Slack-aware |
|---|---|---|
| Slack Vm error | 0.0063 | 0 (input) |
| Physical max mismatch (MW) | 17.4 | **0.66** (26× better) |
| Reported max mismatch (MW) | 0.29 | 0.66 (now honest = physical) |
| State error l2 | **0.049** | 0.076 (55% worse) |
| Va error max (rad) | **0.0071** | ~0.012 (epoch-3 value; final similar) |
| Vm error max (p.u.) | 0.0011 | **0.0003** |
| NR iterations from GENCO | 2.15 | 2.19 |
| Screening, GENCO alone (τ = 1.01): recall / precision | 0.93 / 0.74 | 0.91 / 0.50 |
| Tiered policy, LU / scenario | **1.53** | 1.86 |

- **Reading:** slack-awareness fixes physical consistency and voltage magnitudes, but angle accuracy degrades. Net screening and warm-start value is not improved.
- Open question: is that caused by the slack-aware formulation or by the short fine-tune (4.7k scenarios × 12 epochs vs the original 250k × 200)? The **matched control** (same fine-tune, original task) answers this. It is running.
- **Assumption:** the paper-train ids within the first 6,000 scenarios are representative training data. The fine-tune sees only about 2% of the original training set.

## 2026-10-02: Slack-aware vs matched control: conclusion

The control is the same fine-tune recipe (4.7k scenarios, 12 epochs, lr 1e-4) with the original PowerFlow task. Its best val loss was 2.65e-4 at epoch 9.
- Outputs: `results/exp0{1,2}/control_case118_base_*`.
- All three models are evaluated on the same 638 paper-test scenarios (medians; screening uses τ = 1.01 on the 319-scenario test half).

| Metric | Original | Control | Slack-aware |
|---|---|---|---|
| Physical max mismatch (MW) | 17.4 | 16.0 | **0.66** |
| Reported max mismatch (MW) | **0.29** | 0.37 | 0.66 |
| State error l2 | **0.049** | 0.061 | 0.076 |
| NR iterations from GENCO | **2.15** | 2.20 | 2.19 |
| GENCO-alone recall / precision | 0.93 / **0.74** | **0.94** / 0.65 | 0.91 / 0.50 |
| Tiered policy, LU / scenario | **1.53** | 1.63 | 1.86 |

**Conclusion**
- The short fine-tune itself costs about 23% in state error (original → control).
- Slack-awareness costs a further ~25% (control → slack-aware) and buys 24× lower physical mismatch.
- **At this compute budget the slack-aware model is more honest but not more useful.** Downstream, Newton warm start and the certified screen are equal or better with the original.
- The practical fix is the one-step Newton refinement (1 LU), which removes the slack inconsistency anyway.
- Recommendation for the GENCO authors:
  - Report physical metrics, i.e. evaluate with the known slack setpoint.
  - Test slack-aware training at full scale (250k × 200 epochs), which we cannot do on a laptop.
- Caveats:
  - One seed per arm.
  - The fine-tune data is only 2% of the original training set.
  - The conclusion is about this budget, not about slack-aware training in general.

## 2026-10-02: Refinement of F2: the slack mismatch is local

**Uniform re-anchoring test** (256 IEEE-118 test scenarios)
- Shifting or scaling all PQ-bus Vm by the slack-Vm error makes things much worse: median physical max mismatch 15.9 → 174 MW, state error 0.039 → 0.067.
- So GENCO's PQ voltages are already accurate (median max |ΔVm| at PQ buses 0.00095 p.u.). Only its slack Vm is off.

**Locality test**
- With the true slack setpoint, the bus with the worst mismatch is adjacent to the slack in 98% (IEEE 118, 64 scenarios) and 100% (GOC 2000, 64 scenarios) of cases.
- On IEEE 118 the 6 slack neighbours hold 53% of the total |ΔS| (median). On GOC 2000 the single neighbour holds 15%.

**Corrected interpretation of F2**
- The released PF formulation lets GENCO pick the slack Vm. It returns a state that is accurate almost everywhere, but **inconsistent with the true setpoint around the slack bus**.
- The published power-balance metric cannot see this, because it is computed with GENCO's own slack Vm.
- The 26–60× figure is about **residual metrics**, not about the state error of the whole grid.
- Claims in the write-up must be phrased accordingly: "reported power balance is optimistic", not "GENCO's state is badly wrong".

## 2026-10-02: Stress datasets generated (datakit)

- `configs/datakit/case118_n{1,2,3,5}.yaml` produced `data/stress/n{k}/case118_ieee/raw`.
- 30 load scenarios × 22 topology variants each: 660, 659, 660 and 660 scenarios. N-2 lost one infeasible case.
- About 2–3 minutes per set on the M4 with 6 processes.
- Splits: `splits_all` (test = all scenarios; train/val are dummy, unused by evaluation).
- Assumption: our N-1 set is distributionally identical to the paper's PF data (same config except seed). It serves as the in-distribution control for N-2..N-5.

## 2026-10-02: F6. Stress test: out-of-distribution contingency orders (IEEE 118 Base)

Data: `results/stress_base.log`, `results/stress_policy_base.log`, `results/exp0{1,2}/stress_n*_base_*`.
- Released PF checkpoints are trained on N-1 only.
- Stress sets have exactly 1, 2, 3 or 5 random branch/generator outages.

**GENCO accuracy and warm-start value under shift**

| Outages | Median reported max mismatch (MW) | Worst case (MW) | Median physical max mismatch (MW) | NR iterations from GENCO | NR iterations, flat start | Error after 1 Newton step (l2) | ‖dx‖ vs error, Spearman |
|---|---|---|---|---|---|---|---|
| N-1 | 0.35 | 2.6 | 16.6 | 2.11 | 4.0 | 6.3e-5 | 1.000 |
| N-2 | 0.38 | **972** | 12.2 | 2.14 | 4.0 | 5.4e-5 | 1.000 |
| N-3 | 0.44 | 18 | 18.1 | 2.19 | 4.0 | 8.4e-5 | 1.000 |
| N-5 | 0.82 | 196 | 22.1 | 2.40 | 4.0 | 1.9e-4 | 1.000 |

- GENCO degrades gracefully in the median but develops heavy tails (972 MW worst case at N-2).

**Certified tiered policy: calibrated on N-1 only, evaluated on all scenarios of each set (τ = 1.01, PQ |V| tolerance 0.001)**

| Outages | Violations | DC recall | GENCO recall | GENCO + 1 step: recall (missed) | Policy recall / precision | Policy LU | Flat-start LU | Escalated to tier 2 |
|---|---|---|---|---|---|---|---|---|
| N-1 | 435 | 0.49 | 0.95 | 0.998 (1) | **1.000 / 1.000** | **1.43** | 4.00 | 39% |
| N-2 | 714 | 0.49 | 0.93 | 1.000 (0) | **1.000 / 1.000** | **1.50** | 4.01 | 44% |
| N-3 | 950 | 0.54 | 0.95 | 0.999 (1) | **1.000 / 1.000** | **1.63** | 4.01 | 53% |
| N-5 | 1,763 | 0.52 | 0.90 | 0.997 (6) | **1.000 / 1.000** | **1.97** | 4.03 | 70% |

- **Headline:** with bounds calibrated on N-1 only, the policy kept zero missed violations and zero false alarms on unseen N-2..N-5. It automatically escalated more cases as the contingency order grew, at 37–51% of flat-start NR cost.
- **Convergence rescue: not observed.** Flat-start NR converged on 100% of cases up to N-5. The claim is dropped. Possible cause: datakit only keeps AC-PF-solvable scenarios, and IEEE 118 at these loads is benign for NR.
- **Caveats:**
  - The bounds are empirical (calibrated), not proven. The zero-miss result holds on these ~2,640 scenarios and is not a guarantee.
  - The c1 bound uses ‖dx‖², whose scale is set on N-1.

## 2026-10-02: Scale check: GOC 2000 (Small)

**Held-out set**
- The 2000-bus model was trained on a random 20k subset of the 218k scenarios, so only 11 paper-test scenarios fall in the first 1,000.
- We built `data/pf/case2000_goc/splits_heldout`: 198 scenarios from partitions 0–4 that are outside the training subset **and** share no load scenario with any train/val scenario. The last load scenario on disk is dropped because it may continue into partition 5.
- Assumption: these are as "unseen" as the paper's test set. They are a stricter set, since no load-scenario leakage is possible.

**Exp 01 results**
- Slack Vm error median 0.0186 p.u. (1.9%).
- Reported max residual median 9.1 MW vs physical 238 MW.
- Worst-bus angle error median 0.16 rad, so the state is self-consistent but wrong.
- Newton iterations: flat 6.0, DC 5.9, GENCO 3.0 (median), all converged.
- One Newton step cuts error 62× (4.22 → 0.068 l2).
- ‖dx‖ vs error: Spearman 1.000 (l2), 0.999 (Va), 0.67 (Vm-inf).

**Exp 02 (screening, 99 test scenarios after the calibration split; τ = 1.01; 6 branch + 31 PQ-bus violations).** Source: `results/exp02_case2000.log`.

| Method | Recall | Precision | LU / scenario |
|---|---|---|---|
| DC | 0.11 | 0.10 | 0.5 |
| GENCO as published | 0.11 | 0.03 | 0 |
| GENCO + 1 Newton step | 0.95 | 0.35 | 1 |
| Full NR, flat start | 1.0 | 1.0 | 6.0 |
| Full NR, GENCO warm start | 1.0 | 1.0 | **3.2** |

- The tiered policy sends 100% of scenarios to tier 2.
- **Honest reading: at 2,000 buses GENCO-Small as released is not a usable violation screen.**
  - Violations here are mostly voltage, and PQ-bus Vm errors (median max 0.0084 p.u.) exceed the 0.001 p.u. tolerance band.
  - One Newton step from that far (state error l2 4.2) is not in the quadratic regime.
  - (Corrected later the same day: an earlier note blamed the slack error for shifting all voltages. That was untested and is contradicted by the slack-locality analysis below.)
- Its value at scale is the **warm start: 47% fewer factorizations, AC-exact.**
- The slack-aware fix may matter most at large scale. This needs a 2000-bus fine-tune with more data and compute.
- Caveat: only 37 violations in 99 test scenarios, so recall numbers are coarse.

---

## 2026-10-02: Correction: the paper-test PF data is N-1, not N-k ≤ 10

- `data/pf/case118_ieee/raw/args.log` (the paper's datakit config) has `topology_perturbation: {type: random, k: 1, n_topology_variants: 22, elements: [branch, gen]}`.
- So every PF scenario is **N-1** (one random branch or generator out), plus ±20% admittance perturbation, cost permutation, and an ERCOT-shaped load scaled up to the OPF-feasibility limit.
- The paper's "k=5 / k=10" statement refers to the state-estimation datasets.
- Earlier wording in this log ("N-k") for exp02 should read "N-1".
- **Stress test planned:** datakit sets with exactly 1, 2, 3 and 5 outages (`configs/datakit/case118_n{1,2,3,5}.yaml`).
  - Same settings as the paper except `outage_count_probabilities`, with seeds 2027–2031.
  - 30 load scenarios × 22 variants each.
  - N-2+ is out of distribution for the released PF checkpoints. This tests whether the certificate catches GENCO's failures and whether flat-start NR starts failing (convergence rescue).

## 2026-10-02: Decision: rigorous Newton–Kantorovich certificate deprioritized

- Idea: in rectangular coordinates the PF equations are quadratic, so the Jacobian is affine.
  - Kantorovich then holds with explicit constants: β = ‖J(x0)⁻¹‖, L from |Ybus| row sums, η = ‖Newton step‖.
  - If h = βLη ≤ ½, a unique AC solution lies within r ≤ 2η of x0, and within ≈ (h/2)·η of the Newton point.
- Back-of-envelope with IEEE 118 magnitudes: β ≈ 1/σ_min ≈ 0.1, L ≈ 4·max_i Σ_j|Y_ij| ≈ 10⁴ (physical units), and η at GENCO's output ≈ 10⁻².
  - So h ≫ ½ and the certificate fails at GENCO's raw output.
  - It holds after one Newton step, but computing η there costs a second factorization.
  - Total ≈ 2 LU, the same as just finishing NR (2.13 LU).
- **Conclusion:** rigour without compute savings.
  - Kept as optional future work / an appendix item.
  - The calibrated bound plus "finish NR" is the practical, honest design.
- These are estimates (not computed exactly); revisit if time allows.

## 2026-10-02: Wall-clock (preliminary; MacBook Air M4, idle, torch 4 threads)

Source: `results/runtime/*.json` and `results/bench_*.log` (`scripts/bench_runtime.py`).

**IEEE 118, GENCO Base (128 paper-test scenarios)**

| What | Time |
|---|---|
| GENCO, CPU | 26.3 ms/scenario (bs1), 12.3 (bs16), 14.5 (bs128) |
| GENCO, MPS | 25.2 ms (bs1), **8.7** (bs16), 10.4 (bs128) |
| One sparse LU solve (scipy SuperLU, 181 unknowns) | 0.15 ms |
| Our Python NR from flat start | 3.1 ms total (Jacobian assembly in Python dominates) |
| PowerModels `compute_ac_pf`, per scenario as recorded by datakit (`runtime_data.parquet`, column `ac`, seconds) | 4.2 ms AC / 1.4 ms DC |

**Reading**
- At 118 buses GENCO inference (≈9 ms) is **slower** than simply solving AC-PF (3–4 ms) on this laptop.
- This matches the paper (no speedup ≤ 118 buses on CPU baselines).
- LU-count savings at this size are real but worth only fractions of a millisecond.
- **So the 118-bus contribution is the certified screening logic and the analysis, not speed.** Wall-clock value must be shown at 2,000+ buses.

**Final timings** (re-run on an idle machine; `results/runtime/*.json`)

| ms per scenario | IEEE 118 Base | GOC 2000 Small |
|---|---|---|
| GENCO CPU, bs16 / bs128 or bs64 | 12.1 / 14.5 | 102.8 / 108.5 |
| GENCO MPS, bs1 / bs16 / bs128 or bs64 | 26.3 / **8.7** / 10.3 | **67.2** / 93.7 / 100.3 |
| GENCO early exit (9 of 12 layers), largest batch: CPU / MPS | 10.8 / 7.7 (−26%) | 101.6 / 71.0 (−29% vs MPS bs64) |
| One sparse LU solve (scipy SuperLU) | 0.064 (181 unknowns) | 3.24 (3,821 unknowns) |
| Our Python NR, flat start, full solve | 2.6 (4 iterations) | 28.5 (6 iterations) |
| PowerModels AC-PF (datakit record, other hardware) | 4.2 | 696 |

**Reading (important; must be stated honestly in the write-up)**
- On a laptop, an efficient sparse Newton solver beats GENCO inference at both sizes.
- At 118 buses: GENCO ≈ 8 ms vs a full solve ≈ 2.6 ms.
- At 2000 buses: GENCO ≈ 67–100 ms vs a full solve ≈ 28 ms. MPS batching does not help at 2000 buses (per-scenario time rises with batch size).
- The paper's runtime wins (e.g. 28× on GOC 2000) are H100 GPU inference vs PowerModels. PowerModels is ~24× slower per solve than our lean scipy NR.
- So our LU reductions are real but translate into wall-clock savings only when GENCO inference is cheap relative to a factorization: a large GPU batch, or very large grids where LU fill-in grows faster than GENCO's near-linear cost (paper exponent 0.94–0.97).
- **Claims stay in LU counts, plus an explicit break-even statement:** the hybrid pays off when t_GENCO < (LU_baseline − LU_hybrid) × t_iteration. For IEEE 118 vs the base-case start that is ≈ 1.4 iterations ≈ 0.9 ms.
- Possible follow-ups:
  - GOC 10000 test (Small/Tiny checkpoints exist).
  - Colab T4 timing of batched GENCO inference.

**Assumptions**
- Datakit's `ac` runtime column is seconds per scenario on one worker. It is measured around `compute_ac_pf` plus flow computation in Julia (`process_network.py:318-336`), on unknown hardware.

**Assumptions**
- Datakit's `ac` runtime column is seconds per scenario on a single worker. Inferred from magnitudes; not verified in the datakit code.
- Our scipy NR is a fair *counting* baseline, not an optimized production solver (no KLU, Python-level Jacobian assembly).

## 2026-10-02: Fairness: the industry-standard warm start (pre-contingency base case)

**Discovery**
- Datakit's AC-PF warm-starts from the pre-contingency OPF solution (`_gfm_set_acp_starts!` copies `vm`/`va` into `vm_start`/`va_start`; `process_network.py:198-325`).
- Starting contingency cases from the base-case solution is standard practice.
- So GENCO's warm start must be compared against it, not only against flat/DC starts.

**Implementation**
- `acpf.problem_from_raw` builds PF problems straight from datakit parquet. Raw data is float64, giving |F(truth)| ≈ 1e-9 MW, and flat-start NR reproduces the stored solution to 3e-15.
- `acpf.base_case_voltages` reconstructs each load scenario's intact system:
  - Template: a branch-outage variant, which has all generators in service and intact bus types.
  - The outaged branch is restored with its in-service admittances from another variant. This is valid because r, x, b are identical across the 22 variants of a load scenario (std 0), so admittance perturbation is per load scenario.
  - The base case is then solved from a flat start.
- **Assumption:** this reconstructed base case equals the OPF base state datakit used. Same dispatch, setpoints, loads and admittances, so the AC-PF solution should coincide with the OPF voltages up to solver tolerance.

**First number** (load scenario 0, 22 variants): NR iterations from the base case 2.86 vs GENCO 2.15 vs flat 4. GENCO still saves about 25% of factorizations against the industry-standard start.

**Experiment 02 extended**
- Added `base1` (base case + 1 Newton step, the classical one-iteration screen) and NR from the base case.
- Added a "classical tiered policy" with the same calibrated escalation rule. Running on the paper test set and N-1..N-5.

## 2026-10-02: F7. GENCO vs the industry-standard base-case warm start

Source: `results/policy_with_basecase.log`, `results/exp02_base_baseline.log`.
- τ = 1.01. The paper set is calibrated on its own half; stress sets are calibrated on N-1.
- All "full NR" and tiered-policy rows have recall = precision = 1.000.

| Set | NR flat | NR base-case start | NR GENCO start | Classical tiered (base case) | **GENCO certified tiered** | Base + 1 step: recall (missed) | GENCO + 1 step: recall (missed) |
|---|---|---|---|---|---|---|---|
| Paper N-1 (319) | 4.00 | 2.90 | 2.13 | 2.80 | **1.53** | 0.923 (18) | 1.000 (0) |
| Stress N-1 (660) | 4.00 | 2.88 | 2.11 | 2.86 | **1.43** | 0.952 (21) | 0.998 (1) |
| Stress N-2 (659) | 4.01 | 3.08 | 2.14 | 3.08 | **1.50** | 0.948 (37) | 1.000 (0) |
| Stress N-3 (660) | 4.01 | 3.17 | 2.19 | 3.17 | **1.63** | 0.937 (60) | 0.999 (1) |
| Stress N-5 (660) | 4.03 | 3.32 | 2.40 | 3.32 | **1.97** | 0.927 (129) | 0.997 (6) |

Units are LU factorizations per scenario, except the last two columns.

- **Headline (fair):** against the industry-standard base-case-warm-started Newton, GENCO-certified screening needs **40–51% fewer factorizations** with identical (perfect) violation detection.
- The classical one-iteration screen (base case + 1 Newton step) misses 5–8% of violations. GENCO + 1 step misses ≤ 0.4%.
- The classical tiered policy gains nothing from the calibrated escalation. Its one-step error is too large to certify, so it falls back to full NR.
- **Caveats:**
  - LU counts only; GENCO inference is not counted.
  - At 118 buses GENCO inference (≈9 ms/scenario on the M4 GPU) costs more than a full NR solve (3–4 ms), so the wall-clock win needs bigger grids or a GPU-batched pipeline.
  - The base case is reconstructed (see assumption above).

## 2026-10-02: Operator Space (HF Spaces, Gradio) built and tested locally

**Contents** (`space/`)
- `engine.py`: runs GENCO **live** on a bundled IEEE-118 subset through the exact experiment code. It traces layers, applies the Newton step, decides the tier, and runs the baselines.
- `app.py`, with three tabs:
  1. "Inspect a contingency": an interactive grid map (true loading, overloads, outaged lines, slack bus, GENCO's per-bus mismatch); a convergence chart ("learned layers then Newton" vs base-case and flat Newton); a decision card; and a violation table.
  2. "Screen a whole set".
  3. "Findings": the four figures plus a summary.
- `bundle/`: 30 scenarios per order (N-1 paper test, stress N-2/N-3/N-5), 4.9 MB.
  - The 8 hardest scenarios by GENCO residual are included deliberately, so the bundle statistics are harder than the full-set results. The UI says so.
  - Also contains a Kamada-Kawai layout and precomputed base-case voltages.
  - Built by `scripts/build_space_bundle.py`.
- `requirements.txt` pins torch 2.12.1+cpu, torch-scatter (PyG cpu wheel), graphkit @ 8a8defb, pandas<3 and gradio 6.29.
- `genco_certified/` is a vendored copy of `src/genco_certified`. Re-copy before publishing.

**Verified locally**
- Engine: about 10 s per set on first load, including graphkit preprocessing.
- The Gradio server starts and serves the page (26 components). Callbacks were tested.
- Plotly figures were checked as PNGs via kaleido. The log-axis annotation bug is fixed.

**Not done (needs the user):** publishing to Hugging Face, which requires the user's HF account/token and is outward-facing.
- Use `huggingface_hub.HfApi().upload_folder(..., ignore_patterns=["checkpoints/*", "results/*", "**/processed/*", "__pycache__/*"])`.
- **Risk:** torch 2.12.1+cpu wheel availability on the HF builder, and the PyG torch-scatter wheel for linux cp312 on torch 2.12.0+cpu. Verify on first build; fall back to graphkit's `genco-paper-repro` branch with torch 2.8 like the reference Space if needed.

## 2026-10-02: Correction + F8: GOC 2000 vs the base case; best-of-two warm start

**Bug found and fixed**
- On GOC 2000 the base-case reconstruction silently failed for 100% of scenarios, and the baseline fell back to a flat start.
- Cause: GOC 2000 has elements that are out of service **in the base case itself** (6 branches, 146 generators). "Pick a variant without generator outages" therefore never matched.
- Fix (`acpf.base_case_voltages`): the intact state of each element is its majority status across the 22 variants. Bus types and setpoints are handled the same way.
- Revalidated:
  - IEEE 118 is unchanged (2.86 iterations from the base case on load scenario 0).
  - GOC 2000 now solves (base case in 6 iterations from flat).
  - IEEE 118 exp02 is re-run with the fixed code (`results/policy_with_basecase_v2.log`). **Verified: every row is identical to v1.**

**GOC 2000 (Small, 198 held-out; screening on 99 test after calibration; τ = 1.01; 37 violations)**

| Method | Recall | Precision | LU / contingency |
|---|---|---|---|
| NR flat / base-case / GENCO start | 1.000 | 1.000 | 6.00 / **2.96** / 3.23 |
| Classical tiered (base case) | 1.000 | 1.000 | **2.76** |
| GENCO certified tiered | 1.000 | 1.000 | 3.23 (tier 2 for 100%) |
| Base case + 1 step | 0.68 | 1.00 | 1 |
| GENCO + 1 step | 0.95 | 0.35 | 1 |

- Median error after one step: base case 8e-7 vs GENCO 6e-5 (loading).
- Median ‖dx‖: base case 0.025 vs GENCO 4.28.
- **Honest conclusion:** at 2,000 buses with the released Small checkpoint, the base case is a better prior than GENCO's from-scratch prediction for N-1. The outage perturbs a large grid only locally, while GENCO's global angle errors (worst bus ≈ 0.16 rad) dominate.
- "GENCO beats industry practice" holds at IEEE 118 and **not** at GOC 2000 with this model.
- The certification framework itself stays sound: zero misses with either prior.

**F8. Best-of-two warm start chosen without any factorization** (`scripts/exp05_best_start.py`, `results/exp05.log`)
- Evaluate ‖F‖∞ at GENCO's output and at the base case. These are two function evaluations, no LU. Start Newton from the smaller one.

| Set | Always GENCO | Always base | **Best-of-two** | Oracle | Picks GENCO |
|---|---|---|---|---|---|
| IEEE 118 N-1 (638) | 2.15 | 2.88 | **2.25** | 2.13 | 71% |
| IEEE 118 N-5 (660) | 2.40 | 3.32 | **2.42** | 2.40 | 97% |
| GOC 2000 (198) | 3.23 | 2.95 | **2.95** | 2.91 | 10% |

- A deployable rule that is never worse than industry practice: 22–27% fewer iterations on IEEE 118, and equal on GOC 2000.
- **Recommendation for the narrative:** present GENCO as a *candidate prior* inside a certified Newton framework, selected per contingency at no cost, rather than as a replacement.
- Future work to make GENCO win at scale:
  - predict the *change* from the base case (delta learning);
  - larger or slack-aware models;
  - local refinement around the outage.

## 2026-10-02: F9. Embedding probe: GENCO's latents know where its voltages are wrong

Source: `scripts/exp04_embedding_probe.py`, `results/exp04/case118_ieee_base_probe.json`.
- Model: IEEE 118 Base, 638 paper-test scenarios.
- Per-bus ridge regression (RidgeCV) of log10 true error from four feature sets.
- Train and test are disjoint halves of the scenarios. The slack bus is excluded; |ΔVm| is probed at PQ buses only.

| Features | R² for log\|ΔVm\| (PQ) | Spearman | R² for log\|ΔVa\| | Spearman |
|---|---|---|---|---|
| GENCO's local mismatch \|ΔS_i\| only (ground-truth-free) | 0.24 | 0.39 | 0.08 | 0.44 |
| Structure (bus type, hops to slack, hops to outage) | 0.18 | 0.39 | 0.03 | 0.16 |
| Final-layer bus embedding (384-d) | **0.63** | 0.77 | 0.15 | 0.42 |
| Embedding + local mismatch | **0.71** | 0.83 | **0.36** | 0.67 |

**Reading**
- The decoder-head input linearly encodes much of *where* GENCO's voltage-magnitude error lies (R² 0.63), well beyond the physics residual or the graph structure.
- Angle errors are harder (R² ≤ 0.36). Angle errors are non-local, being integrated along paths.
- PCA of the embeddings shows no strong slack-distance axis: Spearman(PC1, hops to slack) = 0.20.
- **Idea, not done:** a learned per-element error bound from embeddings, calibrated conformally. It could enable "tier 0" (no Newton step) for some decisions. This is future work.

**Caveat:** a linear probe on one checkpoint and one grid. R² is on log errors.

## 2026-10-02: Deliverables status

**Done**
- Report draft `docs/report.md` (numbers cross-checked against the logs).
- README landing page.
- `scripts/run_all.sh` (one-command reproduction) and `scripts/make_splits.py` (rebuilds non-published splits; verified identical: 198 GOC-2000 held-out scenarios, 659–660 per stress set).
- `tests/test_acpf.py`: 6 passing, including a finite-difference Jacobian check.
- Space code, bundle and publish script.
- Draft `docs/video_script.md`. The upstream-issue drafts were deleted on 2026-10-11: the user decided not to post issues upstream.

**Needs the user (outward-facing or account-bound)**
1. Publish the HF Space (`scripts/publish_space.py --repo <user>/genco-certified`). Needs an HF login.
2. ~~Create a GitHub repo~~: created **private** at msarwar20/genco-certified (2026-10-11). Still open: make it public and choose a license before submission.
3. ~~Upstream issues~~: the user decided not to post them (2026-10-11).
4. Watch for the official challenge Space (`gridfm/genco-challengex`) to learn the submission format.
5. Optional: run `bench_runtime.py` on Colab T4 for a GPU wall-clock datapoint.
6. Record the 3-minute video.

## 2026-10-11: Challenge Space opened; repo created; docs rewritten

**Challenge Space** (https://huggingface.co/spaces/gridfm/genco-challenge)
- Entries are posted as discussions in the Community tab.
- Required: title, short self-contained write-up, visuals, links to code/models/data, contact email, and names/affiliations.
- The jury sees only the post (no pitch or Q&A).
- IBM affiliates are ineligible. The jury is not yet named.
- Not submitted yet (user's instruction).

**Entries so far**
- #1 [EXAMPLE] by A. Puech (organizer):
  - Contingency analysis on GOC 500 with GENCO Tiny.
  - F-β-tuned overload threshold.
  - Runtime taken from the paper (GENCO ~2× faster than AC-PF).
  - Conformal abstention listed as next steps.
  - Our entry is positioned as that next step, done with a Newton step against the base-case baseline.
- #2 "another entry" by another author (2026-10-09):
  - AC-PTDF audit plus response-supervised fine-tuning of GENCO Small, with three seeds.
  - Very thorough open-science package.
  - Notably, its PF adapter "marks reference voltage magnitude as known". It independently handles the slack-Vm mask, without quantifying it.
  - Acknowledged in our README/report.

**Repo**
- Local git (main) and a private GitHub repo: https://github.com/msarwar20/genco-certified
- 176 files, about 17 MB.
- Excluded: large regenerated intermediates (exp02 branch/bus tables, MLflow runs) and per-run configs.
- Absolute local paths in committed logs were replaced by `<repo>`.

**Docs**
- README, `docs/report.md` and the Space text were rewritten in plain IEEE-style prose.
- The upstream-issue drafts were deleted; the user decided not to post issues.

## 2026-10-11: External reviews (independent Sonnet reviewers): accepted findings

**A. Adversarial jury review of our entry** (read-only; it reproduced our headline numbers from the stored parquets). Accepted as valid; to be fixed in Plan v3.

1. **Calibration leakage.**
   - `analyze_exp02.py` splits by *scenario*, so the paper set's 29 load scenarios appear in both halves.
   - Re-splitting by load scenario (20 draws) gives c1 between 0.10 and 0.35.
   - In 11/20 draws, 6–13 test scenarios exceed c1. Misses still stay 0.
2. **N-1 stress row in-sample.** It is calibrated on itself (`--cal_tag stress_n1_base`), so zero misses there holds by construction.
3. **The bound is exceeded out of distribution.**
   - c1 = 0.261 is exceeded on N-2/N-3/N-5 by 7/1/1 scenarios, up to 3.2×. There were no misses only because accepted margins were 21–28× the actual error.
   - A ~13× smaller c still gives 0 misses (except 1 at N-5), at 1.03–1.39 LU.
   - So the rule is neither guaranteed nor efficient; a Pareto curve is needed.
4. **LU accounting bias (`analyze_exp02.py:141`, also `:153`).**
   - Escalated scenarios were charged the all-scenario mean NR count, not their own (higher) one.
   - Corrected: 1.49 / 1.57 / 1.72 / 2.08 LU, i.e. savings 37–49%, not 40–51%.
   - `space/engine.py` already uses the per-scenario count.
5. **Baseline strength.**
   - NR to 1e-6 MW is strict. From the base case it takes 2.03 (N-1) / 2.49 (N-5) iterations to reach 1e-2 MW.
   - Base case + 2 steps (2 LU) misses 1/435 (N-1) and 6/1763 (N-5), so GENCO-tiered is only ~25% cheaper at N-1 and at parity at N-5.
   - **Defensible lead:** at equal cost (1 LU), GENCO + 1 step misses 0–0.3% and base + 1 step misses 5–7%.
6. **Industry cost model.**
   - Contingency analysis usually reuses the base-case factorization (chord / FDLF / compensation, 0 new LU).
   - In our own timings, 1 LU = 0.06 ms of a 0.65 ms NR iteration (118 buses), and GENCO inference ≈ 13 iterations.
   - LU count is a weak cost proxy, and inference is excluded from the headline.
7. **Threshold dependence.** The tier-1 share at N-1 falls from 61% (τ = 1.01, ε = 1 mV) to 19% (τ = 1.0), 6.8% (ε = 0) and 2.7% (both), where LU 2.08 vs 2.11 for plain GENCO-warm NR. A sensitivity table is required.
8. **DC fairness.**
   - The DC model uses Im(Y_ft) (r and tap included) rather than 1/x, and the same τ with no tuning.
   - Contingency-level DC recall is 0.78–0.85, not 0.49.
   - The organizers' example tunes its threshold, so we must too.
9. **Realism.** 33–35% of N-1 and 82% of N-5 scenarios contain violations, which contradicts the report's "almost all uneventful".
10. **Probe leakage** (`exp04_embedding_probe.py:86-88`).
    - The scenario-level split leaks load-scenario siblings.
    - There are no bus-ID or random-embedding controls.
    - For generator outages hops_outage = 99, clamped to 20.
11. **Spearman(‖dx‖, error) = 1** is expected, since dx ≈ −e near the solution. It is not an empirical finding and must be reworded.
12. **Reproduction wording.** "Within 6%" holds for mean metrics only. RMSE and PBE-max deviate more (subset effect, untested).
13. **Unscripted analyses.** Mask surgery, re-anchoring and slack locality have no script.
14. **Slack-Vm framing.** It may be an intended convention. Frame it as an "undocumented convention with metric consequences".

The reviewer's scores: originality 6, technical 5.5, exploration 7.5, communication 7, open science 6. 

**B. other entries rigor teardown** (another entry v0.1.0)
- Grouped splits with a test enforcing them.
- A matched control logged per run, with three seeds.
- Validation-only checkpoint selection.
- FD checks with step sweeps.
- All labels re-certified against an independent solver.
- `verify_artifacts.py` checks the hash chain.
- SHA256SUMS release, lockfile, LICENSE/NOTICE/CITATION.
- Figure contract with source CSVs.
- Gaps: n = 3 with no CIs; 2 small fixed grids; unrealistic operating points; no downstream task or UI; nothing on the HF Hub; a dense post.
- No new entries or organizer replies as of 2026-10-11.

## Open questions / TODO

- [ ] Re-run IEEE 118 exp01/exp02 with the fixed DC start for consistent DC numbers.
- [x] Fine-tune outcome: slack-aware vs control. Done (see "Slack-aware vs matched control").
- [x] Wall-clock on Mac CPU/MPS (see "Wall-clock"). Colab T4 / GPU timing still open (needs the user's Colab).
- [x] Convergence rescue: tested up to N-5 on IEEE 118. Flat start never failed, so the claim is dropped (see F6).
- [ ] Rigorous certificate: a Newton–Kantorovich bound instead of the calibrated constant.
- [ ] Element-local tier-0 bounds, so that some scenarios can skip the Newton step entirely.
- [ ] Decide with the user how and when to raise the slack-voltage finding with the GENCO authors (GitHub issue / monthly meeting).
- [ ] Check whether the PFΔ benchmark checkpoints (`genco-pfdelta`) have the same REF-Vm masking. Same code, so probably yes.
