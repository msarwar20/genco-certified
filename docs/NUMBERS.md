# NUMBERS: single source of truth for the write-up

Compiled 2026-10-22 by agent K from `RESEARCH_LOG.md` (authoritative, later entries win), `results/notes/*.md` and the result files they cite. Values marked "(computed here)" were recomputed from a result file while writing this document and are not in the log. Anything in `README.md` / `docs/report.md` is NOT a source (see section 11).

Conventions used everywhere unless a row says otherwise
- Grid IEEE 118, model Base, released seed 0, paper-test split `splits_paper_test` (638 scenarios, 29 load scenarios, all N-1: one random branch or generator out) = "paper set".
- Stress30 sets = `data/stress/n{1,2,3,5}` (660/659/660/660 scenarios, 30 load scenarios each, seeds 2027-2031). Stress100 = `data/stress100/n{1,2,3,5}` (100 load scenarios each). Calibration set = `data/calib/n1`.
- "Newton iterations / NR iterations" = iterations of our float64 NR (`src/genco_certified/acpf.py`) until max nodal mismatch <= 1e-2 / 1e-4 / 1e-6 MW. "LU" = sparse factorisations of the PF Jacobian per contingency (hardware independent; GENCO inference not included).
- "Reported" = mismatch with GENCO's own slack Vm. "Physical" = with the true slack setpoint imposed (REF/PV Vm and REF Va reset to datakit setpoints).
- Screening: violation = branch max(|S_f|,|S_t|)/rate_a > tau = 1.01 (parallel circuits merged by max) or PQ-bus Vm outside limits by more than eps_v = 1 mV. Element = branch or PQ bus. Truth = datakit AC solution. "Tiered" = 1 Newton step then accept if margin > c*||dx||^2, else NR. "Sequential" = check after every step (up to 3), then NR; escalated scenarios charged their own NR count. c is calibrated over LOAD scenarios by rule `max` or split-conformal `conf0.10`/`conf0.05`.
- CIs: 95% cluster bootstrap over load scenarios (2,000 resamples) unless stated; "CP UB" = Clopper-Pearson 95% upper bound (two-sided unless "1-sided").
- Hardware for timings: MacBook Air M4, torch 4 threads, python/scipy. GPU timings do not exist yet.
- Pins: graphkit `8a8defb`, datakit `d8a092d`, Python 3.12, torch 2.12.1, pandas<3.

---

## 1. Reproduction of released checkpoints

Source: `results/repro/summary.md`, `summary_pf.csv`, `summary_opf.csv`; scripts `scripts/reproduce.py`, `scripts/summarize_repro.py`; log entry 2026-10-02. Unmodified `gridfm_graphkit evaluate`.

Definition: 20 checkpoints (PF: IEEE 14/30/57/118 x Base/Small/Tiny = 12; OPF: IEEE 14/30/57/118 x Base/Small = 8), seed 0, on the paper's own held-out scenarios restricted to the first 30 partitions (6,000 scenarios, ~638 paper-test scenarios per grid; IEEE 57: 631), about 3% of each paper test set (~22k).

| Quantity | Value | Basis |
|---|---|---|
| PF mean metrics (Avg active res, Avg reactive res, PBE Mean), relative deviation vs paper | median abs 3.1%, 69% of values within +-6%, max abs 12.7% (IEEE 57 Base avg active res, -12.7%) | n = 36 (computed here from `summary_pf.csv`) |
| PF RMSE-* metrics | median abs deviation 9.7%, max 72% | n = 84 (computed here) |
| PF "PBE Max" | 30% to 94% lower than paper (median -81%) | n = 12; expected on a subset (max statistics) |
| DC baseline avg active residual (GENCO-independent) | +6.9, +7.9, +10.0 (IEEE 118/14/30), -8.1% (IEEE 57) | same drift sign as GENCO mean metrics, points to subset sampling |
| OPF mean metrics (avg active/reactive res, RMSE PG, opt. gap) | IEEE 118: max 7.0%; IEEE 57: max 8.0%; IEEE 30: up to 41%; IEEE 14: up to 64% (tiny absolute values) | n = 8 per grid (computed here) |
| Upstream bug | graphkit OPF DC baseline crashes under pandas 3 (`tasks/opf_ac_dc_baseline.py:56`), worked around by `pandas<3`. Not filed (user decision: no upstream issues). | log 2026-10-02 |

Wording to use: "mean PF metrics agree to within about 6% for most runs (worst -13%); maxima and RMSE deviate more (subset effect, untested)". "Within 6%" holds for mean metrics only.

---

## 2. Layer dynamics and early exit

Sources: `results/exp01/*_layers.parquet` (per layer, per scenario), `scripts/exp01_layers_newton.py`, hooks in `src/genco_certified/tracing.py`; figure `results/figures/fig1_layer_dynamics.png`. Layers are numbered 1-12 below (parquet index + 1).

### 2.1 Median (over 638 paper-test scenarios) mean |F| in MW at each layer's decoded state (reported), IEEE 118

| Layer | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Base | 261 | 11.1 | 5.86 | 3.72 | 2.51 | 1.58 | 0.81 | 0.37 | 0.17 | 0.097 | 0.066 | 0.054 |
| Small | 31.5 | 13.3 | 6.94 | 4.63 | 3.10 | 1.76 | 0.75 | 0.29 | 0.14 | 0.091 | 0.080 | 0.078 |
| Tiny | 294 | 25.4 | 11.8 | 6.41 | 4.22 | 2.74 | 1.60 | 0.72 | 0.32 | 0.16 | 0.12 | 0.099 |

Contraction between layers 2 and 12 for Base: geometric mean 0.59 per layer (computed here); individual ratios 0.45-0.82. Log wording "about 0.5 per layer" is rounded. Not geometric on GOC 500 (Base median reported mismatch 1880, 514, 136, 158, 48, 84, 30, 17, 12, 8.0, 4.6, 4.5 MW, two increases; `results/notes/batch3_H.md` section 2.1). GOC 2000 (879 scenarios, Small) mean |F| median goes 287 -> 38 -> 21 -> ... -> 0.70 (computed here).

### 2.2 Newton iterations to 1e-6 MW when warm-starting from layer k's physical state (mean over scenarios)

| Layer | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| IEEE 118 Base | 8.72 (0% conv.) | 3.50 | 3.13 | 3.08 | 3.03 | 3.02 | 3.00 | 2.67 | **2.26** | 2.19 | 2.16 | **2.15** |
| IEEE 118 Small | 4.00 (100% conv.) | 3.60 | 3.26 | 3.11 | 3.06 | 3.01 | 2.73 | 2.25 | 2.23 | 2.22 | 2.22 | 2.22 |
| IEEE 118 Tiny | 9.07 (0% conv.) | 3.72 | 3.46 | 3.19 | 3.08 | 3.03 | 3.00 | 2.74 | 2.18 | 2.17 | 2.17 | 2.18 |
| IEEE 14 Base | 3.09 | 2.87 | 2.43 | 2.30 | 2.06 | 2.05 | 2.02 | 1.93 | 1.87 | 1.86 | 1.87 | 1.87 |
| IEEE 57 Base | 3.66 | 3.37 | 3.13 | 3.09 | 3.05 | 3.01 | 2.15 | 2.03 | 2.01 | 2.00 | 2.00 | 2.00 |
| GOC 2000 Small (879 held-out) | not conv. | 4.11 | 4.00 | 4.00 | 4.00 | 4.00 | 3.73 | 3.36 | 3.23 | 3.23 | 3.20 | 3.16 |

(computed here from `*_layers.parquet`; 100% convergence from layer 2 onward on all rows.) Early-exit cost = layer 9 minus layer 12: Base +0.12 (log "2.26 vs 2.15"), Small +0.01, Tiny +0.01, IEEE 14 +0.00, IEEE 57 +0.01, GOC 2000 +0.08. Layer-1 warm start diverges for Base and Tiny; it does NOT diverge for Small (converges in 4.00, i.e. flat-start-like). The log sentence "warm-starting from layer 1 always diverges" is true only for Base/Tiny and (GOC 2000) Small. For non-converged rows the iteration counts are caps and are not meaningful.

### 2.3 Early exit inference time (9 of 12 layers, largest batch), M4, ms per scenario

Source `results/runtime/case118_ieee_base.json`, `case2000_goc_small.json`; script `scripts/bench_runtime.py`.

| | 12 layers | 9 layers | change |
|---|---|---|---|
| IEEE 118 Base, CPU bs128 | 14.5 | 10.8 | -26% |
| IEEE 118 Base, MPS bs128 | 10.3 | 7.7 | -26% |
| GOC 2000 Small, CPU bs64 | 108.5 | 101.6 | -6% (computed here) |
| GOC 2000 Small, MPS bs64 | 100.3 | 71.0 | -29% |

The accuracy cost of early exit has only been measured as Newton iterations (table 2.2), not as screening misses or LU. GPU early-exit timing: pending (Colab notebook ready, `notebooks/colab_gpu_timing.ipynb`).

### 2.4 One Newton step and the error estimate (setpoint enforced)

Source `results/exp01/*_scenarios.parquet` (computed here: ratio of median error to median error after one step). Flat/DC/GENCO columns are mean NR iterations to 1e-6 MW.

| Model / set | state err l2 median | after 1 step | ratio | flat | DC | GENCO |
|---|---|---|---|---|---|---|
| IEEE 14 Base | 4.3e-4 | 1.8e-7 | 2,475 | 4.01 | 3.07 | 1.87 |
| IEEE 30 Base | 1.7e-3 | 7.7e-7 | 2,183 | 4.00 | 3.02 | 1.99 |
| IEEE 57 Base | 4.1e-3 | 4.1e-6 | 1,007 | 4.04 | 4.01 | 2.00 |
| IEEE 118 Base (638) | 4.9e-2 | 5.6e-5 | 880 | 4.00 | 4.13 (98.4% conv.; OLD DC start) | 2.15 |
| IEEE 118 Small / Tiny | 6.1e-2 / 7.2e-2 | 7.1e-5 / 1.1e-4 | 860 / 652 | 4.00 | 4.13 (old DC) | 2.22 / 2.18 |
| GOC 500 Base (638) | 0.527 | 2.97e-3 | 177 | 5.00 | 5.00 | 2.92 |
| GOC 2000 Small (198 old) | 4.22 | 0.068 | 62 | 6.00 | 5.89 | 3.23 |
| GOC 2000 Small (879) | 4.77 | 0.114 | 42 | 7.06 (99.9% conv.) | 6.81 | 3.15 |

Fixed (series-susceptance) DC start for IEEE 118: 3.00/3.72/3.97 iterations to 1e-2/1e-4/1e-6 on the paper set (`results/exp02` `nri_dc_*`, `policy_v3.log` 3.5; also `control_`/`slackaware_case118_base` exp01 parquets, 3.97). The exp01 Base/Small/Tiny IEEE-118 parquets still carry the old DC start (4.13); log TODO still open.
Spearman(||dx||, true error) = 1.000 on all IEEE checkpoints and GOC 500; GOC 2000 0.999 (Small) / 0.996 (Tiny) in l2. This is expected analytically (dx ~ -e near the solution), not an empirical finding; do not present it as one.

### 2.5 Accuracy under outage-order shift (IEEE 118 Base, stress30; `results/exp01/stress_n*_base_scenarios.parquet`, `results/stress_base.log`)

| Outages | median reported max mismatch (MW) | worst case (MW) | median physical (MW) | NR iters GENCO | NR iters flat | err l2 after 1 step (median) |
|---|---|---|---|---|---|---|
| N-1 | 0.35 | 2.6 | 16.6 | 2.11 | 4.00 | 6.3e-5 |
| N-2 | 0.38 | 972 | 12.2 | 2.14 | 4.01 | 5.4e-5 |
| N-3 | 0.44 | 18 | 18.1 | 2.19 | 4.01 | 8.4e-5 |
| N-5 | 0.82 | 196 | 22.1 | 2.40 | 4.03 | 1.9e-4 |

Flat-start NR converged on 100% of cases up to N-5: no "convergence rescue" claim.

---

## 3. Slack-Vm convention

Sources: `scripts/exp06_slack_audit.py`; `results/exp06/summary.csv` (+ per-tag `*_summary.json`); `results/notes/ws0_B_slack.md`; GOC 500/2000-879 rows from `results/notes/batch3_H.md` sections 2 and 3; figures `fig2_hidden_slack.png`, `fig2b_slack_locality.png`. Statistics: all test scenarios, seed 0, cluster bootstrap over load scenarios.

Mechanism: `AddPFHeteroMask` (`graphkit datasets/masking.py:129-131`) masks REF Vm (unknown) and keeps REF Va. In datakit PF data REF Vm is a setpoint: std 0 within every load scenario (as for PV buses), range 0.988-1.060 p.u. across load scenarios (`data/pf/case118_ieee/raw/bus_data.parquet`). Frame as an "undocumented convention with metric consequences" (it may be intended). The paper's RMSE table is consistent: IEEE 118 REF Vm RMSE 8.2e-3 vs 2.75e-4 for PQ buses.

### 3.1 Reported vs physical gap (ratio of medians of per-scenario max mismatch, MW; CI over load scenarios)

| Grid / model | n scen (load scen) | slack Vm err median (p.u.) | reported (MW) | physical (MW) | ratio [95% CI] |
|---|---|---|---|---|---|
| IEEE 14 Base | 638 (29) | 0.00027 | 0.0048 | 0.173 | 36x [31, 46] |
| IEEE 30 Base | 638 (29) | 0.00074 | 0.0126 | 0.491 | 39x [31, 46] |
| IEEE 57 Base | 631 (29) | 0.00095 | 0.0329 | 1.04 | 32x [25, 39] |
| IEEE 118 Base | 638 (29) | 0.0063 | 0.288 | 17.4 | 60x [45, 79] |
| IEEE 118 Small | 638 (29) | 0.0057 | 0.454 | 16.8 | 37x [26, 48] |
| IEEE 118 Tiny | 638 (29) | 0.0068 | 0.583 | 20.2 | 35x [25, 43] |
| GOC 500 Base | 638 (29) | 0.0078 | 4.53 | 86.3 | 19x [12, 29] |
| GOC 500 Small | 638 (29) | 0.0055 | 4.68 | 61.0 | 13x [9.3, 19] |
| GOC 500 Tiny | 638 (29) | 0.0052 | 6.77 | 58.4 | 8.6x [6.3, 14] |
| GOC 2000 Small, 198 held-out (old) | 198 (9) | 0.0187 | 9.11 | 238 | 26x [17, 44] |
| GOC 2000 Small, 879 held-out (current) | 879 (40) | 0.0173 | 11.0 | 217 | 19.7x [15.8, 25.7] |
| GOC 2000 Tiny, 879 held-out | 879 (40) | 0.0227 | 11.1 | 289 | 26x [18.5, 32] |

Wording: "about 30-60x on IEEE 14-118 (BASE 60x is the high outlier; Small/Tiny 35-37x), 9-19x on GOC 500, 20-26x on GOC 2000". Use the 879 GOC 2000 rows, not the 198 row. These are residual metrics, not state-error statements: GENCO's PQ voltages are accurate (median max |dVm| at PQ buses 0.00105 p.u. on IEEE 118 Base). Slack Vm error p95/max: IEEE 118 Base 0.019/0.032.

### 3.2 Locality (physical state; "constrained" = P at PV+PQ and Q at PQ buses, matches the max-mismatch metric; use this definition)

| Grid / model | slack neighbours | worst bus adjacent, constrained | worst bus adjacent, full | share of |dS| in 1-hop, constrained | uniform share |
|---|---|---|---|---|---|
| IEEE 14 Base | 2 | 98.9% | 99.2% | 92% | 15% |
| IEEE 30 Base | 2 | 97.3% | 97.6% | 86% | 7% |
| IEEE 57 Base | 4 | 98.1% | 86.7% | 84% | 7% |
| IEEE 118 Base | 6 | 99.2% | 97.0% | 76% (full: 50%) | 5% |
| IEEE 118 Small / Tiny | 6 | 99.2% / 98.4% | 96.6% / 93.9% | 71% / 67% | 5% |
| GOC 500 Base / Small / Tiny | 1 | 99.4% / 96.9% / 94.2% | 99.1% / 95.9% / 93.9% | 24.8% / 17.5% / 14.5% | enrichment 100x / 73x / 62x |
| GOC 2000 Small (879) | 1 | 100% | 95% | 8.9% (full 7.3%) | enrichment 146x |
| GOC 2000 Small (198, old) | 1 | 100% | 99% | 9.2% (full 7.7%), enrichment 155x | 0.05% |

Mean |dS| per hop (IEEE 118 Base, median over scenarios, MW): hop 1 5.6, hop 2 1.2, hop 3 0.075, hops 4-6+ 0.08-0.11; hops 1-2 hold ~85%. Old log claims "98% / 100%" (64 scenarios) and "neighbour holds 15%" on GOC 2000 are superseded (8%, enrichment instead).
IEEE 57 "full" definition is the low outlier (86.7%) because PV-bus Q "mismatch" is counted; state which definition is used.

### 3.3 Uniform re-anchoring (shift or scale all PQ Vm by the slack error): worse in 95-100% of scenarios on every grid (mismatch), 89-100% (state error)
IEEE 118 Base: physical max mismatch 17.4 -> 187 MW (shift) / 183 (scale); state err l2 0.049 -> 0.076 / 0.075. GOC 2000 Small (198): 238 -> 2,260 MW. Source `ws0_B_slack.md`; GOC 500 / 2000-879: worse in 97-100% (`batch3_H.md`). Earlier 256-scenario numbers (15.9 -> 174 MW; 0.039 -> 0.067) superseded by all-scenario numbers.

### 3.4 Zero-shot mask surgery (IEEE 118, 638 scenarios, `exp06_slack_audit.py`; "visible" = setpoint in input and unmasked; "clamp_only" = setpoint only through the output `torch.where` clamp)

| Model | variant | physical max mismatch (MW) | state err l2 | err ratio vs plain | NR iters (1e-6) |
|---|---|---|---|---|---|
| Base | plain | 17.4 | 0.049 | 1 | 2.15 |
| Base | visible | 494 | 0.83 | 16.8x [11.2, 24.9] | 3.48 |
| Base | clamp_only | 10.9 | 0.281 | 5.7x [3.8, 8.4] | 2.98 |
| Small | plain | 16.8 | 0.061 | 1 | 2.22 |
| Small | visible | 4.37 | 0.371 | 6.1x [3.5, 9.7] | 2.82 |
| Small | clamp_only | 1.34 | 0.068 | 1.11x [0.89, 1.48] | 2.13 |
| Tiny | plain | 20.2 | 0.072 | 1 | 2.18 |
| Tiny | visible | 1.25 | 0.105 | 1.46x [1.08, 2.17] | 2.27 |
| Tiny | clamp_only | 0.78 | 0.069 | 0.95x [0.87, 1.13] | 2.14 |

Wording: "mask surgery fails on Base; clamp-only works on Small and Tiny (physical mismatch 13x / 26x lower, state error unchanged, NR iterations unchanged)". Old log numbers (512 MW, 8 MW, 4.5x) were Base on 64 scenarios. "Visible" feeds a value never seen in training (distribution shift).

### 3.5 Slack-aware fine-tune vs matched control (IEEE 118 Base)

Recipe: `PowerFlowSlackAware` plugin (`src/genco_certified/graphkit_plugin.py`), `scripts/finetune.py`; released normalizer kept; 4,724 train / 638 val paper-train ids within the first 6,000 scenarios (~2% of the original 250k x 200 epochs); 12 epochs, lr 1e-4, batch 32, MPS. Control = identical recipe with original PowerFlow task.

(a) Single seed (seed 0), 638 paper-test scenarios, medians (log 2026-10-02; verified against `results/exp01/{case118_ieee_base_seed0,control_case118_base,slackaware_case118_base}_scenarios.parquet`):

| Metric | Released | Control | Slack-aware |
|---|---|---|---|
| Physical max mismatch (MW) | 17.4 | 16.0 | 0.66 |
| Reported max mismatch (MW) | 0.29 | 0.37 | 0.66 |
| State error l2 | 0.049 | 0.061 | 0.076 |
| Slack Vm error | 0.0063 | 0.0057 | 0 (input) |
| Vm error max (p.u.) | 0.00105 | 0.00094 | 0.00028 |
| NR iterations from GENCO (1e-6) | 2.15 | 2.20 | 2.19 |
| GENCO-alone recall / precision (tau 1.01, 319 test scen, old split by scenario) | 0.93 / 0.74 | 0.94 / 0.65 | 0.91 / 0.50 |
| Tiered policy LU (old accounting, superseded) | 1.53 | 1.63 | 1.86 |
| Best val loss | n/a | 2.65e-4 (epoch 9) | 3.92e-4 (epoch 9) |

(b) Three init seeds (0, 1, 42), each on its own per-seed paper split (n 616-814), mean +- SD; `results/genco_delta/residual_summary_by_arm.md/.csv`, `scripts/run_genco_delta.sh`, `eval_genco_delta.py`:

| Set | arm | median state err l2 | median physical max mismatch (MW) | NR iters 1e-2 / 1e-4 / 1e-6 |
|---|---|---|---|---|
| paper | released | 0.054 +- 0.016 | 17.2 | 1.74 / 2.01 / 2.15 |
| paper | control | 0.067 +- 0.025 | 17.6 | 1.76 / 2.01 / 2.19 |
| paper | slack_aware | 0.086 +- 0.009 | 1.02 | 1.74 / 2.00 / 2.19 |
| stress N-5 | released | 0.112 +- 0.024 | 28.9 | 1.92 / 2.11 / 2.50 |
| stress N-5 | control | 0.119 +- 0.028 | 26.7 | 1.91 / 2.11 / 2.52 |
| stress N-5 | slack_aware | 0.133 +- 0.038 | 1.84 | 1.88 / 2.11 / 2.49 |

Conclusion unchanged: short fine-tune costs ~23% state error (0.049 -> 0.061; 3-seed 0.054 -> 0.067), slack-awareness a further ~25% (3-seed +29%) and buys ~17-26x lower physical mismatch, no Newton-iteration gain. Budget-limited conclusion; not about full-scale slack-aware training.

---

## 4. Embedding probe

Sources: `scripts/exp04_embedding_probe.py`, `results/exp04/probe_v2.csv`, `probe_v2_contrasts.csv`, `probe_v2_summary.json`, `results/notes/ws0_C_probe.md`.

### 4.1 Withdrawn (F9 / report Table II / README "R^2 = 0.63")
v1 numbers (embedding R^2 0.63 for log|dVm|, 0.71 with mismatch; angles 0.15 / 0.36; local mismatch 0.24; structure 0.18) came from `results/exp04/case118_ieee_base_probe.json` (STALE; do not cite). Cause: scenario-level split leaked load-scenario siblings; no bus-ID or random-embedding controls; generator outages had hops_outage clamped (99 -> 20). Claim "the latents know where GENCO is wrong beyond bus identity" and the proposed "learned per-element bound from embeddings" are withdrawn.

### 4.2 v2 (grouped 5-fold CV by load scenario, 29 groups; correct outage location; controls). IEEE 118 Base, mean +- SD over 3 init seeds (0, 1, 42); ridge on log10 error; ~41k PQ bus-rows per model (|dVm|), ~75k non-slack bus-rows (|dVa|). Per-seed CI ~ +-0.03-0.04 (cluster bootstrap, excludes retraining).

| Features | R^2 log|dVm| (PQ) | Spearman | R^2 log|dVa| |
|---|---|---|---|
| Local mismatch only | 0.224 +- 0.030 | 0.369 | 0.046 +- 0.041 |
| Structure (type, outage kind, hops to slack/outage) | 0.196 +- 0.013 | 0.417 | n/r |
| Bus-ID one-hot (118-d) | 0.553 +- 0.029 | 0.719 | 0.021 +- 0.002 |
| Bus-ID + mismatch | 0.652 +- 0.008 | 0.797 | 0.246 +- 0.103 |
| Random-init GENCO embedding (384-d) | 0.520 +- 0.027 | 0.700 | 0.003 +- 0.008 |
| Random-init embedding + mismatch | 0.631 +- 0.010 | 0.784 | 0.189 +- 0.097 |
| Trained GENCO embedding (384-d) | 0.580 +- 0.036 | 0.743 | 0.014 +- 0.017 |
| Trained embedding + mismatch | 0.667 +- 0.004 | 0.807 | 0.206 +- 0.073 |
| Trained emb + bus-ID + structure + mismatch | 0.670 +- 0.005 | 0.809 | 0.202 +- 0.079 |
| Small (seed 0): bus-ID / embedding / emb+mismatch | 0.519 / 0.539 / 0.642 | | 0.085 / 0.106 / 0.298 |

Paired contrasts (mean +- SD over seeds, dR^2): embedding - bus-ID +0.027 +- 0.009 (CI excludes 0 in 1/3 seeds); embedding - random embedding +0.060 +- 0.010 (3/3); emb+mismatch - bus-ID+mismatch +0.015 +- 0.004 (2/3, marginal); for |dVa| emb+mismatch - bus-ID+mismatch -0.040 +- 0.032 (negative in 2/3).

### 4.3 What survives
- Per-bus |dVm| error is strongly bus-specific: a 118-number bus lookup reaches R^2 0.553; cross-fitted R^2 of |e| from per-bus mean |e| = 0.447 (Base), 0.491 Small, 0.397 Tiny (`exp08`, `batch3_H.md` 4.2).
- Ground-truth-free local mismatch + bus identity gives R^2 ~0.65; the trained embedding adds only +0.015 (|dVm|), nothing for |dVa|.
- Trained beats random-weights GENCO by +0.06 R^2 consistently, but random already reaches 0.52 (architecture prior).
- Angles: nothing in the embedding beyond mismatch (v1 claim 0.36 / Spearman 0.67 was leakage; v2 0.21 / 0.49).
- Per-bus bias correction (exp08, `scripts/exp08_bias_correction.py`, `results/exp08/`): NEGATIVE. Per-bus mean signed error removes 0.3% (Base; Vm 0.29%, Va 0.34%) of squared error in-sample (Small 0.6% Vm / 3.5% Va; Tiny 2.1% / 1.5%) and nothing out of sample (fit on `calib/n1`: SSE change -0.2% [-3.5, +3.1] on paper set, +0.8% on N-5; NR iterations and 1-step misses unchanged: 1/484 and 18/6617). 63-75% of squared angle error is a per-scenario common-mode shift (Base 75%, Small 73%, Tiny 63%). Linear ridge only; outage-aware correction untried; whether common-mode shift = slack-neighbourhood effect not tested.

---

## 5. Warm-start exchange rate (Newton iterations from each start)

All entries are mean NR iterations to 1e-2 / 1e-4 / 1e-6 MW; all runs converged unless noted. Flat = Vm 1, Va = slack angle. Base start = reconstructed pre-contingency solution (`acpf.base_case_voltages`: each element's majority status across the 22 variants of a load scenario, NR from flat; GOC 2000 has 6 branches and 146 generators out in the base case itself). GENCO start = released checkpoint output, setpoints enforced. Sources: `results/policy_v3.log` 3.5, `results/exp02/v3_nr_iterations.csv`, `v3_seeds_nr_iterations.csv`, `results/notes/ws1_F_baselines_stats.md` section 5, `results/exp02/{big118,goc500,goc2000h}_nr_iterations.csv`; scripts `scripts/exp02_screening.py`, `scripts/analyze_exp02.py --v3/--ci/--xcal`.

### 5.1 IEEE 118 Base, seed 0, paper set (638 / 29 load) and stress30

| Set | flat | DC (series-B) | base start | GENCO start |
|---|---|---|---|---|
| paper N-1 | 3.01 / 4.00 / 4.00 | 3.00 / 3.72 / 3.97 | 2.04 / 2.49 / 2.88 | 1.72 / 2.01 / 2.15 |
| stress N-1 | 3.01 / 4.00 / 4.00 | 3.00 / 3.80 / 4.00 | 2.03 / 2.50 / 2.88 | 1.69 / 2.00 / 2.11 |
| stress N-2 | 3.03 / 4.00 / 4.01 | 3.00 / 3.90 / 4.00 | 2.19 / 2.74 / 3.08 | 1.60 / 2.01 / 2.14 |
| stress N-3 | 3.03 / 4.00 / 4.01 | 3.00 / 3.88 / 4.00 | 2.29 / 2.90 / 3.17 | 1.73 / 2.01 / 2.19 |
| stress N-5 | 3.08 / 4.01 / 4.03 | 3.01 / 3.76 / 4.00 | 2.49 / 3.08 / 3.32 | 1.89 / 2.08 / 2.40 |

Paired GENCO - base (paper set): -0.32 [-0.45, -0.19] / -0.49 [-0.53, -0.45] / -0.73 [-0.83, -0.63]; GENCO fewer iterations in 35% / 48% / 71%, more in 4% / 1% / 2% of contingencies. Exhaustive N-1 (every branch + generator, 6,900 contingencies, `data/exhaustive/n1`+`n1_gen`, exp07): to 1e-4: flat 4.00, base 2.48 [2.47, 2.49], GENCO 2.00; to 1e-2: 3.01 / 2.04 / 1.53 [1.40, 1.67].

### 5.2 Per size and seed (Base seeds 0/1/42 each on its own per-seed paper split; Small, Tiny seed 0). Mean +- SD over 3 Base seeds. `ws1_F` section 5; stress sets identical for all checkpoints.

| Set | tol | flat | base start | GENCO Base (3 seeds) | GENCO Small | GENCO Tiny |
|---|---|---|---|---|---|---|
| paper | 1e-2 | 3.01 | 2.07 +- 0.03 | 1.74 +- 0.14 | 1.68 | 1.83 |
| paper | 1e-4 | 4.00 | 2.52 +- 0.02 | 2.01 +- 0.01 | 2.00 | 2.01 |
| paper | 1e-6 | 4.00 | 2.91 +- 0.03 | 2.15 +- 0.09 | 2.22 | 2.18 |
| stress N-1 | 1e-2 / 1e-4 / 1e-6 | 3.01 / 4.00 / 4.00 | 2.03 / 2.50 / 2.88 | 1.79 +- .09 / 2.00 +- .01 / 2.15 +- .08 | 1.76 / 2.00 / 2.13 | 1.77 / 2.01 / 2.20 |
| stress N-2 | same | 3.03 / 4.00 / 4.01 | 2.19 / 2.74 / 3.08 | 1.68 +- .12 / 2.02 +- .02 / 2.21 +- .08 | 1.71 / 2.01 / 2.18 | 1.81 / 2.02 / 2.29 |
| stress N-3 | same | 3.03 / 4.00 / 4.01 | 2.29 / 2.90 / 3.17 | 1.81 +- .10 / 2.03 +- .02 / 2.30 +- .14 | 1.78 / 2.02 / 2.30 | 1.89 / 2.03 / 2.39 |
| stress N-5 | same | 3.08 / 4.01 / 4.03 | 2.49 / 3.08 / 3.32 | 1.92 +- .06 / 2.11 +- .05 / 2.50 +- .14 | 1.89 / 2.07 / 2.47 | 1.96 / 2.10 / 2.58 |

Paper-set seeds use splits `splits_paper_test` (seed 0, 638/29), `splits_paper_test_seed1` (814/37), `splits_paper_test_seed42` (616/28). The default split is leaky for seeds 1/42 (seed 0 test is 72%/79% in seed 1/42 training). The base-start SD on the paper row comes from the three different test sets.

### 5.3 Stress100 (calibration-free; `data/stress100/n*`, 100 load scenarios each; `big118_nr_iterations.csv`, `batch3_H.md` 1.7), mean [95% CI]

| Set | tol | flat | base | GENCO | GENCO - base (paired) | fewer / more |
|---|---|---|---|---|---|---|
| S100 N-1 | 1e-2 | 3.01 | 2.05 [2.03, 2.07] | 1.69 [1.62, 1.76] | -0.37 [-0.43, -0.30] | 38% / 4% |
| S100 N-1 | 1e-4 | 4.00 | 2.51 | 2.00 | -0.50 [-0.53, -0.48] | 49% / 0% |
| S100 N-1 | 1e-6 | 4.00 | 2.89 [2.87, 2.91] | 2.10 [2.06, 2.14] | -0.79 [-0.83, -0.75] | 75% / 1% |
| S100 N-2 | 1e-6 | 4.01 | 3.10 | 2.17 | -0.93 [-0.97, -0.88] | 85% / 0% |
| S100 N-3 | 1e-6 | 4.02 | 3.20 | 2.21 | -0.99 [-1.02, -0.95] | 88% / 0% |
| S100 N-5 | 1e-2 | 3.08 | 2.51 | 1.85 | -0.65 [-0.69, -0.62] | 59% / 0% |
| S100 N-5 | 1e-4 | 4.01 | 3.08 | 2.07 | -1.01 [-1.02, -0.99] | 94% / 0% |
| S100 N-5 | 1e-6 | 4.03 | 3.34 | 2.38 | -0.96 [-1.00, -0.92] | 83% / 0% |

(Full rows incl. N-2/N-3 at 1e-2, 1e-4 in `batch3_H.md` 1.7.)

### 5.4 Scale trend: GENCO - base start, iterations to 1e-6 MW (paired)

| Grid | model | set | 1e-2 | 1e-4 | 1e-6 |
|---|---|---|---|---|---|
| IEEE 118 | Base | paper (638/29) | -0.32 | -0.49 | -0.73 [-0.83, -0.63] |
| GOC 500 | Base | paper (638/29) | -0.10 [-0.14, -0.07] | -0.33 [-0.46, -0.18] | -0.04 [-0.13, 0.04] (tie) |
| GOC 500 | Small / Tiny | paper | -0.11 / -0.11 | -0.27 / -0.30 | +0.01 / +0.01 |
| GOC 2000 | Small | held-out 879 / 40 | +0.57 [0.40, 0.72] | +0.63 [0.59, 0.67] | +0.22 [0.12, 0.35] |
| GOC 2000 | Tiny | held-out 879 / 40 | +0.57 | +0.63 | +0.25 [0.13, 0.38] |
| GOC 2000 | Small | old 198 / 9 | +0.48 [0.12, 0.84] | +0.60 [0.52, 0.68] | +0.28 [0.03, 0.57] |

GOC 500 absolute (Base): flat 4.00 / 4.79 / 5.00; base 2.11 / 2.60 / 2.96; GENCO 2.00 / 2.27 / 2.92. GOC 2000 (Small, 879): flat 6.20 / 6.55 / 7.06; base 2.06 / 2.37 / 2.93; GENCO 2.63 / 3.00 / 3.15 (base wins in 59-64% of scenarios at 1e-2/1e-4). The 681 scenarios outside the old 198 (31 new load scenarios) replicate: +0.59 / +0.64 / +0.21. Scale trend at 1e-6: -0.73 (118) -> -0.04 (500) -> +0.22 (2000).

### 5.5 Best-of-two start (pick smaller ||F||inf at GENCO output vs base case, 0 LU; `scripts/exp05_best_start.py`, `results/exp05/*.parquet`, `results/exp05.log`), iterations to 1e-6

| Set | always GENCO | always base | best-of-two | oracle | picks GENCO |
|---|---|---|---|---|---|
| IEEE 118 paper (638) | 2.15 | 2.88 | 2.25 | 2.13 | 71% |
| IEEE 118 stress30 N-5 (660) | 2.40 | 3.32 | 2.42 | 2.40 | 97% |
| GOC 500 Base (638) | 2.92 | 2.96 | 2.87 [2.80, 2.92] | 2.80 | 40% |
| GOC 2000 Small 198 (old) | 3.23 | 2.95 | 2.95 | 2.91 | 10% |
| GOC 2000 Small 879 | 3.15 | 2.93 | 2.92 [2.89, 2.94] (-0.01 vs base) | 2.89 | 11% |

### 5.6 Equal-cost ladder: GENCO + 1 step vs base + k steps (element misses of violating elements; tau 1.01, eps_v 1 mV; no calibration)

(a) Seed 0, stress30 + paper (`policy_v3.log` 3.4; misses/false alarms):

| method (LU) | paper | N-1 | N-2 | N-3 | N-5 | GOC 2000 (198) |
|---|---|---|---|---|---|---|
| violating elements | 484 | 435 | 714 | 950 | 1763 | 61 |
| GENCO + 0 steps | 31/178 | 20/129 | 49/92 | 52/97 | 182/106 | 46/237 |
| GENCO + 1 (1 LU) | 1/0 | 1/0 | 0/0 | 1/0 | 6/0 | 4/130 |
| GENCO + 2 (2 LU) | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| base + 1 (1 LU) | 33/6 | 21/12 | 37/20 | 60/32 | 129/46 | 16/0 |
| base + 2 (2 LU) | 0/0 | 1/0 | 0/1 | 2/0 | 6/0 | 0/0 |
| base + 3 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |

(b) Three Base seeds (mean +- SD) / Small / Tiny, element misses (`ws1_F` section 5): GENCO + 1 on paper 1.3 +- 1.5 / 0 / 0; N-1 1.3 +- 0.6 / 2 / 1; N-2 0.7 +- 1.2 / 0 / 0; N-3 1.3 +- 0.6 / 1 / 2; N-5 6.3 +- 2.5 / 7 / 4. Base + 1: paper 41 +- 8 / 33 / 33; N-1 21; N-2 37; N-3 60; N-5 129. Base + 2: 0 / 1 / 0 / 2 / 6 (identical across checkpoints). GENCO + 2: 0 everywhere. Violating elements (Base mean): 567 / 435 / 714 / 950 / 1763.

(c) Stress100 + paper, seed 0 (`batch3_H.md` 1.6): GENCO + 1: 0 / 2 / 11 / 18 (N-1/2/3/5; of 1624/3210/4514/6617) and 1/484 paper; base + 2: 2 / 3 / 8 / 19, paper 0; base + 1: 114 / 213 / 371 / 519, paper 33; GENCO + 2: 0 everywhere. Pooled over six sets (18,030 violating elements; 1-sided CP UB): GENCO + 1 32 misses (0.238%), base + 1 1,369 (7.93%), base + 2 33 (0.245%), GENCO + 2 0 (0.017%).

(d) Pooled over each checkpoint's five 30-load sets (el misses / UB; ws1_F section 6): GENCO + 1: Base s0 9/4346 (0.36%), s1 7/4542, s42 17/4398 (0.58%), Small 10, Tiny 7; Base + 1: 280/4346 (7.09%); base + 2: 9/4346 (0.36%); GENCO + 2: 0.

(e) GOC 500 (133 violating elements in 92 of 638 contingencies): GENCO + 1 0/133 misses but 83 / 30 / 34 false alarms (Base/Small/Tiny); base + 1 1/133, 24 FA; both exact at 2 steps. GOC 2000 (879, 173 violating elements, 64 contingencies): GENCO + 1 4 misses / 941 FA (Tiny 3 / 1046), base + 1 18 misses / 9 FA; both exact at 2 steps.

Defensible lead (IEEE 118): "a GENCO prediction is worth about one Newton iteration for screening": at 1 LU GENCO + 1 misses 0-0.3% (0-0.6% UB pooled) and base + 1 misses 5-7%; GENCO + 1 ~ base + 2. Holds across 3 seeds and Small/Tiny. Does not hold in iteration terms under factor reuse (section 7).

---

## 6. Screening safety

Scripts: `scripts/exp02_screening.py` (per-element tables), `scripts/analyze_exp02.py` (`--v3`, `--ci`, `--xcal`), tests `tests/test_screening.py`. Logs: `results/policy_v3.log`, `policy_v3_ci.log`, `policy_big118.log`, `policy_goc500.log`, `policy_goc2000h.log`; CSVs `results/exp02/*.csv`. All "certification" is empirical ("empirically safe on these sets"), not a proof. Newton-Kantorovich certificate deprioritized (h = beta*L*eta >> 1/2 at GENCO's output; needs a second LU).

### 6.1 Data
| Set | seed | load scen. | scenarios | violating scen. | role |
|---|---|---|---|---|---|
| paper N-1 (`data/pf`, `splits_paper_test`) | released | 29 | 638 | 35.3% (484 violating elements, 225 violating contingencies) | test (and 29-load calibration in v3) |
| stress30 N-1/2/3/5 | 2027-2031 | 30 each | 660/659/660/660 | 32.9/50.7/68.9/81.7% (435/714/950/1763 elements; 217/334/455/539 contingencies) | test |
| `data/calib/n1` | 3101 | 100 | 2,200 | 31.5% (1581 elements, 746 contingencies) | calibration |
| `data/stress100/n1,n2,n3,n5` | 3111, 3102, 3103, 3105 | 100 each | 2200/2199/2199/2194 | 33.0/55.8/67.9/84.5% (1624/3210/4514/6617 elements; 786/1303/1555/1916 contingencies) | test |
| `data/exhaustive/n1` (+`n1_gen`) | 3201 | 30 | 5340 (+1620) | 32.7% of 5310 outage scen. (19.6% of 1590 gen) | baselines (exp07) |
| GOC 500 paper (`splits_paper_test`) | released | 29 | 638 | 133 elements / 92 contingencies | test |
| GOC 2000 held-out `splits_heldout` | released | 40 | 879 | 173 elements / 64 contingencies | test |

Caveats: calibration and test sets share the global load trajectory (datakit cuts the first n hours; corr 0.95, mean abs rel difference 1.8%), so conformal exchangeability holds only conditionally and the "529 load scenarios" are ~100 distinct operating levels (the load-scenario UB is optimistic). ~4% of random N-1 rows are duplicate draws. 33-35% of N-1 and 82-85% of N-5 scenarios contain violations (reviewer point 9; report's "almost all uneventful" is wrong).

### 6.2 Calibration constants (step-1 score = max over a load scenario's elements of |error|/||dx||^2)

| Calibration | GENCO max | GENCO conf0.10 | GENCO conf0.05 | base max | base conf0.10 | base conf0.05 |
|---|---|---|---|---|---|---|
| `calib/n1`, 100 load scen. (headline) | 2.178 | 0.2358 | 0.3797 | 39.81 | 20.93 | 34.44 |
| paper set, 29 load scen. (v3) | 0.3465 | 0.1257 | 0.3465 | 3575 (label-noise inflated) / 42.1 (error vs own NR) | 88.7 / 26.2 | 3575 / 42.1 |
| 3 Base seeds, own paper set (max) | 0.3465 / 1.8678 / 0.2249 (seeds 0/1/42); Small 0.1952; Tiny 0.7554 | conf0.10: 0.1257 / 0.139 / 0.1056; Small 0.1572; Tiny 0.1527 | | base (own NR) 42.1 / 37.7 / 37.2 | 26.2 / 20.1 / 26.6 | |

The max-rule constant is unstable (heavy tail over load scenarios): 0.35 from 29 load scenarios, 2.18 from 100 (6x). Conformal constants are stable (0.11-0.24). Single-split c is fragile (random half of paper load scenarios gave 0.167 vs 0.3465). Label-noise floor: graphkit stores float32, so datakit truth seen through graphkit's problem has median mismatch 1.9e-3 MW (max 7.5e-3), inflating the base-case c (3575 vs 42.1); the headline base-case policies use c measured against our own NR (stronger baseline). GENCO c is unaffected (0.3465 vs 0.3506). Score percentiles per set: `batch3_H.md` 1.2.

### 6.3 Safety result (misses and false alarms)
Calibrated on `calib/n1` (all 100 load scenarios; the calib set itself by 5-fold grouped CV), tested on S100 N-1/2/3/5 + paper set + calib CV: **0 element misses, 0 element false alarms, 0 contingency misses, 0 contingency false alarms in all 54 cells** (3 rules x 3 policies x 6 sets). Source `results/policy_big118.log`, `big118_policies.csv`.

| Pooled (6 sets) | misses | 1-sided CP UB | two-sided UB |
|---|---|---|---|
| elements (18,030 violating) | 0 | 0.017% | 0.020% |
| contingencies (6,531 violating) | 0 | 0.046% | 0.056% |
| load scenarios with a miss (529; optimistic) | 0 | 0.56% | 0.69% |

Per-set UBs (two-sided): elements calib 0.23% / S100 N-1 0.23% / N-2 0.11% / N-3 0.08% / N-5 0.06% / paper 0.76%; contingencies 0.49 / 0.47 / 0.28 / 0.24 / 0.19 / 1.63%; load scenarios 3.6% (n = 100), 11.9% (n = 29).
Earlier pooled bound over each checkpoint's five 30-load sets, seed 0 Base (max or conf0.10 rule): 0/4,346 elements (UB 0.07%), 0/1,770 contingencies (0.17%), 0/149 load scenarios (2.0%); seed 1: 0/4542, 0/1840, 0/157; seed 42: 0/4398, 0/1774, 0/148; Small and Tiny: same counts as seed 0. The five checkpoints share the stress sets (not independent evidence). Zero misses in all 72 (set x policy x rule) cells over 20 grouped-fold draws and all 120 grouped-table rows (`ws0_A` 3.3).
GOC 500 (5-fold cross-fit, 29 load scen.): 0 misses, 0 FA for every rule/policy/size; pooled over three sizes (same scenarios) 0/399 elements (1-sided UB 0.75%), 0/276 contingencies (1.08%), 0/87 load (3.4%). GOC 2000 (879/40): 0/173 elements (2-sided UB 2.11%), 0/64 contingencies (5.60%), 0/40 load (8.8%), 0 FA, for both priors and all rules.
Not a guarantee: the c1 bound IS exceeded on some sets (6.5); zero misses arise because accepted margins are far larger than realised errors (in-sample zero-miss c is 27-87x below the calibrated c).

### 6.4 Cost: sequential policy LU per contingency (GENCO prior vs base-case prior), IEEE 118
Headline (calibrated on `calib/n1`; conf0.10; 95% CI; `big118_policies.csv`, `batch3_H.md` 1.3):

| Set | GENCO seq (conf0.10) | base seq (conf0.10) | GENCO seq (max) | base seq (max) | tier-1 share G / B (conf0.10) |
|---|---|---|---|---|---|
| S100 N-1 | 1.369 [1.299, 1.439] | 2.135 [2.101, 2.168] | 1.842 [1.795, 1.886] | 2.235 [2.200, 2.268] | 0.63 / 0.15 |
| S100 N-2 | 1.465 [1.400, 1.531] | 2.509 | 1.864 | 2.601 | 0.54 / 0.01 |
| S100 N-3 | 1.531 [1.477, 1.586] | 2.726 | 1.915 | 2.815 | 0.47 / 0.00 |
| S100 N-5 | 1.651 [1.612, 1.689] | 2.961 [2.939, 2.983] | 1.964 [1.946, 1.982] | 3.016 | 0.36 / 0.00 |
| paper N-1 | 1.367 [1.234, 1.514] | 2.089 | 1.771 | 2.219 | 0.63 / 0.17 |
| calib N-1 (CV) | 1.345 | 2.126 | 1.763 | 2.219 | 0.66 / 0.14 |

= 35-44% fewer factorisations than the base-case sequential policy at zero misses (conf0.10). Tiered (1 step then NR) costs more: GENCO max-rule 1.87-2.31. Reference (single-step NR to 1e-6): 2.88/3.08/3.17/3.32 from base, 2.11/2.14/2.19/2.40 from GENCO (N-1/2/3/5, stress30).
Older, weaker-calibration versions (do not use as headline): 29-load-scenario paper-set calibration, stress30 test (`policy_v3.log` 3.2): GENCO seq max 1.477/1.487/1.588/1.755, conf0.10 1.224/1.311/1.365/1.559; base seq (own-NR c) 2.22/2.54/2.75/3.00 (max), 2.14/2.48/2.68/2.95 (conf0.10); GENCO tiered 1.585/1.619/1.777/2.124 (max). 5-fold grouped CV within each set: GENCO seq conf0.10 1.215 (paper) / 1.211 / 1.545 / 1.492 / 1.661 (N-1/2/3/5), base seq 2.61 / 2.63 / 2.41 / 2.37 / 2.59 (label-noise c). Three Base seeds (own-paper-set c, `ws1_F` 5): GENCO seq conf0.10 1.239 +- 0.080 (paper), 1.260 +- 0.039, 1.330 +- 0.036, 1.431 +- 0.084, 1.615 +- 0.077 (N-1/2/3/5); base seq 2.176 +- 0.026, 2.134, 2.463, 2.673, 2.945; max rule GENCO 1.558 +- 0.167 ... 1.847 +- 0.093, base 2.258 ... 2.995.
Corrected LU accounting under the OLD calibration (c = 0.261 from all of stress N-1, in-sample at N-1): tiered 1.494 / 1.569 / 1.718 / 2.076 vs reported-then 1.429 / 1.498 / 1.633 / 1.974 (escalated scenarios now charged own NR count); savings vs base-case NR 37-49% (not 40-51%).
GOC 500 (cross-fit): GENCO seq 1.81-1.98 (conf0.10: Base 1.812 [1.676, 1.926], Small 1.851, Tiny 1.980) vs base 2.18 (conf0.10) / 2.245 (max); tier-1 acceptance 2-19%. GOC 2000 (879): GENCO seq 2.003 [2.000, 2.008] (conf0.10, Small) / 2.032 (max) vs base 2.156 [2.122, 2.188] / 2.261; tier-1 acceptance of GENCO 0% (it is the 2-step floor, not a sign of quality). Old 198-set: GENCO 2.005 vs base 2.071 (max); conformal c = inf (9 load scenarios), everything escalates (3.227).

### 6.5 Conformal exceedance vs nominal alpha (share of evaluation load scenarios whose step-1 score exceeds c; `big118_exceedance.csv`, `batch3_H.md` 1.5)

| Set | GENCO max (nominal ~1/(n+1)) | GENCO conf0.10 | GENCO conf0.05 | base conf0.10 |
|---|---|---|---|---|
| calib N-1 (CV, n = 100) | 1/100 = 1.0% | 10/100 = 10.0% [4.9, 17.6] | 6/100 = 6.0% | 10/100 = 10.0% |
| S100 N-1 | 0% | 1/100 = 1.0% [0.0, 5.4] | 0% | 10% |
| S100 N-2 | 0% | 7/100 = 7.0% [2.9, 13.9] | 3% | 0% |
| S100 N-3 | 0% | 2/100 = 2.0% | 1% | 0% |
| S100 N-5 | 0% | 5/100 = 5.0% [1.6, 11.3] | 0% | 0% |
| paper N-1 (n = 29) | 0/29 | 1/29 = 3.4% [0.1, 17.8] | 0/29 | 4/29 = 13.8% [3.9, 31.7] |
| GOC 500 Base/Small/Tiny (cross-fit, n = 29) | 3.4 / 3.4 / 6.9% | 6.9 / 10.3 / 6.9% | | 6.9 / 6.9 / 6.9% |
| GOC 2000 Small/Tiny (879, n = 40) | 5.0 / 2.5% | 7.5 / 7.5% | 5.0 / 2.5% | 12.5 / 12.5% |

All at or below nominal 10% (N-1..N-5: 1 / 7 / 2 / 5%, 3% on the paper set), within sampling noise. Under the OLD 29-load-scenario paper-set calibration (conf0.10, `policy_v3.log` 3.2): 6.7 / 33 / 23 / 13% for N-1/2/3/5 (load scenario level; 0.6-2.9% at scenario level) = exchangeability across outage order failed. Not exchangeable across contingency order in general: N-2..N-5 values at or below nominal are empirical evidence, not guarantees. Calibration tail is heavier than test (p90 0.235 vs 0.14-0.18), so the bound is conservative and not tight. Step-1 constant at step 2: exceeded in 0-6.7% of load scenarios (`v3_step_k_ratio_check.csv`).

### 6.6 Threshold sensitivity and Pareto (only on stress30, seed 0, paper-set conf0.10 calibration; not re-run on stress100)
GENCO tiered, stress N-1, tier-1 share / LU by tau (rows) and eps_v (cols) (`policy_v3.log` 3.7, `v3_sensitivity.csv`):

| tau | eps_v 0 | 0.5 mV | 1 mV | 2 mV |
|---|---|---|---|---|
| 1.000 | 3.6% / 2.07 | 23% / 1.88 | 26% / 1.85 | 28% / 1.83 |
| 1.005 | 8.8% / 2.02 | 62% / 1.48 | 75% / 1.35 | 84% / 1.25 |
| 1.010 | 9.5% / 2.01 | 64% / 1.47 | 78% / 1.32 | 87% / 1.20 |
| 1.020 | 9.8% / 2.01 | 65% / 1.46 | 79% / 1.30 | 89% / 1.18 |

N-5: tier-1 9% to 62%, LU 2.28 down to 1.71. Classical tiered rule accepts at most 9% (N-1), 0% (N-5). Misses at eps_v = 0 or tau = 1.00 equal the exact-NR label-noise floor (exact NR "misses" 21-83 elements at tau = 1.00; 25 elements + ~419 FA at eps_v = 0, stress N-1; OPF-binding limits within 1e-6 of threshold). With eps_v >= 0.5 mV and tau >= 1.005 floor = 0 and all policies have 0 misses. Reviewer's earlier numbers (tier-1 share 61% -> 19% / 6.8% / 2.7%) were under the old calibration.
Pareto (in-sample oracle, 53 values of c, `results/exp02/pareto_*.csv`, `v3_pareto_summary.csv`): smallest c from which every larger c has zero element misses: GENCO tiered 0.004-0.0126 (0 at N-2) at LU 1.00-1.29; base tiered 0.13-0.32 at LU 1.76-3.25. This is 27-87x below the calibrated c = 0.3465. GOC 2000: GENCO c = 1e-4 (LU 2.99 tiered / 1.88 sequential); base c = 0.016 (LU 1.24 / 1.11). In-sample, not a usable rule.

### 6.7 DC baseline (datakit's PowerModels DC; branch overloads only; `v3_dc_tuned.csv`)
Our old DC (Im(Y_ft)) vs datakit DC: element recall 0.427 vs 0.440 (paper set); the old DC was not the problem. Untuned (threshold = tau), element recall 0.38-0.49 on paper/stress (0.38 paper half; 0.49/0.49/0.54/0.52 on stress N-1/2/3/5). Tuned on the paper set (5-fold CV): F3 threshold 0.875 -> element recall 0.87 / 0.95 / 0.92 / 0.97 / 0.91 (paper, N-1, N-2, N-3, N-5), precision 0.15-0.46, flag rate 1.9-2.8% of branches; recall >= 0.99 threshold 0.75 -> recall 0.98-1.00, precision 0.09-0.30. Contingency level untuned: recall 0.81 / 0.87 / 0.74 / 0.89 / 0.87, flags 58-80% of scenarios; tuned: recall 1.00 but flags 97-100% (uninformative). GOC 2000: F3 recall 0.55 at precision 1.0; recall>=0.99 setting recall 0.91 with 201 FA; contingency recall 0.75. DC LU cost not reported in v3 (legacy 0.5 convention).

### 6.8 Verifier details
- Tier-2 answer (our NR vs datakit stored solution): max abs diff in loading 0.9e-5-1.4e-5 (118-bus sets), 3.5e-5 (GOC 2000); in Vm 1.2e-6-2.6e-6, 1.9e-6.
- Recertification (`results/validation/recertify.json`): raw-parquet problem max mismatch 3.8e-7 MW (paper), 9.9e-7 / 8.8e-7 / 9.7e-7 / 9.8e-7 (stress N-1/2/3/5), 1.5e-5 (GOC 2000); no scenario above 1e-3 MW. Graphkit float32 problem: median 1.8-2.2e-3 MW, max 5.5-7.5e-3 (GOC median 1.8e-2, max 3.1e-2).
- Old single-table screening (319-scenario half of paper set, calibrated on own half, superseded): GENCO as published recall 0.93 / precision 0.74 (17 missed); GENCO + 1 step 1.000 / 1.000 (0 missed); tiered 1.53 LU (53% at tier 1); DC recall 0.38 (144 missed). `results/policy_with_basecase_v2.log`, `exp02_base_baseline.log` are the pre-correction logs.

---

## 7. Industry factor reuse (chord, compensated chord, FDLF) and triangular-solve savings

Source: `src/genco_certified/fastpf.py`, `scripts/exp07_industrial_baselines.py`, `results/exp07/*` (`cost_table.csv`, `breakeven*.csv`, `exp07.log`), `ws1_F_baselines_stats.md` sections 3-4, `tests/test_fastpf.py` (16 tests). Base seed 0 only. Definitions: full NR refactorises each iteration; **chord** = one cached base-topology factorisation per (pv,pq) pattern, plain; **chordC** = chord with Woodbury outage compensation (rank <= 4 per branch); **FDLF** = XB fast-decoupled, B', B'' factorised once, Woodbury outage compensation. "Solves" = forward+backward substitution pairs including compensation setup; base-case start assumed free and exact. Plain chord is weak (rate = outaged branch's self-PTDF; fails within 50 iterations for 19.7% of paper-set contingencies from the base start, 8.2% from GENCO; 69.5% at N-5 from base); quote chordC and FDLF.

### 7.1 Paired saving of a GENCO start vs a base-case start in triangular solves (95% CI, cluster bootstrap)

Tolerance 1e-4 MW (paper set, 638 contingencies / 29 load scen.):

| Solver | solve saving | iteration saving | GENCO fewer / more | N-5 stress30 | exhaustive N-1 (6,900) |
|---|---|---|---|---|---|
| Full NR | 20% (18-21) | 20% | 48% / 1% | 32% | 19% |
| chordC | 21% (20-21) | 31% (30-32) | 84% / 1% | 11% | 20% |
| FDLF | **5% (2-8)** | 6% (2-9) | 45% / 23% | 10% | 5% |
| plain chord | 36% (35-38) on 512 converged (126 excluded) | | | | |

Tolerance 1e-2 MW (paper): full NR 16% (9-22); chordC 17% (15-19); FDLF 10% (6-13); N-5: 24%, 11%, 11%. By outage order (1e-4, full NR / chordC / FDLF solve saving): stress N-1 20/20/6%; N-2 27/16/9%; N-3 31/14/12%; N-5 32/11/10%. Generator-only exhaustive N-1: 13/29/4%. With FDLF the GENCO start is worse than the base start in 23% of contingencies (paper set).
Mean iterations to 1e-4 (paper, base -> GENCO): NR 2.49 -> 2.00; chordC 4.11 -> 2.83 (solves 6.23 -> 4.95); FDLF 5.21 -> 4.91 (solves 11.83 -> 11.24); chord 17.77 -> 14.65. "GENCO worth": about 0.5 full-NR, 1.3 chordC, or 0.3 FDLF iterations (to 1e-4; 1.0 NR at N-5). Iteration counts are not a transferable price under reuse.

### 7.2 Other grids (same definitions; `batch3_H.md` 2.3)
GOC 500 Base (638): GENCO saves 11-12% of solves with chordC (1e-4: 11% [10, 12]; 1e-2: 12% [11, 13]), 13% [7, 18] with full NR at 1e-4 (5% at 1e-2), and is worse with FDLF (+0.44 iterations to 1e-4, solve saving -7% [-9, -4]; more iterations in 48%, fewer in 10%). GOC 2000: exp07 not run.

### 7.3 Per-contingency time and break-even GENCO inference (python/scipy, 1 thread; paper set; provisional: load average 2.3-5 on 10 cores, two runs agree within 1.5%)

Break-even = base-start solve time - GENCO-start solve time for the same solver (what GENCO must beat, inference included; negative = no benefit possible).
Measured, 1e-4 MW (`results/exp07/breakeven.csv`; ms): 

| solver | base-start ms | GENCO-start ms | break-even same solver | vs cheapest base-start solver (FDLF 0.349 ms) |
|---|---|---|---|---|
| full NR | 1.553 | 1.247 | 0.306 | -0.898 |
| chord | 0.675 | 0.520 | 0.155 | -0.171 |
| chordC | 0.750 | 0.713 | 0.037 | -0.364 |
| FDLF | 0.349 | 0.334 | 0.015 | +0.015 |

1e-2 MW: break-even 0.202 / 0.208 / 0.029 / 0.020 ms (NR / chord / chordC / FDLF). Stress N-5 (1e-4): 0.630 / 0.126 / 0.058 / 0.041; (1e-2): 0.393 / 0.313 / 0.059 / 0.041. Largest break-even anywhere (measured) 0.63 ms (full NR, N-5). Count-based "model" basis (compiled-like costs, Jacobian assembly = 2 mismatch evals): 0.01-0.08 ms for NR/chordC/FDLF on all stress sets (largest 0.27 ms plain chord N-5 1e-2). Cheapest solver from the base start takes 0.35 ms/contingency (FDLF, 1e-4). Where no base case exists (full NR from flat, 2.55 ms) a GENCO start would be worth up to ~1.3 ms.
GOC 500 (Base, `--timing --repeats 3 --max_loads 8`, 176 contingencies, loaded machine, 1e-4): full NR 8.59 vs 8.67 ms; chordC 2.70 vs 2.52; FDLF 1.51 vs 1.51; largest same-solver break-even 0.28 ms (full NR, 1e-2); full NR at 1e-4: -0.09 ms.
Measured GENCO inference on the M4: 7.7-24 ms (IEEE 118), 166-179 ms (GOC 500 CPU), 67-108 ms (GOC 2000): 12x to 1,600x above the IEEE 118 break-evens (>= 590x GOC 500). No wall-clock benefit on this hardware under any classical solver.

---

## 8. Runtime and break-even

Source: `results/runtime/case118_ieee_base.json`, `case2000_goc_small.json`, `case500_goc_base_cpu.json`; `scripts/bench_runtime.py`, `scripts/bench_genco_cpu.py`. M4, idle (IEEE 118 and GOC 2000), torch 4 threads. Final timings (re-run idle) supersede the preliminary table in the log.

| ms per scenario | IEEE 118 Base | GOC 500 Base (loaded machine, load avg 2.5-4.6) | GOC 2000 Small |
|---|---|---|---|
| GENCO CPU bs1 | 24.3 | 178.9 | 105.2 |
| GENCO CPU bs16 | 12.1 | 165.8 | 102.8 |
| GENCO CPU bs128 (IEEE) / bs64 (GOC 2000) | 14.5 | n/m | 108.5 |
| GENCO MPS bs1 | 26.3 | n/m | **67.2** |
| GENCO MPS bs16 | **8.7** | n/m | 93.7 |
| GENCO MPS bs128 / bs64 | 10.3 | n/m | 100.3 |
| GENCO 9 layers CPU / MPS (largest batch) | 10.8 / 7.7 | n/m | 101.6 / 71.0 |
| One sparse LU solve (SuperLU) | 0.064 (181 unknowns) | n/m | 3.24 (3,821 unknowns) |
| Our python NR, flat start, full solve | 2.62 (4 iterations) | n/m | 28.5 (6 iterations) |
| PowerModels AC-PF (datakit record, other hardware) | 4.2 | n/m | 696 |

(n/m = not measured.) MPS batching does not help at 2,000 buses. The paper's runtime wins (e.g. 28x on GOC 2000) are H100 inference vs PowerModels; PowerModels is ~24x slower per solve than our scipy NR (696 vs 28.5 ms, with caveat: other hardware). GENCO MPS bs1 at IEEE 118 is 26.3 ms in the JSON; the preliminary log table says CPU 26.3 / MPS 25.2 (superseded).
Break-even statements (see 7.3 for the measured tables): hybrid pays when t_GENCO < (LU_baseline - LU_hybrid) x t_iteration. The earlier "roughly 0.9 ms (1.4 LU x 0.65 ms)" used the old tiered LU (2.90 - 1.53) and is superseded; under the paired Newton-start measure the break-even is 0.31 ms (full NR, 1e-4), 0.04 ms (chordC), 0.015 ms (FDLF). GPU: not measured. `notebooks/colab_gpu_timing.ipynb` and `scripts/pack_for_colab.sh` are ready; user to run (planned 2026-10-22); report format `genco_gpu_timing.csv`. One bf16 CPU dry-run observation (not a result): Va moved up to 0.09 rad vs fp32.
Provisional numbers: all exp07 timings (shared machine); GOC 500 timing (loaded). 1 LU = 0.064 ms of a ~0.62-0.65 ms python NR iteration at 118 buses (Jacobian assembly dominates; computed here 1.553 ms / 2.49 iterations).

---

## 9. GENCO-Delta (condition on the pre-contingency state)

Sources: `results/genco_delta/residual_summary_by_arm.md|.csv` (all arms; aggregator fixed to include `residual_vm_va_flows_lr3e-4`), `summary_by_arm.md` (matrix only), `results/genco_delta/eval/*.json`, `runs/*.json`, `results/notes/ws4_E_delta.md`, `ws4_G_residual.md`; scripts `scripts/finetune.py`, `finetune_residual.py`, `eval_genco_delta.py`, `eval_genco_delta_residual.py`, `run_genco_delta.sh`, `run_genco_delta_residual.sh`; code `src/genco_certified/{graphkit_plugin,basecase_store,genco_residual}.py`. Both studies finished 2026-10-12.

Setup: IEEE 118 Base, init = released seed s in {0, 1, 42}, each trained and tested on its own per-seed paper split (paper row n = 616-814); 12 epochs, lr 1e-4 (except lr 3e-4 arm), batch 32, MPS, released normalizer, best checkpoint by validation loss only; 4,724 train ids (~2% of original). Evaluation: NR iterations from the model output; sets paper, stress30 n1/n2/n3/n5. Mean +- SD over init seeds (n_seeds in the table; failures reduce it).
- control = original PowerFlow task, same recipe. slack_aware = `PowerFlowSlackAware`.
- delta_vm_va / delta_vm_va_flows = `PowerFlowContingencyAware`: fill masked VM_H/VA_H (and P_E/Q_E for `_flows`) with base-case values (normalised units); masks unchanged.
- residual_* = `GNS_heterogeneous_residual`: decode = base + zero-initialised delta head (released `mlp_bus` with last Linear zeroed) at every layer; equals the base case exactly at init (tested bit for bit).
- Confound: base-case Vm at PV/REF equals the post-contingency setpoint (6e-8 p.u.), so Delta arms implicitly receive the true slack Vm. The slack_aware arm alone gains nothing, so the residual gain is not the slack effect.

### 9.1 Results (mean +- SD over init seeds). Base start (mean NR iters 1e-6): paper 2.91 +- 0.03, N-1 2.88, N-2 3.08, N-3 3.17, N-5 3.32. Base + 1 step misses: 41 +- 8 (paper), 21, 37, 60, 129.

| Set | arm | seeds | iters 1e-2 | 1e-4 | 1e-6 | median err l2 | misses after 1 step |
|---|---|---|---|---|---|---|---|
| paper | released | 3 | 1.74 +- .14 | 2.01 | 2.15 +- .09 | 0.054 +- .016 | 1.33 +- 1.5 |
| paper | control | 3 | 1.76 | 2.01 | 2.19 +- .07 | 0.067 | 1.0 |
| paper | slack_aware | 3 | 1.74 | 2.00 | 2.19 +- .07 | 0.086 | 1.0 |
| paper | delta_vm_va | 2 | 2.00 | 2.20 | 2.85 +- .10 | 0.167 | 4.5 |
| paper | delta_vm_va_flows | 3 | 5.63 +- 6.3 | 5.75 | 6.18 +- 5.8 | 0.534 +- .66 | 35.7 +- 54 |
| paper | residual_vm_va | 3 | 1.39 +- .08 | 2.07 | 2.24 +- .06 | 0.0199 | 4.33 |
| paper | residual_vm_va_flows | 3 | 1.41 | 2.08 | 2.25 | 0.0224 | 4.0 |
| paper | residual_vm_va_flows_lr3e-4 | 3 | **1.33 +- .08** | 2.05 | 2.19 +- .06 | 0.0219 +- .009 | 1.67 +- .58 |
| N-1 | released | 3 | 1.79 | 2.00 | 2.15 | 0.0595 | 1.33 |
| N-1 | control | 3 | 1.83 | 2.00 | 2.20 | 0.0734 | 1.67 |
| N-1 | slack_aware | 3 | 1.81 | 2.01 | 2.26 | 0.101 | 1.33 |
| N-1 | delta_vm_va | 2 | 2.01 | 2.19 | 2.88 | 0.207 | 4.0 |
| N-1 | delta_vm_va_flows | 3 | 5.58 | 5.76 | 6.12 | 0.543 | 29.7 |
| N-1 | residual_vm_va | 3 | 1.35 | 2.05 | 2.20 | 0.0188 | 2.67 |
| N-1 | residual_vm_va_flows | 3 | 1.37 | 2.06 | 2.21 | 0.0212 | 2.33 |
| N-1 | residual_vm_va_flows_lr3e-4 | 3 | 1.29 +- .05 | 2.05 | 2.16 | 0.0215 | 2.0 |
| N-2 | released | 3 | 1.68 | 2.02 | 2.21 | 0.0573 | 0.67 |
| N-2 | control | 3 | 1.72 | 2.03 | 2.25 | 0.0692 | 0.67 |
| N-2 | slack_aware | 3 | 1.74 | 2.03 | 2.27 | 0.0868 | 0.33 |
| N-2 | delta_vm_va | 2 | 2.03 | 2.21 | 2.95 | 0.204 | 6.0 |
| N-2 | delta_vm_va_flows | 3 | 5.77 | 5.95 | 6.38 | 0.773 | 47 |
| N-2 | residual_vm_va | 3 | 1.58 | 2.12 | 2.36 | 0.040 | 2.0 |
| N-2 | residual_vm_va_flows | 3 | 1.62 | 2.12 | 2.39 | 0.0412 | 3.67 |
| N-2 | residual_vm_va_flows_lr3e-4 | 3 | 1.51 | 2.10 | 2.30 | 0.0379 | 2.33 |
| N-3 | released | 3 | 1.81 | 2.03 | 2.30 | 0.0759 | 1.33 |
| N-3 | control | 3 | 1.86 | 2.03 | 2.36 | 0.096 | 1.0 |
| N-3 | slack_aware | 3 | 1.85 | 2.03 | 2.36 | 0.119 | 1.0 |
| N-3 | delta_vm_va | 2 | 2.03 | 2.29 | 2.95 | 0.24 | 7.0 |
| N-3 | delta_vm_va_flows | 3 | 5.72 | 6.00 | 6.35 | 0.601 | 76.3 |
| N-3 | residual_vm_va | 3 | 1.77 | 2.18 | 2.52 | 0.0754 | 6.33 |
| N-3 | residual_vm_va_flows | 3 | 1.81 | 2.18 | 2.54 | 0.0752 | 7.0 |
| N-3 | residual_vm_va_flows_lr3e-4 | 3 | 1.68 | 2.14 | 2.43 | 0.0704 | 5.0 |
| N-5 | released | 3 | 1.92 | 2.11 | 2.50 | 0.112 | 6.33 |
| N-5 | control | 3 | 1.91 | 2.11 | 2.52 | 0.119 | 6.67 |
| N-5 | slack_aware | 3 | 1.88 | 2.11 | 2.49 | 0.133 | 7.0 |
| N-5 | delta_vm_va | 3 | 5.91 | 6.12 | 6.51 | 0.63 | 125 |
| N-5 | delta_vm_va_flows | 2 | 2.10 | 2.51 | 3.02 | 0.282 | 34.5 |
| N-5 | residual_vm_va | 3 | 1.97 | 2.34 | 2.79 | 0.172 | 22.0 |
| N-5 | residual_vm_va_flows | 3 | 1.98 | 2.36 | 2.81 | 0.168 | 22.7 |
| N-5 | residual_vm_va_flows_lr3e-4 | 3 | 1.89 | 2.28 | 2.69 | 0.155 | 17.7 |

(SDs for the other cells are in the CSV.) Misses after one step = violating elements missed after GENCO + 1 Newton step (of 484 paper / 435 / 714 / 950 / 1763).

### 9.2 Verdicts
- **Plain Delta: negative and unstable.** Seeds 0/1 about base-case quality (2.85-2.95 iterations to 1e-6 vs base 2.88-3.08; released 2.15-2.30). Seed 42 diverged in both plain arms: best val loss 2.9e-2 (delta_vm_va, best epoch 2) and 2.78e-2 (delta_vm_va_flows) vs 2.4e-3 / 2.2e-3 for seeds 0/1 (~10x); Newton needs ~13 iterations from those predictions and five of the ten seed-42 plain-arm evaluations crash with an index error (trajectory never forms), which is why n_seeds is 2 for delta_vm_va on paper/N-1/N-2/N-3 and for delta_vm_va_flows on N-5, and why the 3-seed means of delta_vm_va_flows (6.2 iterations, SD 5.8) are dominated by the diverged seed. Released weights with filled inputs have median state error 7.1 before training. Early seed 0 (log 2026-10-12): delta_vm_va 2.02 / 2.30 / 2.93 (paper), 2.12 / 2.46 / 3.00 (N-5).
- **Residual Delta (lr 3e-4): 2.5x lower N-1 state error, ~25% fewer iterations to 1e-2 MW, no gain at 1e-6, and regression at N-5.** Released -> residual: median err l2 0.054 -> 0.022 (paper), 0.060 -> 0.022 (N-1), 0.057 -> 0.038 (N-2), 0.076 -> 0.070 (N-3), 0.112 -> 0.155 (N-5, worse); iterations to 1e-2: 1.74 -> 1.33, 1.79 -> 1.29, 1.68 -> 1.51, 1.81 -> 1.68, 1.92 -> 1.89; to 1e-6: 2.15 -> 2.19, 2.15 -> 2.16, 2.21 -> 2.30, 2.30 -> 2.43, 2.50 -> 2.69; misses after one step 1.3 -> 1.7 (paper), 1.3 -> 2.0 (N-1), 0.7 -> 2.3, 1.3 -> 5.0, 6.3 -> 17.7. Trained on N-1 only. lr 3e-4 has lower best val loss than lr 1e-4 in all three seeds (3.64e-4 vs 4.85e-4 s0; 5.58e-4 vs 6.57e-4 s1; 6.10e-4 vs 6.95e-4 s42), so the choice is not made on test data.
- Control best val loss 2.76e-4 / 2.37e-4 / 5.58e-4 (s0/1/42); slack_aware 3.95e-4 / 3.38e-4 / 6.84e-4; residual_vm_va 5.17e-4 / 6.09e-4 / 6.74e-4.
- Smokes (plumbing only, `ws4_E`, `ws4_G`): residual at init reproduces base exactly (2.04 / 2.49 / 2.88, state err 0.0502, 33 misses/6 FA); 74 steps 2.03 / 2.49 / 2.87. Plain delta_vm_va_flows after 2 epochs: 2.06 / 2.84 / 3.04.
- Base vs post-contingency label distance (median of per-scenario max, paper set): |dVm| at PQ 4.3e-3 p.u., |dVa| 2.4e-2 rad; N-5: 1.8e-2 / 0.13.
- Not done: N-k training mixtures (agent N, `scripts/run_genco_delta_nk.sh`, data `data/train_nk/case118_ieee`, 5,056 scenarios / 230 load scenarios, ratio of outage counts 1:2:3:4:5 = 36.8/20.5/14.9/14.2/13.6%; no results yet, no `ws4_N_nk.md` yet); GOC 2000 (Colab notebook `notebooks/colab_genco_delta.ipynb` ready, not run; 40 of 182 GOC 2000 load scenarios have no variant in the released model's train/val split).

---

## 10. Negative results (all to be reported as such)

1. **Convergence rescue**: flat-start NR converged on 100% of cases up to N-5 (IEEE 118 stress30); claim dropped (Sec. 2.5).
2. **Tier 0** (accept GENCO with no Newton step using its own residual as bound): never certified a whole scenario (c0 = 0.61 per MW too loose); dropped.
3. **GENCO beats the pre-contingency start at scale**: no. GOC 500 tie (-0.04 [-0.13, 0.04] at 1e-6), GOC 2000 base wins by +0.22 [0.12, 0.35] to +0.63 iterations (Sec. 5.4). FDLF: GENCO start worse than base in 23% of contingencies (Sec. 7).
4. **Wall-clock benefit on M4**: none; GENCO inference 7.7-24 ms vs break-evens 0.015-0.63 ms (Sec. 7.3, 8). GPU unmeasured.
5. **Embedding probe** (F9): withdrawn (Sec. 4.1). Per-bus bias correction: no effect (Sec. 4.3).
6. **Slack-aware fine-tune**: honest balance but ~25% worse state error vs matched control and no downstream gain at this budget (Sec. 3.5).
7. **Zero-shot mask surgery on Base**: visible 494 MW / 16.8x state error; clamp-only 5.7x. (Works on Small/Tiny.)
8. **Uniform re-anchoring of PQ voltages**: worse in 95-100% of scenarios.
9. **Plain GENCO-Delta**: no gain, unstable (seed 42 diverged) (Sec. 9.2). Residual Delta regresses at N-5.
10. **Calibrated bound not exchangeable across outage order** with the 29-load-scenario calibration (6.7 / 33 / 23 / 13% exceedance); fixed with 100-load calibration (1 / 7 / 2 / 5%) but still not a guarantee.
11. **Plain chord Newton** from the base case: fails the 50-iteration cap for 19.7% (paper) to 69.5% (N-5) of contingencies.
12. **Classical tiered rule** (base prior): accepts at most 7-9% after one step; no gain from escalation.
13. **GOC 2000 screening with GENCO alone**: recall 0.11 as published; GENCO + 1 step recall 0.95 at 130 false alarms (198) / 941 (879); tier-1 acceptance 0%.
14. **Rigorous Newton-Kantorovich certificate**: back-of-envelope only; fails at GENCO's raw output (not computed exactly).
15. Spearman(||dx||, error) = 1 is not a finding.

---

## 11. Superseded claims in README.md and docs/report.md

| Claim in README / report | Correct statement (source) |
|---|---|
| README s.1 / report s.I: "PF mean metrics within 6% on most runs" (report abstract unspecific) | Mean metrics only: median 3.1%, max 12.7%; RMSE median 9.7%, PBE Max 30-94% lower (Sec. 1) |
| Report III.A: warm-start from layer 9 "2.26 vs 2.15", early exit cuts inference 26% | Numbers fine for Base/IEEE 118; GOC 2000 -6% CPU / -29% MPS; README "<0.1 extra iteration" is +0.12 on Base, +0.01 on Small/Tiny (Sec. 2.2-2.3) |
| README/report Table II and abstract: embedding R^2 0.63 (0.71 with mismatch), "embeddings encode where voltage errors lie" | Withdrawn; v2: embedding 0.58, bus-ID 0.553, random weights 0.52; angles ~0 (Sec. 4) |
| README/report: slack gap "26 to 60 times" | 35-60x on IEEE 14-118 (Base 60x outlier, Small/Tiny 35-37x), 9-19x GOC 500, 20-26x GOC 2000 (Sec. 3.1) |
| Report IV.C: worst bus adjacent in "98%" IEEE 118 and "all" GOC 2000; six neighbours hold 53% | 97.0% full / 99.2% constrained (638 scen.); neighbours 50% (full) / 76% (constrained); GOC 2000 neighbour share 8-9%, enrichment ~150x; worst bus adjacent 95-100% (Sec. 3.2) |
| Report IV.D: mask surgery does not help (512 MW; 8 MW, 4.5x) | Base only (494 MW; 10.9 MW, 5.7x); clamp-only works on Small and Tiny (Sec. 3.4) |
| Report IV.D / README: fine-tune comparison "one seed per arm" | 3 init seeds now exist (Sec. 3.5b); conclusion unchanged (+29% state error vs control) |
| README Table I / report Table V: two-tier LU 1.43 / 1.50 / 1.63 / 1.97 (stress N-1 in-sample calibration, escalated cases charged mean NR count) | LU accounting bug and in-sample N-1 row. Corrected under old calibration: 1.494 / 1.569 / 1.718 / 2.076. Headline now: sequential with 100-load-scenario conformal calibration 1.37 / 1.47 / 1.53 / 1.65 vs base 2.14 / 2.51 / 2.73 / 2.96 (Sec. 6.4) |
| README / abstract: "40 to 51% fewer factorizations than Newton from the pre-contingency state" | 37-49% (corrected accounting), 35-44% vs the base-case sequential policy on stress100 (conf0.10). Not 40-51% |
| README Table I: DC recall 0.49 | Datakit DC element recall 0.43-0.44 untuned; tuned F3 threshold recall 0.87-0.97 at precision 0.15-0.46 but flags 97-100% of contingencies (Sec. 6.7) |
| Report V.B / README: "bound calibrated on N-1 only ... smallest value covering every element" ; stress N-1 row | N-1 stress row was in-sample; calibration now by load-scenario groups on a separate 100-load set; bound exceeded under shift with the old calibration; wording "empirically safe on these sets" (Sec. 6.2-6.5) |
| Report V.C: 319-scenario half; GENCO + 1 step finds all 233 violations; two-tier 1.53 LU; base-case NR 2.90 | Pre-correction; use Sec. 5.6 and 6.3-6.4 (GENCO + 1: 1/484 missed on paper set; base + 1: 33) |
| Report V.A: Spearman 1.000 as a result | Expected analytically (dx ~ -e); not a finding (Sec. 2.4) |
| Report V.D: "no violation is missed"; zero false alarms | Holds for calibrated policies on 0/18,030 elements (UB 0.017%) but not for GENCO + 1 step (32/18,030) (Sec. 6.3) |
| README limitations: "about 2,600 IEEE-118 contingencies" | 0 misses on 6,531 violating contingencies / 18,030 violating elements over 11,630 scenarios (stress100 + calib + paper; Sec. 6.3), plus 1,770 contingencies / 3,277 scenarios on stress30+paper |
| README/report GOC 2000: 198 scenarios, 2.96/2.95 vs 3.23 iterations | 879 scenarios / 40 load scenarios: base 2.93 vs GENCO 3.15 (Small), 3.18 (Tiny) (Sec. 5.4) |
| Report V.E: "worst-bus angle error ~0.16 rad" GOC 2000 | Still from the 198 run; 879 run median err_va at layer 12 = 0.182 (Sec. 2 table; `case2000_goc_small_seed0_h879_layers.parquet`, computed here) |
| Report VI: break-even "about 0.9 ms per contingency" | Old LU basis. Measured paired break-evens 0.31 / 0.04 / 0.015 ms (NR / chordC / FDLF), max 0.63 ms at N-5 (Sec. 7.3) |
| Report VI Table VII: GENCO best 8.7 (118), 67 (2000) | Still correct (MPS bs16; MPS bs1); GOC 500 CPU 166-179 ms added (Sec. 8) |
| Report VIII future work: "per-element bounds from embeddings", "delta learning ... likely to matter on large grids", "slack-aware at full scale" | Embedding bounds unsupported; delta learning tested (plain negative, residual helps N-1 at loose tolerance, fails N-5); GOC 2000 delta untested (Sec. 9) |
| Report II.A Table I: stress and paper sets only; calibration on paper set | Add `calib/n1`, `stress100`, exhaustive N-1, GOC 500 (Sec. 6.1) |
| Report/README "almost all uneventful" (Intro) | 33-35% of N-1 and 82-85% of N-5 scenarios contain violations |
| README "Another entrant (another entry) also treats this voltage as known" | Still true; other entries ("another entry") now overlaps on NR verification (other entries note) |
| README "pre-contingency states reconstructed" | Unchanged; GOC 2000 majority-status fix (100% silent fallback bug fixed 2026-10-02) |

---

## 12. Gaps and conflicts (flagged)

Conflicts between sources
1. **GOC 2000 slack ratio**: log/report/README 26x [17, 44] (198 scenarios, 9 load scen.) vs 19.7x [15.8, 25.7] (Small, 879/40) and 26x [18.5, 32] (Tiny). Use 879. The log's Batch 3 entry does not repeat this number; it is only in `batch3_H.md`.
2. **GOC 2000 worst-bus adjacency**: log "99-100%"; 879 run is 95% (full) for both Small and Tiny; constrained 100% (Small) / 95% (Tiny).
3. **Layer-1 warm start "always diverges"** (log F1): false for IEEE 118 Small (converges, 4.00).
4. **Early exit**: README "<0.1 extra iteration"; Base is +0.12.
5. **Three "headline" GENCO sequential LU ranges** (conformal 0.10): 1.22-1.56 (29-load calibration, exceedance up to 33%), 1.24-1.62 (3 seeds, own paper-set c), 1.37-1.65 (100-load calibration, correct). Use the last. The log's Batch 2 F entry (1.24 +- 0.08 ... 1.62 +- 0.08) matches `ws1_F`, but note it reuses the 29-load c.
6. **Break-even**: log wall-clock section "0.9 ms (1.4 iterations)" vs paired measured 0.31 ms (full NR). Different definitions (screening-policy LU saving vs Newton-start saving); the 0.9 ms is superseded.
7. **Preliminary vs final timings** in the log: CPU bs1 26.3 vs 24.3, MPS bs1 25.2 vs 26.3, LU 0.15 vs 0.064 ms, python NR 3.1 vs 2.6 ms. Use `results/runtime/*.json`.
8. **GENCO-Delta "paper N-1" rows** are means over three per-seed paper splits (n = 616-814), not the seed-0 638 set. Released 1.74 / 2.01 / 2.15 matches the 3-seed row; seed 0 alone is 1.72 / 2.00 / 2.15. N-count in plain-Delta rows varies (2 vs 3 seeds), and the log's "four of ten evaluations crash with an index error" is five of ten in the files: the seed-42 plain arms have 5 failed evaluations (`IndexError: list index out of range`, `results/genco_delta/logs/eval_delta_vm_va_s42__{paper,n1,n2,n3}.log` and `eval_delta_vm_va_flows_s42__n5.log`); the other five seed-42 evaluations ran but with diverged predictions.
9. **IEEE 118 DC start**: exp01 Base/Small/Tiny parquets still use the old DC start (4.13 iterations, 98.4% convergence); the corrected 3.97 exists only in exp02 v3 and the control/slack-aware parquets. Log TODO still unchecked.
10. **`policy_with_basecase*.log`, `exp02_base_baseline.log`, `policy_big118`-era older pooled numbers**: pre-correction logs remain in `results/` and show superseded LU (1.43-1.97, 2.90 etc.); do not cite.
11. `results/notes/other entries_2026-10-21.md` says "Clopper-Pearson bounds on 11k contingencies": sources give 6,531 violating contingencies (11,630 scenarios) on the big sets and 1,770 (3,277 scenarios) on stress30+paper; the "11k" is scenarios, not contingencies.
12. Slack-aware figures differ between the single-seed fine-tune (0.049 / 0.061 / 0.076) and the 3-seed matrix (0.054 / 0.067 / 0.086, per-seed splits); report the 3-seed version as primary, single seed as the first run.

Headline numbers with partial or missing sources
13. **GPU timing**: no result file exists (`results/gpu_timing/` absent). The notebook has never been run on a GPU. Every statement about GPU break-even is open.
14. **Threshold sensitivity** (tau, eps_v grid) exists only for stress30 with the 29-load-scenario conformal calibration; not rerun on `calib/n1` + `stress100`.
15. **Seeds 1/42, Small/Tiny** not run on the big IEEE 118 sets; exp07 (factor reuse) only Base seed 0 on IEEE 118 and GOC 500 Base; GOC 2000 exp07 not run; GOC 500 seeds 1/42 not run.
16. **Early exit** effect on screening (misses, LU) not measured; only on NR iterations.
17. **GENCO-Delta**: N-k training and GOC 2000 delta not done; residual arm trained on N-1 only. Seed 42's divergence of plain Delta is not diagnosed beyond validation loss.
18. **Common-mode angle shift vs slack-neighbourhood effect** (63-75% of squared Va error): not tested.
19. **Reproduction subset**: claim that maxima/RMSE differences are "subset effect" is untested; full test sets not evaluated.
20. **Base-case reconstruction** equals datakit's OPF base state only by assumption (majority status, same dispatch/admittances); `acpf.base_case_voltages` checked on several load scenarios, no independent reference. In practice base-case solutions are assumed free and exact.
21. **Calibration-test independence**: all calibration and test sets share the 100-hour global load trajectory (corr 0.95); the published paper set's trajectory overlap with `calib/n1` was not checked; load-scenario UBs (0.56%) are optimistic. A calibration set from another `agg_profile` was not generated.
22. **Hidden-layer figures**: `fig1`-`fig4` PNGs were generated before WS0 corrections (`make_figures.py` still calls the legacy table); `results/figures/post/` does not exist yet (agent V). `run_all.sh` does not yet call exp07/`--ci`/`--xcal`/exp06 hooks.
23. **Dataset realism**: duplicates ~4% of random N-1 rows; hard-to-solve cases dropped by datakit (N-2: 1; N-5: 6) so NR flat-start robustness is conditioned on datakit's AC-feasible scenarios.
