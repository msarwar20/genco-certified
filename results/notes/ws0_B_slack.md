# WS0.4 (B): scripted slack-voltage audit

Date: 2026-10-11. Script: `scripts/exp06_slack_audit.py`. Outputs: `results/exp06/` (per checkpoint `<tag>_scenarios.parquet`, `<tag>_hops.parquet`, `<tag>_summary.json`; aggregate `summary.csv`; stdout in `run.log`). Figures: `results/figures/fig2_hidden_slack.png` (now 5 rows, GOC 2000 Small added) and `fig2b_slack_locality.png` (new), both from `scripts/make_figures.py` (`fig_slack`, `fig_slack_locality`).

## What was done

Every test scenario of seven checkpoints was analysed (no 64/256-scenario subsets): IEEE 14/30/57/118 Base, IEEE 118 Small/Tiny (paper-test split, 631-638 scenarios each) and GOC 2000 Small (`splits_heldout`, 198 scenarios from 9 load scenarios). Batch size 32 (8 for GOC 2000). Run time 3-35 s per IEEE checkpoint, 22 s for GOC 2000.

- Slack Vm error, reported mismatch (state as GENCO outputs it) and physical mismatch (REF/PV Vm and REF Va reset to the setpoints), per scenario. Per-scenario `ref_vm_err`, `res_raw_*`, `res_phys_*` and `err_l2` equal the stored `results/exp01` values to the last bit on all seven checkpoints, so the script reproduces the old pipeline exactly.
- Locality: BFS hop distance to the slack from `p.f/p.t`; whether the worst physical-mismatch bus is adjacent to the slack; share of total per-bus |dS| held by 1-hop neighbours; mean |dS| per hop bin 1..5 and 6+.
- Uniform re-anchoring: all PQ-bus Vm shifted by (true minus predicted slack Vm), or scaled by their ratio.
- Mask surgery (IEEE 118 only), through the model on a deep copy of each batch. `visible`: REF Vm in `x` set to the truth and unmasked. `clamp_only`: same, plus a pre-hook on `input_proj_bus` that zeroes REF Vm in the projection input, so the setpoint enters only through the `torch.where` clamp.
- Statistics: 95% percentile CIs from a cluster bootstrap over LOAD scenarios (2,000 draws; `load_scenario_idx` read from raw `bus_data` through the sorted ids in `test.pt`; the order is verified by comparing each scenario's total load with the raw Pd sum).

Two definitions of per-bus |dS| are stored. "full" is the one specified: |V conj(Ybus V) - Sbus| with the slack excluded. It also counts the Q mismatch at PV buses, where Q is a free variable and Sbus holds the true Qg. "constrained" (suffix `_pf`) keeps only the components the power-flow equations fix (P at PV and PQ buses, Q at PQ buses), which is what the "max mismatch" metric measures. Both are reported; they differ on IEEE 57 (see below).

## Results (medians, [95% cluster-bootstrap CI over load scenarios])

Slack Vm error and the reported-vs-physical gap (ratio of medians):

| Grid / model | scenarios (load) | Slack Vm err median (p.u.) | p95 / max | Max mismatch reported (MW) | physical (MW) | Ratio |
|---|---|---|---|---|---|---|
| IEEE 14 Base | 638 (29) | 0.00027 [0.00023, 0.00033] | 0.00088 / 0.0014 | 0.0048 [0.0045, 0.0052] | 0.173 [0.148, 0.221] | 36x [31, 46] |
| IEEE 30 Base | 638 (29) | 0.00074 [0.00058, 0.00085] | 0.0020 / 0.0041 | 0.0126 [0.0116, 0.0140] | 0.491 [0.402, 0.561] | 39x [31, 46] |
| IEEE 57 Base | 631 (29) | 0.00095 [0.00075, 0.0011] | 0.0033 / 0.0079 | 0.0329 [0.0290, 0.0385] | 1.04 [0.853, 1.26] | 32x [25, 39] |
| IEEE 118 Base | 638 (29) | 0.0063 [0.0047, 0.0074] | 0.019 / 0.032 | 0.288 [0.235, 0.356] | 17.4 [13.7, 20.8] | 60x [45, 79] |
| IEEE 118 Small | 638 (29) | 0.0057 [0.0041, 0.0077] | 0.019 / 0.028 | 0.454 [0.350, 0.573] | 16.8 [11.7, 22.2] | 37x [26, 48] |
| IEEE 118 Tiny | 638 (29) | 0.0068 [0.0050, 0.0084] | 0.017 / 0.035 | 0.583 [0.511, 0.648] | 20.2 [14.5, 23.9] | 35x [25, 43] |
| GOC 2000 Small | 198 (9) | 0.019 [0.014, 0.027] | 0.032 / 0.032 | 9.11 [8.30, 9.92] | 238 [164, 402] | 26x [17, 44] |

Locality (physical state; neighbours = 1-hop neighbours of the slack; "uniform" = share of non-slack buses that are neighbours, i.e. what a spatially uniform error would give):

| Grid / model | Slack neighbours | Worst bus adjacent, full | Worst bus adjacent, constrained | Share of |dS| in neighbours, full | constrained | Uniform share |
|---|---|---|---|---|---|---|
| IEEE 14 Base | 2 | 0.992 [0.984, 0.998] | 0.989 [0.980, 0.997] | 0.81 [0.79, 0.82] | 0.92 [0.90, 0.93] | 0.15 |
| IEEE 30 Base | 2 | 0.976 [0.959, 0.991] | 0.973 [0.958, 0.987] | 0.73 [0.70, 0.75] | 0.86 [0.83, 0.89] | 0.07 |
| IEEE 57 Base | 4 | 0.867 [0.815, 0.910] | 0.981 [0.967, 0.992] | 0.61 [0.59, 0.62] | 0.84 [0.80, 0.87] | 0.07 |
| IEEE 118 Base | 6 | 0.970 [0.945, 0.991] | 0.992 [0.978, 1.000] | 0.50 [0.48, 0.53] | 0.76 [0.70, 0.80] | 0.05 |
| IEEE 118 Small | 6 | 0.966 [0.942, 0.984] | 0.992 [0.984, 0.998] | 0.48 [0.44, 0.51] | 0.71 [0.62, 0.77] | 0.05 |
| IEEE 118 Tiny | 6 | 0.939 [0.898, 0.975] | 0.984 [0.969, 0.997] | 0.47 [0.43, 0.49] | 0.67 [0.59, 0.71] | 0.05 |
| GOC 2000 Small | 1 | 0.990 [0.970, 1.000] | 1.000 | 0.08 [0.06, 0.15] | 0.09 [0.07, 0.18] | 0.0005 |

Mean |dS| per bus by hop distance, IEEE 118 Base (median over scenarios, MW): hop 1: 5.6, hop 2: 1.2, hop 3: 0.075, hop 4-6+: 0.08-0.11. Hops 1-2 together hold about 85% of the total. GOC 2000 Small: 238, 21, then a floor of about 1.1-1.5 MW from hop 3 outwards.

Uniform re-anchoring (medians; "worse" = fraction of scenarios where the shift is worse than doing nothing, for mismatch / state error):

| Grid / model | Phys. max mismatch: plain | shift | scale | State err l2: plain | shift | scale | shift worse |
|---|---|---|---|---|---|---|---|
| IEEE 14 Base | 0.173 | 0.295 | 0.279 | 0.00043 | 0.00093 | 0.00091 | 0.95 / 0.98 |
| IEEE 30 Base | 0.491 | 1.87 | 1.79 | 0.0017 | 0.0038 | 0.0037 | 1.00 / 0.98 |
| IEEE 57 Base | 1.04 | 3.02 | 2.94 | 0.0042 | 0.0077 | 0.0077 | 0.99 / 0.89 |
| IEEE 118 Base | 17.4 | 187 [147, 227] | 183 | 0.049 | 0.076 [0.064, 0.091] | 0.075 | 1.00 / 1.00 |
| IEEE 118 Small | 16.8 | 175 | 174 | 0.061 | 0.080 | 0.079 | 1.00 / 1.00 |
| IEEE 118 Tiny | 20.2 | 209 | 207 | 0.072 | 0.094 | 0.093 | 1.00 / 1.00 |
| GOC 2000 Small | 238 | 2,260 [1,710, 3,540] | 2,295 | 4.22 | 4.25 | 4.25 | 1.00 / 1.00 |

Mask surgery, IEEE 118 (638 scenarios; NR iterations to 1e-6 MW from the physical state):

| Model | Variant | Phys. max mismatch (MW) | State err l2 | Err ratio vs plain | NR iterations (mean) |
|---|---|---|---|---|---|
| Base | plain | 17.4 [13.7, 20.8] | 0.049 [0.035, 0.071] | 1 | 2.15 [2.06, 2.25] |
| | visible | 494 [369, 603] | 0.83 [0.65, 0.94] | 16.8 [11.2, 24.9] | 3.48 [3.31, 3.65] |
| | clamp_only | 10.9 [8.6, 13.1] | 0.281 [0.246, 0.329] | 5.7 [3.8, 8.4] | 2.98 [2.96, 2.99] |
| Small | plain | 16.8 [11.7, 22.2] | 0.061 [0.041, 0.092] | 1 | 2.22 [2.10, 2.35] |
| | visible | 4.37 [3.55, 6.11] | 0.371 [0.198, 0.499] | 6.1 [3.5, 9.7] | 2.82 [2.69, 2.93] |
| | clamp_only | 1.34 [1.15, 1.56] | 0.068 [0.045, 0.096] | 1.11 [0.89, 1.48] | 2.13 [2.05, 2.24] |
| Tiny | plain | 20.2 [14.5, 23.9] | 0.072 [0.054, 0.087] | 1 | 2.18 [2.10, 2.26] |
| | visible | 1.25 [1.03, 1.44] | 0.105 [0.082, 0.130] | 1.46 [1.08, 2.17] | 2.27 [2.16, 2.40] |
| | clamp_only | 0.78 [0.71, 0.91] | 0.069 [0.056, 0.086] | 0.95 [0.87, 1.13] | 2.14 [2.08, 2.22] |

## Against the RESEARCH_LOG claims

Confirmed:
- Slack Vm errors and the 26-60x reported-to-physical gaps reproduce on all scenarios (IEEE 14-57: 32-39x; IEEE 118 Base 60x; GOC 2000 26x). The CIs are wide, especially on GOC 2000 (9 load scenarios).
- IEEE 118 Base locality: the worst bus is a slack neighbour in 97.0% of all 638 scenarios (98% in the old 64), and the 6 neighbours hold a median 50% of total |dS| (53% in the old 64; the first 64 scenarios give 98.4% / 52.5% here). The claim holds.
- It also holds on IEEE 14, 30 and 57 and on Small/Tiny: worst bus adjacent in 97-99% of scenarios with the constrained definition (87-99% with the full one, IEEE 57 being the low outlier, see item 2), and the neighbours carry 5-11 times their uniform share.
- Uniform re-anchoring makes things worse everywhere: IEEE 118 Base 17.4 to 187 MW, error 0.049 to 0.076 (old 256-scenario numbers: 15.9 to 174, 0.039 to 0.067). It is worse in about 95-100% of scenarios on every grid for mismatch, and 89-100% for state error.
- The median max PQ-bus |dVm| on IEEE 118 Base is 0.00105 p.u. (old log 0.00095), so PQ voltages are accurate and only the slack is off.

Contradictions and surprises:
1. **GOC 2000 neighbour share was overstated.** The log says the single neighbour holds 15%. On all 198 scenarios the median is 8% [6, 15]; the old 64 scenarios were 3 load scenarios (median 15%), and per load scenario the median ranges 2-18%. The worst-bus-adjacent claim is fine (99% full, 100% constrained). The reason the share is small is that 88% of the total |dS| sits in the 6+ bin: about 2,000 buses each carry a roughly 1 MW floor, against 238 MW at the neighbour. "Local" is true per bus (hop-1 mean is about 200 times the floor) but not in aggregate share. Enrichment over a uniform spread (155x) is the size-independent way to say it.
2. **IEEE 57 under the "full" |dS|.** The worst bus is adjacent in only 86.7% of scenarios (12% at hop 2), against 98.1% with the constrained definition. The PV-bus Q "mismatch" (relative to the true Qg) adds a second spatial pattern. The constrained version matches the "max mismatch" metric and should be the one quoted for physical mismatch; the full version is the one as specified. Choose one and state it. Neighbour shares are higher with the constrained definition on every grid (e.g. IEEE 118 Base 76% vs 50%).
3. **Mask surgery fails only on Base.** The F2 sentence "zero-shot mask surgery fails" is a Base result (visible 494 MW, 16.8x the state error; clamp-only 10.9 MW with 5.7x the state error and 2.98 vs 2.15 NR iterations; old log 512 MW, 8 MW, 4.5x). On Small and Tiny, clamp-only works: physical max mismatch falls 13x (Small) and 26x (Tiny), the state error is unchanged (ratio 1.11 [0.89, 1.48] and 0.95 [0.87, 1.13]) and NR iterations are 2.13 / 2.14, no worse than plain (2.22 / 2.18; CIs overlap). `visible` is mixed: it costs 6.1x state error on Small and 1.46x on Tiny, with mismatch down to 4.4 and 1.25 MW. So the log cannot say mask surgery fails in general. A fair summary is that it depends on the model size, and the Tiny/Small clamp-only variant is a nearly free zero-shot fix for the reported mismatch gap. The slack-aware fine-tune was done on Base only, where surgery does fail, so its conclusion stands, but the motivation "no zero-shot fix exists" should be restricted to Base. Note the max mismatch under clamp-only (0.78-1.3 MW) is still above the reported-metric level (0.45-0.58 MW).
4. **IEEE 118 Base ratio (60x) is higher than Small/Tiny (35-37x)** and than the other grids (31-39x). The CIs only just separate (Base lower bound 45 vs Tiny upper 43), mostly because Base's reported mismatch is lowest (0.29 vs 0.45-0.58 MW) while the physical mismatch is similar (17-20 MW). Say "about 35-60x" rather than singling out 60x.
5. Minor: the old log's re-anchoring numbers (256 scenarios) were slightly lower than the all-scenario ones, within CIs.

## Caveats

- One seed (0) per checkpoint. The load-scenario bootstrap treats the 29 (IEEE) / 9 (GOC) load scenarios as exchangeable; the GOC 2000 intervals are wide and its p95 and max of the slack error coincide (0.0316) because there are at most 9 distinct setpoints.
- The "physical" state resets REF/PV Vm and REF Va to the datakit setpoints (as the earlier analysis did); the true state is datakit's float32 solution.
- Mask surgery `visible` hands the model a value it never saw during training (REF Vm is always masked to zero), so its failure on Base is a distribution-shift result, not evidence about the architecture.
- `scripts/run_all.sh` is not edited here (not in this task's ownership). Lines to add after the exp01 block:
  `for gs in case14_ieee:base case30_ieee:base case57_ieee:base case118_ieee:base case118_ieee:small case118_ieee:tiny; do $PY scripts/exp06_slack_audit.py --grid ${gs%%:*} --size ${gs##*:}; done`
  `$PY scripts/exp06_slack_audit.py --grid case2000_goc --size small --split splits_heldout --batch_size 8`
