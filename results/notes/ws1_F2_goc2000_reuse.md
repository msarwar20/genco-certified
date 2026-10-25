# WS1-F2: factor-reuse baselines on GOC 2000 (Small, seed 0, held-out)

Commands (tag `case2000_goc_small_seed0_h879`; `--grid case2000_goc --size small --seed 0 --split splits_heldout`):
1. `exp07_industrial_baselines.py <above> --tag <tag>` (207 s, 4 threads)
2. `... --timing <above> --tag <tag> --repeats 3 --max_loads 10` (1 thread)
3. `... --summarize --tags <tag> --timing_tag <tag>`; outputs renamed `results/exp07/*_goc2000.*`, existing 118/500 files restored.

Iterations: n = 879 contingencies, 40 load scenarios. Timing: n = 220, first 10 load scenarios. Cluster bootstrap over load scenarios.

Paired GENCO vs base-case start, 1e-4 MW (1e-2 in brackets). Break-even in ms, measured (negative: no benefit possible).

| Solver | solve saving [95% CI] | iter. saving | GENCO fewer / more | break-even, same solver | vs cheapest base-start (FDLF, 1.68 ms) |
|---|---|---|---|---|---|
| Full NR | -26% [-29, -24] (-27%) | -26% | 1% / 64% | -3.29 (-2.81) | -14.6 (-12.7) |
| chordC | 2% [1, 2] (-4%) | 3% | 25% / 18% | +0.11 (+0.03) | -0.86 (-0.95) |
| FDLF | -18% [-20, -16] (-20%) | -23% | 2% / 92% | -0.19 (-0.15) | -0.19 (-0.15) |
| plain chord | 15% [14, 16], 745 both converged (22%) | 15% | 86% / 5% | +0.38 (+0.60) | -2.65 (-1.67) |

Plain chord fails to converge in 50 iterations for 15.0% (base) and 12.9% (GENCO) of contingencies at 1e-4 (9.1% / 6.9% at 1e-2). Mean iterations to 1e-4, base -> GENCO: NR 2.37 -> 3.00, chordC 4.04 -> 3.93, FDLF 5.16 -> 6.34.

Reading. GENCO's start is worse than the base case for full NR and FDLF on this grid. The chordC break-even is 0.03 to 0.11 ms. The largest positive break-even is 0.60 ms (plain chord, 1e-2), against measured GENCO CPU inference of 67 to 108 ms. No wall-clock benefit.

Caveats. Timing ran at load average 5.4 -> 4.7 on 10 cores (another inference job in parallel), three repeats: provisional. The base factorization is amortised over 22 contingencies per load scenario (this set's own count), not 240 as in the 118 tables (adds 0.18 to 0.24 ms). Only 30 of 879 contingencies have a PV->PQ change. `exp07_goc2000.log` also reprints the older timing tables (the summarizer reads every timing json). `timing_run_goc2000.log` was reconstructed from the json after an accidental overwrite.
