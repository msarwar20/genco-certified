# WS1-F: factor-reuse baselines, seeds and sizes, intervals

Date: 2026-10-11. Scope: PLAN WS1.2, WS1.4 (CPU part), WS2.1, WS2.2.
New code: `src/genco_certified/fastpf.py`, `scripts/exp07_industrial_baselines.py`, `tests/test_fastpf.py`, `scripts/analyze_exp02.py --ci` (the `--v3` mode and the legacy CLI behave as before; `--v3` reproduces `results/policy_v3.log` and all `v3_*.csv` bit for bit).
Outputs: `results/exp07/*`, `results/policy_v3_ci.log`, `results/exp02/v3_ci_*.csv`, `v3_seeds_*.csv`, and the new `results/exp02/*` runs for Base seeds 1 and 42 and for Small and Tiny.
Timings are **provisional**: the machine was shared (load average 2.3 to 5 on 10 cores, see section 4).

## 1. Verdict

Under industry-style factor reuse a GENCO start still saves work, but much less than "one Newton iteration", and it does not save time. On IEEE 118, paired over the same contingencies, a GENCO start cuts the triangular solves needed to reach 1e-4 MW by 20% for full Newton with refactorization (95% CI 18 to 21%), by 21% for a Newton step that reuses the base-case factorization with outage compensation (20 to 21%), and by only 5% for fast-decoupled XB with outage compensation (2 to 8%). For the fast-decoupled solver the GENCO start is *worse* than the base-case start in 23% of the contingencies. The saving grows with outage order (N-5: 32%, 11%, 10%) and is the same on the exhaustive N-1 workload (6,900 contingencies: 19%, 20%, 5%).

In time, the cheapest solver from the base-case start (fast-decoupled with compensation) takes 0.35 ms per contingency to 1e-4 MW in this python/scipy implementation, and a GENCO start saves 0.015 ms of that. Break-even GENCO inference time per contingency (the solver time that a GENCO start saves):

| solver | break-even inference time, paper set | N-5 |
|---|---|---|
| full Newton, refactorizing every iteration | 0.31 ms | 0.63 ms |
| chord Newton with outage-compensated Jacobian | 0.04 ms | 0.06 ms |
| fast-decoupled XB with outage compensation | 0.015 ms | 0.04 ms |

Measured GENCO inference is 7.7 to 24 ms per scenario (CPU and MPS, `results/runtime`, measured earlier). That is at least 12 times the largest break-even anywhere in this study (0.63 ms) and 500 to 1,600 times the break-even against the cheapest solver on the paper set (190 to 590 times at N-5). So on CPU at 118 buses a GENCO start does not pay for itself. A GPU or batched inference would have to reach roughly 0.02 ms per scenario to matter against fast-decoupled; that is not measured and looks out of reach. A count-based cost model that charges a compiled-code-like Jacobian assembly gives the same ranking with smaller break-evens (0.01 to 0.08 ms for full Newton, chordC and fast-decoupled, section 4), so the python overhead of the solvers does not drive the conclusion. Where no base case is available (full Newton from a flat start, 2.55 ms), a GENCO start would be worth up to 1.3 ms.

The two earlier headlines survive in a narrower form. The equal-cost screening result (GENCO + 1 step misses as few elements as base case + 2 steps) holds for all three Base seeds and for Small and Tiny (section 5). But "a GENCO prediction is worth about one Newton iteration" is a statement in iterations of full Newton: with factor reuse a GENCO start is worth 1.3 iterations of the compensated chord, 0.3 iterations of fast-decoupled XB, and about 0.5 iterations of full Newton to 1e-4 MW (1.0 at N-5). Because an iteration with reuse costs a solve rather than a factorization, iteration counts are not a transferable price. The screening claim should be tied to the verifier, and the "worth" claim should be stated in time.

## 2. What was built

`fastpf.py` works on `acpf.PFProblem` objects: the contingency problem from `acpf.extract_problems`, and a base-case problem. The base-case problem comes from `solve_base_case`, which calls `acpf.base_case_voltages` itself and only intercepts its `problem_from_raw` call, so the voltages are bit-identical to what exp02 uses (tested).

- Chord Newton (`ChordContext`). The Jacobian of the base topology at the base-case solution is factorized once per load scenario (splu); every contingency iterates x <- x - J0^-1 F_c(x) on its own mismatch. A generator outage that turns a PV bus into PQ changes the unknown set. Default: one extra factorization per distinct (pv, pq) pattern and load scenario, cached and counted (`new_factors`). Option `bordered=True` keeps the base factorization and solves the enlarged system with a Schur complement (one setup solve per new PQ bus, two base solves per iteration); iterates are identical (tested).
- Added by me, not in the brief: chord Newton with an outage-compensated Jacobian (`chordC`). The plain chord needs the base Jacobian to be close to the outaged one. It is not: the linear rate of the iteration is the self-PTDF of the outaged branch, (baseMVA/x) u' B'^-1 u, which I checked against the spectral radius of I - J0^-1 Jc on 15 single-branch outages (agreement within 0.03, range 0.2 to 0.87; a test covers it), so plain chord needs 18 iterations on average at 1e-4 MW (paper set) and does not reach it in 50 for 20% of the contingencies. `chordC` evaluates the Jacobian at the base-case voltages as before, but removes the outaged branches' contribution by Woodbury (rank at most 4 per branch, about 3 setup solves per outaged branch). It is the fair "no refactorization" Newton baseline and converges in 4 iterations.
- Fast-decoupled XB (`FDLFContext`, Stott and Alsac; MATPOWER `makeB` alg 2). B' is the Laplacian of baseMVA/x over PV and PQ buses (r, shunts, taps and charging ignored), B'' is -Im(Ybus) over PQ buses. Both are factorized once for the base topology. Branch outages are Sherman-Morrison-Woodbury updates on the base factors (rank 1 per branch in B', rank at most 2 in B''; one base solve per half-step plus a small dense update, about 2 setup solves per outaged branch). Iterations alternate P-theta and Q-V half-steps until max |mismatch| <= tolerance, cap 50 iterations. A PQ-pattern change costs one extra B'' factorization (or `bordered=True`). x and tap come from the raw branch table (`problem_from_raw` input), because graphkit's edge attributes do not carry them.
- Counters. `solves` are forward-plus-backward substitutions with one LU (a "triangular-solve pair"), inside the iteration; `setup_solves` are compensation or bordering solves, once per contingency; `new_factors` are factorizations created for the contingency; the per-load-scenario base factorization is counted once (1 for chord, 2 for fast-decoupled) and amortised in section 4. An FDLF iteration is a P and a Q half-step; a crossing after the P half-step counts as k - 0.5.
- Tests (`tests/test_fastpf.py`, 16, all pass; whole suite 58 pass at the time of writing): chord, chordC and FDLF converge to datakit's solution on IEEE 118 scenarios with branch and with generator outages (state error < 1e-5); compensated solves equal a fresh factorization of the outaged B', B'' and Jacobian (1, 3 and 5 outaged branches, plus a PQ-pattern change); bordered equals refactorized; factorization counters; an islanding outage is reported as a failure, not a crash; Clopper-Pearson against `scipy.stats.binomtest`; cluster-bootstrap coverage. Two bugs found and fixed while building it, now covered by tests: the B' ordering after a PV-to-PQ change (unknowns are ordered [pv; pq], so a generator outage permutes them), and the attribution of cached pattern factorizations between the two starts and between solvers sharing a context.

## 3. Iterations and solves from a base-case and from a GENCO start (exp07)

Sets: the paper test set (Base seed 0, 638 contingencies, 29 load scenarios), stress N-1/2/3/5 (30 load scenarios each), and the new exhaustive N-1 sets in `data/exhaustive` (every branch outage, 5,310 contingencies, and every generator outage, 1,590, on the same 30 load scenarios; the intact N-0 variants are dropped). GENCO start: released Base seed 0, setpoints enforced (`p.V_from`, exactly as exp02). Base-case start: `acpf.base_case_voltages`. All CIs are 95% cluster bootstraps over load scenarios (2,000 resamples). "Solves" include the compensation setup solves. Means are over contingencies that reached the tolerance in 50 iterations (30 for full Newton); failures are in the last table.

Tolerance 1e-4 MW:

| set | solver | start | iterations [95% CI] | solves incl. setup [95% CI] | LU per contingency | not converged |
|---|---|---|---|---|---|---|
| paper N-1 | nr | flat | 4.00 [4.00, 4.00] | 4.00 [4.00, 4.00] | 4.00 | 0/638 |
| paper N-1 | nr | base | 2.49 [2.45, 2.53] | 2.49 [2.45, 2.53] | 2.49 | 0/638 |
| paper N-1 | nr | genco | 2.00 [2.00, 2.01] | 2.00 [2.00, 2.01] | 2.00 | 0/638 |
| paper N-1 | chord | base | 17.77 [16.46, 19.02] | 17.77 [16.46, 19.02] | 0.26 | 126/638 |
| paper N-1 | chord | genco | 14.65 [13.64, 15.63] | 14.65 [13.64, 15.63] | 0.26 | 52/638 |
| paper N-1 | chordC | base | 4.11 [3.98, 4.25] | 6.23 [6.03, 6.42] | 0.26 | 0/638 |
| paper N-1 | chordC | genco | 2.83 [2.72, 2.95] | 4.95 [4.78, 5.10] | 0.26 | 0/638 |
| paper N-1 | fdlf | base | 5.21 [5.13, 5.29] | 11.83 [11.64, 12.01] | 0.26 | 0/638 |
| paper N-1 | fdlf | genco | 4.91 [4.74, 5.11] | 11.24 [10.88, 11.62] | 0.26 | 0/638 |
| stress N-5 | nr | flat | 4.01 [4.00, 4.02] | 4.01 [4.00, 4.02] | 4.01 | 0/660 |
| stress N-5 | nr | base | 3.08 [3.04, 3.11] | 3.08 [3.04, 3.11] | 3.08 | 0/660 |
| stress N-5 | nr | genco | 2.08 [2.06, 2.11] | 2.08 [2.06, 2.11] | 2.08 | 0/660 |
| stress N-5 | chord | base | 36.15 [34.59, 37.71] | 36.15 [34.59, 37.71] | 0.70 | 459/660 |
| stress N-5 | chord | genco | 32.11 [31.09, 33.13] | 32.11 [31.09, 33.13] | 0.70 | 308/660 |
| stress N-5 | chordC | base | 6.69 [6.35, 7.04] | 17.59 [17.17, 18.05] | 0.70 | 1/660 |
| stress N-5 | chordC | genco | 4.76 [4.51, 5.02] | 15.66 [15.32, 16.01] | 0.70 | 0/660 |
| stress N-5 | fdlf | base | 6.54 [6.35, 6.75] | 20.46 [20.02, 20.92] | 0.70 | 0/660 |
| stress N-5 | fdlf | genco | 5.53 [5.34, 5.73] | 18.44 [17.98, 18.92] | 0.70 | 0/660 |
| exhaustive N-1, branches + generators | nr | flat | 4.00 [4.00, 4.00] | 4.00 [4.00, 4.00] | 4.00 | 0/6900 |
| exhaustive N-1, branches + generators | nr | base | 2.48 [2.47, 2.49] | 2.48 [2.47, 2.49] | 2.48 | 0/6900 |
| exhaustive N-1, branches + generators | nr | genco | 2.00 [2.00, 2.00] | 2.00 [2.00, 2.00] | 2.00 | 0/6900 |
| exhaustive N-1, branches + generators | chord | base | 18.16 [18.01, 18.29] | 18.16 [18.01, 18.29] | 0.23 | 1442/6900 |
| exhaustive N-1, branches + generators | chord | genco | 15.13 [14.97, 15.32] | 15.13 [14.97, 15.32] | 0.23 | 597/6900 |
| exhaustive N-1, branches + generators | chordC | base | 4.05 [3.99, 4.11] | 6.26 [6.20, 6.32] | 0.23 | 10/6900 |
| exhaustive N-1, branches + generators | chordC | genco | 2.81 [2.77, 2.85] | 5.01 [4.98, 5.05] | 0.23 | 3/6900 |
| exhaustive N-1, branches + generators | fdlf | base | 5.10 [5.01, 5.19] | 11.65 [11.48, 11.83] | 0.23 | 0/6900 |
| exhaustive N-1, branches + generators | fdlf | genco | 4.78 [4.61, 4.96] | 11.02 [10.68, 11.39] | 0.23 | 0/6900 |

Tolerance 1e-2 MW:

| set | solver | start | iterations [95% CI] | solves incl. setup [95% CI] | LU per contingency | not converged |
|---|---|---|---|---|---|---|
| paper N-1 | nr | flat | 3.01 [3.00, 3.02] | 3.01 [3.00, 3.02] | 3.01 | 0/638 |
| paper N-1 | nr | base | 2.04 [2.00, 2.07] | 2.04 [2.00, 2.07] | 2.04 | 0/638 |
| paper N-1 | nr | genco | 1.72 [1.58, 1.85] | 1.72 [1.58, 1.85] | 1.72 | 0/638 |
| paper N-1 | chord | base | 14.64 [13.72, 15.57] | 14.64 [13.72, 15.57] | 0.26 | 59/638 |
| paper N-1 | chord | genco | 7.80 [7.16, 8.59] | 7.80 [7.16, 8.59] | 0.26 | 16/638 |
| paper N-1 | chordC | base | 2.74 [2.64, 2.84] | 4.86 [4.69, 5.01] | 0.26 | 0/638 |
| paper N-1 | chordC | genco | 1.93 [1.80, 2.05] | 4.05 [3.86, 4.22] | 0.26 | 0/638 |
| paper N-1 | fdlf | base | 3.33 [3.27, 3.40] | 8.08 [7.94, 8.23] | 0.26 | 0/638 |
| paper N-1 | fdlf | genco | 2.95 [2.80, 3.10] | 7.31 [7.02, 7.62] | 0.26 | 0/638 |
| stress N-5 | nr | flat | 3.08 [3.06, 3.11] | 3.08 [3.06, 3.11] | 3.08 | 0/660 |
| stress N-5 | nr | base | 2.49 [2.44, 2.54] | 2.49 [2.44, 2.54] | 2.49 | 0/660 |
| stress N-5 | nr | genco | 1.88 [1.84, 1.93] | 1.88 [1.84, 1.93] | 1.88 | 0/660 |
| stress N-5 | chord | base | 31.86 [30.88, 32.83] | 31.86 [30.88, 32.83] | 0.70 | 280/660 |
| stress N-5 | chord | genco | 21.55 [20.38, 22.75] | 21.55 [20.38, 22.75] | 0.70 | 102/660 |
| stress N-5 | chordC | base | 4.44 [4.21, 4.68] | 15.35 [15.02, 15.71] | 0.70 | 1/660 |
| stress N-5 | chordC | genco | 2.79 [2.65, 2.92] | 13.69 [13.42, 13.97] | 0.70 | 0/660 |
| stress N-5 | fdlf | base | 4.29 [4.15, 4.44] | 15.95 [15.62, 16.30] | 0.70 | 0/660 |
| stress N-5 | fdlf | genco | 3.40 [3.27, 3.53] | 14.17 [13.84, 14.50] | 0.70 | 0/660 |
| exhaustive N-1, branches + generators | nr | flat | 3.01 [3.01, 3.01] | 3.01 [3.01, 3.01] | 3.01 | 0/6900 |
| exhaustive N-1, branches + generators | nr | base | 2.04 [2.03, 2.05] | 2.04 [2.03, 2.05] | 2.04 | 0/6900 |
| exhaustive N-1, branches + generators | nr | genco | 1.53 [1.40, 1.67] | 1.53 [1.40, 1.67] | 1.53 | 0/6900 |
| exhaustive N-1, branches + generators | chord | base | 15.31 [15.16, 15.46] | 15.31 [15.16, 15.46] | 0.23 | 635/6900 |
| exhaustive N-1, branches + generators | chord | genco | 8.16 [7.90, 8.43] | 8.16 [7.90, 8.43] | 0.23 | 101/6900 |
| exhaustive N-1, branches + generators | chordC | base | 2.74 [2.71, 2.78] | 4.95 [4.91, 4.98] | 0.23 | 3/6900 |
| exhaustive N-1, branches + generators | chordC | genco | 1.79 [1.68, 1.89] | 3.99 [3.89, 4.10] | 0.23 | 0/6900 |
| exhaustive N-1, branches + generators | fdlf | base | 3.27 [3.22, 3.32] | 7.99 [7.90, 8.09] | 0.23 | 0/6900 |
| exhaustive N-1, branches + generators | fdlf | genco | 2.83 [2.70, 2.98] | 7.13 [6.86, 7.42] | 0.23 | 0/6900 |

Paired difference, GENCO start minus base-case start (tolerance 1e-4 MW; the plain chord rows only use contingencies where both starts converged, and the excluded count is large):

| set | solver | d iterations [95% CI] | iteration saving | solve saving | GENCO fewer/more | n (excluded) |
|---|---|---|---|---|---|---|
| paper N-1 | nr | -0.49 [-0.53, -0.45] | 0.20 [0.18, 0.21] | 0.20 [0.18, 0.21] | 48%/1% | 638 (0) |
| paper N-1 | chord | -6.45 [-7.04, -5.90] | 0.36 [0.35, 0.38] | 0.36 [0.35, 0.38] | 93%/0% | 512 (126) |
| paper N-1 | chordC | -1.28 [-1.34, -1.21] | 0.31 [0.30, 0.32] | 0.21 [0.20, 0.21] | 84%/1% | 638 (0) |
| paper N-1 | fdlf | -0.29 [-0.46, -0.12] | 0.06 [0.02, 0.09] | 0.05 [0.02, 0.08] | 45%/23% | 638 (0) |
| stress N-1 | nr | -0.50 [-0.55, -0.45] | 0.20 [0.18, 0.22] | 0.20 [0.18, 0.22] | 49%/1% | 660 (0) |
| stress N-1 | chord | -6.96 [-7.46, -6.48] | 0.36 [0.34, 0.38] | 0.36 [0.34, 0.38] | 92%/2% | 520 (140) |
| stress N-1 | chordC | -1.24 [-1.34, -1.15] | 0.31 [0.29, 0.32] | 0.20 [0.18, 0.21] | 83%/1% | 660 (0) |
| stress N-1 | fdlf | -0.36 [-0.48, -0.23] | 0.07 [0.05, 0.09] | 0.06 [0.04, 0.08] | 45%/20% | 660 (0) |
| stress N-2 | nr | -0.73 [-0.78, -0.69] | 0.27 [0.25, 0.28] | 0.27 [0.25, 0.28] | 70%/0% | 659 (0) |
| stress N-2 | chord | -9.41 [-9.84, -8.95] | 0.34 [0.33, 0.36] | 0.34 [0.33, 0.36] | 97%/1% | 405 (254) |
| stress N-2 | chordC | -1.54 [-1.65, -1.45] | 0.32 [0.30, 0.33] | 0.16 [0.16, 0.17] | 90%/0% | 659 (0) |
| stress N-2 | fdlf | -0.65 [-0.77, -0.53] | 0.12 [0.10, 0.14] | 0.09 [0.08, 0.11] | 63%/9% | 659 (0) |
| stress N-3 | nr | -0.89 [-0.92, -0.85] | 0.31 [0.30, 0.31] | 0.31 [0.30, 0.31] | 84%/0% | 660 (0) |
| stress N-3 | chord | -9.88 [-10.50, -9.25] | 0.34 [0.32, 0.36] | 0.34 [0.32, 0.36] | 99%/0% | 330 (330) |
| stress N-3 | chordC | -1.71 [-1.83, -1.59] | 0.31 [0.30, 0.33] | 0.14 [0.13, 0.15] | 92%/0% | 656 (4) |
| stress N-3 | fdlf | -0.93 [-1.04, -0.82] | 0.16 [0.14, 0.18] | 0.12 [0.10, 0.13] | 79%/4% | 660 (0) |
| stress N-5 | nr | -0.99 [-1.02, -0.96] | 0.32 [0.32, 0.33] | 0.32 [0.32, 0.33] | 93%/0% | 660 (0) |
| stress N-5 | chord | -11.19 [-11.98, -10.40] | 0.31 [0.29, 0.33] | 0.31 [0.29, 0.33] | 100%/0% | 200 (460) |
| stress N-5 | chordC | -1.99 [-2.15, -1.83] | 0.30 [0.28, 0.31] | 0.11 [0.11, 0.12] | 96%/0% | 659 (1) |
| stress N-5 | fdlf | -1.01 [-1.09, -0.93] | 0.15 [0.14, 0.17] | 0.10 [0.09, 0.11] | 82%/2% | 660 (0) |
| exhaustive N-1, branches | nr | -0.53 [-0.55, -0.51] | 0.21 [0.20, 0.21] | 0.21 [0.20, 0.21] | 53%/1% | 5310 (0) |
| exhaustive N-1, branches | chord | -8.74 [-8.98, -8.47] | 0.36 [0.35, 0.38] | 0.36 [0.35, 0.38] | 99%/1% | 3878 (1432) |
| exhaustive N-1, branches | chordC | -1.30 [-1.34, -1.25] | 0.32 [0.31, 0.33] | 0.19 [0.18, 0.19] | 84%/1% | 5310 (0) |
| exhaustive N-1, branches | fdlf | -0.36 [-0.51, -0.19] | 0.07 [0.04, 0.10] | 0.06 [0.03, 0.08] | 47%/21% | 5310 (0) |
| exhaustive N-1, generators | nr | -0.31 [-0.32, -0.29] | 0.13 [0.13, 0.14] | 0.13 [0.13, 0.14] | 27%/1% | 1590 (0) |
| exhaustive N-1, generators | chord | -1.15 [-1.22, -1.10] | 0.29 [0.28, 0.30] | 0.29 [0.28, 0.30] | 75%/1% | 1580 (10) |
| exhaustive N-1, generators | chordC | -1.15 [-1.22, -1.10] | 0.29 [0.28, 0.30] | 0.29 [0.28, 0.30] | 75%/1% | 1580 (10) |
| exhaustive N-1, generators | fdlf | -0.18 [-0.30, -0.04] | 0.04 [0.01, 0.06] | 0.04 [0.01, 0.06] | 40%/29% | 1590 (0) |
| exhaustive N-1, branches + generators | nr | -0.48 [-0.49, -0.46] | 0.19 [0.19, 0.20] | 0.19 [0.19, 0.20] | 47%/1% | 6900 (0) |
| exhaustive N-1, branches + generators | chord | -6.54 [-6.72, -6.35] | 0.36 [0.35, 0.37] | 0.36 [0.35, 0.37] | 92%/1% | 5458 (1442) |
| exhaustive N-1, branches + generators | chordC | -1.26 [-1.30, -1.23] | 0.31 [0.30, 0.32] | 0.20 [0.20, 0.21] | 82%/1% | 6890 (10) |
| exhaustive N-1, branches + generators | fdlf | -0.32 [-0.46, -0.16] | 0.06 [0.03, 0.09] | 0.05 [0.03, 0.08] | 45%/23% | 6900 (0) |

Tolerance 1e-2 MW:

| set | solver | d iterations [95% CI] | iteration saving | solve saving | GENCO fewer/more | n (excluded) |
|---|---|---|---|---|---|---|
| paper N-1 | nr | -0.32 [-0.45, -0.19] | 0.16 [0.09, 0.22] | 0.16 [0.09, 0.22] | 35%/4% | 638 (0) |
| paper N-1 | chord | -8.25 [-8.81, -7.67] | 0.56 [0.53, 0.59] | 0.56 [0.53, 0.59] | 84%/2% | 579 (59) |
| paper N-1 | chordC | -0.81 [-0.90, -0.71] | 0.29 [0.26, 0.33] | 0.17 [0.15, 0.19] | 60%/4% | 638 (0) |
| paper N-1 | fdlf | -0.39 [-0.52, -0.24] | 0.12 [0.07, 0.16] | 0.10 [0.06, 0.13] | 58%/19% | 638 (0) |
| stress N-5 | nr | -0.60 [-0.67, -0.54] | 0.24 [0.22, 0.27] | 0.24 [0.22, 0.27] | 54%/0% | 660 (0) |
| stress N-5 | chord | -14.69 [-15.36, -14.05] | 0.46 [0.44, 0.48] | 0.46 [0.44, 0.48] | 99%/0% | 380 (280) |
| stress N-5 | chordC | -1.68 [-1.85, -1.53] | 0.38 [0.36, 0.40] | 0.11 [0.10, 0.12] | 93%/0% | 659 (1) |
| stress N-5 | fdlf | -0.89 [-0.96, -0.81] | 0.21 [0.19, 0.22] | 0.11 [0.10, 0.12] | 80%/1% | 660 (0) |
| exhaustive N-1, branches + generators | nr | -0.51 [-0.64, -0.37] | 0.25 [0.18, 0.32] | 0.25 [0.18, 0.32] | 51%/3% | 6900 (0) |
| exhaustive N-1, branches + generators | chord | -8.67 [-8.96, -8.37] | 0.57 [0.55, 0.58] | 0.57 [0.55, 0.58] | 88%/2% | 6265 (635) |
| exhaustive N-1, branches + generators | chordC | -0.96 [-1.07, -0.86] | 0.35 [0.31, 0.39] | 0.19 [0.17, 0.22] | 67%/3% | 6897 (3) |
| exhaustive N-1, branches + generators | fdlf | -0.43 [-0.56, -0.30] | 0.13 [0.09, 0.17] | 0.11 [0.07, 0.14] | 58%/16% | 6900 (0) |

Not converged within the iteration cap (tolerance 1e-4 MW). Full Newton never failed from any start. Every failure is the iteration cap (`max_it`); there was no divergence and no singular compensation. The few chordC failures are at higher outage order or after a PV-to-PQ change, where the unchanged base Jacobian converges slowly:

| set | chord base | chord genco | chordC base | chordC genco | fdlf base | fdlf genco |
|---|---|---|---|---|---|---|
| paper N-1 | 126/638 (19.7%) | 52/638 (8.2%) | 0/638 (0.0%) | 0/638 (0.0%) | 0/638 (0.0%) | 0/638 (0.0%) |
| stress N-1 | 140/660 (21.2%) | 59/660 (8.9%) | 0/660 (0.0%) | 0/660 (0.0%) | 0/660 (0.0%) | 0/660 (0.0%) |
| stress N-2 | 254/659 (38.5%) | 124/659 (18.8%) | 0/659 (0.0%) | 0/659 (0.0%) | 0/659 (0.0%) | 0/659 (0.0%) |
| stress N-3 | 330/660 (50.0%) | 165/660 (25.0%) | 4/660 (0.6%) | 1/660 (0.2%) | 0/660 (0.0%) | 0/660 (0.0%) |
| stress N-5 | 459/660 (69.5%) | 308/660 (46.7%) | 1/660 (0.2%) | 0/660 (0.0%) | 0/660 (0.0%) | 0/660 (0.0%) |
| exhaustive N-1, branches | 1432/5310 (27.0%) | 594/5310 (11.2%) | 0/5310 (0.0%) | 0/5310 (0.0%) | 0/5310 (0.0%) | 0/5310 (0.0%) |
| exhaustive N-1, generators | 10/1590 (0.6%) | 3/1590 (0.2%) | 10/1590 (0.6%) | 3/1590 (0.2%) | 0/1590 (0.0%) | 0/1590 (0.0%) |
| exhaustive N-1, branches + generators | 1442/6900 (20.9%) | 597/6900 (8.7%) | 10/6900 (0.1%) | 3/6900 (0.0%) | 0/6900 (0.0%) | 0/6900 (0.0%) |

Branch outages against generator outages, paper set. Generator outages that turn a PV bus into PQ need one extra factorization unless bordered (0.95 per such contingency, since two contingencies can share a pattern), and for them the plain chord is the same as chordC:

| contingencies | n | solver | start | iterations [95% CI] | solves incl. setup | LU per contingency | not converged |
|---|---|---|---|---|---|---|---|
| branch-only | 466 | nr | base | 2.55 [2.51, 2.59] | 2.55 | 2.55 | 0 |
| branch-only | 466 | nr | genco | 2.01 [2.00, 2.01] | 2.01 | 2.01 | 0 |
| branch-only | 466 | chord | base | 24.70 [23.47, 25.92] | 24.70 | 0.00 | 126 |
| branch-only | 466 | chord | genco | 19.54 [18.29, 20.79] | 19.54 | 0.00 | 52 |
| branch-only | 466 | chordC | base | 4.12 [3.99, 4.26] | 7.02 | 0.00 | 0 |
| branch-only | 466 | chordC | genco | 2.82 [2.70, 2.93] | 5.71 | 0.00 | 0 |
| branch-only | 466 | fdlf | base | 5.25 [5.16, 5.35] | 12.43 | 0.00 | 0 |
| branch-only | 466 | fdlf | genco | 4.89 [4.71, 5.10] | 11.72 | 0.00 | 0 |
| gen-outage (PV->PQ) | 172 | nr | base | 2.35 [2.25, 2.46] | 2.35 | 2.35 | 0 |
| gen-outage (PV->PQ) | 172 | nr | genco | 2.00 [2.00, 2.00] | 2.00 | 2.00 | 0 |
| gen-outage (PV->PQ) | 172 | chord | base | 4.08 [3.68, 4.55] | 4.08 | 0.95 | 0 |
| gen-outage (PV->PQ) | 172 | chord | genco | 2.87 [2.62, 3.18] | 2.87 | 0.95 | 0 |
| gen-outage (PV->PQ) | 172 | chordC | base | 4.08 [3.68, 4.55] | 4.08 | 0.95 | 0 |
| gen-outage (PV->PQ) | 172 | chordC | genco | 2.87 [2.62, 3.18] | 2.87 | 0.95 | 0 |
| gen-outage (PV->PQ) | 172 | fdlf | base | 5.10 [4.91, 5.32] | 10.20 | 0.95 | 0 |
| gen-outage (PV->PQ) | 172 | fdlf | genco | 4.97 [4.77, 5.18] | 9.94 | 0.95 | 0 |

## 4. End-to-end cost and break-even

Method. `--timing` runs single-threaded (BLAS threads pinned to 1) over all 638 paper-set contingencies, five repeats, with fresh solver contexts per repeat, and takes the per-contingency minimum. The timed region contains the solve, the outage-compensation setup, and the cold creation of any new factorization pattern. The per-load-scenario base factorization (Jacobian derivatives and LU for chord, B' and B'' LUs for fast-decoupled) is timed separately and added as base/240, i.e. an exhaustive N-1 sweep of the 118-bus case (186 branches + 54 generators per operating point). For the 22 contingencies per load scenario of the paper set it would be 10 times larger (0.03 ms for chord, 0.02 ms for fdlf), which does not change any ranking. The base-case solution itself is assumed available and free, as in exp02. Full Newton rebuilds and refactorizes the Jacobian each iteration (`acpf.jacobian` + `spsolve`). The problems are the raw-parquet problems (float64); their iteration counts match the graphkit-problem counts of section 3 (e.g. 2.49 and 2.00 for NR).

Load. Two independent runs of the paper set, the first at load average 4.9 to 3.2 and the second at 2.3 to 2.5 on 10 cores (quiet would be < 2.0), agree within 1.5% on every row (`results/exp07/timing_*_run1.json` and `timing_case118_ieee_base_seed0.json`); the N-5 run was at 2.5. I still call all of it provisional. The tables use the second run. Absolute times are for pure python/scipy code: assembling the Jacobian costs 106 triangular-solve pairs here, a mismatch evaluation 1.2, one Jacobian LU 10.6. Compiled code would be faster everywhere. The 'model' column therefore recomputes each cost as counts times measured building-block times, with the Jacobian assembly charged as 2 mismatch evaluations; this favours the classical baseline.

GENCO inference cost (`results/runtime/case118_ieee_base.json`, not re-measured, 4 torch threads):

| GENCO inference | ms per scenario |
|---|---|
| genco_cpu_bs1_ms_per_scen | 24.3 |
| genco_cpu_bs16_ms_per_scen | 12.1 |
| genco_cpu_bs128_ms_per_scen | 14.5 |
| genco_mps_bs16_ms_per_scen | 8.7 |
| genco_mps_bs128_ms_per_scen | 10.3 |
| genco_mps_bs128_9layers_ms_per_scen | 7.7 |

Per-contingency cost, paper set, tolerance 1e-4 MW (ms; solve [95% CI over load scenarios], base LU amortised over 240, `_bd` rows are the bordered variants):

| solver | start | iters | measured solve ms [95% CI] | + base LU / 240 | measured total | model total |
|---|---|---|---|---|---|---|
| nr | flat | 4.00 | 2.547 [2.543, 2.552] | 0.000 | 2.547 | 0.338 |
| nr | base | 2.49 | 1.553 [1.527, 1.578] | 0.000 | 1.553 | 0.213 |
| nr | genco | 2.00 | 1.247 [1.244, 1.250] | 0.000 | 1.247 | 0.172 |
| chord | base | 17.77 | 0.672 [0.653, 0.694] | 0.003 | 0.675 | 0.469 |
| chord | genco | 14.65 | 0.517 [0.498, 0.539] | 0.003 | 0.520 | 0.347 |
| chordC | base | 4.11 | 0.747 [0.731, 0.763] | 0.003 | 0.750 | 0.109 |
| chordC | genco | 2.83 | 0.710 [0.694, 0.726] | 0.003 | 0.713 | 0.086 |
| chordC_bd | base | 4.11 | 0.776 [0.763, 0.788] | 0.003 | 0.778 | n/a |
| chordC_bd | genco | 2.83 | 0.738 [0.725, 0.749] | 0.003 | 0.740 | n/a |
| fdlf | base | 5.21 | 0.347 [0.342, 0.351] | 0.002 | 0.349 | 0.204 |
| fdlf | genco | 4.91 | 0.332 [0.325, 0.339] | 0.002 | 0.334 | 0.193 |
| fdlf_bd | base | 5.21 | 0.367 [0.363, 0.371] | 0.002 | 0.368 | n/a |
| fdlf_bd | genco | 4.91 | 0.354 [0.347, 0.363] | 0.002 | 0.356 | n/a |

Tolerance 1e-2 MW:

| solver | start | iters | measured solve ms [95% CI] | + base LU / 240 | measured total | model total |
|---|---|---|---|---|---|---|
| nr | flat | 3.01 | 1.954 [1.948, 1.960] | 0.000 | 1.954 | 0.256 |
| nr | base | 2.04 | 1.273 [1.253, 1.293] | 0.000 | 1.273 | 0.175 |
| nr | genco | 1.72 | 1.071 [0.986, 1.153] | 0.000 | 1.071 | 0.149 |
| chord | base | 14.64 | 0.531 [0.514, 0.551] | 0.003 | 0.534 | 0.354 |
| chord | genco | 7.80 | 0.323 [0.303, 0.345] | 0.003 | 0.326 | 0.186 |
| chordC | base | 2.74 | 0.715 [0.698, 0.730] | 0.003 | 0.717 | 0.084 |
| chordC | genco | 1.93 | 0.686 [0.669, 0.701] | 0.003 | 0.689 | 0.069 |
| chordC_bd | base | 2.74 | 0.738 [0.725, 0.750] | 0.003 | 0.740 | n/a |
| chordC_bd | genco | 1.93 | 0.711 [0.698, 0.723] | 0.003 | 0.714 | n/a |
| fdlf | base | 3.33 | 0.282 [0.278, 0.285] | 0.002 | 0.284 | 0.136 |
| fdlf | genco | 2.95 | 0.262 [0.256, 0.268] | 0.002 | 0.264 | 0.122 |
| fdlf_bd | base | 3.33 | 0.299 [0.295, 0.302] | 0.002 | 0.300 | n/a |
| fdlf_bd | genco | 2.95 | 0.282 [0.275, 0.288] | 0.002 | 0.283 | n/a |

Break-even GENCO inference time (ms), measured, paper set (the last column compares GENCO + the solver with the cheapest solver run from the base-case start; positive only for fast-decoupled itself):

| solver | tol (MW) | base-start ms | GENCO-start ms | break-even inference ms, same solver | break-even vs cheapest base-start solver (FDLF) |
|---|---|---|---|---|---|
| nr | 0.01 | 1.273 | 1.071 | 0.202 | -0.787 |
| chord | 0.01 | 0.534 | 0.326 | 0.208 | -0.042 |
| chordC | 0.01 | 0.717 | 0.689 | 0.029 | -0.405 |
| fdlf | 0.01 | 0.284 | 0.264 | 0.020 | 0.020 |
| nr | 0.0001 | 1.553 | 1.247 | 0.306 | -0.898 |
| chord | 0.0001 | 0.675 | 0.520 | 0.155 | -0.171 |
| chordC | 0.0001 | 0.750 | 0.713 | 0.037 | -0.364 |
| fdlf | 0.0001 | 0.349 | 0.334 | 0.015 | 0.015 |

Stress N-5 (`results/exp07/timing_stress_n5_base.json`):

| solver | tol (MW) | base-start ms | GENCO-start ms | break-even inference ms, same solver | break-even vs cheapest base-start solver (FDLF) |
|---|---|---|---|---|---|
| nr | 0.01 | 1.581 | 1.188 | 0.393 | -0.761 |
| chord | 0.01 | 1.224 | 0.911 | 0.313 | -0.484 |
| chordC | 0.01 | 1.173 | 1.113 | 0.059 | -0.686 |
| fdlf | 0.01 | 0.427 | 0.386 | 0.041 | 0.041 |
| nr | 0.0001 | 1.941 | 1.311 | 0.630 | -0.802 |
| chord | 0.0001 | 1.362 | 1.237 | 0.126 | -0.728 |
| chordC | 0.0001 | 1.225 | 1.168 | 0.058 | -0.659 |
| fdlf | 0.0001 | 0.509 | 0.467 | 0.041 | 0.041 |

Count-based model, break-even (ms, same solver) on every set; the last column is the cost of the cheapest solver from the base-case start (the whole contingency):

| set | tol (MW) | n | nr | chord | chordC | fdlf | cheapest base-start model cost |
|---|---|---|---|---|---|---|---|
| paper N-1 | 0.01 | 638 | 0.026 | 0.167 | 0.015 | 0.014 | 0.084 |
| paper N-1 | 0.0001 | 638 | 0.041 | 0.122 | 0.024 | 0.011 | 0.109 |
| stress N-1 | 0.01 | 660 | 0.028 | 0.183 | 0.015 | 0.014 | 0.081 |
| stress N-1 | 0.0001 | 660 | 0.042 | 0.126 | 0.023 | 0.013 | 0.105 |
| stress N-2 | 0.01 | 659 | 0.050 | 0.258 | 0.023 | 0.024 | 0.111 |
| stress N-2 | 0.0001 | 659 | 0.061 | 0.155 | 0.029 | 0.024 | 0.141 |
| stress N-3 | 0.01 | 660 | 0.047 | 0.265 | 0.029 | 0.030 | 0.143 |
| stress N-3 | 0.0001 | 660 | 0.074 | 0.137 | 0.033 | 0.034 | 0.177 |
| stress N-5 | 0.01 | 660 | 0.050 | 0.252 | 0.032 | 0.032 | 0.189 |
| stress N-5 | 0.0001 | 660 | 0.082 | 0.098 | 0.037 | 0.036 | 0.231 |
| exhaustive N-1, branches | 0.01 | 5310 | 0.044 | 0.228 | 0.018 | 0.016 | 0.071 |
| exhaustive N-1, branches | 0.0001 | 5310 | 0.044 | 0.153 | 0.024 | 0.013 | 0.096 |
| exhaustive N-1, generators | 0.01 | 1590 | 0.037 | 0.018 | 0.018 | 0.014 | 0.124 |
| exhaustive N-1, generators | 0.0001 | 1590 | 0.025 | 0.023 | 0.023 | 0.006 | 0.149 |
| exhaustive N-1, branches + generators | 0.01 | 6900 | 0.042 | 0.180 | 0.018 | 0.016 | 0.083 |
| exhaustive N-1, branches + generators | 0.0001 | 6900 | 0.040 | 0.123 | 0.024 | 0.011 | 0.108 |

Reading. The largest break-even anywhere is 0.63 ms (full Newton, N-5, measured) and the largest under the model is 0.27 ms (plain chord, N-5, 1e-2; not a solver anyone would run). Against the cheapest base-case-start solver the break-even is 0.01 to 0.04 ms. Measured GENCO inference is 7.7 to 24 ms. Bordering a PV-to-PQ change instead of refactorizing costs slightly more time at 118 buses (+3 to 16%): a 118-bus factorization is cheap, so the generator-outage caveat matters little here.

## 5. Seeds and sizes (WS2.1)

Base: mean ± SD over the three released seeds (0, 1, 42), n = 3. Small and Tiny: seed 0 only. The stress sets are the same for every checkpoint. The paper-set test split is **not**: seeds 1 and 42 were trained on other splits than seed 0, so their paper sets are `splits_paper_test_seed1` (814 contingencies, 37 load scenarios) and `splits_paper_test_seed42` (616, 28). My first seed 1 and 42 paper-set runs used the default split (leaky); they were discarded and re-run with the per-seed split before any number here was produced from them. Small and Tiny seed 0 use the default split.

Calibration. The constant c is calibrated on each checkpoint's own paper set (cross-set), evaluated on the stress sets; on the paper set itself it is 5-fold CV grouped by load scenario. The max-rule constant moves a lot with the calibration sample (heavy tail over load scenarios), which is the main source of the SD in the LU columns:

| prior | rule | base seed 0 | base seed 1 | base seed 42 | small seed 0 | tiny seed 0 |
|---|---|---|---|---|---|---|
| GENCO | max | 0.3465 | 1.8678 | 0.2249 | 0.1952 | 0.7554 |
| GENCO | conf0.10 | 0.1257 | 0.139 | 0.1056 | 0.1572 | 0.1527 |
| GENCO | conf0.05 | 0.3465 | 1.8678 | 0.2249 | 0.1952 | 0.7554 |
| base case (error vs own NR) | max | 42.0956 | 37.6895 | 37.1554 | 42.0956 | 42.0956 |
| base case (error vs own NR) | conf0.10 | 26.2307 | 20.061 | 26.5919 | 26.2307 | 26.2307 |
| base case (error vs own NR) | conf0.05 | 42.0956 | 37.6895 | 37.1554 | 42.0956 | 42.0956 |

Cost in LU per contingency, tier-1 share and element misses (policies; base-case rows use c measured against our own NR, the stronger baseline). Zero element misses, zero contingency misses in every cell:

| set | policy | rule | LU Base (3 seeds) | LU Small | LU Tiny | tier-1 Base | misses Base/Small/Tiny |
|---|---|---|---|---|---|---|---|
| paper N-1 | Base case sequential | conf0.10 | 2.176 ± 0.026 | 2.165 | 2.165 | 0.13 ± 0.02 | 0/0/0 |
| stress N-1 | Base case sequential | conf0.10 | 2.134 ± 0.016 | 2.141 | 2.141 | 0.13 ± 0.00 | 0/0/0 |
| stress N-2 | Base case sequential | conf0.10 | 2.463 ± 0.025 | 2.476 | 2.476 | 0.02 ± 0.00 | 0/0/0 |
| stress N-3 | Base case sequential | conf0.10 | 2.673 ± 0.016 | 2.682 | 2.682 | 0.00 ± 0.00 | 0/0/0 |
| stress N-5 | Base case sequential | conf0.10 | 2.945 ± 0.014 | 2.952 | 2.952 | 0.00 ± 0.00 | 0/0/0 |
| paper N-1 | GENCO sequential | conf0.10 | 1.239 ± 0.080 | 1.384 | 1.461 | 0.76 ± 0.08 | 0/0/0 |
| stress N-1 | GENCO sequential | conf0.10 | 1.260 ± 0.039 | 1.359 | 1.374 | 0.74 ± 0.04 | 0/0/0 |
| stress N-2 | GENCO sequential | conf0.10 | 1.330 ± 0.036 | 1.417 | 1.454 | 0.67 ± 0.04 | 0/0/0 |
| stress N-3 | GENCO sequential | conf0.10 | 1.431 ± 0.084 | 1.479 | 1.600 | 0.57 ± 0.08 | 0/0/0 |
| stress N-5 | GENCO sequential | conf0.10 | 1.615 ± 0.077 | 1.615 | 1.736 | 0.39 ± 0.07 | 0/0/0 |
| paper N-1 | GENCO tiered | conf0.10 | 1.373 ± 0.166 | 1.588 | 1.611 | 0.76 ± 0.08 | 0/0/0 |
| stress N-1 | GENCO tiered | conf0.10 | 1.379 ± 0.097 | 1.456 | 1.536 | 0.74 ± 0.04 | 0/0/0 |
| stress N-2 | GENCO tiered | conf0.10 | 1.511 ± 0.097 | 1.590 | 1.701 | 0.67 ± 0.04 | 0/0/0 |
| stress N-3 | GENCO tiered | conf0.10 | 1.698 ± 0.203 | 1.762 | 1.976 | 0.57 ± 0.08 | 0/0/0 |
| stress N-5 | GENCO tiered | conf0.10 | 2.064 ± 0.195 | 2.024 | 2.265 | 0.39 ± 0.07 | 0/0/0 |
| paper N-1 | Base case sequential | max | 2.258 ± 0.042 | 2.210 | 2.210 | 0.09 ± 0.02 | 0/0/0 |
| stress N-1 | Base case sequential | max | 2.206 ± 0.012 | 2.220 | 2.220 | 0.10 ± 0.00 | 0/0/0 |
| stress N-2 | Base case sequential | max | 2.533 ± 0.004 | 2.537 | 2.537 | 0.01 ± 0.00 | 0/0/0 |
| stress N-3 | Base case sequential | max | 2.747 ± 0.007 | 2.755 | 2.755 | 0.00 ± 0.00 | 0/0/0 |
| stress N-5 | Base case sequential | max | 2.995 ± 0.007 | 3.003 | 3.003 | 0.00 ± 0.00 | 0/0/0 |
| paper N-1 | GENCO sequential | max | 1.558 ± 0.167 | 1.406 | 1.743 | 0.44 ± 0.17 | 0/0/0 |
| stress N-1 | GENCO sequential | max | 1.611 ± 0.250 | 1.424 | 1.756 | 0.39 ± 0.25 | 0/0/0 |
| stress N-2 | GENCO sequential | max | 1.602 ± 0.157 | 1.470 | 1.804 | 0.40 ± 0.16 | 0/0/0 |
| stress N-3 | GENCO sequential | max | 1.707 ± 0.135 | 1.530 | 1.889 | 0.29 ± 0.13 | 0/0/0 |
| stress N-5 | GENCO sequential | max | 1.847 ± 0.093 | 1.648 | 1.944 | 0.17 ± 0.09 | 0/0/0 |
| paper N-1 | GENCO tiered | max | 1.704 ± 0.138 | 1.610 | 1.915 | 0.44 ± 0.17 | 0/0/0 |
| stress N-1 | GENCO tiered | max | 1.748 ± 0.220 | 1.533 | 1.953 | 0.39 ± 0.25 | 0/0/0 |
| stress N-2 | GENCO tiered | max | 1.798 ± 0.177 | 1.643 | 2.080 | 0.40 ± 0.16 | 0/0/0 |
| stress N-3 | GENCO tiered | max | 1.995 ± 0.189 | 1.818 | 2.279 | 0.29 ± 0.13 | 0/0/0 |
| stress N-5 | GENCO tiered | max | 2.319 ± 0.175 | 2.064 | 2.500 | 0.17 ± 0.09 | 0/0/0 |

Equal-cost ladder, element misses of violating elements (Base mean ± SD over seeds / Small / Tiny):

| set | GENCO + 1 step | GENCO + 2 steps | Base case + 1 step | Base case + 2 steps | violating elements (Base mean / Small / Tiny) |
|---|---|---|---|---|---|
| paper N-1 | 1.3 ± 1.5 / 0 / 0 | 0.0 ± 0.0 / 0 / 0 | 41.0 ± 8.0 / 33 / 33 | 0.0 ± 0.0 / 0 / 0 | 567 / 484 / 484 |
| stress N-1 | 1.3 ± 0.6 / 2 / 1 | 0.0 ± 0.0 / 0 / 0 | 21.0 ± 0.0 / 21 / 21 | 1.0 ± 0.0 / 1 / 1 | 435 / 435 / 435 |
| stress N-2 | 0.7 ± 1.2 / 0 / 0 | 0.0 ± 0.0 / 0 / 0 | 37.0 ± 0.0 / 37 / 37 | 0.0 ± 0.0 / 0 / 0 | 714 / 714 / 714 |
| stress N-3 | 1.3 ± 0.6 / 1 / 2 | 0.0 ± 0.0 / 0 / 0 | 60.0 ± 0.0 / 60 / 60 | 2.0 ± 0.0 / 2 / 2 | 950 / 950 / 950 |
| stress N-5 | 6.3 ± 2.5 / 7 / 4 | 0.0 ± 0.0 / 0 / 0 | 129.0 ± 0.0 / 129 / 129 | 6.0 ± 0.0 / 6 / 6 | 1763 / 1763 / 1763 |

GENCO + 1 step (one LU) misses as few elements as base case + 2 steps (two LU) for all five checkpoints, within a handful of elements (seed 42 is the weakest, 17 and one contingency against 9 and none for base + 2, pooled over the five sets). Newton-Raphson iterations by tolerance (mean ± SD over the 3 Base seeds for flat, base, GENCO; GENCO for Small and Tiny. The flat and base-case starts do not depend on the checkpoint; their SD on the paper row comes from the three different test sets):

| set | tol (MW) | flat | base | genco | GENCO Small | GENCO Tiny |
|---|---|---|---|---|---|---|
| paper N-1 | 0.01 | 3.01 ± 0.00 | 2.07 ± 0.03 | 1.74 ± 0.14 | 1.68 | 1.83 |
| paper N-1 | 0.0001 | 4.00 ± 0.00 | 2.52 ± 0.02 | 2.01 ± 0.01 | 2.00 | 2.01 |
| paper N-1 | 1e-06 | 4.00 ± 0.00 | 2.91 ± 0.03 | 2.15 ± 0.09 | 2.22 | 2.18 |
| stress N-1 | 0.01 | 3.01 ± 0.00 | 2.03 ± 0.00 | 1.79 ± 0.09 | 1.76 | 1.77 |
| stress N-1 | 0.0001 | 4.00 ± 0.00 | 2.50 ± 0.00 | 2.00 ± 0.01 | 2.00 | 2.01 |
| stress N-1 | 1e-06 | 4.00 ± 0.00 | 2.88 ± 0.00 | 2.15 ± 0.08 | 2.13 | 2.20 |
| stress N-2 | 0.01 | 3.03 ± 0.00 | 2.19 ± 0.00 | 1.68 ± 0.12 | 1.71 | 1.81 |
| stress N-2 | 0.0001 | 4.00 ± 0.00 | 2.74 ± 0.00 | 2.02 ± 0.02 | 2.01 | 2.02 |
| stress N-2 | 1e-06 | 4.01 ± 0.00 | 3.08 ± 0.00 | 2.21 ± 0.08 | 2.18 | 2.29 |
| stress N-3 | 0.01 | 3.03 ± 0.00 | 2.29 ± 0.00 | 1.81 ± 0.10 | 1.78 | 1.89 |
| stress N-3 | 0.0001 | 4.00 ± 0.00 | 2.90 ± 0.00 | 2.03 ± 0.02 | 2.02 | 2.03 |
| stress N-3 | 1e-06 | 4.01 ± 0.00 | 3.17 ± 0.00 | 2.30 ± 0.14 | 2.30 | 2.39 |
| stress N-5 | 0.01 | 3.08 ± 0.00 | 2.49 ± 0.00 | 1.92 ± 0.06 | 1.89 | 1.96 |
| stress N-5 | 0.0001 | 4.01 ± 0.00 | 3.08 ± 0.00 | 2.11 ± 0.05 | 2.07 | 2.10 |
| stress N-5 | 1e-06 | 4.03 ± 0.00 | 3.32 ± 0.00 | 2.50 ± 0.14 | 2.47 | 2.58 |

Full tables, with the 'dc' start, the conformal 0.05 rule and the GENCO-minus-base paired differences: `results/policy_v3_ci.log` (sections A to E) and `results/exp02/v3_seeds_*.csv`.

## 6. Intervals and miss-rate bounds (WS2.2)

Cluster bootstrap over load scenarios (2,000 resamples) for LU, tier-1 share, recall, miss rate and NR iterations; Clopper-Pearson bounds at element level, contingency level and load-scenario level (fraction of load scenarios that contain a miss). The load-scenario bound does not need independence between the contingencies of one load scenario and is the one to quote. Base seed 0, c from its paper set, 5-fold CV on the paper set:

| set | policy | rule | c | LU [95% CI] | tier-1 [95% CI] | el misses | el 95% UB | cont misses | cont 95% UB | load scen. with miss | load 95% UB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| paper N-1 | Base seq | conf0.10 | 29.5 | 2.165 [2.105, 2.219] | 0.13 [0.10, 0.16] | 0/484 | 0.76% | 0/225 | 1.63% | 0/29 | 11.9% |
| stress N-1 | Base seq | conf0.10 | 26.2 | 2.141 [2.074, 2.208] | 0.13 [0.10, 0.17] | 0/435 | 0.84% | 0/217 | 1.69% | 0/30 | 11.6% |
| stress N-2 | Base seq | conf0.10 | 26.2 | 2.476 [2.426, 2.527] | 0.02 [0.01, 0.03] | 0/714 | 0.52% | 0/334 | 1.10% | 0/30 | 11.6% |
| stress N-3 | Base seq | conf0.10 | 26.2 | 2.682 [2.629, 2.732] | 0.00 [0.00, 0.00] | 0/950 | 0.39% | 0/455 | 0.81% | 0/30 | 11.6% |
| stress N-5 | Base seq | conf0.10 | 26.2 | 2.952 [2.912, 2.992] | 0.00 [0.00, 0.00] | 0/1763 | 0.21% | 0/539 | 0.68% | 0/30 | 11.6% |
| paper N-1 | GENCO seq | conf0.10 | 0.167 | 1.215 [1.113, 1.339] | 0.79 [0.66, 0.89] | 0/484 | 0.76% | 0/225 | 1.63% | 0/29 | 11.9% |
| stress N-1 | GENCO seq | conf0.10 | 0.126 | 1.224 [1.126, 1.335] | 0.78 [0.67, 0.87] | 0/435 | 0.84% | 0/217 | 1.69% | 0/30 | 11.6% |
| stress N-2 | GENCO seq | conf0.10 | 0.126 | 1.311 [1.200, 1.434] | 0.69 [0.57, 0.80] | 0/714 | 0.52% | 0/334 | 1.10% | 0/30 | 11.6% |
| stress N-3 | GENCO seq | conf0.10 | 0.126 | 1.365 [1.276, 1.462] | 0.63 [0.54, 0.72] | 0/950 | 0.39% | 0/455 | 0.81% | 0/30 | 11.6% |
| stress N-5 | GENCO seq | conf0.10 | 0.126 | 1.559 [1.488, 1.630] | 0.45 [0.38, 0.52] | 0/1763 | 0.21% | 0/539 | 0.68% | 0/30 | 11.6% |
| paper N-1 | Base seq | max | 42.1 | 2.210 [2.149, 2.265] | 0.11 [0.08, 0.14] | 0/484 | 0.76% | 0/225 | 1.63% | 0/29 | 11.9% |
| stress N-1 | Base seq | max | 42.1 | 2.220 [2.158, 2.282] | 0.10 [0.07, 0.13] | 0/435 | 0.84% | 0/217 | 1.69% | 0/30 | 11.6% |
| stress N-2 | Base seq | max | 42.1 | 2.537 [2.483, 2.592] | 0.01 [0.01, 0.02] | 0/714 | 0.52% | 0/334 | 1.10% | 0/30 | 11.6% |
| stress N-3 | Base seq | max | 42.1 | 2.755 [2.705, 2.803] | 0.00 [0.00, 0.00] | 0/950 | 0.39% | 0/455 | 0.81% | 0/30 | 11.6% |
| stress N-5 | Base seq | max | 42.1 | 3.003 [2.967, 3.039] | 0.00 [0.00, 0.00] | 0/1763 | 0.21% | 0/539 | 0.68% | 0/30 | 11.6% |
| paper N-1 | GENCO seq | max | 0.347 | 1.403 [1.262, 1.555] | 0.60 [0.45, 0.74] | 0/484 | 0.76% | 0/225 | 1.63% | 0/29 | 11.9% |
| stress N-1 | GENCO seq | max | 0.347 | 1.477 [1.353, 1.608] | 0.52 [0.39, 0.65] | 0/435 | 0.84% | 0/217 | 1.69% | 0/30 | 11.6% |
| stress N-2 | GENCO seq | max | 0.347 | 1.487 [1.366, 1.614] | 0.52 [0.39, 0.64] | 0/714 | 0.52% | 0/334 | 1.10% | 0/30 | 11.6% |
| stress N-3 | GENCO seq | max | 0.347 | 1.588 [1.486, 1.691] | 0.41 [0.31, 0.51] | 0/950 | 0.39% | 0/455 | 0.81% | 0/30 | 11.6% |
| stress N-5 | GENCO seq | max | 0.347 | 1.755 [1.683, 1.821] | 0.26 [0.19, 0.33] | 0/1763 | 0.21% | 0/539 | 0.68% | 0/30 | 11.6% |

Equal-cost ladder, Base seed 0 (the 'UB' columns are the Clopper-Pearson two-sided 95% upper limits):

| set | method | recall [95% CI] | el misses | el 95% UB | cont misses | cont 95% UB | load scen. with miss | load 95% UB |
|---|---|---|---|---|---|---|---|---|
| paper N-1 | Base case + 1 step | 0.9318 [0.9085, 0.9532] | 33/484 | 9.44% | 4/225 | 4.49% | 21/29 | 87.3% |
| stress N-1 | Base case + 1 step | 0.9517 [0.9232, 0.9744] | 21/435 | 7.29% | 3/217 | 3.99% | 13/30 | 62.6% |
| stress N-2 | Base case + 1 step | 0.9482 [0.9234, 0.9714] | 37/714 | 7.07% | 3/334 | 2.60% | 15/30 | 68.7% |
| stress N-3 | Base case + 1 step | 0.9368 [0.9206, 0.9521] | 60/950 | 8.06% | 6/455 | 2.85% | 24/30 | 92.3% |
| stress N-5 | Base case + 1 step | 0.9268 [0.9087, 0.9438] | 129/1763 | 8.63% | 5/539 | 2.15% | 26/30 | 96.2% |
| paper N-1 | Base case + 2 steps | 1.0000 [1.0000, 1.0000] | 0/484 | 0.76% | 0/225 | 1.63% | 0/29 | 11.9% |
| stress N-1 | Base case + 2 steps | 0.9977 [0.9926, 1.0000] | 1/435 | 1.27% | 0/217 | 1.69% | 1/30 | 17.2% |
| stress N-2 | Base case + 2 steps | 1.0000 [1.0000, 1.0000] | 0/714 | 0.52% | 0/334 | 1.10% | 0/30 | 11.6% |
| stress N-3 | Base case + 2 steps | 0.9979 [0.9946, 1.0000] | 2/950 | 0.76% | 0/455 | 0.81% | 2/30 | 22.1% |
| stress N-5 | Base case + 2 steps | 0.9966 [0.9914, 1.0000] | 6/1763 | 0.74% | 0/539 | 0.68% | 2/30 | 22.1% |
| paper N-1 | GENCO + 1 step | 0.9979 [0.9932, 1.0000] | 1/484 | 1.15% | 0/225 | 1.63% | 1/29 | 17.8% |
| stress N-1 | GENCO + 1 step | 0.9977 [0.9929, 1.0000] | 1/435 | 1.27% | 0/217 | 1.69% | 1/30 | 17.2% |
| stress N-2 | GENCO + 1 step | 1.0000 [1.0000, 1.0000] | 0/714 | 0.52% | 0/334 | 1.10% | 0/30 | 11.6% |
| stress N-3 | GENCO + 1 step | 0.9989 [0.9968, 1.0000] | 1/950 | 0.59% | 0/455 | 0.81% | 1/30 | 17.2% |
| stress N-5 | GENCO + 1 step | 0.9966 [0.9926, 1.0000] | 6/1763 | 0.74% | 0/539 | 0.68% | 3/30 | 26.5% |
| paper N-1 | GENCO + 2 steps | 1.0000 [1.0000, 1.0000] | 0/484 | 0.76% | 0/225 | 1.63% | 0/29 | 11.9% |
| stress N-1 | GENCO + 2 steps | 1.0000 [1.0000, 1.0000] | 0/435 | 0.84% | 0/217 | 1.69% | 0/30 | 11.6% |
| stress N-2 | GENCO + 2 steps | 1.0000 [1.0000, 1.0000] | 0/714 | 0.52% | 0/334 | 1.10% | 0/30 | 11.6% |
| stress N-3 | GENCO + 2 steps | 1.0000 [1.0000, 1.0000] | 0/950 | 0.39% | 0/455 | 0.81% | 0/30 | 11.6% |
| stress N-5 | GENCO + 2 steps | 1.0000 [1.0000, 1.0000] | 0/1763 | 0.21% | 0/539 | 0.68% | 0/30 | 11.6% |

Pooled over the five test sets of each checkpoint (one-sided 95% upper bounds). The sets use different load scenarios, so the load-scenario denominator is 148 to 157. For comparison the base-case ladder is included:

| checkpoint | method | el misses | el 95% UB | cont misses | cont 95% UB | load scen. with miss | load 95% UB |
|---|---|---|---|---|---|---|---|
| base seed 0 | GENCO sequential, max rule | 0/4346 | 0.07% | 0/1770 | 0.17% | 0/149 | 2.0% |
| base seed 1 | GENCO sequential, max rule | 0/4542 | 0.07% | 0/1840 | 0.16% | 0/157 | 1.9% |
| base seed 42 | GENCO sequential, max rule | 0/4398 | 0.07% | 0/1774 | 0.17% | 0/148 | 2.0% |
| small seed 0 | GENCO sequential, max rule | 0/4346 | 0.07% | 0/1770 | 0.17% | 0/149 | 2.0% |
| tiny seed 0 | GENCO sequential, max rule | 0/4346 | 0.07% | 0/1770 | 0.17% | 0/149 | 2.0% |
| base seed 0 | GENCO sequential, conformal 0.10 | 0/4346 | 0.07% | 0/1770 | 0.17% | 0/149 | 2.0% |
| base seed 1 | GENCO sequential, conformal 0.10 | 0/4542 | 0.07% | 0/1840 | 0.16% | 0/157 | 1.9% |
| base seed 42 | GENCO sequential, conformal 0.10 | 0/4398 | 0.07% | 0/1774 | 0.17% | 0/148 | 2.0% |
| small seed 0 | GENCO sequential, conformal 0.10 | 0/4346 | 0.07% | 0/1770 | 0.17% | 0/149 | 2.0% |
| tiny seed 0 | GENCO sequential, conformal 0.10 | 0/4346 | 0.07% | 0/1770 | 0.17% | 0/149 | 2.0% |
| base seed 0 | GENCO + 1 step | 9/4346 | 0.36% | 0/1770 | 0.17% | 6/149 | 7.8% |
| base seed 1 | GENCO + 1 step | 7/4542 | 0.29% | 0/1840 | 0.16% | 6/157 | 7.4% |
| base seed 42 | GENCO + 1 step | 17/4398 | 0.58% | 1/1774 | 0.27% | 10/148 | 11.2% |
| small seed 0 | GENCO + 1 step | 10/4346 | 0.39% | 0/1770 | 0.17% | 6/149 | 7.8% |
| tiny seed 0 | GENCO + 1 step | 7/4346 | 0.30% | 0/1770 | 0.17% | 6/149 | 7.8% |
| base seed 0 | Base case + 1 step | 280/4346 | 7.09% | 21/1770 | 1.70% | 99/149 | 72.9% |
| base seed 1 | Base case + 1 step | 296/4542 | 7.15% | 23/1840 | 1.77% | 103/157 | 71.9% |
| base seed 42 | Base case + 1 step | 288/4398 | 7.20% | 22/1774 | 1.77% | 95/148 | 70.7% |
| small seed 0 | Base case + 1 step | 280/4346 | 7.09% | 21/1770 | 1.70% | 99/149 | 72.9% |
| tiny seed 0 | Base case + 1 step | 280/4346 | 7.09% | 21/1770 | 1.70% | 99/149 | 72.9% |
| base seed 0 | Base case + 2 steps | 9/4346 | 0.36% | 0/1770 | 0.17% | 5/149 | 6.9% |
| base seed 1 | Base case + 2 steps | 9/4542 | 0.35% | 0/1840 | 0.16% | 5/157 | 6.6% |
| base seed 42 | Base case + 2 steps | 9/4398 | 0.36% | 0/1774 | 0.17% | 5/148 | 7.0% |
| small seed 0 | Base case + 2 steps | 9/4346 | 0.36% | 0/1770 | 0.17% | 5/149 | 6.9% |
| tiny seed 0 | Base case + 2 steps | 9/4346 | 0.36% | 0/1770 | 0.17% | 5/149 | 6.9% |
| base seed 0 | GENCO + 2 steps | 0/4346 | 0.07% | 0/1770 | 0.17% | 0/149 | 2.0% |
| base seed 1 | GENCO + 2 steps | 0/4542 | 0.07% | 0/1840 | 0.16% | 0/157 | 1.9% |
| base seed 42 | GENCO + 2 steps | 0/4398 | 0.07% | 0/1774 | 0.17% | 0/148 | 2.0% |
| small seed 0 | GENCO + 2 steps | 0/4346 | 0.07% | 0/1770 | 0.17% | 0/149 | 2.0% |
| tiny seed 0 | GENCO + 2 steps | 0/4346 | 0.07% | 0/1770 | 0.17% | 0/149 | 2.0% |

With 29 to 37 load scenarios per set the load-scenario bound for "zero misses" is 9.5% one-sided (11.6% two-sided); pooled over the five sets it is 1.9 to 2.0%. Stress sets are shared by all checkpoints, so the five checkpoints are not independent evidence. Tighter bounds need more load scenarios: see section 8.

## 7. Data and split choices per run

| run | data | split | scenarios / load scenarios |
|---|---|---|---|
| Base seed 0 paper | `data/pf/case118_ieee` | `splits_paper_test` | 638 / 29 |
| Base seed 1 paper | `data/pf/case118_ieee` | `splits_paper_test_seed1` | 814 / 37 |
| Base seed 42 paper | `data/pf/case118_ieee` | `splits_paper_test_seed42` | 616 / 28 |
| Small, Tiny seed 0 paper | `data/pf/case118_ieee` | `splits_paper_test` | 638 / 29 |
| all stress N-1/2/3/5 | `data/stress/n{1,2,3,5}` | `splits_all` | 660, 659, 660, 660 / 30 each |
| exp07 (Base seed 0 only) | paper set, stress, `data/exhaustive/n1` and `n1_gen` | as above, `splits_all` | 5,340 + 1,620 (30 load scenarios; N-0 dropped) |

Tags: `case118_ieee_base_seed{0,1,42}`, `case118_ieee_{small,tiny}_seed0`, `stress_n{k}_base[_seed{1,42}]`, `stress_n{k}_{small,tiny}`.

## 8. Caveats and open items

- Timings are provisional (shared machine, section 4) and for python/scipy code; GENCO inference is the earlier 4-thread CPU/MPS measurement, not a re-measurement and not GPU. The break-even tables give the number to beat; a Colab GPU run only needs `ms per scenario` to compare against them.
- Base-case solutions are assumed free and exact (reconstructed from all variants of a load scenario, as in exp02). In an operator's tool the base case is a state-estimator or previous-case solution.
- The factor-reuse baselines are mine and at 118 buses. Plain chord is a weak baseline for branch outages (see above); chordC and FDLF are the ones to quote. FDLF with B', B'' from the base topology and Woodbury compensation is a standard construction, but production codes add refinements (limits, tap adjustment) that are out of scope.
- The 'model' basis is an idealisation; it is there to show that python overhead does not decide the answer, not as a prediction.
- Cluster-level bounds are weak with 30 load scenarios per set. New larger sets exist and are the intended next evaluation sets: `data/calib/n1` (2,200 N-1 scenarios, 100 load scenarios; for conformal calibration, WS2.3), `data/stress100/n{1,2,3,5}` (2,200 scenarios, 100 load scenarios each; WS2.5), and the exhaustive sets (exp07 already runs on them; the exp02 screening analysis does not yet). Running `exp02_screening.py --data_root data/stress100/nK --split splits_all` and adding the tags to `CI_SETS` is the only change needed.
- exp07 was run for Base seed 0 only; seeds 1, 42 and Small/Tiny would change the GENCO-start columns slightly (their NR iteration counts at 1e-4 differ by less than 0.06, section 5).
- The legacy table and `make_figures.py` are unchanged. `scripts/run_all.sh` does not yet call exp07 or `--ci`.

## 9. Reproduce

```
python scripts/exp02_screening.py --grid case118_ieee --size base --seed 1  --split splits_paper_test_seed1
python scripts/exp02_screening.py --grid case118_ieee --size base --seed 42 --split splits_paper_test_seed42
python scripts/exp02_screening.py --grid case118_ieee --size base --seed S --data_root data/stress/nK --split splits_all --tag stress_nK_base_seedS   # S in 1, 42; K in 1 2 3 5
python scripts/exp02_screening.py --grid case118_ieee --size small|tiny [--data_root ... --tag stress_nK_small|tiny]
python scripts/analyze_exp02.py --ci            # -> results/policy_v3_ci.log, results/exp02/v3_ci_*.csv, v3_seeds_*.csv
python scripts/exp07_industrial_baselines.py --grid case118_ieee --size base [--data_root data/stress/nK --split splits_all --tag stress_nK_base]
python scripts/exp07_industrial_baselines.py --timing --repeats 5      # run on an idle machine; paper set
python scripts/exp07_industrial_baselines.py --summarize               # -> results/exp07/*.csv, exp07.log
python -m pytest -q tests
```
