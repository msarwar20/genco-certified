# Confirmatory test on fresh IEEE 118 data (agent Z)

Protocol `results/confirm/PROTOCOL.md`, written 2026-10-25 00:15 EDT before any data existed; sha256 of its text
`027cf172823fdb3e3b0d8eac592d3b80018cbb50b558352b04d0a1803551aa42`. Rule frozen as in NUMBERS 6.2 (calib/n1, tau 1.01, eps_v 1 mV, <=3 steps; exact c in the protocol). The run reproduced all six constants. Output: `results/confirm/policy_confirm.log`, `confirm_*.csv`.

**Data.** Seeds 5101/5102/5103/5105 (never used), 100 load scenarios x 22 variants, N-1/2/3/5, `data/confirm/n{1,2,3,5}`. Load profile ERCOT 2024 East (first 100 h) instead of the default profile: total demand correlates -0.44 to -0.50 with calib/stress100 per index, mean 4,193 MW vs 3,960 MW. Scenarios 2200/2200/2200/2196; violating scenarios 34.9/56.5/71.6/85.9%; max loading/rate 3.9-5.3. Splits: `splits_all`.

**Safety** (0 element misses, 0 element false alarms, 0 contingency misses, 0 contingency false alarms in all 36 cells: 3 rules x 3 policies x 4 sets)

| Set | viol. elem / cont | GENCO seq LU (conf0.10) | base seq LU | tier-1 G / B | exceed. conf0.10 (G) | NR 1e-6 base / GENCO / diff |
|---|---|---|---|---|---|---|
| C N-1 | 1775 / 820 | 1.378 [1.307, 1.455] | 2.187 [2.156, 2.219] | 0.62 / 0.12 | 5% | 2.92 / 2.10 / -0.83 [-0.87, -0.78] |
| C N-2 | 3269 / 1311 | 1.424 [1.366, 1.482] | 2.534 [2.505, 2.564] | 0.58 / 0.01 | 6% | 3.12 / 2.14 / -0.98 [-1.01, -0.94] |
| C N-3 | 4860 / 1640 | 1.565 [1.511, 1.618] | 2.767 [2.740, 2.793] | 0.44 / 0.00 | 10% | 3.22 / 2.27 / -0.95 [-1.00, -0.90] |
| C N-5 | 7191 / 1942 | 1.706 [1.661, 1.752] | 2.981 [2.960, 3.003] | 0.31 / 0.00 | 6% | 3.36 / 2.42 / -0.94 [-0.98, -0.90] |

GENCO seq LU, max rule: 1.81-1.97; conf0.05: 1.47-1.79. GENCO-minus-base at 1e-2 / 1e-4: -0.47/-0.54 (N-1) to -0.65/-1.03 (N-5); all CIs exclude 0.

**Pooled misses (Clopper-Pearson 95%, one- / two-sided upper limit):** elements 0/17,095 (0.018% / 0.022%); contingencies 0/5,713 (0.052% / 0.065%); load scenarios 0/400 (0.75% / 0.92%; the four sets share one load trajectory, so this is not 400 independent draws).

**Exceedance** (share of load scenarios with step-1 score above c, GENCO): conf0.10 5/6/10/6%, conf0.05 3/2/4/3%, max 0/0/0/1% (N-5, one load scenario scored 2.203 > 2.178 without a miss). Base prior: 0-5%. All at or below nominal.

**Deviations.** (1) Standalone `results/confirm/make_confirm_splits.py` instead of `make_splits.py` (not edited; same convention). (2) `num_processes` 4 instead of 6. (3) Profile chosen from its statistics alone. No reruns. Zero misses is empirical, not a proof.
