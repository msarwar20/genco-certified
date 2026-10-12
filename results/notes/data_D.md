# Data workstream D: calibration, bigger stress, exhaustive N-1, GOC 500 / 2000

Written 2026-10-11. Covers PLAN WS2.3, WS2.5, WS3.1, WS3.2. Everything below was generated or checked in this
session; numbers come from the raw parquet unless stated.

## 1. New IEEE 118 datakit sets

All configs are copies of `configs/datakit/case118_n2.yaml` with only these fields changed: `load.scenarios`,
`settings.data_dir`, `settings.seed`, `topology_perturbation.{k, outage_count_probabilities}` (and, for the exhaustive
set, `type`). Generator/admittance perturbation, load profile, `n_topology_variants: 22`, `elements: [branch, gen]`,
`num_processes: 6`, `pf_fast` etc. are untouched. Each set lives in `<data_dir>/case118_ieee/{raw,processed,splits_all}`.

"Violating scenario" = at least one in-service branch with max(|S_f|, |S_t|)/rate_a > 1.01, computed from raw
pf/qf/pt/qt/rate_a (parallel circuits are tested individually, so the flag equals the merged-by-max definition).
N-0 (base-topology) scenarios have 0 violations in every set that contains them.

| set (data_dir) | config | seed | load scen. | scenarios | dropped (PF did not converge) | violating scen. | max load /rate | raw / total MB | wall |
|---|---|---|---|---|---|---|---|---|---|
| `data/calib/n1` (calibration) | `case118_calib_n1.yaml` | 3101 | 100 | 2200 (22/ls) | 0 | 693 (31.5%) | 3.00 | 29 / 141 | 180 s |
| `data/stress100/n1` | `case118_stress100_n1.yaml` | 3111 | 100 | 2200 | 0 | 727 (33.0%) | 3.09 | 29 / 141 | 129 s |
| `data/stress100/n2` | `case118_stress100_n2.yaml` | 3102 | 100 | 2199 | 1 | 1227 (55.8%) | 3.53 | 32 / 144 | 180 s |
| `data/stress100/n3` | `case118_stress100_n3.yaml` | 3103 | 100 | 2199 | 1 | 1494 (67.9%) | 4.23 | 33 / 145 | 75 s |
| `data/stress100/n5` | `case118_stress100_n5.yaml` | 3105 | 100 | 2194 | 6 | 1855 (84.5%) | 6.22 | 35 / 146 | 143 s |
| `data/exhaustive/n1` (all branches) | `case118_exhaustive_n1.yaml` | 3201 | 30 | 5340 (178/ls) | 9 branches islanding, 0 PF failures | 1735 of 5310 outage scen. (32.7%) | 3.88 | 66 / 337 | 133 s |
| `data/exhaustive/n1_gen` (all generators) | `case118_exhaustive_n1_gen.yaml` + `data/exhaustive/run_exhaustive_gen.py` | 3201 | 30 | 1620 (54/ls) | 0 | 312 of 1590 outage scen. (19.6%) | 2.76 | 17 / 100 | 143 s |

Wall times are with two datakit jobs (6 workers each) running side by side plus other agents' CPU jobs (load average
up to ~35), so they are upper bounds. Total new IEEE-118 data: about 1.2 GB incl. processed caches. For comparison the
old 30-load-scenario N-1 set (`data/stress/n1`, 660 scenarios) has 190 violating scenarios (28.8%).

Checks done for every set above:
- scenario ids are exactly 0..N-1 (asserted in `scripts/make_splits.py::scenario_table`; datakit renumbers after dropping
  non-converged variants, so ids stay contiguous);
- `splits_all` written: test = all ids, train/val = first 2 / next 2;
- outage count per scenario equals k in every random set (histograms are single-valued);
- one batch through `load_genco` + GENCO forward + `acpf.extract_problems` + `acpf.mismatch` / `acpf.newton_pf`, all
  finite, NR converged (Base size; see section 5).

Dropped scenarios in the random sets are variants whose AC power flow did not converge (see each
`raw/error.log`); a few load scenarios therefore have 21 instead of 22 variants.

Other properties of the random N-1 sets (calib, stress100/n1): outage draws are with replacement within a load scenario
(22 draws from 230 feasible single outages), so about 87 of 2200 rows (4%) are exact duplicate (load scenario, outage)
samples. Roughly 23% of the N-1 rows are generator outages. N-2 has 1 duplicate, N-3 and N-5 none.

### 1.1 Exhaustive N-1 semantics (read from `NMinusKGenerator`)

- `NMinusKGenerator(k, net)` enumerates `combinations(idx_branches_in_service, c)` for c = 0..k, i.e. for k = 1 the **N-0
  base topology first, then every in-service branch** (lines and transformers): 1 + 186 = 187 candidate topologies per
  load scenario ("Number of possible topologies ... 187" in the log).
- It is **branches only**. `topology_perturbation.elements` is ignored (datakit warns that `elements`/`n_topology_variants`
  are unused). Generator outages are not enumerated by `n_minus_k`. The config keeps `elements: [branch, gen]` as asked,
  with a comment saying it has no effect.
- Islanding outages are dropped silently (`check_single_connected_component`). 9 of the 186 branches island the grid
  (0-indexed branch idx 6, 8, 112, 132, 133, 175, 176, 182, 183 = from/to buses 7-8, 8-9, 70-72, 84-85, 85-86, 109-110,
  109-111, 67-115, 11-116 as stored in `branch_data`, 0-indexed; add 1 for pglib numbering; all are radial feeders), so **178 scenarios per load scenario = 1 N-0 + 177 N-1**, 30 load scenarios =
  5340 scenarios, 270 islanding variants dropped (9 x 30). No AC power-flow failures (empty `error.log`). The same 9
  branches are dropped for every load scenario.
- The operating point (OPF dispatch after cost permutation / admittance perturbation) is computed once per load scenario
  and shared by all its topology variants (same as in the random sets), so setpoints are not re-dispatched after the outage.
- Generators: added a companion set `data/exhaustive/n1_gen` using a drop-in `TopologyGenerator`
  (`data/exhaustive/run_exhaustive_gen.py`, config type `gen_n_minus_1`, not a stock datakit type). It yields the N-0
  topology plus one topology per in-service generator not at the reference bus: 1 + 53 = 54 per load scenario, 1620
  scenarios, nothing dropped (generator outages never island; PV buses that lose their only generator become PQ via
  `deactivate_gens`). Same seed 3201, same 30 load scenarios, and the N-0 states match the branch set to the last digit
  (checked: max |dPg|, |dPd|, |dVm| = 0), so the two sets pair by `load_scenario_idx`. Run it with
  `python data/exhaustive/run_exhaustive_gen.py configs/datakit/case118_exhaustive_n1_gen.yaml` (activated venv).
  Together: 5340 + 1620 scenarios, with N-0 duplicated; "every branch and generator per operating point" is `n1` + `n1_gen`.
  The reference-bus generator is not dropped (same convention as the random generator).

## 2. Caveat that matters for calibration (WS2.3)

datakit's `agg_load_profile` generator **cuts the first n hours of the aggregate profile** when `load.scenarios` <= 8760
and uses the seed only for the per-bus +-20% uniform noise (and for OPF / admittance / topology draws). So load
scenario i has nearly the same global load level in *every* set: total demand of calib vs stress100/n1 correlates 0.95
per index (mean absolute relative difference 1.8%); the old 30-scenario stress sets and the exhaustive set share the
first 30 hours with correlation 0.79 to 0.87. The outages, noise and dispatch are independent (outage sequences agree
in 0.3% of rows, i.e. chance), so no scenario is duplicated, but calibration and test sets are **not independent in the
global load level**. For split-conformal this means the exchangeability argument holds only conditionally on the
same 100-hour load trajectory; say so in the write-up, or generate the calibration set from another `agg_profile`
(`external/gridfm-datakit/gridfm_datakit/load_profiles/ercot_load_act_hr_2024_*.csv`) if full independence is wanted.

## 3. Splits (scripts/make_splits.py, scripts/fetch_data.py)

- `make_splits.py` now iterates `data/{stress,stress100,calib,exhaustive}/*/case118_ieee/raw`, asserts contiguity, writes
  `splits_all`, and skips a dir without `bus_data.parquet` (generation unfinished) with a message.
- `fetch_data.py`: `find_split_file(task, grid, seed=0)`; new `--seeds` option writes `splits_paper_test` (seed 0, unchanged
  name) and `splits_paper_test_seed<S>` for other seeds. Default behaviour is unchanged.
- **The published splits differ across model seeds** (0/1/42) but are identical across sizes. For GOC 500 the splits are
  80/10/10 over all 219,746 scenarios by load scenario, so every seed's test scenarios were trained on by the other seeds
  (seed 0 test: 528/638 seen by seed 1 and 638/638 by seed 42). Use `splits_paper_test_seed<S>` for seed-S checkpoints;
  `splits_paper_test` is only clean for seed 0. The same holds for GOC 2000 (seed 0 test: 4/40 seen by seed 42).

## 4. GOC 500 (WS3.1) and GOC 2000 (WS3.2)

`find_split_file` works for `case500_goc` unchanged: the split lives under `mlflow/eval/case500_goc/<size>/seed<S>/<run>/
artifacts/stats/case500_goc_scenario_splits.json` (GOC 2000 uses `mlflow/train/...`; the `*` covers both). The three
sizes of a seed have byte-identical splits. Checkpoint weights for case500_goc base/small/tiny (seed 0) download
on first `load_genco` call (`checkpoints/genco-pf-datakit/case500_goc/<size>/seed0/`).

GOC 500 (`data/pf/case500_goc`, 30 partitions): 6000 scenarios on disk (ids 0..5999, contiguous), 273 load scenarios
(~22 variants each; the last, idx 272, is cut at 16 rows and is in no test set), raw 418 MB, processed cache 1.1 GB.
Paper test scenarios in range: seed 0 = 638 (29 load scenarios), seed 1 = 814 (37), seed 42 = 616 (28); test and
train/val share no load scenario within a seed.

GOC 2000 (`data/pf/case2000_goc`, 20 partitions, previously 5): 4000 scenarios (ids 0..3999, contiguous), 182 load
scenarios on disk (load idx 0..182, idx 160 absent, idx 182 partial with 20 rows), raw 1.7 GB, rebuilt processed cache
3.0 GB (old cache deleted first). Paper test scenarios in range (`splits_paper_test`): seed 0 = 40 (18 load scenarios,
was 11), seed 42 = 31 (11).

New held-out sets (`make_splits.py`; scenario's load scenario never appears in the train/val of the model seed(s)
concerned; the last load scenario on disk is dropped; each seed trained on its own random 20k-scenario subset of the
218,208, the subsets overlap at chance level):

| split | for model seed(s) | scenarios | load scenarios | composition |
|---|---|---|---|---|
| `splits_heldout` | 0 (default, small and tiny) | **879** | **40** | 40 are seed-0 paper-test, 839 lie outside seed 0's 20k subset (was 198 / 9) |
| `splits_heldout_seed42` | 42 | 638 | 29 | 31 paper-test, rest outside the subset |
| `splits_heldout_allseeds` | clean for both released seeds | 110 | 5 | strictest option |

Using `splits_heldout` with a seed-42 checkpoint leaks (84 of its 839 outside-subset scenarios are in seed 42's
subset). train/val inside these folders are the seen scenarios split in half and are unused by evaluation.

Anything that cached graphs for GOC 2000 before today must be rebuilt: the old `processed/` was removed and the smoke
run rebuilt it for the 4000 scenarios.

## 5. Smoke validation

Script (kept out of the repo): load_genco(..., data_root=..., split_dir=...) -> one batch -> GENCO forward with
`trace_layers` -> `inverse_transform` + `RemovePFMask` -> `acpf.extract_problems` -> finite `mismatch` /
`branch_loading` / `newton_pf`. All passed (batch of 16, or 8 / 2 for GOC):

| dataset | model | result |
|---|---|---|
| calib/n1, stress100/n1,n2,n3,n5, exhaustive/n1, exhaustive/n1_gen (`splits_all`) | case118 base | finite; NR converged in 2.0 to 2.3 iterations from GENCO; ground-truth residual <= 0.005 MW |
| case500_goc `splits_paper_test` | base, small, tiny | finite; NR 3.0 iterations; truth residual 0.014 MW |
| case2000_goc `splits_heldout` (small), `splits_heldout_allseeds` (tiny) | small, tiny | finite; NR 3.0 / 4.0 iterations; truth residual 0.022 MW |

Smoke side effects removed: the `results/configs/PowerFlow_case500_goc_*_seed0.yaml` and
`PowerFlow_case2000_goc_tiny_seed0.yaml` files that `load_genco` writes. Processed caches under each new data root
were kept (they are valid and avoid a rebuild). `results/` contains nothing else from this workstream.

## 6. Problems and notes

- `NMinusKGenerator` does not enumerate generators (docstring: "Only considers lines and transformers"); handled with the
  extra driver above, not by editing `external/`.
- The N-0 base topology is the first variant of every exhaustive load scenario; evaluation code that assumes "every
  scenario is an outage" must filter it (branch count out = 0 and no generator out).
- `acpf.base_case_voltages` (majority status) was checked on load scenarios 0 and 7 of `exhaustive/n1`,
  `exhaustive/n1_gen` and `calib/n1`: converged, finite.
- Disk: about 6.2 GB new (IEEE 118 sets 1.2 GB, GOC 500 1.5 GB, GOC 2000 +3.5 GB incl. a 3.0 GB processed cache). Free
  space went from 74 GB to 64 GB (shared with other jobs).
