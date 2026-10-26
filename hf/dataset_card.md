---
license: apache-2.0
pretty_name: GENCO contingency sets for IEEE 118 (datakit, N-1 to N-5)
task_categories:
  - tabular-regression
tags:
  - power-systems
  - power-flow
  - contingency-analysis
  - n-minus-k
  - gridfm-datakit
viewer: false
size_categories:
  - 10K<n<100K
---

# GENCO contingency sets for IEEE 118

Raw gridfm-datakit output for the IEEE 118-bus system (pglib case), generated for the post "What is GENCO worth to a Newton solver? Pricing a grid foundation model against the pre-contingency state" (GridFM GENCO Challenge 2026). Each set is a list of AC power-flow solutions: a number of load scenarios (operating points), each with about 22 topology variants in which one to five branches or generators are out of service. Code, configs and numbers: [GitHub](https://github.com/msarwar20/genco-certified), [checkpoints](https://huggingface.co/sarwar21255/genco-ieee118-finetunes), [data](https://huggingface.co/datasets/sarwar21255/genco-contingency-ieee118), [demo](https://huggingface.co/spaces/sarwar21255/genco-operator-console).

The sets are in the same format as the released `gridfm/pf_small_case118_ieee`, so graphkit and the released GENCO checkpoints read them directly. Only the raw parquet is uploaded. The `processed/` caches that graphkit builds on first use are not (they are 2 to 3 times larger and are rebuilt automatically).

## Sets

Every set was generated with `gridfm_datakit generate configs/datakit/<config>.yaml` (datakit commit `d8a092d`; Julia PowerModels for the AC power flow). The config is copied into each `raw/config.yaml`. "Violating" = at least one in-service branch with max(|S_f|, |S_t|) / rate_a > 1.01. Counts come from `results/notes/data_D.md`, `ws4_N_nk.md`, `confirm_Z.md` and the `raw/n_scenarios.txt` files.

| Directory | Config | Seed | Load scen. | Outages (k) | Scenarios | Violating | Raw parquet | Used in the post for |
|---|---|---|---|---|---|---|---|---|
| `calib/n1` | `case118_calib_n1.yaml` | 3101 | 100 | random, 1 | 2,200 | 31.5% | 30.2 MB | calibration of the screening constants |
| `stress100/n1` | `case118_stress100_n1.yaml` | 3111 | 100 | random, 1 | 2,200 | 33.0% | 30.3 MB | held-out test (TABLE I, III) |
| `stress100/n2` | `case118_stress100_n2.yaml` | 3102 | 100 | random, 2 | 2,199 | 55.8% | 33.7 MB | held-out test |
| `stress100/n3` | `case118_stress100_n3.yaml` | 3103 | 100 | random, 3 | 2,199 | 67.9% | 34.8 MB | held-out test |
| `stress100/n5` | `case118_stress100_n5.yaml` | 3105 | 100 | random, 5 | 2,194 | 84.5% | 36.5 MB | held-out test |
| `confirm/n1` | `case118_confirm_n1.yaml` | 5101 | 100 | random, 1 | 2,200 | 34.9% | 30.4 MB | confirmation run |
| `confirm/n2` | `case118_confirm_n2.yaml` | 5102 | 100 | random, 2 | 2,200 | 56.5% | 33.7 MB | confirmation run |
| `confirm/n3` | `case118_confirm_n3.yaml` | 5103 | 100 | random, 3 | 2,200 | 71.6% | 35.1 MB | confirmation run |
| `confirm/n5` | `case118_confirm_n5.yaml` | 5105 | 100 | random, 5 | 2,196 | 85.9% | 36.6 MB | confirmation run |
| `exhaustive/n1` | `case118_exhaustive_n1.yaml` | 3201 | 30 | every branch, 1 | 5,340 (178 per load scen.) | 32.7% of 5,310 outage scen. | 68.6 MB | factor-reuse baselines (TABLE II) |
| `exhaustive/n1_gen` | `case118_exhaustive_n1_gen.yaml` | 3201 | 30 | every generator, 1 | 1,620 (54 per load scen.) | 19.6% of 1,590 outage scen. | 18.0 MB | factor-reuse baselines |
| `train_nk` | `case118_train_nk.yaml` | 4001 | 230 | random, mixed 1 to 5 | 5,056 | 58.2% | 77.4 MB | training the mixed N-k GENCO-Δ |
| `stress/n1,n2,n3,n5` | `case118_n{1,2,3,5}.yaml` | 2027, 2028, 2029, 2031 | 30 | random, 1/2/3/5 | 660 / 659 / 660 / 660 | 32.9 / 50.7 / 68.9 / 81.7% | 9.2 / 10.2 / 10.6 / 11.1 MB | GENCO-Δ evaluation (Section VI, Fig. 5), the Space bundle |

Notes on the contents:
- Dropped scenarios. In the random sets a few variants are missing because the AC power flow did not converge (`raw/error.log`), so some load scenarios have 21 variants. Scenario ids are contiguous from 0 in every set, which graphkit's streaming loader requires.
- `train_nk`. Outage counts k = 1..5 appear as 1,860 / 1,036 / 752 / 719 / 689 scenarios (probabilities 0.35 / 0.20 / 0.15 / 0.15 / 0.15 before the PF failures); 23.7% of the outages are generators. It is disjoint from the test sets: the closest load vectors differ by at least 14 MW at some bus.
- `exhaustive/n1`. Branches only (datakit's `n_minus_k` does not enumerate generators). The first variant of every load scenario is the intact N-0 topology; 9 branches that island the grid are dropped. `n1_gen` was produced by `data/exhaustive/run_exhaustive_gen.py` (a drop-in topology generator in the code repository), with the same seed and load scenarios, so the two pair by load scenario.
- `confirm/*`. The load profile is `ercot_load_act_hr_2024_east` instead of datakit's default. Its total demand correlates between -0.44 and -0.50 with the calibration and stress100 sets, per load-scenario index. The configs and the evaluation protocol (SHA-256 `027cf172823fdb3e3b0d8eac592d3b80018cbb50b558352b04d0a1803551aa42`) were written before the data existed.

## Layout

```
<set>/case118_ieee/raw/
  bus_data.parquet/scenario_partition=*/     bus voltages, loads, types, setpoints
  branch_data.parquet/scenario_partition=*/  flows, admittances, rate_a, br_status
  gen_data.parquet/scenario_partition=*/     dispatch, in_service
  runtime_data.parquet/ y_bus_data.parquet/
  scenarios_agg_load_profile.parquet         the load scenarios
  config.yaml  args.log  error.log  n_scenarios.txt
```

Column definitions are those of gridfm-datakit. Outages are identified by `br_status == 0` (branches) and `in_service == 0` (generators).

## How to use

```bash
hf download sarwar21255/genco-contingency-ieee118 --repo-type dataset --local-dir data
python scripts/make_splits.py        # writes splits_all/ for every set, asserts contiguous ids
```

Then `load_genco(..., data_root="data/stress100/n2", split_dir="data/stress100/n2/case118_ieee/splits_all")` runs a released or fine-tuned checkpoint on the whole set. `docs/REPRODUCE.md` in the code repository lists the script behind each table.

## Caveats

- Shared load trajectory. Datakit's `agg_load_profile` takes the first n hours of the aggregate profile, so load scenario i has nearly the same global load level in every set that uses the default profile (correlation 0.95 per index between `calib` and `stress100`). Outages, per-bus noise and dispatch are independent. The `confirm` sets use another profile for this reason.
- Duplicates. About 4% of the random N-1 rows are exact duplicate (load scenario, outage) draws, because outages are drawn with replacement within a load scenario. N-2 has one, N-3 and N-5 none.
- Not real operating data. Everything is simulated: perturbed pglib IEEE 118 with datakit load scenarios, and OPF dispatch that is computed once per load scenario and not re-dispatched after the outage.
- Setpoints. Slack and PV voltage magnitudes are dispatch setpoints that are constant across the outages of a load scenario.

## Sources and license

Generated data, Apache-2.0. The network is the IEEE 118-bus case from PGLib-OPF (IEEE PES Task Force on Benchmarks for Validation of Emerging Power System Algorithms, CC BY 4.0) read through gridfm-datakit (Apache-2.0). The default and `ercot_load_act_hr_2024_east` load profiles ship with gridfm-datakit; the scenarios contain derived load shapes only. Please cite the post and the GENCO paper (arXiv:2608.09921).

## Authors

Muhammad Sarwar (Iowa State University) and Muhammad Rizwan (Pakistan Institute of Engineering & Applied Sciences). Contact: msarwar@iastate.edu.
