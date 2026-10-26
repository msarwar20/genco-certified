# Reproducing the results

This file lists, in order, the commands that take a fresh clone to every result table and figure. It was assembled from `scripts/run_all.sh`, the `--help` of each script, the batch drivers `scripts/run_batch3*.sh`, `scripts/run_genco_delta*.sh`, and the "Reproduce" sections of `results/notes/*.md`, and where a table could not be tied to one command, section 10 says so explicitly.

All commands run from the repository root on the reference machine, a MacBook Air M4 (24 GB) with macOS 15 and Python 3.12, using zsh (which has no `timeout`, so use `${var}` rather than `$var[...]`).

## 0. What you need

| Item | Size |
|---|---|
| `external/` (two toolkit clones) | 17 MB |
| `.venv` (torch, PyG, mlflow, Julia packages) | 2.1 GB, plus a Julia install under `.venv/julia_env` |
| `data/pf` IEEE 14 to 118 (30 partitions each), GOC 500 (30), GOC 2000 (20) | 7.0 GB (GOC 2000 is 4.8 GB of that) |
| `data/opf` (IEEE 14 to 118, 30 partitions) | 1.0 GB |
| `data/stress`, `stress100`, `calib`, `exhaustive` (generated) | 1.3 GB |
| `data/train_nk` (generated, only for the mixed N-k study) | 0.3 GB |
| `checkpoints/genco-pf-datakit` + `genco-opf-datakit` (released) | 1.1 GB |
| `checkpoints/finetuned` (fine-tuned weights, 29 runs plus smoke runs) | 9.9 GB |
| `results/` after a full run, including git-ignored parquet intermediates | 2.3 GB (2.0 GB is `results/exp02`) |

A complete run needs about 25 GB, and the core reproduction (steps 1 to 6, no fine-tuning, no GOC 2000) needs about 8 GB. The tracked part of `results/` is only about 33 MB (JSON, CSV, Markdown, logs, small parquet).

Network access is needed for steps 1 and 2 (Hugging Face: `gridfm/genco-pf-datakit`, `gridfm/genco-opf-datakit`, `gridfm/pf_small_<grid>`, `gridfm/opf_small_<grid>`) and for the Julia packages that datakit installs on first use (PowerModels 0.21.6, Ipopt 1.15.0, Memento 1.5.0; see `external/gridfm-datakit/gridfm_datakit/juliapkg.json`). Nothing else touches the network.

## 1. Environment

```bash
./scripts/setup_env.sh                 # clones gridfm-graphkit 8a8defb and gridfm-datakit d8a092d, builds .venv (uv, Python 3.12)
source .venv/bin/activate
export MLFLOW_ALLOW_FILE_STORE=true HF_HUB_DISABLE_PROGRESS_BARS=1
pip install -e .                       # the genco_certified package (setup_env.sh installs the toolkits, not this)
```

`requirements.lock.txt` records the exact versions that produced the results and `results/environment.json` the platform and Julia versions. `pandas` must stay below 3 because graphkit's OPF baseline breaks under pandas 3. Setup takes 10 to 15 minutes, most of it downloads.

## 2. Data

```bash
# released test subsets: first 30 partitions (6,000 scenarios) per grid, plus the paper's published splits
python scripts/fetch_data.py --task pf  --partitions 30                     # IEEE 14, 30, 57, 118
python scripts/fetch_data.py --task opf --partitions 30
python scripts/fetch_data.py --task pf  --grids case118_ieee --partitions 30 --seeds 0 1 42   # per-seed test splits (splits_paper_test_seed1, _seed42)
python scripts/fetch_data.py --task pf  --grids case500_goc  --partitions 30 --seeds 0 1 42
python scripts/fetch_data.py --task pf  --grids case2000_goc --partitions 20 --seeds 0 42
```

`scripts/run_all.sh` fetches only 5 GOC 2000 partitions; the later held-out set (879 scenarios, 40 load scenarios) needs 20 (RESEARCH_LOG, "GOC grids"). Whether the seed lists above are exactly what was used for GOC 500 and 2000 is not recorded in a script: `make_splits.py` needs the published splits of seeds 0 and 42 to exist for GOC 2000, and the commands above produce them, but treat the `--seeds` arguments as reconstructed.

Stress, calibration and exhaustive scenario sets are generated with datakit (Julia + PowerModels; 6 processes). The seeds are fixed in the YAML files, so the sets are reproducible; generation takes minutes per set (the per-set wall time was not logged in the repository).

```bash
for c in case118_n1 case118_n2 case118_n3 case118_n5 \
         case118_stress100_n1 case118_stress100_n2 case118_stress100_n3 case118_stress100_n5 \
         case118_calib_n1 case118_exhaustive_n1; do
  gridfm_datakit generate configs/datakit/${c}.yaml
done
python scripts/gen_exhaustive_gen_outages.py configs/datakit/case118_exhaustive_n1_gen.yaml   # every single-generator outage
python scripts/make_splits.py                  # splits_all for every generated set; GOC 2000 held-out splits
```

`configs/datakit/trial.yaml` is a short trial configuration and is not needed, whereas `make_splits.py` takes no arguments and writes into `data/`, so run it once after the generation. Datakit renumbers scenarios, and the scripts assert that ids are contiguous 0..N-1 (graphkit's streaming loader needs this).

Note (RESEARCH_LOG, `results/notes/data_D.md`): the calibration, stress100 and paper sets share the global load trajectory (datakit takes the first n hours of the aggregate profile), so the conformal exchangeability argument holds only conditionally on it.

## 3. Reproduction of the published metrics

```bash
python scripts/reproduce.py --task pf  --sizes base small tiny     # about 20 checkpoints in total across tasks
python scripts/reproduce.py --task opf --sizes base small
python scripts/summarize_repro.py                                  # -> results/repro/summary_*.csv and the comparison table
```

This takes part of the 1.5 to 2 h budget of `run_all.sh` and writes to `results/repro/`.

## 4. Experiments

`scripts/run_all.sh` runs steps 1 to 4 of this section in one go for the original (pre-WS0) experiment set: about 1.5 to 2 hours on the M4 without fine-tuning, about 80 minutes more with `--with-finetune`. It does not call the later experiments (`--ci`, `--xcal`, exp07, exp08, the big sets, GOC 500, the GENCO-Δ matrix), which the commands below cover.

### 4.1 Layer dynamics and one Newton step (exp01), screening runs (exp02), slack audit (exp06), best start (exp05)

```bash
for gs in case118_ieee:base case118_ieee:small case118_ieee:tiny case14_ieee:base case30_ieee:base case57_ieee:base; do
  python scripts/exp01_layers_newton.py --grid ${gs%%:*} --size ${gs##*:}
  python scripts/exp06_slack_audit.py   --grid ${gs%%:*} --size ${gs##*:}
done
python scripts/exp02_screening.py --grid case118_ieee --size base
python scripts/exp05_best_start.py --grid case118_ieee --size base
for k in 1 2 3 5; do
  python scripts/exp01_layers_newton.py --grid case118_ieee --size base --data_root data/stress/n$k --split splits_all --tag stress_n${k}_base
  python scripts/exp02_screening.py     --grid case118_ieee --size base --data_root data/stress/n$k --split splits_all --tag stress_n${k}_base
done
python scripts/exp05_best_start.py --grid case118_ieee --size base --data_root data/stress/n5 --split splits_all --tag stress_n5
# GOC 2000 (small held-out, 198 scenarios, the original set)
python scripts/exp01_layers_newton.py --grid case2000_goc --size small --split splits_heldout --batch_size 8
python scripts/exp02_screening.py     --grid case2000_goc --size small --split splits_heldout
python scripts/exp05_best_start.py    --grid case2000_goc --size small --split splits_heldout
python scripts/exp06_slack_audit.py   --grid case2000_goc --size small --split splits_heldout --batch_size 8
```

Runtime: exp02 takes 20 to 30 s on the 638-scenario paper set and 2 to 4 minutes on a 2,200-scenario set. On GOC 2000, exp01 is the slowest job in the repository (tens of minutes to hours depending on the scenario count; timestamps are in `results/batch3_logs/`). Outputs land in `results/exp01`, `exp02` (git-ignored `*_branches.parquet`, `*_buses.parquet`), `exp05` and `exp06`.

Seeds and sizes used by the `--ci` tables:

```bash
python scripts/exp02_screening.py --grid case118_ieee --size base --seed 1  --split splits_paper_test_seed1
python scripts/exp02_screening.py --grid case118_ieee --size base --seed 42 --split splits_paper_test_seed42
for S in 1 42; do for k in 1 2 3 5; do
  python scripts/exp02_screening.py --grid case118_ieee --size base --seed $S --data_root data/stress/n$k --split splits_all --tag stress_n${k}_base_seed$S
done; done
for sz in small tiny; do
  python scripts/exp02_screening.py --grid case118_ieee --size $sz
  for k in 1 2 3 5; do python scripts/exp02_screening.py --grid case118_ieee --size $sz --data_root data/stress/n$k --split splits_all --tag stress_n${k}_${sz}; done
done
```

### 4.2 Screening analysis (no model runs; minutes)

```bash
python scripts/analyze_exp02.py --v3     # grouped/conformal calibration, ladders, DC baseline, sensitivity, Pareto -> results/policy_v3.log, exp02/v3_*.csv, results/validation/recertify.json
python scripts/analyze_exp02.py --ci     # cluster-bootstrap CIs, Clopper-Pearson bounds, seeds 0/1/42, Small/Tiny -> results/policy_v3_ci.log, exp02/v3_ci_*.csv, v3_seeds_*.csv
```

### 4.3 Large IEEE 118 sets, GOC 500, GOC 2000 held-out (batch 3)

`scripts/run_batch3.sh`, `run_batch3b.sh` and `run_batch3c.sh` hold the exact job list (nice 15, resumable through `results/batch3_logs/*.done`; zsh). They run exp02 on `data/calib/n1` and `data/stress100/n{1,2,3,5}`, exp01/02/05/06 (and exp07 for Base) on GOC 500 Base/Small/Tiny, the same on the GOC 2000 held-out set with tag suffix `_h879`, and exp08.

```bash
mkdir -p results/batch3_logs
nohup ./scripts/run_batch3.sh  > results/batch3_logs/driver.log  2>&1 &
nohup ./scripts/run_batch3b.sh > results/batch3_logs/driver_b.log 2>&1 &   # run after the first has finished, they share jobs
./scripts/run_batch3c.sh                                                   # the slow GOC 2000 exp01 runs
```

Then the cross-set analyses (the calibration constant used by the demo Space comes from the first one):

```bash
python scripts/analyze_exp02.py --xcal --cal_tag calib_n1_base \
  --eval_tags calib_n1_base stress100_n1_base stress100_n2_base stress100_n3_base stress100_n5_base case118_ieee_base_seed0 \
  --labels "calib N-1 (CV)" "S100 N-1" "S100 N-2" "S100 N-3" "S100 N-5" "paper N-1" --prefix big118
python scripts/analyze_exp02.py --xcal --cal_tag none --eval_tags case500_goc_base_seed0 case500_goc_small_seed0 case500_goc_tiny_seed0 \
  --labels "GOC500 Base" "GOC500 Small" "GOC500 Tiny" --prefix goc500
python scripts/analyze_exp02.py --xcal --cal_tag none --eval_tags case2000_goc_small_seed0_h879 case2000_goc_tiny_seed0_h879 case2000_goc_small_seed0 \
  --labels "GOC2000 Small (879)" "GOC2000 Tiny (879)" "GOC2000 Small (old 198)" --prefix goc2000h
python scripts/summarize_batch3.py --layers      # prints the batch-3 summary tables
```

(The `--xcal` commands are copied from `results/notes/batch3_H.md`, where the big118 call is complete and the goc500 and goc2000h calls are shown as in the note.) The outputs are `results/policy_{big118,goc500,goc2000h}.log` and `results/exp02/{big118,goc500,goc2000h}_*.csv`.

### 4.4 Factor-reuse baselines and break-even (exp07)

```bash
python scripts/exp07_industrial_baselines.py --grid case118_ieee --size base
for k in 1 2 3 5; do python scripts/exp07_industrial_baselines.py --grid case118_ieee --size base --data_root data/stress/n$k --split splits_all --tag stress_n${k}_base; done
python scripts/exp07_industrial_baselines.py --timing --repeats 5       # idle machine, single thread
python scripts/exp07_industrial_baselines.py --summarize                # -> results/exp07/*.csv, exp07.log
```

The exhaustive N-1 runs (`exh_n1_base`, `exh_n1gen_base`) are in `results/exp07/run_exhaustive.log`; the invocation is the same with `--data_root data/exhaustive/n1` (or `n1_gen`) `--split splits_all` and that tag, and GOC 500 Base is in `scripts/run_batch3.sh`.

### 4.5 Embedding probe (exp04), per-bus bias correction (exp08), runtime

```bash
python scripts/exp04_embedding_probe.py                       # grouped CV, controls, seeds 0/1/42 + Small, about 10 min
python scripts/exp04_embedding_probe.py --alpha_mode loo --tag _loo_seed0    # the leave-one-out variant (file names *_loo_seed0.*); assumed, see section 10
python scripts/exp08_bias_correction.py --size base --fit calib=data/calib/n1:splits_all \
    --sets paper=data/pf:splits_paper_test stress100_n5=data/stress100/n5:splits_all     # exact calls in run_batch3b.sh (Small and Tiny too)
python scripts/bench_runtime.py --grid case118_ieee --size base --n_scen 128              # idle machine
python scripts/bench_runtime.py --grid case2000_goc --size small --split splits_heldout --n_scen 64
python scripts/bench_genco_cpu.py --grid case500_goc --size base                          # CPU inference timing on GOC 500
```

GPU timings were taken on Colab with `notebooks/colab_gpu_timing.ipynb` (see `notebooks/README.md`); that run is outside this repository's scripts.

### 4.6 Slack-aware fine-tune and matched control (optional)

```bash
./scripts/run_all.sh --with-finetune      # or only the block below
for task in PowerFlowSlackAware PowerFlow; do
  python scripts/finetune.py --grid case118_ieee --size base --task_name $task --epochs 12 --lr 1e-4 --accelerator mps    # ~40 min each
done
python scripts/exp01_layers_newton.py --grid case118_ieee --size base --task_name PowerFlowSlackAware \
   --weights checkpoints/finetuned/PowerFlowSlackAware_case118_ieee_base_seed0/best_model_state_dict.pt --tag slackaware_case118_base
python scripts/exp01_layers_newton.py --grid case118_ieee --size base \
   --weights checkpoints/finetuned/PowerFlow_case118_ieee_base_seed0/best_model_state_dict.pt --tag control_case118_base
```

## 5. GENCO-Δ studies (training; Apple GPU)

```bash
nohup ./scripts/run_genco_delta.sh          > results/genco_delta/run_matrix.log   2>&1 &   # arms {control, slack_aware, delta_vm_va, delta_vm_va_flows} x seeds {0,1,42}
nohup ./scripts/run_genco_delta_residual.sh > results/genco_delta/run_residual.log 2>&1 &   # residual arms; run after the matrix
ARMS="residual_vm_va_flows" LR=3e-4 SUFFIX="_lr3e-4" ./scripts/run_genco_delta_residual.sh > results/genco_delta/run_residual_lr3e-4.log 2>&1
```

`scripts/queue_residual_after_matrix.sh <pid>` chains these. Both drivers build the base-case stores first (`scripts/build_basecase_store.py --all --seed_splits 1 42`), are resumable (finished run records and evaluation JSONs are skipped, `FORCE=1` redoes), and take the knobs documented at the top of each script (`SEEDS`, `ARMS`, `EPOCHS`, `LR`, `SETS`, `SMOKE=1` for a one-epoch dry run, `NICE`). Do not run two evaluations of the same task name at once, since they share a generated config file, and do not use the GPU for anything else during a run.

On an M4 a run takes about 3.3 minutes per epoch for 12 epochs, so roughly 40 minutes per run. The matrix has 12 runs (about 8 h), each residual arm has 3 runs, and evaluations take about 25 s each. Results are in `results/genco_delta/{runs,eval,logs,train_logs}` and the tables `summary_by_arm.md`, `residual_summary_by_arm.md` (aggregator: `python scripts/eval_genco_delta.py --aggregate --out_dir results/genco_delta/eval`).

The mixed N-k study (`data/train_nk`, `scripts/make_splits_nk.py`, `scripts/run_genco_delta_nk.sh`, config `configs/datakit/case118_train_nk.yaml`) was still running when this file was written; its commands are in the header of `scripts/run_genco_delta_nk.sh` and `results/notes/ws4_N_nk.md`, and are not repeated here.

GOC 2000 GENCO-Δ was prepared for Colab (`notebooks/colab_genco_delta.ipynb`, `scripts/pack_for_colab.sh`, `notebooks/README.md`).

## 6. Figures

```bash
python scripts/make_figures.py          # results/figures/fig1..fig4 (PNG), from results/exp01 and results/exp02
python scripts/make_post_figures.py     # results/figures/post/<name>.{png,svg} + data/<name>.csv with the plotted values
```

`make_figures.py`, `make_splits.py` and `build_space_bundle.py` take no arguments and run immediately, even with `--help`: do not call them to inspect usage. `make_figures.py` re-renders the committed PNGs, which can change their bytes without changing their content.

## 7. Demo Space

```bash
python scripts/build_space_bundle.py    # rebuilds space/bundle/ (about 5 MB; deterministic)
```

`space/bundle/calibration.json` holds the escalation constant read by `space/engine.py`, which was derived from `results/exp02/big118_policies.csv` (rows `set == "S100 N-1"`, policies `genco_seq` and `base_seq`, rule `conf0.10` by default); regenerate it from that file if the calibration is redone. Publishing the Space (`scripts/publish_space.py`) is outward-facing and is not part of reproduction.

## 8. Tests

```bash
python -m pytest -m "not data" -q       # offline: no downloads, 14 tests, about 5 s
python -m pytest -q tests               # full suite with data and released checkpoints present: 69 tests, about 20 s
```

`results/validation/pytest.log` is the log of a full run, and tests that need `data/pf/case118_ieee` or the released checkpoints carry the `data` marker and skip when the files are missing.

## 9. Verify the artifacts

```bash
python scripts/verify_artifacts.py                 # SHA256SUMS, every JSON parses, numbers are finite
python scripts/verify_artifacts.py --checkpoints   # also: released checkpoints match the digests in results/genco_delta/runs/*.json
python scripts/make_checksums.py                   # regenerate SHA256SUMS after results/ changes (and after `git add` of new result files)
```

`SHA256SUMS` covers the git-tracked `results/**/*.{json,csv,md,log}`, and logs written by running jobs change the digest, so regenerate it after a job finishes. A few JSON files contain NaN on purpose (a standard deviation over one seed, a GPU timing that was not measured, a precision with no predicted positives); they are listed in `NAN_OK` in `scripts/verify_artifacts.py`. Only the data-free parts were re-run to check this document, and on the same platform the screening tables are expected to reproduce to the printed precision, while timings and the last digits of mismatches will vary by machine. Training is seeded but is not expected to be bit-deterministic on MPS.

CI (`.github/workflows/ci.yml`) runs the offline tests and the checksum/finite check on Ubuntu with Python 3.12.

## 10. Tables and files without a single command

- Report Tables I to VII (`docs/report.md`) predate the revised analysis and are being rewritten. The mapping above is by result file, not by table number.
- Table II (embedding probe) comes from `exp04_embedding_probe.py`; the exact `--alpha_mode loo --tag _loo_seed0` call that produced `probe_v2_*_loo_seed0.*` is inferred from the file names and the script's options, not from a log.
- `results/runtime/*.json` (CPU/MPS timings): the three files correspond to `bench_runtime.py` (IEEE 118, GOC 2000) and `bench_genco_cpu.py` (GOC 500 CPU). The thread and batch settings used are inside the JSON, not in a driver.
- The Colab GPU timings are stored in `results/colab/` as exported by the notebook; the Colab GOC 2000 GENCO-Δ run was not done.
- The `--fit`/`--sets` arguments for exp08 Small and Tiny are in `run_batch3b.sh`; the Base call above is abbreviated.
- Seeds 1 and 42 for exp07 were not run (RESEARCH_LOG, WS1 notes): the GENCO-start columns of the factor-reuse tables are Base seed 0 only.
- `results/exp02/run_seeds_sizes.log` and `run_v3.log` are logs of runs of the commands in 4.1 and 4.2; the driver that looped over seeds and sizes was not saved as a script, so the loops in 4.1 are reconstructed from `results/notes/ws1_F_baselines_stats.md`, section 9.
