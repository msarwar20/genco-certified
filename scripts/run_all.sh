#!/usr/bin/env bash
# Reproduce every result in docs/report.md on a laptop (~1.5-2 h without fine-tuning, ~+80 min with).
# Usage: ./scripts/run_all.sh [--with-finetune]
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export MLFLOW_ALLOW_FILE_STORE=true HF_HUB_DISABLE_PROGRESS_BARS=1
PY="python -W ignore"

# 0. data: paper-test subsets (+ published splits), GOC 2000 subset, stress sets
$PY scripts/fetch_data.py --task pf  --partitions 30
$PY scripts/fetch_data.py --task opf --partitions 30
$PY scripts/fetch_data.py --task pf  --grids case2000_goc --partitions 5
for k in 1 2 3 5; do [ -f data/stress/n$k/case118_ieee/raw/n_scenarios.txt ] || gridfm_datakit generate configs/datakit/case118_n$k.yaml; done
$PY scripts/make_splits.py

# 1. reproduction of published metrics
$PY scripts/reproduce.py --task pf  --sizes base small tiny
$PY scripts/reproduce.py --task opf --sizes base small
$PY scripts/summarize_repro.py

# 2. experiments
for gs in case118_ieee:base case118_ieee:small case118_ieee:tiny case14_ieee:base case30_ieee:base case57_ieee:base; do
  $PY scripts/exp01_layers_newton.py --grid ${gs%%:*} --size ${gs##*:}
done
$PY scripts/exp01_layers_newton.py --grid case2000_goc --size small --split splits_heldout --batch_size 8
$PY scripts/exp02_screening.py --grid case118_ieee --size base
$PY scripts/exp02_screening.py --grid case2000_goc --size small --split splits_heldout
for k in 1 2 3 5; do
  $PY scripts/exp01_layers_newton.py --grid case118_ieee --size base --data_root data/stress/n$k --split splits_all --tag stress_n${k}_base
  $PY scripts/exp02_screening.py     --grid case118_ieee --size base --data_root data/stress/n$k --split splits_all --tag stress_n${k}_base
done
( $PY scripts/analyze_exp02.py --tag case118_ieee_base_seed0 --tau 1.01
  for k in 1 2 3 5; do $PY scripts/analyze_exp02.py --tag stress_n${k}_base --cal_tag stress_n1_base --tau 1.01; done
) | grep -v "DC voltage" > results/policy_with_basecase.log
$PY scripts/exp04_embedding_probe.py
$PY scripts/exp05_best_start.py --grid case118_ieee --size base
$PY scripts/exp05_best_start.py --grid case118_ieee --size base --data_root data/stress/n5 --split splits_all --tag stress_n5
$PY scripts/exp05_best_start.py --grid case2000_goc --size small --split splits_heldout

# 3. optional: slack-aware fine-tune + matched control (~40 min each on Apple M-series GPU)
if [ "${1:-}" = "--with-finetune" ]; then
  for task in PowerFlowSlackAware PowerFlow; do
    $PY scripts/finetune.py --grid case118_ieee --size base --task_name $task --epochs 12 --lr 1e-4 --accelerator mps
  done
  $PY scripts/exp01_layers_newton.py --grid case118_ieee --size base --task_name PowerFlowSlackAware \
      --weights checkpoints/finetuned/PowerFlowSlackAware_case118_ieee_base_seed0/best_model_state_dict.pt --tag slackaware_case118_base
  $PY scripts/exp01_layers_newton.py --grid case118_ieee --size base \
      --weights checkpoints/finetuned/PowerFlow_case118_ieee_base_seed0/best_model_state_dict.pt --tag control_case118_base
fi

# 4. runtime (run on an idle machine) and figures
$PY scripts/bench_runtime.py --grid case118_ieee --size base --n_scen 128
$PY scripts/bench_runtime.py --grid case2000_goc --size small --split splits_heldout --n_scen 64
$PY scripts/make_figures.py
python -m pytest -q tests
