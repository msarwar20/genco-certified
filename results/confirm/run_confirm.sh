#!/bin/zsh
# Confirmatory pipeline (generation -> splits -> exp02). Resumable via results/confirm/*.done
cd "$(dirname "$0")/../.."
source .venv/bin/activate
export MLFLOW_ALLOW_FILE_STORE=true HF_HUB_DISABLE_PROGRESS_BARS=1
for k in 1 2 3 5; do
  if [[ ! -f results/confirm/gen_n${k}.done ]]; then
    gridfm_datakit generate configs/datakit/case118_confirm_n${k}.yaml > results/confirm/gen_n${k}.log 2>&1 && touch results/confirm/gen_n${k}.done || { echo "gen n${k} failed"; exit 1; }
  fi
done
python results/confirm/make_confirm_splits.py > results/confirm/splits.log 2>&1 || exit 1
for k in 1 2 3 5; do
  if [[ ! -f results/confirm/exp02_n${k}.done ]]; then
    python scripts/exp02_screening.py --grid case118_ieee --size base --seed 0 --data_root data/confirm/n${k} --split splits_all --tag confirm_n${k}_base > results/confirm/exp02_n${k}.log 2>&1 && touch results/confirm/exp02_n${k}.done || { echo "exp02 n${k} failed"; exit 1; }
  fi
done
python scripts/analyze_exp02.py --xcal --cal_tag calib_n1_base --eval_tags confirm_n1_base confirm_n2_base confirm_n3_base confirm_n5_base --labels "C N-1" "C N-2" "C N-3" "C N-5" --prefix confirm --out results/confirm --log results/confirm/policy_confirm.log > results/confirm/analysis.out 2>&1
echo ALLDONE
