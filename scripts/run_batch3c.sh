#!/bin/zsh
# Batch 3 driver, part c: the slow exp01 runs on the GOC 2000 held-out set. Sequential, nice 15.
cd "$(dirname "$0")/.." || exit 1
source .venv/bin/activate
export MLFLOW_ALLOW_FILE_STORE=true HF_HUB_DISABLE_PROGRESS_BARS=1
L=results/batch3_logs
for s in small tiny; do
  name=t3_exp01_$s
  [[ -f $L/$name.done ]] && { echo "skip $name"; continue; }
  echo "== $(date '+%F %T') start $name"
  nice -n 15 python -W ignore scripts/exp01_layers_newton.py --grid case2000_goc --size $s --split splits_heldout --batch_size 8 --tag case2000_goc_${s}_seed0_h879 > $L/$name.log 2>&1 && touch $L/$name.done
  echo "== $(date '+%F %T') end $name"
done
echo "ALL DONE c $(date '+%F %T')"
