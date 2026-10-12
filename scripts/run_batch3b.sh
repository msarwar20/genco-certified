#!/bin/zsh
# Batch 3 driver, part b: remaining GOC 2000 jobs (exp01 last because it is slow), the exp08 bias-correction runs
# in between. Sequential, nice 15. Resumable via results/batch3_logs/*.done.
cd "$(dirname "$0")/.." || exit 1
source .venv/bin/activate
export MLFLOW_ALLOW_FILE_STORE=true HF_HUB_DISABLE_PROGRESS_BARS=1
L=results/batch3_logs
run() {
  local name=$1; shift
  if [[ -f $L/$name.done ]]; then echo "skip $name"; return; fi
  echo "== $(date '+%F %T') start $name"
  nice -n 15 "$@" > $L/$name.log 2>&1 && touch $L/$name.done
  echo "== $(date '+%F %T') end $name"
}
P="python -W ignore"
run t3_exp02_tiny ${=P} scripts/exp02_screening.py --grid case2000_goc --size tiny --split splits_heldout --tag case2000_goc_tiny_seed0_h879
for s in small tiny; do
  run t3_exp05_$s ${=P} scripts/exp05_best_start.py --grid case2000_goc --size $s --split splits_heldout --tag case2000_goc_${s}_h879
  run t3_exp06_$s ${=P} scripts/exp06_slack_audit.py --grid case2000_goc --size $s --split splits_heldout --batch_size 8 --tag case2000_goc_${s}_seed0_h879
done
# ---- Task 4: per-bus bias correction (IEEE 118)
run t4_base ${=P} scripts/exp08_bias_correction.py --size base --fit calib=data/calib/n1:splits_all --sets paper=data/pf:splits_paper_test stress100_n5=data/stress100/n5:splits_all
for s in small tiny; do
  run t4_$s ${=P} scripts/exp08_bias_correction.py --size $s --fit calib=data/calib/n1:splits_all --sets paper=data/pf:splits_paper_test
done
# ---- slow one last
for s in small tiny; do
  run t3_exp01_$s ${=P} scripts/exp01_layers_newton.py --grid case2000_goc --size $s --split splits_heldout --batch_size 8 --tag case2000_goc_${s}_seed0_h879
done
echo "ALL DONE b $(date '+%F %T')"
