#!/bin/zsh
# Batch 3 driver: big IEEE118 sets, GOC 500, GOC 2000 held-out. Sequential, low priority (nice 15).
# Usage: nohup ./scripts/run_batch3.sh > results/batch3_logs/driver.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
source .venv/bin/activate
export MLFLOW_ALLOW_FILE_STORE=true HF_HUB_DISABLE_PROGRESS_BARS=1
L=results/batch3_logs
run() {  # run <logname> <cmd...>
  local name=$1; shift
  if [[ -f $L/$name.done ]]; then echo "skip $name"; return; fi
  echo "== $(date '+%F %T') start $name"
  nice -n 15 "$@" > $L/$name.log 2>&1 && touch $L/$name.done
  echo "== $(date '+%F %T') end $name (rc=$?)"
}
P="python -W ignore"
# ---- Task 1: big IEEE 118 sets (Base seed 0)
run t1_calib_n1 ${=P} scripts/exp02_screening.py --grid case118_ieee --size base --data_root data/calib/n1 --split splits_all --tag calib_n1_base
for k in 1 2 3 5; do
  run t1_stress100_n$k ${=P} scripts/exp02_screening.py --grid case118_ieee --size base --data_root data/stress100/n$k --split splits_all --tag stress100_n${k}_base
done
# ---- Task 2: GOC 500
for s in base small tiny; do
  run t2_exp02_$s ${=P} scripts/exp02_screening.py --grid case500_goc --size $s
  run t2_exp01_$s ${=P} scripts/exp01_layers_newton.py --grid case500_goc --size $s
  run t2_exp05_$s ${=P} scripts/exp05_best_start.py --grid case500_goc --size $s
  run t2_exp06_$s ${=P} scripts/exp06_slack_audit.py --grid case500_goc --size $s
done
run t2_exp07_base ${=P} scripts/exp07_industrial_baselines.py --grid case500_goc --size base
# ---- Task 3: GOC 2000 held-out (879 scenarios)
for s in small tiny; do
  run t3_exp02_$s ${=P} scripts/exp02_screening.py --grid case2000_goc --size $s --split splits_heldout --tag case2000_goc_${s}_seed0_h879
  run t3_exp01_$s ${=P} scripts/exp01_layers_newton.py --grid case2000_goc --size $s --split splits_heldout --batch_size 8 --tag case2000_goc_${s}_seed0_h879
  run t3_exp05_$s ${=P} scripts/exp05_best_start.py --grid case2000_goc --size $s --split splits_heldout --tag case2000_goc_${s}_h879
  run t3_exp06_$s ${=P} scripts/exp06_slack_audit.py --grid case2000_goc --size $s --split splits_heldout --batch_size 8 --tag case2000_goc_${s}_seed0_h879
done
echo "ALL DONE $(date '+%F %T')"
