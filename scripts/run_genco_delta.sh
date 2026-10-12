#!/usr/bin/env bash
# GENCO-Delta study on IEEE 118 (Apple M-series GPU): control vs delta_vm_va vs delta_vm_va_flows,
# released Base seeds {0,1,42} as initialisation, training seed = init seed, 12 epochs, lr 1e-4, batch 32.
#
# Arms (all fine-tuned from the released Base checkpoint on the same paper-train ids, selected on the paper-val ids):
#   control            task PowerFlow                 (masked inputs = 0, as released)
#   delta_vm_va        task PowerFlowContingencyAware (masked Vm/Va <- pre-contingency Vm/Va)
#   delta_vm_va_flows  task PowerFlowContingencyAware (masked Vm/Va and P_E/Q_E <- pre-contingency state / flows)
#   slack_aware        task PowerFlowSlackAware        (only the slack Vm setpoint becomes a known input). The delta arms
#                      implicitly receive the true slack Vm via the base case, so this arm separates "slack known"
#                      from "base-case information".
# Every fine-tuned model and the released checkpoint are evaluated on the paper test split and on the stress sets
# n1, n2, n3, n5 with scripts/eval_genco_delta.py (NR iterations, state error, equal-cost screening, cluster bootstrap).
#
# Usage (about 6.5 h on an M4, run it in the background):
#   nohup ./scripts/run_genco_delta.sh > results/genco_delta/run_matrix.log 2>&1 &
# Resumable: finished runs/evals (their JSON exists) are skipped; FORCE=1 redoes everything.
# Knobs (environment): SEEDS="0 1 42"  ARMS="control delta_vm_va delta_vm_va_flows"  EPOCHS=12  LR=1e-4  BS=32
#   (default ARMS now also includes slack_aware)  ACCEL=mps  SETS="paper n1 n2 n3 n5"  SMOKE=1 (1 epoch, 10% of the training batches, tag prefix smoke_)  NO_TRAIN=1  NO_EVAL=1
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export MLFLOW_ALLOW_FILE_STORE=true HF_HUB_DISABLE_PROGRESS_BARS=1
PY="python -W ignore"

SEEDS="${SEEDS:-0 1 42}"
ARMS="${ARMS:-control slack_aware delta_vm_va delta_vm_va_flows}"
SETS="${SETS:-paper n1 n2 n3 n5}"
EPOCHS="${EPOCHS:-12}"; LR="${LR:-1e-4}"; BS="${BS:-32}"; ACCEL="${ACCEL:-mps}"
PREFIX=""; FT_EXTRA=""
if [ "${SMOKE:-0}" = "1" ]; then EPOCHS=1; PREFIX="smoke_"; FT_EXTRA="--limit_train_batches 0.1"; fi
OUT=results/genco_delta; mkdir -p ${OUT}/logs ${OUT}/runs ${OUT}/eval

# data root and split of each evaluation set
data_root() { case "$1" in paper) echo data/pf ;; n*) echo data/stress/$1 ;; *) echo "unknown set $1" >&2; exit 1 ;; esac; }
# The released seed-s checkpoint was trained on seed s's own published split (they differ between seeds: ~75% of
# seed 0's test ids are in seed 1/42's training sets), so seed s is fine-tuned and tested on ITS split.
split_of()  { case "$1" in paper) if [ "$2" = 0 ]; then echo splits_paper_test; else echo splits_paper_test_seed$2; fi ;; *) echo splits_all ;; esac; }

echo "== $(date '+%F %T') GENCO-Delta matrix: arms=[${ARMS}] seeds=[${SEEDS}] epochs=${EPOCHS} sets=[${SETS}] smoke=${SMOKE:-0}"
$PY - <<'EOF'
import torch, sys
print("torch", torch.__version__, "mps", torch.backends.mps.is_available())
EOF

# 0. base-case stores (cover every scenario id of every split of every evaluation root) + per-seed paper splits
$PY scripts/build_basecase_store.py --all --seed_splits 1 42

arm_flags() {  # task name + delta flag of an arm
  case "$1" in
    control)           echo "--task_name PowerFlow" ;;
    slack_aware)       echo "--task_name PowerFlowSlackAware" ;;
    delta_vm_va)       echo "--task_name PowerFlowContingencyAware --delta_inputs vm_va" ;;
    delta_vm_va_flows) echo "--task_name PowerFlowContingencyAware --delta_inputs vm_va_flows" ;;
    *) echo "unknown arm $1" >&2; exit 1 ;;
  esac
}

eval_one() {  # eval_one <tag-prefix> <weights|-> <arm-or-released> <set>
  local name=$1 weights=$2 arm=$3 set=$4 root split tag
  root=$(data_root ${set}); split=$(split_of ${set} ${S}); tag="${name}__${set}"
  if [ -f ${OUT}/eval/${tag}.json ] && [ "${FORCE:-0}" != "1" ]; then echo "   skip eval ${tag}"; return; fi
  local flags; if [ "${arm}" = released ]; then flags="--task_name PowerFlow"; else flags=$(arm_flags ${arm}); fi
  local w=""; if [ "${weights}" != "-" ]; then w="--weights ${weights}"; fi
  $PY scripts/eval_genco_delta.py ${flags} ${w} --seed ${S} --data_root ${root} --split ${split} --tag ${tag} \
      > ${OUT}/logs/eval_${tag}.log 2>&1 && tail -n 12 ${OUT}/logs/eval_${tag}.log | sed 's/^/   /' || { echo "   EVAL FAILED ${tag}"; tail -n 20 ${OUT}/logs/eval_${tag}.log; }
}

for S in ${SEEDS}; do
  # released checkpoint, evaluated the same way (the reference every arm is compared with)
  if [ "${NO_EVAL:-0}" != "1" ]; then
    for set in ${SETS}; do eval_one "${PREFIX}released_s${S}" - released ${set}; done
  fi
  for ARM in ${ARMS}; do
    TAG="${PREFIX}${ARM}_s${S}"
    W=checkpoints/finetuned/${TAG}/best_model_state_dict.pt
    if [ "${NO_TRAIN:-0}" != "1" ]; then
      if [ -f ${OUT}/runs/${TAG}.json ] && [ -f ${W} ] && [ "${FORCE:-0}" != "1" ]; then
        echo "== skip training ${TAG} (run record exists)"
      else
        echo "== $(date '+%F %T') train ${TAG}"
        $PY scripts/finetune.py $(arm_flags ${ARM}) --grid case118_ieee --size base --seed ${S} --seed_train ${S} --split $(split_of paper ${S}) \
            --epochs ${EPOCHS} --lr ${LR} --batch_size ${BS} --accelerator ${ACCEL} ${FT_EXTRA} --tag ${TAG} --log_dir ${OUT}/train_logs/${TAG} \
            > ${OUT}/logs/train_${TAG}.log 2>&1 || { echo "TRAIN FAILED ${TAG}"; tail -n 30 ${OUT}/logs/train_${TAG}.log; continue; }
        tail -n 3 ${OUT}/logs/train_${TAG}.log | sed 's/^/   /'
      fi
    fi
    if [ "${NO_EVAL:-0}" != "1" ] && [ -f ${W} ]; then
      echo "== $(date '+%F %T') evaluate ${TAG}"
      for set in ${SETS}; do eval_one "${TAG}" ${W} ${ARM} ${set}; done
    fi
  done
done

# 1. tables: one row per (run, set), then mean/SD over init seeds per (arm, set)
if [ "${NO_EVAL:-0}" != "1" ]; then
  $PY scripts/eval_genco_delta.py --aggregate --out_dir ${OUT}/eval --prefix "${PREFIX}"
fi
echo "== $(date '+%F %T') done"
