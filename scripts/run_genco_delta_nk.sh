#!/usr/bin/env bash
# Residual GENCO-Delta trained on a MIXED N-k set (data/train_nk: k=1..5, probs 0.35/0.2/0.15/0.15/0.15, seed 4001,
# 230 load scenarios x 22 variants, splits_nk = 207/23 load scenarios). Companion of scripts/run_genco_delta_residual.sh:
# same model/task (PowerFlowContingencyResidual), arm flags residual_vm_va_flows, lr 3e-4, 12 epochs, batch 32, MPS, init from
# released IEEE 118 Base seeds {0,1,42}, training seed = init seed. Evaluation sets and per-seed splits are IDENTICAL to the residual
# study (paper = data/pf with the seed's published test split; n1 n2 n3 n5 = data/stress/n*, splits_all).
# Tag: residual_nk_vm_va_flows_lr3e-4_s<seed>. Outputs: results/genco_delta/{eval,logs,runs,train_logs}.
# Resumable (run record + weights exist -> skip training; eval JSON exists -> skip eval). FORCE=1 redoes.
# Knobs: SEEDS="0 1 42" EPOCHS=12 LR=3e-4 BS=32 ACCEL=mps SETS="paper n1 n2 n3 n5" NO_TRAIN=1 NO_EVAL=1 NICE=15
#   nohup ./scripts/run_genco_delta_nk.sh > results/genco_delta/run_nk.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export MLFLOW_ALLOW_FILE_STORE=true HF_HUB_DISABLE_PROGRESS_BARS=1
PY="python -W ignore"
if [ -n "${NICE:-}" ]; then PY="nice -n ${NICE} ${PY}"; fi

SEEDS="${SEEDS:-0 1 42}"
SETS="${SETS:-paper n1 n2 n3 n5}"
EPOCHS="${EPOCHS:-12}"; LR="${LR:-3e-4}"; BS="${BS:-32}"; ACCEL="${ACCEL:-mps}"
ARM=residual_nk_vm_va_flows_lr3e-4
OUT=results/genco_delta; mkdir -p ${OUT}/logs ${OUT}/runs ${OUT}/eval
RES_TASK=PowerFlowContingencyResidual
TRAIN_ROOT=data/train_nk; TRAIN_SPLIT=splits_nk

# same logic as run_genco_delta_residual.sh
data_root() { case "$1" in paper) echo data/pf ;; n*) echo data/stress/$1 ;; *) echo "unknown set $1" >&2; exit 1 ;; esac; }
split_of()  { case "$1" in paper) if [ "$2" = 0 ]; then echo splits_paper_test; else echo splits_paper_test_seed$2; fi ;; *) echo splits_all ;; esac; }

echo "== $(date '+%F %T') GENCO-Delta N-k: arm=${ARM} seeds=[${SEEDS}] epochs=${EPOCHS} lr=${LR} sets=[${SETS}]"
for f in ${TRAIN_ROOT}/case118_ieee/basecase_store.pt ${TRAIN_ROOT}/case118_ieee/${TRAIN_SPLIT}/train.pt data/pf/case118_ieee/basecase_store.pt; do
  [ -f ${f} ] || { echo "missing ${f}" >&2; exit 1; }
done

for S in ${SEEDS}; do
  TAG="${ARM}_s${S}"
  W=checkpoints/finetuned/${TAG}/best_model_state_dict.pt
  if [ "${NO_TRAIN:-0}" != "1" ]; then
    if [ -f ${OUT}/runs/${TAG}.json ] && [ -f ${W} ] && [ "${FORCE:-0}" != "1" ]; then
      echo "== skip training ${TAG} (run record exists)"
    else
      echo "== $(date '+%F %T') train ${TAG}"
      ${PY} scripts/finetune_residual.py --task_name ${RES_TASK} --delta_inputs vm_va_flows --grid case118_ieee --size base \
          --seed ${S} --seed_train ${S} --data_root ${TRAIN_ROOT} --split ${TRAIN_SPLIT} \
          --epochs ${EPOCHS} --lr ${LR} --batch_size ${BS} --accelerator ${ACCEL} --tag ${TAG} --log_dir ${OUT}/train_logs/${TAG} \
          > ${OUT}/logs/train_${TAG}.log 2>&1 || { echo "TRAIN FAILED ${TAG}"; tail -n 30 ${OUT}/logs/train_${TAG}.log; continue; }
      tail -n 3 ${OUT}/logs/train_${TAG}.log | sed 's/^/   /'
    fi
  fi
  if [ "${NO_EVAL:-0}" != "1" ] && [ -f ${W} ]; then
    echo "== $(date '+%F %T') evaluate ${TAG}"
    for set in ${SETS}; do
      root=$(data_root ${set}); split=$(split_of ${set} ${S}); tag="${TAG}__${set}"
      if [ -f ${OUT}/eval/${tag}.json ] && [ "${FORCE:-0}" != "1" ]; then echo "   skip eval ${tag}"; continue; fi
      ${PY} scripts/eval_genco_delta_residual.py --task_name ${RES_TASK} --delta_inputs vm_va_flows --weights ${W} --seed ${S} \
          --data_root ${root} --split ${split} --tag ${tag} > ${OUT}/logs/eval_${tag}.log 2>&1 \
          && tail -n 12 ${OUT}/logs/eval_${tag}.log | sed 's/^/   /' || { echo "   EVAL FAILED ${tag}"; tail -n 20 ${OUT}/logs/eval_${tag}.log; }
    done
  fi
done

if [ "${NO_EVAL:-0}" != "1" ]; then
  ${PY} scripts/eval_genco_delta_residual.py --aggregate --out_dir ${OUT}/eval --prefix ""
fi
echo "== $(date '+%F %T') done"
