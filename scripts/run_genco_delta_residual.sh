#!/usr/bin/env bash
# GENCO-Delta RESIDUAL study on IEEE 118 (Apple M-series GPU): the model predicts base case + correction instead of the
# absolute state (src/genco_certified/genco_residual.py). Companion of scripts/run_genco_delta.sh, same protocol:
# released Base seeds {0,1,42} as initialisation, training seed = init seed, per-seed paper splits, 12 epochs, lr 1e-4,
# batch 32, model selection on the paper-val ids, evaluation on the seed's paper test split and the stress sets.
#
# Arms (task PowerFlowContingencyResidual, model GNS_heterogeneous_residual; delta head = released mlp_bus with its last
# layer zeroed, so training starts exactly at the base case):
#   residual_vm_va        masked Vm/Va inputs <- pre-contingency Vm/Va
#   residual_vm_va_flows  masked Vm/Va and P_E/Q_E inputs <- pre-contingency state / flows
#   residual_init         (evaluation only, INIT_EVAL=1) the same model with 0 training steps; must equal the base case
# Evaluation tags: <arm>_s<seed>__<set>, written to results/genco_delta/eval next to the matrix' own runs. The released
# checkpoints are evaluated as in the matrix (released_s<seed>__<set>, skipped if the matrix already did it).
# Table: results/genco_delta/residual_summary_{runs.csv,by_arm.csv,by_arm.md}; the matrix' summary_* files are not touched.
#
# Usage (about as long as the delta arms of the matrix: ~3.3 min/epoch/run on an M4, 6 runs, ~25 s per evaluation):
#   nohup ./scripts/run_genco_delta_residual.sh > results/genco_delta/run_residual.log 2>&1 &
# Resumable: finished runs/evals (their JSON exists) are skipped; FORCE=1 redoes everything.
# Knobs (environment): SEEDS="0 1 42"  ARMS="residual_vm_va residual_vm_va_flows"  EPOCHS=12  LR=1e-4  BS=32  ACCEL=mps
#   SETS="paper n1 n2 n3 n5"  SMOKE=1 (1 epoch, 10% of the training batches, tag prefix smoke_)  NO_TRAIN=1  NO_EVAL=1
#   INIT_EVAL=1 (also evaluate residual_init)  NO_RELEASED=1 (do not evaluate the released checkpoints)  NICE=15 (prefix python with nice -n)
# Do not run it while another evaluation of task PowerFlowContingencyResidual is running (load_genco writes a shared
# results/configs/<task>_<grid>_<size>_seed<s>.yaml and reads it back immediately). The matrix' tasks have other names.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export MLFLOW_ALLOW_FILE_STORE=true HF_HUB_DISABLE_PROGRESS_BARS=1
PY="python -W ignore"
if [ -n "${NICE:-}" ]; then PY="nice -n ${NICE} ${PY}"; fi

SEEDS="${SEEDS:-0 1 42}"
ARMS="${ARMS:-residual_vm_va residual_vm_va_flows}"
SETS="${SETS:-paper n1 n2 n3 n5}"
EPOCHS="${EPOCHS:-12}"; LR="${LR:-1e-4}"; BS="${BS:-32}"; ACCEL="${ACCEL:-mps}"
PREFIX=""; FT_EXTRA=""; SUFFIX="${SUFFIX:-}"   # SUFFIX distinguishes e.g. learning-rate variants in run tags
if [ "${SMOKE:-0}" = "1" ]; then EPOCHS=1; PREFIX="smoke_"; FT_EXTRA="--limit_train_batches 0.1"; fi
OUT=results/genco_delta; mkdir -p ${OUT}/logs ${OUT}/runs ${OUT}/eval
RES_TASK=PowerFlowContingencyResidual

# data root and split of each evaluation set (identical to run_genco_delta.sh)
data_root() { case "$1" in paper) echo data/pf ;; n*) echo data/stress/$1 ;; *) echo "unknown set $1" >&2; exit 1 ;; esac; }
split_of()  { case "$1" in paper) if [ "$2" = 0 ]; then echo splits_paper_test; else echo splits_paper_test_seed$2; fi ;; *) echo splits_all ;; esac; }

echo "== $(date '+%F %T') GENCO-Delta residual: arms=[${ARMS}] seeds=[${SEEDS}] epochs=${EPOCHS} lr=${LR} sets=[${SETS}] smoke=${SMOKE:-0}"
$PY - <<'EOF'
import torch
print("torch", torch.__version__, "mps", torch.backends.mps.is_available())
EOF

# 0. base-case stores and per-seed paper splits: built by the matrix; only (re)build what is missing
need_build=0
for s in data/pf data/stress/n1 data/stress/n2 data/stress/n3 data/stress/n5; do
  [ -f ${s}/case118_ieee/basecase_store.pt ] || need_build=1
done
for sd in ${SEEDS}; do
  [ "${sd}" = 0 ] || [ -f data/pf/case118_ieee/splits_paper_test_seed${sd}/test.pt ] || need_build=1
done
if [ "${need_build}" = 1 ]; then $PY scripts/build_basecase_store.py --all --seed_splits 1 42; else echo "stores and per-seed splits present"; fi

arm_flags() {  # task name + delta flag of an arm
  case "$1" in
    residual_vm_va)       echo "--task_name ${RES_TASK} --delta_inputs vm_va" ;;
    residual_vm_va_flows) echo "--task_name ${RES_TASK} --delta_inputs vm_va_flows" ;;
    residual_init)        echo "--task_name ${RES_TASK} --delta_inputs vm_va_flows" ;;
    *) echo "unknown arm $1" >&2; exit 1 ;;
  esac
}

eval_one() {  # eval_one <tag-prefix> <weights|-> <arm-or-released> <set>
  local name=$1 weights=$2 arm=$3 set=$4 root split tag
  root=$(data_root ${set}); split=$(split_of ${set} ${S}); tag="${name}__${set}"
  if [ -f ${OUT}/eval/${tag}.json ] && [ "${FORCE:-0}" != "1" ]; then echo "   skip eval ${tag}"; return; fi
  local flags; if [ "${arm}" = released ]; then flags="--task_name PowerFlow"; else flags=$(arm_flags ${arm}); fi
  local w=""; if [ "${weights}" != "-" ]; then w="--weights ${weights}"; fi
  $PY scripts/eval_genco_delta_residual.py ${flags} ${w} --seed ${S} --data_root ${root} --split ${split} --tag ${tag} \
      > ${OUT}/logs/eval_${tag}.log 2>&1 && tail -n 12 ${OUT}/logs/eval_${tag}.log | sed 's/^/   /' || { echo "   EVAL FAILED ${tag}"; tail -n 20 ${OUT}/logs/eval_${tag}.log; }
}

for S in ${SEEDS}; do
  if [ "${NO_EVAL:-0}" != "1" ]; then
    if [ "${NO_RELEASED:-0}" != "1" ]; then   # reference every arm is compared with (same tags as the matrix)
      for set in ${SETS}; do eval_one "${PREFIX}released_s${S}" - released ${set}; done
    fi
    if [ "${INIT_EVAL:-0}" = "1" ]; then     # 0 training steps: must reproduce the base case
      for set in ${SETS}; do eval_one "${PREFIX}residual_init_s${S}" - residual_init ${set}; done
    fi
  fi
  for ARM in ${ARMS}; do
    TAG="${PREFIX}${ARM}${SUFFIX}_s${S}"
    W=checkpoints/finetuned/${TAG}/best_model_state_dict.pt
    if [ "${NO_TRAIN:-0}" != "1" ]; then
      if [ -f ${OUT}/runs/${TAG}.json ] && [ -f ${W} ] && [ "${FORCE:-0}" != "1" ]; then
        echo "== skip training ${TAG} (run record exists)"
      else
        echo "== $(date '+%F %T') train ${TAG}"
        $PY scripts/finetune_residual.py $(arm_flags ${ARM}) --grid case118_ieee --size base --seed ${S} --seed_train ${S} --split $(split_of paper ${S}) \
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

# tables: one row per (run, set), then mean/SD over init seeds per (arm, set); released / control / delta arms of the matrix
# appear in the same tables when their evaluations are in the same folder
if [ "${NO_EVAL:-0}" != "1" ]; then
  $PY scripts/eval_genco_delta_residual.py --aggregate --out_dir ${OUT}/eval --prefix "${PREFIX}"
fi
echo "== $(date '+%F %T') done"
