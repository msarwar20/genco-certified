#!/usr/bin/env bash
# Wait for the running GENCO-Delta matrix (PID given) to finish, then run the residual study:
# (1) both residual arms at the matched lr 1e-4; (2) residual_vm_va_flows at lr 3e-4 (tag suffix _lr3e-4).
cd "$(dirname "$0")/.."
PID="$1"
while kill -0 "${PID}" 2>/dev/null; do sleep 60; done
echo "== $(date '+%F %T') matrix finished; starting residual study"
./scripts/run_genco_delta_residual.sh > results/genco_delta/run_residual.log 2>&1
ARMS="residual_vm_va_flows" LR=3e-4 SUFFIX="_lr3e-4" ./scripts/run_genco_delta_residual.sh > results/genco_delta/run_residual_lr3e-4.log 2>&1
echo "== $(date '+%F %T') residual study done"
