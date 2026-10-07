#!/usr/bin/env bash
# The paper's T2 accuracy table: five seeds per expression length, each seed its
# own process. CPU: the engine evaluates every proof-tree node as a separate
# scalar op, and a length-7 seed took 4.6 h on CPU against 11.2 h on GPU.
# Usage: bash commands/t2_hwf.sh [released|paper] ["1 3 5 7"]
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/upstream${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""

MODEL="${1:-released}"
LENGTHS="${2:-1 3 5 7}"
LOG_ROOT="${DEEPSTOCHLOG_LOG_DIR:-log}/t2_$MODEL"

pids=()
for LENGTH in $LENGTHS; do
  for SEED in 0 1 2 3 4; do
    mkdir -p "$LOG_ROOT/seed$SEED"
    DEEPSTOCHLOG_LOG_DIR="$LOG_ROOT/seed$SEED" python3 -m src.experiments.t2.hwf \
      --length "$LENGTH" \
      --model "$MODEL" \
      --epochs 20 \
      --batch-size 2 \
      --lr 0.003 \
      --seed "$SEED" \
      --runs 1 \
      --device cpu > /dev/null 2> "$LOG_ROOT/seed$SEED/t2_hwf_l$LENGTH.stderr" &
    pids+=("$!")
  done
done
for pid in "${pids[@]}"; do wait "$pid"; done
python3 -m src.experiments.t2.hwf --model "$MODEL" --report "$LOG_ROOT"
