#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Single-threaded on purpose: these tensors are too small for torch's
# intra-op threading to be anything but contention.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

TRAIN_LENGTHS="2 3 4 5 6"
TEST_LENGTHS="8 64"

# One run per cell of Table 1a. lr: official is 1.0, which never solved the
# task over three runs; 0.1 solved it in all three. --track_test produces
# Table 1b, so only test length 8 needs it.
for L in $TRAIN_LENGTHS; do
  for M in $TEST_LENGTHS; do
    echo "=== T4 Sort | training length $L | test length $M ==="
    track=()
    [ "$M" = "8" ] && track=(--track_test)
    python3 -m src.experiments.t4.deepproblog \
      --train_length "$L" \
      --test_length "$M" \
      --batch_size 16 \
      --epochs 40 \
      --log_period 50 \
      --test_period 50 \
      --lr 0.1 \
      --stop_at_accuracy 1.0 \
      "${track[@]}"
  done
done
