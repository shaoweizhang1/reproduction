#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

TRAIN_LENGTHS="2 4 8"
TEST_LENGTHS="8 64"

# One run per cell of the paper's Table 1a. --stop_at_accuracy replaces
# the official fixed 40 epochs, which stops while dev accuracy is still
# climbing.
for L in $TRAIN_LENGTHS; do
  for M in $TEST_LENGTHS; do
    echo "=== T3 Add | training length $L | test length $M ==="
    python3 -m src.experiments.t3.deepproblog \
      --train_length "$L" \
      --test_length "$M" \
      --batch_size 50 \
      --epochs 500 \
      --log_period 50 \
      --test_period 50 \
      --lr 0.02 \
      --stop_at_accuracy 1.0
  done
done
