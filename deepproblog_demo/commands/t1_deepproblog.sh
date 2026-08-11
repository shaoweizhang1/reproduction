#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Single-threaded on purpose: these tensors are too small for torch's
# intra-op threading to be anything but contention.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python3 -m src.experiments.t1.deepproblog \
  --batch_size 2 \
  --total_iterations 60000 \
  --log_period 100 \
  --test_period 1000 \
  --test_subset 500 \
  --lr 1e-3
