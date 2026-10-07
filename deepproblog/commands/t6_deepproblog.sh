#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# Our reconstruction -- the library has no implementation of this task.
# 256 train / 64 test are the paper's sizes. 20 epochs rather than the
# paper's 5: nothing here is at 100% by 5, and dev accuracy is flat well
# before 20.
python3 -m src.experiments.t6.coin_ball \
  --train_size 256 \
  --test_size 64 \
  --batch_size 5 \
  --epochs 20 \
  --lr 1e-3 \
  --colour_lr 1.0 \
  --param_lr 1e-2 \
  --param_warm_up_lr 1e-4 \
  --param_warm_up_epochs 4 \
  --log_period 52 \
  --test_period 52
