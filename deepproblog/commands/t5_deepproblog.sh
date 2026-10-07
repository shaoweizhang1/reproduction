#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

python3 -m src.experiments.t5.deepproblog \
  --batch_size 10 \
  --epochs 40 \
  --log_period 10 \
  --test_period 30 \
  --lr 0.005 \
  --p_drop 0.5
