#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

for MODEL in concat separate; do
  echo "=== T1 baseline | $MODEL ==="
  # Both baselines the paper mentions. 60000 rather than the paper's
  # 30000: the concat baseline and DeepProbLog were both still climbing
  # there. The separate baseline was not -- it ends slightly lower at
  # 60000 than at 30000. The 30000 runs are kept in log/archive/.
  python3 -m src.experiments.t1.baseline \
    --model "$MODEL" \
    --batch_size 2 \
    --total_iterations 60000 \
    --log_period 100 \
    --test_period 1000 \
    --test_subset 500 \
    --lr 1e-3 \
    --weight_decay 1e-2
done
