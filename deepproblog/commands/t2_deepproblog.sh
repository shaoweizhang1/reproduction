#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# A test query leaves the sum unbound, so two-digit addition enumerates all
# 199 possible sums where T1 enumerates 19. Evaluation is far more
# expensive as a result: score a smaller slice, less often.
python3 -m src.experiments.t2.deepproblog \
  --batch_size 2 \
  --total_iterations 30000 \
  --log_period 100 \
  --test_period 2000 \
  --test_subset 100 \
  --lr 1e-3
