#!/usr/bin/env bash
# The paper's Table 3. CPU: the cost is SLG resolution in SWI-Prolog.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/upstream${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

export CUDA_VISIBLE_DEVICES=""

for MAX_LENGTH in 10 14 18; do
  echo "=== T3 brackets | max length $MAX_LENGTH ==="
  python3 -m src.experiments.t3.bracket \
    --max-length "$MAX_LENGTH"
done
