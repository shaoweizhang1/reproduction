#!/usr/bin/env bash
# The paper's Table 1 with the original accuracy pass, spread over 16 CPU processes.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/upstream${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""

for DIGIT_LENGTH in 1 2 3 4; do
  echo "=== T1 addition | digit length $DIGIT_LENGTH ==="
  python3 -m src.experiments.t1.addition \
    --digit-length "$DIGIT_LENGTH" \
    --epochs 25 \
    --runs 5 \
    --workers 16
done
