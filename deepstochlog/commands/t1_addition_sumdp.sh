#!/usr/bin/env bash
# T1 original training/proof engine with exact sum-DP accuracy evaluation.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/upstream${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${CUDA_VISIBLE_DEVICES:?An idle GPU ID is required.}"
for digit_length in 1 2 3 4; do
  echo "=== T1 sum-DP accuracy | length ${digit_length} ==="
  python3 -u -m src.experiments.t1.addition \
    --digit-length "$digit_length" \
    --epochs 25 \
    --runs 5 \
    --workers 1 \
    --accuracy-mode sum_dp
done
