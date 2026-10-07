#!/usr/bin/env bash
# The paper's Table 4, max lengths 3-12, 3-15, 3-18. CPU, see t3_bracket.sh.
# Upstream's parser defaults to 5 epochs and the paper uses 1, so it is spelled
# out here rather than left to the default.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/upstream${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

export CUDA_VISIBLE_DEVICES=""

for MAX_LENGTH in 12 15 18; do
  echo "=== T4 a^n b^n c^n | max length $MAX_LENGTH ==="
  python3 -m src.experiments.t4.anbncn \
    --max-length "$MAX_LENGTH" \
    --epochs 1 \
    --runs 5
done
