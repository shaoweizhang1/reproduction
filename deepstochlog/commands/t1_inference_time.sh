#!/usr/bin/env bash
# Table 7: original CPU inference, 100 queries per run, five runs per length.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/upstream${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
python3 -u -m src.experiments.t1.inference_time "$@"
