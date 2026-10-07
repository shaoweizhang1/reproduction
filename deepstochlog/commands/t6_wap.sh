#!/usr/bin/env bash
# The paper's word algebra problem experiment, reported in prose rather than in
# a table. Upstream's entry point runs the paper's protocol already -- 40
# epochs, batch 32, 5 runs -- and takes no arguments.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/upstream${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

: "${CUDA_VISIBLE_DEVICES:?An idle GPU ID is required.}"
export CUDA_VISIBLE_DEVICES

python3 -m src.experiments.t6.wap
