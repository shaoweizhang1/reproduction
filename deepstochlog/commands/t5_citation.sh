#!/usr/bin/env bash
# Paper T5: depth-2 base programs, classifier Linear(50), five seeds.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/upstream${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=''
for dataset in citeseer cora; do
  python3 -m src.experiments.t5.citation \
    --dataset "$dataset" --seed 0 --runs 5 \
    --epochs 100 --lr 0.01 --log-freq 1 --workers 16 --device cpu
done
