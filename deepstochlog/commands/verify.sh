#!/usr/bin/env bash
# Non-destructive CPU checks, isolated from formal experiments.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/upstream${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=''
export DEEPSTOCHLOG_LOG_DIR
DEEPSTOCHLOG_LOG_DIR="$(mktemp -d)"
echo "Check records: $DEEPSTOCHLOG_LOG_DIR"
for script in commands/*.sh; do bash -n "$script"; done
python3 -m compileall -q src
mkdir "$DEEPSTOCHLOG_LOG_DIR/release"
git -C upstream archive 2a6982f4e48bf8559df441e0c33859d149eaab0b \
  | tar -x -C "$DEEPSTOCHLOG_LOG_DIR/release"
python3 -m src.verify "$@" 2>&1 | tee "$DEEPSTOCHLOG_LOG_DIR/verify.log"
