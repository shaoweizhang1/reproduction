#!/usr/bin/env bash
# Table 6: freeze sources, validate answer sets, then run 36 serial cells.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/upstream${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
run_root="${1:-${DEEPSTOCHLOG_LOG_DIR:-log}/table6_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$(dirname "$run_root")"
mkdir "$run_root"
run_root="$(cd "$run_root" && pwd)"
mkdir "$run_root/snapshot"
git -C upstream archive 2a6982f4e48bf8559df441e0c33859d149eaab0b | tar -x -C "$run_root/snapshot"
cp src/experiments/t2/table6_unified.py "$run_root/driver.py"
cp commands/t2_parsing_time.sh "$run_root/run.sh"
find "$run_root/snapshot" -type f -print0 | sort -z | xargs -0 sha256sum > "$run_root/source.sha256"
sha256sum "$run_root/driver.py" "$run_root/run.sh" >> "$run_root/source.sha256"
exec > >(tee "$run_root/runner.log") 2>&1
date -Is
swipl --version
python3 "$run_root/driver.py" validate --release "$run_root/snapshot" --out "$run_root/validation"
nice -n 19 python3 "$run_root/driver.py" batch \
  --release "$run_root/snapshot" \
  --validation "$run_root/validation" \
  --out "$run_root/results" \
  --repetitions 3 \
  --wall-limit 3600 \
  --table-space 16000000000 \
  --stack-limit 1g
sha256sum -c "$run_root/source.sha256"
date -Is
echo 'COMPLETE: 36 cells (SLD length 11 may time out).'
