#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PY=${PYTHON_BIN:-python3}
EVAL="${SCRIPT_DIR}/evaluate_w0-w4.py"
OUT="${OUT_DIR:-${EVAL_ROOT}/table1}"
BASE="${BASELINE_ROOT:-${EVAL_ROOT}/baseline_results}"
mkdir -p "$OUT"

run_one() {
  local name="$1" root="$2"
  local result="$OUT/${name}_gpt54_all_metrics.json"
  echo "[$(date '+%F %T')] START name=$name root=$root output=$result"
  "$PY" "$EVAL" --metrics rq --workers 20 --trajectory-root "$root" --output "$result"
  echo "[$(date '+%F %T')] DONE name=$name"
}

if [ -d "$BASE" ]; then
  for d in "$BASE"/*; do
    [ -d "$d" ] || continue
    find "$d" -type f -name '*.mp4' -print -quit | grep -q . || continue
    name=$(basename "$d")
    run_one "$name" "$d"
  done
fi

run_one STAP "${STAP_TRAJECTORY_ROOT:-${EVAL_ROOT}/results/showcase_step50_eval}"
