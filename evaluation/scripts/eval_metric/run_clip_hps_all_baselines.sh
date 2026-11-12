#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SCRIPT="${SCRIPT_DIR}/evaluate_clip_hps.py"
PY=${PYTHON_BIN:-python3}
BASE="${BASELINE_ROOT:-${EVAL_ROOT}/baseline_results}"
OUT="${OUT_DIR:-${EVAL_ROOT}/table1}"
mkdir -p "$OUT"

if [ -d "$BASE" ]; then
  for d in "$BASE"/*; do
    [ -d "$d" ] || continue
    name=$(basename "$d")
    echo "[$(date '+%F %T')] evaluating $name" | tee -a "$OUT/clip_hps_all.log"
    "$PY" "$SCRIPT" \
      --trajectory-root "$d" \
      --output "$OUT/${name}_metrics.json" \
      >> "$OUT/${name}_clip_hps.log" 2>&1
  done
fi

STAP_TRAJ="${STAP_TRAJECTORY_ROOT:-${EVAL_ROOT}/results/showcase_step50_eval}"
"$PY" "$SCRIPT" \
  --trajectory-root "$STAP_TRAJ" \
  --output "$OUT/STAP_metrics.json" \
  >> "$OUT/STAP_clip_hps.log" 2>&1
