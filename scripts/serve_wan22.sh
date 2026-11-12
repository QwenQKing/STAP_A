#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONDA_SH=${CONDA_BASE:-$HOME/miniconda3}/bin/activate
ENV_NAME="${CONDA_ENV:-vllm-omni}"
GPUS="${GPUS:-4,5,6,7}"
PORT="${PORT:-4022}"
TP="${TP:-4}"
GMU="${GMU:-0.8}"
MODEL_ROOT=${MODEL_ROOT:-${STAP_ROOT}/models}
MODEL_DIR="${MODEL_DIR:-${MODEL_ROOT}/Wan2.2-T2V-A14B}"
SERVED_NAME="${SERVED_NAME:-wan22-a14b}"
LOG_DIR="${LOG_DIR:-${STAP_ROOT}/logs}"
DAEMON=0
[ "${1:-}" = "-d" ] && DAEMON=1

if curl -sf "localhost:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "[serve] a service is ALREADY healthy on port ${PORT} - nothing to do."
  echo "        (stop it first with: pkill -f \"vllm serve.*${SERVED_NAME}\")"
  exit 0
fi
if [ ! -d "$MODEL_DIR" ]; then
  echo "[serve] ERROR: model dir not found: $MODEL_DIR" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
source "$CONDA_SH" "$ENV_NAME"
export CUDA_VISIBLE_DEVICES="$GPUS"

echo "==============================================="
echo "  serving T2V backbone for the STAP rollout"
echo "  model        : $SERVED_NAME  ($MODEL_DIR)"
echo "  GPUs         : $GPUS   (TP=$TP)"
echo "  port         : $PORT   gpu-mem-util=$GMU"
echo "  env          : $ENV_NAME  ($(which vllm))"
echo "  mode         : $([ $DAEMON -eq 1 ] && echo background || echo foreground)"
echo "==============================================="

CMD=(vllm serve "$MODEL_DIR"
     --omni
     --port "$PORT"
     --host 0.0.0.0
     --tensor-parallel-size "$TP"
     --gpu-memory-utilization "$GMU"
     --dtype bfloat16
     --served-model-name "$SERVED_NAME")

if [ $DAEMON -eq 1 ]; then
  LOG="$LOG_DIR/serve_${SERVED_NAME}_${PORT}.log"
  nohup "${CMD[@]}" > "$LOG" 2>&1 < /dev/null &
  disown
  echo "[serve] launched in background (pid $!). log: $LOG"
  echo "[serve] waiting for health on :${PORT} (up to 30 min for model load) ..."
  for i in $(seq 1 180); do
    if curl -sf "localhost:${PORT}/v1/models" >/dev/null 2>&1; then
      echo "[serve] HEALTHY on localhost:${PORT}"
      exit 0
    fi
    sleep 10
  done
  echo "[serve] not healthy yet - check the log: $LOG"
  exit 1
else
  exec "${CMD[@]}"
fi

