#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONDA_BASE=${CONDA_BASE:-$HOME/miniconda3}
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-vllmapi}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export NO_PROXY="localhost,127.0.0.1,::1,${NO_PROXY:-}"
export no_proxy="$NO_PROXY"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

MODEL_ROOT=${MODEL_ROOT:-${STAP_ROOT}/models}
MODEL=${MODEL:-${MODEL_ROOT}/Qwen2.5-VL-72B-Instruct}
SERVED_NAME=${SERVED_NAME:-Qwen2.5-VL-72B-Instruct}
PORT=${PORT:-3032}
TP=${TP:-8}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.40}
MAX_LEN=${MAX_LEN:-32768}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}

echo "[serve_qwen] env=$(python -c 'import vllm;print(vllm.__version__)') TP=$TP util=$GPU_MEM_UTIL max_len=$MAX_LEN port=$PORT"
echo "[serve_qwen] model=$MODEL  served-as=$SERVED_NAME"

DAEMON=0
if [ "${1:-}" = "-d" ]; then DAEMON=1; fi
LOG_DIR="${LOG_DIR:-${STAP_ROOT}/logs}"
mkdir -p "$LOG_DIR"

CMD=(python -m vllm.entrypoints.openai.api_server
  --model "$MODEL"
  --served-model-name "$SERVED_NAME"
  --port "$PORT"
  --host 0.0.0.0
  --tensor-parallel-size "$TP"
  --dtype bfloat16
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --max-model-len "$MAX_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --limit-mm-per-prompt '{"image":4,"video":1}'
  --enforce-eager
  --trust-remote-code)

if [ $DAEMON -eq 1 ]; then
  LOG="$LOG_DIR/serve_qwen_vlm_${PORT}.log"
  nohup "${CMD[@]}" > "$LOG" 2>&1 < /dev/null &
  PID=$!
  disown
  echo "[serve_qwen] launched in background (pid $PID). log: $LOG"
  echo "[serve_qwen] waiting for the server to become healthy ..."
  START=$(date +%s)
  for i in $(seq 1 360); do
    if curl -sf "localhost:${PORT}/v1/models" >/dev/null 2>&1; then
      printf "\n[serve_qwen] HEALTHY on localhost:%s (%ss)\n" "${PORT}" "$(( $(date +%s) - START ))"
      exit 0
    fi
    if ! kill -0 "$PID" 2>/dev/null; then
      printf "\n[serve_qwen] ERROR: server process exited, last log lines:\n"; tail -25 "$LOG"; exit 1
    fi
    PROG=$(tail -c 3000 "$LOG" 2>/dev/null | tr '\r' '\n' | grep -avE '^[[:space:]]*$' | tail -1)
    printf "\r[serve_qwen] +%-4ds | %s\033[K" "$(( $(date +%s) - START ))" "${PROG:0:120}"
    sleep 5
  done
  printf "\n[serve_qwen] not healthy yet, check the log: %s\n" "$LOG"
  exit 1
else
  exec "${CMD[@]}"
fi

