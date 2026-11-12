#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAP_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONDA_BASE=${CONDA_BASE:-$HOME/miniconda3}
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-STAP}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export NO_PROXY="localhost,127.0.0.1,::1,${NO_PROXY:-}"
export no_proxy="$NO_PROXY"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

MODEL=${MODEL:-${STAP_ROOT}/checkpoints/stap/stap_video_grpo_step50/global_step_50/actor/huggingface}
SERVED_NAME=${SERVED_NAME:-STAP-qwen25-vl-step50}
PORT=${PORT:-3033}
TP=${TP:-4}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.6}
MAX_LEN=${MAX_LEN:-32768}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}
LIMIT_VIDEO=${LIMIT_VIDEO:-4}

echo "[serve_stap_agent] env=$(python -c 'import vllm;print(vllm.__version__)') TP=$TP util=$GPU_MEM_UTIL max_len=$MAX_LEN port=$PORT"
echo "[serve_stap_agent] model=$MODEL served-as=$SERVED_NAME"

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
  --limit-mm-per-prompt "image=1,video=${LIMIT_VIDEO}"
  --trust-remote-code)

if [ $DAEMON -eq 1 ]; then
  LOG="$LOG_DIR/serve_stap_agent_${PORT}.log"
  nohup "${CMD[@]}" > "$LOG" 2>&1 < /dev/null &
  PID=$!
  disown
  echo "[serve_stap_agent] launched in background pid=$PID log=$LOG"
  START=$(date +%s)
  for _ in $(seq 1 240); do
    if curl -sf "localhost:${PORT}/v1/models" >/dev/null 2>&1; then
      printf "\n[serve_stap_agent] HEALTHY on localhost:%s  (%ss)\n" "$PORT" "$(( $(date +%s) - START ))"
      exit 0
    fi
    if ! kill -0 "$PID" 2>/dev/null; then
      printf "\n[serve_stap_agent] process exited; log tail:\n"
      tail -40 "$LOG"
      exit 1
    fi
    PROG=$(tail -c 3000 "$LOG" 2>/dev/null | tr '\r' '\n' | grep -avE '^[[:space:]]*$' | tail -1)
    printf "\r[serve_stap_agent] +%-4ds | %s\033[K" "$(( $(date +%s) - START ))" "${PROG:0:120}"
    sleep 5
  done
  printf "\n[serve_stap_agent] timeout; see log: %s\n" "$LOG"
  exit 1
else
  exec "${CMD[@]}"
fi

