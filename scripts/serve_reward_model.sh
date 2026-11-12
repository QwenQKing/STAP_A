#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=${STAP_RL:-$(cd "${SCRIPT_DIR}/.." && pwd)}
PY=${PYTHON_BIN:-${CONDA_BASE:-$HOME/miniconda3}/envs/stap/bin/python}
RM=$ROOT/rewardmodel
LOG=$RM/logs
mkdir -p "$LOG"
export CLIP_MODEL_PATH=${CLIP_MODEL_PATH:-$ROOT/models/clip-vit-large-patch14}
export HPS_MODEL_DIR=${HPS_MODEL_DIR:-$ROOT/models/HPSv2}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NO_PROXY="localhost,127.0.0.1,::1,${NO_PROXY:-}"
export no_proxy="$NO_PROXY"

start_clip() {
  local gpu=$1 port=$2
  CUDA_VISIBLE_DEVICES=$gpu CLIP_SERVER_PORT=$port \
    CLIP_MAX_BATCH_SIZE=${CLIP_MAX_BATCH_SIZE:-32} \
    CLIP_BATCH_TIMEOUT=${CLIP_BATCH_TIMEOUT:-0.05} \
    nohup "$PY" "$RM/clip_server.py" > "$LOG/clip_${port}.log" 2>&1 &
  echo "CLIP gpu=$gpu port=$port pid=$!"
}
start_hps() {
  local gpu=$1 port=$2
  CUDA_VISIBLE_DEVICES=$gpu HPS_SERVER_PORT=$port \
    HPS_MAX_BATCH_SIZE=${HPS_MAX_BATCH_SIZE:-32} \
    HPS_BATCH_TIMEOUT=${HPS_BATCH_TIMEOUT:-0.05} \
    nohup "$PY" "$RM/hps_server.py" > "$LOG/hps_${port}.log" 2>&1 &
  echo "HPS gpu=$gpu port=$port pid=$!"
}

test -f "$HPS_MODEL_DIR/HPS_v2.1_compressed.pt"
test -d "$CLIP_MODEL_PATH"
start_clip 0 8031
start_hps 0 8021
start_clip 1 8032
start_hps 1 8022
echo "POOL_STARTED $(date '+%F %T')"

