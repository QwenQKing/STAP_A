#!/usr/bin/env bash
set -euo pipefail

if [ "$(basename "${CONDA_PREFIX:-}")" != "stap" ]; then
  source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-stap}"
fi

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export NO_PROXY="localhost,127.0.0.1,::1,${NO_PROXY:-}"
export no_proxy="localhost,127.0.0.1,::1,${no_proxy:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"

MODEL_PATH="${MODEL_PATH:-checkpoints/stap/stap_video_grpo_step50/global_step_50/actor/huggingface}"
TP="${TP:-4}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "${SCRIPT_DIR}/run_stap_pure_inference.py" --model-path "${MODEL_PATH}" --tensor-parallel-size "${TP}" "$@"

