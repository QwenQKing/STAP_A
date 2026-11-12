#!/usr/bin/env bash
set -xe

if [ "$(basename "${CONDA_PREFIX:-}")" != "stap" ]; then
  source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-stap}"
fi

export PYTHONUNBUFFERED=1
exec > >(python3 -u -c '
import signal
import sys
import time

signal.signal(signal.SIGHUP, signal.SIG_IGN)
for line in sys.stdin:
    sys.stdout.write(time.strftime("[%Y-%m-%d %H:%M:%S] ") + line)
    sys.stdout.flush()
') 2>&1

echo "[run_grpo] python: $(which python3) | ray: $(which ray)"

export STAP_RL="${STAP_RL:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PYTHONPATH="$(dirname "${STAP_RL}")${PYTHONPATH:+:${PYTHONPATH}}"
export MODEL_ROOT="${MODEL_ROOT:-${STAP_RL}/models}"
export BASE_MODEL="${BASE_MODEL:-${MODEL_ROOT}/Qwen2.5-VL-7B-Instruct}"
export PROJECT_NAME="${PROJECT_NAME:-stap}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-stap_video_grpo}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export RAY_PORT="${RAY_PORT:-6580}"
export RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8280}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray_stap}"

export TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-${STAP_RL}/data/train.parquet}"
export VAL_DATA_PATH="${VAL_DATA_PATH:-${STAP_RL}/data/val.parquet}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-${STAP_RL}/logs/wandb}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-stap}"
export WANDB_START_METHOD="${WANDB_START_METHOD:-thread}"
export WANDB_DISABLE_SERVICE="${WANDB_DISABLE_SERVICE:-true}"
export STAP_METRICS_JSONL="${STAP_METRICS_JSONL:-${STAP_RL}/logs/train_metrics.jsonl}"
mkdir -p "${WANDB_DIR}" "$(dirname "${STAP_METRICS_JSONL}")"

export NO_PROXY="localhost,127.0.0.1,::1,${NO_PROXY:-}"
export no_proxy="localhost,127.0.0.1,::1,${no_proxy:-}"

export STAP_VLM_API_BASE="${STAP_VLM_API_BASE:-}"
export STAP_VLM_MODEL="${STAP_VLM_MODEL:-Qwen2.5-VL-72B-Instruct}"
export STAP_VLM_TIMEOUT="${STAP_VLM_TIMEOUT:-300}"
export CLIP_SERVER_URL="${CLIP_SERVER_URL:-}"
export HPS_SERVER_URL="${HPS_SERVER_URL:-}"
export STAP_JUDGE_FRAMES="${STAP_JUDGE_FRAMES:-8}"

export RAY_SPILL_DIR="${RAY_SPILL_DIR:-${STAP_RL}/ray_spill}"
export RAY_object_spilling_config="{\"type\":\"filesystem\",\"params\":{\"directory_path\":\"${RAY_SPILL_DIR}\"}}"
mkdir -p "${RAY_SPILL_DIR}" "${STAP_RL}/data"

OLD_PID=$(lsof -ti :"${RAY_PORT}" 2>/dev/null || true)
if [ -n "${OLD_PID}" ]; then
  kill -9 ${OLD_PID} 2>/dev/null || true
  sleep 2
fi

rm -rf "${RAY_TEMP_DIR}"
mkdir -p "${RAY_TEMP_DIR}"

unset RAY_ADDRESS
NODE_IP=$(hostname -i | awk '{print $1}')

ray start --head \
  --node-ip-address="${NODE_IP}" \
  --num-cpus=8 \
  --num-gpus=4 \
  --port="${RAY_PORT}" \
  --include-dashboard=false \
  --temp-dir="${RAY_TEMP_DIR}" \
  --object-store-memory=8000000000

export RAY_ADDRESS="${NODE_IP}:${RAY_PORT}"
sleep 10
ray status --address="${NODE_IP}:${RAY_PORT}" || exit 1

python3 -m STAP.trainer.multimodal.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="${TRAIN_DATA_PATH}" \
  data.val_files="${VAL_DATA_PATH}" \
  data.train_batch_size=32 \
  data.max_prompt_length=8192 \
  data.max_response_length=8192 \
  data.image_key=images \
  data.user_prompt_round_1="${STAP_RL}/prompts/round_1_user_prompt_qwenvl_stap.pkl" \
  +data.user_prompt_after_generate="${STAP_RL}/prompts/user_prompt_after_video_generate_stap.pkl" \
  actor_rollout_ref.model.path="${BASE_MODEL}" \
  actor_rollout_ref.actor.optim.lr=2e-6 \
  actor_rollout_ref.actor.optim.lr_sigmoid_decay_warmup=True \
  actor_rollout_ref.actor.optim.lr_sigmoid_decay_ratio=0.95 \
  actor_rollout_ref.actor.optim.lr_sigmoid_decay_warmup_steps=45 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_multi_turn_response_mask=True \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm_multiturn_stap \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.max_gen_round=5 \
  actor_rollout_ref.rollout.response_length_total=16384 \
  +actor_rollout_ref.rollout.generate.video_generate_limit=4 \
  +actor_rollout_ref.rollout.generate.base_url="${WAN22_BASE_URL:-}" \
  +actor_rollout_ref.rollout.generate.model=wan22-a14b \
  +actor_rollout_ref.rollout.generate.n_frames=4 \
  +actor_rollout_ref.rollout.generate.parallel_tool_call_threads=1 \
  +actor_rollout_ref.rollout.generate.user_prompt_after_generate="${STAP_RL}/prompts/user_prompt_after_video_generate_stap.pkl" \
  +actor_rollout_ref.rollout.generate.user_prompt_last_round="${STAP_RL}/prompts/user_prompt_last_round_stap.pkl" \
  actor_rollout_ref.rollout.limit_images=null \
  '+actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt={image:1,video:4}' \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.kl_ctrl.kl_coef=0.001 \
  trainer.critic_warmup=0 \
  trainer.logger="['console','wandb']" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.save_freq=1 \
  trainer.resume_mode="${RESUME_FROM:-auto}" \
  trainer.test_freq=-1 \
  trainer.total_epochs=1 \
  +trainer.format_penalty=0.1 \
  +trainer.reward_mode=stap_multi_anchor \
  +trainer.val_before_train=False \
  +algorithm.filter_groups.enable=False

