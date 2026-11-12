# STAP

STAP is a reinforcement-learning framework for improving text-to-video generation with multi-turn prompting, video generation feedback, and multimodal reward signals.

This repository contains the anonymized training code, training data, evaluation data, and inference/evaluation scripts used for STAP.

## Repository Structure

```text
STAP/
- data/                 # Training and validation parquet files
- evaluation/           # Evaluation data, inference runners, and metric scripts
- monkey_patch/         # Runtime patches used by training
- prompts/              # Prompt templates used by training
- rewardmodel/          # CLIP and HPS reward model servers
- scripts/              # Training and service launch scripts
- trainer/              # GRPO/PPO trainer implementation
- utils/                # Dataset, metrics, and tool utilities
- workers/              # Actor, rollout, and reward workers
```

## Training

The main training entrypoint is:

```bash
bash scripts/run_STAP_grpo.sh
```

Important environment variables:

```bash
export MODEL_ROOT=$PWD/models
export BASE_MODEL=$MODEL_ROOT/Qwen2.5-VL-7B-Instruct
export WAN22_BASE_URL=<wan22-video-generation-endpoint>
export STAP_VLM_API_BASE=<vlm-judge-endpoint>
export CLIP_SERVER_URL=<clip-reward-endpoint>
export HPS_SERVER_URL=<hps-reward-endpoint>
```

Training data defaults to:

```text
data/train.parquet
data/val.parquet
```

## Evaluation and Inference

Evaluation files are under `evaluation/`.

Pure local inference:

```bash
bash evaluation/scripts/run_stap_pure_inference.sh \
  --wan22-base-url "$WAN22_BASE_URL"
```

API-based showcase inference:

```bash
WAN22_BASE_URLS="$WAN22_BASE_URL_1 $WAN22_BASE_URL_2" \
bash evaluation/scripts/run_stap_showcase_api.sh \
  --qwen-api-base "$STAP_QWEN_API_BASE"
```

Metric scripts are under:

```text
evaluation/scripts/eval_metric/
```

## Notes

This repository is anonymized. Local server addresses, private paths, and credentials are intentionally omitted. Configure all private endpoints through environment variables or command-line arguments.
