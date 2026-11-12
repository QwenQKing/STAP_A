# STAP pure inference

This directory contains a pure inference runner for the STAP loop:
Qwen2.5-VL proposes/refines `<generate>...</generate>` prompts, WAN22 generates videos,
and generated video frames are fed back to Qwen for the next round until `<complete>end</complete>`
or the generation limit is reached.

Quick dry run:

```bash
cd STAP
bash evaluation/scripts/run_stap_pure_inference.sh --dry-run --datasets aigve_60k --limit 2
```

Run a small real job:

```bash
cd STAP
CUDA_VISIBLE_DEVICES=0 bash evaluation/scripts/run_stap_pure_inference.sh \
  --datasets aigve_60k \
  --limit 1 \
  --batch-size 1 \
  --wan22-base-url "$WAN22_BASE_URL"
```

Defaults mirror `STAP/scripts/run_STAP_grpo.sh` where relevant:
- Qwen model: `$MODEL_ROOT/Qwen2.5-VL-7B-Instruct`
- WAN22 endpoint/model: set by `WAN22_BASE_URL`, `wan22-a14b`
- `max_gen_round=5`, `video_generate_limit=4`
- WAN22 video params: `512x320`, `80` frames, `8` fps, `40` steps, guidance `5.0`
- Qwen receives `4` sampled video frames per generated mp4

Outputs are written under:

```text
evaluation/outputs/<dataset>/<prompt_id>/
```

Each sample directory contains per-round mp4 files and `trajectory.json`; each dataset directory also gets a `manifest.jsonl`.



cd STAP

WAN22_BASE_URLS="$WAN22_BASE_URL_1 $WAN22_BASE_URL_2" \
nohup bash evaluation/scripts/run_stap_showcase_api.sh \
  --qwen-api-base "$STAP_QWEN_API_BASE" \
  --qwen-model STAP-qwen25-vl-step50 \
  --num-workers 2 \
  --output-root evaluation/outputs/showcase_step50_eval \
  > evaluation/outputs/showcase_step50_eval.log 2>&1 &
