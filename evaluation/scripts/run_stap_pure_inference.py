from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

EVAL_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.environ.get("STAP_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
PACKAGE_PARENT = PROJECT_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from STAP.utils.tools.video_generate import call_video_generate_batch

PROMPT_ROOT = Path(os.environ.get("STAP_PROMPT_ROOT", EVAL_ROOT / "prompts"))
VIDEO_PLACEHOLDER_TOKEN_ID = 151656
QWEN_IM_END_TOKEN_ID = 151645
MAX_QWEN_TOKEN_ID = 151664


@dataclass
class Trajectory:
    dataset: str
    prompt_id: str
    index: int
    source_prompt: str
    out_dir: Path
    prompt_token_ids: list[int]
    multi_modal_data: dict[str, list[np.ndarray]] = field(default_factory=lambda: {"video": []})
    generated_videos: int = 0
    rounds: list[dict[str, Any]] = field(default_factory=list)
    final_response: str | None = None
    complete: bool = False
    error: str | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=str(EVAL_ROOT / "data/eval"))
    p.add_argument("--output-root", default=str(EVAL_ROOT / "outputs"))
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")

    p.add_argument("--model-path", default=str(PROJECT_ROOT / "models/Qwen2.5-VL-7B-Instruct"))
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--limit-mm-video", type=int, default=4)
    p.add_argument("--dtype", default="bfloat16")

    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=-1)
    p.add_argument("--max-tokens", type=int, default=2048)

    p.add_argument("--max-gen-round", type=int, default=5)
    p.add_argument("--video-generate-limit", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=1, help="Qwen trajectory batch size.")

    p.add_argument("--wan22-base-url", default=os.environ.get("WAN22_BASE_URL", ""))
    p.add_argument("--wan22-model", default="wan22-a14b")
    p.add_argument("--wan22-max-concurrent", type=int, default=1)
    p.add_argument("--wan22-timeout", type=int, default=3600)
    p.add_argument("--wan22-poll-interval", type=float, default=10.0)
    p.add_argument("--n-frames-for-qwen", type=int, default=4)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=320)
    p.add_argument("--num-frames", type=int, default=80)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--num-inference-steps", type=int, default=40)
    p.add_argument("--guidance-scale", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--negative-prompt", default="low quality, blurry, distorted, watermark, text")

    p.add_argument("--round1-prompt", default=str(PROMPT_ROOT / "round_1_user_prompt_qwenvl_stap.pkl"))
    p.add_argument("--after-generate-prompt", default=str(PROMPT_ROOT / "user_prompt_after_video_generate_stap.pkl"))
    p.add_argument("--last-round-prompt", default=str(PROMPT_ROOT / "user_prompt_last_round_stap.pkl"))
    return p.parse_args()


def load_pickle_text(path: str) -> str:
    with open(path, "rb") as f:
        value = pickle.load(f)
    if not isinstance(value, str):
        raise TypeError(f"{path} did not contain a string")
    return value


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def extract_prompt_text(sample: dict[str, Any]) -> str:
    content = sample.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    prompt = sample.get("prompt")
    if isinstance(prompt, list):
        for item in prompt:
            if isinstance(item, dict) and isinstance(item.get("content"), str) and item["content"].strip():
                return item["content"].strip()
    raise ValueError(f"sample {sample.get('prompt_id')} has no usable prompt text")


def dataset_dirs(data_root: Path, only: list[str] | None) -> list[Path]:
    dirs = [p for p in sorted(data_root.iterdir()) if p.is_dir() and (p / "data.json").exists()]
    if only is None:
        return dirs
    wanted = set(only)
    return [p for p in dirs if p.name in wanted]


def encode_user_assistant(tokenizer: Any, user_text: str) -> list[int]:
    text = f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
    return tokenizer.encode(text)


def ensure_im_end(token_ids: list[int]) -> list[int]:
    filtered = [int(t) for t in token_ids if int(t) <= MAX_QWEN_TOKEN_ID]
    if not filtered:
        return [QWEN_IM_END_TOKEN_ID]
    if QWEN_IM_END_TOKEN_ID not in filtered:
        filtered[-1] = QWEN_IM_END_TOKEN_ID
    return filtered


def last_tag(text: str, tag: str) -> str | None:
    matches = list(re.finditer(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def write_video(path: Path, mp4_bytes: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.mp4")
    tmp.write_bytes(mp4_bytes)
    tmp.replace(path)


def build_trajectories(cfg: argparse.Namespace, tokenizer: Any | None, round1_prompt: str) -> list[Trajectory]:
    trajectories: list[Trajectory] = []
    for d in dataset_dirs(Path(cfg.data_root), cfg.datasets):
        samples = load_json(d / "data.json")
        if cfg.limit is not None:
            samples = samples[: cfg.limit]
        for idx, sample in enumerate(samples, start=1):
            prompt_id = str(sample.get("prompt_id") or f"{d.name}_{idx:05d}")
            out_dir = Path(cfg.output_root) / d.name / prompt_id
            final_path = out_dir / "trajectory.json"
            if final_path.exists() and not cfg.force:
                print(f"[{d.name}] skip existing {prompt_id}", flush=True)
                continue
            source = extract_prompt_text(sample)
            first_user = round1_prompt + source
            prompt_token_ids = encode_user_assistant(tokenizer, first_user) if tokenizer is not None else []
            trajectories.append(Trajectory(d.name, prompt_id, idx, source, out_dir, prompt_token_ids))
    return trajectories


def qwen_inputs(batch: list[Trajectory]) -> list[dict[str, Any]]:
    inputs = []
    for t in batch:
        item: dict[str, Any] = {"prompt_token_ids": t.prompt_token_ids}
        if t.multi_modal_data["video"]:
            item["multi_modal_data"] = {"video": t.multi_modal_data["video"]}
        inputs.append(item)
    return inputs


def save_result(t: Trajectory) -> None:
    record = {
        "dataset": t.dataset,
        "prompt_id": t.prompt_id,
        "source_prompt": t.source_prompt,
        "success": t.error is None,
        "complete": t.complete,
        "error": t.error,
        "final_response": t.final_response,
        "rounds": t.rounds,
    }
    dump_json(t.out_dir / "trajectory.json", record)
    manifest = t.out_dir.parent / "manifest.jsonl"
    with manifest.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_batch(cfg: argparse.Namespace, llm: Any, tokenizer: Any, batch: list[Trajectory], after_prompt: str, last_prompt: str) -> None:
    from vllm import SamplingParams

    sampling = SamplingParams(
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        max_tokens=cfg.max_tokens,
        stop_token_ids=[QWEN_IM_END_TOKEN_ID],
    )
    active = list(batch)
    gen_params = {
        "width": cfg.width,
        "height": cfg.height,
        "num_frames": cfg.num_frames,
        "fps": cfg.fps,
        "negative_prompt": cfg.negative_prompt,
        "num_inference_steps": cfg.num_inference_steps,
        "guidance_scale": cfg.guidance_scale,
        "seed": cfg.seed,
    }

    for round_idx in range(cfg.max_gen_round):
        if not active:
            break
        print(f"[qwen] round={round_idx} active={len(active)}", flush=True)
        outputs = llm.generate(qwen_inputs(active), sampling, use_tqdm=False)
        requests: list[tuple[Trajectory, str, str]] = []

        for t, out in zip(active, outputs):
            token_ids = ensure_im_end(list(out.outputs[0].token_ids))
            t.prompt_token_ids.extend(token_ids)
            response = tokenizer.decode(token_ids, skip_special_tokens=True)
            t.final_response = response
            generate_prompt = last_tag(response, "generate")
            complete = last_tag(response, "complete")
            round_record = {"round": round_idx, "assistant": response, "generate_prompt": generate_prompt, "complete": complete}
            t.rounds.append(round_record)

            if generate_prompt and not complete:
                if t.generated_videos >= cfg.video_generate_limit or round_idx == cfg.max_gen_round - 1:
                    t.complete = False
                    t.error = "generation_limit_reached_before_complete"
                    continue
                requests.append((t, generate_prompt, response))
            else:
                t.complete = bool(complete)

        active = [t for t, _, _ in requests]
        if not active:
            break

        print(f"[wan22] round={round_idx} jobs={len(requests)}", flush=True)
        results = call_video_generate_batch(
            prompts=[p for _, p, _ in requests],
            base_url=cfg.wan22_base_url,
            model=cfg.wan22_model,
            n_frames=cfg.n_frames_for_qwen,
            timeout_sec=cfg.wan22_timeout,
            poll_interval_sec=cfg.wan22_poll_interval,
            gen_params=gen_params,
            max_concurrent=cfg.wan22_max_concurrent,
        )

        next_active: list[Trajectory] = []
        for (t, prompt, _), (tool_text, frames, stat) in zip(requests, results):
            t.generated_videos += 1
            h = hashlib.md5(prompt.encode("utf-8")).hexdigest()[:8]
            video_path = t.out_dir / f"round_{round_idx:02d}_{h}.mp4"
            if stat.get("success") and stat.get("mp4_bytes"):
                write_video(video_path, stat["mp4_bytes"])
                t.rounds[-1]["video_path"] = str(video_path)
                t.rounds[-1]["wan22"] = {k: v for k, v in stat.items() if k != "mp4_bytes"}
            else:
                t.error = stat.get("error") or "wan22_failed"
                t.rounds[-1]["wan22"] = {k: v for k, v in stat.items() if k != "mp4_bytes"}
                continue

            next_round = round_idx + 1
            is_last = next_round >= cfg.max_gen_round - 1 or t.generated_videos >= cfg.video_generate_limit
            followup = last_prompt if is_last else after_prompt
            next_user = tool_text + followup if "[Video Generation Results]" in tool_text and "There is an error" not in tool_text else tool_text
            t.prompt_token_ids.extend(encode_user_assistant(tokenizer, next_user))
            if frames:
                t.multi_modal_data["video"].append(np.asarray(frames, dtype=np.uint8))
            next_active.append(t)
        active = next_active

    for t in batch:
        if not t.complete and t.error is None:
            t.error = "max_round_reached_without_complete"
        save_result(t)
        print(f"[{t.dataset}] saved {t.prompt_id} complete={t.complete} error={t.error}", flush=True)


def main() -> int:
    cfg = parse_args()
    Path(cfg.output_root).mkdir(parents=True, exist_ok=True)
    round1 = load_pickle_text(cfg.round1_prompt)
    after = load_pickle_text(cfg.after_generate_prompt)
    last = load_pickle_text(cfg.last_round_prompt)

    if cfg.dry_run:
        trajectories = build_trajectories(cfg, None, round1)
        print(f"pending trajectories: {len(trajectories)}", flush=True)
        return 0

    try:
        from transformers import AutoProcessor, AutoTokenizer
        from vllm import LLM
    except ModuleNotFoundError as exc:
        raise RuntimeError("Pure inference requires transformers and vllm in the runtime environment.") from exc

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
    trajectories = build_trajectories(cfg, tokenizer, round1)
    print(f"pending trajectories: {len(trajectories)}", flush=True)
    if not trajectories:
        return 0

    AutoProcessor.from_pretrained(cfg.model_path, trust_remote_code=True)
    llm = LLM(
        model=cfg.model_path,
        tensor_parallel_size=cfg.tensor_parallel_size,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        max_model_len=cfg.max_model_len,
        dtype=cfg.dtype,
        trust_remote_code=True,
        limit_mm_per_prompt={"video": cfg.limit_mm_video},
        enforce_eager=False,
        enable_chunked_prefill=False,
    )

    for start in range(0, len(trajectories), cfg.batch_size):
        run_batch(cfg, llm, tokenizer, trajectories[start : start + cfg.batch_size], after, last)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130)

