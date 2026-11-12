from __future__ import annotations

import argparse
import base64
import builtins
import io
import hashlib
import json
import os
import pickle
import queue
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from PIL import Image


_ORIG_PRINT = builtins.print


def print(*args: Any, **kwargs: Any) -> None:
    from datetime import datetime

    prefix = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    _ORIG_PRINT(prefix, *args, **kwargs)


EVAL_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.environ.get("STAP_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
PACKAGE_PARENT = PROJECT_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from STAP.utils.tools.video_generate import call_video_generate

PROMPT_ROOT = Path(os.environ.get("STAP_PROMPT_ROOT", EVAL_ROOT / "prompts"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=str(EVAL_ROOT / "data/eval"))
    p.add_argument("--output-root", default=str(EVAL_ROOT / "outputs/showcase_step50"))
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument(
        "--sample-id-files",
        nargs="+",
        default=None,
        help="JSON selection files with a datasets mapping of dataset names to prompt IDs.",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--retry-failed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume mode: skip completed samples, retry failed or incomplete trajectory.json files by default.",
    )
    p.add_argument("--dry-run", action="store_true")

    p.add_argument("--qwen-api-base", default="")
    p.add_argument("--qwen-model", default="STAP-qwen25-vl-step50")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--qwen-timeout", type=int, default=600)

    p.add_argument(
        "--wan22-base-url",
        default=os.environ.get("WAN22_BASE_URL", ""),
        help="Wan2.2 endpoint. Required unless --wan22-base-urls is set.",
    )
    p.add_argument(
        "--wan22-base-urls",
        nargs="*",
        default=None,
        help=(
            "One or more Wan2.2 endpoints. When set, video generations are distributed round-robin. "
            "Required when --wan22-base-url is not set."
        ),
    )
    p.add_argument("--wan22-model", default="wan22-a14b")
    p.add_argument("--wan22-timeout", type=int, default=3600)
    p.add_argument("--wan22-poll-interval", type=float, default=10.0)
    p.add_argument("--n-frames-for-qwen", type=int, default=4)
    p.add_argument(
        "--feedback-frames",
        type=int,
        default=1,
        help="Number of decoded video frames sent back to Qwen as images. Keep 1 for the current agent server limit.",
    )
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=320)
    p.add_argument("--num-frames", type=int, default=80)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--num-inference-steps", type=int, default=40)
    p.add_argument("--guidance-scale", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--negative-prompt", default="low quality, blurry, distorted, watermark, text")

    p.add_argument("--max-gen-round", type=int, default=4)
    p.add_argument("--video-generate-limit", type=int, default=3)
    p.add_argument(
        "--num-workers",
        type=int,
        default=3,
        help="Number of samples to process concurrently. Use 2 to drive two Wan2.2 servers at once.",
    )
    p.add_argument("--round1-prompt", default=str(PROMPT_ROOT / "round_1_user_prompt_qwenvl_stap.pkl"))
    p.add_argument("--after-generate-prompt", default=str(PROMPT_ROOT / "user_prompt_after_video_generate_stap.pkl"))
    p.add_argument("--last-round-prompt", default=str(PROMPT_ROOT / "user_prompt_last_round_stap.pkl"))
    return p.parse_args()


def load_pickle_text(path: str) -> str:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_existing_result(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        obj = load_json(path)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def is_completed_result(obj: dict[str, Any] | None) -> bool:
    if not obj:
        return False
    return obj.get("success") is True and obj.get("complete") is True and obj.get("error") is None


def extract_prompt_text(sample: dict[str, Any]) -> str:
    content = sample.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    prompt = sample.get("prompt")
    if isinstance(prompt, list):
        for item in prompt:
            if isinstance(item, dict):
                text = item.get("content")
                if isinstance(text, str) and text.strip():
                    return text.strip()
    raise ValueError(f"sample {sample.get('prompt_id')} has no usable prompt text")


def dataset_dirs(data_root: Path, only: list[str] | None) -> list[Path]:
    dirs = [p for p in sorted(data_root.iterdir()) if p.is_dir() and (p / "data.json").exists()]
    if only is None:
        return dirs
    wanted = set(only)
    return [p for p in dirs if p.name in wanted]


def load_selected_prompt_ids(paths: list[str] | None) -> dict[str, set[str]] | None:
    if not paths:
        return None
    selected: dict[str, set[str]] = {}
    for raw_path in paths:
        path = Path(raw_path)
        selection = load_json(path)
        datasets = selection.get("datasets") if isinstance(selection, dict) else None
        if not isinstance(datasets, dict):
            raise ValueError(f"selection file has no datasets mapping: {path}")
        for dataset, prompt_ids in datasets.items():
            if not isinstance(prompt_ids, list):
                raise ValueError(f"selection file has non-list prompt IDs for {dataset}: {path}")
            selected.setdefault(str(dataset), set()).update(str(prompt_id) for prompt_id in prompt_ids)
    return selected


def last_tag(text: str, tag: str) -> str | None:
    matches = list(re.finditer(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL))
    return matches[-1].group(1).strip() if matches else None


def write_video(path: Path, mp4_bytes: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.mp4")
    tmp.write_bytes(mp4_bytes)
    tmp.replace(path)


def qwen_chat(cfg: argparse.Namespace, messages: list[dict[str, Any]]) -> str:
    if not cfg.qwen_api_base.strip():
        raise ValueError("Set --qwen-api-base before running inference.")
    api_messages = []
    last_index = len(messages) - 1
    for idx, message in enumerate(messages):
        if idx == last_index or isinstance(message.get("content"), str):
            api_messages.append(message)
        else:
            api_messages.append({"role": message["role"], "content": qwen_text_content(message)})
    payload = {
        "model": cfg.qwen_model,
        "messages": api_messages,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "max_tokens": cfg.max_tokens,
    }
    sess = requests.Session()
    sess.trust_env = False
    r = sess.post(f"{cfg.qwen_api_base.rstrip('/')}/v1/chat/completions", json=payload, timeout=cfg.qwen_timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def normalize_wan22_endpoints(cfg: argparse.Namespace) -> list[str]:
    urls = cfg.wan22_base_urls if cfg.wan22_base_urls else [cfg.wan22_base_url]
    normalized = []
    for url in urls:
        url = url.rstrip("/")
        if url and url not in normalized:
            normalized.append(url)
    if not normalized:
        raise ValueError("at least one Wan2.2 endpoint is required")
    return normalized


def next_wan22_endpoint(cfg: argparse.Namespace) -> str:
    return cfg._wan22_queue.get()


def release_wan22_endpoint(cfg: argparse.Namespace, endpoint: str) -> None:
    cfg._wan22_queue.put(endpoint)


def save_feedback_frames(video_path: Path, frames: list[Any], max_frames: int) -> list[Path]:
    frame_paths: list[Path] = []
    if not frames or max_frames <= 0:
        return frame_paths
    if len(frames) <= max_frames:
        indices = list(range(len(frames)))
    elif max_frames == 1:
        indices = [len(frames) // 2]
    else:
        indices = [round(i * (len(frames) - 1) / (max_frames - 1)) for i in range(max_frames)]

    for out_idx, frame_idx in enumerate(indices):
        img = Image.fromarray(frames[frame_idx]).convert("RGB")
        frame_path = video_path.with_name(f"{video_path.stem}_frame_{out_idx:02d}.jpg")
        img.save(frame_path, format="JPEG", quality=92)
        frame_paths.append(frame_path)
    return frame_paths


def image_feedback_content(frame_paths: list[Path], text: str) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for frame_path in frame_paths:
        b64 = base64.b64encode(frame_path.read_bytes()).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    content.append({"type": "text", "text": text})
    return content


def qwen_text_content(message: dict[str, Any]) -> str:
    content = message["content"]
    if isinstance(content, str):
        return content
    parts = []
    for item in content:
        if item.get("type") == "text":
            parts.append(item.get("text", ""))
        elif item.get("type") == "image_url":
            parts.append("[image_url: decoded feedback frame saved next to round video_path]")
        elif item.get("type") == "video_url":
            parts.append("[video_url: saved in round video_path]")
    return "\n".join(parts)


def sample_jobs(cfg: argparse.Namespace) -> list[tuple[str, int, dict[str, Any]]]:
    jobs = []
    selected = load_selected_prompt_ids(cfg.sample_id_files)
    allowed_datasets = cfg.datasets
    if selected is not None:
        selected_names = set(selected)
        allowed_datasets = sorted(selected_names if allowed_datasets is None else selected_names.intersection(allowed_datasets))
    for d in dataset_dirs(Path(cfg.data_root), allowed_datasets):
        samples = load_json(d / "data.json")
        if selected is not None:
            wanted = selected.get(d.name, set())
            samples = [sample for sample in samples if str(sample.get("prompt_id")) in wanted]
        if cfg.limit is not None:
            samples = samples[: cfg.limit]
        for idx, sample in enumerate(samples, start=1):
            jobs.append((d.name, idx, sample))
    return jobs


def run_one(cfg: argparse.Namespace, prompts: dict[str, str], dataset: str, index: int, sample: dict[str, Any]) -> None:
    prompt_id = str(sample.get("prompt_id") or f"{dataset}_{index:05d}")
    source_prompt = extract_prompt_text(sample)
    out_dir = Path(cfg.output_root) / dataset / prompt_id
    final_path = out_dir / "trajectory.json"
    existing = load_existing_result(final_path)
    if final_path.exists() and not cfg.force and is_completed_result(existing):
        rounds = len(existing.get("rounds", [])) if existing else "?"
        print(f"[{dataset}] skip completed {prompt_id} rounds={rounds}", flush=True)
        return
    if final_path.exists() and not cfg.force and not cfg.retry_failed:
        print(f"[{dataset}] skip existing incomplete/failed {prompt_id} (use --retry-failed or --force to rerun)", flush=True)
        return
    if final_path.exists() and not cfg.force and cfg.retry_failed:
        old_error = existing.get("error") if existing else "unreadable trajectory"
        print(f"[{dataset}] retry incomplete/failed {prompt_id}: {old_error}", flush=True)

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": prompts["round1"] + source_prompt}
    ]
    rounds: list[dict[str, Any]] = []
    generated = 0
    complete = False
    error = None

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
        user_message = messages[-1]
        record: dict[str, Any] = {
            "round": round_idx,
            "qwen_input_user": qwen_text_content(user_message),
        }
        try:
            assistant = qwen_chat(cfg, messages)
        except Exception as exc:
            error = f"qwen_api_error: {type(exc).__name__}: {exc}"
            record["error"] = error
            rounds.append(record)
            break

        messages.append({"role": "assistant", "content": assistant})
        generate_prompt = last_tag(assistant, "generate")
        complete_tag = last_tag(assistant, "complete")
        record.update(
            {
                "qwen_output": assistant,
                "generate_prompt": generate_prompt,
                "complete": complete_tag,
            }
        )

        if complete_tag or not generate_prompt:
            complete = bool(complete_tag)
            rounds.append(record)
            break

        if generated >= cfg.video_generate_limit or round_idx == cfg.max_gen_round - 1:
            error = "generation_limit_reached_before_complete"
            record["error"] = error
            rounds.append(record)
            break

        wan22_endpoint = next_wan22_endpoint(cfg)
        print(f"[{dataset}] {prompt_id} round={round_idx} wan22_start endpoint={wan22_endpoint}", flush=True)
        try:
            tool_text, frames, stat = call_video_generate(
                generate_prompt,
                base_url=wan22_endpoint,
                model=cfg.wan22_model,
                n_frames=cfg.n_frames_for_qwen,
                timeout_sec=cfg.wan22_timeout,
                poll_interval_sec=cfg.wan22_poll_interval,
                gen_params=gen_params,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            generated += 1
            record["wan22_endpoint"] = wan22_endpoint
            record["error"] = error
            rounds.append(record)
            print(f"[{dataset}] {prompt_id} round={round_idx} wan22_result endpoint={wan22_endpoint} success=False error={error}", flush=True)
            break
        finally:
            release_wan22_endpoint(cfg, wan22_endpoint)
        generated += 1
        record["wan22"] = {k: v for k, v in stat.items() if k != "mp4_bytes"}
        record["wan22_endpoint"] = wan22_endpoint

        if not stat.get("success") or not stat.get("mp4_bytes"):
            error = stat.get("error") or "wan22_failed"
            record["error"] = error
            rounds.append(record)
            print(f"[{dataset}] {prompt_id} round={round_idx} wan22_result endpoint={wan22_endpoint} success=False error={error}", flush=True)
            break

        print(f"[{dataset}] {prompt_id} round={round_idx} wan22_result endpoint={wan22_endpoint} success=True duration={stat.get('duration')}s", flush=True)
        h = hashlib.md5(generate_prompt.encode("utf-8")).hexdigest()[:8]
        video_path = out_dir / f"round_{round_idx:02d}_{h}.mp4"
        write_video(video_path, stat["mp4_bytes"])
        record["video_path"] = str(video_path)
        print(
            f"[{dataset}] {prompt_id} round={round_idx} wan22_saved "
            f"endpoint={wan22_endpoint} success=True video_path={video_path} "
            f"bytes={video_path.stat().st_size}",
            flush=True,
        )
        frame_paths = save_feedback_frames(video_path, frames, cfg.feedback_frames)
        record["feedback_frame_paths"] = [str(x) for x in frame_paths]
        rounds.append(record)

        next_round = round_idx + 1
        is_last = next_round >= cfg.max_gen_round - 1 or generated >= cfg.video_generate_limit
        next_prompt = prompts["last"] if is_last else prompts["after"]
        next_user_text = (
            "The video generated from your previous <generate> prompt is represented by the attached sampled frame(s). "
            "Use them to assess the generated video; the full mp4 is saved in the trajectory metadata.\n\n"
            + next_prompt
        )
        messages.append({"role": "user", "content": image_feedback_content(frame_paths, next_user_text)})

    result = {
        "dataset": dataset,
        "index": index,
        "prompt_id": prompt_id,
        "source_prompt": source_prompt,
        "sample_metadata": sample,
        "qwen_api_base": cfg.qwen_api_base,
        "qwen_model": cfg.qwen_model,
        "wan22_base_url": cfg.wan22_base_url,
        "wan22_base_urls": cfg._wan22_endpoints,
        "wan22_model": cfg.wan22_model,
        "success": error is None,
        "complete": complete,
        "error": error,
        "rounds": rounds,
    }
    dump_json(final_path, result)
    with (out_dir.parent / "manifest.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"[{dataset}] saved {prompt_id} rounds={len(rounds)} complete={complete} error={error}", flush=True)


def main() -> int:
    cfg = parse_args()
    cfg._wan22_endpoints = normalize_wan22_endpoints(cfg)
    cfg._wan22_queue = queue.Queue()
    for endpoint in cfg._wan22_endpoints:
        cfg._wan22_queue.put(endpoint)
    prompts = {
        "round1": load_pickle_text(cfg.round1_prompt),
        "after": load_pickle_text(cfg.after_generate_prompt),
        "last": load_pickle_text(cfg.last_round_prompt),
    }
    jobs = sample_jobs(cfg)
    pending = []
    completed = failed_existing = new_jobs = 0
    for dataset, index, sample in jobs:
        prompt_id = str(sample.get("prompt_id") or f"{dataset}_{index:05d}")
        final_path = Path(cfg.output_root) / dataset / prompt_id / "trajectory.json"
        existing = load_existing_result(final_path)
        if final_path.exists() and not cfg.force and is_completed_result(existing):
            completed += 1
            continue
        if final_path.exists() and not cfg.force and not cfg.retry_failed:
            failed_existing += 1
            continue
        if final_path.exists():
            failed_existing += 1
        else:
            new_jobs += 1
        pending.append((dataset, index, sample))
    print(
        f"jobs={len(jobs)} pending={len(pending)} completed_skip={completed} "
        f"existing_to_retry_or_skip={failed_existing} new={new_jobs} data_root={cfg.data_root}",
        flush=True,
    )
    print(f"wan22_endpoints={cfg._wan22_endpoints} num_workers={cfg.num_workers}", flush=True)
    if cfg.dry_run:
        for dataset, index, sample in pending[:20]:
            print(dataset, index, sample.get("prompt_id"), repr(extract_prompt_text(sample)[:120]), flush=True)
        return 0
    Path(cfg.output_root).mkdir(parents=True, exist_ok=True)
    if cfg.num_workers <= 1:
        for dataset, index, sample in pending:
            run_one(cfg, prompts, dataset, index, sample)
        return 0

    with ThreadPoolExecutor(max_workers=cfg.num_workers) as pool:
        futures = {
            pool.submit(run_one, cfg, prompts, dataset, index, sample): (dataset, sample.get("prompt_id"))
            for dataset, index, sample in pending
        }
        for future in as_completed(futures):
            dataset, prompt_id = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"[{dataset}] worker error {prompt_id}: {type(exc).__name__}: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

