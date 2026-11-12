from __future__ import annotations

import asyncio
import io
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import requests

STAP_T2V_BASE_URL = os.environ.get("STAP_T2V_BASE_URL", "")
STAP_T2V_MODEL = os.environ.get("STAP_T2V_MODEL", "wan22-a14b")
_DEFAULT_MAX_CONCURRENT = int(os.environ.get("STAP_T2V_MAX_CONCURRENT", "16"))
_DEFAULT_RETRIES = int(os.environ.get("STAP_T2V_RETRIES", "3"))

VIDEO_PLACEHOLDER = "<|vision_start|><|video_pad|><|vision_end|>"

DEFAULT_GEN = dict(
    num_frames=80, width=512, height=320, fps=8,
    num_inference_steps=40, guidance_scale=5.0,
    negative_prompt="low quality, blurry, distorted, watermark, text", seed=42,
)


def _form(prompt: str, model: str, gen_params: Dict[str, Any] | None) -> Dict[str, str]:
    gp = {**DEFAULT_GEN, **(gen_params or {})}
    return {
        "prompt": prompt, "model": model,
        "width": str(gp["width"]), "height": str(gp["height"]),
        "num_frames": str(gp["num_frames"]), "fps": str(gp["fps"]),
        "negative_prompt": gp["negative_prompt"],
        "num_inference_steps": str(gp["num_inference_steps"]),
        "guidance_scale": str(gp["guidance_scale"]), "seed": str(gp["seed"]),
    }


def _ok_text() -> str:
    return (
        "[Video Generation Results] "
        "The video has been successfully generated based on your prompt. "
        "The video is shown below:\n" + VIDEO_PLACEHOLDER + "\n"
    )


def _err_text(err: str) -> str:
    return f"[Video Generation Results] There is an error encountered: {err}\n"


def _decode_mp4_to_frames(mp4_bytes: bytes, n_frames: int) -> List[np.ndarray]:
    try:
        import imageio.v2 as imageio
        rdr = imageio.get_reader(io.BytesIO(mp4_bytes), format="mp4")
        frames = [np.asarray(f) for f in rdr]
        rdr.close()
    except Exception:
        import subprocess
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
            tf.write(mp4_bytes)
            path = tf.name
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", path],
                check=True, capture_output=True, text=True,
            ).stdout.strip()
            width, height = (int(x) for x in probe.split("x"))
            raw = subprocess.check_output(
                ["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo",
                 "-pix_fmt", "rgb24", "-"], stderr=subprocess.STDOUT,
            )
            size = width * height * 3
            frames = [np.frombuffer(raw[i:i + size], dtype=np.uint8).reshape(height, width, 3)
                      for i in range(0, len(raw) - size + 1, size)]
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
    if not frames:
        return []
    arr = np.asarray(frames)
    idx = np.linspace(0, len(arr) - 1, min(n_frames, len(arr))).astype(int)
    return [np.asarray(arr[i]).astype(np.uint8) for i in idx]


def call_video_generate(
    prompt: str,
    base_url: str | None = None,
    model: str | None = None,
    n_frames: int = 4,
    timeout_sec: int = 600,
    poll_interval_sec: float = 2.0,
    gen_params: Dict[str, Any] | None = None,
) -> Tuple[str, List[np.ndarray], Dict]:
    base = (base_url or STAP_T2V_BASE_URL).rstrip("/")
    if not base:
        raise ValueError("Set STAP_T2V_BASE_URL or pass base_url.")
    model = model or STAP_T2V_MODEL
    t0 = time.time()
    tool_stat = {"success": False, "prompt": prompt, "model": model, "n_frames": 0,
                 "mp4_bytes": None, "duration": None, "error": None}
    sess = requests.Session()
    sess.trust_env = False
    try:
        r = sess.post(f"{base}/v1/videos", data=_form(prompt, model, gen_params), timeout=60)
        r.raise_for_status()
        job = r.json()
        job_id = job.get("id") or job.get("video_id")
        if not job_id:
            raise ValueError(f"submit response missing id: {job}")
        deadline = time.time() + timeout_sec
        status = None
        while time.time() < deadline:
            pr = sess.get(f"{base}/v1/videos/{job_id}", timeout=30)
            pr.raise_for_status()
            status = pr.json().get("status")
            if status == "completed":
                break
            if status == "failed":
                raise RuntimeError(f"job {job_id} failed: {pr.json().get('error', 'unknown')}")
            time.sleep(poll_interval_sec)
        if status != "completed":
            raise TimeoutError(f"job {job_id} not completed within {timeout_sec}s")
        cr = sess.get(f"{base}/v1/videos/{job_id}/content", timeout=timeout_sec)
        cr.raise_for_status()
        mp4 = cr.content
        if not mp4 or len(mp4) < 100 or b"ftyp" not in mp4[:32]:
            raise ValueError(f"invalid mp4 ({len(mp4)} bytes)")
        frames = [] if n_frames <= 0 else _decode_mp4_to_frames(mp4, n_frames)
        if n_frames > 0 and not frames:
            raise ValueError("decoded 0 frames from mp4")
        tool_stat.update(success=True, n_frames=len(frames), mp4_bytes=mp4,
                         duration=round(time.time() - t0, 2))
        return _ok_text(), frames, tool_stat
    except Exception as e:
        tool_stat["error"] = f"{type(e).__name__}: {e}"
        tool_stat["duration"] = round(time.time() - t0, 2)
        return _err_text(tool_stat["error"]), [], tool_stat


async def _async_one(session, sem, prompt, base, model, n_frames, timeout_sec,
                     poll_interval_sec, gen_params, retries):
    import aiohttp
    async with sem:
        for attempt in range(retries):
            t0 = time.time()
            stat = {"success": False, "prompt": prompt, "model": model, "n_frames": 0,
                    "mp4_bytes": None, "duration": None, "error": None}
            try:
                async with session.post(f"{base}/v1/videos", data=_form(prompt, model, gen_params),
                                        timeout=aiohttp.ClientTimeout(total=60)) as r:
                    if r.status != 200:
                        raise RuntimeError(f"submit HTTP {r.status}: {(await r.text())[:120]}")
                    job = await r.json()
                job_id = job.get("id") or job.get("video_id")
                if not job_id:
                    raise ValueError(f"submit missing id: {job}")
                deadline = time.time() + timeout_sec
                status = None
                while time.time() < deadline:
                    async with session.get(f"{base}/v1/videos/{job_id}",
                                           timeout=aiohttp.ClientTimeout(total=30)) as pr:
                        pj = await pr.json()
                    status = pj.get("status")
                    if status == "completed":
                        break
                    if status == "failed":
                        raise RuntimeError(f"job {job_id} failed: {pj.get('error', 'unknown')}")
                    await asyncio.sleep(poll_interval_sec)
                if status != "completed":
                    raise TimeoutError(f"job {job_id} not completed within {timeout_sec}s")
                async with session.get(f"{base}/v1/videos/{job_id}/content",
                                       timeout=aiohttp.ClientTimeout(total=timeout_sec)) as cr:
                    mp4 = await cr.read()
                if not mp4 or len(mp4) < 100 or b"ftyp" not in mp4[:32]:
                    raise ValueError(f"invalid mp4 ({len(mp4)} bytes)")
                frames = [] if n_frames <= 0 else _decode_mp4_to_frames(mp4, n_frames)
                if n_frames > 0 and not frames:
                    raise ValueError("decoded 0 frames from mp4")
                stat.update(success=True, n_frames=len(frames), mp4_bytes=mp4,
                            duration=round(time.time() - t0, 2))
                return _ok_text(), frames, stat
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                stat["error"] = err
                stat["duration"] = round(time.time() - t0, 2)
                return _err_text(err), [], stat


async def _async_batch(prompts, base, model, n_frames, timeout_sec, poll_interval_sec,
                       gen_params, max_concurrent, retries):
    import aiohttp
    sem = asyncio.Semaphore(max_concurrent)
    async with aiohttp.ClientSession() as session:
        tasks = [_async_one(session, sem, p, base, model, n_frames, timeout_sec,
                            poll_interval_sec, gen_params, retries) for p in prompts]
        return await asyncio.gather(*tasks)


def call_video_generate_batch(
    prompts: List[str],
    base_url: str | None = None,
    model: str | None = None,
    n_frames: int = 4,
    timeout_sec: int = 600,
    poll_interval_sec: float = 2.0,
    gen_params: Dict[str, Any] | None = None,
    max_concurrent: int = _DEFAULT_MAX_CONCURRENT,
    retries: int = _DEFAULT_RETRIES,
) -> List[Tuple[str, List[np.ndarray], Dict]]:
    base = (base_url or STAP_T2V_BASE_URL).rstrip("/")
    if not base:
        raise ValueError("Set STAP_T2V_BASE_URL or pass base_url.")
    model = model or STAP_T2V_MODEL
    if not prompts:
        return []

    try:
        import aiohttp
        coro = _async_batch(prompts, base, model, n_frames, timeout_sec,
                            poll_interval_sec, gen_params, max_concurrent, retries)
        try:
            asyncio.get_running_loop()
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=1) as ex:
                return ex.submit(asyncio.run, coro).result()
        except RuntimeError:
            return asyncio.run(coro)
    except Exception as e:
        print(f"[call_video_generate_batch] async path unavailable ({type(e).__name__}: {e}); "
              f"falling back to thread pool")
        from concurrent.futures import ThreadPoolExecutor
        outs: List = [None] * len(prompts)
        workers = max(1, min(max_concurrent, len(prompts)))
        with ThreadPoolExecutor(workers) as pool:
            fut = {pool.submit(call_video_generate, p, base, model, n_frames,
                               timeout_sec, poll_interval_sec, gen_params): i
                   for i, p in enumerate(prompts)}
            for f in fut:
                outs[fut[f]] = f.result()
        return outs


if __name__ == "__main__":
    import sys
    ps = sys.argv[1:] or [
        "a calm ocean at sunset, gentle waves rolling slowly toward the shore",
        "a red sports car driving fast on a coastal road",
    ]
    t0 = time.time()
    res = call_video_generate_batch(ps, model="wan22-a14b",
                                    n_frames=4, max_concurrent=8)
    print(f"batch of {len(ps)} done in {round(time.time()-t0,1)}s")
    for (txt, frames, st) in res:
        print("  success:", st["success"], "| n_frames:", st["n_frames"], "| err:", st["error"])

