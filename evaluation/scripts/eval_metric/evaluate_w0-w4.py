import argparse
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(os.environ.get("STAP_PROJECT_ROOT", EVAL_ROOT.parent))
DATA_ROOT = EVAL_ROOT / "data/eval"
TRAJ_ROOT = EVAL_ROOT / "results/showcase_step50_eval"
DEFAULT_OUT = EVAL_ROOT / "table1/STAP_eval_metrics.json"
SCORE_FILE = PROJECT_ROOT / "workers/multimodal/reward/reward_score_mm/stap_score.py"


def load_score_module():
    import sys
    import types
    sys.path.insert(0, str(PROJECT_ROOT.parent))
    name = "STAP.workers.multimodal.reward.reward_score_mm.stap_format_score"
    shim = types.ModuleType(name)
    shim.format_reward = lambda prediction: (1.0, len(prediction))
    sys.modules[name] = shim
    import importlib.util
    spec = importlib.util.spec_from_file_location("stap_score_table1", SCORE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def text_from_sample(s):
    gt = s.get("reward_model", {}).get("ground_truth")
    if isinstance(gt, list) and gt and isinstance(gt[0], str):
        return gt[0]
    p = s.get("prompt", "")
    if isinstance(p, list):
        for item in p:
            if isinstance(item, dict) and isinstance(item.get("content"), str):
                return item["content"]
    return p if isinstance(p, str) else ""


def read_data():
    out = {}
    for p in sorted(DATA_ROOT.iterdir()):
        if p.is_dir() and (p / "data.json").is_file():
            rows = json.loads((p / "data.json").read_text())
        elif p.suffix == ".parquet":
            try:
                import pandas as pd
                rows = pd.read_parquet(p).to_dict("records")
            except Exception as e:
                print(f"[data] skip {p}: {e}")
                continue
        else:
            continue
        ds = p.name if p.is_file() else p.name
        out[ds] = {str(x.get("prompt_id")): x for x in rows if x.get("prompt_id")}
    return out


def load_json(p):
    try:
        return json.loads(p.read_text())
    except Exception as e:
        return {"_load_error": repr(e)}


def final_video(traj, sample_dir=None):
    direct = traj.get("video_path")
    if direct and Path(direct).is_file():
        return str(direct), traj.get("final_round", traj.get("round"))
    rounds = traj.get("rounds", [])
    for r in reversed(rounds if isinstance(rounds, list) else []):
        p = r.get("video_path") if isinstance(r, dict) else None
        if p and Path(p).is_file():
            return str(p), r.get("round")
    if sample_dir:
        videos = sorted(Path(sample_dir).glob("*.mp4"))
        if videos:
            return str(videos[-1]), None
    return None, None


def atomic_write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def decode_with_av(path, n_frames):
    import av
    import numpy as np
    container = av.open(str(path))
    frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    container.close()
    if not frames: return []
    idx = np.linspace(0, len(frames) - 1, min(n_frames, len(frames))).astype(int)
    return [np.asarray(frames[i]).astype(np.uint8) for i in idx]

def evaluate_one(item, score, args):
    ds, pid, traj, source, existing, jp = item
    prompt = text_from_sample(source) or traj.get("source_prompt", "") or traj.get("original_prompt", "")
    video, rnd = final_video(traj, jp.parent)
    rec = dict(existing)
    rec.update({"prompt_id": pid, "original_prompt": prompt, "video_path": video,
        "final_round": rnd, "status": "error", "r_Q": None, "r_CLIP": None,
        "r_HPS": None, "r_Q_per_layer": None, "w0": None, "w1": None,
        "w2": None, "w3": None, "w4": None, "r_Q_equal_mean": None,
        "vlm_usage": None, "errors": {}})
    for key in ("r_Q", "r_Q_per_layer", "w0", "w1", "w2", "w3", "w4", "r_Q_equal_mean", "vlm_usage"):
        if key in existing: rec[key] = existing[key]
    try:
        if not video: raise RuntimeError("no existing final-round video")
        primary = source.get("layer", {}) if isinstance(source, dict) else {}
        if isinstance(primary, dict): primary = primary.get("primary", "W0")
        primary = primary or pid.split("_", 1)[0]
        frames = score._decode_frames(video, args.judge_frames)
        if not frames: raise RuntimeError("video decode returned no frames")
        if args.metrics in ("rq", "all"):
            try:
                rec["r_Q"], rec["r_Q_per_layer"] = score.compute_rQ(video, prompt, primary)
                layers = rec["r_Q_per_layer"] or {}
                vals = [layers.get(w) for w in ("W0", "W1", "W2", "W3", "W4")]
                if not all(isinstance(v, (int, float)) for v in vals) or all(float(v) == 0.0 for v in vals):
                    raise RuntimeError("VLM judge output invalid: W0-W4 were missing, unparsable, or all zero")
                for w in ("W0", "W1", "W2", "W3", "W4"): rec[w.lower()] = layers[w]
                rec["r_Q_equal_mean"] = sum(rec[w.lower()] for w in ("W0", "W1", "W2", "W3", "W4")) / 5.0
                rec["vlm_usage"] = getattr(score, "_VLM_LAST_USAGE", {}) or {}
                rec["vlm_raw_response"] = getattr(score, "_VLM_LAST_RAW_RESPONSE", "")[:30000]
            except Exception as e:
                rec["errors"]["r_Q"] = repr(e)
                rec["r_Q"] = None
                rec["r_Q_per_layer"] = None
                for w in ("w0", "w1", "w2", "w3", "w4"): rec[w] = None
                rec["r_Q_equal_mean"] = None
        if args.metrics in ("clip_hps", "all"):
            try: rec["r_CLIP"] = score.compute_clip_frames(frames, prompt)
            except Exception as e: rec["errors"]["r_CLIP"] = repr(e)
            try: rec["r_HPS"] = score.compute_hps_frames(frames, prompt)
            except Exception as e: rec["errors"]["r_HPS"] = repr(e)
        rec["status"] = "ok" if not rec["errors"] else "partial_error"
    except Exception as e: rec["errors"]["sample"] = repr(e)
    return ds, pid, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--judge-frames", type=int, default=8)
    ap.add_argument("--metrics", choices=["rq", "clip_hps", "all"], default="rq")
    ap.add_argument("--trajectory-root", type=Path, default=TRAJ_ROOT)
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()
    os.environ.setdefault("CLIP_SERVER_URLS", "")
    os.environ.setdefault("HPS_SERVER_URLS", "")
    os.environ.setdefault("STAP_VLM_API_BASE", "")
    os.environ.setdefault("STAP_VLM_MODEL", "gpt-5.4")
    os.environ.setdefault("STAP_VLM_MAX_TOKENS", "8096")
    os.environ.setdefault("STAP_VLM_THINKING", "false")
    if not os.environ.get("STAP_VLM_API_KEY"):
        print("[warn] STAP_VLM_API_KEY is not set; authenticated VLM APIs may fail.", flush=True)
    os.environ.setdefault("STAP_VLM_API_MODE", "responses")
    os.environ["STAP_JUDGE_FRAMES"] = str(args.judge_frames)
    score = load_score_module()
    data = read_data()
    allowed = set(args.datasets) if args.datasets else None
    result = {"metadata": {
        "source_data": str(DATA_ROOT), "source_trajectories": str(args.trajectory_root),
        "metric_source": str(SCORE_FILE), "judge_frames": args.judge_frames,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metric_names": ["r_Q", "r_CLIP", "r_HPS"],
        "vlm_token_totals": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    }, "datasets": {}}
    if args.output.exists() and not args.force:
        try: result = json.loads(args.output.read_text())
        except Exception: pass
    if args.workers <= 0: raise ValueError("--workers must be positive")
    jobs = []
    for dsdir in sorted(args.trajectory_root.iterdir()):
        if not dsdir.is_dir() or (allowed and dsdir.name not in allowed): continue
        ds = dsdir.name; result["datasets"].setdefault(ds, {})
        rows = data.get(ds, {})
        json_files = list(dsdir.glob("[!_]*.json")) + list(dsdir.glob("*/[!_]*.json"))
        for jp in sorted(json_files):
            traj = load_json(jp)
            pid = str(traj.get("prompt_id") or jp.parent.name)
            existing = result["datasets"][ds].get(pid, {})
            valid_rq = existing.get("status") == "ok" and existing.get("r_Q") is not None and not all(float(existing.get(w, 0.0) or 0.0) == 0.0 for w in ("w0", "w1", "w2", "w3", "w4"))
            if not args.force and ((args.metrics == "rq" and valid_rq) or
                                   (args.metrics == "clip_hps" and existing.get("r_CLIP") is not None and existing.get("r_HPS") is not None) or
                                   (args.metrics == "all" and existing.get("status") == "ok")):
                continue
            if args.limit and len(jobs) >= args.limit: break
            jobs.append((ds, pid, traj, rows.get(pid, traj.get("sample_metadata", {})), existing, jp))
        if args.limit and len(jobs) >= args.limit: break
    print(f"[start] jobs={len(jobs)} workers={args.workers}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(evaluate_one, job, score, args) for job in jobs]
        for fut in as_completed(futures):
            ds, pid, rec = fut.result(); result["datasets"].setdefault(ds, {})[pid] = rec
            usage = rec.get("vlm_usage") or {}
            totals = result["metadata"].setdefault("vlm_token_totals", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
            for key in ("input_tokens", "output_tokens", "total_tokens"): totals[key] += int(usage.get(key, 0) or 0)
            atomic_write(args.output, result)
            print(f"[{time.strftime('%F %T')}] {ds}/{pid} status={rec['status']} r_Q={rec['r_Q']}", flush=True)
    atomic_write(args.output, result)
    print(f"DONE output={args.output}", flush=True)


if __name__ == "__main__": main()

