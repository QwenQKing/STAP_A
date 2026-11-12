import argparse
import json
import os
import sys
from pathlib import Path


_DOVER_CACHE = {}


def dover_score(path, device, dover_root):
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("DOVER evaluation requires torch in the runtime environment.") from exc
    sys.path.insert(0, str(dover_root))
    import yaml
    from dover.datasets import UnifiedFrameSampler, spatial_temporal_view_decomposition
    from dover.models import DOVER

    cfg_path = dover_root / "dover.yml"
    with cfg_path.open() as f:
        opt = yaml.safe_load(f)

    dev = torch.device(device)
    cache_key = (str(dover_root.resolve()), str(dev))
    if cache_key not in _DOVER_CACHE:
        evaluator = DOVER(**opt["model"]["args"]).to(dev).eval()
        weight_path = Path(opt["test_load_path"])
        if not weight_path.is_absolute():
            weight_path = dover_root / weight_path
        if not weight_path.exists():
            flat_weight = dover_root.parent / weight_path.name
            if flat_weight.exists():
                weight_path = flat_weight
        evaluator.load_state_dict(torch.load(weight_path.resolve(), map_location=dev))

        dopt = opt["data"]["val-l1080p"]["args"]
        samplers = {}
        for key, spec in dopt["sample_types"].items():
            if "t_frag" in spec:
                samplers[key] = UnifiedFrameSampler(
                    spec["clip_len"] // spec["t_frag"],
                    spec["t_frag"],
                    spec["frame_interval"],
                    spec["num_clips"],
                )
            else:
                samplers[key] = UnifiedFrameSampler(
                    spec["clip_len"],
                    spec["num_clips"],
                    spec["frame_interval"],
                )
        _DOVER_CACHE[cache_key] = (evaluator, dopt, samplers)

    evaluator, dopt, samplers = _DOVER_CACHE[cache_key]
    views, _ = spatial_temporal_view_decomposition(str(path), dopt["sample_types"], samplers)
    mean = torch.tensor([123.675, 116.28, 103.53], device=dev)
    std = torch.tensor([58.395, 57.12, 57.375], device=dev)

    for key, val in views.items():
        n = dopt["sample_types"][key].get("num_clips", 1)
        views[key] = (
            ((val.permute(1, 2, 3, 0).to(dev) - mean) / std)
            .permute(3, 0, 1, 2)
            .reshape(val.shape[0], n, -1, *val.shape[2:])
            .transpose(0, 1)
        )

    with torch.inference_mode():
        result = [float(x.mean().item()) for x in evaluator(views)]
    return {"dover_technical": result[0], "dover_aesthetic": result[1]}


def dover_scores_batch(paths, device, dover_root):
    return [dover_score(path, device, dover_root) for path in paths]


def load_records(output):
    if not output.exists():
        return {}
    with output.open() as f:
        return json.load(f)


def collect_videos(root, input_list):
    if input_list:
        with Path(input_list).open() as f:
            return [Path(line.strip()) for line in f if line.strip()]
    return sorted(root.rglob("*.mp4"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", required=True)
    ap.add_argument("--input-list", default="", help="optional newline-delimited video path manifest")
    ap.add_argument("--output", required=True)
    ap.add_argument("--dover-root", default="EVAL/eval_model/DOVER")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    root = Path(args.input_root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = load_records(output)

    videos = collect_videos(root, args.input_list)
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard-index/num-shards")
    videos = [v for i, v in enumerate(videos) if i % args.num_shards == args.shard_index]
    if args.limit:
        videos = videos[: args.limit]

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("DOVER evaluation requires torch in the runtime environment.") from exc
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    pending = [v for v in videos if records.get(str(v.relative_to(root)), {}).get("status") != "ok"]

    for start in range(0, len(pending), max(1, args.batch_size)):
        batch = pending[start : start + max(1, args.batch_size)]
        try:
            dover = dover_scores_batch(batch, device, Path(args.dover_root))
        except Exception as exc:
            dover = [{"status": "error", "error": f"{type(exc).__name__}: {exc}"} for _ in batch]

        for video, dover_rec in zip(batch, dover):
            key = str(video.relative_to(root))
            rec = {"video": str(video), "status": "ok"}
            try:
                rec.update(dover_rec)
            except Exception as exc:
                rec["status"] = "error"
                rec["error"] = f"{type(exc).__name__}: {exc}"
            records[key] = rec
            print(key, rec["status"], flush=True)

        tmp = output.with_suffix(output.suffix + ".tmp")
        with tmp.open("w") as f:
            json.dump(records, f, ensure_ascii=True, indent=2)
        os.replace(tmp, output)


if __name__ == "__main__":
    main()
