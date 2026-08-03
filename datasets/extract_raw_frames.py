#!/usr/bin/env python3
"""Extract raw video files into PRVR raw-frame directories.

This prepares frames shared by CLIP4Clip raw-frame experiments and candidate
mining. TVR is excluded because TVQA provides pre-extracted 3fps HQ frames.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm


DATASETS = ("activitynet", "charades", "msrvtt")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".avi", ".mov")
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("PRVR_DATA_ROOT", REPO_ROOT / "datasets"))


def parse_frame_rate(rate: Optional[str]) -> Optional[float]:
    if not rate or rate == "0/0":
        return None
    if "/" not in rate:
        value = float(rate)
        return value if value > 0 else None
    numerator, denominator = rate.split("/", 1)
    denominator_value = float(denominator)
    if denominator_value == 0:
        return None
    value = float(numerator) / denominator_value
    return value if value > 0 else None


def probe_video(video_path: Path) -> Dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_frames,duration,avg_frame_rate,r_frame_rate:format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    data = json.loads(result.stdout)
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError(f"no video stream found: {video_path}")
    stream = streams[0]
    duration = stream.get("duration")
    if duration in (None, "N/A"):
        duration = (data.get("format") or {}).get("duration")
    nb_frames = stream.get("nb_frames")
    return {
        "width": int(stream["width"]) if stream.get("width") is not None else None,
        "height": int(stream["height"]) if stream.get("height") is not None else None,
        "duration": float(duration) if duration not in (None, "N/A") else None,
        "source_fps": (
            parse_frame_rate(stream.get("avg_frame_rate"))
            or parse_frame_rate(stream.get("r_frame_rate"))
        ),
        "source_frames": int(nb_frames) if nb_frames not in (None, "N/A") else None,
    }


def normalize_video_id(dataset: str, video_path: Path) -> str:
    video_id = video_path.stem
    if dataset == "activitynet" and not video_id.startswith("v_"):
        return f"v_{video_id}"
    return video_id


def discover_videos(video_root: Path, dataset: str) -> List[Tuple[str, Path]]:
    if not video_root.exists():
        raise FileNotFoundError(f"raw video root does not exist: {video_root}")

    videos: Dict[str, Path] = {}
    for path in sorted(video_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        video_id = normalize_video_id(dataset, path)
        videos.setdefault(video_id, path)
    return sorted(videos.items())


def read_video_ids(path: Optional[str]) -> Optional[set[str]]:
    if not path:
        return None
    ids: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            value = line.strip()
            if value:
                ids.add(value)
    return ids


def frame_paths(output_dir: Path) -> List[Path]:
    paths = [
        Path(path)
        for path in glob.glob(str(output_dir / "*.jpg"))
    ]
    paths.sort()
    return paths


def build_filter(fps: float, short_side: int) -> str:
    filters = [f"fps={fps:g}"]
    if short_side > 0:
        filters.append(
            "scale='if(gt(iw,ih),-2,%d)':'if(gt(iw,ih),%d,-2)'"
            % (short_side, short_side)
        )
    return ",".join(filters)


def extract_one(
    dataset: str,
    video_id: str,
    video_path: Path,
    output_root: Path,
    fps: float,
    short_side: int,
    jpeg_quality: int,
    overwrite: bool,
    ffmpeg_threads: int,
    dry_run: bool,
) -> Dict[str, Any]:
    output_dir = output_root / video_id
    existing = frame_paths(output_dir) if output_dir.exists() else []
    if existing and not overwrite:
        info = probe_video(video_path)
        return {
            "dataset": dataset,
            "video_id": video_id,
            "video_path": str(video_path),
            "output_dir": str(output_dir),
            "target_fps": fps,
            "short_side": short_side,
            "status": "exists",
            "existing_frames": len(existing),
            **info,
        }

    info = probe_video(video_path)
    if dry_run:
        return {
            "dataset": dataset,
            "video_id": video_id,
            "video_path": str(video_path),
            "output_dir": str(output_dir),
            "target_fps": fps,
            "short_side": short_side,
            "status": "dry_run",
            **info,
        }

    if overwrite and output_dir.exists():
        for path in frame_paths(output_dir):
            path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-threads",
        str(ffmpeg_threads),
        "-y",
        "-i",
        str(video_path),
        "-vf",
        build_filter(fps, short_side),
        "-q:v",
        str(jpeg_quality),
        str(output_dir / "%06d.jpg"),
    ]
    subprocess.run(cmd, check=True)
    written = frame_paths(output_dir)
    if not written:
        raise RuntimeError(f"ffmpeg produced no frames for {video_path}")

    return {
        "dataset": dataset,
        "video_id": video_id,
        "video_path": str(video_path),
        "output_dir": str(output_dir),
        "target_fps": fps,
        "short_side": short_side,
        "status": "written",
        "written_frames": len(written),
        **info,
    }


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]], append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument(
        "--video-root",
        default=None,
        help="Defaults to $PRVR_DATA_ROOT/<dataset>/raw_videos.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Defaults to $PRVR_DATA_ROOT/<dataset>/raw_frames.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.5,
        help="Target extraction FPS. 1.5 matches the prepared CLIP-B/32 video feature lengths.",
    )
    parser.add_argument(
        "--short-side",
        type=int,
        default=224,
        help="Resize the shorter image side to this value. Use 0 to keep source resolution.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ffmpeg-threads", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--video-id-list",
        default=None,
        help="Optional text file with one video_id per line.",
    )
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.short_side < 0:
        raise ValueError("--short-side must be >= 0")
    if args.jpeg_quality < 1:
        raise ValueError("--jpeg-quality must be >= 1")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.ffmpeg_threads < 1:
        raise ValueError("--ffmpeg-threads must be >= 1")

    dataset_root = DATA_ROOT / args.dataset
    if args.video_root is None:
        args.video_root = str(dataset_root / "raw_videos")
    if args.output_root is None:
        args.output_root = str(dataset_root / "raw_frames")
    if args.manifest is None:
        args.manifest = str(Path(args.output_root) / "manifest.jsonl")
    return args


def main() -> None:
    args = parse_args()
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("ffmpeg was not found in PATH")
    if shutil.which("ffprobe") is None:
        raise FileNotFoundError("ffprobe was not found in PATH")

    video_root = Path(args.video_root)
    output_root = Path(args.output_root)
    requested_ids = read_video_ids(args.video_id_list)
    videos = discover_videos(video_root, args.dataset)
    if requested_ids is not None:
        videos = [(video_id, path) for video_id, path in videos if video_id in requested_ids]
    if args.limit is not None:
        videos = videos[: args.limit]
    if not videos:
        raise FileNotFoundError(f"no videos found under {video_root}")

    summary = {
        "dataset": args.dataset,
        "video_root": str(video_root),
        "output_root": str(output_root),
        "manifest": args.manifest,
        "videos": len(videos),
        "fps": args.fps,
        "short_side": args.short_side,
        "workers": args.workers,
        "ffmpeg_threads": args.ffmpeg_threads,
        "dry_run": args.dry_run,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_to_video = {
            pool.submit(
                extract_one,
                args.dataset,
                video_id,
                video_path,
                output_root,
                args.fps,
                args.short_side,
                args.jpeg_quality,
                args.overwrite,
                args.ffmpeg_threads,
                args.dry_run,
            ): (video_id, video_path)
            for video_id, video_path in videos
        }

        with tqdm(total=len(future_to_video), desc="extracting raw frames", unit="video", dynamic_ncols=True) as pbar:
            for future in as_completed(future_to_video):
                video_id, video_path = future_to_video[future]
                try:
                    records.append(future.result())
                except Exception as exc:
                    failed.append(
                        {
                            "dataset": args.dataset,
                            "video_id": video_id,
                            "video_path": str(video_path),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                pbar.update(1)

    write_jsonl(manifest_path, sorted(records, key=lambda item: item["video_id"]), append=False)
    if failed:
        failure_path = manifest_path.with_suffix(".failed.jsonl")
        write_jsonl(failure_path, sorted(failed, key=lambda item: item["video_id"]), append=False)
        raise RuntimeError(f"{len(failed)} videos failed; see {failure_path}")

    counts = {
        "written": sum(1 for item in records if item["status"] == "written"),
        "exists": sum(1 for item in records if item["status"] == "exists"),
        "dry_run": sum(1 for item in records if item["status"] == "dry_run"),
    }
    print(json.dumps({"done": True, **counts}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
