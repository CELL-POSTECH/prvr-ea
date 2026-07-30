#!/usr/bin/env python3
"""Materialize candidate frame images listed in a pseudo-GT JSONL.

ActivityNet candidates built with frame-source=paths contain frame timestamps and
output paths, but no image files yet. Run this on the GPU/server that has the raw
mp4 files before running Qwen verification.

Example:
    python qwen/materialize_candidate_frames.py \
      --dataset activitynet \
      --video-root /data/activitynet/val \
      --workers 8
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

DATASETS = ("tvr", "activitynet", "charades")
VIDEO_STREAM_CACHE: Dict[str, Optional[Tuple[float, Optional[float]]]] = {}
VIDEO_STREAM_CACHE_LOCK = Lock()


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


def probe_video_stream(video_path: str) -> Optional[Tuple[float, Optional[float]]]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=duration,avg_frame_rate,r_frame_rate:format=duration",
        "-of",
        "json",
        video_path,
    ]

    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    data = json.loads(result.stdout)
    streams = data.get("streams") or []

    if not streams:
        return None

    stream = streams[0]
    duration = stream.get("duration")

    if duration in (None, "N/A"):
        duration = (data.get("format") or {}).get("duration")

    if duration in (None, "N/A"):
        return None

    fps = (
        parse_frame_rate(stream.get("avg_frame_rate"))
        or parse_frame_rate(stream.get("r_frame_rate"))
    )

    return float(duration), fps


def get_video_stream_info(video_path: str) -> Optional[Tuple[float, Optional[float]]]:
    with VIDEO_STREAM_CACHE_LOCK:
        if video_path in VIDEO_STREAM_CACHE:
            return VIDEO_STREAM_CACHE[video_path]

    stream_info = probe_video_stream(video_path)

    with VIDEO_STREAM_CACHE_LOCK:
        return VIDEO_STREAM_CACHE.setdefault(video_path, stream_info)


def clamp_to_video_stream(video_path: str, time_sec: float) -> Optional[float]:
    stream_info = get_video_stream_info(video_path)

    if stream_info is None:
        return None

    duration, fps = stream_info
    frame_margin = max(2.0 / fps, 0.25) if fps else 0.25
    max_time = max(0.0, duration - frame_margin)

    if time_sec <= max_time:
        return None

    return max_time


def default_input_for_dataset(dataset: str) -> str:
    return f"outputs/upstream/pseudo_gt_candidates.{dataset}.jsonl"


def read_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if line.strip():
                yield line_idx, json.loads(line)


def find_video_file(video_root: str, video_id: str, cache: Dict[str, Optional[str]]) -> Optional[str]:
    if video_id in cache:
        return cache[video_id]

    bare_id = video_id[2:] if video_id.startswith("v_") else video_id

    candidates = [
        os.path.join(video_root, video_id),
        os.path.join(video_root, f"{video_id}.mp4"),
        os.path.join(video_root, f"{video_id}.mkv"),
        os.path.join(video_root, bare_id),
        os.path.join(video_root, f"{bare_id}.mp4"),
        os.path.join(video_root, f"{bare_id}.mkv"),
    ]

    candidates.extend(glob.glob(os.path.join(video_root, "*", f"{video_id}.mp4")))
    candidates.extend(glob.glob(os.path.join(video_root, "*", f"{bare_id}.mp4")))

    for candidate in candidates:
        if os.path.isfile(candidate):
            cache[video_id] = candidate
            return candidate

    cache[video_id] = None
    return None


def rewrite_path(path: str, prefix_from: Optional[str], prefix_to: Optional[str]) -> str:
    if prefix_from and prefix_to and path.startswith(prefix_from):
        return prefix_to.rstrip("/") + path[len(prefix_from.rstrip("/")) :]
    return path


def rewrite_row_frame_paths(
    row: Dict[str, Any],
    prefix_from: Optional[str],
    prefix_to: Optional[str],
) -> Dict[str, Any]:
    if not (prefix_from and prefix_to):
        return row

    rewritten = dict(row)

    for key in ("gt_frame_paths", "pseudo_frame_paths"):
        if key in rewritten and rewritten[key] is not None:
            rewritten[key] = [
                rewrite_path(str(path), prefix_from, prefix_to)
                for path in rewritten[key]
            ]

    return rewritten


def write_rewritten_jsonl(args: argparse.Namespace) -> None:
    if args.output_jsonl is None:
        return

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out_f:
        for _line_idx, row in read_jsonl(Path(args.input)):
            out_f.write(
                json.dumps(
                    rewrite_row_frame_paths(
                        row,
                        args.path_prefix_from,
                        args.path_prefix_to,
                    ),
                    ensure_ascii=False,
                )
                + "\n"
            )


def extract_frame(
    video_path: str,
    time_sec: float,
    output_path: str,
    overwrite: bool,
    ffmpeg_threads: int,
) -> str:
    if os.path.exists(output_path) and not overwrite:
        return "exists"

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    tmp_path = f"{output_path}.tmp.{os.getpid()}.jpg"

    def format_seek_time(seek_time: float) -> str:
        floored_time = math.floor(max(0.0, seek_time) * 1000.0) / 1000.0
        return f"{floored_time:.3f}"

    def run_ffmpeg_extract(seek_time: float) -> subprocess.CompletedProcess[str]:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

        cmd = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads",
            str(ffmpeg_threads),
            "-y",
            "-ss",
            format_seek_time(seek_time),
            "-i",
            video_path,
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-frames:v",
            "1",
            "-q:v",
            "2",
            tmp_path,
        ]

        return subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )

    requested_time = max(0.0, float(time_sec))
    attempted_times = [requested_time]
    result = run_ffmpeg_extract(requested_time)

    if not os.path.exists(tmp_path):
        fallback_times: List[float] = []
        clamped_time = clamp_to_video_stream(video_path, requested_time)

        if clamped_time is not None:
            fallback_times.append(clamped_time)

        fallback_times.extend(
            max(0.0, requested_time - offset)
            for offset in (0.25, 0.5, 1.0)
        )

        for fallback_time in fallback_times:
            formatted_time = format_seek_time(fallback_time)

            if formatted_time in {format_seek_time(item) for item in attempted_times}:
                continue

            attempted_times.append(fallback_time)
            result = run_ffmpeg_extract(fallback_time)

            if os.path.exists(tmp_path):
                break

    if not os.path.exists(tmp_path):
        stderr = result.stderr.strip()
        detail = f"; ffmpeg stderr: {stderr}" if stderr else ""
        attempted = ", ".join(format_seek_time(item) for item in attempted_times)

        raise RuntimeError(
            f"ffmpeg did not write a frame for {video_path} at {requested_time:.3f}s"
            f" after attempts [{attempted}]"
            f"{detail}"
        )

    os.replace(tmp_path, output_path)

    return "written"


def add_tasks_for_role(
    tasks: Dict[str, Tuple[str, float, str, int, str]],
    row: Dict[str, Any],
    line_idx: int,
    role: str,
    video_id: str,
    times_key: str,
    paths_key: str,
    video_root: str,
    video_cache: Dict[str, Optional[str]],
    prefix_from: Optional[str],
    prefix_to: Optional[str],
) -> int:
    times = row.get(times_key)
    paths = row.get(paths_key)

    if times is None or paths is None:
        raise KeyError(
            f"line {line_idx}: missing {times_key} or {paths_key}; "
            "rebuild candidates with --frame-source paths/videos"
        )

    if len(times) != len(paths):
        raise ValueError(
            f"line {line_idx}: {times_key}/{paths_key} length mismatch: "
            f"{len(times)} vs {len(paths)}"
        )

    video_path = find_video_file(video_root, str(video_id), video_cache)

    if video_path is None:
        raise FileNotFoundError(
            f"line {line_idx}: could not find raw video for {video_id} under {video_root}"
        )

    added = 0

    for time_sec, path in zip(times, paths):
        out_path = rewrite_path(str(path), prefix_from, prefix_to)

        if out_path not in tasks:
            tasks[out_path] = (
                video_path,
                float(time_sec),
                role,
                line_idx,
                str(video_id),
            )
            added += 1

    return added


def collect_tasks(args: argparse.Namespace) -> Dict[str, Tuple[str, float, str, int, str]]:
    tasks: Dict[str, Tuple[str, float, str, int, str]] = {}
    video_cache: Dict[str, Optional[str]] = {}
    seen_rows = 0

    for line_idx, row in read_jsonl(Path(args.input)):
        if line_idx < args.start_index:
            continue

        if args.limit is not None and seen_rows >= args.limit:
            break

        seen_rows += 1

        gt_video_id = (
            row.get("original_gt_video_id")
            or row.get("gt_video_id")
            or row.get("query_gt_video_id")
        )
        pseudo_video_id = row.get("pseudo_video_id") or row.get("candidate_video_id")

        if gt_video_id is None or pseudo_video_id is None:
            raise KeyError(f"line {line_idx}: missing gt or pseudo video id")

        add_tasks_for_role(
            tasks,
            row,
            line_idx,
            "gt",
            str(gt_video_id),
            "gt_frame_times",
            "gt_frame_paths",
            args.video_root,
            video_cache,
            args.path_prefix_from,
            args.path_prefix_to,
        )

        add_tasks_for_role(
            tasks,
            row,
            line_idx,
            "pseudo",
            str(pseudo_video_id),
            "pseudo_frame_times",
            "pseudo_frame_paths",
            args.video_root,
            video_cache,
            args.path_prefix_from,
            args.path_prefix_to,
        )

    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--dataset", choices=DATASETS, default="activitynet")
    parser.add_argument(
        "--input",
        default=None,
        help="Defaults to outputs/upstream/pseudo_gt_candidates.<dataset>.jsonl",
    )
    parser.add_argument(
        "--video-root",
        required=True,
        help="Root containing raw mp4/mkv files.",
    )
    parser.add_argument(
        "--path-prefix-from",
        default=None,
        help="Rewrite frame output paths that start with this prefix.",
    )
    parser.add_argument(
        "--path-prefix-to",
        default=None,
        help="Replacement prefix for frame output paths.",
    )
    parser.add_argument(
        "--output-jsonl",
        default=None,
        help=(
            "Optional candidates JSONL to write with rewritten frame paths. "
            "Use this with --path-prefix-from/to before Qwen verification."
        ),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--ffmpeg-threads",
        type=int,
        default=1,
        help="Threads per ffmpeg process. Keep this low when using many workers.",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--failure-jsonl",
        default=None,
        help=(
            "Path to write failed frame cases as they happen. Defaults to "
            "<input>.materialize_failed.jsonl."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if args.input is None:
        args.input = default_input_for_dataset(args.dataset)

    if bool(args.path_prefix_from) != bool(args.path_prefix_to):
        raise ValueError("--path-prefix-from and --path-prefix-to must be provided together")

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    if args.ffmpeg_threads < 1:
        raise ValueError("--ffmpeg-threads must be >= 1")

    return args


def main() -> None:
    args = parse_args()

    tasks = collect_tasks(args)
    write_rewritten_jsonl(args)
    fail_path = (
        Path(args.failure_jsonl)
        if args.failure_jsonl is not None
        else Path(args.input).with_suffix(".materialize_failed.jsonl")
    )

    existing = sum(1 for path in tasks if os.path.exists(path))
    pending = {
        path: task
        for path, task in tasks.items()
        if args.overwrite or not os.path.exists(path)
    }

    summary = {
        "input": args.input,
        "video_root": args.video_root,
        "total_unique_frames": len(tasks),
        "existing_frames": existing,
        "pending_frames": len(pending),
        "workers": args.workers,
        "ffmpeg_threads": args.ffmpeg_threads,
        "dry_run": args.dry_run,
        "output_jsonl": args.output_jsonl,
        "failure_jsonl": str(fail_path),
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    if args.dry_run or not pending:
        return

    written = 0
    failed: List[Dict[str, Any]] = []
    fail_path.parent.mkdir(parents=True, exist_ok=True)

    with fail_path.open("w", encoding="utf-8") as fail_f:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            future_to_path = {
                pool.submit(
                    extract_frame,
                    video_path,
                    time_sec,
                    out_path,
                    args.overwrite,
                    args.ffmpeg_threads,
                ): out_path
                for out_path, (video_path, time_sec, _role, _line_idx, _video_id)
                in pending.items()
            }

            with tqdm(
                total=len(future_to_path),
                desc="Extracting frames",
                unit="frame",
                dynamic_ncols=True,
            ) as pbar:
                for future in as_completed(future_to_path):
                    out_path = future_to_path[future]
                    video_path, time_sec, role, line_idx, video_id = pending[out_path]

                    try:
                        status = future.result()

                        if status == "written":
                            written += 1

                    except Exception as exc:
                        failure = {
                            "output_path": out_path,
                            "video_path": video_path,
                            "time_sec": time_sec,
                            "role": role,
                            "line_idx": line_idx,
                            "video_id": video_id,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                        failed.append(failure)
                        fail_f.write(json.dumps(failure, ensure_ascii=False) + "\n")
                        fail_f.flush()
                        tqdm.write(
                            "[frame_failed] "
                            + json.dumps(
                                failure,
                                ensure_ascii=False,
                                sort_keys=True,
                            )
                        )

                    pbar.update(1)
                    pbar.set_postfix(
                        written=written,
                        failed=len(failed),
                    )

    result = {
        "written_frames": written,
        "failed_frames": len(failed),
    }

    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)

    if failed:
        raise RuntimeError(
            f"Failed to materialize {len(failed)} frames; see {fail_path}"
        )


if __name__ == "__main__":
    main()
