#!/usr/bin/env python3
"""Sanity-check frame paths referenced by pseudo-GT candidate JSONL files."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DATASETS = ("tvr", "activitynet", "msrvtt")
DEFAULT_CAPTION_FILES = {
    "tvr": "tvrval.caption.txt",
    "activitynet": "datasets/activitynet/activitynetval.caption.txt",
    "msrvtt": None,
}
FRAME_PATH_KEYS = ("gt_frame_paths", "pseudo_frame_paths", "candidate_frame_paths")
GT_VIDEO_KEYS = ("original_gt_video_id", "gt_video_id", "query_gt_video_id")
CANDIDATE_VIDEO_KEYS = ("pseudo_video_id", "candidate_video_id")


def default_input_for_dataset(dataset: str) -> str:
    return f"outputs/upstream/pseudo_gt_candidates.{dataset}.jsonl"


def default_output_for_dataset(dataset: str, skip_frame_check: bool, mode: str) -> str:
    suffix = "video_sanity" if skip_frame_check else f"frame_sanity.{mode}"
    return f"outputs/upstream/pseudo_gt_candidates.{dataset}.{suffix}.json"


def read_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if line.strip():
                yield line_idx, json.loads(line)


def collect_frame_paths(
    input_path: Path,
    expected_per_key: int,
    max_bad_rows: int,
) -> Tuple[
    int,
    List[str],
    List[Dict[str, Any]],
    Dict[str, int],
    List[str],
    List[str],
]:
    unique_paths = set()
    gt_videos = set()
    candidate_videos = set()
    rows = 0
    bad_rows: List[Dict[str, Any]] = []
    key_counts: Counter[str] = Counter()

    for line_idx, row in read_jsonl(input_path):
        rows += 1
        for key in FRAME_PATH_KEYS:
            if key not in row:
                continue

            paths = row.get(key) or []
            key_counts[key] += len(paths)
            if len(paths) != expected_per_key and len(bad_rows) < max_bad_rows:
                bad_rows.append(
                    {
                        "line_idx": line_idx,
                        "key": key,
                        "count": len(paths),
                        "query_key": row.get("query_key"),
                        "desc_id": row.get("desc_id"),
                    }
                )
            unique_paths.update(str(path) for path in paths)

        for key in GT_VIDEO_KEYS:
            value = row.get(key)
            if value:
                gt_videos.add(str(value))
                break

        for key in CANDIDATE_VIDEO_KEYS:
            value = row.get(key)
            if value:
                candidate_videos.add(str(value))
                break

    return (
        rows,
        sorted(unique_paths),
        bad_rows,
        dict(key_counts),
        sorted(gt_videos),
        sorted(candidate_videos),
    )


def video_id_from_caption_key(key: str) -> str:
    return key.split("#enc#", 1)[0]


def read_caption_videos(path: Path) -> Tuple[int, List[str]]:
    rows = 0
    videos = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows += 1
            key = line.split(None, 1)[0]
            videos.add(video_id_from_caption_key(key))
    return rows, sorted(videos)


def make_caption_video_summary(
    caption_path: Path,
    gt_videos: Sequence[str],
    candidate_videos: Sequence[str],
    max_samples: int,
) -> Dict[str, Any]:
    caption_rows, caption_videos = read_caption_videos(caption_path)
    candidate_all_videos = sorted(set(gt_videos) | set(candidate_videos))
    caption_video_set = set(caption_videos)
    candidate_all_video_set = set(candidate_all_videos)
    missing_from_caption = sorted(candidate_all_video_set - caption_video_set)
    caption_not_in_candidates = sorted(caption_video_set - candidate_all_video_set)

    return {
        "caption_file": str(caption_path),
        "caption_rows": caption_rows,
        "caption_unique_videos": len(caption_videos),
        "candidate_unique_gt_videos": len(set(gt_videos)),
        "candidate_unique_pseudo_videos": len(set(candidate_videos)),
        "candidate_unique_all_videos": len(candidate_all_videos),
        "candidate_videos_all_in_caption": not missing_from_caption,
        "candidate_video_count_matches_caption": len(candidate_all_videos) == len(caption_videos),
        "candidate_video_set_matches_caption": not missing_from_caption and not caption_not_in_candidates,
        "candidate_videos_missing_from_caption_count": len(missing_from_caption),
        "candidate_videos_missing_from_caption_sample": missing_from_caption[:max_samples],
        "caption_videos_not_in_candidates_count": len(caption_not_in_candidates),
        "caption_videos_not_in_candidates_sample": caption_not_in_candidates[:max_samples],
    }


def check_jpeg_markers(path: str) -> Optional[Tuple[Any, ...]]:
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return ("missing", path)
    except Exception as exc:
        return ("stat_error", path, type(exc).__name__, str(exc))

    if st.st_size <= 0:
        return ("zero_size", path, st.st_size)

    try:
        with open(path, "rb") as f:
            head = f.read(2)
            if st.st_size >= 2:
                f.seek(-2, os.SEEK_END)
                tail = f.read(2)
            else:
                tail = b""
    except Exception as exc:
        return ("read_error", path, type(exc).__name__, str(exc))

    if head != b"\xff\xd8":
        return ("bad_jpeg_soi", path, st.st_size, head.hex())
    if tail != b"\xff\xd9":
        return ("bad_jpeg_eoi", path, st.st_size, tail.hex())

    return None


def check_pil_decode(path: str) -> Optional[Tuple[Any, ...]]:
    marker_issue = check_jpeg_markers(path)
    if marker_issue is not None:
        return marker_issue

    try:
        from PIL import Image

        with Image.open(path) as img:
            img.verify()
    except Exception as exc:
        return ("pil_decode_error", path, type(exc).__name__, str(exc))

    return None


def check_paths(
    paths: Sequence[str],
    workers: int,
    mode: str,
    progress_every: int,
    max_samples: int,
) -> Tuple[Dict[str, int], List[Tuple[Any, ...]], float]:
    check_fn = check_pil_decode if mode == "pil" else check_jpeg_markers
    counts: Counter[str] = Counter()
    samples: List[Tuple[Any, ...]] = []

    start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(check_fn, path) for path in paths]
        for checked, future in enumerate(as_completed(futures), 1):
            issue = future.result()
            if issue is not None:
                counts[str(issue[0])] += 1
                if len(samples) < max_samples:
                    samples.append(issue)
            if progress_every > 0 and checked % progress_every == 0:
                print(
                    json.dumps(
                        {
                            "checked": checked,
                            "total": len(paths),
                            "elapsed_sec": round(time.time() - start, 1),
                            "issues": dict(counts),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    return dict(counts), samples, time.time() - start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, default="activitynet")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument(
        "--mode",
        choices=["markers", "pil"],
        default="markers",
        help=(
            "markers checks existence, nonzero size, and JPEG SOI/EOI bytes. "
            "pil additionally decodes every unique image with Pillow and is slower."
        ),
    )
    parser.add_argument("--expected-per-key", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=250000)
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument(
        "--caption-file",
        default=None,
        help=(
            "Optional split caption file. Checks that unique videos referenced by "
            "the candidate JSONL are present in this split."
        ),
    )
    parser.add_argument(
        "--require-video-set-match",
        action="store_true",
        help="Fail unless the candidate unique video set exactly matches --caption-file.",
    )
    parser.add_argument(
        "--skip-frame-check",
        action="store_true",
        help="Only collect JSONL/frame-path/video-set metadata; skip per-file frame checks.",
    )
    parser.add_argument("--fail-on-issue", action="store_true")
    args = parser.parse_args()

    if args.input is None:
        args.input = default_input_for_dataset(args.dataset)
    if args.caption_file is None:
        args.caption_file = DEFAULT_CAPTION_FILES[args.dataset]
    if args.output is None:
        args.output = default_output_for_dataset(
            dataset=args.dataset,
            skip_frame_check=args.skip_frame_check,
            mode=args.mode,
        )

    return args


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    start = time.time()
    rows, paths, bad_rows, key_counts, gt_videos, candidate_videos = collect_frame_paths(
        input_path=input_path,
        expected_per_key=args.expected_per_key,
        max_bad_rows=args.max_samples,
    )

    caption_video_summary = None
    if args.caption_file is not None:
        caption_video_summary = make_caption_video_summary(
            caption_path=Path(args.caption_file),
            gt_videos=gt_videos,
            candidate_videos=candidate_videos,
            max_samples=args.max_samples,
        )

    collect_summary = {
        "input": str(input_path),
        "candidate_rows": rows,
        "unique_frame_paths": len(paths),
        "unique_gt_videos": len(set(gt_videos)),
        "unique_pseudo_videos": len(set(candidate_videos)),
        "unique_all_videos": len(set(gt_videos) | set(candidate_videos)),
        "frame_path_key_counts": key_counts,
        "bad_frame_count_rows_sample": bad_rows,
        "caption_video_summary": caption_video_summary,
        "collect_sec": round(time.time() - start, 2),
    }
    print(json.dumps(collect_summary, ensure_ascii=False, indent=2), flush=True)

    if args.skip_frame_check:
        issues: Dict[str, int] = {}
        samples: List[Tuple[Any, ...]] = []
        check_sec = 0.0
    else:
        issues, samples, check_sec = check_paths(
            paths=paths,
            workers=args.workers,
            mode=args.mode,
            progress_every=args.progress_every,
            max_samples=args.max_samples,
        )

    summary = {
        **collect_summary,
        "mode": args.mode,
        "workers": args.workers,
        "skip_frame_check": args.skip_frame_check,
        "checked_paths": 0 if args.skip_frame_check else len(paths),
        "issues": issues,
        "issue_samples": samples,
        "check_sec": round(check_sec, 2),
        "total_sec": round(time.time() - start, 2),
        "ok": not issues
        and not bad_rows
        and (
            caption_video_summary is None
            or not args.require_video_set_match
            or caption_video_summary["candidate_video_set_matches_caption"]
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {output_path}", flush=True)

    if args.fail_on_issue and not summary["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
