#!/usr/bin/env python3
"""Build a dense caption file from Qwen-accepted extra GT rows.

The output keeps the original caption file order, inserting accepted pseudo-GT
captions immediately after the existing captions for each accepted candidate
video id. New rows use the next available ``#enc#`` index for that video.

Example:
    python build_dense_caption.py \
      --dataset tvr \
      --split val \
      --extra-gt-all outputs/full_accept_reverify_current_prompt/extra_gt_all.jsonl \
      --overwrite
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ACCEPT_VALUES = {"accept", "accepted", "true", "1", "yes"}


def existing_path(candidates: Sequence[Path]) -> Optional[Path]:
    for path in candidates:
        if path.exists():
            return path
    return None


def default_caption_candidates(dataset: str, split: str) -> List[Path]:
    filename = f"{dataset}{split}.caption.txt"
    return [
        Path(filename),
        Path("data") / filename,
        Path("datasets") / dataset / filename,
    ]


def default_extra_gt_candidates(dataset: str, split: str) -> List[Path]:
    candidates = [
        Path("outputs") / "runs" / f"{dataset}_{split}" / "extra_gt_all.jsonl",
        Path("outputs") / "runs" / dataset / "extra_gt_all.jsonl",
        Path("outputs") / f"{dataset}_{split}" / "extra_gt_all.jsonl",
    ]
    if dataset == "tvr" and split == "val":
        candidates.extend(
            [
                Path("outputs") / "full_accept_reverify_current_prompt" / "extra_gt_all.jsonl",
                Path("outputs") / "runs" / "full" / "extra_gt_all.jsonl",
            ]
        )
    return candidates


def default_verification_candidates(dataset: str, split: str) -> List[Path]:
    candidates = [
        Path("outputs") / "runs" / f"{dataset}_{split}" / "verification.jsonl",
        Path("outputs") / "runs" / dataset / "verification.jsonl",
        Path("outputs") / f"{dataset}_{split}" / "verification.jsonl",
    ]
    if dataset == "tvr" and split == "val":
        candidates.extend(
            [
                Path("outputs") / "full_accept_reverify_current_prompt" / "verification.jsonl",
                Path("outputs") / "runs" / "full" / "verification.jsonl",
            ]
        )
    return candidates


def default_output_path(dataset: str, split: str) -> Path:
    return Path(f"{dataset}dense{split}.caption.txt")


def read_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, 1):
            if line.strip():
                yield line_idx, json.loads(line)


def parse_caption_key(key: str) -> Tuple[str, int]:
    video_id, enc = key.rsplit("#enc#", 1)
    return video_id, int(enc)


def read_caption_file(path: Path) -> Tuple[List[Tuple[str, str]], Dict[str, int], set[str]]:
    entries: List[Tuple[str, str]] = []
    max_enc_by_video: Dict[str, int] = {}
    video_ids: set[str] = set()
    key_re = re.compile(r"^(.+?#enc#\d+)(?:\s|$)")

    with path.open("r", encoding="utf-8") as f:
        for line_idx, raw_line in enumerate(f, 1):
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue
            match = key_re.match(line)
            if not match:
                raise ValueError(f"Malformed caption line {line_idx} in {path}: {line[:120]}")
            video_id, enc = parse_caption_key(match.group(1))
            max_enc_by_video[video_id] = max(max_enc_by_video.get(video_id, -1), enc)
            video_ids.add(video_id)
            entries.append((line, video_id))

    return entries, max_enc_by_video, video_ids


def get_candidate_video_id(row: Dict[str, Any]) -> Optional[str]:
    value = row.get("candidate_video_id") or row.get("pseudo_video_id")
    return str(value) if value else None


def get_query_key(row: Dict[str, Any]) -> Optional[str]:
    value = row.get("query_key") or row.get("query_id")
    return str(value) if value else None


def get_query(row: Dict[str, Any]) -> Optional[str]:
    value = row.get("query")
    if value is None:
        return None
    return re.sub(r"\s+", " ", str(value)).strip()


def is_accepted(row: Dict[str, Any]) -> bool:
    if row.get("add_to_extra_gt") is True:
        return True
    recommendation = row.get("qwen_recommendation")
    if recommendation is None:
        recommendation = row.get("qwen_result", {}).get("gt_label_recommendation")
    return str(recommendation).strip().lower() in ACCEPT_VALUES


def iter_accepted_rows(path: Path, filter_accept: bool) -> Iterable[Tuple[int, Dict[str, Any]]]:
    for line_idx, row in read_jsonl(path):
        if filter_accept and not is_accepted(row):
            continue
        yield line_idx, row


def interleave_caption_lines(
    base_entries: Sequence[Tuple[str, str]],
    extra_lines_by_video: Dict[str, List[str]],
) -> List[str]:
    last_index_by_video = {
        video_id: index
        for index, (_line, video_id) in enumerate(base_entries)
    }
    output_lines: List[str] = []
    inserted_videos: set[str] = set()

    for index, (line, video_id) in enumerate(base_entries):
        output_lines.append(line)
        if index != last_index_by_video[video_id]:
            continue
        output_lines.extend(extra_lines_by_video.get(video_id, []))
        inserted_videos.add(video_id)

    for video_id, extra_lines in extra_lines_by_video.items():
        if video_id not in inserted_videos:
            output_lines.extend(extra_lines)

    return output_lines


def resolve_input_paths(args: argparse.Namespace) -> Tuple[Path, Path, bool]:
    caption_path = Path(args.caption_file) if args.caption_file else existing_path(
        default_caption_candidates(args.dataset, args.split)
    )
    if caption_path is None:
        searched = [str(path) for path in default_caption_candidates(args.dataset, args.split)]
        raise FileNotFoundError(f"Could not find caption file. Searched: {searched}")

    if args.extra_gt_all and args.verification:
        raise ValueError("Pass only one of --extra-gt-all or --verification.")

    if args.extra_gt_all:
        return caption_path, Path(args.extra_gt_all), False
    if args.verification:
        return caption_path, Path(args.verification), True

    extra_gt_path = existing_path(default_extra_gt_candidates(args.dataset, args.split))
    if extra_gt_path is not None:
        return caption_path, extra_gt_path, False

    verification_path = existing_path(default_verification_candidates(args.dataset, args.split))
    if verification_path is not None:
        return caption_path, verification_path, True

    searched = [
        str(path)
        for path in default_extra_gt_candidates(args.dataset, args.split)
        + default_verification_candidates(args.dataset, args.split)
    ]
    raise FileNotFoundError(f"Could not find extra GT or verification JSONL. Searched: {searched}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. tvr, activitynet, charades.")
    parser.add_argument("--split", required=True, help="Split name, e.g. train, val, test.")
    parser.add_argument("--caption-file", default=None, help="Defaults to <dataset><split>.caption.txt.")
    parser.add_argument("--extra-gt-all", default=None, help="Accepted rows JSONL. Used without additional filtering.")
    parser.add_argument("--verification", default=None, help="Verification JSONL. Only accepted rows are appended.")
    parser.add_argument("--output", default=None, help="Defaults to <dataset>dense<split>.caption.txt.")
    parser.add_argument("--allow-new-videos", action="store_true", help="Allow appending captions for videos absent from the base caption file.")
    parser.add_argument("--dry-run", action="store_true", help="Print summary without writing the output file.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    caption_path, accepted_path, filter_accept = resolve_input_paths(args)
    output_path = Path(args.output) if args.output else default_output_path(args.dataset, args.split)

    base_entries, max_enc_by_video, base_video_ids = read_caption_file(caption_path)
    extra_lines_by_video: Dict[str, List[str]] = {}
    seen_pairs: set[Tuple[str, str]] = set()
    skipped_duplicates = 0
    skipped_missing_fields = 0
    missing_videos: set[str] = set()
    accepted_rows = 0

    for _line_idx, row in iter_accepted_rows(accepted_path, filter_accept=filter_accept):
        accepted_rows += 1
        candidate_video_id = get_candidate_video_id(row)
        query_key = get_query_key(row)
        query = get_query(row)
        if not candidate_video_id or not query_key or not query:
            skipped_missing_fields += 1
            continue

        if candidate_video_id not in base_video_ids and not args.allow_new_videos:
            missing_videos.add(candidate_video_id)
            continue

        pair = (query_key, candidate_video_id)
        if pair in seen_pairs:
            skipped_duplicates += 1
            continue
        seen_pairs.add(pair)

        next_enc = max_enc_by_video.get(candidate_video_id, -1) + 1
        max_enc_by_video[candidate_video_id] = next_enc
        extra_lines_by_video.setdefault(candidate_video_id, []).append(f"{candidate_video_id}#enc#{next_enc} {query}")

    if missing_videos:
        sample = sorted(missing_videos)[:20]
        raise ValueError(
            f"{len(missing_videos)} candidate video(s) are absent from {caption_path}. "
            f"Sample: {sample}. Use --allow-new-videos to append them anyway."
        )

    output_lines = interleave_caption_lines(base_entries, extra_lines_by_video)

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "caption_file": str(caption_path),
        "accepted_input": str(accepted_path),
        "accepted_input_filtered": filter_accept,
        "output": str(output_path),
        "base_lines": len(base_entries),
        "accepted_rows_seen": accepted_rows,
        "inserted_lines": sum(len(lines) for lines in extra_lines_by_video.values()),
        "skipped_duplicates": skipped_duplicates,
        "skipped_missing_fields": skipped_missing_fields,
        "total_output_lines": len(output_lines),
        "dry_run": args.dry_run,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.dry_run:
        return
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for line in output_lines:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
