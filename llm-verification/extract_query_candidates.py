#!/usr/bin/env python3
"""Extract a small pseudo-GT candidate JSONL for one query.

Example:
    python qwen/extract_query_candidates.py \
      --dataset tvr \
      --query "Sheldon sits down in his spot on the couch" \
      --output outputs/runs/sheldon_sits_couch/candidates.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable


DATASETS = ("tvr", "activitynet", "msrvtt")
DEFAULT_OUTPUT = "outputs/runs/query_subset/candidates.jsonl"


def default_input_for_dataset(dataset: str) -> str:
    return f"outputs/upstream/pseudo_gt_candidates.{dataset}.jsonl"


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def normalize_query(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    return text.rstrip(".").strip().lower()


def rewrite_path(path: str, prefix_from: str | None, prefix_to: str | None) -> str:
    if prefix_from and prefix_to and path.startswith(prefix_from):
        return prefix_to.rstrip("/") + path[len(prefix_from.rstrip("/")) :]
    return path


def rewrite_row_frame_paths(
    row: Dict[str, Any],
    prefix_from: str | None,
    prefix_to: str | None,
) -> Dict[str, Any]:
    if not (prefix_from and prefix_to):
        return row

    rewritten = dict(row)
    for key in ("gt_frame_paths", "pseudo_frame_paths", "candidate_frame_paths"):
        if key in rewritten and rewritten[key] is not None:
            rewritten[key] = [
                rewrite_path(str(path), prefix_from, prefix_to)
                for path in rewritten[key]
            ]
    return rewritten


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, default="tvr")
    parser.add_argument("--input", default=None, help="Defaults to outputs/upstream/pseudo_gt_candidates.<dataset>.jsonl")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--query",
        action="append",
        default=[],
        help="Query text to extract. Can be passed multiple times.",
    )
    parser.add_argument(
        "--query-key",
        action="append",
        default=[],
        help="Exact query_key/desc_id to extract. Can be passed multiple times.",
    )
    parser.add_argument(
        "--path-prefix-from",
        default=None,
        help="Rewrite frame paths that start with this prefix.",
    )
    parser.add_argument(
        "--path-prefix-to",
        default=None,
        help="Replacement prefix for frame paths.",
    )
    parser.add_argument("--exact", action="store_true", help="Require exact normalized query equality.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.input is None:
        args.input = default_input_for_dataset(args.dataset)
    if not args.query and not args.query_key:
        raise ValueError("At least one --query or --query-key is required.")
    if bool(args.path_prefix_from) != bool(args.path_prefix_to):
        raise ValueError("--path-prefix-from and --path-prefix-to must be provided together")
    return args


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    targets = [normalize_query(query) for query in args.query]
    target_keys = {str(key) for key in args.query_key}
    count = 0
    query_keys = set()
    with output_path.open("w", encoding="utf-8") as out_f:
        for row in read_jsonl(input_path):
            row_query = normalize_query(str(row.get("query", "")))
            row_key = str(row.get("query_key") or row.get("desc_id"))
            query_matched = any(
                row_query == target if args.exact else target in row_query
                for target in targets
            )
            key_matched = row_key in target_keys
            matched = query_matched or key_matched
            if not matched:
                continue
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
            count += 1
            query_keys.add(row.get("query_key") or row.get("desc_id"))
            if args.limit is not None and count >= args.limit:
                break

    print(
        json.dumps(
            {
                "input": str(input_path),
                "output": str(output_path),
                "queries": args.query,
                "query_keys_requested": args.query_key,
                "num_records": count,
                "query_keys": sorted(str(key) for key in query_keys),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
