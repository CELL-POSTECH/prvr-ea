"""Shared evaluation-only metrics for PRVR dense multi-GT annotations.

Models continue to produce their original query-to-video scores.  When
``PRVR_MULTI_GT=1`` is set, evaluators use the dense JSONL mapping to replace
the original one-positive-per-query target with every verified positive video.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np


def enabled() -> bool:
    return os.environ.get("PRVR_MULTI_GT", "").lower() in {"1", "true", "yes"}


def _annotation_path() -> Path:
    configured = os.environ.get("PRVR_MULTI_GT_FILE")
    if configured:
        path = Path(configured)
        if path.exists():
            return path
        raise FileNotFoundError(f"multi-GT annotation file not found: {path}")
    project_root = Path(os.environ.get("PRVR_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
    data_root = Path(os.environ.get(
        "PRVR_MULTI_GT_DATA_ROOT",
        os.environ.get("PRVR_DATA_ROOT", str(project_root / "datasets")),
    ))
    collection = os.environ.get("PRVR_MULTI_GT_COLLECTION")
    if not collection:
        alias = os.environ.get("PRVR_MULTI_GT_DATASET", "").removesuffix("_clip")
        collection = {"act": "activitynet", "cha": "charades"}.get(alias, alias)
    if not collection:
        raise RuntimeError("multi-GT dataset/collection was not configured")
    if collection == "tvr":
        candidates = (
            data_root / collection / "tvrdenseval_v.gt.jsonl",
            data_root / collection / "TextData" / "tvrdenseval.gt.jsonl",
            data_root / collection / "tvrdenseval.gt.jsonl",
        )
    else:
        candidates = (
            data_root / collection / "TextData" / f"{collection}denseval.gt.jsonl",
            data_root / collection / f"{collection}denseval.gt.jsonl",
        )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("multi-GT annotation file not found; tried: " + ", ".join(map(str, candidates)))


@lru_cache(maxsize=8)
def _load_mapping(path_string: str) -> dict[str, tuple[str, ...]]:
    mapping: dict[str, tuple[str, ...]] = {}
    with Path(path_string).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            query_id = item.get("query_id")
            video_ids = item.get("gt_video_ids") or item.get("video_ids") or item.get("videos")
            if not query_id or not isinstance(video_ids, list) or not video_ids:
                raise ValueError(f"invalid multi-GT item at {path_string}:{line_number}")
            mapping[query_id] = tuple(dict.fromkeys(video_ids))
    return mapping


def multi_gt_indices(video_metas: Iterable[str], query_metas: Iterable[str]) -> dict[int, list[int]]:
    """Return query-index -> all candidate positive-video indices.

    Every dense GT video must be part of the unchanged standard test candidate
    set. Failing loudly prevents silently inflating scores by dropping
    positives that the model never had a chance to retrieve.
    """
    videos = list(video_metas)
    queries = list(query_metas)
    mapping = _load_mapping(str(_annotation_path()))
    video_to_index = {video_id: index for index, video_id in enumerate(videos)}
    targets: dict[int, list[int]] = {}
    missing_queries = []
    missing_videos: dict[str, list[str]] = {}
    for query_index, query_id in enumerate(queries):
        gt_video_ids = mapping.get(query_id)
        if gt_video_ids is None:
            missing_queries.append(query_id)
            continue
        absent = [video_id for video_id in gt_video_ids if video_id not in video_to_index]
        if absent:
            missing_videos[query_id] = absent
            continue
        targets[query_index] = [video_to_index[video_id] for video_id in gt_video_ids]
    if missing_queries:
        raise ValueError(f"multi-GT file lacks {len(missing_queries)} evaluated queries; first: {missing_queries[:3]}")
    if missing_videos:
        first_query = next(iter(missing_videos))
        raise ValueError(
            f"multi-GT positives are absent from the candidate videos for {len(missing_videos)} queries; "
            f"first {first_query}: {missing_videos[first_query]}"
        )
    return targets


def coverage_recall_from_errors(errors, targets: dict[int, list[int]], ks=(1, 5, 10, 100)) -> tuple[float, ...]:
    """Macro recall of *all* verified GT videos retrieved in each top-K.

    ``errors`` has shape [query, video] and lower is better, matching the
    existing PRVR evaluators which pass ``-similarity``. For a query q:
    ``Recall@K(q) = |topK(q) intersect GT(q)| / |GT(q)|``.
    """
    array = errors.detach().cpu().numpy() if hasattr(errors, "detach") else np.asarray(errors)
    if array.ndim != 2:
        raise ValueError(f"expected [query, video] errors, got {array.shape}")
    if len(targets) != array.shape[0]:
        raise ValueError(f"multi-GT targets ({len(targets)}) do not match scores ({array.shape[0]})")
    ranks = np.argsort(array, axis=1)
    recalls = {k: [] for k in ks}
    for query_index in range(array.shape[0]):
        positives = set(targets[query_index])
        for k in ks:
            retrieved = set(ranks[query_index, :min(k, array.shape[1])])
            recalls[k].append(len(retrieved & positives) / len(positives))
    return tuple(100.0 * float(np.mean(recalls[k])) for k in ks)


def first_positive_rank_stats_from_errors(errors, targets: dict[int, list[int]]) -> tuple[float, float]:
    """Median/mean best-positive rank retained for legacy evaluator logs."""
    array = errors.detach().cpu().numpy() if hasattr(errors, "detach") else np.asarray(errors)
    ranks = np.argsort(array, axis=1)
    first_ranks = []
    for query_index in range(array.shape[0]):
        position = {video_index: rank + 1 for rank, video_index in enumerate(ranks[query_index])}
        first_ranks.append(min(position[index] for index in targets[query_index]))
    return float(np.median(first_ranks)), float(np.mean(first_ranks))


def average_precision_from_errors(errors, targets: dict[int, list[int]]) -> float:
    """Mean AP over all verified positives, for evaluators that log mAP."""
    array = errors.detach().cpu().numpy() if hasattr(errors, "detach") else np.asarray(errors)
    ranks = np.argsort(array, axis=1)
    aps = []
    for query_index in range(array.shape[0]):
        positives = set(targets[query_index])
        hits = 0
        precision_sum = 0.0
        for rank, video_index in enumerate(ranks[query_index], start=1):
            if video_index in positives:
                hits += 1
                precision_sum += hits / rank
        aps.append(precision_sum / len(positives))
    return float(np.mean(aps))
