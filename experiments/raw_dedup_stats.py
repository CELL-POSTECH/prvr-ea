"""Opt-in statistics for raw representation retrieval scores.

The model adapters call :func:`capture_from_env` immediately before their
unchanged max-over-representations reduction.  This module is evaluation-only:
it neither changes scores nor writes a full ``[query, repr, video]`` ranking.
Instead it stores small per-query scalar diagnostics and a compact summary.
"""

from __future__ import annotations

import atexit
import csv
import gzip
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict

import torch

try:
    import branch_ablation_recorder
except ImportError:
    branch_ablation_recorder = None


DEFAULT_DEPTH_KS = (1, 5, 10)
SUMMARY_FIELDS = [
    "method", "dataset", "branch", "metric", "statistic", "value",
    "repr_per_video", "raw_top_l", "num_queries", "coverage_pct",
]
BASE_PER_QUERY_FIELDS = [
    "method", "dataset", "branch", "query_index", "query_id", "gt_video_id",
    "repr_per_video", "raw_top_l",
]
TAIL_PER_QUERY_FIELDS = ["video_rank", "gt_first_raw_rank"]


def _configured_depth_ks() -> tuple[int, ...]:
    configured = os.environ.get("PRVR_RAW_DEDUP_UNIQUE_KS")
    if not configured:
        return DEFAULT_DEPTH_KS
    values = tuple(sorted({int(item) for item in configured.replace(";", ",").split(",") if item.strip()}))
    if not values or any(value < 1 for value in values):
        raise ValueError("PRVR_RAW_DEDUP_UNIQUE_KS must contain positive integers")
    return values


def _per_query_fields(depth_ks: tuple[int, ...]) -> list[str]:
    return [
        *BASE_PER_QUERY_FIELDS,
        *(f"raw_depth_unique_{k}" for k in depth_ks),
        *(f"unique_videos_at_l{k}" for k in depth_ks if k > 1),
        *(f"duplicate_rate_at_l{k}" for k in depth_ks if k > 1),
        *TAIL_PER_QUERY_FIELDS,
    ]


def _percentile(values: list[float], pct: int) -> float | str:
    if not values:
        return ""
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct / 100
    lo, hi = int(index), min(int(index) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def _stats(values: list[float]) -> tuple[tuple[str, float | str], ...]:
    if not values:
        return tuple((name, "") for name in ("min", "mean", "p50", "median", "p90", "p95", "p99", "max"))
    return (
        ("min", min(values)), ("mean", sum(values) / len(values)),
        ("p50", _percentile(values, 50)), ("median", _percentile(values, 50)),
        ("p90", _percentile(values, 90)), ("p95", _percentile(values, 95)),
        ("p99", _percentile(values, 99)), ("max", max(values)),
    )


class RawDedupSummary:
    """Accumulate raw-depth, duplicate, and GT-rank statistics by branch."""

    def __init__(self, output: str, per_query_output: str | None, method: str, dataset: str):
        self.output = Path(output)
        self.per_query_output = Path(per_query_output) if per_query_output else None
        self.method = method
        self.dataset = dataset
        self.counts: Dict[str, int] = defaultdict(int)
        self.depths: Dict[str, Dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
        self.duplicates: Dict[str, Dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
        self.video_ranks: Dict[str, list[float]] = defaultdict(list)
        self.gt_raw_ranks: Dict[str, list[float]] = defaultdict(list)
        self.reprs: Dict[str, int] = {}
        self.top_ls: Dict[str, int] = {}
        self.per_query_rows: list[dict[str, object]] = []
        self.depth_ks = _configured_depth_ks()
        self.query_ids, self.video_ids, self.video_index = self._load_metadata()
        self.offsets: Dict[str, int] = defaultdict(int)
        self.closed = False

    @staticmethod
    def _load_metadata() -> tuple[list[str], list[str], dict[str, int]]:
        """Read the unchanged test caption order used by every PRVR evaluator."""
        query_file = os.environ.get("PRVR_RAW_DEDUP_QUERY_FILE")
        if not query_file:
            return [], [], {}
        queries: list[str] = []
        videos: list[str] = []
        seen: set[str] = set()
        with Path(query_file).open(encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                query_id = stripped.split(" ", 1)[0]
                video_id = query_id.split("#", 1)[0]
                queries.append(query_id)
                if video_id not in seen:
                    seen.add(video_id)
                    videos.append(video_id)
        return queries, videos, {video_id: index for index, video_id in enumerate(videos)}

    def _batch_metadata(self, branch: str, q_count: int, num_videos: int) -> tuple[list[str], list[str], list[int]]:
        if not self.query_ids:
            return [""] * q_count, [""] * q_count, []
        if len(self.video_ids) != num_videos:
            raise RuntimeError(
                f"{branch}: evaluator supplied {num_videos} videos but test captions define "
                f"{len(self.video_ids)} unique videos; refusing to write incorrect GT ranks"
            )
        start = self.offsets[branch]
        stop = start + q_count
        if stop > len(self.query_ids):
            raise RuntimeError(
                f"{branch}: captured {stop} queries but test split has only {len(self.query_ids)} queries"
            )
        query_ids = self.query_ids[start:stop]
        gt_video_ids = [query_id.split("#", 1)[0] for query_id in query_ids]
        try:
            gt_indices = [self.video_index[video_id] for video_id in gt_video_ids]
        except KeyError as exc:
            raise RuntimeError(f"GT video is absent from evaluator context: {exc.args[0]}") from exc
        self.offsets[branch] = stop
        return query_ids, gt_video_ids, gt_indices

    def add(self, branch: str, raw_scores: torch.Tensor, repr_per_video: int) -> None:
        """Consume a ``[Q, R, V]`` raw score tensor before its original max.

        ``video_rank`` is the GT rank after max-pooling R representations per
        video. ``gt_first_raw_rank`` is the earliest rank of any GT-video
        representation before deduplication. Ties are counted as non-better,
        matching the usual optimistic retrieval-rank convention.
        """
        if raw_scores.ndim != 3:
            raise ValueError(f"Expected [Q,R,V] raw scores, got {tuple(raw_scores.shape)}")
        q_count, raw_repr, num_videos = raw_scores.shape
        if raw_repr != repr_per_video:
            raise ValueError(
                f"{branch}: expected {repr_per_video} repr/video, received {raw_repr}; "
                "do not silently analyse a different branch definition"
            )
        query_ids, gt_video_ids, gt_indices = self._batch_metadata(branch, q_count, num_videos)
        default_top_l = (max(self.depth_ks) - 1) * repr_per_video + 1
        requested_top_l = int(os.environ.get("PRVR_RAW_DEDUP_TOP_L", str(default_top_l)))
        if requested_top_l < max(self.depth_ks):
            raise ValueError("PRVR_RAW_DEDUP_TOP_L must be >= max(PRVR_RAW_DEDUP_UNIQUE_KS)")
        top_l = min(requested_top_l, raw_repr * num_videos)
        flat = raw_scores.reshape(q_count, -1)
        top_indices = torch.topk(flat, k=top_l, dim=1, largest=True, sorted=True).indices.cpu().tolist()

        video_ranks: list[int] = []
        gt_raw_ranks: list[int] = []
        if gt_indices:
            device = raw_scores.device
            batch_indices = torch.arange(q_count, device=device)
            gt_index_tensor = torch.tensor(gt_indices, device=device)
            video_scores = raw_scores.max(dim=1).values
            gt_video_scores = video_scores[batch_indices, gt_index_tensor]
            video_ranks = ((video_scores > gt_video_scores.unsqueeze(1)).sum(dim=1) + 1).cpu().tolist()
            gt_raw_scores = raw_scores[batch_indices, :, gt_index_tensor].max(dim=1).values
            gt_raw_ranks = ((flat > gt_raw_scores.unsqueeze(1)).sum(dim=1) + 1).cpu().tolist()

        self.reprs[branch] = repr_per_video
        self.top_ls[branch] = top_l
        checkpoints = {k: min((k - 1) * repr_per_video + 1, top_l) for k in self.depth_ks if k > 1}
        for row_index, row in enumerate(top_indices):
            seen: set[int] = set()
            depth: dict[int, int] = {}
            unique_at_l: dict[int, int] = {}
            for raw_rank, flat_index in enumerate(row, start=1):
                video_index = flat_index % num_videos
                if video_index not in seen:
                    seen.add(video_index)
                    if len(seen) in self.depth_ks:
                        depth[len(seen)] = raw_rank
                for k, checkpoint in checkpoints.items():
                    if raw_rank == checkpoint:
                        unique_at_l[k] = len(seen)
            self.counts[branch] += 1
            for k in self.depth_ks:
                if k in depth:
                    self.depths[branch][k].append(float(depth[k]))
            duplicate_rates = {}
            for k, checkpoint in checkpoints.items():
                # ``checkpoints`` can coincide when the requested raw top-L
                # clips two target depths (e.g. 10 and 20) at the same L.
                # Derive the rate from the unique count at that depth instead
                # of accumulating duplicate counters keyed by the checkpoint.
                # This is both simpler and bounded in [0, 100].
                unique_count = unique_at_l.get(k, 0)
                duplicate_rates[k] = 100.0 * (checkpoint - unique_count) / checkpoint
                self.duplicates[branch][k].append(duplicate_rates[k])
            if video_ranks:
                self.video_ranks[branch].append(float(video_ranks[row_index]))
                self.gt_raw_ranks[branch].append(float(gt_raw_ranks[row_index]))
            if self.per_query_output is not None:
                per_query_row = {
                    "method": self.method, "dataset": self.dataset, "branch": branch,
                    "query_index": self.offsets[branch] - q_count + row_index,
                    "query_id": query_ids[row_index], "gt_video_id": gt_video_ids[row_index],
                    "repr_per_video": repr_per_video, "raw_top_l": top_l,
                    "video_rank": video_ranks[row_index] if video_ranks else "",
                    "gt_first_raw_rank": gt_raw_ranks[row_index] if gt_raw_ranks else "",
                }
                for k in self.depth_ks:
                    per_query_row[f"raw_depth_unique_{k}"] = depth.get(k, "")
                    if k > 1:
                        per_query_row[f"unique_videos_at_l{k}"] = unique_at_l.get(k, "")
                        per_query_row[f"duplicate_rate_at_l{k}"] = duplicate_rates.get(k, "")
                self.per_query_rows.append(per_query_row)

    def _write_row(self, writer: csv.DictWriter, branch: str, metric: str, statistic: str, value: float | str,
                   coverage: float = 100.0) -> None:
        writer.writerow({
            "method": self.method, "dataset": self.dataset, "branch": branch,
            "metric": metric, "statistic": statistic, "value": value,
            "repr_per_video": self.reprs[branch], "raw_top_l": self.top_ls[branch],
            "num_queries": self.counts[branch], "coverage_pct": coverage,
        })

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with self.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            for branch in sorted(self.counts):
                count = self.counts[branch]
                for k in self.depth_ks:
                    values = self.depths[branch][k]
                    coverage = 100.0 * len(values) / count if count else 0.0
                    for statistic, value in _stats(values):
                        self._write_row(writer, branch, f"raw_depth_for_unique_video_at_{k}", statistic, value, coverage)
                for k in self.depth_ks:
                    if k == 1:
                        continue
                    for statistic, value in _stats(self.duplicates[branch][k]):
                        self._write_row(writer, branch, f"duplicate_rate_at_l{k}", statistic, value)
                for metric, values in (("video_rank", self.video_ranks[branch]),
                                       ("gt_first_raw_rank", self.gt_raw_ranks[branch])):
                    for statistic, value in _stats(values):
                        self._write_row(writer, branch, metric, statistic, value,
                                        100.0 * len(values) / count if count else 0.0)
                    if metric == "video_rank" and values:
                        for k in self.depth_ks:
                            recall = 100.0 * sum(rank <= k for rank in values) / len(values)
                            self._write_row(writer, branch, f"video_recall_at_{k}", "value", recall)
        if self.per_query_output is not None:
            self.per_query_output.parent.mkdir(parents=True, exist_ok=True)
            opener = gzip.open if ".gz" in self.per_query_output.suffixes else open
            with opener(self.per_query_output, "wt", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=_per_query_fields(self.depth_ks))
                writer.writeheader()
                writer.writerows(self.per_query_rows)


_SUMMARY: RawDedupSummary | None = None
_SEQUENCE_INDEX = 0
_ABLATION_SEQUENCE_INDEX = 0


def get_summary() -> RawDedupSummary | None:
    """Return the process-wide opt-in collector, or ``None`` during normal eval."""
    global _SUMMARY
    output = os.environ.get("PRVR_RAW_DEDUP_SUMMARY")
    if not output:
        return None
    if _SUMMARY is None:
        _SUMMARY = RawDedupSummary(
            output, os.environ.get("PRVR_RAW_DEDUP_PER_QUERY"),
            os.environ.get("PRVR_RAW_DEDUP_METHOD", ""),
            os.environ.get("PRVR_RAW_DEDUP_DATASET", ""),
        )
        atexit.register(_SUMMARY.close)
    return _SUMMARY


def capture(branch: str, raw_scores: torch.Tensor, repr_per_video: int) -> None:
    if branch_ablation_recorder is not None:
        branch_ablation_recorder.capture_raw(branch, raw_scores, repr_per_video)
    summary = get_summary()
    if summary is not None:
        summary.add(branch, raw_scores, repr_per_video)


def capture_vector_pools(left_query, left_context, right_query, right_context,
                         left_weight=None, right_weight=None) -> None:
    """Forward opt-in representation-level pooling data to branch ablation."""
    if branch_ablation_recorder is not None:
        if left_weight is None:
            left_weight = float(os.environ.get("PRVR_BRANCH_ABLATION_LEFT_WEIGHT", "0.7"))
        if right_weight is None:
            right_weight = float(os.environ.get("PRVR_BRANCH_ABLATION_RIGHT_WEIGHT", "0.3"))
        branch_ablation_recorder.capture_vector_pools(
            left_query, left_context, right_query, right_context,
            left_weight, right_weight,
        )


def capture_from_env(raw_scores: torch.Tensor) -> None:
    """Capture a [Q,R,V] score tensor using the launcher-provided repr map."""
    if raw_scores.ndim != 3:
        return
    if branch_ablation_recorder is not None and branch_ablation_recorder.enabled():
        global _ABLATION_SEQUENCE_INDEX
        sequence = json.loads(os.environ.get("PRVR_BRANCH_ABLATION_SEQUENCE", "[]"))
        if sequence:
            branch = sequence[_ABLATION_SEQUENCE_INDEX % len(sequence)]
            _ABLATION_SEQUENCE_INDEX += 1
            branch_ablation_recorder.capture_raw(branch, raw_scores, raw_scores.shape[1])
        else:
            mapping = json.loads(os.environ.get("PRVR_BRANCH_ABLATION_BRANCHES", "{}"))
            branch = mapping.get(str(raw_scores.shape[1]))
            if branch:
                branch_ablation_recorder.capture_raw(branch, raw_scores, raw_scores.shape[1])
    if not os.environ.get("PRVR_RAW_DEDUP_SUMMARY"):
        return
    global _SEQUENCE_INDEX
    sequence = json.loads(os.environ.get("PRVR_RAW_DEDUP_SEQUENCE", "[]"))
    if sequence:
        branch = sequence[_SEQUENCE_INDEX % len(sequence)]
        _SEQUENCE_INDEX += 1
        capture(branch, raw_scores, raw_scores.shape[1])
        return
    mapping = json.loads(os.environ.get("PRVR_RAW_DEDUP_BRANCHES", "{}"))
    repr_per_video = raw_scores.shape[1]
    branch = mapping.get(str(repr_per_video))
    if branch:
        capture(branch, raw_scores, repr_per_video)
