"""Opt-in dual-branch ablation recorder for PRVR evaluation.

This module is intentionally evaluation-only.  Model/evaluator code calls into
the existing branch-rank and raw-dedup hooks; when ``PRVR_BRANCH_ABLATION_OUTPUT``
is set, those hooks forward the already-computed score tensors here.

The ablations are:
  - Base: original weighted branch score used by the evaluator.
  - Clip-level branch: left/clip branch max-sim score.
  - Frame-level branch: right/frame branch max-sim score.
  - Branch Mean Pooling: mean of all raw representation-level similarities.
  - Weighed Branch Mean Pooling: config-weighted mean of branch-wise raw
    representation-level similarities.

The mean-pooling scores are computed from the normalized representation-level
similarities captured immediately before each model's unchanged max reduction.
This is equivalent to scoring the unnormalized mean-pooled branch representation
with the same query vector used by the original max-sim primitive.
"""

from __future__ import annotations

import atexit
import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch


SUMMARY_FIELDS = [
    "Method",
    "Base",
    "Clip-level branch",
    "Frame-level branch",
    "Branch Mean Pooling",
    "Weighed Branch Mean Pooling",
]
LONG_FIELDS = ["method", "dataset", "condition", "r1", "r5", "r10", "num_queries", "checkpoint"]


def _video_id(query_id: str) -> str:
    return str(query_id).split("#", 1)[0]


def _find_video_index(video_ids: list[str], query_id: str) -> int:
    qid = _video_id(query_id)
    for candidate in (qid, qid[2:] if qid.startswith("v_") else None, "v_" + qid):
        if candidate is None:
            continue
        try:
            return video_ids.index(candidate)
        except ValueError:
            pass
    raise KeyError(f"No evaluation video for query {query_id!r}")


def _to_tensor(scores) -> torch.Tensor:
    if isinstance(scores, torch.Tensor):
        return scores.detach().float().cpu()
    return torch.as_tensor(scores, dtype=torch.float32).detach().cpu()


def _recall(scores: torch.Tensor, query_metas: list[str], video_ids: list[str]) -> tuple[float, float, float]:
    gt = torch.tensor([_find_video_index(video_ids, q) for q in query_metas], dtype=torch.long)
    row = torch.arange(scores.shape[0])
    gt_scores = scores[row, gt]
    ranks = (scores > gt_scores.unsqueeze(1)).sum(dim=1) + 1
    denom = float(len(query_metas)) if query_metas else 1.0
    return tuple(100.0 * (ranks <= k).sum().item() / denom for k in (1, 5, 10))


def _format_triplet(values: tuple[float, float, float]) -> str:
    return "/".join(f"{v:.4f}" for v in values)


class BranchAblationRecorder:
    def __init__(self) -> None:
        self.output = Path(os.environ["PRVR_BRANCH_ABLATION_OUTPUT"])
        self.long_output = Path(os.environ.get(
            "PRVR_BRANCH_ABLATION_LONG_OUTPUT",
            str(self.output.with_name(self.output.stem + "_long.csv")),
        ))
        self.method = os.environ.get("PRVR_BRANCH_ABLATION_METHOD", "")
        self.dataset = os.environ.get("PRVR_BRANCH_ABLATION_DATASET", "")
        self.checkpoint = os.environ.get("PRVR_BRANCH_ABLATION_CHECKPOINT", "")
        self.video_ids: list[str] | None = None
        self.query_metas: list[str] = []
        self.left_name = "clip"
        self.right_name = "frame"
        self.left_weight = 0.7
        self.right_weight = 0.3
        self.max_scores: Dict[str, list[torch.Tensor]] = defaultdict(list)
        self.raw_mean_scores: Dict[str, list[torch.Tensor]] = defaultdict(list)
        self.repr_counts: Dict[str, int] = {}
        self.closed = False

    def capture_raw(self, branch: str, raw_scores: torch.Tensor, repr_per_video: int) -> None:
        if raw_scores.ndim != 3:
            return
        self.raw_mean_scores[branch].append(raw_scores.detach().float().mean(dim=1).cpu())
        self.repr_counts[branch] = int(repr_per_video)

    def add_scores(
        self,
        video_metas: Iterable[str],
        query_metas: Iterable[str],
        left_scores,
        right_scores,
        left_name: str = "clip",
        right_name: str = "frame",
        left_weight: float = 0.7,
        right_weight: float = 0.3,
        exact_scores=None,
    ) -> None:
        video_ids = [str(v) for v in video_metas]
        if self.video_ids is None:
            self.video_ids = video_ids
        elif self.video_ids != video_ids:
            raise RuntimeError("Ablation recorder received inconsistent video ordering")
        query_list = [str(q) for q in query_metas]
        left = _to_tensor(left_scores)
        right = _to_tensor(right_scores)
        exact = left_weight * left + right_weight * right if exact_scores is None else _to_tensor(exact_scores)
        self.left_name = left_name
        self.right_name = right_name
        self.left_weight = float(left_weight)
        self.right_weight = float(right_weight)
        self.query_metas.extend(query_list)
        self.max_scores["base"].append(exact)
        self.max_scores["clip"].append(left)
        self.max_scores["frame"].append(right)

    def _concat_raw_mean(self, branch: str, expected_rows: int) -> torch.Tensor | None:
        if branch not in self.raw_mean_scores:
            return None
        merged = torch.cat(self.raw_mean_scores[branch], dim=0)
        if merged.shape[0] < expected_rows:
            raise RuntimeError(
                f"Ablation recorder has only {merged.shape[0]} raw rows for {branch}; "
                f"need {expected_rows}"
            )
        return merged[:expected_rows]

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if not self.query_metas or self.video_ids is None:
            return

        scores: Dict[str, torch.Tensor] = {
            "Base": torch.cat(self.max_scores["base"], dim=0),
            "Clip-level branch": torch.cat(self.max_scores["clip"], dim=0),
            "Frame-level branch": torch.cat(self.max_scores["frame"], dim=0),
        }
        n_rows = len(self.query_metas)
        left_mean = self._concat_raw_mean(self.left_name, n_rows)
        right_mean = self._concat_raw_mean(self.right_name, n_rows)
        if left_mean is None:
            left_mean = scores["Clip-level branch"]
            self.repr_counts.setdefault(self.left_name, 1)
        if right_mean is None:
            right_mean = scores["Frame-level branch"]
            self.repr_counts.setdefault(self.right_name, 1)
        left_repr = self.repr_counts.get(self.left_name, 1)
        right_repr = self.repr_counts.get(self.right_name, 1)
        scores["Branch Mean Pooling"] = (
            left_mean * float(left_repr) + right_mean * float(right_repr)
        ) / float(left_repr + right_repr)
        scores["Weighed Branch Mean Pooling"] = (
            self.left_weight * left_mean + self.right_weight * right_mean
        )

        metrics = {
            condition: _recall(matrix, self.query_metas, self.video_ids)
            for condition, matrix in scores.items()
        }
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with self.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerow({
                "Method": self.method,
                "Base": _format_triplet(metrics["Base"]),
                "Clip-level branch": _format_triplet(metrics["Clip-level branch"]),
                "Frame-level branch": _format_triplet(metrics["Frame-level branch"]),
                "Branch Mean Pooling": _format_triplet(metrics["Branch Mean Pooling"]),
                "Weighed Branch Mean Pooling": _format_triplet(metrics["Weighed Branch Mean Pooling"]),
            })
        with self.long_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=LONG_FIELDS)
            writer.writeheader()
            for condition, values in metrics.items():
                writer.writerow({
                    "method": self.method,
                    "dataset": self.dataset,
                    "condition": condition,
                    "r1": values[0],
                    "r5": values[1],
                    "r10": values[2],
                    "num_queries": len(self.query_metas),
                    "checkpoint": self.checkpoint,
                })


_RECORDER: BranchAblationRecorder | None = None


def enabled() -> bool:
    return bool(os.environ.get("PRVR_BRANCH_ABLATION_OUTPUT"))


def get_recorder() -> BranchAblationRecorder | None:
    global _RECORDER
    if not enabled():
        return None
    if _RECORDER is None:
        _RECORDER = BranchAblationRecorder()
        atexit.register(_RECORDER.close)
    return _RECORDER


def capture_raw(branch: str, raw_scores: torch.Tensor, repr_per_video: int) -> None:
    recorder = get_recorder()
    if recorder is not None:
        recorder.capture_raw(branch, raw_scores, repr_per_video)


def record_full(
    video_metas,
    query_metas,
    left_scores,
    right_scores,
    left_name: str = "clip",
    right_name: str = "frame",
    left_weight: float = 0.7,
    right_weight: float = 0.3,
    exact_scores=None,
) -> None:
    recorder = get_recorder()
    if recorder is not None:
        recorder.add_scores(
            video_metas,
            query_metas,
            left_scores,
            right_scores,
            left_name,
            right_name,
            left_weight,
            right_weight,
            exact_scores,
        )


class ProxyRecorder:
    """Recorder-compatible adapter used by config-style validators."""

    def __init__(self, video_metas, left_name="clip", right_name="frame", left_weight=None, right_weight=None):
        self.video_metas = video_metas
        self.left_name = left_name
        self.right_name = right_name
        self.left_weight = 0.7 if left_weight is None else float(left_weight)
        self.right_weight = 0.3 if right_weight is None else float(right_weight)

    def add(self, left_scores, right_scores, exact_scores, query_metas):
        record_full(
            self.video_metas,
            query_metas,
            left_scores,
            right_scores,
            self.left_name,
            self.right_name,
            self.left_weight,
            self.right_weight,
            exact_scores,
        )

    def close(self):
        recorder = get_recorder()
        if recorder is not None:
            recorder.close()
