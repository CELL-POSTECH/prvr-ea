"""Opt-in, streaming recorder for PRVR two-branch evaluation ranks.

The recorder is deliberately separate from the model code.  Evaluators call it
only when PRVR_BRANCH_RANK_OUTPUT is set, so normal training/evaluation uses
the original code path and performs no CSV work.
"""
import csv
import os
from pathlib import Path

import torch

try:
    import branch_ablation_recorder
except ImportError:
    branch_ablation_recorder = None


def _video_id(query_id):
    return str(query_id).split("#", 1)[0]


def _find_video_index(video_ids, query_id):
    qid = _video_id(query_id)
    try:
        return video_ids.index(qid)
    except ValueError:
        # ActivityNet releases sometimes disagree only on the v_ prefix.
        if qid.startswith("v_"):
            try:
                return video_ids.index(qid[2:])
            except ValueError:
                pass
        prefixed = "v_" + qid
        try:
            return video_ids.index(prefixed)
        except ValueError as exc:
            raise KeyError(f"No evaluation video for query {query_id!r}") from exc


class BranchRankRecorder:
    """Write one row/query without retaining full QxV score matrices."""

    fields = [
        "model", "dataset", "feature", "checkpoint", "query_id", "gt_video_id",
        "num_candidates", "left_name", "right_name", "left_weight", "right_weight",
        "rank_exact", "rank_left", "rank_right", "delta_left", "delta_right",
        "exact_top1_video_id", "left_top1_video_id", "right_top1_video_id",
        "exact_gt_score", "left_gt_score", "right_gt_score",
    ]

    def __init__(self, video_metas, left_name="clip", right_name="frame", left_weight=None, right_weight=None):
        output = os.environ.get("PRVR_BRANCH_RANK_OUTPUT")
        self.enabled = bool(output)
        if not self.enabled:
            return
        self.video_ids = [str(x) for x in video_metas]
        self.left_name = left_name
        self.right_name = right_name
        self.left_weight = left_weight
        self.right_weight = right_weight
        self.output = Path(output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.output.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=self.fields)
        self.writer.writeheader()

    @staticmethod
    def _ranks(scores, gt):
        # Upstream evaluation calls argsort(-similarity): lower error is better.
        order = torch.argsort(-scores, dim=1)
        ranks = torch.nonzero(order.eq(gt[:, None]), as_tuple=False)[:, 1] + 1
        return ranks, order[:, 0]

    def add(self, left_scores, right_scores, exact_scores, query_metas):
        if not self.enabled:
            return
        if len(query_metas) != left_scores.shape[0]:
            raise ValueError("query metadata and score batch length differ")
        gt = torch.tensor([_find_video_index(self.video_ids, q) for q in query_metas],
                          device=left_scores.device, dtype=torch.long)
        exact_rank, exact_top1 = self._ranks(exact_scores, gt)
        left_rank, left_top1 = self._ranks(left_scores, gt)
        right_rank, right_top1 = self._ranks(right_scores, gt)
        exact_rank = exact_rank.detach().cpu().tolist()
        left_rank = left_rank.detach().cpu().tolist()
        right_rank = right_rank.detach().cpu().tolist()
        exact_top1 = exact_top1.detach().cpu().tolist()
        left_top1 = left_top1.detach().cpu().tolist()
        right_top1 = right_top1.detach().cpu().tolist()
        exact_gt = exact_scores[torch.arange(len(query_metas), device=gt.device), gt].detach().cpu().tolist()
        left_gt = left_scores[torch.arange(len(query_metas), device=gt.device), gt].detach().cpu().tolist()
        right_gt = right_scores[torch.arange(len(query_metas), device=gt.device), gt].detach().cpu().tolist()
        common = {
            "model": os.environ.get("PRVR_BRANCH_RANK_MODEL", ""),
            "dataset": os.environ.get("PRVR_BRANCH_RANK_DATASET", ""),
            "feature": os.environ.get("PRVR_BRANCH_RANK_FEATURE", "clip"),
            "checkpoint": os.environ.get("PRVR_BRANCH_RANK_CHECKPOINT", ""),
            "num_candidates": len(self.video_ids),
            "left_name": self.left_name, "right_name": self.right_name,
            "left_weight": self.left_weight, "right_weight": self.right_weight,
        }
        for i, query_id in enumerate(query_metas):
            self.writer.writerow({
                **common, "query_id": query_id, "gt_video_id": _video_id(query_id),
                "rank_exact": exact_rank[i], "rank_left": left_rank[i], "rank_right": right_rank[i],
                "delta_left": left_rank[i] - exact_rank[i],
                "delta_right": right_rank[i] - exact_rank[i],
                "exact_top1_video_id": self.video_ids[exact_top1[i]],
                "left_top1_video_id": self.video_ids[left_top1[i]],
                "right_top1_video_id": self.video_ids[right_top1[i]],
                "exact_gt_score": exact_gt[i], "left_gt_score": left_gt[i], "right_gt_score": right_gt[i],
            })
        self.handle.flush()

    def close(self):
        if self.enabled:
            self.handle.close()


class BranchTopKRecorder:
    """Write exact/branch top-K video lists for every evaluated query."""

    def __init__(self, video_metas, left_name="clip", right_name="frame", left_weight=None, right_weight=None):
        output = os.environ.get("PRVR_BRANCH_TOPK_OUTPUT")
        self.enabled = bool(output)
        if not self.enabled:
            return
        self.video_ids = [str(x) for x in video_metas]
        self.left_name = left_name
        self.right_name = right_name
        self.left_weight = left_weight
        self.right_weight = right_weight
        self.top_k = int(os.environ.get("PRVR_BRANCH_TOPK_K", "100"))
        self.exact_k = int(os.environ.get("PRVR_BRANCH_TOPK_EXACT_K", "10"))
        if self.top_k < 1 or self.exact_k < 1:
            raise ValueError("PRVR_BRANCH_TOPK_K and PRVR_BRANCH_TOPK_EXACT_K must be positive")
        if self.exact_k > self.top_k:
            raise ValueError("exact_k must be <= top_k")
        label = f"top{self.top_k}"
        exact_label = f"top{self.exact_k}"
        self.fields = [
            "model", "dataset", "feature", "checkpoint", "query_id", "num_candidates",
            "top_k", "exact_k", "clip_branch_name", "frame_branch_name",
            "clip_weight", "frame_weight",
            f"exact_{label}_video_ids", f"clip_{label}_video_ids", f"frame_{label}_video_ids",
            f"exact_{label}_scores", f"clip_{label}_scores", f"frame_{label}_scores",
            f"exact_{label}_clip_ranks", f"exact_{label}_frame_ranks",
            f"exact_{exact_label}_clip_max_rank", f"exact_{exact_label}_frame_max_rank",
            f"exact_{exact_label}_clip_coverage_at_{self.top_k}",
            f"exact_{exact_label}_frame_coverage_at_{self.top_k}",
            f"exact_{label}_clip_max_rank", f"exact_{label}_frame_max_rank",
            f"exact_{label}_clip_coverage_at_{self.top_k}",
            f"exact_{label}_frame_coverage_at_{self.top_k}",
        ]
        self.label = label
        self.exact_label = exact_label
        self.output = Path(output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.output.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=self.fields)
        self.writer.writeheader()

    @staticmethod
    def _rank_matrix(scores):
        order = torch.argsort(-scores, dim=1)
        ranks = torch.empty_like(order)
        positions = torch.arange(1, scores.shape[1] + 1, device=scores.device).expand_as(order)
        ranks.scatter_(1, order, positions)
        return order, ranks

    def _video_list(self, indices):
        return "|".join(self.video_ids[int(index)] for index in indices)

    @staticmethod
    def _score_list(scores):
        return "|".join(f"{float(score):.8g}" for score in scores)

    @staticmethod
    def _rank_list(ranks):
        return "|".join(str(int(rank)) for rank in ranks)

    def add(self, left_scores, right_scores, exact_scores, query_metas):
        if not self.enabled:
            return
        if len(query_metas) != left_scores.shape[0]:
            raise ValueError("query metadata and score batch length differ")
        top_k = min(self.top_k, left_scores.shape[1])
        exact_k = min(self.exact_k, top_k)
        exact_order, _ = self._rank_matrix(exact_scores)
        left_order, left_ranks = self._rank_matrix(left_scores)
        right_order, right_ranks = self._rank_matrix(right_scores)
        exact_top = exact_order[:, :top_k]
        left_top = left_order[:, :top_k]
        right_top = right_order[:, :top_k]
        exact_top_scores = exact_scores.gather(1, exact_top)
        left_top_scores = left_scores.gather(1, left_top)
        right_top_scores = right_scores.gather(1, right_top)
        exact_focus = exact_order[:, :exact_k]
        exact_top_left_ranks = left_ranks.gather(1, exact_top)
        exact_top_right_ranks = right_ranks.gather(1, exact_top)
        exact_focus_left_ranks = left_ranks.gather(1, exact_focus)
        exact_focus_right_ranks = right_ranks.gather(1, exact_focus)
        common = {
            "model": os.environ.get("PRVR_BRANCH_TOPK_MODEL", ""),
            "dataset": os.environ.get("PRVR_BRANCH_TOPK_DATASET", ""),
            "feature": os.environ.get("PRVR_BRANCH_TOPK_FEATURE", "clip"),
            "checkpoint": os.environ.get("PRVR_BRANCH_TOPK_CHECKPOINT", ""),
            "num_candidates": len(self.video_ids),
            "top_k": top_k,
            "exact_k": exact_k,
            "clip_branch_name": self.left_name,
            "frame_branch_name": self.right_name,
            "clip_weight": self.left_weight,
            "frame_weight": self.right_weight,
        }
        exact_top = exact_top.detach().cpu().tolist()
        left_top = left_top.detach().cpu().tolist()
        right_top = right_top.detach().cpu().tolist()
        exact_top_scores = exact_top_scores.detach().cpu().tolist()
        left_top_scores = left_top_scores.detach().cpu().tolist()
        right_top_scores = right_top_scores.detach().cpu().tolist()
        exact_top_left_ranks = exact_top_left_ranks.detach().cpu().tolist()
        exact_top_right_ranks = exact_top_right_ranks.detach().cpu().tolist()
        exact_focus_left_ranks = exact_focus_left_ranks.detach().cpu().tolist()
        exact_focus_right_ranks = exact_focus_right_ranks.detach().cpu().tolist()
        for i, query_id in enumerate(query_metas):
            self.writer.writerow({
                **common,
                "query_id": query_id,
                f"exact_{self.label}_video_ids": self._video_list(exact_top[i]),
                f"clip_{self.label}_video_ids": self._video_list(left_top[i]),
                f"frame_{self.label}_video_ids": self._video_list(right_top[i]),
                f"exact_{self.label}_scores": self._score_list(exact_top_scores[i]),
                f"clip_{self.label}_scores": self._score_list(left_top_scores[i]),
                f"frame_{self.label}_scores": self._score_list(right_top_scores[i]),
                f"exact_{self.label}_clip_ranks": self._rank_list(exact_top_left_ranks[i]),
                f"exact_{self.label}_frame_ranks": self._rank_list(exact_top_right_ranks[i]),
                f"exact_{self.exact_label}_clip_max_rank": max(exact_focus_left_ranks[i]),
                f"exact_{self.exact_label}_frame_max_rank": max(exact_focus_right_ranks[i]),
                f"exact_{self.exact_label}_clip_coverage_at_{top_k}": sum(rank <= top_k for rank in exact_focus_left_ranks[i]),
                f"exact_{self.exact_label}_frame_coverage_at_{top_k}": sum(rank <= top_k for rank in exact_focus_right_ranks[i]),
                f"exact_{self.label}_clip_max_rank": max(exact_top_left_ranks[i]),
                f"exact_{self.label}_frame_max_rank": max(exact_top_right_ranks[i]),
                f"exact_{self.label}_clip_coverage_at_{top_k}": sum(rank <= top_k for rank in exact_top_left_ranks[i]),
                f"exact_{self.label}_frame_coverage_at_{top_k}": sum(rank <= top_k for rank in exact_top_right_ranks[i]),
            })
        self.handle.flush()

    def close(self):
        if self.enabled:
            self.handle.close()


def make_recorder(video_metas, left_name="clip", right_name="frame", left_weight=None, right_weight=None):
    if os.environ.get("PRVR_BRANCH_TOPK_OUTPUT"):
        return BranchTopKRecorder(video_metas, left_name, right_name, left_weight, right_weight)
    if not os.environ.get("PRVR_BRANCH_RANK_OUTPUT"):
        if branch_ablation_recorder is not None and branch_ablation_recorder.enabled():
            return branch_ablation_recorder.ProxyRecorder(
                video_metas, left_name, right_name, left_weight, right_weight
            )
        return None
    return BranchRankRecorder(video_metas, left_name, right_name, left_weight, right_weight)


def record_full(video_metas, query_metas, left_scores, right_scores, left_name="clip", right_name="frame", left_weight=0.7, right_weight=0.3, exact_scores=None):
    """Record a full score matrix already materialized by legacy evaluators."""
    if os.environ.get("PRVR_BRANCH_TOPK_OUTPUT"):
        recorder = BranchTopKRecorder(video_metas, left_name, right_name, left_weight, right_weight)
        left = torch.as_tensor(left_scores)
        right = torch.as_tensor(right_scores, device=left.device)
        exact = left_weight * left + right_weight * right if exact_scores is None else torch.as_tensor(exact_scores, device=left.device)
        recorder.add(left, right, exact, query_metas)
        recorder.close()
        return
    if branch_ablation_recorder is not None and branch_ablation_recorder.enabled():
        branch_ablation_recorder.record_full(
            video_metas, query_metas, left_scores, right_scores,
            left_name, right_name, left_weight, right_weight, exact_scores,
        )
        if not os.environ.get("PRVR_BRANCH_RANK_OUTPUT"):
            return
    recorder = make_recorder(video_metas, left_name, right_name, left_weight, right_weight)
    if recorder is None:
        return
    left = torch.as_tensor(left_scores)
    right = torch.as_tensor(right_scores, device=left.device)
    exact = left_weight * left + right_weight * right if exact_scores is None else torch.as_tensor(exact_scores, device=left.device)
    recorder.add(left, right, exact, query_metas)
    recorder.close()
