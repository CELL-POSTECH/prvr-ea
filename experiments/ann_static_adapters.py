"""Small model-family adapters for :mod:`ann_static_benchmark`.

Each adapter only exposes the already encoded static context vectors and the
same query vectors used by the original evaluator.  It is called exclusively
by the opt-in ANN path.
"""
from __future__ import annotations

import os

import torch

from ann_static_benchmark import run


def _net(model):
    return model.module if hasattr(model, "module") else model


def _paths(cfg, method: str):
    output = os.environ.get("PRVR_STATIC_ANN_OUTPUT")
    if not output:
        raise RuntimeError("PRVR_STATIC_ANN_OUTPUT is required")
    index_dir = os.environ.get(
        "PRVR_STATIC_ANN_INDEX_DIR",
        os.path.join(cfg["model_root"], "ann_static_indices"),
    )
    checkpoint = os.environ.get("PRVR_STATIC_ANN_CHECKPOINT", "")
    if not checkpoint:
        raise RuntimeError("PRVR_STATIC_ANN_CHECKPOINT is required")
    return output, index_dir, checkpoint


def gmm_style(*, model, validator, context_loader, query_loader, cfg, method: str, batch_to_gpu):
    """Adapter shared by GMMFormer(-v2), HLFormer, DreamPRVR and Holmes."""
    net = _net(model)
    net.eval()
    context = validator.compute_context_info(model, context_loader)

    def batches():
        for batch in query_loader:
            batch = batch_to_gpu(batch)
            query = net.encode_query(batch[0], batch[1])
            yield query, query, batch[-1]

    output, index_dir, checkpoint = _paths(cfg, method)
    return run(
        method=method,
        checkpoint=checkpoint,
        output=output,
        index_dir=index_dir,
        video_ids=context["video_metas"],
        left_bank=context["video_proposal_feat"],
        right_bank=context["video_feat"],
        query_batches=batches(),
        left_weight=float(cfg["clip_scale_w"]),
        right_weight=float(cfg["frame_scale_w"]),
        left_name="clip",
        right_name="frame",
    )


def dual_query_gmm_style(*, model, validator, context_loader, query_loader, cfg, method: str, batch_to_gpu,
                         query_from_batch, context_kwargs=None):
    """Static-bank adapter for models with distinct query vectors per branch."""
    net = _net(model)
    net.eval()
    context = validator.compute_context_info(model, context_loader, **(context_kwargs or {}))

    def batches():
        for batch in query_loader:
            batch = batch_to_gpu(batch)
            left_query, right_query, query_ids = query_from_batch(net, batch)
            yield left_query, right_query, query_ids

    output, index_dir, checkpoint = _paths(cfg, method)
    return run(
        method=method,
        checkpoint=checkpoint,
        output=output,
        index_dir=index_dir,
        video_ids=context["video_metas"],
        left_bank=context["video_proposal_feat"],
        right_bank=context["video_feat"],
        query_batches=batches(),
        left_weight=float(cfg["clip_scale_w"]),
        right_weight=float(cfg["frame_scale_w"]),
        left_name="clip",
        right_name="frame",
    )
