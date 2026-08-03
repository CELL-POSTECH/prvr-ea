#!/usr/bin/env python3
"""Dump top-k rankings from PRVR checkpoints, then build pseudo-GT candidates.

This is a one-command wrapper around:
  1. checkpoint inference for user-specified PRVR methods
  2. pure CLIP max-frame top-k computation
  3. PRVR+CLIP top-k agreement candidate construction

Generic example:
    conda run -n prvr python scripts/build_pseudo_gt_from_ckpts.py \
      --dataset activitynet \
      --prvr-model method_a=dreamprvr=/path/to/method_a.ckpt \
      --prvr-model method_b=gmmformer=/path/to/method_b.ckpt \
      --prvr-model method_c=hlformer=/path/to/method_c.ckpt \
      --prvr-model method_d=holmes=/path/to/method_d.ckpt

TVR remains the default. Use ``--dataset activitynet`` or ``--dataset charades``
for the other validation splits.
Run with the prvr conda environment.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = REPO_ROOT / "all_prvr"
DATA_ROOT = Path(os.environ.get("PRVR_DATA_ROOT", REPO_ROOT / "datasets"))
DATASET_INTERNAL_NAMES = {
    "tvr": "tvr_clip",
    "activitynet": "act_clip",
    "charades": "cha_clip",
    "msrvtt": "msrvtt_clip",
}
DATASET_DEFAULTS = {
    "tvr": {
        "dreamprvr_ckpt": MODEL_ROOT / "CVPR26-DreamPRVR/results/clip/tvr/DreamPRVR/best.ckpt",
        "ms_sl_ckpt": MODEL_ROOT / "ms-sl/results/tvr/tvr-tvr_clip_ms_sl-2026_06_11_09_55_13/model.ckpt",
        "gmmformer_ckpt": MODEL_ROOT / "GMMFormer_v2/results/clip/tvr/gmmformer_v2/best.ckpt",
        "hlformer_ckpt": MODEL_ROOT / "ICCV25-HLFormer/results/clip/tvr/HLFormer/best.ckpt",
        "holmes_ckpt": MODEL_ROOT / "ICML26-Holmes/results/clip/tvr/Holmes/20260721-155445/best.ckpt",
        "output": REPO_ROOT / "outputs/upstream/pseudo_gt_candidates.tvr.jsonl",
        "rank_dir": REPO_ROOT / "outputs/tvr_rankings_from_ckpts",
        "pure_clip_cache": REPO_ROOT / "outputs/tvr_rankings_from_ckpts/pure_clip_frame_topP1000_top100.jsonl",
        "frame_source": "frames",
        "sampled_frame_root": None,
        "sampled_frame_layout": "video_time",
        "pseudo_window_mode": "fixed",
        "pseudo_window_sec": 4.0,
        "pseudo_window_min_sec": 4.0,
        "pseudo_window_max_sec": 20.0,
    },
    "activitynet": {
        "dreamprvr_ckpt": MODEL_ROOT / "CVPR26-DreamPRVR/results/clip/activitynet/DreamPRVR/best.ckpt",
        "ms_sl_ckpt": MODEL_ROOT / "ms-sl/results/clip/activitynet/activitynet-act_clip-2026_07_21_17_44_33/model.ckpt",
        "gmmformer_ckpt": MODEL_ROOT / "GMMFormer_v2/results/clip/activitynet/gmmformer_v2/best.ckpt",
        "hlformer_ckpt": MODEL_ROOT / "ICCV25-HLFormer/results/clip/activitynet/HLFormer/best.ckpt",
        "holmes_ckpt": MODEL_ROOT / "ICML26-Holmes/results/clip/activitynet/Holmes/20260721-142334/best.ckpt",
        "output": REPO_ROOT / "outputs/upstream/pseudo_gt_candidates.activitynet.jsonl",
        "rank_dir": REPO_ROOT / "outputs/activitynet_rankings_from_ckpts",
        "pure_clip_cache": REPO_ROOT / "outputs/activitynet_rankings_from_ckpts/pure_clip_frame_topP1000_top100.jsonl",
        "frame_source": "paths",
        "sampled_frame_root": str(DATA_ROOT / "activitynet/raw_frames"),
        "sampled_frame_layout": "video_time",
        "pseudo_window_mode": "gt_clamped",
        "pseudo_window_sec": 4.0,
        "pseudo_window_min_sec": 4.0,
        "pseudo_window_max_sec": 20.0,
    },
    "charades": {
        "dreamprvr_ckpt": MODEL_ROOT / "CVPR26-DreamPRVR/results/clip/charades/DreamPRVR/best.ckpt",
        "ms_sl_ckpt": MODEL_ROOT / "ms-sl/results/charades/charades-charades_clip_ms_sl-2026_06_19_16_20_37/model.ckpt",
        "gmmformer_ckpt": MODEL_ROOT / "GMMFormer_v2/results/clip/charades/gmmformer_v2/best.ckpt",
        "hlformer_ckpt": MODEL_ROOT / "ICCV25-HLFormer/results/clip/charades/HLFormer/best.ckpt",
        "holmes_ckpt": MODEL_ROOT / "ICML26-Holmes/results/clip/charades/Holmes/20260721-151633/best.ckpt",
        "output": REPO_ROOT / "outputs/upstream/pseudo_gt_candidates.charades.jsonl",
        "rank_dir": REPO_ROOT / "outputs/charades_rankings_from_ckpts",
        "pure_clip_cache": REPO_ROOT / "outputs/charades_rankings_from_ckpts/pure_clip_frame_topP1000_top100.jsonl",
        "frame_source": "paths",
        "sampled_frame_root": str(DATA_ROOT / "charades/raw_frames"),
        "sampled_frame_layout": "video_time",
        "pseudo_window_mode": "gt_clamped",
        "pseudo_window_sec": 4.0,
        "pseudo_window_min_sec": 4.0,
        "pseudo_window_max_sec": 20.0,
    },
    "msrvtt": {
        "dreamprvr_ckpt": MODEL_ROOT / "CVPR26-DreamPRVR/results/clip/msrvtt/DreamPRVR/best.ckpt",
        "ms_sl_ckpt": MODEL_ROOT / "ms-sl/results/msrvtt/msrvtt-msrvtt_clip_ms_sl-2026_06_22_15_44_42/model.ckpt",
        "gmmformer_ckpt": MODEL_ROOT / "GMMFormer_v2/results/clip/msrvtt/gmmformer_v2/best.ckpt",
        "hlformer_ckpt": MODEL_ROOT / "ICCV25-HLFormer/results/clip/msrvtt/HLFormer/best.ckpt",
        "holmes_ckpt": MODEL_ROOT / "ICML26-Holmes/results/clip/msrvtt/Holmes/20260721-160641/best.ckpt",
        "output": REPO_ROOT / "outputs/upstream/pseudo_gt_candidates.msrvtt.jsonl",
        "rank_dir": REPO_ROOT / "outputs/msrvtt_rankings_from_ckpts",
        "pure_clip_cache": REPO_ROOT / "outputs/msrvtt_rankings_from_ckpts/pure_clip_frame_topP1000_top100.jsonl",
        "frame_source": "frames",
        "sampled_frame_root": str(DATA_ROOT / "msrvtt/raw_frames"),
        "sampled_frame_layout": "video_time",
        "pseudo_window_mode": "fixed",
        "pseudo_window_sec": 4.0,
        "pseudo_window_min_sec": 4.0,
        "pseudo_window_max_sec": 20.0,
    },
}

MODEL_CKPT_ARGS = {
    "dreamprvr": "dreamprvr_ckpt",
    "ms-sl": "ms_sl_ckpt",
    "gmmformer": "gmmformer_ckpt",
    "hlformer": "hlformer_ckpt",
    "holmes": "holmes_ckpt",
}

DEFAULT_PRVR_ADAPTERS = ("dreamprvr", "gmmformer", "hlformer", "holmes")
PRVR_FAMILY_ADAPTERS = {
    "dreamprvr": (MODEL_ROOT / "CVPR26-DreamPRVR/src", "DreamPRVR"),
    "gmmformer": (MODEL_ROOT / "GMMFormer_v2/src", "gmmformer_v2"),
    "hlformer": (MODEL_ROOT / "ICCV25-HLFormer/src", "HLFormer"),
    "holmes": (MODEL_ROOT / "ICML26-Holmes/src", None),
}

ADAPTER_DISPLAY_NAMES = {
    "dreamprvr": "DreamPRVR",
    "ms-sl": "ms-sl",
    "gmmformer": "GMMFormerv2",
    "hlformer": "HLFormer",
    "holmes": "Holmes",
}

LEGACY_RANK_FILENAMES = {
    "dreamprvr": "dreamprvr_top100.jsonl",
    "ms-sl": "ms_sl_top100.jsonl",
    "gmmformer": "gmmformer_v2_top100.jsonl",
    "hlformer": "hlformer_top100.jsonl",
    "holmes": "holmes_top100.jsonl",
}

SPLIT_PATH_OVERRIDES = {
    "activitynet": {
        "train": {
            "annotation": str(DATA_ROOT / "activitynet/TextData/activitynet_train.jsonl"),
            "caption": str(DATA_ROOT / "activitynet/TextData/activitynettrain.caption.txt"),
        },
    },
    "tvr": {
        "train": {
            "annotation": str(DATA_ROOT / "tvr/TextData/tvr_train_release.jsonl"),
            "caption": str(DATA_ROOT / "tvr/TextData/tvrtrain.caption.txt"),
        },
    },
    "charades": {
        "train": {
            "annotation": str(DATA_ROOT / "charades/TextData/charades_train.jsonl"),
            "caption": str(DATA_ROOT / "charades/TextData/charadestrain.caption.txt"),
        },
    },
    "msrvtt": {
        "train": {
            "annotation": str(DATA_ROOT / "msrvtt/MSRVTT_data.videos.jsonl"),
            "caption": str(DATA_ROOT / "msrvtt/TextData/msrvtttrain.caption.txt"),
        },
    },
}


def write_topk_jsonl(
    output_path: str,
    query_ids: Sequence[str],
    video_ids: Sequence[str],
    scores: np.ndarray,
    topk: int,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for qi, query_id in enumerate(tqdm(query_ids, desc=f"write {Path(output_path).name}")):
            row = scores[qi]
            idxs = np.argsort(-row)[:topk]
            top = [
                {
                    "video_id": video_ids[int(vi)],
                    "rank": rank + 1,
                    "score": float(row[int(vi)]),
                }
                for rank, vi in enumerate(idxs)
            ]
            f.write(json.dumps({"query_id": query_id, "top100": top}, ensure_ascii=False) + "\n")


def merge_topk_scores(
    top_scores: np.ndarray,
    top_video_idxs: np.ndarray,
    query_indices: np.ndarray,
    score_np: np.ndarray,
    video_indices: np.ndarray,
    topk: int,
) -> None:
    old_scores = top_scores[query_indices]
    old_idxs = top_video_idxs[query_indices]
    new_idxs = np.broadcast_to(video_indices[None, :], score_np.shape)
    combined_scores = np.concatenate([old_scores, score_np], axis=1)
    combined_idxs = np.concatenate([old_idxs, new_idxs], axis=1)
    kth = min(topk - 1, combined_scores.shape[1] - 1)
    selected = np.argpartition(-combined_scores, kth=kth, axis=1)[:, :topk]
    selected_scores = np.take_along_axis(combined_scores, selected, axis=1)
    selected_idxs = np.take_along_axis(combined_idxs, selected, axis=1)
    order = np.argsort(-selected_scores, axis=1)
    top_scores[query_indices] = np.take_along_axis(selected_scores, order, axis=1)
    top_video_idxs[query_indices] = np.take_along_axis(selected_idxs, order, axis=1)


def write_topk_indices_jsonl(
    output_path: str,
    query_ids: Sequence[str],
    video_ids: Sequence[str],
    top_scores: np.ndarray,
    top_video_idxs: np.ndarray,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for qi, query_id in enumerate(tqdm(query_ids, desc=f"write {Path(output_path).name}")):
            top = [
                {
                    "video_id": video_ids[int(video_idx)],
                    "rank": rank + 1,
                    "score": float(top_scores[qi, rank]),
                }
                for rank, video_idx in enumerate(top_video_idxs[qi])
                if video_idx >= 0
            ]
            f.write(json.dumps({"query_id": query_id, "top100": top}, ensure_ascii=False) + "\n")


def clear_conflicting_modules() -> None:
    prefixes = ("Configs", "Models", "Datasets", "Opts", "Losses", "Validations", "Utils", "method", "utils")
    for name in list(sys.modules):
        if name in prefixes or name.startswith(tuple(p + "." for p in prefixes)):
            del sys.modules[name]


def import_config_without_makedirs(module_name: str) -> Any:
    """Import legacy config modules without creating old absolute result dirs."""
    original_makedirs = os.makedirs

    def safe_makedirs(path: str, *args: Any, **kwargs: Any) -> None:
        if str(path).startswith(("/data1/", "/data2/")):
            return
        original_makedirs(path, *args, **kwargs)

    os.makedirs = safe_makedirs
    try:
        return importlib.import_module(module_name)
    finally:
        os.makedirs = original_makedirs


def load_prvr_family_config(repo_src: Path, dataset: str) -> Dict[str, Any]:
    dataset_name = DATASET_INTERNAL_NAMES[dataset]
    base_name = dataset_name.replace("_clip", "")
    if base_name == "tvr":
        module_name = "Configs.tvr"
    elif base_name in ("act", "msrvtt"):
        module_name = "Configs.act"
    elif base_name == "cha":
        module_name = "Configs.cha"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    config_mod = import_config_without_makedirs(module_name)
    configure_cfg = importlib.import_module("prvr_compat").configure_cfg
    return configure_cfg(copy.deepcopy(config_mod.cfg), dataset_name, str(repo_src.parent))


def get_clip_feature_paths(rootpath: str, collection: str, clip_feature: str = "b32") -> tuple[str, str]:
    if clip_feature != "b32":
        raise ValueError(f"Unsupported clip_feature: {clip_feature}")
    visual_feats = os.path.join(rootpath, collection, "FeatureData", f"new_clip_vit_32_{collection}_vid_features.hdf5")
    if not os.path.exists(visual_feats):
        visual_feats = os.path.join(
            rootpath,
            collection,
            "FeatureData",
            "i3d_rgb_lgi",
            f"new_clip_vit_32_{collection}_vid_features.hdf5",
        )
    text_feat_path = os.path.join(rootpath, collection, "TextData", f"clip_ViT_B_32_{collection}_query_feat.hdf5")
    if collection == "charades":
        full_text_feat_path = os.path.join(
            rootpath,
            collection,
            "TextData",
            "other",
            f"clip_ViT_B_32_{collection}_query_feat.hdf5",
        )
        if os.path.exists(full_text_feat_path):
            text_feat_path = full_text_feat_path
    return visual_feats, text_feat_path


def build_prvr_eval_loaders(cfg: Dict[str, Any], data_mod: Any, split: str):
    rootpath = cfg["data_root"]
    collection = cfg["collection"]
    caption_file = os.path.join(rootpath, collection, "TextData", f"{collection}{split}.caption.txt")
    visual_feats, text_feat_path = get_clip_feature_paths(rootpath, collection, cfg.get("clip_feature", "b32"))
    video_ids = data_mod.read_video_ids(caption_file)

    try:
        video_dataset = data_mod.VisDataSet4PRVR(visual_feats, None, cfg, video_ids=video_ids, is_clip=True)
    except TypeError:
        video_dataset = data_mod.VisDataSet4PRVR(visual_feats, None, cfg, video_ids=video_ids)
    text_dataset = data_mod.TxtDataSet4PRVR(caption_file, text_feat_path, cfg)

    context_loader = DataLoader(
        video_dataset,
        collate_fn=data_mod.collate_frame_val,
        batch_size=cfg["eval_context_bsz"],
        num_workers=cfg["num_workers"],
        shuffle=False,
        pin_memory=cfg["pin_memory"],
    )
    query_loader = DataLoader(
        text_dataset,
        collate_fn=data_mod.collate_text_val,
        batch_size=cfg["eval_query_bsz"],
        num_workers=cfg["num_workers"],
        shuffle=False,
        pin_memory=cfg["pin_memory"],
    )
    return context_loader, query_loader


def load_state_dict_flex(model: torch.nn.Module, state_dict: Dict[str, Any]) -> None:
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError:
        pass

    stripped = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            stripped[key[len("module.") :]] = value
        else:
            stripped[key] = value
    model.load_state_dict(stripped)


def dump_prvr_family_rankings(
    repo_src: Path,
    ckpt_path: str,
    output_path: str,
    model_name: str | None,
    dataset: str,
    topk: int,
    gpu: str,
    num_workers: int | None,
    eval_query_bsz: int | None,
    eval_context_bsz: int | None,
    split: str,
    ) -> None:
    """Dump rankings for GMMFormer_v2, HLFormer, and Holmes-style repos."""
    clear_conflicting_modules()
    sys.path.insert(0, str(MODEL_ROOT))
    sys.path.insert(0, str(repo_src))
    old_cwd = os.getcwd()
    os.chdir(str(repo_src))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    try:
        get_datasets = importlib.import_module("Datasets.builder").get_datasets
        data_mod = importlib.import_module("Datasets.data_provider")
        get_models = importlib.import_module("Models.builder").get_models
        get_validations = importlib.import_module("Validations.builder").get_validations
        load_ckpt = importlib.import_module("Utils.utils").load_ckpt

        cfg = load_prvr_family_config(repo_src, dataset)
        if model_name is not None:
            cfg["model_name"] = model_name
        if num_workers is not None:
            cfg["num_workers"] = num_workers
        if eval_query_bsz is not None:
            cfg["eval_query_bsz"] = eval_query_bsz
        if eval_context_bsz is not None:
            cfg["eval_context_bsz"] = eval_context_bsz
        cfg["pin_memory"] = not cfg.get("no_pin_memory", False)

        print(f"[{model_name or cfg['model_name']}] loading {dataset} {split} dataloaders")
        if split == "val":
            cfg, _, context_loader, query_loader, _, _ = get_datasets(cfg)
        else:
            context_loader, query_loader = build_prvr_eval_loaders(cfg, data_mod, split)

        print(f"[{model_name or cfg['model_name']}] loading checkpoint: {ckpt_path}")
        model = get_models(cfg)
        _, state_dict, _, _, _ = load_ckpt(ckpt_path)
        load_state_dict_flex(model, state_dict)
        model = model.cuda()
        model.eval()

        validator = get_validations(cfg)
        with torch.no_grad():
            ctx_info = validator.compute_context_info(model, context_loader)
            video_ids = list(ctx_info["video_metas"])
            query_ids = list(getattr(query_loader.dataset, "cap_ids"))
            top_scores = np.full((len(query_ids), topk), -np.inf, dtype=np.float32)
            top_video_idxs = np.full((len(query_ids), topk), -1, dtype=np.int64)
            video_indices = np.arange(len(video_ids), dtype=np.int64)

            for batch in tqdm(
                query_loader,
                desc=f"[{model_name or cfg['model_name']}] merge query top-{topk}",
                total=len(query_loader),
            ):
                query_feat = batch[0].cuda()
                query_mask = batch[1].cuda()
                query_indices = np.asarray(batch[2], dtype=np.int64)
                clip_scores, frame_scores = model.get_pred_from_raw_query(
                    query_feat,
                    query_mask,
                    None,
                    ctx_info["video_proposal_feat"],
                    ctx_info["video_feat"],
                )
                score_sum = cfg["clip_scale_w"] * clip_scores + cfg["frame_scale_w"] * frame_scores
                score_np = score_sum.detach().cpu().numpy().astype(np.float32, copy=False)
                merge_topk_scores(top_scores, top_video_idxs, query_indices, score_np, video_indices, topk)

        write_topk_indices_jsonl(output_path, query_ids, video_ids, top_scores, top_video_idxs)
    finally:
        os.chdir(old_cwd)
        if str(repo_src) in sys.path:
            sys.path.remove(str(repo_src))
        if str(MODEL_ROOT) in sys.path:
            sys.path.remove(str(MODEL_ROOT))
        clear_conflicting_modules()


def dump_mssl_rankings(
    repo_dir: Path,
    ckpt_path: str,
    output_path: str,
    topk: int,
    gpu: str,
    num_workers: int | None,
    eval_query_bsz: int | None,
    eval_context_bsz: int | None,
    split: str,
    video_chunk_size: int,
) -> None:
    clear_conflicting_modules()
    sys.path.insert(0, str(repo_dir))
    old_cwd = os.getcwd()
    os.chdir(str(repo_dir))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    try:
        eval_mod = importlib.import_module("method.eval")
        data_mod = importlib.import_module("method.data_provider")
        basic_mod = importlib.import_module("utils.basic_utils")

        opt_path = Path(ckpt_path).resolve().parent / "opt.json"
        if not opt_path.exists():
            raise FileNotFoundError(f"ms-sl opt.json not found next to checkpoint: {opt_path}")
        with open(opt_path, "r", encoding="utf-8") as f:
            opt_dict = json.load(f)
        opt_dict["ckpt_filepath"] = ckpt_path
        if num_workers is not None:
            opt_dict["num_workers"] = num_workers
        if eval_query_bsz is not None:
            opt_dict["eval_query_bsz"] = eval_query_bsz
        if eval_context_bsz is not None:
            opt_dict["eval_context_bsz"] = eval_context_bsz
        opt_dict["device_ids"] = [int(gpu.split(",")[0])]
        opt_dict["device"] = torch.device("cuda:0")
        opt_dict["h5driver"] = None if opt_dict.get("no_core_driver") else "core"
        opt_dict["pin_memory"] = not opt_dict.get("no_pin_memory", False)
        opt = SimpleNamespace(**opt_dict)

        rootpath = opt.root_path
        collection = opt.collection
        caption_file = os.path.join(rootpath, collection, "TextData", f"{collection}{split}.caption.txt")
        visual_feats, text_feat_path = eval_mod.get_clip_feature_paths(rootpath, collection)
        opt.q_feat_size = 512
        video_ids = data_mod.read_video_ids(caption_file)
        val_text_dataset = data_mod.TxtDataSet4MS_SL(caption_file, text_feat_path, opt)

        print(f"[ms-sl] loading checkpoint: {ckpt_path}")
        model = eval_mod.setup_model(opt)
        model.eval()
        query_ids = list(val_text_dataset.cap_ids)
        top_scores = np.full((len(query_ids), topk), -np.inf, dtype=np.float32)
        top_video_idxs = np.full((len(query_ids), topk), -1, dtype=np.int64)

        for start in range(0, len(video_ids), video_chunk_size):
            end = min(start + video_chunk_size, len(video_ids))
            chunk_video_ids = video_ids[start:end]
            print(
                f"[ms-sl] scoring video chunk {start}:{end} / {len(video_ids)}",
                flush=True,
            )
            val_video_dataset = data_mod.VisDataSet4MS_SL(
                visual_feats, None, opt, video_ids=chunk_video_ids, is_clip=True
            )
            query_eval_loader = DataLoader(
                val_text_dataset,
                collate_fn=data_mod.collate_text_val,
                batch_size=opt.eval_query_bsz,
                num_workers=opt.num_workers,
                shuffle=False,
                pin_memory=opt.pin_memory,
            )

            with torch.no_grad():
                ctx_info = eval_mod.compute_context_info(model, val_video_dataset, opt)
                chunk_video_idxs = np.arange(start, end, dtype=np.int64)
                for batch in tqdm(
                    query_eval_loader,
                    desc=f"[ms-sl] merge query top-{topk}",
                    total=len(query_eval_loader),
                ):
                    query_feat = batch[0].to(opt.device)
                    query_mask = batch[1].to(opt.device)
                    query_indices = np.asarray(batch[2], dtype=np.int64)
                    clip_scores, frame_scores = model.get_pred_from_raw_query(
                        query_feat,
                        query_mask,
                        None,
                        ctx_info["video_proposal_feat"],
                        ctx_info["video_feat"],
                        ctx_info["video_mask"],
                    )
                    score_sum = opt.clip_scale_w * clip_scores + opt.frame_scale_w * frame_scores
                    score_np = score_sum.detach().cpu().numpy().astype(np.float32, copy=False)
                    merge_topk_scores(
                        top_scores,
                        top_video_idxs,
                        query_indices,
                        score_np,
                        chunk_video_idxs,
                        topk,
                    )

            del ctx_info
            torch.cuda.empty_cache()

        write_topk_indices_jsonl(output_path, query_ids, video_ids, top_scores, top_video_idxs)
    finally:
        os.chdir(old_cwd)
        if str(repo_dir) in sys.path:
            sys.path.remove(str(repo_dir))
        clear_conflicting_modules()


def run_child(args: Sequence[str]) -> None:
    print("$ " + " ".join(args), flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONUTF8"] = "1"
    subprocess.run(args, check=True, cwd=str(REPO_ROOT), env=env)


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.=-]+", "_", value.strip())
    return value.strip("_") or "model"


def parse_prvr_model_spec(value: str) -> Dict[str, str]:
    parts = value.split("=", 2)
    if len(parts) != 3:
        raise ValueError(
            "Expected --prvr-model NAME=ADAPTER=CKPT, "
            f"where ADAPTER is one of {sorted(MODEL_CKPT_ARGS)}; got: {value}"
        )
    name, adapter, ckpt = [part.strip() for part in parts]
    if not name or not adapter or not ckpt:
        raise ValueError(f"Expected non-empty NAME=ADAPTER=CKPT, got: {value}")
    if adapter not in MODEL_CKPT_ARGS:
        raise ValueError(f"Unsupported PRVR adapter {adapter!r}; choose one of {sorted(MODEL_CKPT_ARGS)}")
    return {"name": name, "adapter": adapter, "ckpt": ckpt}


def legacy_prvr_model_specs(args: argparse.Namespace) -> List[Dict[str, str]]:
    specs = []
    for adapter in DEFAULT_PRVR_ADAPTERS:
        attr = MODEL_CKPT_ARGS[adapter]
        ckpt = getattr(args, attr)
        if ckpt:
            specs.append(
                {
                    "name": ADAPTER_DISPLAY_NAMES[adapter],
                    "adapter": adapter,
                    "ckpt": ckpt,
                }
            )
    return specs


def resolve_prvr_model_specs(args: argparse.Namespace) -> List[Dict[str, str]]:
    specs = [parse_prvr_model_spec(item) for item in args.prvr_model] if args.prvr_model else legacy_prvr_model_specs(args)
    if len(specs) != args.expected_prvr_models:
        raise ValueError(
            f"Expected {args.expected_prvr_models} PRVR models, got {len(specs)}. "
            "Pass repeated --prvr-model NAME=ADAPTER=CKPT."
        )
    seen = set()
    for spec in specs:
        if spec["name"] in seen:
            raise ValueError(f"Duplicate PRVR model name: {spec['name']}")
        seen.add(spec["name"])
        if not Path(spec["ckpt"]).exists():
            raise FileNotFoundError(f"PRVR checkpoint does not exist for {spec['name']}: {spec['ckpt']}")
    return specs


def rank_filename_for_spec(spec: Dict[str, str], topk: int, legacy: bool) -> str:
    if legacy:
        filename = LEGACY_RANK_FILENAMES[spec["adapter"]]
        if topk == 100:
            return filename
        return filename.replace("top100", f"top{topk}")
    return f"{slugify(spec['name'])}_top{topk}.jsonl"


def apply_dataset_defaults(args: argparse.Namespace) -> argparse.Namespace:
    defaults = DATASET_DEFAULTS[args.dataset]
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, str(value) if isinstance(value, Path) else value)
    if args.frame_source is None:
        args.frame_source = defaults["frame_source"]
    if args.split != "val":
        output_default = str(defaults["output"])
        rank_dir_default = str(defaults["rank_dir"])
        pure_clip_default = str(defaults["pure_clip_cache"])
        if args.output == output_default:
            args.output = str(REPO_ROOT / f"outputs/upstream/pseudo_gt_candidates.{args.dataset}.{args.split}.jsonl")
        if args.rank_dir == rank_dir_default:
            args.rank_dir = str(REPO_ROOT / f"outputs/{args.dataset}_{args.split}_rankings_from_ckpts")
        if args.pure_clip_cache == pure_clip_default:
            args.pure_clip_cache = str(
                REPO_ROOT / f"outputs/{args.dataset}_{args.split}_rankings_from_ckpts/pure_clip_frame_topP1000_top100.jsonl"
            )
    return args


def get_split_override(dataset: str, split: str, key: str) -> str | None:
    return SPLIT_PATH_OVERRIDES.get(dataset, {}).get(split, {}).get(key)


def require_checkpoint(args: argparse.Namespace, kind: str) -> None:
    attr = MODEL_CKPT_ARGS[kind]
    value = getattr(args, attr)
    option = "--" + attr.replace("_", "-")
    if not value:
        raise FileNotFoundError(
            f"{kind} checkpoint is not configured for dataset={args.dataset}. "
            f"Pass {option} after training or choosing a checkpoint."
        )
    if not Path(value).exists():
        raise FileNotFoundError(f"{kind} checkpoint does not exist: {value}")


def add_optional_checkpoint_args(cmd: List[str], args: argparse.Namespace) -> None:
    for attr in MODEL_CKPT_ARGS.values():
        value = getattr(args, attr)
        if value:
            cmd.extend(["--" + attr.replace("_", "-"), value])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-rank", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model-kind", choices=sorted(MODEL_CKPT_ARGS), help=argparse.SUPPRESS)
    parser.add_argument("--rank-output", help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", help=argparse.SUPPRESS)

    parser.add_argument("--dataset", choices=sorted(DATASET_DEFAULTS), default="tvr")
    parser.add_argument("--split", choices=["val", "train"], default="val")
    parser.add_argument(
        "--prvr-model",
        action="append",
        default=[],
        metavar="NAME=ADAPTER=CKPT",
        help=(
            "PRVR model to include in agreement mining. ADAPTER must be one of "
            "dreamprvr, gmmformer, hlformer, holmes, ms-sl. Pass this option four times."
        ),
    )
    parser.add_argument("--expected-prvr-models", type=int, default=4)
    parser.add_argument("--dreamprvr-ckpt", default=None)
    parser.add_argument("--ms-sl-ckpt", default=None)
    parser.add_argument("--gmmformer-ckpt", default=None)
    parser.add_argument("--hlformer-ckpt", default=None)
    parser.add_argument("--holmes-ckpt", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--rank-dir", default=None)
    parser.add_argument("--pure-clip-cache", default=None)
    parser.add_argument(
        "--frame-source",
        choices=["frames", "videos", "paths"],
        default=None,
        help="How candidate frame paths are materialized. ActivityNet defaults to paths.",
    )
    parser.add_argument("--video-root", default=None, help="Raw video root used with --frame-source videos.")
    parser.add_argument("--sampled-frame-root", default=None)
    parser.add_argument("--sampled-frame-layout", choices=["video_time", "query_role"], default=None)
    parser.add_argument("--overwrite-sampled-frames", action="store_true")
    parser.add_argument("--pseudo-window-sec", type=float, default=None)
    parser.add_argument("--pseudo-window-mode", choices=["fixed", "gt_clamped"], default=None)
    parser.add_argument("--pseudo-window-min-sec", type=float, default=None)
    parser.add_argument("--pseudo-window-max-sec", type=float, default=None)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument(
        "--pure-clip-query-batch-size",
        type=int,
        default=128,
        help="Query batch size used when computing pure CLIP top-k.",
    )
    parser.add_argument(
        "--pure-clip-video-batch-size",
        type=int,
        default=512,
        help="Video batch size used when computing pure CLIP top-k.",
    )
    parser.add_argument(
        "--pure-clip-top-frames",
        type=int,
        default=1000,
        help="For each query, take this many best val-set frame embeddings, then dedupe videos to topk.",
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--eval-query-bsz", type=int, default=None)
    parser.add_argument("--eval-context-bsz", type=int, default=None)
    parser.add_argument(
        "--video-chunk-size",
        type=int,
        default=1024,
        help="Number of videos scored per chunk for memory-safe train split ranking.",
    )
    parser.add_argument("--skip-existing-ranks", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved commands without running inference.")
    return apply_dataset_defaults(parser.parse_args())


def dump_rank_entry(args: argparse.Namespace) -> None:
    ckpt_path = args.checkpoint
    if not ckpt_path:
        require_checkpoint(args, args.model_kind)
        ckpt_path = getattr(args, MODEL_CKPT_ARGS[args.model_kind])
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"{args.model_kind} checkpoint does not exist: {ckpt_path}")

    if args.model_kind == "ms-sl":
        dump_mssl_rankings(
            MODEL_ROOT / "ms-sl",
            ckpt_path,
            args.rank_output,
            args.topk,
            args.gpu,
            args.num_workers,
            args.eval_query_bsz,
            args.eval_context_bsz,
            args.split,
            args.video_chunk_size,
        )
    elif args.model_kind in PRVR_FAMILY_ADAPTERS:
        repo_src, model_name = PRVR_FAMILY_ADAPTERS[args.model_kind]
        dump_prvr_family_rankings(
            repo_src,
            ckpt_path,
            args.rank_output,
            model_name,
            args.dataset,
            args.topk,
            args.gpu,
            args.num_workers,
            args.eval_query_bsz,
            args.eval_context_bsz,
            args.split,
        )
    else:
        raise ValueError(f"Unknown model kind: {args.model_kind}")


def main() -> None:
    args = parse_args()
    if args.dump_rank:
        dump_rank_entry(args)
        return

    prvr_model_specs = resolve_prvr_model_specs(args)
    using_legacy_models = not args.prvr_model

    rank_dir = Path(args.rank_dir)
    rank_dir.mkdir(parents=True, exist_ok=True)
    rank_paths = {
        spec["name"]: rank_dir / rank_filename_for_spec(spec, args.topk, using_legacy_models)
        for spec in prvr_model_specs
    }

    for spec in prvr_model_specs:
        path = rank_paths[spec["name"]]
        if args.skip_existing_ranks and path.exists():
            print(f"[skip] existing rank file: {path}", flush=True)
            continue
        print(
            f"[stage] dumping {spec['name']} ({spec['adapter']}) top-{args.topk} rankings -> {path}",
            flush=True,
        )
        child_cmd = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--dump-rank",
            "--model-kind",
            spec["adapter"],
            "--rank-output",
            str(path),
            "--checkpoint",
            spec["ckpt"],
            "--dataset",
            args.dataset,
            "--split",
            args.split,
            "--topk",
            str(args.topk),
            "--gpu",
            args.gpu,
            "--num-workers",
            str(args.num_workers),
        ]
        if spec["adapter"] == "ms-sl":
            child_cmd += ["--video-chunk-size", str(args.video_chunk_size)]
        if args.eval_query_bsz is not None:
            child_cmd += ["--eval-query-bsz", str(args.eval_query_bsz)]
        if args.eval_context_bsz is not None:
            child_cmd += ["--eval-context-bsz", str(args.eval_context_bsz)]
        if args.dry_run:
            print("$ " + " ".join(child_cmd), flush=True)
        else:
            run_child(child_cmd)

    print(f"[stage] building PRVR+CLIP agreement candidates -> {args.output}", flush=True)
    candidate_cmd = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "scripts/build_pseudo_gt_candidates.py"),
        "--dataset",
        args.dataset,
        "--split",
        args.split,
        "--output",
        args.output,
        "--pure-clip-cache",
        args.pure_clip_cache,
        "--topk",
        str(args.topk),
        "--num-frames",
        str(args.num_frames),
        "--query-batch-size",
        str(args.pure_clip_query_batch_size),
        "--video-batch-size",
        str(args.pure_clip_video_batch_size),
        "--pure-clip-top-frames",
        str(args.pure_clip_top_frames),
        "--frame-source",
        args.frame_source,
        "--sampled-frame-layout",
        args.sampled_frame_layout,
        "--pseudo-window-sec",
        str(args.pseudo_window_sec),
        "--pseudo-window-mode",
        args.pseudo_window_mode,
        "--pseudo-window-min-sec",
        str(args.pseudo_window_min_sec),
        "--pseudo-window-max-sec",
        str(args.pseudo_window_max_sec),
    ]
    for spec in prvr_model_specs:
        candidate_cmd += ["--model-rank", f"{spec['name']}={rank_paths[spec['name']]}"]
    if args.video_root is not None:
        candidate_cmd += ["--video-root", args.video_root]
    if args.sampled_frame_root is not None:
        candidate_cmd += ["--sampled-frame-root", args.sampled_frame_root]
    annotation_override = get_split_override(args.dataset, args.split, "annotation")
    caption_override = get_split_override(args.dataset, args.split, "caption")
    if annotation_override is not None:
        candidate_cmd += ["--annotation", annotation_override]
    if caption_override is not None:
        candidate_cmd += ["--val-caption", caption_override]
    if args.overwrite_sampled_frames:
        candidate_cmd += ["--overwrite-sampled-frames"]
    if args.dry_run:
        print("$ " + " ".join(candidate_cmd), flush=True)
    else:
        run_child(candidate_cmd)


if __name__ == "__main__":
    main()
