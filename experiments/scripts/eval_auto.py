#!/usr/bin/env python3
"""Run the original BGM-Net/MS-SL/DL-DKD inference entrypoint from a checkpoint."""
import argparse
import os
import sys
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--repo", required=True)
parser.add_argument("--model-dir", required=True)
parser.add_argument("--gpu", required=True)
parser.add_argument("--multiGT", action="store_true", help="evaluate dense multi-positive ground truth")
args = parser.parse_args()

PROJECT_ROOT = Path(os.environ.get("PRVR_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
DEFAULT_DATA_ROOT = str(PROJECT_ROOT / "datasets")

# Must precede imports from the model repository, which import PyTorch.
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
if args.multiGT:
    os.environ["PRVR_MULTI_GT"] = "1"
sys.path.insert(0, args.repo)
sys.argv = [sys.argv[0], "--model_dir", args.model_dir, "--eval_split_name", "test"]

import torch  # noqa: E402
from method.config import TestOptions  # noqa: E402
from method.eval import start_inference  # noqa: E402


def fallback_options():
    """Recover options for a checkpoint whose opt.json was removed.

    The original runner configuration is deterministic by repository, feature,
    and dataset.  Architecture is still loaded exclusively from model.ckpt.
    This only restores data paths and the original evaluation weights.
    """
    model_dir = Path(args.model_dir)
    try:
        collection = model_dir.parents[0].name
        feature_mode = model_dir.parents[1].name
    except IndexError as exc:
        raise RuntimeError(f"Cannot infer feature/dataset from {model_dir}") from exc
    visual_features = {
        "activitynet": "i3d", "tvr": "i3d_resnet",
        "charades": "i3d_rgb_lgi", "msrvtt": "resnext101-resnet152",
    }
    if collection not in visual_features:
        raise RuntimeError(f"Unsupported collection for option recovery: {collection}")

    options = TestOptions()
    options.initialize()
    opt = options.parser.parse_args([])
    opt.model_dir = str(model_dir)
    opt.results_dir = str(model_dir)
    opt.ckpt_filepath = str(model_dir / "model.ckpt")
    opt.train_log_filepath = str(model_dir / "train.log.txt")
    opt.eval_log_filepath = str(model_dir / "eval.log.txt")
    opt.tensorboard_log_dir = str(model_dir / "tensorboard_log")
    opt.root_path = os.environ.get("PRVR_DATA_ROOT", DEFAULT_DATA_ROOT)
    opt.collection = collection
    opt.dset_name = collection
    opt.feature_mode = feature_mode
    opt.visual_feature = visual_features[collection]
    opt.eval_split_name = "test"
    opt.eval_id = "test"
    opt.no_core_driver = False
    opt.h5driver = "core"
    opt.no_pin_memory = False
    opt.pin_memory = True
    opt.num_workers = 8

    repo_name = Path(args.repo).name
    if repo_name == "ms-sl":
        opt.model_name = "MS_SL_Net"
        if collection == "charades":
            opt.clip_scale_w, opt.frame_scale_w = 0.5, 0.5
    elif repo_name == "BGM-Net":
        opt.model_name = "BGM_Net"
        if collection == "charades":
            opt.clip_scale_w, opt.frame_scale_w = 0.6, 0.4
    elif repo_name == "DL-DKD":
        opt.model_name = "DLDKD"
        opt.feature_mode = "resnet"
        opt.double_branch = True
        opt.teacher, opt.student = "clip", "i3d"
        opt.distill_loss_decay = "exp"
        opt.label_style = "soft"
        if collection == "tvr":
            opt.q_feat_size, opt.margin, opt.n_heads = 768, 0.1, 4
            opt.drop = opt.input_drop = 0.2
        elif collection == "charades":
            opt.drop = opt.input_drop = 0.15
        else:
            opt.drop = opt.input_drop = 0.25
    else:
        raise RuntimeError(f"No option recovery mapping for {repo_name}")
    print(f"[branch-rank] opt.json missing; recovered original {repo_name} {collection}/{opt.feature_mode} eval options")
    return opt


opt_path = Path(args.model_dir) / "opt.json"
opt = TestOptions().parse() if opt_path.is_file() else fallback_options()
# `opt.json` deliberately captures the original run directory but its data
# root is machine-specific. Always use this benchmark's data root at eval.
opt.root_path = os.environ.get("PRVR_DATA_ROOT", DEFAULT_DATA_ROOT)
if args.multiGT:
    os.environ["PRVR_MULTI_GT_COLLECTION"] = opt.collection
    os.environ["PRVR_MULTI_GT_DATA_ROOT"] = opt.root_path
# Saved training options include the original physical GPU. CUDA visibility
# remaps the requested physical GPU to logical index zero for evaluation.
opt.device_ids = [0]
opt.device = torch.device("cuda:0")

# This is analysis-only batching: it changes neither model weights nor score
# definitions.  BGM-Net's 1176 proposal representations can otherwise make a
# full ActivityNet query batch exceed 24 GiB before max-over-representations.
query_batch_size = os.environ.get("PRVR_RAW_DEDUP_EVAL_QUERY_BSZ")
if query_batch_size:
    opt.eval_query_bsz = int(query_batch_size)
start_inference(opt)
