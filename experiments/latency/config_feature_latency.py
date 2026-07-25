#!/usr/bin/env python3
"""Latency worker for the config-based CLIP-feature PRVR models.

It loads an existing checkpoint and calls each model's own query/context
encoders.  Contexts are streamed in the model's configured evaluation-context
batch size, while only score computation and global top-k are timed.  This is
therefore exact with respect to the model's original maxsim/fusion rule yet
does not require materialising a multi-million-video representation bank.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--repo", type=Path, required=True)
parser.add_argument("--method", required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--corpus-dir", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--gpu", default="0")
parser.add_argument("--context-bsz", type=int, default=0,
                    help="0 keeps the method's configured context-encoding batch size")
parser.add_argument("--search-vector-budget", type=int, default=7_237_555,
                    help="raw video-representation budget per GPU search chunk")
parser.add_argument("--search-chunk-videos", type=int, default=0,
                    help="explicit search chunk size; 0 derives it from the representation budget")
parser.add_argument("--seed", type=int, default=9527)
args = parser.parse_args()

# Must happen before importing torch.
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(args.repo / "src" if (args.repo / "src").is_dir() else args.repo))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from latency.synthetic_corpus import open_video_features  # noqa: E402
from latency.results import append_detail, export_wide_qps  # noqa: E402
from Configs.builder import get_configs  # noqa: E402
from Models.builder import get_models  # noqa: E402
from Utils.utils import load_ckpt  # noqa: E402


METHOD_LABELS = {
    "AMDNet": "AMDNet",
    "GMMFormer": "GMMFormer",
    "GMMFormer-v2": "GMMFormerV2",
    "HLFormer": "HLFormer",
    "DreamPRVR": "DreamPRVR",
    "Holmes": "Holmes",
    "BOA": "BOA",
    "MSC-PRVR": "MSC-PRVR",
}


def cuda_ms(fn):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    value = fn()
    end.record()
    torch.cuda.synchronize()
    return value, float(start.elapsed_time(end))


def unwrap(model):
    return model.module if hasattr(model, "module") else model


def branch_scores(query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
    """Exact original normalize → maxsim score for [V,R,D] context tensors."""
    if context.ndim == 2:
        # GMMFormer/MS-SL-style frame aggregation produces one representation
        # per video.  It is still maxsim over R=1.
        context = context.unsqueeze(1)
    query = F.normalize(query, dim=-1)
    context = F.normalize(context, dim=-1)
    return torch.matmul(context, query.t()).amax(dim=1).t()


def normalized_context(context):
    """Prepare a compact, normalized gallery bank outside the timed search."""
    if context is None:
        return None
    if context.ndim == 2:
        context = context.unsqueeze(1)
    return F.normalize(context, dim=-1).to(dtype=torch.float16).contiguous()


def normalized_query(query):
    if isinstance(query, tuple):
        return tuple(F.normalize(item, dim=-1).to(dtype=torch.float16).contiguous() for item in query)
    return F.normalize(query, dim=-1).to(dtype=torch.float16).contiguous()


def prepared_branch_scores(query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
    """MaxSim against an already normalized gallery bank."""
    return torch.matmul(context, query.t()).amax(dim=1).t()


def make_query_inputs(cfg, batch_size: int, method: str, device: torch.device):
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 17)
    q_dim = int(cfg.get("text_feat_dim", cfg["q_feat_size"])) if method == "BOA" else int(cfg["q_feat_size"])
    if method == "AMDNet":
        return torch.randn((batch_size, q_dim), generator=generator).to(device), None
    query = torch.randn((batch_size, int(cfg["max_desc_l"]), q_dim), generator=generator)
    mask = torch.ones((batch_size, int(cfg["max_desc_l"])))
    query, mask = query.to(device), mask.to(device)
    if method != "BOA":
        return (query, mask)

    # BOA's semantic-concept augmentation is part of its encoder interface.
    # Empty concept masks preserve the raw-query path while fixed concept banks
    # retain its unchanged context-side retrieval computation.
    hidden = int(cfg["hidden_size"])
    concepts = max(int(getattr(cfg, "top_k", 0) or cfg.get("top_k", 0) or 30), 30)
    bank_generator = torch.Generator(device="cpu").manual_seed(args.seed + 19)
    semantic = (
        torch.randn((concepts, hidden), generator=bank_generator).to(device),
        torch.randn((concepts, hidden), generator=bank_generator).to(device),
    )
    query_semantic = (
        torch.zeros((int(cfg["max_desc_l"]), hidden), device=device),
        torch.zeros((int(cfg["max_desc_l"]), hidden), device=device),
    )
    query_sc_mask = torch.zeros((batch_size, int(cfg["max_desc_l"])), dtype=torch.bool, device=device)
    return query, mask, query_semantic, query_sc_mask, semantic


def encode_query(net, method: str, inputs):
    if method == "AMDNet":
        return net.encode_query(inputs[0], None)
    if method == "BOA":
        query, mask, semantic, query_sc_mask, _ = inputs
        encoded, _ = net.encode_query(query, mask.clone(), semantic, query_sc_mask)
        return encoded
    encoded = net.encode_query(inputs[0], inputs[1])
    if method == "MSC-PRVR":
        encoded = encoded[0]
    return encoded


def encode_context(net, method: str, clip: torch.Tensor, frame: torch.Tensor,
                   mask: torch.Tensor, inputs):
    if method == "AMDNet":
        return None, net.encode_context(clip)[0]
    if method == "BOA":
        return net.encode_context(clip, frame, mask, inputs[-1])
    if method == "MSC-PRVR":
        return net.encode_context(clip, frame, mask, eval=True)
    return net.encode_context(clip, frame, mask)


def score_context(method: str, cfg, query, frame_context, clip_context):
    if method == "BOA":
        clip_query, frame_query = query
        clip_score = branch_scores(clip_query, clip_context)
        frame_score = branch_scores(frame_query, frame_context)
    else:
        clip_score = branch_scores(query, clip_context)
        if method == "AMDNet":
            return clip_score
        frame_score = branch_scores(query, frame_context)
    return float(cfg["clip_scale_w"]) * clip_score + float(cfg["frame_scale_w"]) * frame_score


def score_prepared_context(method: str, cfg, query, frame_context, clip_context):
    if method == "BOA":
        clip_query, frame_query = query
        clip_score = prepared_branch_scores(clip_query, clip_context)
        frame_score = prepared_branch_scores(frame_query, frame_context)
    else:
        clip_score = prepared_branch_scores(query, clip_context)
        if method == "AMDNet":
            return clip_score
        frame_score = prepared_branch_scores(query, frame_context)
    return float(cfg["clip_scale_w"]) * clip_score + float(cfg["frame_scale_w"]) * frame_score


def main():
    if args.method not in METHOD_LABELS:
        raise ValueError(f"unsupported config model: {args.method}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    cfg = get_configs("act_clip")
    if args.method == "MSC-PRVR":
        # These are CLI-injected by MSC_PRVR/src/main.py in ordinary eval.
        # They configure training losses/model construction, not the scoring
        # equation, but must be present for the original builder.
        cfg.update({"model_name": "N_np", "map_size": 32, "vl_coef": 0.1,
                    "sim_thr": 0.5, "rkd_d_coef": 10.0, "rkd_a_coef": 20.0,
                    "rkd_angle_chunk_size": 0})
    model = get_models(cfg)
    _, state, _, _, _ = load_ckpt(str(args.checkpoint))
    model.load_state_dict(state)
    model = model.cuda().eval()
    net = unwrap(model)
    device = next(net.parameters()).device
    query_bsz = int(cfg["eval_query_bsz"])
    context_bsz = int(args.context_bsz or cfg["eval_context_bsz"])
    video_features = open_video_features(args.corpus_dir)
    num_videos = int(video_features.shape[0])

    inputs = make_query_inputs(cfg, query_bsz, args.method, device)
    # One uncaptured warm-up removes CUDA lazy initialization from batch time.
    with torch.no_grad():
        encode_query(net, args.method, inputs)
    torch.cuda.synchronize()

    def encode_source_batch(start: int, stop: int):
        source = np.array(video_features[start:stop], copy=True)
        source = torch.from_numpy(source).to(device, non_blocking=True).unsqueeze(1)
        frame = source.repeat(1, 128, 1)
        clip = source.repeat(1, int(cfg["map_size"]), 1)
        mask = torch.ones((stop - start, 128), device=device)

        def encode_and_normalize():
            frame_context, clip_context = encode_context(net, args.method, clip, frame, mask, inputs)
            return normalized_context(frame_context), normalized_context(clip_context)

        return cuda_ms(encode_and_normalize)

    with torch.no_grad():
        query, query_ms = cuda_ms(lambda: normalized_query(encode_query(net, args.method, inputs)))
        global_scores = torch.full((query_bsz, 10), float("-inf"), dtype=torch.float16, device=device)
        gallery_prepare_ms = search_ms = 0.0
        encoded_videos = retrieved_videos = 0
        progress_step = max(1, num_videos // 20)
        next_progress = progress_step
        print(f"gallery preparation: 0/{num_videos} (0%)", flush=True)

        search_start = 0
        while search_start < num_videos:
            # The loop advances manually after the first encoded mini-batch
            # reveals the model's true branch representation counts.
            first_stop = min(search_start + context_bsz, num_videos)
            (first_frame, first_clip), elapsed = encode_source_batch(search_start, first_stop)
            gallery_prepare_ms += elapsed
            reps_per_video = (0 if first_frame is None else first_frame.shape[1]) + first_clip.shape[1]
            chunk_videos = int(args.search_chunk_videos or max(1, args.search_vector_budget // reps_per_video))
            chunk_stop = min(search_start + chunk_videos, num_videos)
            count = chunk_stop - search_start
            frame_bank = None if first_frame is None else torch.empty(
                (count, *first_frame.shape[1:]), dtype=torch.float16, device=device)
            clip_bank = torch.empty((count, *first_clip.shape[1:]), dtype=torch.float16, device=device)
            first_count = first_stop - search_start
            if frame_bank is not None:
                frame_bank[:first_count].copy_(first_frame)
            clip_bank[:first_count].copy_(first_clip)
            encoded_videos += first_count
            if encoded_videos >= next_progress or encoded_videos == num_videos:
                print(f"gallery preparation: {encoded_videos}/{num_videos} ({100.0 * encoded_videos / num_videos:.0f}%)", flush=True)
                while next_progress <= encoded_videos:
                    next_progress += progress_step
            offset = first_count

            for encode_start in range(first_stop, chunk_stop, context_bsz):
                encode_stop = min(encode_start + context_bsz, chunk_stop)
                (frame_context, clip_context), elapsed = encode_source_batch(encode_start, encode_stop)
                gallery_prepare_ms += elapsed
                size = encode_stop - encode_start
                if frame_bank is not None:
                    frame_bank[offset:offset + size].copy_(frame_context)
                clip_bank[offset:offset + size].copy_(clip_context)
                offset += size
                encoded_videos += size
                if encoded_videos >= next_progress or encoded_videos == num_videos:
                    print(f"gallery preparation: {encoded_videos}/{num_videos} ({100.0 * encoded_videos / num_videos:.0f}%)", flush=True)
                    while next_progress <= encoded_videos:
                        next_progress += progress_step

            def score_and_merge():
                nonlocal global_scores
                scores = score_prepared_context(args.method, cfg, query, frame_bank, clip_bank)
                local = torch.topk(scores, k=min(10, scores.shape[1]), dim=1).values
                global_scores = torch.topk(torch.cat((global_scores, local), dim=1), k=10, dim=1).values
                return global_scores

            _, elapsed = cuda_ms(score_and_merge)
            search_ms += elapsed
            retrieved_videos += count
            print(f"retrieval: {retrieved_videos}/{num_videos} ({100.0 * retrieved_videos / num_videos:.0f}%), search chunk={count} videos / {reps_per_video} repr-video", flush=True)
            del frame_bank, clip_bank
            search_start = chunk_stop

    e2e_ms = query_ms + search_ms
    detail = args.output
    append_detail(detail, {
        "corpus": num_videos,
        "method": METHOD_LABELS[args.method],
        "backend": "torch_gpu_exact",
        "device": str(device),
        "query_bsz": query_bsz,
        "gallery_encoder_bsz": context_bsz,
        "search_chunk_videos": chunk_videos,
        "reprs_per_video": reps_per_video,
        "query_emb_time_ms": query_ms,
        "search_time_ms": search_ms,
        "e2e_latency_ms": e2e_ms,
        "e2e_qps": 1000.0 * query_bsz / e2e_ms if e2e_ms else 0.0,
        "gallery_prepare_sec": gallery_prepare_ms / 1000.0,
        "checkpoint": str(args.checkpoint),
        "status": "ok",
    })
    export_wide_qps(detail, detail.parent / "qps.csv")
    print(f"latency complete: {METHOD_LABELS[args.method]} V={num_videos} qbsz={query_bsz}")


if __name__ == "__main__":
    main()
