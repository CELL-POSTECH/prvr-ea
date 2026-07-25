#!/usr/bin/env python3
"""Exact chunked latency worker for MS-SL and BGM-Net checkpoints."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--repo", type=Path, required=True)
parser.add_argument("--method", choices=("MS-SL", "BGM-Net"), required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--corpus-dir", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--gpu", default="0")
parser.add_argument("--search-vector-budget", type=int, default=7_237_555)
parser.add_argument("--search-chunk-videos", type=int, default=0)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(args.repo))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from latency.synthetic_corpus import open_video_features  # noqa: E402
from latency.results import append_detail, export_wide_qps  # noqa: E402

if args.method == "MS-SL":
    from method.model import MS_SL_Net as Model  # noqa: E402
    label = "MS-SL"
else:
    from method.model import BGM_Net as Model  # noqa: E402
    label = "BGMNet"


def cuda_ms(fn):
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record(); value = fn(); end.record(); torch.cuda.synchronize()
    return value, float(start.elapsed_time(end))


def branch_scores(query, context):
    if context.ndim == 2:
        context = context.unsqueeze(1)
    return torch.matmul(F.normalize(context, dim=-1), F.normalize(query, dim=-1).t()).amax(dim=1).t()


def prepare_context(frame_context, clip_context):
    """Keep the exact MS-SL/BGM inference inputs as compact gallery banks.

    Their frame score is query-guided attention, not maxsim over all raw
    frames.  Frame features must remain FP32 for the original linear layers;
    proposal features can safely be stored FP16 and are cast only after the
    query-specific proposal has been selected.
    """
    return frame_context.contiguous(), clip_context.to(dtype=torch.float16).contiguous()


def inference_scores(model, query, frame_bank, proposal_bank, clip_w, frame_w):
    """Original MS-SL/BGM inference: proposal maxsim + guided frame attention."""
    # Equivalent to get_clip_scale_scores without materialising a normalized
    # proposal copy.  [V,P,Q] -> [Q,P,V] -> max over P.
    query_half = query.to(dtype=torch.float16)
    proposal_norm = torch.linalg.vector_norm(proposal_bank, dim=-1).clamp_min(1e-6)
    proposal_sim = torch.matmul(proposal_bank, query_half.t()) / proposal_norm.unsqueeze(-1)
    clip_scores, key_indices = proposal_sim.permute(2, 1, 0).max(dim=1)

    # This is key_clip_guided_attention_in_inference written inline so the
    # already encoded query is not encoded again for every gallery chunk.
    key = model.mapping_linear[0](frame_bank)
    value = model.mapping_linear[1](frame_bank)
    videos = frame_bank.shape[0]
    video_index = torch.arange(videos, device=frame_bank.device).unsqueeze(1)
    selected_proposals = proposal_bank[video_index, key_indices.t()].float()
    attention = torch.bmm(key, selected_proposals.transpose(2, 1))
    attention = torch.softmax(attention, dim=1)
    frame_features = torch.bmm(attention.transpose(1, 2), value)
    frame_scores = (F.normalize(frame_features, dim=-1) * query.unsqueeze(0)).sum(dim=-1).transpose(1, 0)
    return float(clip_w) * clip_scores.float() + float(frame_w) * frame_scores


def option_values(checkpoint: Path):
    path = checkpoint.parent / "opt.json"
    if not path.exists():
        return 50, 100, 0.7, 0.3
    data = json.loads(path.read_text(encoding="utf-8"))
    return (int(data.get("eval_query_bsz", 50)), int(data.get("eval_context_bsz", 100)),
            float(data.get("clip_scale_w", 0.7)), float(data.get("frame_scale_w", 0.3)))


def main():
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["model_cfg"]
    model = Model(config).cuda().eval()
    device = next(model.parameters()).device
    query_bsz, context_bsz, clip_w, frame_w = option_values(args.checkpoint)
    features = open_video_features(args.corpus_dir)
    num_videos = int(features.shape[0])
    generator = torch.Generator(device="cpu").manual_seed(9544)
    query_input = torch.randn((query_bsz, config.max_desc_l, config.query_input_size), generator=generator).to(device)
    query_mask = torch.ones((query_bsz, config.max_desc_l), device=device)
    def encode_source_batch(start, stop):
        src = torch.from_numpy(np.array(features[start:stop], copy=True)).to(device).unsqueeze(1)
        clip = src.repeat(1, config.map_size, 1)
        frame = src.repeat(1, config.max_ctx_l, 1)
        mask = torch.ones((stop - start, config.max_ctx_l), device=device)
        def encode_and_normalize():
            frame_ctx, clip_ctx = model.encode_context(clip, frame, mask)
            return prepare_context(frame_ctx, clip_ctx)
        return cuda_ms(encode_and_normalize)

    with torch.no_grad():
        model.encode_query(query_input, query_mask)
        query, query_ms = cuda_ms(lambda: F.normalize(model.encode_query(query_input, query_mask), dim=-1).contiguous())
        global_top = torch.full((query_bsz, 10), float("-inf"), device=device)
        prepare_ms = search_ms = 0.0
        encoded_videos = retrieved_videos = 0
        progress_step = max(1, num_videos // 20); next_progress = progress_step
        print(f"gallery preparation: 0/{num_videos} (0%)", flush=True)
        search_start = 0
        while search_start < num_videos:
            first_stop = min(search_start + context_bsz, num_videos)
            (first_frame, first_clip), elapsed = encode_source_batch(search_start, first_stop)
            prepare_ms += elapsed
            raw_reprs_per_video = first_frame.shape[1] + first_clip.shape[1]
            # The published representation count is proposal features plus
            # the one query-guided frame representation.  The raw frames are
            # retained internally only to form that final representation.
            reps_per_video = first_clip.shape[1] + 1
            chunk_videos = int(args.search_chunk_videos or max(1, args.search_vector_budget // raw_reprs_per_video))
            chunk_stop = min(search_start + chunk_videos, num_videos); count = chunk_stop - search_start
            frame_bank = torch.empty((count, *first_frame.shape[1:]), dtype=torch.float32, device=device)
            clip_bank = torch.empty((count, *first_clip.shape[1:]), dtype=torch.float16, device=device)
            first_count = first_stop - search_start
            frame_bank[:first_count].copy_(first_frame); clip_bank[:first_count].copy_(first_clip)
            encoded_videos += first_count; offset = first_count
            if encoded_videos >= next_progress or encoded_videos == num_videos:
                print(f"gallery preparation: {encoded_videos}/{num_videos} ({100.0 * encoded_videos / num_videos:.0f}%)", flush=True)
                while next_progress <= encoded_videos: next_progress += progress_step
            for encode_start in range(first_stop, chunk_stop, context_bsz):
                encode_stop = min(encode_start + context_bsz, chunk_stop)
                (frame_ctx, clip_ctx), elapsed = encode_source_batch(encode_start, encode_stop)
                prepare_ms += elapsed; size = encode_stop - encode_start
                frame_bank[offset:offset + size].copy_(frame_ctx); clip_bank[offset:offset + size].copy_(clip_ctx)
                offset += size; encoded_videos += size
                if encoded_videos >= next_progress or encoded_videos == num_videos:
                    print(f"gallery preparation: {encoded_videos}/{num_videos} ({100.0 * encoded_videos / num_videos:.0f}%)", flush=True)
                    while next_progress <= encoded_videos: next_progress += progress_step
            def score():
                nonlocal global_top
                values = inference_scores(model, query, frame_bank, clip_bank, clip_w, frame_w)
                local = torch.topk(values, min(10, values.shape[1]), dim=1).values
                global_top = torch.topk(torch.cat((global_top, local), dim=1), 10, dim=1).values
            _, elapsed = cuda_ms(score); search_ms += elapsed
            retrieved_videos += count
            print(f"retrieval: {retrieved_videos}/{num_videos} ({100.0 * retrieved_videos / num_videos:.0f}%), search chunk={count} videos / {reps_per_video} repr-video", flush=True)
            del frame_bank, clip_bank
            search_start = chunk_stop
    e2e = query_ms + search_ms
    append_detail(args.output, {
        "corpus": num_videos, "method": label, "backend": "torch_gpu_exact", "device": str(device),
        "query_bsz": query_bsz, "gallery_encoder_bsz": context_bsz,
        "search_chunk_videos": chunk_videos, "reprs_per_video": reps_per_video,
        "query_emb_time_ms": query_ms, "search_time_ms": search_ms,
        "e2e_latency_ms": e2e, "e2e_qps": 1000.0 * query_bsz / e2e if e2e else 0.0,
        "gallery_prepare_sec": prepare_ms / 1000.0, "checkpoint": str(args.checkpoint), "status": "ok",
    })
    export_wide_qps(args.output, args.output.parent / "qps.csv")
    print(f"latency complete: {label} V={num_videos} qbsz={query_bsz}")


if __name__ == "__main__":
    main()
