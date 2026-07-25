#!/usr/bin/env python3
"""Exact dual-branch synthetic latency worker for the original DL-DKD model."""
from __future__ import annotations

import argparse
import json
import os
import sys
from argparse import Namespace
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--repo", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--corpus-dir", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--gpu", default="0")
parser.add_argument("--search-vector-budget", type=int, default=7_237_555)
parser.add_argument("--search-chunk-videos", type=int, default=0)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(args.repo)); sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from latency.synthetic_corpus import open_video_features  # noqa: E402
from latency.results import append_detail, export_wide_qps  # noqa: E402
from method.model import DLDKD  # noqa: E402


def cuda_ms(fn):
    torch.cuda.synchronize(); start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record(); value = fn(); end.record(); torch.cuda.synchronize()
    return value, float(start.elapsed_time(end))


def score(query, context, mask):
    raw = torch.einsum("md,nld->mln", F.normalize(query, dim=-1), F.normalize(context, dim=-1))
    raw = raw.masked_fill(mask.t().unsqueeze(0) == 0, -1e10)
    return raw.amax(dim=1)


def normalized_context(context):
    return F.normalize(context, dim=-1).to(dtype=torch.float16).contiguous()


def prepared_score(query, context):
    return torch.einsum("md,nld->mln", query, context).amax(dim=1)


def options(path: Path):
    data = json.loads((path.parent / "opt.json").read_text()) if (path.parent / "opt.json").exists() else {}
    return int(data.get("eval_query_bsz", 50)), int(data.get("eval_context_bsz", 100))


def pad_input(source, dim):
    if source.shape[-1] == dim:
        return source
    output = source.new_zeros((*source.shape[:-1], dim))
    output[..., :source.shape[-1]] = source
    return output


def main():
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    cfg = checkpoint["model_cfg"]
    # Constructor-only loss options do not affect inference representations.
    opt = Namespace(double_branch=True, kl_intra_weight=0.0, inher_nce_weight=0.0,
                    explore_nce_weight=0.0, collection="activitynet", alpha=0.0, belta=0.0)
    model = DLDKD(cfg, opt).cuda().eval()
    model.load_state_dict(checkpoint["model"])
    device = next(model.parameters()).device
    qbsz, cbsz = options(args.checkpoint)
    bank = open_video_features(args.corpus_dir); videos = int(bank.shape[0])
    generator = torch.Generator(device="cpu").manual_seed(9561)
    q_in = torch.randn((qbsz, cfg.max_desc_l, cfg.query_input_size), generator=generator).to(device)
    q_mask = torch.ones((qbsz, cfg.max_desc_l), device=device)
    def encode_source_batch(start, stop):
        source = torch.from_numpy(np.array(bank[start:stop], copy=True)).to(device).unsqueeze(1)
        source = pad_input(source, cfg.visual_input_size).repeat(1, cfg.max_ctx_l, 1)
        mask = torch.ones((stop - start, cfg.max_ctx_l), device=device)
        def encode_and_normalize():
            inheritance_c, exploration_c = model.encode_context(source, mask)
            return normalized_context(inheritance_c), normalized_context(exploration_c)
        return cuda_ms(encode_and_normalize)

    with torch.no_grad():
        model.encode_query(q_in, q_mask)
        (inheritance_q, exploration_q), q_ms = cuda_ms(
            lambda: tuple(F.normalize(item, dim=-1).to(dtype=torch.float16).contiguous() for item in model.encode_query(q_in, q_mask)))
        top = torch.full((qbsz, 10), float("-inf"), dtype=torch.float16, device=device)
        prep = search_ms = 0.0
        encoded_videos = retrieved_videos = 0
        progress_step = max(1, videos // 20); next_progress = progress_step
        print(f"gallery preparation: 0/{videos} (0%)", flush=True)
        search_start = 0
        while search_start < videos:
            first_stop = min(search_start + cbsz, videos)
            (first_inheritance, first_exploration), elapsed = encode_source_batch(search_start, first_stop)
            prep += elapsed
            reps_per_video = first_inheritance.shape[1] + first_exploration.shape[1]
            chunk_videos = int(args.search_chunk_videos or max(1, args.search_vector_budget // reps_per_video))
            chunk_stop = min(search_start + chunk_videos, videos); count = chunk_stop - search_start
            inheritance_bank = torch.empty((count, *first_inheritance.shape[1:]), dtype=torch.float16, device=device)
            exploration_bank = torch.empty((count, *first_exploration.shape[1:]), dtype=torch.float16, device=device)
            first_count = first_stop - search_start
            inheritance_bank[:first_count].copy_(first_inheritance); exploration_bank[:first_count].copy_(first_exploration)
            encoded_videos += first_count; offset = first_count
            if encoded_videos >= next_progress or encoded_videos == videos:
                print(f"gallery preparation: {encoded_videos}/{videos} ({100.0 * encoded_videos / videos:.0f}%)", flush=True)
                while next_progress <= encoded_videos: next_progress += progress_step
            for encode_start in range(first_stop, chunk_stop, cbsz):
                encode_stop = min(encode_start + cbsz, chunk_stop)
                (inheritance_c, exploration_c), elapsed = encode_source_batch(encode_start, encode_stop)
                prep += elapsed; size = encode_stop - encode_start
                inheritance_bank[offset:offset + size].copy_(inheritance_c); exploration_bank[offset:offset + size].copy_(exploration_c)
                offset += size; encoded_videos += size
                if encoded_videos >= next_progress or encoded_videos == videos:
                    print(f"gallery preparation: {encoded_videos}/{videos} ({100.0 * encoded_videos / videos:.0f}%)", flush=True)
                    while next_progress <= encoded_videos: next_progress += progress_step
            def retrieve():
                nonlocal top
                values = 0.7 * prepared_score(inheritance_q, inheritance_bank) + 0.3 * prepared_score(exploration_q, exploration_bank)
                local = torch.topk(values, min(10, values.shape[1]), dim=1).values
                top = torch.topk(torch.cat((top, local), dim=1), 10, dim=1).values
            _, elapsed = cuda_ms(retrieve); search_ms += elapsed
            retrieved_videos += count
            print(f"retrieval: {retrieved_videos}/{videos} ({100.0 * retrieved_videos / videos:.0f}%), search chunk={count} videos / {reps_per_video} repr-video", flush=True)
            del inheritance_bank, exploration_bank
            search_start = chunk_stop
    e2e = q_ms + search_ms
    append_detail(args.output, {"corpus": videos, "method": "DL-DKD", "backend": "torch_gpu_exact",
        "device": str(device), "query_bsz": qbsz, "gallery_encoder_bsz": cbsz,
        "search_chunk_videos": chunk_videos, "reprs_per_video": reps_per_video,
        "query_emb_time_ms": q_ms, "search_time_ms": search_ms,
        "e2e_latency_ms": e2e, "e2e_qps": 1000.0 * qbsz / e2e if e2e else 0.0,
        "gallery_prepare_sec": prep / 1000.0, "checkpoint": str(args.checkpoint), "status": "ok"})
    export_wide_qps(args.output, args.output.parent / "qps.csv")
    print(f"latency complete: DL-DKD V={videos} qbsz={qbsz}")


if __name__ == "__main__":
    main()
