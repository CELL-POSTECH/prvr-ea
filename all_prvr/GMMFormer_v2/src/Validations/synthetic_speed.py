"""Exact, memory-bounded synthetic speed evaluation for GMMFormer-v2.

The model's context/query encoders and its dual-branch maxsim/fusion rule are
unchanged.  Only gallery storage is sharded so a 100k-video gallery fits on a
24 GiB GPU.  No ground truth, recall matrix, or ranking artifact is created.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from Utils.utils import gpu


def _model(model):
    return model.module if hasattr(model, 'module') else model


def _elapsed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = fn()
    torch.cuda.synchronize()
    return value, (time.perf_counter() - start) * 1000.0


def _as_batch(query):
    return query.unsqueeze(0) if query.ndim == 1 else query


def _write_row(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open('a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _merge_topk(current_scores, current_ids, scores, ids, topk):
    """Merge exact local shard top-k with a running global top-k."""
    values, positions = torch.topk(torch.cat((current_scores, scores), dim=1), topk, dim=1)
    source_ids = torch.cat((current_ids, ids), dim=1)
    return values, source_ids.gather(1, positions)


def _encode_queries(net, query_loader):
    queries, query_ids = [], []
    total_ms = 0.0
    for batch in query_loader:
        def encode():
            moved = gpu(batch)
            encoded = _as_batch(net.encode_query(moved[0], moved[1]))
            return F.normalize(encoded, dim=-1), moved[-1]
        (encoded, ids), elapsed = _elapsed(encode)
        queries.append(encoded)
        query_ids.extend(ids)
        total_ms += elapsed
    return torch.cat(queries, dim=0), query_ids, total_ms


def run_synthetic_speed_eval(model, context_loader, query_loader, cfg, args, logger):
    """Profile original GMMFormer-v2 scoring on a synthetic 100k gallery."""
    net = _model(model)
    net.eval()
    num_videos = len(context_loader.dataset)
    if num_videos < 1:
        raise ValueError('synthetic context dataset is empty')
    if args.synthetic_context_shard_videos < 1:
        raise ValueError('--synthetic_context_shard_videos must be positive')
    topk = min(int(args.synthetic_topk), num_videos)
    shard_size = min(int(args.synthetic_context_shard_videos), num_videos)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    queries, query_ids, text_total_ms = _encode_queries(net, query_loader)
    num_queries = len(query_ids)
    if num_queries < 1:
        raise ValueError('synthetic query dataset is empty')
    query_batch = int(cfg['eval_query_bsz'])
    global_scores = torch.full((num_queries, topk), float('-inf'), device=queries.device)
    global_ids = torch.full((num_queries, topk), -1, dtype=torch.long, device=queries.device)

    # These buffers retain a single gallery shard.  Their dimensions are the
    # post-encoder GMMFormer-v2 representations, not the 512-D HDF5 inputs.
    clip_buffer = torch.empty((shard_size, 32, cfg['hidden_size']), device=queries.device)
    frame_buffer = torch.empty((shard_size, 128, cfg['hidden_size']), device=queries.device)
    write_offset = 0
    video_offset = 0
    video_total_ms = 0.0
    search_total_ms = 0.0

    def score_shard(size, start):
        nonlocal global_scores, global_ids, search_total_ms
        clip_context = clip_buffer[:size]
        frame_context = frame_buffer[:size]
        for q_start in range(0, num_queries, query_batch):
            q_end = min(q_start + query_batch, num_queries)
            q = queries[q_start:q_end]
            def search():
                # Algebraically identical to get_clip_scale_scores for each
                # branch, while operating one gallery shard at a time.
                clip_scores = torch.matmul(clip_context, q.t()).amax(dim=1).t()
                frame_scores = torch.matmul(frame_context, q.t()).amax(dim=1).t()
                fused = cfg['clip_scale_w'] * clip_scores + cfg['frame_scale_w'] * frame_scores
                local_scores, local_pos = torch.topk(fused, topk, dim=1)
                local_ids = local_pos + start
                return _merge_topk(global_scores[q_start:q_end], global_ids[q_start:q_end],
                                   local_scores, local_ids, topk)
            (merged_scores, merged_ids), elapsed = _elapsed(search)
            global_scores[q_start:q_end] = merged_scores
            global_ids[q_start:q_end] = merged_ids
            search_total_ms += elapsed

    for batch in context_loader:
        def encode_context():
            moved = gpu(batch)
            frame, clip = net.encode_context(moved[0], moved[1], moved[2])
            # Normalizing once here is identical to the normal evaluator's
            # F.normalize inside every branch score call, but prevents a second
            # full-shard allocation for each query.
            return F.normalize(clip, dim=-1), F.normalize(frame, dim=-1)
        (clip, frame), elapsed = _elapsed(encode_context)
        video_total_ms += elapsed
        batch_size = clip.shape[0]
        source = 0
        while source < batch_size:
            room = shard_size - write_offset
            take = min(room, batch_size - source)
            clip_buffer[write_offset:write_offset + take].copy_(clip[source:source + take])
            frame_buffer[write_offset:write_offset + take].copy_(frame[source:source + take])
            write_offset += take
            source += take
            if write_offset == shard_size:
                score_shard(write_offset, video_offset)
                video_offset += write_offset
                write_offset = 0
    if write_offset:
        score_shard(write_offset, video_offset)

    # Ensure the final top-k is materialized before recording peak memory.
    torch.cuda.synchronize()
    vectors_per_video = 32 + 128
    vectors_per_query = num_videos * vectors_per_video
    bytes_per_vector = 512 * 4
    video_ms = video_total_ms
    text_ms = text_total_ms / num_queries
    search_ms = search_total_ms / num_queries
    output = Path(args.synthetic_output) if args.synthetic_output else \
        Path(cfg['root']).parents[1] / 'experiments' / 'processing_time_results_synthetic_100k_top10' / 'gmmformer_v2.csv'
    row = {
        'method': 'GMMFormerV2',
        'video_emb_time_ms': video_ms,
        # Query encoding and gallery encoding are reported separately.  The
        # end-to-end retrieval latency below intentionally excludes the
        # one-time gallery encoding cost, assuming an encoded gallery is
        # reused for online queries.
        'query_emb_time_ms': text_ms,
        'search_time_ms': search_ms,
        'e2e_latency_ms': text_ms + search_ms,
        'search_only_qps': 1000.0 / search_ms if search_ms else 0.0,
        'e2e_qps': 1000.0 / (text_ms + search_ms) if (text_ms + search_ms) else 0.0,
        'clip_dim': 512,
        'frame_dim': 512,
        'vectors_per_query': vectors_per_query,
        'equivalent_512_vectors_per_query': vectors_per_query,
        'vectors_per_video': vectors_per_video,
        'clip_vectors_per_video': 32,
        'frame_vectors_per_video': 128,
        'vector_memory_mib_per_query': vectors_per_query * bytes_per_vector / 2**20,
        'clip_vector_memory_mib_per_query': num_videos * 32 * bytes_per_vector / 2**20,
        'frame_vector_memory_mib_per_query': num_videos * 128 * bytes_per_vector / 2**20,
        'peak_gpu_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
        'peak_gpu_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
        'chunk_vector_budget': int(args.synthetic_chunk_vector_budget),
        'chunk_videos': shard_size,
        'resident_gallery': str(shard_size >= num_videos).upper(),
        'profile': f'gmmformer_v2_synthetic_full_pipeline_chunked_qbsz{query_batch}',
        'device': str(queries.device),
        'log_path': str(Path(cfg['model_root']) / 'log.txt'),
        'error': '',
    }
    _write_row(output, row)
    logger.info('Synthetic speed row written: %s', output)
    logger.info('Synthetic speed: videos=%d queries=%d shard=%d video=%.3fms text=%.3fms search=%.3fms',
                num_videos, num_queries, shard_size, video_ms, text_ms, search_ms)
    return row
