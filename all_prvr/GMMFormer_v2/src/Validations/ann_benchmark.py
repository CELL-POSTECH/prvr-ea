"""Opt-in ANN benchmark for GMMFormer-v2 evaluation.

This module never changes training or the normal evaluator.  It stores the
post-checkpoint context representations once, then benchmarks either exhaustive
FlatIP or IVF/HNSW candidate retrieval for the same encoded query vectors.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from Utils.utils import gpu


def _model(model):
    return model.module if hasattr(model, "module") else model


def _default_bank_path(cfg, checkpoint):
    token = _checkpoint_token(checkpoint)
    return Path(cfg['model_root']) / 'ann_context_bank' / f'{cfg["dataset_name"]}_{token}.pt'


def _checkpoint_token(checkpoint):
    return hashlib.sha1(str(Path(checkpoint).resolve()).encode()).hexdigest()[:12]


def build_context_bank(model, validator, context_loader, cfg, checkpoint, requested_path, logger):
    """Encode all evaluation videos once and persist query-independent vectors."""
    # ``main.py`` constructs the model in training mode.  Normal validation
    # switches it to eval mode in ``validations.forward``; a persisted bank
    # must do the same or dropout makes it inconsistent with original eval.
    model.eval()
    output = Path(requested_path) if requested_path else _default_bank_path(cfg, checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    context = validator.compute_context_info(model, context_loader)
    payload = {
        'schema_version': 1,
        'dataset_name': cfg['dataset_name'],
        'checkpoint': str(Path(checkpoint).resolve()),
        'video_ids': [str(v) for v in context['video_metas']],
        # Store post-encoder FP32 vectors.  Normalization is done once on GPU
        # during benchmark setup, exactly matching model.get_clip_scale_scores.
        'clip_bank': context['video_proposal_feat'].detach().cpu().float(),
        'frame_bank': context['video_feat'].detach().cpu().float(),
    }
    torch.save(payload, output)
    logger.info('ANN context bank: clip=%s frame=%s', tuple(payload['clip_bank'].shape), tuple(payload['frame_bank'].shape))
    return output


def _load_bank(cfg, checkpoint, requested_path):
    path = Path(requested_path) if requested_path else _default_bank_path(cfg, checkpoint)
    if not path.exists():
        raise FileNotFoundError(f'ANN context bank missing: {path}. Run with --ann_build_context_bank first.')
    bank = torch.load(path, map_location='cpu')
    if bank.get('dataset_name') != cfg['dataset_name']:
        raise ValueError('context bank dataset does not match --dataset')
    clip = F.normalize(bank['clip_bank'].cuda(non_blocking=True), dim=-1).contiguous()
    frame = F.normalize(bank['frame_bank'].cuda(non_blocking=True), dim=-1).contiguous()
    return path, bank['video_ids'], clip, frame


def _require_faiss():
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError('ANN benchmark requires faiss-gpu in the active environment') from exc
    return faiss


def _resolve_ivf_nlist(requested_nlist, corpus_size):
    """Return an explicit IVF list count for a branch raw-representation corpus."""
    if int(requested_nlist) > 0:
        return min(int(requested_nlist), int(corpus_size))
    # 2^floor(log2(sqrt(N))).  The power-of-two rounding keeps the value
    # stable and makes clip/frame branches use 256/512 for ActivityNet.
    exponent = max(0, int(math.floor(math.log2(math.sqrt(max(1, corpus_size))))))
    return min(1 << exponent, int(corpus_size))


def _faiss_index(faiss, vectors, index_type, nlist, nprobe, ef_search):
    """Build a serializable FAISS CPU index; IVF may later be cloned to GPU."""
    d = vectors.shape[-1]
    xb = vectors.reshape(-1, d).detach().float().cpu().numpy()
    if index_type == 'flat_full':
        index = faiss.IndexFlatIP(d)
        index.add(xb)
        return index
    if index_type in ('ivf', 'ivf-gpu'):
        nlist = _resolve_ivf_nlist(nlist, xb.shape[0])
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        train_n = min(xb.shape[0], max(10000, nlist * 40))
        sample = xb[torch.randperm(xb.shape[0])[:train_n].numpy()]
        index.train(sample)
        index.add(xb)
        index.nprobe = min(int(nprobe), nlist)
        return index
    if index_type == 'hnsw':
        index = faiss.IndexHNSWFlat(d, 128, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = max(40, int(ef_search))
        index.hnsw.efSearch = int(ef_search)
        index.add(xb)
        return index
    raise ValueError(f'unsupported ANN index: {index_type}')


def _index_paths(cfg, checkpoint, args, bank_path):
    """Stable filenames prevent rebuilding an index for the same context bank.

    The bank is an encoded model output, not merely an alias for a checkpoint
    path.  Include its stat signature so rebuilding a bank (for example after
    correcting eval/train mode) cannot accidentally reuse an index built from
    different vectors.
    """
    root = Path(args.ann_index_dir) if args.ann_index_dir else Path(cfg['model_root']) / 'ann_faiss_indices'
    token = _checkpoint_token(checkpoint)
    stat = Path(bank_path).stat()
    bank_token = hashlib.sha1(f'{Path(bank_path).resolve()}:{stat.st_size}:{stat.st_mtime_ns}'.encode()).hexdigest()[:12]
    if args.ann_index in ('ivf', 'ivf-gpu'):
        nlist_spec = f'nlist{args.ann_nlist}' if args.ann_nlist else 'nlistauto_sqrtpow2'
        spec = f'{args.ann_index}_{nlist_spec}'
    elif args.ann_index == 'hnsw':
        spec = f'hnsw_m128_efc{max(40, int(args.ann_ef_search))}'
    else:
        spec = 'flat_full'
    stem = f'{cfg["dataset_name"]}_{token}_{bank_token}_{spec}'
    return root, root / f'{stem}_clip.faiss', root / f'{stem}_frame.faiss'


def _load_or_build_faiss_index(faiss, vectors, branch, path, cfg, args, logger):
    """Load a persisted CPU FAISS index or build and atomically publish one."""
    if path.exists() and not args.ann_rebuild_index:
        index = faiss.read_index(str(path))
        logger.info('ANN %s index cache hit: %s', branch, path)
        return index

    logger.info('ANN %s index cache miss; building %s (%d raw vectors)',
                branch, args.ann_index, vectors.shape[0] * vectors.shape[1])
    index = _faiss_index(
        faiss, vectors, args.ann_index, args.ann_nlist,
        args.ann_nprobe, args.ann_ef_search)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    faiss.write_index(index, str(temporary))
    temporary.replace(path)
    logger.info('ANN %s index saved: %s', branch, path)
    return index


def _search(index, q, k, gpu_index=False):
    if gpu_index:
        # faiss.contrib.torch_utils installs zero-copy torch bindings for
        # GPU indices.  It is imported when the IVF-GPU index is constructed.
        scores, ids = index.search(q.detach().float().contiguous(), int(k))
        return scores, ids.long()
    scores, ids = index.search(q.detach().float().cpu().numpy(), int(k))
    return torch.from_numpy(scores), torch.from_numpy(ids).long()


def _branch_candidates(scores, raw_ids, repr_count, num_videos, k):
    """Deduplicate raw results by video, retaining the highest returned score."""
    # HNSW can return -1 for unfilled result slots when efSearch is smaller
    # than the requested raw depth.  Those are sentinels, not raw vectors.
    valid_raw = (raw_ids >= 0) & torch.isfinite(scores)
    scores, raw_ids = scores[valid_raw], raw_ids[valid_raw]
    if not len(raw_ids):
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=scores.dtype)
    video_ids = torch.div(raw_ids, repr_count, rounding_mode='floor')
    per_video = torch.full((num_videos,), float('-inf'), device=scores.device)
    per_video.scatter_reduce_(0, video_ids, scores, reduce='amax', include_self=True)
    values, ids = torch.topk(per_video, k=min(k, num_videos))
    valid = torch.isfinite(values)
    return ids[valid], values[valid]


def _elapsed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = fn()
    torch.cuda.synchronize()
    return value, (time.perf_counter() - start) * 1000.0


def _write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _write_summary(path, rows, args, context_source, num_videos):
    """Write a compact timing summary beside the per-query measurements."""
    timing_fields = (
        'clip_search_ms', 'frame_search_ms', 'dedup_ms',
        'clip_fetch_maxsim_ms', 'frame_fetch_maxsim_ms',
        'fusion_top10_ms', 'total_ms',
    )
    summary = {
        'index_type': args.ann_index,
        'faiss_backend': ('none_torch_cuda' if args.ann_index == 'origin'
                          else 'gpu' if args.ann_index == 'ivf-gpu' else 'cpu'),
        'context_source': str(context_source),
        'num_queries': len(rows),
    }
    if args.ann_index == 'origin':
        summary['ann_parameters_applied'] = False
    else:
        summary.update({
            'ann_parameters_applied': True,
            'clip_raw_k': args.ann_clip_raw_k,
            'frame_raw_k': args.ann_frame_raw_k,
            'candidate_k': args.ann_candidate_k,
        })
        if args.ann_index in ('ivf', 'ivf-gpu'):
            clip_nlist = _resolve_ivf_nlist(args.ann_nlist, num_videos * 32)
            frame_nlist = _resolve_ivf_nlist(args.ann_nlist, num_videos * 128)
            summary.update({
                'ivf_nlist_rule': ('explicit' if args.ann_nlist else '2^floor(log2(sqrt(raw_corpus)))'),
                'clip_ivf_nlist': clip_nlist,
                'frame_ivf_nlist': frame_nlist,
                'clip_ivf_nprobe': min(int(args.ann_nprobe), clip_nlist),
                'frame_ivf_nprobe': min(int(args.ann_nprobe), frame_nlist),
            })
        elif args.ann_index == 'hnsw':
            summary.update({
                'hnsw_m': 128,
                'hnsw_ef_construction': max(40, int(args.ann_ef_search)),
                # HNSW must inspect at least k entries to return k usable raw
                # candidates; these are the effective query-time values.
                'clip_hnsw_ef_search': max(int(args.ann_ef_search), int(args.ann_clip_raw_k)),
                'frame_hnsw_ef_search': max(int(args.ann_ef_search), int(args.ann_frame_raw_k)),
            })
    for field in timing_fields:
        values = sorted(float(row[field]) for row in rows)
        if not values:
            continue
        def percentile(p):
            pos = (len(values) - 1) * p
            lo, hi = int(pos), min(int(pos) + 1, len(values) - 1)
            return values[lo] + (values[hi] - values[lo]) * (pos - lo)
        summary[f'{field}_mean'] = sum(values) / len(values)
        summary[f'{field}_p50'] = percentile(.50)
        summary[f'{field}_p95'] = percentile(.95)
    # The normal single-GT evaluator maps ``video_id#enc#...`` to ``video_id``.
    # Keep the exact same convention here so an ANN run reports retrieval
    # quality next to its latency.  Metrics are deliberately calculated after
    # the timed region; converting indices to IDs must not affect timing.
    if rows and 'top10_video_ids' in rows[0]:
        n = len(rows)
        for k in (1, 5, 10):
            hits = sum(
                row['gt_video_id'] in row['top10_video_ids'].split('|')[:k]
                for row in rows
            )
            summary[f'r{k}'] = 100.0 * hits / n
    summary_path = path.with_name(f'{path.stem}_summary.json')
    with summary_path.open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)
    return summary_path


def run_ann_benchmark(model, validator, context_loader, query_loader, cfg, args, logger):
    """Run original evaluation scoring or FAISS retrieval with cross-branch reranking."""
    net = _model(model)
    net.eval()
    faiss = None if args.ann_index == 'origin' else _require_faiss()
    if args.ann_index == 'origin':
        # Follow normal evaluation exactly for context preparation.  Only the
        # retrieval sub-steps below are timed; context/query encoding stays
        # outside the requested search-latency breakdown.
        context = validator.compute_context_info(model, context_loader)
        video_ids = [str(video_id) for video_id in context['video_metas']]
        clip_bank = F.normalize(context['video_proposal_feat'], dim=-1).contiguous()
        frame_bank = F.normalize(context['video_feat'], dim=-1).contiguous()
        context_source = 'origin: validator.compute_context_info(test_context_dataloader)'
        logger.info('Origin context encoded by normal evaluator; V=%d D=%d; original GMMFormer-v2 GPU retrieval backend', clip_bank.shape[0], clip_bank.shape[-1])
        clip_index = frame_index = None
    else:
        bank_path, video_ids, clip_bank, frame_bank = _load_bank(cfg, args.resume, args.ann_context_bank)
        context_source = f'bank: {bank_path}'
        V = clip_bank.shape[0]
        backend = 'GPU (after CPU cache load)' if args.ann_index == 'ivf-gpu' else 'CPU'
        logger.info('ANN bank loaded from %s; V=%d D=%d; FAISS search backend=%s',
                    bank_path, V, clip_bank.shape[-1], backend)
        _, clip_index_path, frame_index_path = _index_paths(cfg, args.resume, args, bank_path)
        clip_index = _load_or_build_faiss_index(
            faiss, clip_bank, 'clip', clip_index_path, cfg, args, logger)
        frame_index = _load_or_build_faiss_index(
            faiss, frame_bank, 'frame', frame_index_path, cfg, args, logger)
        if args.ann_index in ('ivf', 'ivf-gpu'):
            # nprobe is a query-time setting, so it remains configurable even
            # when the trained IVF partition is loaded from disk.
            clip_index.nprobe = min(int(args.ann_nprobe), clip_index.nlist)
            frame_index.nprobe = min(int(args.ann_nprobe), frame_index.nlist)
        gpu_resources = None
        if args.ann_index == 'ivf-gpu':
            try:
                import faiss.contrib.torch_utils  # installs torch CUDA bindings
            except ImportError as exc:
                raise RuntimeError('ivf-gpu requires the FAISS GPU torch bindings') from exc
            if faiss.get_num_gpus() < 1:
                raise RuntimeError('ivf-gpu requested but this FAISS build sees no CUDA GPU')
            gpu_resources = faiss.StandardGpuResources()
            gpu_device = torch.cuda.current_device()
            clip_index = faiss.index_cpu_to_gpu(gpu_resources, gpu_device, clip_index)
            frame_index = faiss.index_cpu_to_gpu(gpu_resources, gpu_device, frame_index)
            clip_index.nprobe = min(int(args.ann_nprobe), clip_index.nlist)
            frame_index.nprobe = min(int(args.ann_nprobe), frame_index.nlist)
            logger.info('ANN IVF indices cloned to FAISS GPU %d; CPU cache remains on disk', gpu_device)
        if args.ann_index == 'hnsw':
            # Faiss HNSW must examine at least k entries to return k valid raw
            # candidates.  Keep a user-specified larger efSearch, but never allow
            # it to silently truncate the requested 832/2948-depth searches.
            clip_index.hnsw.efSearch = max(int(args.ann_ef_search), int(args.ann_clip_raw_k))
            frame_index.hnsw.efSearch = max(int(args.ann_ef_search), int(args.ann_frame_raw_k))
    V = clip_bank.shape[0]
    if frame_bank.shape[0] != V or clip_bank.shape[1] != 32 or frame_bank.shape[1] != 128:
        raise ValueError('expected GMMFormer-v2 contexts [V,32,D] and [V,128,D]')
    out = Path(args.ann_output) if args.ann_output else Path(cfg['model_root']) / 'ann_benchmark' / f'{args.ann_index}_cross_branch_only.csv'
    rows = []
    for batch in query_loader:
        batch = gpu(batch)
        query_ids = batch[-1]
        q = F.normalize(net.encode_query(batch[0], batch[1]), dim=-1)
        for qi, query_id in enumerate(query_ids):
            if args.ann_max_queries and len(rows) >= args.ann_max_queries:
                break
            one_q = q[qi:qi + 1]
            gt_video_id = query_id.split('#', 1)[0]
            if args.ann_index == 'origin':
                # This is algebraically the original GMMFormer-v2 evaluator:
                # normalized [V,R,D] context × normalized q, max over R, then
                # the configured branch fusion and final top-k.  Query encoding
                # is intentionally outside this timer, matching ANN timings.
                (clip_score, clip_ms) = _elapsed(lambda: torch.matmul(clip_bank, one_q[0]).amax(dim=1))
                (frame_score, frame_ms) = _elapsed(lambda: torch.matmul(frame_bank, one_q[0]).amax(dim=1))
                (top_out, top_ms) = _elapsed(lambda: torch.topk(
                    cfg['clip_scale_w'] * clip_score + cfg['frame_scale_w'] * frame_score, 10))
                _, top_indices = top_out
                rows.append({'query_id': query_id, 'gt_video_id': gt_video_id, 'index_type': 'origin', 'clip_search_ms': clip_ms, 'frame_search_ms': frame_ms, 'dedup_ms': 0.0, 'clip_fetch_maxsim_ms': 0.0, 'frame_fetch_maxsim_ms': 0.0, 'fusion_top10_ms': top_ms, 'total_ms': clip_ms + frame_ms + top_ms, 'clip_unique_count': V, 'frame_unique_count': V, 'both_count': V, 'clip_only_count': 0, 'frame_only_count': 0, 'top10_video_ids': '|'.join(video_ids[i] for i in top_indices.detach().cpu().tolist())})
                continue
            if args.ann_index == 'flat_full':
                (clip_out, clip_ms) = _elapsed(lambda: _search(clip_index, one_q, V * 32))
                (frame_out, frame_ms) = _elapsed(lambda: _search(frame_index, one_q, V * 128))
                clip_s, clip_i = clip_out; frame_s, frame_i = frame_out
                # FAISS search returns CPU arrays.  Candidate reduction and
                # every later stage are deliberately measured on GPU.
                (cand, dedup_ms) = _elapsed(lambda: (
                    _branch_candidates(clip_s[0].cuda(), clip_i[0].cuda(), 32, V, V),
                    _branch_candidates(frame_s[0].cuda(), frame_i[0].cuda(), 128, V, V),
                ))
                (clip_v, clip_score), (frame_v, frame_score) = cand
                # Both vectors contain every video, ordered by their exact maxsim.
                cv = torch.empty(V, device=q.device); fv = torch.empty(V, device=q.device)
                cv[clip_v] = clip_score; fv[frame_v] = frame_score
                (top_out, top_ms) = _elapsed(lambda: torch.topk(cfg['clip_scale_w'] * cv + cfg['frame_scale_w'] * fv, 10))
                _, top_indices = top_out
                rows.append({'query_id': query_id, 'gt_video_id': gt_video_id, 'index_type': 'flat_full', 'clip_search_ms': clip_ms, 'frame_search_ms': frame_ms, 'dedup_ms': dedup_ms, 'clip_fetch_maxsim_ms': 0.0, 'frame_fetch_maxsim_ms': 0.0, 'fusion_top10_ms': top_ms, 'total_ms': clip_ms + frame_ms + dedup_ms + top_ms, 'clip_unique_count': V, 'frame_unique_count': V, 'both_count': V, 'clip_only_count': 0, 'frame_only_count': 0, 'top10_video_ids': '|'.join(video_ids[i] for i in top_indices.detach().cpu().tolist())})
                continue
            gpu_index = args.ann_index == 'ivf-gpu'
            (clip_out, clip_ms) = _elapsed(lambda: _search(clip_index, one_q, args.ann_clip_raw_k, gpu_index))
            (frame_out, frame_ms) = _elapsed(lambda: _search(frame_index, one_q, args.ann_frame_raw_k, gpu_index))
            (cand, dedup_ms) = _elapsed(lambda: (
                _branch_candidates(clip_out[0][0].cuda(), clip_out[1][0].cuda(), 32, V, args.ann_candidate_k),
                _branch_candidates(frame_out[0][0].cuda(), frame_out[1][0].cuda(), 128, V, args.ann_candidate_k),
            ))
            (clip_v, clip_s), (frame_v, frame_s) = cand
            clip_full = torch.full((V,), float('-inf'), device=q.device); clip_full[clip_v] = clip_s
            frame_full = torch.full((V,), float('-inf'), device=q.device); frame_full[frame_v] = frame_s
            both_mask = torch.isfinite(clip_full) & torch.isfinite(frame_full)
            clip_only = torch.nonzero(torch.isfinite(clip_full) & ~torch.isfinite(frame_full), as_tuple=False).squeeze(1)
            frame_only = torch.nonzero(torch.isfinite(frame_full) & ~torch.isfinite(clip_full), as_tuple=False).squeeze(1)
            def clip_fetch():
                if not len(frame_only): return torch.empty(0, device=q.device)
                return torch.einsum('d,nrd->nr', one_q[0], clip_bank.index_select(0, frame_only)).amax(1)
            def frame_fetch():
                if not len(clip_only): return torch.empty(0, device=q.device)
                return torch.einsum('d,nrd->nr', one_q[0], frame_bank.index_select(0, clip_only)).amax(1)
            clip_missing, clip_fetch_ms = _elapsed(clip_fetch)
            frame_missing, frame_fetch_ms = _elapsed(frame_fetch)
            clip_full[frame_only] = clip_missing; frame_full[clip_only] = frame_missing
            union = torch.nonzero(torch.isfinite(clip_full) & torch.isfinite(frame_full), as_tuple=False).squeeze(1)
            def fuse(): return torch.topk(cfg['clip_scale_w'] * clip_full[union] + cfg['frame_scale_w'] * frame_full[union], min(10, len(union)))
            top_out, top_ms = _elapsed(fuse)
            _, top_positions = top_out
            top_indices = union[top_positions]
            total = clip_ms + frame_ms + dedup_ms + clip_fetch_ms + frame_fetch_ms + top_ms
            rows.append({'query_id': query_id, 'gt_video_id': gt_video_id, 'index_type': args.ann_index, 'clip_search_ms': clip_ms, 'frame_search_ms': frame_ms, 'dedup_ms': dedup_ms, 'clip_fetch_maxsim_ms': clip_fetch_ms, 'frame_fetch_maxsim_ms': frame_fetch_ms, 'fusion_top10_ms': top_ms, 'total_ms': total, 'clip_unique_count': len(clip_v), 'frame_unique_count': len(frame_v), 'both_count': int(both_mask.sum()), 'clip_only_count': len(clip_only), 'frame_only_count': len(frame_only), 'top10_video_ids': '|'.join(video_ids[i] for i in top_indices.detach().cpu().tolist())})
            if args.ann_max_queries and len(rows) >= args.ann_max_queries:
                break
        if args.ann_max_queries and len(rows) >= args.ann_max_queries:
            break
    _write_rows(out, rows)
    summary_path = _write_summary(out, rows, args, context_source, V)
    logger.info('ANN benchmark written: %s (%d queries); summary: %s', out, len(rows), summary_path)
    if rows:
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        logger.info('ANN single-GT recall: R@1 %.2f R@5 %.2f R@10 %.2f',
                    summary.get('r1', float('nan')), summary.get('r5', float('nan')),
                    summary.get('r10', float('nan')))
