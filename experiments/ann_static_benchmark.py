"""Shared ANN benchmark for PRVR models with static dual-branch context banks.

This module is opt-in: model evaluators invoke it only when
``PRVR_STATIC_ANN_MODE`` is set.  It never changes their normal evaluation
path.  Both branches must expose query-independent context vectors shaped
``[videos, representations_per_video, dim]`` and query vectors shaped
``[queries, dim]``.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def enabled() -> bool:
    return bool(os.environ.get("PRVR_STATIC_ANN_MODE"))


def _int(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _mode() -> str:
    mode = os.environ["PRVR_STATIC_ANN_MODE"]
    if mode not in {"origin", "ivf", "ivf-gpu", "hnsw"}:
        raise ValueError(f"unsupported PRVR_STATIC_ANN_MODE: {mode}")
    return mode


def _auto_nlist(corpus: int) -> int:
    return 1 << max(0, int(math.floor(math.log2(math.sqrt(max(1, corpus))))))


def _branch_setting(prefix: str, raw_corpus: int, defaults: dict[str, int]) -> dict[str, int]:
    nlist = int(os.environ.get(f"PRVR_STATIC_ANN_{prefix}_NLIST", "0"))
    if nlist < 0:
        raise ValueError("nlist must be >= 0")
    nlist = _auto_nlist(raw_corpus) if nlist == 0 else min(nlist, raw_corpus)
    return {
        "k": _int(f"PRVR_STATIC_ANN_{prefix}_K", defaults["k"]),
        "nlist": nlist,
        "nprobe": min(_int(f"PRVR_STATIC_ANN_{prefix}_NPROBE", defaults["nprobe"]), nlist),
        "ef": _int(f"PRVR_STATIC_ANN_{prefix}_EF_SEARCH", defaults["ef"]),
    }


def _elapsed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = fn()
    torch.cuda.synchronize()
    return value, (time.perf_counter() - start) * 1000.0


def _faiss():
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("ANN benchmark needs faiss-cpu/faiss-gpu in the active environment") from exc
    return faiss


def _build_index(faiss, vectors: torch.Tensor, mode: str, setting: dict[str, int], hnsw_m: int, hnsw_efc: int):
    vectors_np = vectors.reshape(-1, vectors.shape[-1]).detach().float().cpu().numpy()
    dim = vectors_np.shape[1]
    if mode in {"ivf", "ivf-gpu"}:
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, setting["nlist"], faiss.METRIC_INNER_PRODUCT)
        train_n = min(len(vectors_np), max(10_000, setting["nlist"] * 40))
        index.train(vectors_np[torch.randperm(len(vectors_np))[:train_n].numpy()])
        index.add(vectors_np)
        index.nprobe = setting["nprobe"]
        return index
    index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = hnsw_efc
    index.hnsw.efSearch = max(setting["ef"], setting["k"])
    index.add(vectors_np)
    return index


def _index_cache_path(root: Path, method: str, branch: str, mode: str, checkpoint: str, setting: dict[str, int], hnsw_m: int, hnsw_efc: int) -> Path:
    checkpoint_path = Path(checkpoint)
    stat = checkpoint_path.stat()
    token = hashlib.sha1(f"{checkpoint_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()[:12]
    if mode in {"ivf", "ivf-gpu"}:
        spec = f"{mode}_nlist{setting['nlist']}"
    else:
        spec = f"hnsw_m{hnsw_m}_efc{hnsw_efc}"
    return root / f"{method}_{token}_{spec}_{branch}.faiss"


def _load_index(faiss, bank: torch.Tensor, branch: str, cache_root: Path, method: str, mode: str,
                checkpoint: str, setting: dict[str, int], hnsw_m: int, hnsw_efc: int):
    path = _index_cache_path(cache_root, method, branch, mode, checkpoint, setting, hnsw_m, hnsw_efc)
    if path.exists():
        index = faiss.read_index(str(path))
        print(f"ANN {method}/{branch}: index cache hit {path}", flush=True)
    else:
        print(f"ANN {method}/{branch}: building {mode} index ({bank.shape[0] * bank.shape[1]} raw repr)", flush=True)
        index = _build_index(faiss, bank, mode, setting, hnsw_m, hnsw_efc)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        faiss.write_index(index, str(temp))
        temp.replace(path)
    if mode in {"ivf", "ivf-gpu"}:
        index.nprobe = setting["nprobe"]
    elif mode == "hnsw":
        index.hnsw.efSearch = max(setting["ef"], setting["k"])
    return index


def _search(index, query: torch.Tensor, k: int, gpu_index: bool):
    if gpu_index:
        scores, ids = index.search(query.detach().float().contiguous(), k)
        return scores, ids.long()
    scores, ids = index.search(query.detach().float().cpu().numpy(), k)
    return torch.from_numpy(scores), torch.from_numpy(ids).long()


def _dedup(scores: torch.Tensor, raw_ids: torch.Tensor, reprs: int, videos: int, keep: int):
    valid = (raw_ids >= 0) & torch.isfinite(scores)
    scores, raw_ids = scores[valid], raw_ids[valid]
    if not len(raw_ids):
        return torch.empty(0, dtype=torch.long, device=scores.device), torch.empty(0, device=scores.device)
    video_ids = torch.div(raw_ids, reprs, rounding_mode="floor")
    per_video = torch.full((videos,), float("-inf"), device=scores.device)
    per_video.scatter_reduce_(0, video_ids, scores, reduce="amax", include_self=True)
    values, ids = torch.topk(per_video, min(keep, videos))
    valid = torch.isfinite(values)
    return ids[valid], values[valid]


def _gt_index(video_ids: list[str], query_id: str) -> int:
    raw = str(query_id).split("#", 1)[0]
    candidates = (raw, raw[2:] if raw.startswith("v_") else None, "v_" + raw)
    for candidate in candidates:
        if candidate is not None:
            try:
                return video_ids.index(candidate)
            except ValueError:
                continue
    raise KeyError(f"ground-truth video absent from context bank: {raw}")


def _as_bank(value: torch.Tensor, name: str) -> torch.Tensor:
    """Canonicalize a static context branch to ``[V, R, D]``.

    GMMFormer-like single-frame branches are emitted as ``[V, D]`` by their
    original encoders, whereas multi-representation branches are already
    ``[V, R, D]``.  This is only a shape convention: it does not pool,
    truncate, or otherwise alter any model representation.
    """
    if value.ndim == 2:
        value = value.unsqueeze(1)
    if value.ndim != 3:
        raise ValueError(f"{name} context bank must be [V,D] or [V,R,D], got {tuple(value.shape)}")
    return value


def _write(path: Path, rows: list[dict], summary: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with path.with_name(path.stem + "_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def run(*, method: str, checkpoint: str, output: str, index_dir: str, video_ids, left_bank: torch.Tensor,
        right_bank: torch.Tensor, query_batches, left_weight: float, right_weight: float,
        left_name: str = "clip", right_name: str = "frame") -> list[float]:
    """Benchmark one model and return normal-evaluator compatible recall metrics.

    ``query_batches`` yields ``(left_query, right_query, query_ids)``.  Query
    encoding and context encoding happen in the caller and are deliberately
    outside the timed retrieval interval.
    """
    mode = _mode()
    video_ids = [str(item) for item in video_ids]
    left_bank = F.normalize(_as_bank(left_bank, left_name), dim=-1).contiguous()
    right_bank = F.normalize(_as_bank(right_bank, right_name), dim=-1).contiguous()
    videos, left_reprs, dim = left_bank.shape
    if right_bank.shape[0] != videos or right_bank.shape[2] != dim:
        raise ValueError("ANN branches must share video count and embedding dimension")
    right_reprs = right_bank.shape[1]
    defaults = {"k": 1, "nprobe": 64, "ef": 1}
    left = _branch_setting("LEFT", videos * left_reprs, {**defaults, "k": left_reprs})
    right = _branch_setting("RIGHT", videos * right_reprs, {**defaults, "k": right_reprs})
    # Model-specific wrapper supplies the L@30-derived default K values.
    keep = _int("PRVR_STATIC_ANN_CANDIDATE_K", 30)
    hnsw_m = _int("PRVR_STATIC_ANN_HNSW_M", 128)
    hnsw_efc = _int("PRVR_STATIC_ANN_HNSW_EF_CONSTRUCTION", 256)
    checkpoint = str(Path(checkpoint).resolve())
    output_path = Path(output)
    if mode == "ivf-gpu":
        left["k"] = min(left["k"], 2048)
        right["k"] = min(right["k"], 2048)

    faiss = None
    left_index = right_index = None
    if mode != "origin":
        faiss = _faiss()
        cache_root = Path(index_dir)
        left_index = _load_index(faiss, left_bank, left_name, cache_root, method, mode, checkpoint, left, hnsw_m, hnsw_efc)
        right_index = _load_index(faiss, right_bank, right_name, cache_root, method, mode, checkpoint, right, hnsw_m, hnsw_efc)
        if mode == "ivf-gpu":
            import faiss.contrib.torch_utils  # noqa: F401
            resources = faiss.StandardGpuResources()
            device = torch.cuda.current_device()
            left_index = faiss.index_cpu_to_gpu(resources, device, left_index)
            right_index = faiss.index_cpu_to_gpu(resources, device, right_index)
            left_index.nprobe, right_index.nprobe = left["nprobe"], right["nprobe"]
            print(f"ANN {method}: IVF indices cloned to GPU {device}; k={left['k']}/{right['k']}", flush=True)

    rows: list[dict] = []
    max_queries = int(os.environ.get("PRVR_STATIC_ANN_MAX_QUERIES", "0"))
    for left_query, right_query, query_ids in query_batches:
        left_query = F.normalize(left_query, dim=-1)
        right_query = F.normalize(right_query, dim=-1)
        for index, query_id in enumerate(query_ids):
            if max_queries and len(rows) >= max_queries:
                break
            lq, rq = left_query[index], right_query[index]
            gt = _gt_index(video_ids, query_id)
            if mode == "origin":
                left_score, left_ms = _elapsed(lambda: torch.matmul(left_bank, lq).amax(dim=1))
                right_score, right_ms = _elapsed(lambda: torch.matmul(right_bank, rq).amax(dim=1))
                top, fusion_ms = _elapsed(lambda: torch.topk(left_weight * left_score + right_weight * right_score, 10))
                top_ids = top.indices
                dedup_ms = left_fetch_ms = right_fetch_ms = 0.0
            else:
                gpu_index = mode == "ivf-gpu"
                (left_out, left_ms) = _elapsed(lambda: _search(left_index, lq[None], left["k"], gpu_index))
                (right_out, right_ms) = _elapsed(lambda: _search(right_index, rq[None], right["k"], gpu_index))
                ((left_ids, left_scores), (right_ids, right_scores)), dedup_ms = _elapsed(lambda: (
                    _dedup(left_out[0][0].cuda(), left_out[1][0].cuda(), left_reprs, videos, keep),
                    _dedup(right_out[0][0].cuda(), right_out[1][0].cuda(), right_reprs, videos, keep),
                ))
                left_full = torch.full((videos,), float("-inf"), device=lq.device)
                right_full = torch.full((videos,), float("-inf"), device=lq.device)
                left_full[left_ids] = left_scores
                right_full[right_ids] = right_scores
                left_only = torch.nonzero(torch.isfinite(left_full) & ~torch.isfinite(right_full), as_tuple=False).squeeze(1)
                right_only = torch.nonzero(torch.isfinite(right_full) & ~torch.isfinite(left_full), as_tuple=False).squeeze(1)
                left_missing, left_fetch_ms = _elapsed(lambda: (
                    torch.einsum("d,nrd->nr", lq, left_bank.index_select(0, right_only)).amax(1)
                    if len(right_only) else torch.empty(0, device=lq.device)
                ))
                right_missing, right_fetch_ms = _elapsed(lambda: (
                    torch.einsum("d,nrd->nr", rq, right_bank.index_select(0, left_only)).amax(1)
                    if len(left_only) else torch.empty(0, device=lq.device)
                ))
                left_full[right_only] = left_missing
                right_full[left_only] = right_missing
                union = torch.nonzero(torch.isfinite(left_full) & torch.isfinite(right_full), as_tuple=False).squeeze(1)
                top, fusion_ms = _elapsed(lambda: torch.topk(
                    left_weight * left_full[union] + right_weight * right_full[union], min(10, len(union))))
                top_ids = union[top.indices]
            total = left_ms + right_ms + dedup_ms + left_fetch_ms + right_fetch_ms + fusion_ms
            rows.append({
                "query_id": str(query_id), "gt_video_id": str(query_id).split("#", 1)[0], "index_type": mode,
                "left_search_ms": left_ms, "right_search_ms": right_ms, "dedup_ms": dedup_ms,
                "left_fetch_maxsim_ms": left_fetch_ms, "right_fetch_maxsim_ms": right_fetch_ms,
                "fusion_top10_ms": fusion_ms, "total_ms": total,
                "top10_video_ids": "|".join(video_ids[int(item)] for item in top_ids.detach().cpu().tolist()),
            })
            if len(rows) % 1000 == 0:
                print(f"ANN {method}/{mode}: {len(rows)} queries complete", flush=True)
        if max_queries and len(rows) >= max_queries:
            break

    def mean(field):
        return sum(float(row[field]) for row in rows) / max(1, len(rows))
    summary = {
        "method": method, "index_type": mode, "num_queries": len(rows),
        "left_branch": left_name, "right_branch": right_name,
        "left_reprs_per_video": left_reprs, "right_reprs_per_video": right_reprs,
        "left_weight": left_weight, "right_weight": right_weight,
        "left": left, "right": right, "candidate_k": keep,
        "hnsw_m": hnsw_m, "hnsw_ef_construction": hnsw_efc,
        "faiss_backend": "torch_cuda" if mode == "origin" else "gpu" if mode == "ivf-gpu" else "cpu",
    }
    for field in ("left_search_ms", "right_search_ms", "dedup_ms", "left_fetch_maxsim_ms", "right_fetch_maxsim_ms", "fusion_top10_ms", "total_ms"):
        summary[f"{field}_mean"] = mean(field)
    for k in (1, 5, 10):
        summary[f"r{k}"] = 100.0 * sum(row["gt_video_id"] in row["top10_video_ids"].split("|")[:k] for row in rows) / max(1, len(rows))
    _write(output_path, rows, summary)
    print(f"ANN {method}/{mode}: wrote {output_path} ({len(rows)} queries)", flush=True)
    return [summary["r1"], summary["r5"], summary["r10"], float("nan"), summary["r1"] + summary["r5"] + summary["r10"]]
