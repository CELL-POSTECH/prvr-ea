#!/usr/bin/env python3
"""CLIP4Clip meanP latency worker with CPU FAISS IVF/HNSW variants.

Gallery vectors are post-visual-encoder, mean-pooled 512-D features.  This is
intentional: the benchmark's e2e definition excludes one-time gallery/video
encoding and measures CLIP text encoding plus retrieval only.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from argparse import Namespace
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--repo", type=Path, required=True)
parser.add_argument("--variant", choices=("flat", "ivf", "hnsw"), required=True)
parser.add_argument("--corpus-dir", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--gpu", default="0")
parser.add_argument("--faiss-threads", type=int, default=16)
parser.add_argument("--search-vector-budget", type=int, default=7_237_555)
parser.add_argument("--search-chunk-videos", type=int, default=0)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(args.repo)); sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

import faiss  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from latency.synthetic_corpus import open_video_features  # noqa: E402
from latency.results import append_detail, export_wide_qps  # noqa: E402
from modules.modeling import CLIP4Clip  # noqa: E402


QUERY_BSZ = 32  # CLIP4Clip raw-frame eval script's BATCH_SIZE_VAL default


def cuda_ms(fn):
    torch.cuda.synchronize(); start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record(); value = fn(); end.record(); torch.cuda.synchronize()
    return value, float(start.elapsed_time(end))


def faiss_parameters(videos: int):
    if videos <= 100_000: return 1024, 32
    if videos <= 500_000: return 2048, 32
    if videos <= 1_000_000: return 4096, 32
    return 8192, 32


def normalized_bank(corpus_dir: Path):
    print("loading and normalizing synthetic gallery on CPU...", flush=True)
    raw = open_video_features(corpus_dir)
    # FAISS needs a contiguous in-memory CPU bank. [5M,512] FP32 is 9.54 GiB.
    bank = np.asarray(raw, dtype=np.float32).copy()
    faiss.normalize_L2(bank)
    return bank


def cached_index(corpus_dir: Path, bank: np.ndarray, variant: str):
    index_dir = corpus_dir / "faiss_indices"; index_dir.mkdir(exist_ok=True)
    nlist, nprobe = faiss_parameters(bank.shape[0])
    filename = index_dir / (f"clip4clip_{variant}_nlist{nlist}.faiss" if variant == "ivf" else "clip4clip_hnsw_m32.faiss")
    if filename.exists():
        print(f"loading cached {variant} index: {filename}", flush=True)
        index = faiss.read_index(str(filename))
        if variant == "ivf": index.nprobe = nprobe
        else: index.hnsw.efSearch = 128
        return index, 0.0, nlist if variant == "ivf" else "", nprobe if variant == "ivf" else ""
    start = time.perf_counter()
    print(f"building CPU FAISS {variant} index for {bank.shape[0]} videos (one-time)...", flush=True)
    if variant == "ivf":
        quantizer = faiss.IndexFlatIP(512)
        index = faiss.IndexIVFFlat(quantizer, 512, nlist, faiss.METRIC_INNER_PRODUCT)
        train_count = min(bank.shape[0], max(39 * nlist, 100_000))
        sample = bank[np.linspace(0, bank.shape[0] - 1, train_count, dtype=np.int64)]
        index.train(sample); index.add(bank); index.nprobe = nprobe
    else:
        index = faiss.IndexHNSWFlat(512, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200; index.hnsw.efSearch = 128; index.add(bank)
    elapsed = time.perf_counter() - start
    faiss.write_index(index, str(filename))
    return index, elapsed, nlist if variant == "ivf" else "", nprobe if variant == "ivf" else ""


def make_model():
    config = Namespace(local_rank=0, max_words=32, max_frames=128, loose_type=True,
                       sim_header="meanP", linear_patch="2d", cross_num_hidden_layers=4,
                       pretrained_clip_name="ViT-B/32")
    return CLIP4Clip.from_pretrained("cross-base", cache_dir="", task_config=config).cuda().eval()


def gpu_flat(query, bank, chunk=100_000):
    k = min(10, int(bank.shape[0]))
    top_values = torch.full((query.shape[0], k), float("-inf"), device=query.device)
    top_ids = torch.full((query.shape[0], k), -1, dtype=torch.long, device=query.device)
    for start in range(0, bank.shape[0], chunk):
        stop = min(start + chunk, bank.shape[0])
        context = torch.from_numpy(bank[start:stop]).to(query.device, non_blocking=True)
        scores = query @ context.t()
        values, local = torch.topk(scores, k, dim=1)
        ids = local + start
        merged_values = torch.cat((top_values, values), dim=1)
        merged_ids = torch.cat((top_ids, ids), dim=1)
        top_values, order = torch.topk(merged_values, k, dim=1)
        top_ids = merged_ids.gather(1, order)
    return top_values, top_ids


def main():
    faiss.omp_set_num_threads(args.faiss_threads)
    bank = normalized_bank(args.corpus_dir)
    videos = int(bank.shape[0])
    model = make_model()
    device = next(model.parameters()).device
    generator = torch.Generator(device="cpu").manual_seed(9571)
    # Valid token IDs; the largest position acts as CLIP's EOT selection.
    tokens = torch.randint(1, 49_000, (QUERY_BSZ, 32), generator=generator, dtype=torch.long)
    # OpenAI CLIP's EOT token.  `get_sequence_output` selects it with argmax.
    tokens[:, -1] = 49_407
    tokens, mask = tokens.to(device), torch.ones((QUERY_BSZ, 32), dtype=torch.long, device=device)
    segments = torch.zeros_like(tokens)
    with torch.no_grad():
        model.get_sequence_output(tokens, segments, mask)
        sequence, query_ms = cuda_ms(lambda: model.get_sequence_output(tokens, segments, mask))
        query = F.normalize(sequence.squeeze(1), dim=-1)
        search_chunk = int(args.search_chunk_videos or args.search_vector_budget)
        _, exact_ids = cuda_ms(lambda: gpu_flat(query, bank, chunk=search_chunk))[0]
        if args.variant == "flat":
            # Re-time after the warm-up to report the native CLIP4Clip path.
            (_, ids), search_ms = cuda_ms(lambda: gpu_flat(query, bank, chunk=search_chunk))
            method, backend, build_sec, nlist, nprobe, overlap = "CLIP4Clip", "torch_gpu_exact", 0.0, "", "", 1.0
        else:
            index, build_sec, nlist, nprobe = cached_index(args.corpus_dir, bank, args.variant)
            torch.cuda.synchronize()
            start = time.perf_counter()
            # D2H transfer is required by the CPU ANN service and is included.
            _, ids = index.search(query.detach().float().cpu().numpy(), min(10, videos))
            search_ms = (time.perf_counter() - start) * 1000.0
            overlap = float(np.mean([len(set(a.tolist()) & set(b.tolist())) / float(min(10, videos)) for a, b in zip(ids, exact_ids.cpu().numpy())]))
            method = "CLIP4Clip+IVF" if args.variant == "ivf" else "CLIP4Clip+HNSW"
            backend = "cpu_faiss"
    e2e = query_ms + search_ms
    append_detail(args.output, {"corpus": videos, "method": method, "backend": backend, "device": str(device),
        "query_bsz": QUERY_BSZ, "gallery_encoder_bsz": "",
        "search_chunk_videos": min(search_chunk, videos), "reprs_per_video": 1,
        "query_emb_time_ms": query_ms, "search_time_ms": search_ms,
        "e2e_latency_ms": e2e, "e2e_qps": 1000.0 * QUERY_BSZ / e2e if e2e else 0.0,
        "index_build_sec": build_sec, "nlist": nlist, "nprobe": nprobe, "hnsw_m": 32 if args.variant == "hnsw" else "",
        "ef_search": 128 if args.variant == "hnsw" else "", "top10_overlap_vs_exact": overlap, "status": "ok"})
    export_wide_qps(args.output, args.output.parent / "qps.csv")
    print(f"latency complete: {method} V={videos} qbsz={QUERY_BSZ}")


if __name__ == "__main__":
    main()
