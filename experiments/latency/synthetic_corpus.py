"""Contiguous 512-D synthetic gallery generation for latency benchmarks.

The corpus intentionally uses a memmap instead of one HDF5 dataset per video:
the latter becomes impractical at multi-million-video scale.  Feature values
are deterministic FP32 normal samples and are only used to exercise the model
and retrieval kernels; no recall metric is defined for this corpus.
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np


DIM = 512
DEFAULT_QUERY_COUNT = 256
DEFAULT_QUERY_LENGTH = 30


def corpus_dir(root: Path, corpus_size: int) -> Path:
    return root / f"corpus_{corpus_size}"


def paths(root: Path, corpus_size: int) -> tuple[Path, Path, Path]:
    directory = corpus_dir(root, corpus_size)
    return directory, directory / "video_features.f32", directory / "manifest.json"


def ensure_corpus(root: Path, corpus_size: int, *, seed: int = 9527,
                  chunk_videos: int = 8192) -> Path:
    """Create (or validate) a ``[V, 1, 512]`` FP32 video memmap."""
    if corpus_size < 1:
        raise ValueError("corpus_size must be positive")
    directory, features, manifest = paths(root, corpus_size)
    expected_bytes = corpus_size * DIM * np.dtype(np.float32).itemsize
    if manifest.exists() and features.exists():
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        valid = (
            metadata.get("num_videos") == corpus_size
            and metadata.get("shape") == [corpus_size, 1, DIM]
            and metadata.get("dtype") == "float32"
            and features.stat().st_size == expected_bytes
        )
        if valid:
            return directory
        raise RuntimeError(f"invalid existing synthetic corpus: {directory}")
    if features.exists() or manifest.exists():
        raise RuntimeError(f"partial synthetic corpus exists: {directory}; remove it before retrying")

    directory.mkdir(parents=True, exist_ok=True)
    temporary = features.with_suffix(".f32.partial")
    if temporary.exists():
        raise RuntimeError(f"partial synthetic feature file exists: {temporary}")
    rng = np.random.default_rng(seed)
    bank = np.memmap(temporary, mode="w+", dtype=np.float32, shape=(corpus_size, DIM))
    for start in range(0, corpus_size, chunk_videos):
        stop = min(start + chunk_videos, corpus_size)
        bank[start:stop] = rng.standard_normal((stop - start, DIM), dtype=np.float32)
        if stop % max(chunk_videos * 32, 1) == 0 or stop == corpus_size:
            print(f"synthetic gallery: {stop}/{corpus_size}", flush=True)
    bank.flush()
    del bank
    temporary.replace(features)
    manifest.write_text(json.dumps({
        "num_videos": corpus_size,
        "shape": [corpus_size, 1, DIM],
        "dtype": "float32",
        "seed": seed,
        "query_count": DEFAULT_QUERY_COUNT,
        "query_length": DEFAULT_QUERY_LENGTH,
    }, indent=2) + "\n", encoding="utf-8")
    return directory


def open_video_features(directory: Path) -> np.memmap:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    shape = manifest["shape"]
    if len(shape) != 3 or shape[1:] != [1, DIM]:
        raise ValueError(f"unsupported synthetic gallery shape: {shape}")
    return np.memmap(directory / "video_features.f32", mode="r", dtype=np.float32,
                     shape=(int(shape[0]), DIM))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description="Create a deterministic [V,1,512] synthetic gallery")
    cli.add_argument("--root", type=Path, required=True)
    cli.add_argument("--videos", type=int, required=True)
    cli.add_argument("--seed", type=int, default=9527)
    cli.add_argument("--chunk-videos", type=int, default=8192)
    ns = cli.parse_args()
    print(ensure_corpus(ns.root, ns.videos, seed=ns.seed, chunk_videos=ns.chunk_videos))
