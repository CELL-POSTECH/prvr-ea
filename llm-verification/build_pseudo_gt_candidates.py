#!/usr/bin/env python3
"""Build dense pseudo-GT candidates for PRVR validation queries.

The script intersects top-k text-to-video retrieval results from PRVR model
outputs and a pure CLIP max-frame-similarity ranker. TVR remains the default
configuration. ActivityNet and Charades can be selected with ``--dataset``.

TVR example:
    conda run -n prvr python llm-verification/build_pseudo_gt_candidates.py \
      --ms-sl-rank outputs/tvr_rankings_from_ckpts/ms_sl_top100.jsonl \
      --gmmformer-rank outputs/tvr_rankings_from_ckpts/gmmformer_v2_top100.jsonl \
      --hlformer-rank outputs/tvr_rankings_from_ckpts/hlformer_top100.jsonl \
      --holmes-rank outputs/tvr_rankings_from_ckpts/holmes_top100.jsonl \
      --pure-clip-cache outputs/tvr_rankings_from_ckpts/pure_clip_frame_topP1000_top100.jsonl \
      --output outputs/upstream/pseudo_gt_candidates.tvr.jsonl

ActivityNet example with local raw videos:
    conda run -n prvr python llm-verification/build_pseudo_gt_candidates.py \
      --dataset activitynet \
      --ms-sl-rank outputs/activitynet_rankings_from_ckpts/ms_sl_top100.jsonl \
      --gmmformer-rank outputs/activitynet_rankings_from_ckpts/gmmformer_v2_top100.jsonl \
      --hlformer-rank outputs/activitynet_rankings_from_ckpts/hlformer_top100.jsonl \
      --holmes-rank outputs/activitynet_rankings_from_ckpts/holmes_top100.jsonl \
      --pure-clip-cache outputs/activitynet_rankings_from_ckpts/pure_clip_frame_topP1000_top100.jsonl \
      --frame-source videos \
      --video-root datasets/activitynet/raw_videos \
      --sampled-frame-root datasets/activitynet/verification_frames \
      --output outputs/upstream/pseudo_gt_candidates.activitynet.jsonl

Generic rank inputs:
    conda run -n prvr python llm-verification/build_pseudo_gt_candidates.py \
      --dataset activitynet \
      --model-rank method_a=outputs/ranks/method_a_top100.jsonl \
      --model-rank method_b=outputs/ranks/method_b_top100.jsonl \
      --model-rank method_c=outputs/ranks/method_c_top100.jsonl \
      --model-rank method_d=outputs/ranks/method_d_top100.jsonl \
      --pure-clip-cache outputs/ranks/pure_clip_frame_topP1000_top100.jsonl \
      --output outputs/upstream/pseudo_gt_candidates.activitynet.jsonl
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import shutil
import subprocess
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("PRVR_DATA_ROOT", REPO_ROOT / "datasets"))


def data_path(*parts: str) -> str:
    return str(DATA_ROOT.joinpath(*parts))


DATASET_DEFAULTS: Dict[str, Dict[str, Optional[str]]] = {
    "tvr": {
        "annotation": data_path("tvr", "TextData", "tvr_val_release.jsonl"),
        "val_caption": data_path("tvr", "TextData", "tvrval.caption.txt"),
        "query_feat": data_path("tvr", "TextData", "clip_ViT_B_32_tvr_query_feat.hdf5"),
        "video_feat": data_path("tvr", "FeatureData", "new_clip_vit_32_tvr_vid_features.hdf5"),
        "frame_root": data_path("tvr", "raw_frames", "frames_hq"),
        "sampled_frame_root": data_path("tvr", "verification_frames"),
        "frame_source": "frames",
    },
    "activitynet": {
        "annotation": data_path("activitynet", "TextData", "activitynet_val.jsonl"),
        "val_caption": data_path("activitynet", "TextData", "activitynetval.caption.txt"),
        "query_feat": data_path("activitynet", "TextData", "clip_ViT_B_32_activitynet_query_feat.hdf5"),
        "video_feat": data_path("activitynet", "FeatureData", "new_clip_vit_32_activitynet_vid_features.hdf5"),
        "frame_root": data_path("activitynet", "raw_frames"),
        "sampled_frame_root": data_path("activitynet", "verification_frames"),
        "frame_source": "paths",
    },
    "charades": {
        "annotation": data_path("charades", "TextData", "charades_val.jsonl"),
        "val_caption": data_path("charades", "TextData", "charadesval.caption.txt"),
        "query_feat": data_path("charades", "TextData", "clip_ViT_B_32_charades_query_feat.hdf5"),
        "video_feat": data_path("charades", "FeatureData", "new_clip_vit_32_charades_vid_features.hdf5"),
        "frame_root": data_path("charades", "raw_frames"),
        "sampled_frame_root": data_path("charades", "verification_frames"),
        "frame_source": "paths",
    },
    "msrvtt": {
        "annotation": data_path("msrvtt", "MSRVTT_data.videos.jsonl"),
        "val_caption": data_path("msrvtt", "TextData", "msrvttval.caption.txt"),
        "query_feat": data_path("msrvtt", "TextData", "clip_ViT_B_32_msrvtt_query_feat.hdf5"),
        "video_feat": data_path("msrvtt", "FeatureData", "new_clip_vit_32_msrvtt_vid_features.hdf5"),
        "frame_root": data_path("msrvtt", "raw_frames"),
        "sampled_frame_root": data_path("msrvtt", "verification_frames"),
        "frame_source": "paths",
    },
}

SPLIT_PATH_OVERRIDES: Dict[str, Dict[str, Dict[str, str]]] = {
    "activitynet": {
        "train": {
            "annotation": data_path("activitynet", "TextData", "activitynet_train.jsonl"),
            "val_caption": data_path("activitynet", "TextData", "activitynettrain.caption.txt"),
        },
    },
    "tvr": {
        "train": {
            "annotation": data_path("tvr", "TextData", "tvr_train_release.jsonl"),
            "val_caption": data_path("tvr", "TextData", "tvrtrain.caption.txt"),
        },
    },
    "charades": {
        "train": {
            "annotation": data_path("charades", "TextData", "charades_train.jsonl"),
            "val_caption": data_path("charades", "TextData", "charadestrain.caption.txt"),
        },
    },
    "msrvtt": {
        "train": {
            "annotation": data_path("msrvtt", "MSRVTT_data.videos.jsonl"),
            "val_caption": data_path("msrvtt", "TextData", "msrvtttrain.caption.txt"),
        },
    },
}


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def read_caption_rows(path: str) -> List[Tuple[str, str]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            query_key, desc = line.split(" ", 1)
            rows.append((query_key, desc))
    return rows


def read_video_ids_from_caption(path: str) -> List[str]:
    video_ids = []
    seen = set()
    for query_key, _ in read_caption_rows(path):
        video_id = query_key.split("#", 1)[0]
        if video_id not in seen:
            seen.add(video_id)
            video_ids.append(video_id)
    return video_ids


def read_tvr_caption_key_map(path: str) -> Dict[Tuple[str, str], deque]:
    key_map: Dict[Tuple[str, str], deque] = defaultdict(deque)
    for query_key, desc in read_caption_rows(path):
        video_id = query_key.split("#enc#", 1)[0]
        key_map[(video_id, desc)].append(query_key)
    return key_map


def add_common_aliases(
    aliases: Dict[str, str],
    query_key: str,
    desc_id: str,
    query_index: int,
    video_id: str,
) -> None:
    for alias in {
        query_key,
        desc_id,
        str(query_index),
        f"{video_id}#{desc_id}",
        f"{video_id}#enc#{query_key.rsplit('#enc#', 1)[-1]}",
    }:
        aliases[alias] = query_key


def build_tvr_query_meta(annotation: str, val_caption: str) -> Tuple[List[Dict[str, Any]], Dict[str, str], Dict[str, float]]:
    records = read_jsonl(annotation)
    caption_keys_by_text = read_tvr_caption_key_map(val_caption)

    aliases: Dict[str, str] = {}
    metas: List[Dict[str, Any]] = []
    durations: Dict[str, float] = {}
    for record in records:
        desc_id = str(record["desc_id"])
        video_id = record["vid_name"]
        key_queue = caption_keys_by_text.get((video_id, record["desc"]))
        if not key_queue:
            continue
        query_key = key_queue.popleft()
        duration = float(record["duration"])
        durations[video_id] = duration
        query_index = len(metas)
        meta = {
            "query_key": query_key,
            "desc_id": desc_id,
            "query_index": query_index,
            "query": record["desc"],
            "type": record.get("type"),
            "gt_video_id": video_id,
            "duration": duration,
            "gt_ts": [float(record["ts"][0]), float(record["ts"][1])],
        }
        metas.append(meta)
        add_common_aliases(aliases, query_key, desc_id, query_index, video_id)

    leftovers = [
        (video_id, desc, list(keys))
        for (video_id, desc), keys in caption_keys_by_text.items()
        if keys
    ]
    if leftovers:
        video_id, desc, keys = leftovers[0]
        raise KeyError(
            "Could not match every TVR caption row to the annotation. "
            f"First unmatched vid_name={video_id!r}, query_keys={keys[:3]!r}, desc={desc!r}"
        )
    return metas, aliases, durations


def build_activitynet_query_meta(annotation: str, val_caption: str) -> Tuple[List[Dict[str, Any]], Dict[str, str], Dict[str, float]]:
    if annotation.endswith(".jsonl"):
        data = {str(record["desc_id"]): record for record in read_jsonl(annotation)}
    else:
        data = read_json(annotation)
    caption_rows = read_caption_rows(val_caption)
    aliases: Dict[str, str] = {}
    metas: List[Dict[str, Any]] = []
    durations: Dict[str, float] = {}

    for query_index, (query_key, caption) in enumerate(caption_rows):
        if "#enc#" not in query_key:
            raise ValueError(f"Unexpected ActivityNet query key: {query_key}")
        video_id, enc_idx = query_key.split("#enc#", 1)
        sent_idx = int(enc_idx)
        if annotation.endswith(".jsonl"):
            if query_key not in data:
                raise KeyError(f"ActivityNet caption query key not found in annotation: {query_key}")
            record = data[query_key]
            duration = float(record["duration"])
            ts = record["ts"]
        else:
            if video_id not in data:
                raise KeyError(f"ActivityNet caption video id not found in annotation: {video_id}")
            record = data[video_id]
            duration = float(record["duration"])
            timestamps = record.get("timestamps", [])
            sentences = record.get("sentences", [])
            if sent_idx >= len(timestamps) or sent_idx >= len(sentences):
                raise IndexError(
                    f"ActivityNet query key {query_key} points past annotation lengths: "
                    f"{len(timestamps)} timestamps, {len(sentences)} sentences"
                )
            ts = timestamps[sent_idx]
        durations[video_id] = duration
        desc_id = str(query_index)
        meta = {
            "query_key": query_key,
            "desc_id": desc_id,
            "query_index": query_index,
            "query": normalize_text(caption),
            "type": "v",
            "gt_video_id": video_id,
            "duration": duration,
            "gt_ts": [float(ts[0]), float(ts[1])],
        }
        metas.append(meta)
        add_common_aliases(aliases, query_key, desc_id, query_index, video_id)

    return metas, aliases, durations


def build_charades_query_meta(annotation: str, val_caption: str) -> Tuple[List[Dict[str, Any]], Dict[str, str], Dict[str, float]]:
    records = read_jsonl(annotation)
    caption_rows = read_caption_rows(val_caption)
    if len(records) != len(caption_rows):
        raise ValueError(
            f"Charades validation jsonl/caption length mismatch: {len(records)} vs {len(caption_rows)}"
        )

    captions_by_key = {query_key: desc for query_key, desc in caption_rows}
    aliases: Dict[str, str] = {}
    metas: List[Dict[str, Any]] = []
    durations: Dict[str, float] = {}

    for idx, record in enumerate(records):
        query_key = str(record["desc_id"])
        if query_key not in captions_by_key:
            raise KeyError(f"Charades desc_id not found in caption file: {query_key}")
        video_id = record["vid_name"]
        duration = float(record["duration"])
        durations[video_id] = duration
        meta = {
            "query_key": query_key,
            "desc_id": query_key,
            "query_index": idx,
            "query": normalize_text(captions_by_key[query_key]),
            "type": record.get("type", "v"),
            "gt_video_id": video_id,
            "duration": duration,
            "gt_ts": [float(record["ts"][0]), float(record["ts"][1])],
        }
        metas.append(meta)
        add_common_aliases(aliases, query_key, query_key, idx, video_id)

    return metas, aliases, durations


def build_msrvtt_query_meta(annotation: str, val_caption: str) -> Tuple[List[Dict[str, Any]], Dict[str, str], Dict[str, float]]:
    videos = {str(record["video_id"]): record for record in read_jsonl(annotation)}
    caption_rows = read_caption_rows(val_caption)
    aliases: Dict[str, str] = {}
    metas: List[Dict[str, Any]] = []
    durations: Dict[str, float] = {}

    for query_index, (query_key, caption) in enumerate(caption_rows):
        if "#enc#" not in query_key:
            raise ValueError(f"Unexpected MSRVTT query key: {query_key}")
        video_id, enc_idx = query_key.split("#enc#", 1)
        if video_id not in videos:
            raise KeyError(f"MSRVTT caption video id not found in annotation: {video_id}")
        record = videos[video_id]
        start_time = float(record.get("start time", 0.0))
        end_time = float(record["end time"])
        duration = max(0.0, end_time - start_time)
        durations[video_id] = duration
        desc_id = f"{video_id}#enc#{enc_idx}"
        meta = {
            "query_key": query_key,
            "desc_id": desc_id,
            "query_index": query_index,
            "query": normalize_text(caption),
            "type": "v",
            "gt_video_id": video_id,
            "duration": duration,
            "gt_ts": [0.0, duration],
        }
        metas.append(meta)
        add_common_aliases(aliases, query_key, desc_id, query_index, video_id)

    return metas, aliases, durations


def build_query_meta(dataset: str, annotation: str, val_caption: str) -> Tuple[List[Dict[str, Any]], Dict[str, str], Dict[str, float]]:
    if dataset == "tvr":
        return build_tvr_query_meta(annotation, val_caption)
    if dataset == "activitynet":
        return build_activitynet_query_meta(annotation, val_caption)
    if dataset == "charades":
        return build_charades_query_meta(annotation, val_caption)
    if dataset == "msrvtt":
        return build_msrvtt_query_meta(annotation, val_caption)
    raise ValueError(f"Unsupported dataset: {dataset}")


def as_video_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return None
        return str(int(value)) if float(value).is_integer() else str(value)
    value = str(value).strip()
    return value or None


def normalize_query_id(value: Any, aliases: Dict[str, str]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, (int, np.integer)):
        value = str(int(value))
    elif isinstance(value, (float, np.floating)) and float(value).is_integer():
        value = str(int(value))
    else:
        value = str(value).strip()
    return aliases.get(value)


def extract_query_id(record: Dict[str, Any]) -> Any:
    for key in ("query_id", "qid", "desc_id", "query_key", "caption_id", "text_id", "query"):
        if key in record:
            return record[key]
    return None


def extract_video_id(record: Dict[str, Any]) -> Any:
    for key in ("video_id", "vid", "vid_name", "video", "ctx_id", "movie_id"):
        if key in record:
            return record[key]
    return None


def extract_rank(record: Dict[str, Any], default: int) -> int:
    for key in ("rank", "ranking", "position", "topk_idx"):
        if key in record and record[key] is not None:
            return int(record[key])
    return default


def iter_video_entries(value: Any) -> Iterable[Tuple[str, Optional[float], Optional[int]]]:
    if isinstance(value, dict):
        if "top100" in value:
            value = value["top100"]
        elif "predictions" in value:
            value = value["predictions"]
        elif "ranking" in value:
            value = value["ranking"]
        elif "videos" in value:
            value = value["videos"]
        elif "video_ids" in value:
            value = value["video_ids"]
        else:
            vid = as_video_id(extract_video_id(value))
            if vid is not None:
                score = value.get("score")
                yield vid, float(score) if score is not None else None, None
            return

    if not isinstance(value, (list, tuple, np.ndarray)):
        vid = as_video_id(value)
        if vid is not None:
            yield vid, None, None
        return

    for rank0, item in enumerate(value):
        if isinstance(item, dict):
            vid = as_video_id(extract_video_id(item))
            if vid is None:
                continue
            score = item.get("score")
            rank = extract_rank(item, rank0 + 1)
            yield vid, float(score) if score is not None else None, rank
        elif isinstance(item, (list, tuple, np.ndarray)) and len(item) >= 1:
            vid = as_video_id(item[0])
            score = float(item[1]) if len(item) >= 2 and item[1] is not None else None
            yield vid, score, rank0 + 1
        else:
            vid = as_video_id(item)
            if vid is not None:
                yield vid, None, rank0 + 1


def load_json_rankings(path: str, aliases: Dict[str, str], topk: int) -> Dict[str, Dict[str, Dict[str, Any]]]:
    data = read_json(path)
    result: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    if isinstance(data, dict):
        if "results" in data and isinstance(data["results"], list):
            records = data["results"]
        else:
            for qid, videos in data.items():
                qkey = normalize_query_id(qid, aliases)
                if qkey is None:
                    continue
                for rank0, (vid, score, rank) in enumerate(iter_video_entries(videos)):
                    rank = rank or rank0 + 1
                    if rank <= topk and vid not in result[qkey]:
                        result[qkey][vid] = {"rank": rank, "score": score}
            return result
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError(f"Unsupported JSON ranking structure: {path}")

    for rec in records:
        qkey = normalize_query_id(extract_query_id(rec), aliases)
        if qkey is None:
            continue
        videos_value = None
        for key in ("top100", "predictions", "ranking", "videos", "video_ids"):
            if key in rec:
                videos_value = rec[key]
                break
        if videos_value is None:
            vid = as_video_id(extract_video_id(rec))
            rank = extract_rank(rec, 1)
            if vid is not None and rank <= topk and vid not in result[qkey]:
                score = rec.get("score")
                result[qkey][vid] = {"rank": rank, "score": float(score) if score is not None else None}
            continue
        for rank0, (vid, score, rank) in enumerate(iter_video_entries(videos_value)):
            rank = rank or rank0 + 1
            if rank <= topk and vid not in result[qkey]:
                result[qkey][vid] = {"rank": rank, "score": score}
    return result


def load_jsonl_rankings(path: str, aliases: Dict[str, str], topk: int) -> Dict[str, Dict[str, Dict[str, Any]]]:
    result: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue
            rec = json.loads(line)
            qkey = normalize_query_id(extract_query_id(rec), aliases)
            if qkey is None:
                continue
            videos_value = None
            for key in ("top100", "predictions", "ranking", "videos", "video_ids"):
                if key in rec:
                    videos_value = rec[key]
                    break
            if videos_value is None:
                vid = as_video_id(extract_video_id(rec))
                rank = extract_rank(rec, line_idx + 1)
                if vid is not None and rank <= topk and vid not in result[qkey]:
                    score = rec.get("score")
                    result[qkey][vid] = {"rank": rank, "score": float(score) if score is not None else None}
                continue
            for rank0, (vid, score, rank) in enumerate(iter_video_entries(videos_value)):
                rank = rank or rank0 + 1
                if rank <= topk and vid not in result[qkey]:
                    result[qkey][vid] = {"rank": rank, "score": score}
    return result


def load_table_rankings(path: str, aliases: Dict[str, str], topk: int) -> Dict[str, Dict[str, Dict[str, Any]]]:
    result: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    with open(path, "r", encoding="utf-8", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        reader = csv.DictReader(f, dialect=dialect)
        for row_idx, row in enumerate(reader):
            qkey = normalize_query_id(extract_query_id(row), aliases)
            vid = as_video_id(extract_video_id(row))
            if qkey is None or vid is None:
                continue
            rank = extract_rank(row, row_idx + 1)
            if rank > topk or vid in result[qkey]:
                continue
            score = row.get("score")
            result[qkey][vid] = {"rank": rank, "score": float(score) if score not in (None, "") else None}
    return result


def bytes_to_str_list(values: Sequence[Any]) -> List[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def load_npz_rankings(
    path: str,
    aliases: Dict[str, str],
    video_ids: Sequence[str],
    topk: int,
    lower_is_better: bool,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    data = np.load(path, allow_pickle=True)
    result: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    qids = bytes_to_str_list(data["query_ids"] if "query_ids" in data else data["queries"])
    vids = bytes_to_str_list(data["video_ids"]) if "video_ids" in data else list(video_ids)

    if "ranking_indices" in data:
        ranking = data["ranking_indices"]
        scores = data["scores"] if "scores" in data else None
        for qi, qid in enumerate(qids):
            qkey = normalize_query_id(qid, aliases)
            if qkey is None:
                continue
            for rank0, vi in enumerate(ranking[qi, :topk]):
                vi = int(vi)
                score = float(scores[qi, vi]) if scores is not None and scores.ndim == 2 else None
                result[qkey][vids[vi]] = {"rank": rank0 + 1, "score": score}
        return result

    if "scores" not in data:
        raise ValueError(f"NPZ ranking needs either ranking_indices or scores: {path}")
    scores = data["scores"]
    if scores.shape != (len(qids), len(vids)):
        raise ValueError(f"Unexpected score matrix shape in {path}: {scores.shape}")
    for qi, qid in enumerate(qids):
        qkey = normalize_query_id(qid, aliases)
        if qkey is None:
            continue
        row = scores[qi]
        idxs = np.argsort(row)[:topk] if lower_is_better else np.argsort(-row)[:topk]
        for rank0, vi in enumerate(idxs):
            result[qkey][vids[int(vi)]] = {"rank": rank0 + 1, "score": float(row[int(vi)])}
    return result


def load_model_rankings(
    path: str,
    aliases: Dict[str, str],
    video_ids: Sequence[str],
    topk: int,
    lower_is_better: bool,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    suffix = Path(path).suffix.lower()
    if suffix == ".json":
        return load_json_rankings(path, aliases, topk)
    if suffix == ".jsonl":
        return load_jsonl_rankings(path, aliases, topk)
    if suffix in {".csv", ".tsv"}:
        return load_table_rankings(path, aliases, topk)
    if suffix == ".npz":
        return load_npz_rankings(path, aliases, video_ids, topk, lower_is_better)
    raise ValueError(f"Unsupported ranking file extension: {path}")


def load_pure_clip_cache(path: str, aliases: Dict[str, str], topk: int) -> Dict[str, Dict[str, Dict[str, Any]]]:
    result: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            qkey = normalize_query_id(extract_query_id(rec), aliases)
            if qkey is None:
                continue
            for rank0, item in enumerate(rec.get("top100", [])):
                if rank0 >= topk:
                    break
                vid = as_video_id(extract_video_id(item))
                if vid is None:
                    continue
                rank = extract_rank(item, rank0 + 1)
                if rank > topk:
                    continue
                result[qkey][vid] = {
                    "rank": rank,
                    "score": item.get("score"),
                    "clip_feature_index": int(item["clip_feature_index"]),
                }
    return result


def normalize_np(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)


def load_query_feature(handle: h5py.File, key: str) -> np.ndarray:
    feat = np.asarray(handle[key][()], dtype="float32")
    feat = np.squeeze(feat)
    if feat.ndim != 1:
        raise ValueError(f"Expected 1D query feature for {key}, got shape {feat.shape}")
    return feat


def load_video_feature(handle: h5py.File, video_id: str) -> np.ndarray:
    feat = np.asarray(handle[video_id][()], dtype="float32")
    feat = np.squeeze(feat)
    if feat.ndim == 1:
        feat = feat[None, :]
    if feat.ndim != 2:
        raise ValueError(f"Expected 2D video feature for {video_id}, got shape {feat.shape}")
    return feat


def compute_pure_clip_topk(
    query_keys: Sequence[str],
    video_ids: Sequence[str],
    query_feat_path: str,
    video_feat_path: str,
    topk: int,
    top_frames: int,
    query_batch_size: int,
    video_batch_size: int,
    device: str,
    cache_path: Optional[str] = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    result: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    torch_device = torch.device(device)
    cache_f = None
    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        cache_f = open(cache_path, "w", encoding="utf-8")

    try:
        with h5py.File(query_feat_path, "r") as qf, h5py.File(video_feat_path, "r") as vf:
            q_ranges = list(range(0, len(query_keys), query_batch_size))
            for q_batch_idx, q_start in enumerate(tqdm(q_ranges, desc="pure CLIP query batches")):
                q_batch_keys = list(query_keys[q_start : q_start + query_batch_size])
                print(
                    f"[pure_clip] query batch {q_batch_idx + 1}/{len(q_ranges)} "
                    f"({len(q_batch_keys)} queries), scanning {len(video_ids)} videos",
                    flush=True,
                )
                q_np = np.stack([load_query_feature(qf, k) for k in q_batch_keys]).astype("float32")
                q_np = normalize_np(q_np)
                q = torch.from_numpy(q_np).to(torch_device)

                best_frame_scores = None
                best_frame_video_indices = None
                best_frame_indices = None

                v_ranges = range(0, len(video_ids), video_batch_size)
                for v_start in tqdm(
                    v_ranges,
                    desc=f"pure CLIP videos qbatch {q_batch_idx + 1}/{len(q_ranges)}",
                    leave=False,
                ):
                    batch_vids = list(video_ids[v_start : v_start + video_batch_size])
                    feats = [normalize_np(load_video_feature(vf, vid)) for vid in batch_vids]
                    lengths = [feat.shape[0] for feat in feats]
                    max_len = max(lengths)
                    padded = np.zeros((len(feats), max_len, q_np.shape[1]), dtype="float32")
                    mask = np.zeros((len(feats), max_len), dtype=bool)
                    for i, feat in enumerate(feats):
                        if feat.shape[1] != q_np.shape[1]:
                            raise ValueError(
                                f"Feature dim mismatch for {batch_vids[i]}: "
                                f"video dim {feat.shape[1]} vs query dim {q_np.shape[1]}"
                            )
                        padded[i, : feat.shape[0]] = feat
                        mask[i, : feat.shape[0]] = True

                    v = torch.from_numpy(padded).to(torch_device)
                    valid_mask = torch.from_numpy(mask).to(torch_device)
                    sim = torch.einsum("qd,vtd->qvt", q, v)
                    sim = sim.masked_fill(~valid_mask.unsqueeze(0), -1e9)
                    flat_sim = sim.reshape(sim.shape[0], -1)
                    local_k = min(top_frames, flat_sim.shape[1])
                    vals, flat_pos = torch.topk(flat_sim, k=local_k, dim=1)
                    local_vid_pos = flat_pos // sim.shape[2]
                    local_frame_pos = flat_pos % sim.shape[2]
                    global_vid_indices = local_vid_pos + v_start

                    if best_frame_scores is None:
                        best_frame_scores = vals
                        best_frame_video_indices = global_vid_indices
                        best_frame_indices = local_frame_pos
                    else:
                        merged_scores = torch.cat([best_frame_scores, vals], dim=1)
                        merged_video_indices = torch.cat([best_frame_video_indices, global_vid_indices], dim=1)
                        merged_frame_indices = torch.cat([best_frame_indices, local_frame_pos], dim=1)
                        keep_k = min(top_frames, merged_scores.shape[1])
                        best_frame_scores, keep_pos = torch.topk(merged_scores, k=keep_k, dim=1)
                        best_frame_video_indices = torch.gather(merged_video_indices, 1, keep_pos)
                        best_frame_indices = torch.gather(merged_frame_indices, 1, keep_pos)

                assert (
                    best_frame_scores is not None
                    and best_frame_video_indices is not None
                    and best_frame_indices is not None
                )
                top_scores = best_frame_scores.detach().cpu().numpy()
                top_video_indices = best_frame_video_indices.detach().cpu().numpy()
                top_frame_indices = best_frame_indices.detach().cpu().numpy()
                for local_qi, qkey in enumerate(q_batch_keys):
                    seen_videos = set()
                    video_rank = 0
                    ranking = []
                    for frame_rank0, video_index in enumerate(top_video_indices[local_qi]):
                        video_index = int(video_index)
                        vid = video_ids[video_index]
                        if vid in seen_videos:
                            continue
                        seen_videos.add(vid)
                        video_rank += 1
                        frame_index = int(top_frame_indices[local_qi, frame_rank0])
                        item = {
                            "video_id": vid,
                            "rank": video_rank,
                            "score": float(top_scores[local_qi, frame_rank0]),
                            "clip_feature_index": frame_index,
                            "frame_rank": int(frame_rank0 + 1),
                        }
                        ranking.append(item)
                        if not cache_f:
                            result[qkey][vid] = {
                                "rank": item["rank"],
                                "score": item["score"],
                                "clip_feature_index": item["clip_feature_index"],
                                "frame_rank": item["frame_rank"],
                            }
                        if video_rank >= topk:
                            break
                    if cache_f:
                        cache_f.write(json.dumps({"query_id": qkey, "top100": ranking}, ensure_ascii=False) + "\n")
                if cache_f:
                    cache_f.flush()
    finally:
        if cache_f:
            cache_f.close()
    return result


def find_frame_dir(frame_root: str, video_id: str, cache: Dict[str, Optional[str]]) -> Optional[str]:
    if video_id in cache:
        return cache[video_id]
    show = video_id.split("_s", 1)[0]
    candidates = [
        os.path.join(frame_root, f"{show}_frames", video_id),
        os.path.join(frame_root, video_id),
    ]
    candidates.extend(glob.glob(os.path.join(frame_root, "*_frames", video_id)))
    for candidate in candidates:
        if os.path.isdir(candidate):
            cache[video_id] = candidate
            return candidate
    cache[video_id] = None
    return None


def get_frame_files(frame_root: str, video_id: str, dir_cache: Dict[str, Optional[str]]) -> List[str]:
    frame_dir = find_frame_dir(frame_root, video_id, dir_cache)
    if frame_dir is None:
        return []
    files = glob.glob(os.path.join(frame_dir, "*.jpg"))
    files.sort(key=lambda p: int(Path(p).stem) if Path(p).stem.isdigit() else Path(p).stem)
    return files


def find_video_file(video_root: str, video_id: str, cache: Dict[str, Optional[str]]) -> Optional[str]:
    if video_id in cache:
        return cache[video_id]
    candidates = [
        os.path.join(video_root, video_id),
        os.path.join(video_root, f"{video_id}.mp4"),
        os.path.join(video_root, f"{video_id}.mkv"),
        os.path.join(video_root, video_id[2:] if video_id.startswith("v_") else video_id),
        os.path.join(video_root, f"{video_id[2:]}.mp4" if video_id.startswith("v_") else f"{video_id}.mp4"),
    ]
    candidates.extend(glob.glob(os.path.join(video_root, "*", f"{video_id}.mp4")))
    for candidate in candidates:
        if os.path.isfile(candidate):
            cache[video_id] = candidate
            return candidate
    cache[video_id] = None
    return None


def slugify(value: str, max_len: int = 160) -> str:
    value = re.sub(r"[^A-Za-z0-9_.#=-]+", "_", value.strip())
    value = value.strip("_") or "item"
    return value[:max_len]


def clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def clamp_float(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def sample_uniform_indices(start_idx: int, end_idx: int, num_frames: int, max_idx: int) -> List[int]:
    start_idx = clamp(start_idx, 1, max_idx)
    end_idx = clamp(end_idx, 1, max_idx)
    if end_idx < start_idx:
        start_idx, end_idx = end_idx, start_idx
    if num_frames <= 1:
        return [start_idx]
    values = np.linspace(start_idx, end_idx, num_frames)
    return [clamp(int(round(v)), 1, max_idx) for v in values]


def sample_centered_indices(center_idx: int, num_frames: int, max_idx: int) -> List[int]:
    half_left = (num_frames - 1) // 2
    start = center_idx - half_left
    end = start + num_frames - 1
    if start < 1:
        end += 1 - start
        start = 1
    if end > max_idx:
        start -= end - max_idx
        end = max_idx
    start = max(1, start)
    return [clamp(i, 1, max_idx) for i in range(start, end + 1)]


def sample_uniform_times(start_sec: float, end_sec: float, num_frames: int, duration: float) -> List[float]:
    start_sec = clamp_float(start_sec, 0.0, max(0.0, duration))
    end_sec = clamp_float(end_sec, 0.0, max(0.0, duration))
    if end_sec < start_sec:
        start_sec, end_sec = end_sec, start_sec
    if num_frames <= 1:
        return [(start_sec + end_sec) / 2.0]
    return [float(x) for x in np.linspace(start_sec, end_sec, num_frames)]


def sample_centered_times(center_sec: float, num_frames: int, duration: float, window_sec: float) -> List[float]:
    if num_frames <= 1:
        return [clamp_float(center_sec, 0.0, duration)]
    half_window = max(window_sec, 0.0) / 2.0
    start = center_sec - half_window
    end = center_sec + half_window
    if start < 0.0:
        end -= start
        start = 0.0
    if end > duration:
        start -= end - duration
        end = duration
    start = max(0.0, start)
    return [clamp_float(float(x), 0.0, duration) for x in np.linspace(start, end, num_frames)]


def resolve_pseudo_window_sec(meta: Dict[str, Any], args: argparse.Namespace) -> float:
    if args.pseudo_window_mode == "fixed":
        return max(0.0, float(args.pseudo_window_sec))
    gt_start, gt_end = meta["gt_ts"]
    gt_len = max(0.0, float(gt_end) - float(gt_start))
    return clamp_float(gt_len, float(args.pseudo_window_min_sec), float(args.pseudo_window_max_sec))


def time_to_frame_idx(sec: float, duration: float, num_raw_frames: int) -> int:
    if num_raw_frames <= 1 or duration <= 0:
        return 1
    ratio = clamp_float(sec / duration, 0.0, 1.0)
    return int(round(ratio * (num_raw_frames - 1))) + 1


def clip_index_to_frame_idx(clip_index: int, clip_len: int, num_raw_frames: int) -> int:
    if num_raw_frames <= 1 or clip_len <= 1:
        return 1
    ratio = clamp_float(clip_index / float(clip_len - 1), 0.0, 1.0)
    return int(round(ratio * (num_raw_frames - 1))) + 1


def clip_index_to_time(clip_index: int, clip_len: int, duration: float) -> float:
    if duration <= 0 or clip_len <= 1:
        return 0.0
    ratio = clamp_float(clip_index / float(clip_len - 1), 0.0, 1.0)
    return ratio * duration


def indices_to_paths(files: Sequence[str], indices: Sequence[int]) -> List[Optional[str]]:
    out: List[Optional[str]] = []
    for idx in indices:
        if 1 <= idx <= len(files):
            out.append(files[idx - 1])
        else:
            out.append(None)
    return out


def copy_indexed_frames_to_root(
    files: Sequence[str],
    indices: Sequence[int],
    sampled_frame_root: Optional[str],
    video_id: str,
    overwrite: bool,
) -> List[Optional[str]]:
    source_paths = indices_to_paths(files, indices)
    if sampled_frame_root is None:
        return source_paths

    out_dir = os.path.join(sampled_frame_root, slugify(video_id))
    out_paths: List[Optional[str]] = []
    for source_path in source_paths:
        if source_path is None:
            out_paths.append(None)
            continue
        out_path = os.path.join(out_dir, Path(source_path).name)
        if overwrite or not os.path.exists(out_path):
            os.makedirs(out_dir, exist_ok=True)
            shutil.copy2(source_path, out_path)
        out_paths.append(out_path)
    return out_paths


def extract_frame(video_path: str, time_sec: float, output_path: str, overwrite: bool = False) -> None:
    if os.path.exists(output_path) and not overwrite:
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_path = f"{output_path}.tmp.jpg"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{max(0.0, time_sec):.3f}",
        "-i",
        video_path,
        "-frames:v",
        "1",
        "-q:v",
        "2",
        tmp_path,
    ]
    subprocess.run(cmd, check=True)
    os.replace(tmp_path, output_path)


def time_to_sampled_frame_name(time_sec: float) -> str:
    return f"{int(round(max(0.0, float(time_sec)) * 1000.0)):010d}.jpg"


def materialize_video_frames(
    video_root: str,
    sampled_frame_root: str,
    query_key: str,
    video_id: str,
    role: str,
    times: Sequence[float],
    video_cache: Dict[str, Optional[str]],
    paths_only: bool,
    overwrite: bool,
    layout: str,
) -> List[str]:
    if layout == "video_time":
        sample_dir = os.path.join(sampled_frame_root, slugify(video_id))
        paths = [os.path.join(sample_dir, time_to_sampled_frame_name(time_sec)) for time_sec in times]
    else:
        sample_dir = os.path.join(
            sampled_frame_root,
            role,
            slugify(query_key),
            slugify(video_id),
        )
        paths = [os.path.join(sample_dir, f"{idx:06d}.jpg") for idx in range(1, len(times) + 1)]
    if paths_only:
        return paths
    if not video_root:
        raise ValueError("--video-root is required when --frame-source=videos")
    video_path = find_video_file(video_root, video_id, video_cache)
    if video_path is None:
        raise FileNotFoundError(f"Could not find raw video for {video_id} under {video_root}")
    for time_sec, path in zip(times, paths):
        extract_frame(video_path, time_sec, path, overwrite=overwrite)
    return paths


def get_video_feature_lengths(video_feat_path: str, video_ids: Sequence[str]) -> Dict[str, int]:
    with h5py.File(video_feat_path, "r") as f:
        return {vid: int(f[vid].shape[0]) for vid in video_ids}


def write_jsonl(records: Sequence[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def count_jsonl_rows(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def parse_name_path(value: str) -> Tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"Expected NAME=PATH, got: {value}")
    name, path = value.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise ValueError(f"Expected non-empty NAME=PATH, got: {value}")
    return name, path


def build_model_rank_paths(args: argparse.Namespace) -> Dict[str, str]:
    if args.model_rank:
        paths: Dict[str, str] = {}
        for item in args.model_rank:
            name, path = parse_name_path(item)
            if name in paths:
                raise ValueError(f"Duplicate --model-rank name: {name}")
            paths[name] = path
        return paths

    legacy = {
        "ms-sl": args.ms_sl_rank,
        "GMMFormerv2": args.gmmformer_rank,
        "HLFormer": args.hlformer_rank,
        "Holmes": args.holmes_rank,
    }
    missing = [name for name, path in legacy.items() if not path]
    if missing:
        raise ValueError(
            "Provide either repeated --model-rank NAME=PATH or all legacy rank "
            f"options. Missing legacy ranks: {', '.join(missing)}"
        )
    return legacy


def rank_record_to_videos(
    rec: Dict[str, Any],
    aliases: Dict[str, str],
    topk: int,
    pure_clip: bool = False,
) -> Tuple[Optional[str], Dict[str, Dict[str, Any]]]:
    qkey = normalize_query_id(extract_query_id(rec), aliases)
    if qkey is None:
        return None, {}
    videos: Dict[str, Dict[str, Any]] = {}
    videos_value = None
    for key in ("top100", "predictions", "ranking", "videos", "video_ids"):
        if key in rec:
            videos_value = rec[key]
            break
    if videos_value is None:
        vid = as_video_id(extract_video_id(rec))
        rank = extract_rank(rec, 1)
        if vid is not None and rank <= topk:
            score = rec.get("score")
            videos[vid] = {"rank": rank, "score": float(score) if score is not None else None}
            if pure_clip:
                videos[vid]["clip_feature_index"] = int(rec["clip_feature_index"])
        return qkey, videos

    for rank0, item in enumerate(videos_value):
        if rank0 >= topk:
            break
        if isinstance(item, dict):
            vid = as_video_id(extract_video_id(item))
            if vid is None:
                continue
            score = item.get("score")
            rank = extract_rank(item, rank0 + 1)
            if rank > topk or vid in videos:
                continue
            videos[vid] = {"rank": rank, "score": float(score) if score is not None else None}
            if pure_clip:
                videos[vid]["clip_feature_index"] = int(item["clip_feature_index"])
        else:
            for vid, score, rank in iter_video_entries([item]):
                rank = rank or rank0 + 1
                if rank <= topk and vid not in videos:
                    videos[vid] = {"rank": rank, "score": score}
    return qkey, videos


def read_rank_jsonl_line(
    handle: Any,
    aliases: Dict[str, str],
    topk: int,
    pure_clip: bool = False,
) -> Tuple[Optional[str], Dict[str, Dict[str, Any]]]:
    while True:
        line = handle.readline()
        if not line:
            return None, {}
        if line.strip():
            return rank_record_to_videos(json.loads(line), aliases, topk, pure_clip=pure_clip)


def build_candidate_records_for_query(
    qkey: str,
    all_models: Dict[str, Dict[str, Dict[str, Any]]],
    meta_by_query: Dict[str, Dict[str, Any]],
    duration_by_video: Dict[str, float],
    video_feature_lengths: Dict[str, int],
    args: argparse.Namespace,
    frame_dir_cache: Dict[str, Optional[str]],
    frame_files_cache: Dict[str, List[str]],
    video_file_cache: Dict[str, Optional[str]],
) -> Tuple[List[Dict[str, Any]], int]:
    meta = meta_by_query[qkey]
    gt_video_id = meta["gt_video_id"]
    sets = []
    for videos in all_models.values():
        if not videos:
            return [], 0
        sets.append(set(videos.keys()))

    records: List[Dict[str, Any]] = []
    skipped_missing_frames = 0
    agreed_video_ids = sorted(set.intersection(*sets))
    pure_clip = all_models["pure_clip"]
    for pseudo_video_id in agreed_video_ids:
        if pseudo_video_id == gt_video_id:
            continue

        pure_info = pure_clip[pseudo_video_id]
        clip_index = int(pure_info["clip_feature_index"])
        pseudo_clip_len = video_feature_lengths[pseudo_video_id]

        record: Dict[str, Any] = {
            "dataset": args.dataset,
            "query_key": qkey,
            "desc_id": meta["desc_id"],
            "query": meta["query"],
            "type": meta.get("type"),
            "original_gt_video_id": gt_video_id,
            "pseudo_video_id": pseudo_video_id,
            "gt_ts": meta["gt_ts"],
            "gt_duration": meta["duration"],
            "pseudo_clip_feature_index": clip_index,
            "pseudo_clip_feature_len": pseudo_clip_len,
        }
        if args.dataset == "msrvtt":
            pseudo_duration = duration_by_video.get(pseudo_video_id)
            if pseudo_duration is None:
                raise KeyError(f"Missing duration for candidate video: {pseudo_video_id}")
            record.update(
                {
                    "pseudo_ts": [0.0, pseudo_duration],
                    "pseudo_duration": pseudo_duration,
                }
            )

        if args.frame_source == "frames":
            if gt_video_id not in frame_files_cache:
                frame_files_cache[gt_video_id] = get_frame_files(args.frame_root, gt_video_id, frame_dir_cache)
            if pseudo_video_id not in frame_files_cache:
                frame_files_cache[pseudo_video_id] = get_frame_files(args.frame_root, pseudo_video_id, frame_dir_cache)
            gt_files = frame_files_cache[gt_video_id]
            pseudo_files = frame_files_cache[pseudo_video_id]
            if not gt_files or not pseudo_files:
                skipped_missing_frames += 1

            gt_start_idx = time_to_frame_idx(meta["gt_ts"][0], meta["duration"], max(1, len(gt_files)))
            gt_end_idx = time_to_frame_idx(meta["gt_ts"][1], meta["duration"], max(1, len(gt_files)))
            gt_frame_indices = sample_uniform_indices(
                gt_start_idx, gt_end_idx, args.num_frames, max(1, len(gt_files))
            )
            if args.dataset == "msrvtt":
                pseudo_frame_indices = sample_uniform_indices(
                    1, max(1, len(pseudo_files)), args.num_frames, max(1, len(pseudo_files))
                )
                record.update(
                    {
                        "gt_frame_indices": gt_frame_indices,
                        "gt_frame_paths": copy_indexed_frames_to_root(
                            gt_files,
                            gt_frame_indices,
                            args.sampled_frame_root,
                            gt_video_id,
                            args.overwrite_sampled_frames,
                        ),
                        "pseudo_frame_indices": pseudo_frame_indices,
                        "pseudo_frame_paths": copy_indexed_frames_to_root(
                            pseudo_files,
                            pseudo_frame_indices,
                            args.sampled_frame_root,
                            pseudo_video_id,
                            args.overwrite_sampled_frames,
                        ),
                    }
                )
            else:
                pseudo_center_idx = clip_index_to_frame_idx(
                    clip_index, pseudo_clip_len, max(1, len(pseudo_files))
                )
                pseudo_frame_indices = sample_centered_indices(
                    pseudo_center_idx, args.num_frames, max(1, len(pseudo_files))
                )
                record.update(
                    {
                        "gt_frame_indices": gt_frame_indices,
                        "gt_frame_paths": copy_indexed_frames_to_root(
                            gt_files,
                            gt_frame_indices,
                            args.sampled_frame_root,
                            gt_video_id,
                            args.overwrite_sampled_frames,
                        ),
                        "pseudo_center_frame_index": pseudo_center_idx,
                        "pseudo_frame_indices": pseudo_frame_indices,
                        "pseudo_frame_paths": copy_indexed_frames_to_root(
                            pseudo_files,
                            pseudo_frame_indices,
                            args.sampled_frame_root,
                            pseudo_video_id,
                            args.overwrite_sampled_frames,
                        ),
                    }
                )
        else:
            if args.sampled_frame_root is None:
                raise ValueError("--sampled-frame-root is required for --frame-source videos/paths")
            paths_only = args.frame_source == "paths"
            pseudo_duration = duration_by_video.get(pseudo_video_id)
            if pseudo_duration is None:
                raise KeyError(f"Missing duration for candidate video: {pseudo_video_id}")
            gt_times = sample_uniform_times(
                meta["gt_ts"][0], meta["gt_ts"][1], args.num_frames, meta["duration"]
            )
            pseudo_center_time = clip_index_to_time(clip_index, pseudo_clip_len, pseudo_duration)
            pseudo_window_sec = resolve_pseudo_window_sec(meta, args)
            pseudo_times = sample_centered_times(
                pseudo_center_time, args.num_frames, pseudo_duration, pseudo_window_sec
            )
            gt_paths = materialize_video_frames(
                args.video_root or "",
                args.sampled_frame_root,
                qkey,
                gt_video_id,
                "gt",
                gt_times,
                video_file_cache,
                paths_only,
                args.overwrite_sampled_frames,
                args.sampled_frame_layout,
            )
            pseudo_paths = materialize_video_frames(
                args.video_root or "",
                args.sampled_frame_root,
                qkey,
                pseudo_video_id,
                "pseudo",
                pseudo_times,
                video_file_cache,
                paths_only,
                args.overwrite_sampled_frames,
                args.sampled_frame_layout,
            )
            record.update(
                {
                    "gt_frame_times": gt_times,
                    "gt_frame_paths": gt_paths,
                    "pseudo_center_time": pseudo_center_time,
                    "pseudo_window_mode": args.pseudo_window_mode,
                    "pseudo_window_sec": pseudo_window_sec,
                    "pseudo_frame_times": pseudo_times,
                    "pseudo_frame_paths": pseudo_paths,
                }
            )

        model_info = {}
        for name, rankings in all_models.items():
            info = rankings[pseudo_video_id]
            model_info[name] = {
                "rank": int(info["rank"]),
                "score": info.get("score"),
            }
            if name == "pure_clip":
                model_info[name]["clip_feature_index"] = clip_index
        record["model_agreement"] = model_info
        records.append(record)

    return records, skipped_missing_frames


def apply_dataset_defaults(args: argparse.Namespace) -> None:
    defaults = DATASET_DEFAULTS[args.dataset]
    split_defaults = SPLIT_PATH_OVERRIDES.get(args.dataset, {}).get(args.split, {})
    if args.annotation is None:
        args.annotation = split_defaults.get("annotation", defaults["annotation"])
    if args.val_jsonl is None:
        args.val_jsonl = args.annotation
    if args.val_caption is None:
        args.val_caption = split_defaults.get("val_caption", defaults["val_caption"])
    if args.query_feat is None:
        args.query_feat = defaults["query_feat"]
    if args.video_feat is None:
        args.video_feat = defaults["video_feat"]
    if args.frame_root is None:
        args.frame_root = defaults["frame_root"]
    if args.sampled_frame_root is None:
        args.sampled_frame_root = defaults["sampled_frame_root"]
    if args.frame_source is None:
        args.frame_source = defaults["frame_source"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASET_DEFAULTS), default="tvr")
    parser.add_argument("--split", choices=["val", "train"], default="val")
    parser.add_argument("--annotation", default=None, help="Dataset validation annotation. ActivityNet supports jsonl or legacy json.")
    parser.add_argument("--val-jsonl", default=None, help="Backward-compatible alias for --annotation on TVR.")
    parser.add_argument("--val-caption", default=None)
    parser.add_argument("--query-feat", default=None)
    parser.add_argument("--video-feat", default=None)
    parser.add_argument("--frame-root", default=None, help="Root containing pre-extracted frame directories.")
    parser.add_argument("--video-root", default=None, help="Root containing raw videos, used with --frame-source videos.")
    parser.add_argument("--sampled-frame-root", default=None, help="Output root for frames sampled from raw videos.")
    parser.add_argument(
        "--sampled-frame-layout",
        choices=["video_time", "query_role"],
        default="video_time",
        help="video_time: <root>/<video_id>/<milliseconds>.jpg; query_role: <root>/<role>/<query>/<video_id>/000001.jpg.",
    )
    parser.add_argument(
        "--frame-source",
        choices=["frames", "videos", "paths"],
        default=None,
        help="frames: use existing frame dirs; videos: extract frames from raw videos; paths: write planned paths only.",
    )
    parser.add_argument("--overwrite-sampled-frames", action="store_true")
    parser.add_argument(
        "--model-rank",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="PRVR ranking JSONL. Pass once per PRVR method.",
    )
    parser.add_argument("--ms-sl-rank", default=None, help="Legacy rank input.")
    parser.add_argument("--gmmformer-rank", default=None, help="Legacy rank input.")
    parser.add_argument("--hlformer-rank", default=None, help="Legacy rank input.")
    parser.add_argument("--holmes-rank", default=None, help="Legacy rank input.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument(
        "--pure-clip-top-frames",
        type=int,
        default=1000,
        help="For each query, take this many best val-set frame embeddings, then dedupe videos to topk.",
    )
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument(
        "--pseudo-window-sec",
        type=float,
        default=4.0,
        help="Fixed temporal window around pure-CLIP center for raw-video sampling.",
    )
    parser.add_argument(
        "--pseudo-window-mode",
        choices=["fixed", "gt_clamped"],
        default="fixed",
        help="fixed: use --pseudo-window-sec; gt_clamped: clamp GT segment length and use it as pseudo window.",
    )
    parser.add_argument("--pseudo-window-min-sec", type=float, default=4.0)
    parser.add_argument("--pseudo-window-max-sec", type=float, default=20.0)
    parser.add_argument("--query-batch-size", type=int, default=128)
    parser.add_argument("--video-batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--ranking-lower-is-better",
        action="store_true",
        help="Use ascending order for npz score matrices from model ranking files.",
    )
    parser.add_argument(
        "--pure-clip-cache",
        default=None,
        help="Optional JSONL cache path for pure CLIP top-k results.",
    )
    args = parser.parse_args()
    if args.val_jsonl is not None and args.annotation is None:
        args.annotation = args.val_jsonl
    apply_dataset_defaults(args)
    if args.pseudo_window_min_sec > args.pseudo_window_max_sec:
        raise ValueError("--pseudo-window-min-sec must be <= --pseudo-window-max-sec")
    return args


def main() -> None:
    args = parse_args()
    metas, query_aliases, duration_by_video = build_query_meta(args.dataset, args.annotation, args.val_caption)
    meta_by_query = {m["query_key"]: m for m in metas}
    query_keys = [m["query_key"] for m in metas if args.dataset != "tvr" or m.get("type") == "v"]
    video_ids = read_video_ids_from_caption(args.val_caption)

    model_paths = build_model_rank_paths(args)
    if args.pure_clip_cache and os.path.exists(args.pure_clip_cache):
        cache_rows = count_jsonl_rows(args.pure_clip_cache)
        if cache_rows == len(query_keys):
            print(f"Using pure CLIP cache: {args.pure_clip_cache}", flush=True)
            pure_clip_ready = True
        else:
            print(
                f"Discarding incomplete pure CLIP cache: {args.pure_clip_cache} "
                f"({cache_rows} rows, expected {len(query_keys)})",
                flush=True,
            )
            os.remove(args.pure_clip_cache)
            pure_clip_ready = False
    else:
        pure_clip_ready = False

    if not pure_clip_ready:
        print("Computing pure CLIP top-k cache", flush=True)
        compute_pure_clip_topk(
            query_keys,
            video_ids,
            args.query_feat,
            args.video_feat,
            args.topk,
            args.pure_clip_top_frames,
            args.query_batch_size,
            args.video_batch_size,
            args.device,
            args.pure_clip_cache,
        )
    if not args.pure_clip_cache or not os.path.exists(args.pure_clip_cache):
        raise FileNotFoundError("A pure CLIP JSONL cache is required for streaming candidate construction.")

    video_feature_lengths = get_video_feature_lengths(args.video_feat, video_ids)
    frame_dir_cache: Dict[str, Optional[str]] = {}
    frame_files_cache: Dict[str, List[str]] = {}
    video_file_cache: Dict[str, Optional[str]] = {}

    skipped_missing_frames = 0
    num_candidates = 0
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    loaded_rankings = {
        name: load_model_rankings(path, query_aliases, video_ids, args.topk, args.ranking_lower_is_better)
        for name, path in model_paths.items()
    }
    loaded_rankings["pure_clip"] = load_pure_clip_cache(args.pure_clip_cache, query_aliases, args.topk)

    with open(args.output, "w", encoding="utf-8") as out_f:
        for expected_qkey in tqdm(query_keys, desc=f"building {args.dataset} candidates"):
            missing = [name for name, ranks in loaded_rankings.items() if expected_qkey not in ranks]
            if missing:
                raise KeyError(f"Missing rankings for {expected_qkey}: {', '.join(missing)}")
            all_models = {name: ranks[expected_qkey] for name, ranks in loaded_rankings.items()}
            records, skipped = build_candidate_records_for_query(
                expected_qkey,
                all_models,
                meta_by_query,
                duration_by_video,
                video_feature_lengths,
                args,
                frame_dir_cache,
                frame_files_cache,
                video_file_cache,
            )
            skipped_missing_frames += skipped
            for record in records:
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            num_candidates += len(records)

    summary_path = str(Path(args.output).with_suffix(".summary.json"))
    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "output": args.output,
        "num_queries": len(query_keys),
        "num_candidates": num_candidates,
        "skipped_candidates_with_missing_frame_dirs": skipped_missing_frames,
        "topk": args.topk,
        "num_frames": args.num_frames,
        "frame_source": args.frame_source,
        "annotation": args.annotation,
        "val_caption": args.val_caption,
        "query_feat": args.query_feat,
        "video_feat": args.video_feat,
        "frame_root": args.frame_root,
        "video_root": args.video_root,
        "sampled_frame_root": args.sampled_frame_root,
        "model_rank_files": model_paths,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
