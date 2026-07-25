#!/usr/bin/env python3
"""Export full-test GMMFormer-v2 exact/clip/frame ranking diagnostics.

``exact`` is the original video-level weighted score.  ``clip`` and ``frame``
columns are intentionally *raw representation* ranks: a video may therefore
occur repeatedly.  The accompanying fields make it possible to reconstruct
the first-occurrence (max-sim video) ranking without re-running the model.
"""
import argparse
import csv
import gzip
import os
import statistics
import sys
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=("act", "tvr", "cha", "msrvtt"), required=True)
parser.add_argument("--gpu", required=True, help="Physical CUDA GPU index")
parser.add_argument("--top-l", type=int, default=500,
                    help="Raw repr / exact video rows written for each query (default: 500)")
parser.add_argument("--query-batch-size", type=int, default=16)
parser.add_argument("--checkpoint", default="")
parser.add_argument("--output-dir", default="")
args = parser.parse_args()

if args.top_l < 10:
    raise ValueError("--top-l must be at least 10 to report Recall@10 depth")
if args.query_batch_size < 1:
    raise ValueError("--query-batch-size must be positive")

# Must be set before importing torch/project modules.
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

PROJECT_ROOT = Path(os.environ.get("PRVR_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
ALL_PRVR_ROOT = Path(os.environ.get("PRVR_ROOT", PROJECT_ROOT / "all_prvr"))
PRVR_ROOT = ALL_PRVR_ROOT / "GMMFormer_v2"
SRC_ROOT = PRVR_ROOT / "src"
EXP_ROOT = Path(os.environ.get("PRVR_EXP_ROOT", PROJECT_ROOT / "experiments"))
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(PRVR_ROOT.parent))

from Configs.builder import get_configs  # noqa: E402
from Datasets.builder import get_datasets  # noqa: E402
from Datasets.data_provider import collate_text_val  # noqa: E402
from Models.builder import get_models  # noqa: E402
from Utils.utils import load_ckpt, set_seed  # noqa: E402
from Validations.builder import get_validations  # noqa: E402


COLLECTION = {"act": "activitynet", "tvr": "tvr", "cha": "charades", "msrvtt": "msrvtt"}
DEPTH_KS = (1, 5, 10)


def gzip_writer(path, fields):
    handle = gzip.open(path, "wt", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    return handle, writer


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def inverse_ranks(scores):
    """One-based video ranks, with the exact argsort convention of evaluation."""
    order = torch.argsort(scores, dim=1, descending=True)
    ranks = torch.empty_like(order)
    positions = torch.arange(1, scores.shape[1] + 1, device=scores.device).expand_as(order)
    ranks.scatter_(1, order, positions)
    return order, ranks


def raw_topk(context, query_vectors, top_l):
    """Return raw (Q,L,V) max scores and top-L raw-representation indices."""
    raw = torch.matmul(context, query_vectors.t()).permute(2, 1, 0)
    video_scores, winning_repr = torch.max(raw, dim=1)
    raw_values, raw_indices = torch.topk(raw.reshape(raw.shape[0], -1), k=top_l, dim=1)
    return video_scores, winning_repr, raw_values, raw_indices


def raw_dedupe(raw_indices, num_videos):
    """Stats for first-occurrence deduplication inside exported raw top-L."""
    seen, depth_for_k, dedup_rank = set(), {}, []
    for raw_rank, flat_index in enumerate(raw_indices, start=1):
        video_index = flat_index % num_videos
        if video_index in seen:
            dedup_rank.append("")
            continue
        seen.add(video_index)
        dedup_rank.append(len(seen))
        if len(seen) in DEPTH_KS:
            depth_for_k[len(seen)] = raw_rank
    return len(seen), depth_for_k, dedup_rank


def percentile(values, pct):
    if not values:
        return ""
    ordered = sorted(values)
    position = (len(ordered) - 1) * pct / 100.0
    lo, hi = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def rank_metrics(ranks):
    return {
        "r1": 100.0 * sum(rank <= 1 for rank in ranks) / len(ranks),
        "r5": 100.0 * sum(rank <= 5 for rank in ranks) / len(ranks),
        "r10": 100.0 * sum(rank <= 10 for rank in ranks) / len(ranks),
        "mean_rank": statistics.mean(ranks),
        "median_rank": statistics.median(ranks),
        "p90_rank": percentile(ranks, 90),
        "p95_rank": percentile(ranks, 95),
    }


def main():
    collection = COLLECTION[args.dataset]
    checkpoint = Path(args.checkpoint) if args.checkpoint else (
        PRVR_ROOT / "results" / "clip" / collection / "gmmformer_v2" / "best.ckpt"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    output_dir = Path(args.output_dir) if args.output_dir else (
        EXP_ROOT / "branch_rank" / "full_repr" / f"GMMFormer_v2_{args.dataset}_clip_top{args.top_l}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = get_configs(f"{args.dataset}_clip")
    cfg["num_workers"] = 0
    set_seed(cfg["seed"], cuda_deterministic=True)
    cfg, _, _, _, test_context_loader, test_query_loader = get_datasets(cfg)
    full_query_loader = DataLoader(
        test_query_loader.dataset,
        batch_size=args.query_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=cfg["pin_memory"],
        collate_fn=collate_text_val,
    )

    model = get_models(cfg)
    _, state_dict, _, _, _ = load_ckpt(str(checkpoint))
    model.load_state_dict(state_dict)
    model = model.cuda().eval()
    validator = get_validations(cfg)

    paired_fields = [
        "dataset", "checkpoint", "query_dataset_index", "query_id", "gt_video_id", "rank_position",
        "exact_video_id", "exact_score",
        "clip_raw_video_id", "clip_repr_index", "clip_raw_score", "clip_is_first_occurrence", "clip_dedup_rank",
        "frame_raw_video_id", "frame_repr_index", "frame_raw_score", "frame_is_valid_repr",
        "frame_is_first_occurrence", "frame_dedup_rank",
    ]
    query_fields = [
        "dataset", "checkpoint", "query_dataset_index", "query_id", "gt_video_id",
        "rank_exact", "rank_clip", "rank_frame",
        "gt_clip_raw_rank_within_top_l", "gt_frame_raw_rank_within_top_l",
    ]
    dedupe_fields = [
        "dataset", "checkpoint", "query_dataset_index", "query_id", "gt_video_id", "top_l",
        "clip_unique_videos", "clip_duplicate_count", "clip_duplicate_pct",
        "frame_unique_videos", "frame_duplicate_count", "frame_duplicate_pct",
        "clip_raw_depth_for_dedup_at_1", "clip_raw_depth_for_dedup_at_5", "clip_raw_depth_for_dedup_at_10",
        "frame_raw_depth_for_dedup_at_1", "frame_raw_depth_for_dedup_at_5", "frame_raw_depth_for_dedup_at_10",
    ]
    paired_handle, paired_writer = gzip_writer(output_dir / f"ranked_top{args.top_l}.csv.gz", paired_fields)
    query_handle, query_writer = gzip_writer(output_dir / "query_rank_stats.csv.gz", query_fields)
    dedupe_handle, dedupe_writer = gzip_writer(output_dir / "dedupe_per_query.csv.gz", dedupe_fields)

    exact_ranks_all, clip_ranks_all, frame_ranks_all = [], [], []
    unique_counts = {"clip": [], "frame": []}
    depths = {"clip": {k: [] for k in DEPTH_KS}, "frame": {k: [] for k in DEPTH_KS}}
    try:
        with torch.no_grad():
            context_info = validator.compute_context_info(model, test_context_loader)
            video_ids = [str(v) for v in context_info["video_metas"]]
            video_to_index = {video_id: i for i, video_id in enumerate(video_ids)}
            num_videos = len(video_ids)
            if args.top_l > num_videos:
                raise ValueError(f"top-l={args.top_l} exceeds {num_videos} test videos")
            clip_context = F.normalize(context_info["video_proposal_feat"], dim=-1)
            frame_context = F.normalize(context_info["video_feat"], dim=-1)
            frame_valid_mask = context_info["video_mask"]
            if args.top_l > num_videos * clip_context.shape[1] or args.top_l > num_videos * frame_context.shape[1]:
                raise ValueError("top-l exceeds the number of raw representations")

            processed = 0
            for batch_index, batch in enumerate(full_query_loader, start=1):
                query_features, query_mask = batch[0].cuda(), batch[1].cuda()
                source_indices = [int(index) for index in batch[2]]
                query_ids = [str(query_id) for query_id in batch[-1]]
                query_vectors = F.normalize(model.encode_query(query_features, query_mask), dim=-1)

                clip_scores, _, clip_raw_values, clip_raw_indices = raw_topk(clip_context, query_vectors, args.top_l)
                frame_scores, _, frame_raw_values, frame_raw_indices = raw_topk(frame_context, query_vectors, args.top_l)
                exact_scores = cfg["clip_scale_w"] * clip_scores + cfg["frame_scale_w"] * frame_scores
                exact_order, exact_ranks = inverse_ranks(exact_scores)
                _, clip_ranks = inverse_ranks(clip_scores)
                _, frame_ranks = inverse_ranks(frame_scores)

                # Materialize only exported top-L and per-query rank vectors on CPU.
                exact_order = exact_order[:, :args.top_l].cpu().tolist()
                clip_raw_indices = clip_raw_indices.cpu().tolist()
                frame_raw_indices = frame_raw_indices.cpu().tolist()
                clip_raw_values = clip_raw_values.cpu().tolist()
                frame_raw_values = frame_raw_values.cpu().tolist()
                exact_scores_cpu = exact_scores.cpu()
                exact_ranks_cpu, clip_ranks_cpu, frame_ranks_cpu = exact_ranks.cpu(), clip_ranks.cpu(), frame_ranks.cpu()

                for q_pos, (query_id, source_index) in enumerate(zip(query_ids, source_indices)):
                    gt_video_id = query_id.split("#", 1)[0]
                    if gt_video_id not in video_to_index:
                        raise KeyError(f"GT video {gt_video_id!r} is absent from test candidates")
                    gt_index = video_to_index[gt_video_id]
                    exact_rank = int(exact_ranks_cpu[q_pos, gt_index].item())
                    clip_rank = int(clip_ranks_cpu[q_pos, gt_index].item())
                    frame_rank = int(frame_ranks_cpu[q_pos, gt_index].item())
                    exact_ranks_all.append(exact_rank)
                    clip_ranks_all.append(clip_rank)
                    frame_ranks_all.append(frame_rank)

                    clip_unique, clip_depth, clip_dedup = raw_dedupe(clip_raw_indices[q_pos], num_videos)
                    frame_unique, frame_depth, frame_dedup = raw_dedupe(frame_raw_indices[q_pos], num_videos)
                    unique_counts["clip"].append(clip_unique)
                    unique_counts["frame"].append(frame_unique)
                    for k in DEPTH_KS:
                        if k in clip_depth:
                            depths["clip"][k].append(clip_depth[k])
                        if k in frame_depth:
                            depths["frame"][k].append(frame_depth[k])

                    clip_gt_raw = next((i + 1 for i, flat in enumerate(clip_raw_indices[q_pos])
                                        if flat % num_videos == gt_index), "")
                    frame_gt_raw = next((i + 1 for i, flat in enumerate(frame_raw_indices[q_pos])
                                         if flat % num_videos == gt_index), "")
                    base = {
                        "dataset": collection, "checkpoint": str(checkpoint),
                        "query_dataset_index": source_index, "query_id": query_id, "gt_video_id": gt_video_id,
                    }
                    query_writer.writerow({
                        **base, "rank_exact": exact_rank, "rank_clip": clip_rank, "rank_frame": frame_rank,
                        "gt_clip_raw_rank_within_top_l": clip_gt_raw,
                        "gt_frame_raw_rank_within_top_l": frame_gt_raw,
                    })
                    dedupe_writer.writerow({
                        **base, "top_l": args.top_l,
                        "clip_unique_videos": clip_unique, "clip_duplicate_count": args.top_l - clip_unique,
                        "clip_duplicate_pct": 100.0 * (args.top_l - clip_unique) / args.top_l,
                        "frame_unique_videos": frame_unique, "frame_duplicate_count": args.top_l - frame_unique,
                        "frame_duplicate_pct": 100.0 * (args.top_l - frame_unique) / args.top_l,
                        **{f"clip_raw_depth_for_dedup_at_{k}": clip_depth.get(k, "") for k in DEPTH_KS},
                        **{f"frame_raw_depth_for_dedup_at_{k}": frame_depth.get(k, "") for k in DEPTH_KS},
                    })

                    for position in range(args.top_l):
                        exact_index = exact_order[q_pos][position]
                        clip_flat = clip_raw_indices[q_pos][position]
                        frame_flat = frame_raw_indices[q_pos][position]
                        clip_repr, clip_index = divmod(clip_flat, num_videos)
                        frame_repr, frame_index = divmod(frame_flat, num_videos)
                        paired_writer.writerow({
                            **base, "rank_position": position + 1,
                            "exact_video_id": video_ids[exact_index],
                            "exact_score": float(exact_scores_cpu[q_pos, exact_index].item()),
                            "clip_raw_video_id": video_ids[clip_index], "clip_repr_index": clip_repr + 1,
                            "clip_raw_score": clip_raw_values[q_pos][position],
                            "clip_is_first_occurrence": int(clip_dedup[position] != ""),
                            "clip_dedup_rank": clip_dedup[position],
                            "frame_raw_video_id": video_ids[frame_index], "frame_repr_index": frame_repr + 1,
                            "frame_raw_score": frame_raw_values[q_pos][position],
                            "frame_is_valid_repr": int(frame_valid_mask[frame_index, frame_repr].item()),
                            "frame_is_first_occurrence": int(frame_dedup[position] != ""),
                            "frame_dedup_rank": frame_dedup[position],
                        })
                processed += len(query_ids)
                print(f"[{collection}] query batch {batch_index}/{len(full_query_loader)}: {processed} queries", flush=True)
    finally:
        paired_handle.close()
        query_handle.close()
        dedupe_handle.close()

    summary_fields = ["section", "branch", "metric", "value", "top_l", "num_queries", "coverage_pct"]
    summary_rows = []
    for branch, ranks in (("exact", exact_ranks_all), ("clip", clip_ranks_all), ("frame", frame_ranks_all)):
        for metric, value in rank_metrics(ranks).items():
            summary_rows.append({"section": "video_rank", "branch": branch, "metric": metric, "value": value,
                                 "top_l": args.top_l, "num_queries": len(ranks), "coverage_pct": 100.0})
    for branch in ("clip", "frame"):
        counts = unique_counts[branch]
        duplicates = [args.top_l - value for value in counts]
        for metric, value in (
            ("mean_unique_videos", statistics.mean(counts)),
            ("median_unique_videos", statistics.median(counts)),
            ("mean_duplicate_pct", 100.0 * statistics.mean(duplicates) / args.top_l),
            ("median_duplicate_pct", 100.0 * statistics.median(duplicates) / args.top_l),
        ):
            summary_rows.append({"section": "dedupe_at_raw_top_l", "branch": branch, "metric": metric,
                                 "value": value, "top_l": args.top_l, "num_queries": len(counts), "coverage_pct": 100.0})
        for k in DEPTH_KS:
            values = depths[branch][k]
            coverage = 100.0 * len(values) / len(counts)
            for metric, value in (("p50_raw_depth", percentile(values, 50)), ("p90_raw_depth", percentile(values, 90)),
                                  ("p95_raw_depth", percentile(values, 95)), ("max_raw_depth", max(values) if values else "")):
                summary_rows.append({"section": f"raw_depth_for_dedup_at_{k}", "branch": branch, "metric": metric,
                                     "value": value, "top_l": args.top_l, "num_queries": len(counts), "coverage_pct": coverage})
    write_csv(output_dir / "summary.csv", summary_fields, summary_rows)
    (output_dir / "README.txt").write_text(
        f"Full test-set export for GMMFormer-v2 CLIP checkpoint: {checkpoint}\n\n"
        f"ranked_top{args.top_l}.csv.gz has one row per query and rank position. exact_video_id is the original "
        "weighted video-level ranking. clip_raw_video_id and frame_raw_video_id are raw representation rankings; "
        "repeated IDs are expected. *_is_first_occurrence and *_dedup_rank reconstruct the branch video ranking.\n\n"
        "query_rank_stats.csv.gz contains true full-candidate GT ranks and is the source of video Recall@1/5/10. "
        "dedupe_per_query.csv.gz reports duplicate rates in raw top-L and the raw depth required to reach 1/5/10 "
        "unique videos. summary.csv aggregates these statistics. A fixed raw top-L is not itself video Recall@K; "
        "deduplicate by first occurrence before computing video-level Recall@K.\n",
        encoding="utf-8",
    )
    print(f"Completed {collection}: {len(exact_ranks_all)} queries -> {output_dir}")


if __name__ == "__main__":
    main()
