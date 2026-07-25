#!/usr/bin/env python3
"""Export GMMFormer-v2 CLIP-branch representation and deduplicated ranks."""
import argparse
import csv
import os
import random
import sys
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=("act", "tvr", "cha", "msrvtt"), default="act")
parser.add_argument("--branch", choices=("clip", "frame", "exact"), default="clip")
parser.add_argument("--gpu", required=True, help="physical CUDA GPU index")
parser.add_argument("--num-queries", type=int, default=50)
parser.add_argument("--seed", type=int, default=9527)
parser.add_argument("--raw-topk", type=int, default=500)
parser.add_argument("--dedup-topk", type=int, default=500)
parser.add_argument("--checkpoint", default="")
parser.add_argument("--output-dir", default="")
args = parser.parse_args()

# Must precede torch/project imports.
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

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


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def score_and_rank(context, query_vectors):
    """Return raw (Q,L,V), max-video score (Q,V), winning repr and rank."""
    raw = torch.matmul(context, query_vectors.t()).permute(2, 1, 0)
    video_score, winning_repr = torch.max(raw, dim=1)
    order = torch.argsort(video_score, dim=1, descending=True)
    ranks = torch.empty_like(order)
    positions = torch.arange(1, video_score.shape[1] + 1, device=video_score.device).expand_as(order)
    ranks.scatter_(1, order, positions)
    return raw, video_score, winning_repr, ranks


def main():
    collection = COLLECTION[args.dataset]
    checkpoint = Path(args.checkpoint) if args.checkpoint else \
        PRVR_ROOT / "results" / "clip" / collection / "gmmformer_v2" / "best.ckpt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    output_dir = Path(args.output_dir) if args.output_dir else \
        EXP_ROOT / "branch_rank" / "raw_repr" / f"GMMFormer_v2_{args.dataset}_clip_seed{args.seed}_n{args.num_queries}"
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = get_configs(f"{args.dataset}_clip")
    cfg["num_workers"] = 0
    set_seed(cfg["seed"], cuda_deterministic=True)
    cfg, _, _, _, test_context_loader, test_query_loader = get_datasets(cfg)
    if args.num_queries > len(test_query_loader.dataset):
        raise ValueError("num-queries exceeds test split size")
    sampled_indices = random.Random(args.seed).sample(range(len(test_query_loader.dataset)), args.num_queries)
    sampled_loader = DataLoader(Subset(test_query_loader.dataset, sampled_indices), batch_size=args.num_queries,
                                shuffle=False, num_workers=0, pin_memory=cfg["pin_memory"], collate_fn=collate_text_val)

    model = get_models(cfg)
    _, state_dict, _, _, _ = load_ckpt(str(checkpoint))
    model.load_state_dict(state_dict)
    model = model.cuda().eval()
    validator = get_validations(cfg)

    with torch.no_grad():
        context_info = validator.compute_context_info(model, test_context_loader)
        video_ids = [str(v) for v in context_info["video_metas"]]
        clip_context = F.normalize(context_info["video_proposal_feat"], dim=-1)  # (V, 32, D)
        frame_context = F.normalize(context_info["video_feat"], dim=-1)  # (V, 128, D)
        frame_valid_mask = context_info["video_mask"]
        if args.branch == "clip":
            context, valid_repr_mask = clip_context, None
        elif args.branch == "frame":
            # This is exactly the encoded_frame_feat supplied to
            # get_clip_scale_scores in the original frame branch. Invalid
            # padded positions remain zero, as they do in the original code.
            context, valid_repr_mask = frame_context, frame_valid_mask
        candidate_count = clip_context.shape[0]
        if args.dedup_topk > candidate_count:
            raise ValueError("dedup-topk exceeds the number of videos")
        if args.branch != "exact":
            repr_per_video = context.shape[1]
            if args.raw_topk > candidate_count * repr_per_video:
                raise ValueError("raw-topk exceeds the number of representations")
        batch = next(iter(sampled_loader))
        query_features, query_mask = batch[0].cuda(), batch[1].cuda()
        # collate_text_val sorts by caption length, so use its returned source
        # indices instead of the original random-sample order.
        source_indices = [int(index) for index in batch[2]]
        query_ids = [str(q) for q in batch[-1]]
        query_vectors = F.normalize(model.encode_query(query_features, query_mask), dim=-1)
        if args.branch == "exact":
            _, clip_video_scores, clip_winning_repr, clip_ranks = score_and_rank(clip_context, query_vectors)
            _, frame_video_scores, frame_winning_repr, frame_ranks = score_and_rank(frame_context, query_vectors)
            exact_scores = cfg["clip_scale_w"] * clip_video_scores + cfg["frame_scale_w"] * frame_video_scores
            exact_order = torch.argsort(exact_scores, dim=1, descending=True)
        else:
            # Same primitive as get_clip_scale_scores before max over representation dim.
            scores = torch.matmul(context, query_vectors.t()).permute(2, 1, 0)
            flat_scores = scores.reshape(len(query_ids), -1)
            full_order = torch.argsort(flat_scores, dim=1, descending=True)

    common = ["dataset", "branch", "checkpoint", "sample_seed", "query_sample_index", "query_dataset_index", "query_id", "gt_video_id"]
    sample_fields = ["query_sample_index", "query_dataset_index", "query_id", "gt_video_id"]
    sample_rows = [{"query_sample_index": q_pos, "query_dataset_index": source_index,
                    "query_id": query_id, "gt_video_id": query_id.split("#", 1)[0]}
                   for q_pos, (query_id, source_index) in enumerate(zip(query_ids, source_indices))]
    if args.branch == "exact":
        exact_fields = common + [
            "exact_rank", "exact_score", "clip_weight", "frame_weight", "video_id",
            "clip_score", "frame_score", "clip_video_rank", "frame_video_rank",
            "clip_winning_repr_index", "clip_winning_repr_id",
            "frame_winning_repr_index", "frame_winning_repr_id", "frame_winning_repr_is_valid",
        ]
        exact_rows = []
        for q_pos, (query_id, source_index) in enumerate(zip(query_ids, source_indices)):
            gt_video_id = query_id.split("#", 1)[0]
            base = {"dataset": collection, "branch": "exact", "checkpoint": str(checkpoint), "sample_seed": args.seed,
                    "query_sample_index": q_pos, "query_dataset_index": source_index,
                    "query_id": query_id, "gt_video_id": gt_video_id}
            for exact_rank, video_index in enumerate(exact_order[q_pos, :args.dedup_topk].cpu().tolist(), start=1):
                clip_repr = int(clip_winning_repr[q_pos, video_index].item()) + 1
                frame_repr = int(frame_winning_repr[q_pos, video_index].item()) + 1
                video_id = video_ids[video_index]
                exact_rows.append({
                    **base, "exact_rank": exact_rank, "exact_score": float(exact_scores[q_pos, video_index].cpu()),
                    "clip_weight": cfg["clip_scale_w"], "frame_weight": cfg["frame_scale_w"], "video_id": video_id,
                    "clip_score": float(clip_video_scores[q_pos, video_index].cpu()),
                    "frame_score": float(frame_video_scores[q_pos, video_index].cpu()),
                    "clip_video_rank": int(clip_ranks[q_pos, video_index].item()),
                    "frame_video_rank": int(frame_ranks[q_pos, video_index].item()),
                    "clip_winning_repr_index": clip_repr, "clip_winning_repr_id": f"{video_id}_{clip_repr}",
                    "frame_winning_repr_index": frame_repr, "frame_winning_repr_id": f"{video_id}_frame_{frame_repr}",
                    "frame_winning_repr_is_valid": int(frame_valid_mask[video_index, frame_repr - 1].item()),
                })
        write_csv(output_dir / "exact_dedup_top500.csv", exact_fields, exact_rows)
        write_csv(output_dir / "sampled_queries.csv", sample_fields, sample_rows)
        print(f"Wrote exact {len(exact_rows)} video rows: {output_dir}")
        return

    raw_fields = common + ["raw_rank", "raw_score", "video_id", "repr_index", "repr_id", "is_valid_repr", "is_first_occurrence", "dedup_rank_within_raw_topk"]
    dedup_fields = common + ["dedup_rank", "first_raw_rank", "winning_score", "video_id", "winning_repr_index", "winning_repr_id", "winning_repr_is_valid"]
    raw_rows, dedup_rows = [], []

    for q_pos, (query_id, source_index) in enumerate(zip(query_ids, source_indices)):
        gt_video_id = query_id.split("#", 1)[0]
        order, flat = full_order[q_pos].cpu().tolist(), flat_scores[q_pos]
        base = {"dataset": collection, "branch": args.branch, "checkpoint": str(checkpoint), "sample_seed": args.seed,
                "query_sample_index": q_pos, "query_dataset_index": source_index,
                "query_id": query_id, "gt_video_id": gt_video_id}
        seen_topk, dedup_rank = set(), 0
        for raw_rank, flat_index in enumerate(order[:args.raw_topk], start=1):
            repr_index, video_index = flat_index // candidate_count + 1, flat_index % candidate_count
            first = video_index not in seen_topk
            if first:
                seen_topk.add(video_index)
                dedup_rank += 1
            video_id = video_ids[video_index]
            is_valid = 1 if valid_repr_mask is None else int(valid_repr_mask[video_index, repr_index - 1].item())
            repr_id = f"{video_id}_{repr_index}" if args.branch == "clip" else f"{video_id}_frame_{repr_index}"
            raw_rows.append({**base, "raw_rank": raw_rank, "raw_score": float(flat[flat_index].cpu()),
                             "video_id": video_id, "repr_index": repr_index, "repr_id": repr_id, "is_valid_repr": is_valid,
                             "is_first_occurrence": int(first), "dedup_rank_within_raw_topk": dedup_rank if first else ""})

        # Scan the whole raw order until 500 unique videos. The first occurrence
        # of a video is its max-sim representation, hence this is the original
        # clip-branch video ranking except for exact score ties.
        seen = set()
        for raw_rank, flat_index in enumerate(order, start=1):
            repr_index, video_index = flat_index // candidate_count + 1, flat_index % candidate_count
            if video_index in seen:
                continue
            seen.add(video_index)
            video_id = video_ids[video_index]
            is_valid = 1 if valid_repr_mask is None else int(valid_repr_mask[video_index, repr_index - 1].item())
            repr_id = f"{video_id}_{repr_index}" if args.branch == "clip" else f"{video_id}_frame_{repr_index}"
            dedup_rows.append({**base, "dedup_rank": len(seen), "first_raw_rank": raw_rank,
                               "winning_score": float(flat[flat_index].cpu()), "video_id": video_id,
                               "winning_repr_index": repr_index, "winning_repr_id": repr_id, "winning_repr_is_valid": is_valid})
            if len(seen) == args.dedup_topk:
                break
    prefix = "" if args.branch == "clip" else "frame_"
    write_csv(output_dir / f"{prefix}raw_top500.csv", raw_fields, raw_rows)
    write_csv(output_dir / f"{prefix}dedup_top500.csv", dedup_fields, dedup_rows)
    write_csv(output_dir / "sampled_queries.csv", sample_fields, sample_rows)
    (output_dir / "README.txt").write_text(
        "raw_top500.csv lists 500 individual clip representations per query.\n"
        "dedup_top500.csv lists 500 unique clip videos after first-occurrence deduplication.\n"
        "frame_raw_top500.csv/frame_dedup_top500.csv are analogous frame-branch exports.\n"
        "Each deduplicated order equals its original GMMFormer-v2 max-sim branch ranking, except score ties.\n",
        encoding="utf-8")
    print(f"Wrote {args.branch} {len(raw_rows)} raw and {len(dedup_rows)} deduplicated rows: {output_dir}")


if __name__ == "__main__":
    main()
