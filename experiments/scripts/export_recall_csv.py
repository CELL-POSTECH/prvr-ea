#!/usr/bin/env python3
"""Create compact dataset-wise Recall@1/5/10 CSVs from the evaluation ledger."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


DATASETS = ("tvr", "act", "cha", "msrvtt")
PROJECT_ROOT = Path(os.environ.get("PRVR_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
EXP_ROOT = Path(os.environ.get("PRVR_EXP_ROOT", PROJECT_ROOT / "experiments"))
DEFAULT_OUTPUT_DIR = EXP_ROOT / "recall_results"
METHODS = {
    "resnet": (
        ("AMDNet", "AMDNet"), ("GMMFormer", "GMMFormer"),
        ("GMMFormer-v2", "GMMFormerV2"), ("HLFormer", "HLFormer"),
        ("Holmes", "Holmes"), ("DreamPRVR", "DreamPRVR"), ("BOA", "BOA"),
        ("DL-DKD", "DL-DKD"), ("MS-SL", "MS-SL"), ("BGM-Net", "BGMNet"),
    ),
    "clip": (
        ("CLIP4Clip", "CLIP4Clip"), ("AMDNet", "AMDNet"),
        ("GMMFormer", "GMMFormer"), ("GMMFormer-v2", "GMMFormerV2"),
        ("HLFormer", "HLFormer"), ("Holmes", "Holmes"),
        ("DreamPRVR", "DreamPRVR"), ("BOA", "BOA"),
        ("MSC-PRVR", "MSC-PRVR"), ("DL-DKD", "DL-DKD"),
        ("MS-SL", "MS-SL"), ("BGM-Net", "BGMNet"),
    ),
}


def matches_feature(row: dict[str, str], feature: str, multi_gt: bool) -> bool:
    suffix = "_multiGT" if multi_gt else ""
    if feature == "resnet":
        return row["feature"] == "resnet" + suffix
    # CLIP4Clip is raw-frame zero-shot; all other CLIP-table models consume
    # the prepared CLIP-B/32 features. DL-DKD retains its original dual input.
    return (row["model"] == "CLIP4Clip" and row["feature"] == "rawframes128" + suffix) or (
        row["model"] != "CLIP4Clip" and row["feature"] == "clip" + suffix
    )


def latest_ok_rows(
    ledger: Path,
    feature: str,
    multi_gt: bool,
) -> dict[tuple[str, str], dict[str, str]]:
    if not ledger.exists():
        return {}
    selected: dict[tuple[str, str], dict[str, str]] = {}
    dldkd_resnet: dict[str, dict[str, str]] = {}
    suffix = "_multiGT" if multi_gt else ""
    with ledger.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "ok":
                continue
            if not all(row.get(metric, "") for metric in ("r1", "r5", "r10")):
                continue
            if (
                feature == "clip"
                and row["model"] == "DL-DKD"
                and row["feature"] == "resnet" + suffix
                and (
                    row["dataset"] not in dldkd_resnet
                    or row["finished_at_utc"] > dldkd_resnet[row["dataset"]]["finished_at_utc"]
                )
            ):
                dldkd_resnet[row["dataset"]] = row
            if not matches_feature(row, feature, multi_gt):
                continue
            key = (row["dataset"], row["model"])
            if key not in selected or row["finished_at_utc"] > selected[key]["finished_at_utc"]:
                selected[key] = row
    if feature == "clip":
        for dataset, row in dldkd_resnet.items():
            selected.setdefault((dataset, "DL-DKD"), row)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", choices=METHODS, required=True)
    parser.add_argument("--ledger", type=Path, default=EXP_ROOT / "recall_clip.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    parser.add_argument("--multi-gt", action="store_true", help="export the dense multi-GT ledger rows")
    args = parser.parse_args()

    selected = latest_ok_rows(args.ledger, args.feature, args.multi_gt)
    if args.multi_gt and args.output_dir == DEFAULT_OUTPUT_DIR:
        args.output_dir = args.output_dir / "multiGT"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    for dataset in datasets:
        suffix = "_multiGT" if args.multi_gt else ""
        output = args.output_dir / f"recall_{args.feature}_{dataset}{suffix}.csv"
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("Method", "R@1", "R@5", "R@10"))
            writer.writeheader()
            for internal_name, display_name in METHODS[args.feature]:
                row = selected.get((dataset, internal_name), {})
                writer.writerow({
                    "Method": display_name,
                    "R@1": row.get("r1", ""),
                    "R@5": row.get("r5", ""),
                    "R@10": row.get("r10", ""),
                })
        print(output)


if __name__ == "__main__":
    main()
