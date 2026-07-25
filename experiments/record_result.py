#!/usr/bin/env python3
"""Append one PRVR run result to a shared CSV without concurrent-write races."""

import argparse
import csv
import fcntl
import re
from datetime import datetime, timezone
from pathlib import Path


FIELDNAMES = [
    "finished_at_utc", "gpu", "model", "dataset", "feature", "status",
    "r1", "r5", "r10", "r100", "rsum", "run_dir", "log_path", "command",
]

DATASET_ORDER = {"tvr": 0, "act": 1, "cha": 2, "msrvtt": 3}
RESNET_ORDER = {
    "AMDNet": 0, "GMMFormer": 1, "GMMFormer-v2": 2, "HLFormer": 3,
    "Holmes": 4, "DreamPRVR": 5, "BOA": 6, "DL-DKD": 7,
    "MS-SL": 8, "BGM-Net": 9,
}
CLIP_ORDER = {
    "CLIP4Clip": 0, "AMDNet": 1, "GMMFormer": 2, "GMMFormer-v2": 3,
    "HLFormer": 4, "DreamPRVR": 5, "Holmes": 6, "BOA": 7,
    "MSC-PRVR": 8, "DL-DKD": 9, "MS-SL": 10, "BGM-Net": 11,
}


def row_sort_key(row):
    comparison_group = 1 if row["model"] == "CLIP4Clip" or row["feature"] != "resnet" else 0
    method_order = RESNET_ORDER if comparison_group == 0 else CLIP_ORDER
    return (
        comparison_group,
        DATASET_ORDER.get(row["dataset"], len(DATASET_ORDER)),
        method_order.get(row["model"], len(method_order)),
        row["finished_at_utc"],
    )


def last_metrics(text: str, model: str = ""):
    """Read the last test/validation metric block written by the PRVR repos."""
    metrics = {}

    # CLIP4Clip logs T2V first and V2T second.  The comparison table is T2V,
    # so do not let the generic R@ regex below accidentally select the later
    # ``V2T$R@...`` line.
    if model == "CLIP4Clip":
        clip4clip_blocks = re.findall(
            r"(?m)^.*>>>\s+R@1:\s*([-+]?\d*\.?\d+)\s*-\s*"
            r"R@5:\s*([-+]?\d*\.?\d+)\s*-\s*R@10:\s*([-+]?\d*\.?\d+)",
            text,
        )
        if clip4clip_blocks:
            metrics.update(dict(zip(("r1", "r5", "r10"), clip4clip_blocks[-1])))

    # BGM-Net, MS-SL, and DL-DKD.
    list_blocks = re.findall(r"r_1_5_10_100(?:, medr, meanr)?:\s*\[([^\]]+)\]", text, re.I)
    if list_blocks:
        values = re.findall(r"[-+]?\d*\.?\d+", list_blocks[-1])
        if len(values) >= 4:
            metrics.update(dict(zip(("r1", "r5", "r10", "r100"), values[:4])))

    # Config-based repos print each value independently.  Taking the final
    # occurrence means the explicit post-training test evaluation wins.
    if model != "CLIP4Clip":
        for key, rank in (("r1", "1"), ("r5", "5"), ("r10", "10"), ("r100", "100")):
            values = re.findall(rf"R@{rank}:\s*([-+]?\d*\.?\d+)", text, re.I)
            if values:
                metrics[key] = values[-1]

    # MSC-PRVR prints all five values on one line.
    msc_blocks = re.findall(
        r"(?<![A-Za-z])R@1\s+5\s+10\s+100\s+Rsum:\s*"
        r"([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*"
        r"([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*"
        r"([-+]?\d*\.?\d+)", text, re.I)
    if msc_blocks:
        metrics.update(dict(zip(("r1", "r5", "r10", "r100", "rsum"), msc_blocks[-1])))

    rsum = re.findall(r"(?:recall sum|Rsum):\s*([-+]?\d*\.?\d+)", text, re.I)
    if rsum:
        metrics["rsum"] = rsum[-1]
    if "rsum" not in metrics and all(k in metrics for k in ("r1", "r5", "r10", "r100")):
        metrics["rsum"] = f"{sum(float(metrics[k]) for k in ('r1', 'r5', 'r10', 'r100')):.1f}"
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--feature", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--command", required=True)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log)
    metrics = last_metrics(log_path.read_text(errors="replace"), args.model) if log_path.exists() else {}
    row = {
        "finished_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gpu": args.gpu, "model": args.model, "dataset": args.dataset,
        "feature": args.feature, "status": args.status,
        "run_dir": args.run_dir, "log_path": str(log_path), "command": args.command,
        **{key: metrics.get(key, "") for key in ("r1", "r5", "r10", "r100", "rsum")},
    }

    # Locking the CSV file itself makes header creation and row append one
    # atomic critical section across all four runner processes.
    with csv_path.open("a+", newline="", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        if not handle.read(1):
            csv.DictWriter(handle, fieldnames=FIELDNAMES).writeheader()
        handle.seek(0)
        rows = list(csv.DictReader(handle))
        rows.append(row)
        rows.sort(key=row_sort_key)
        handle.seek(0)
        handle.truncate()
        csv.DictWriter(handle, fieldnames=FIELDNAMES).writeheader()
        csv.DictWriter(handle, fieldnames=FIELDNAMES).writerows(rows)
        handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


if __name__ == "__main__":
    main()
