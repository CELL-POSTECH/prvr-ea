#!/usr/bin/env python3
"""Plot per-video frame-count histograms from PRVR CLIP visual HDF5 files."""

import argparse
import csv
import os
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


DATASETS = (
    ("ActivityNet", "activitynet", "new_clip_vit_32_activitynet_vid_features.hdf5", "activitynettest.caption.txt"),
    ("TVR", "tvr", "new_clip_vit_32_tvr_vid_features.hdf5", "tvrtest.caption.txt"),
    ("Charades", "charades", "new_clip_vit_32_charades_vid_features.hdf5", "charadestest.caption.txt"),
    ("MSR-VTT", "msrvtt", "new_clip_vit_32_msrvtt_vid_features.hdf5", "msrvtttest.caption.txt"),
)

# Match the manuscript reference figure: fixed axes make the four datasets
# directly comparable across re-runs. MSR-VTT has a small 44--46-frame tail
# in the current HDF5; it remains plotted to x=46 while the major ticks match
# the supplied figure.
AXIS_STYLE = {
    "activitynet": {
        "xlim": (2, 1165),
        "xticks": [2, 147, 292, 438, 583, 728, 874, 1019, 1165],
        "ylim": (0, 30), "yticks": [0, 6, 12, 18, 24, 30],
    },
    "tvr": {
        "xlim": (6, 123),
        "xticks": [6, 20, 35, 49, 64, 79, 93, 108, 123],
        "ylim": (0, 400), "yticks": [0, 80, 160, 240, 320, 400],
    },
    "charades": {
        "xlim": (21, 227),
        "xticks": [21, 46, 72, 98, 124, 149, 175, 201, 227],
        "ylim": (0, 125), "yticks": [0, 25, 50, 75, 100, 125],
    },
    "msrvtt": {
        "xlim": (16, 46),
        "xticks": [16, 21, 27, 31, 37, 43],
        "ylim": (0, 200), "yticks": [0, 40, 80, 120, 160, 200],
    },
}


def test_video_ids(caption_path: Path):
    """Return test-candidate video ids in caption-file order, without duplicates."""
    if not caption_path.is_file():
        raise FileNotFoundError(caption_path)
    video_ids = []
    seen = set()
    for line in caption_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        query_id = line.split(None, 1)[0]
        video_id = query_id.split("#", 1)[0]
        if video_id not in seen:
            video_ids.append(video_id)
            seen.add(video_id)
    if not video_ids:
        raise ValueError(f"No query/video ids in {caption_path}")
    return video_ids


def frame_counts(path: Path, video_ids) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        missing = [video_id for video_id in video_ids if video_id not in handle]
        if missing:
            raise KeyError(f"{len(missing)} test video ids missing from {path}; first: {missing[:5]}")
        counts = [handle[video_id].shape[0] for video_id in video_ids]
    if not counts:
        raise ValueError(f"No video datasets in {path}")
    return np.asarray(counts, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--data-root", type=Path,
                        default=Path(os.environ.get("PRVR_DATA_ROOT", project_root / "datasets")))
    parser.add_argument("--output-dir", type=Path,
                        default=project_root / "experiments" / "figures" / "histogram")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    # Arial is the requested manuscript font. Matplotlib will use its normal
    # configured fallback only when Arial has not been installed on the host.
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 15,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.linewidth": 1.7,
        "xtick.major.width": 1.7,
        "ytick.major.width": 1.7,
    })

    loaded = []
    for title, collection, filename, caption_filename in DATASETS:
        path = args.data_root / collection / "FeatureData" / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        caption_path = args.data_root / collection / "TextData" / caption_filename
        loaded.append((title, collection, path, caption_path,
                       frame_counts(path, test_video_ids(caption_path))))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(17.16, 2.75))
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.28, top=0.90, wspace=0.26)
    color = "#9ecae1"
    summary_rows = []

    for axis, (title, collection, path, caption_path, counts) in zip(axes, loaded):
        low, high = int(counts.min()), int(counts.max())
        style = AXIS_STYLE[collection]
        # Integer-centered bins make each thin bar represent exactly one
        # stored CLIP frame count, matching the supplied reference style.
        bins = np.arange(low - 0.5, high + 1.5, 1.0)
        axis.hist(counts, bins=bins, color=color, edgecolor=color, linewidth=0.15, rwidth=0.82)
        axis.set_title(title, pad=4)
        axis.set_xlim(*style["xlim"])
        axis.set_xticks(style["xticks"])
        axis.set_ylim(*style["ylim"])
        axis.set_yticks(style["yticks"])
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(direction="out", length=5, width=1.7)
        summary_rows.append({
            "dataset": collection,
            "hdf5": str(path),
            "test_caption_file": str(caption_path),
            "num_videos": len(counts),
            "min_frames": low,
            "mean_frames": f"{counts.mean():.4f}",
            "median_frames": f"{np.median(counts):.4f}",
            "p95_frames": f"{np.percentile(counts, 95):.4f}",
            "p99_frames": f"{np.percentile(counts, 99):.4f}",
            "max_frames": high,
        })

    axes[0].set_ylabel("Frequency")
    fig.supxlabel("Frames per Video", y=0.02)
    png_path = args.output_dir / "clip_frames_per_video_histogram.png"
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    csv_path = args.output_dir / "clip_frames_per_video_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"wrote {png_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
