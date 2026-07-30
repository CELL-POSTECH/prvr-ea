#!/usr/bin/env python3
"""Plot TVR Qwen pseudo-GT verification frame galleries.

Each output image contains one query, the 8 GT/reference frames, and the 8
candidate frames with the Qwen accept or reject decision.

Example:
    python qwen/plot_tvr_qwen_verification.py \
      --input outputs/runs/sheldon_sits_couch/verification.jsonl \
      --output-dir outputs/runs/sheldon_sits_couch/plots
"""

from __future__ import annotations

import argparse
import json
import math
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageOps
from tqdm import tqdm


DEFAULT_INPUT = "outputs/runs/full/verification.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/runs/full/plots"
DEFAULT_BACKEND = "matplotlib"


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def slugify(value: str, max_len: int = 120) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    value = re.sub(r"_+", "_", value).strip("_")
    return (value or "record")[:max_len]


def get_candidate_paths(record: Dict[str, Any]) -> List[str]:
    paths = record.get("candidate_frame_paths") or record.get("pseudo_frame_paths")
    if not paths:
        raise KeyError("Missing candidate_frame_paths or pseudo_frame_paths")
    return list(paths)


def get_gt_paths(record: Dict[str, Any]) -> List[str]:
    paths = record.get("gt_frame_paths")
    if not paths:
        raise KeyError("Missing gt_frame_paths")
    return list(paths)


def get_candidate_video_id(record: Dict[str, Any]) -> str:
    return str(record.get("candidate_video_id") or record.get("pseudo_video_id") or "unknown_candidate")


def get_gt_video_id(record: Dict[str, Any]) -> str:
    return str(
        record.get("gt_video_id")
        or record.get("original_gt_video_id")
        or record.get("query_gt_video_id")
        or "unknown_gt"
    )


def get_query_id(record: Dict[str, Any]) -> str:
    return str(record.get("query_id") or record.get("query_key") or record.get("desc_id") or "unknown_query")


def get_decision(record: Dict[str, Any]) -> str:
    return str(record.get("qwen_recommendation") or record.get("qwen_result", {}).get("gt_label_recommendation") or "none")


def frame_title(path: str, index: int) -> str:
    return f"{index + 1}: {Path(path).stem}"


def resize_to_max_pixels(img: Image.Image, max_pixels: Optional[int]) -> Image.Image:
    if max_pixels is None or max_pixels <= 0:
        return img

    width, height = img.size
    pixels = width * height
    if pixels <= max_pixels:
        return img

    scale = math.sqrt(max_pixels / pixels)
    target_size = (
        max(1, int(width * scale)),
        max(1, int(height * scale)),
    )
    return img.resize(target_size, Image.Resampling.LANCZOS)


def load_image(path: str, max_pixels: Optional[int]) -> Image.Image:
    img = Image.open(path)
    if max_pixels is not None and max_pixels > 0:
        width, height = img.size
        if width * height > max_pixels:
            scale = math.sqrt(max_pixels / (width * height))
            img.draft("RGB", (max(1, int(width * scale)), max(1, int(height * scale))))
    return resize_to_max_pixels(img.convert("RGB"), max_pixels)


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    candidates = names if bold else tuple(reversed(names))
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    left, _top, right, _bottom = draw.textbbox((0, 0), text, font=font)
    return right - left


def wrap_text_to_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> List[str]:
    lines: List[str] = []
    for paragraph in text.splitlines() or [""]:
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if text_width(draw, candidate, font) <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def draw_frame_row(
    axes: Sequence[Any],
    paths: Sequence[str],
    row_label: str,
    title_color: str,
    show_filenames: bool,
    max_pixels: Optional[int],
) -> None:
    for idx, (ax, path) in enumerate(zip(axes, paths)):
        img = load_image(path, max_pixels=max_pixels)
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        if idx == 0:
            ax.set_ylabel(row_label, fontsize=12, fontweight="bold", rotation=0, labelpad=55, va="center")
        if show_filenames:
            ax.set_title(frame_title(path, idx), fontsize=8, color=title_color)
        else:
            ax.set_title(str(idx + 1), fontsize=8, color=title_color)


def make_plot(
    record: Dict[str, Any],
    output_path: Path,
    dpi: int,
    show_filenames: bool,
    max_pixels: Optional[int] = None,
) -> None:
    gt_paths = get_gt_paths(record)
    candidate_paths = get_candidate_paths(record)
    if len(gt_paths) != 8 or len(candidate_paths) != 8:
        raise ValueError(
            f"Expected 8 GT and 8 candidate frames, got {len(gt_paths)} GT and {len(candidate_paths)} candidate"
        )

    decision = get_decision(record)
    query = str(record.get("query", "")).strip()
    qwen_reason = str(record.get("qwen_result", {}).get("reason") or "")
    confidence = record.get("qwen_result", {}).get("confidence")
    candidate_color = "#138a36" if decision == "accept" else "#b3261e"

    fig, axes = plt.subplots(2, 8, figsize=(24, 7.4), constrained_layout=False)
    fig.patch.set_facecolor("white")

    draw_frame_row(
        axes=axes[0],
        paths=gt_paths,
        row_label="GT",
        title_color="#2f5597",
        show_filenames=show_filenames,
        max_pixels=max_pixels,
    )
    draw_frame_row(
        axes=axes[1],
        paths=candidate_paths,
        row_label="Candidate",
        title_color=candidate_color,
        show_filenames=show_filenames,
        max_pixels=max_pixels,
    )

    title = (
        f"Query: {query}\n"
        f"GT: {get_gt_video_id(record)}    Candidate: {get_candidate_video_id(record)}    "
        f"Qwen decision: {decision}"
    )
    if confidence is not None:
        title += f"    Confidence: {confidence}"

    wrapped_title = "\n".join(textwrap.wrap(title, width=170, replace_whitespace=False))
    fig.suptitle(wrapped_title, fontsize=14, fontweight="bold", x=0.5, y=0.985)

    footer_parts = []
    if qwen_reason:
        footer_parts.append(f"Qwen reason: {qwen_reason}")
    if footer_parts:
        footer = "    ".join(footer_parts)
        fig.text(
            0.01,
            0.02,
            "\n".join(textwrap.wrap(footer, width=190, replace_whitespace=False)),
            fontsize=10,
            ha="left",
            va="bottom",
        )

    plt.subplots_adjust(left=0.055, right=0.995, top=0.82, bottom=0.12, wspace=0.035, hspace=0.16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def make_plot_pil(
    record: Dict[str, Any],
    output_path: Path,
    show_filenames: bool,
    max_pixels: Optional[int] = None,
) -> None:
    gt_paths = get_gt_paths(record)
    candidate_paths = get_candidate_paths(record)
    if len(gt_paths) != 8 or len(candidate_paths) != 8:
        raise ValueError(
            f"Expected 8 GT and 8 candidate frames, got {len(gt_paths)} GT and {len(candidate_paths)} candidate"
        )

    gt_images = [load_image(path, max_pixels=max_pixels) for path in gt_paths]
    candidate_images = [load_image(path, max_pixels=max_pixels) for path in candidate_paths]
    all_images = gt_images + candidate_images

    tile_w = max(img.width for img in all_images)
    tile_h = max(img.height for img in all_images)
    label_w = 110
    margin = 24
    gap = 8
    row_gap = 36 if show_filenames else 20
    title_gap = 16
    footer_gap = 18

    title_font = load_font(22, bold=True)
    body_font = load_font(18)
    label_font = load_font(22, bold=True)
    small_font = load_font(13)
    footer_font = load_font(16)

    query = str(record.get("query", "")).strip()
    decision = get_decision(record)
    confidence = record.get("qwen_result", {}).get("confidence")
    title = (
        f"Query: {query}\n"
        f"GT: {get_gt_video_id(record)}    Candidate: {get_candidate_video_id(record)}    "
        f"Qwen decision: {decision}"
    )
    if confidence is not None:
        title += f"    Confidence: {confidence}"

    canvas_w = margin * 2 + label_w + (tile_w * 8) + (gap * 7)
    scratch = Image.new("RGB", (canvas_w, 10), "white")
    draw = ImageDraw.Draw(scratch)
    title_lines = wrap_text_to_width(draw, title, title_font, canvas_w - margin * 2)
    title_line_h = title_font.size + 7
    title_h = len(title_lines) * title_line_h

    reason = str(record.get("qwen_result", {}).get("reason") or "").strip()
    footer_text = f"Qwen reason: {reason}" if reason else ""
    footer_lines = wrap_text_to_width(draw, footer_text, footer_font, canvas_w - margin * 2) if footer_text else []
    footer_line_h = footer_font.size + 6
    footer_h = len(footer_lines) * footer_line_h

    filename_h = small_font.size + 8 if show_filenames else 0
    row_h = tile_h + filename_h
    grid_y = margin + title_h + title_gap
    footer_y = grid_y + row_h * 2 + row_gap + footer_gap
    canvas_h = footer_y + footer_h + margin

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    y = margin
    for line in title_lines:
        draw.text((margin, y), line, font=title_font, fill=(0, 0, 0))
        y += title_line_h

    def draw_row(
        images: Sequence[Image.Image],
        paths: Sequence[str],
        row_y: int,
        label: str,
        label_color: Tuple[int, int, int],
    ) -> None:
        draw.text((margin, row_y + tile_h // 2 - label_font.size // 2), label, font=label_font, fill=label_color)
        x = margin + label_w
        for idx, (img, path) in enumerate(zip(images, paths)):
            tile = Image.new("RGB", (tile_w, tile_h), (245, 245, 245))
            fitted = ImageOps.contain(img, (tile_w, tile_h), method=Image.Resampling.LANCZOS)
            tile.paste(fitted, ((tile_w - fitted.width) // 2, (tile_h - fitted.height) // 2))
            canvas.paste(tile, (x, row_y))
            draw.rectangle((x, row_y, x + tile_w - 1, row_y + tile_h - 1), outline=(210, 210, 210), width=1)
            title_text = frame_title(path, idx) if show_filenames else str(idx + 1)
            text_color = label_color
            text_x = x + max(0, (tile_w - text_width(draw, title_text, small_font)) // 2)
            draw.text((text_x, row_y + tile_h + 4), title_text, font=small_font, fill=text_color)
            x += tile_w + gap

    candidate_color = (19, 138, 54) if decision == "accept" else (179, 38, 30)
    draw_row(gt_images, gt_paths, grid_y, "GT", (47, 85, 151))
    draw_row(candidate_images, candidate_paths, grid_y + row_h + row_gap, "Candidate", candidate_color)

    for line in footer_lines:
        draw.text((margin, footer_y), line, font=footer_font, fill=(0, 0, 0))
        footer_y += footer_line_h

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, compress_level=1)


def make_plot_with_backend(
    record: Dict[str, Any],
    output_path: Path,
    dpi: int,
    show_filenames: bool,
    max_pixels: Optional[int],
    backend: str,
) -> None:
    if backend == "matplotlib":
        make_plot(
            record,
            output_path=output_path,
            dpi=dpi,
            show_filenames=show_filenames,
            max_pixels=max_pixels,
        )
        return
    make_plot_pil(
        record,
        output_path=output_path,
        show_filenames=show_filenames,
        max_pixels=max_pixels,
    )


def should_keep(record: Dict[str, Any], decision_filter: Optional[str], query_filter: Optional[str]) -> bool:
    if decision_filter and get_decision(record) != decision_filter:
        return False
    if query_filter and query_filter.lower() not in str(record.get("query", "")).lower():
        return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--decision", choices=["accept", "reject", "none"], default=None)
    parser.add_argument("--query-contains", default=None)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument(
        "--backend",
        choices=["pil", "matplotlib"],
        default=DEFAULT_BACKEND,
        help="Plot renderer. pil is much faster for large ActivityNet frames.",
    )
    parser.add_argument("--show-filenames", action="store_true")
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=None,
        help="Resize each frame in memory to this pixel budget before plotting.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    total = count_jsonl(input_path)
    for record_idx, record in enumerate(tqdm(read_jsonl(input_path), total=total, desc="plot TVR galleries")):
        if record_idx < args.start_index:
            continue
        if args.limit is not None and written >= args.limit:
            break
        if not should_keep(record, args.decision, args.query_contains):
            continue

        query_id = slugify(get_query_id(record), max_len=70)
        candidate_id = slugify(get_candidate_video_id(record), max_len=70)
        decision = slugify(get_decision(record), max_len=30)
        filename = f"{record_idx:06d}_{query_id}_{candidate_id}_{decision}.png"
        output_path = output_dir / filename
        if output_path.exists() and not args.overwrite:
            continue

        make_plot_with_backend(
            record,
            output_path=output_path,
            dpi=args.dpi,
            show_filenames=args.show_filenames,
            max_pixels=args.max_pixels,
            backend=args.backend,
        )
        written += 1

    print(f"wrote {written} plot(s) to {output_dir}")


if __name__ == "__main__":
    main()
