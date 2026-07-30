#!/usr/bin/env python3
"""Replace histogram titles in the manuscript PDF with a consistent Arial face.

The source page remains vector content. A transparent vector overlay hides the
old title and places the replacement at the same title centre and baseline.
"""

import argparse
import shutil
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib import patheffects
from matplotlib.patches import Rectangle
from matplotlib.textpath import TextPath
from pypdf import PdfReader, PdfWriter


PAGE_WIDTH_PT = 1584.0
PAGE_HEIGHT_PT = 273.6
TITLE_FONT_SIZE_PT = 22.0
TITLE_BASELINE_Y_PT = 224.88
# x positions are the original PDF title baselines, extracted from its vector
# text. The original title strings establish each existing title's centre.
TITLE_LAYOUT = (
    (183.4696638446, "ActivityNet", "ActivityNet Captions"),
    (610.8742629482, "TVR", "TVR"),
    (982.6304245518, "Charades", "Charades"),
    (1378.8943986554, "MSR-VTT", "MSR-VTT"),
)


def text_width(text: str, font: font_manager.FontProperties) -> float:
    return TextPath((0, 0), text, prop=font).get_extents().width


def make_overlay(path: Path, font_path: Path, stroke_width: float):
    font = font_manager.FontProperties(fname=str(font_path), size=TITLE_FONT_SIZE_PT)

    fig = plt.figure(figsize=(PAGE_WIDTH_PT / 72.0, PAGE_HEIGHT_PT / 72.0), frameon=False)
    fig.patch.set_alpha(0.0)
    axis = fig.add_axes((0, 0, 1, 1))
    axis.set_xlim(0, PAGE_WIDTH_PT)
    axis.set_ylim(0, PAGE_HEIGHT_PT)
    axis.set_axis_off()
    text_kwargs = {}
    if stroke_width > 0:
        # A sub-point stroke preserves the Arial glyph shapes while making the
        # replacement just perceptibly heavier than regular Arial.
        text_kwargs["path_effects"] = [
            patheffects.withStroke(linewidth=stroke_width, foreground="black")
        ]
    for old_x, old_title, replacement in TITLE_LAYOUT:
        old_width = text_width(old_title, font)
        new_width = text_width(replacement, font)
        title_center = old_x + old_width / 2.0
        new_x = title_center - new_width / 2.0
        # Erase only this title's original bounding region. The broader first
        # rectangle accommodates "ActivityNet Captions" without touching the
        # plotting area or neighbouring titles.
        left = min(old_x, new_x) - 3.0
        right = max(old_x + old_width, new_x + new_width) + 3.0
        axis.add_patch(Rectangle((left, 220.0), right - left, 31.0,
                                 facecolor="white", edgecolor="none", zorder=1))
        axis.text(new_x, TITLE_BASELINE_Y_PT, replacement, fontproperties=font,
                  color="black", ha="left", va="baseline", zorder=2, **text_kwargs)
    fig.savefig(path, format="pdf", transparent=True, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--font", type=Path, required=True)
    parser.add_argument("--stroke-width", type=float, default=0.0,
                        help="Extra title stroke in PDF points; 0 keeps regular Arial.")
    parser.add_argument("--backup", type=Path, default=None)
    args = parser.parse_args()
    if not args.font.is_file():
        raise FileNotFoundError(args.font)

    backup = args.backup or args.pdf.with_name(args.pdf.stem + "_original.pdf")
    if not args.pdf.is_file() and not backup.is_file():
        raise FileNotFoundError(f"Neither target nor backup exists: {args.pdf}, {backup}")
    if not backup.exists():
        shutil.copy2(args.pdf, backup)

    with tempfile.TemporaryDirectory(prefix="histogram_title_overlay_") as directory:
        overlay = Path(directory) / "overlay.pdf"
        make_overlay(overlay, args.font, args.stroke_width)
        source = PdfReader(str(backup))
        overlay_page = PdfReader(str(overlay)).pages[0]
        source.pages[0].merge_page(overlay_page, over=True)
        writer = PdfWriter()
        for page in source.pages:
            writer.add_page(page)
        with args.pdf.open("wb") as handle:
            writer.write(handle)
    print(f"updated: {args.pdf}")
    print(f"backup:  {backup}")


if __name__ == "__main__":
    main()
