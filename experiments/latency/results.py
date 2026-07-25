"""Atomic append and wide-QPS export for latency benchmark workers."""
from __future__ import annotations

import csv
import fcntl
from pathlib import Path


DETAIL_FIELDS = [
    "corpus", "method", "backend", "device", "query_bsz", "gallery_encoder_bsz",
    "search_chunk_videos", "reprs_per_video", "query_emb_time_ms",
    "search_time_ms", "e2e_latency_ms", "e2e_qps", "gallery_prepare_sec",
    "index_build_sec", "nlist", "nprobe", "hnsw_m", "ef_search",
    "top10_overlap_vs_exact", "checkpoint", "status", "error",
]

METHOD_ORDER = [
    "CLIP4Clip", "AMDNet", "GMMFormer",
    "GMMFormerV2", "HLFormer", "Holmes", "DreamPRVR", "BOA", "DL-DKD",
    "MSC-PRVR", "MS-SL", "BGMNet",
]


def append_detail(output: Path, row: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    _migrate_detail_schema(output)
    normalized = {field: row.get(field, "") for field in DETAIL_FIELDS}
    with output.open("a+", newline="", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0, 2)
        writer = csv.DictWriter(handle, fieldnames=DETAIL_FIELDS)
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerow(normalized)
        handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _migrate_detail_schema(output: Path) -> None:
    """Upgrade a prior detail CSV without losing valid historical rows.

    The latency worker schema gained gallery/search-chunk metadata.  Earlier
    rows have the old header length, whereas rows written during an interrupted
    upgrade already have the new value count but the old header.  Both forms
    are unambiguous and can be normalised safely before appending.
    """
    if not output.exists() or output.stat().st_size == 0:
        return
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows or rows[0] == DETAIL_FIELDS:
        return
    old_fields = rows[0]
    converted: list[dict] = []
    for values in rows[1:]:
        if len(values) == len(DETAIL_FIELDS):
            # Values were emitted using the new writer but retained an old
            # header; map them positionally to the new schema.
            converted.append(dict(zip(DETAIL_FIELDS, values)))
        else:
            converted.append(dict(zip(old_fields, values)))
    temporary = output.with_suffix(".csv.migrating")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DETAIL_FIELDS)
        writer.writeheader()
        for value in converted:
            writer.writerow({field: value.get(field, "") for field in DETAIL_FIELDS})
    temporary.replace(output)


def export_wide_qps(detail_csv: Path, qps_csv: Path) -> None:
    if not detail_csv.exists():
        return
    _migrate_detail_schema(detail_csv)
    latest: dict[tuple[str, str], dict] = {}
    with detail_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") == "ok":
                latest[(row["corpus"], row["method"])] = row
    corpora = sorted({key[0] for key in latest}, key=lambda value: int(value))
    qps_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = qps_csv.with_suffix(".csv.partial")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Corpus", *METHOD_ORDER])
        writer.writeheader()
        for corpus in corpora:
            row = {"Corpus": f"{int(corpus) // 1000}K" if int(corpus) < 1_000_000 else f"{int(corpus) // 1_000_000}M"}
            for method in METHOD_ORDER:
                value = latest.get((corpus, method), {}).get("e2e_qps", "")
                row[method] = value
            writer.writerow(row)
    temporary.replace(qps_csv)
