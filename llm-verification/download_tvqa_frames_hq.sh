#!/usr/bin/env bash
set -euo pipefail

BASE_URL="https://nlp.cs.unc.edu/data/jielei/tvqa/frames_hq"
DATA_ROOT="${DATA_ROOT:-datasets}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-${DATA_ROOT}/tvr/raw_frames_download/tvqa_video_frames_fps3_hq}"
TARGET_DIR="${TARGET_DIR:-${DATA_ROOT}/tvr/raw_frames/frames_hq}"
EXTRACT_DIR="${EXTRACT_DIR:-${DATA_ROOT}/tvr/raw_frames_download/extracted}"

parts=(
  aa ab ac ad ae af ag ah ai aj ak al am an ao ap aq ar as at au av aw ax ay az
  ba bb bc bd be bf bg bh bi
)

mkdir -p "$DOWNLOAD_DIR" "$TARGET_DIR" "$EXTRACT_DIR"
cd "$DOWNLOAD_DIR"

echo "[1/5] Downloading checksum"
wget -c "${BASE_URL}/tvqa_video_frames_fps3_hq.checksum.txt"

echo "[2/5] Downloading split tar.gz parts"
for part in "${parts[@]}"; do
  wget -c "${BASE_URL}/tvqa_video_frames_fps3_hq.tar.gz.${part}"
done

echo "[3/5] Validating md5 checksums"
md5sum -c tvqa_video_frames_fps3_hq.checksum.txt

echo "[4/5] Extracting archive"
cat tvqa_video_frames_fps3_hq.tar.gz.* | tar xz -C "$EXTRACT_DIR"

echo "[5/5] Normalizing extracted directory into: $TARGET_DIR"
source_dir=""
if [[ -d "$EXTRACT_DIR/frames_hq" ]]; then
  source_dir="$EXTRACT_DIR/frames_hq"
elif [[ -d "$EXTRACT_DIR/tvqa_video_frames_fps3_hq/frames_hq" ]]; then
  source_dir="$EXTRACT_DIR/tvqa_video_frames_fps3_hq/frames_hq"
elif compgen -G "$EXTRACT_DIR/*_frames" > /dev/null; then
  source_dir="$EXTRACT_DIR"
elif compgen -G "$EXTRACT_DIR/tvqa_video_frames_fps3_hq/*_frames" > /dev/null; then
  source_dir="$EXTRACT_DIR/tvqa_video_frames_fps3_hq"
else
  echo "Could not find extracted *_frames directories under $EXTRACT_DIR" >&2
  echo "Inspect with: find '$EXTRACT_DIR' -maxdepth 3 -type d | head -50" >&2
  exit 1
fi

for show_dir in "$source_dir"/*_frames; do
  [[ -d "$show_dir" ]] || continue
  name="$(basename "$show_dir")"
  if [[ -e "$TARGET_DIR/$name" ]]; then
    echo "Skipping existing $TARGET_DIR/$name"
  else
    mv "$show_dir" "$TARGET_DIR/"
  fi
done

echo "Done. Final layout:"
find "$TARGET_DIR" -maxdepth 1 -type d -name "*_frames" -printf "  %f\n" | sort
echo
echo "Optional cleanup after verifying frames:"
echo "  rm -rf '$DOWNLOAD_DIR' '$EXTRACT_DIR'"
