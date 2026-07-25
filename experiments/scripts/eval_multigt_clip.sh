#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MULTIGT=1
source "$SCRIPT_DIR/common.sh"
[[ "${1:-}" != "--help" && "${1:-}" != "-h" ]] || { usage eval clip; exit 0; }
[[ $# -ge 2 && $# -le 3 ]] || { usage eval clip >&2; exit 2; }
DATASET="$(normalize_dataset "$1")" || { echo "unknown dataset: $1" >&2; exit 2; }
EXPORT_DONE=0
export_results() {
    [[ "$EXPORT_DONE" -eq 1 ]] && return 0
    EXPORT_DONE=1
    echo "[$(date -u +%FT%TZ)] exporting CLIP multiGT recall CSVs for dataset=$DATASET"
    "$PYTHON_BIN" "$SCRIPT_DIR/export_recall_csv.py" --feature clip --multi-gt --dataset "$DATASET"
}
trap export_results EXIT
set +e
run_matrix eval clip "$@"
EVAL_RC=$?
set -e
export_results
trap - EXIT
exit "$EVAL_RC"
