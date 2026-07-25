#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
[[ "${1:-}" != "--help" && "${1:-}" != "-h" ]] || { usage eval resnet; exit 0; }
[[ $# -ge 2 && $# -le 3 ]] || { usage eval resnet >&2; exit 2; }
DATASET="$(normalize_dataset "$1")" || { echo "unknown dataset: $1" >&2; exit 2; }

EXPORT_DONE=0
export_results() {
    [[ "$EXPORT_DONE" -eq 1 ]] && return 0
    EXPORT_DONE=1
    echo "[$(date -u +%FT%TZ)] exporting ResNet recall CSVs for dataset=$DATASET"
    "$PYTHON_BIN" "$SCRIPT_DIR/export_recall_csv.py" --feature resnet --dataset "$DATASET"
}
trap export_results EXIT

# Always refresh the compact comparison-table CSVs, including when one
# condition fails and run_matrix returns non-zero after continuing.
set +e
run_matrix eval resnet "$@"
EVAL_RC=$?
set -e
export_results
trap - EXIT
exit "$EVAL_RC"
