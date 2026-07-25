#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
[[ "${1:-}" != "--help" && "${1:-}" != "-h" ]] || { usage train clip; exit 0; }
[[ $# -ge 2 && $# -le 3 ]] || { usage train clip >&2; exit 2; }
run_matrix train clip "$@"
