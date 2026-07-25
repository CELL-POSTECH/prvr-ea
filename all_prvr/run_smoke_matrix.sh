#!/usr/bin/env bash
# Four-GPU smoke matrix: each job runs one epoch and then test-split eval.
# Logs and a tab-separated PASS/FAIL ledger are written under smoke_logs/.
set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="${PRVR_PROJECT_ROOT:-$(cd "$script_dir/.." && pwd)}"
root="${PRVR_ROOT:-$project_root/all_prvr}"
log_root="$root/smoke_logs"
mkdir -p "$log_root"
status_file="$log_root/status.tsv"
printf 'timestamp\tmodel\tcollection\tfeature\tgpu\tstatus\tlog\n' > "$status_file"

models=(amd boa gmm gmmv2 hlformer dream holmes bgm mssl dldkd)
collections=(activitynet tvr charades msrvtt)
modes=(resnet clip)
tasks=()
for model in "${models[@]}"; do
  for collection in "${collections[@]}"; do
    for mode in "${modes[@]}"; do
      tasks+=("$model $collection $mode")
    done
  done
done
for collection in "${collections[@]}"; do
  tasks+=("msc $collection clip")
done

run_task() {
  local task=$1 gpu=$2 model collection mode log status
  read -r model collection mode <<< "$task"
  log="$log_root/${model}_${collection}_${mode}.log"
  if bash "$root/run_smoke.sh" "$model" "$collection" "$mode" "$gpu" >"$log" 2>&1; then
    status=PASS
  else
    status=FAIL
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -u +%FT%TZ)" "$model" "$collection" "$mode" "$gpu" "$status" "$log" \
    >> "$status_file"
  printf '[%s] %s %s %s: %s\n' "$gpu" "$model" "$collection" "$mode" "$status"
}

worker() {
  local gpu=$1 index=$2
  while (( index < ${#tasks[@]} )); do
    run_task "${tasks[index]}" "$gpu"
    index=$((index + 4))
  done
}

for gpu in 0 1 2 3; do
  worker "$gpu" "$gpu" &
done
wait
awk -F '\t' 'NR == 1 || $6 != "PASS"' "$status_file"
