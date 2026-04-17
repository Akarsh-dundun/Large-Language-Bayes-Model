#!/usr/bin/env bash
# Live progress dashboard for the paper sweep.
#
# Walks experiment_results/<root>/<task>/<llm>/ and prints, every interval,
# a one-row-per-cell table of distinct codes generated, samples completed,
# samples failed, and the age of the most recent activity per stage.
#
# Usage:
#   bash scripts/watch_progress.sh                              # default root: experiment_results/paper
#   bash scripts/watch_progress.sh experiment_results/paper_smoke
#   bash scripts/watch_progress.sh experiment_results/paper 30  # 30s interval

set -u

ROOT="${1:-experiment_results/paper}"
INTERVAL="${2:-60}"

if [[ ! -d "${ROOT}" ]]; then
  echo "no such directory: ${ROOT}"
  exit 1
fi

print_table() {
  local now
  now=$(date -u +%s)
  printf "\n=== %s (root=%s, interval=%ss) ===\n" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${ROOT}" "${INTERVAL}"
  printf "%-30s %-15s %10s %10s %10s %10s %10s %15s\n" \
    "task" "llm" "codes" "sam_ok" "sam_part" "sam_fail" "manifests" "last_code_age_s"
  printf "%-30s %-15s %10s %10s %10s %10s %10s %15s\n" \
    "----" "---" "-----" "------" "--------" "--------" "---------" "---------------"

  for task_dir in "${ROOT}"/*/; do
    [[ -d "${task_dir}" ]] || continue
    task_name=$(basename "${task_dir}")
    if [[ "${task_name}" == _* ]]; then
      continue
    fi
    for llm_dir in "${task_dir}"*/; do
      [[ -d "${llm_dir}" ]] || continue
      llm_name=$(basename "${llm_dir}")

      codes_count=$(ls "${llm_dir}codes/"code_*.code.json 2>/dev/null | wc -l)
      sam_ok=0
      sam_partial=0
      sam_fail=0
      if [[ -d "${llm_dir}samples" ]]; then
        for meta in "${llm_dir}samples/"sample_*.meta.json; do
          [[ -e "${meta}" ]] || break
          status=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('status',''))" "${meta}" 2>/dev/null)
          case "${status}" in
            ok) sam_ok=$((sam_ok+1));;
            partial|inference_partial) sam_partial=$((sam_partial+1));;
            *) sam_fail=$((sam_fail+1));;
          esac
        done
      fi
      manifests=$(ls "${llm_dir}_manifest/"shard_*.json 2>/dev/null | wc -l)

      last_code_age="--"
      newest=$(ls -t "${llm_dir}codes/"code_*.code.json 2>/dev/null | head -1)
      if [[ -n "${newest}" ]]; then
        mtime=$(stat -c %Y "${newest}" 2>/dev/null || echo "")
        if [[ -n "${mtime}" ]]; then
          last_code_age=$((now - mtime))
        fi
      fi

      printf "%-30s %-15s %10s %10s %10s %10s %10s %15s\n" \
        "${task_name:0:30}" "${llm_name:0:15}" \
        "${codes_count}" "${sam_ok}" "${sam_partial}" "${sam_fail}" \
        "${manifests}" "${last_code_age}"
    done
  done
}

while true; do
  print_table
  if [[ "${INTERVAL}" -le 0 ]]; then
    break
  fi
  sleep "${INTERVAL}"
done
