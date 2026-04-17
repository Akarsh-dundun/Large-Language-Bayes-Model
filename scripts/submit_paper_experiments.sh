#!/usr/bin/env bash
# Submit the paper sweep across the (task, LLM) matrix.
#
# Usage:
#   bash scripts/submit_paper_experiments.sh --all
#   bash scripts/submit_paper_experiments.sh --only task=hurricane_eal_counties,llm=qwen25_coder
#   bash scripts/submit_paper_experiments.sh --only task=tornado_counts_plains,llm=gemma4_e4b --smoke
#
# For each selected cell, submits Stage A on gpu-preempt and Stage B on
# cpu-preempt with --dependency=afterany on Stage A. All submitted job ids
# land in experiment_results/paper/_submissions/<timestamp>.txt.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

ALL_TASKS=(
  "hurricane_eal_counties"
  "tornado_counts_plains"
  "earthquake_frequency_west"
  "wildfire_eal_west"
  "inland_flood_eal"
)

# Map LLM short name -> (config path, ollama model tag).
declare -A LLM_CONFIG=(
  ["qwen25_coder"]="llm_configs/qwen25_coder.json"
  ["gemma4_e4b"]="llm_configs/gemma4.json"
  ["llama32"]="llm_configs/llama32.json"
)

declare -A LLM_OLLAMA=(
  ["qwen25_coder"]="qwen2.5-coder:latest"
  ["gemma4_e4b"]="gemma4:e4b"
  ["llama32"]="llama3.2:latest"
)

ALL_LLMS=(qwen25_coder gemma4_e4b llama32)

# --- argument parsing ---
SUBMIT_ALL=0
SMOKE=0
ONLY_PAIRS=()

usage() {
  cat <<EOF
Usage:
  --all                              submit every (task, llm) cell
  --only task=<task>,llm=<llm>       submit only the specified cell (repeatable)
  --smoke                            tiny target (5 valid codes) for quick validation
  -h, --help                         show this help

Examples:
  $0 --all
  $0 --only task=hurricane_eal_counties,llm=qwen25_coder
  $0 --only task=tornado_counts_plains,llm=qwen25_coder --smoke
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all) SUBMIT_ALL=1; shift ;;
    --only)
      shift
      ONLY_PAIRS+=("$1")
      shift
      ;;
    --smoke) SMOKE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ $SUBMIT_ALL -eq 0 && ${#ONLY_PAIRS[@]} -eq 0 ]]; then
  echo "error: pass --all or one or more --only task=...,llm=..."
  usage
  exit 1
fi

# Build the list of (task, llm) cells to submit.
SELECTED=()
if [[ $SUBMIT_ALL -eq 1 ]]; then
  for t in "${ALL_TASKS[@]}"; do
    for l in "${ALL_LLMS[@]}"; do
      SELECTED+=("${t}:${l}")
    done
  done
else
  for pair in "${ONLY_PAIRS[@]}"; do
    t=$(echo "$pair" | sed -n 's/.*task=\([^,]*\).*/\1/p')
    l=$(echo "$pair" | sed -n 's/.*llm=\([^,]*\).*/\1/p')
    if [[ -z "$t" || -z "$l" ]]; then
      echo "bad --only spec: $pair"; exit 1
    fi
    SELECTED+=("${t}:${l}")
  done
fi

if [[ $SMOKE -eq 1 ]]; then
  CELL_VALID_TARGET=5
  SHARD_VALID_TARGET=5
  SHARD_MAX_INDICES=20
  STAGE_A_ARRAY="0-0"
  STAGE_B_ARRAY="0-0"
  ARRAY_COUNT=1
  RESULTS_PARENT="experiment_results/paper_smoke"
else
  CELL_VALID_TARGET=10000
  SHARD_VALID_TARGET=100
  SHARD_MAX_INDICES=150
  STAGE_A_ARRAY="0-99"
  STAGE_B_ARRAY="0-99"
  ARRAY_COUNT=100
  RESULTS_PARENT="experiment_results/paper"
fi

mkdir -p "${RESULTS_PARENT}/_submissions" slurm_logs
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
SUBMISSION_LOG="${RESULTS_PARENT}/_submissions/${TIMESTAMP}.txt"
echo "timestamp,task,llm,stage_a_jobid,stage_b_jobid,cell_dir" > "${SUBMISSION_LOG}"

echo "Selected cells:"
for cell in "${SELECTED[@]}"; do
  echo "  ${cell}"
done
echo "Smoke mode: ${SMOKE}; cell_target=${CELL_VALID_TARGET}; array=${STAGE_A_ARRAY}"
echo "Submission log: ${SUBMISSION_LOG}"
echo

CANCEL_LINE=""

for cell in "${SELECTED[@]}"; do
  task="${cell%%:*}"
  llm="${cell##*:}"
  task_path="tasks/${task}.json"
  llm_config="${LLM_CONFIG[${llm}]:-}"
  ollama_model="${LLM_OLLAMA[${llm}]:-}"
  if [[ -z "${llm_config}" || -z "${ollama_model}" ]]; then
    echo "error: unknown LLM ${llm}; allowed: ${!LLM_CONFIG[*]}"; exit 1
  fi
  if [[ ! -f "${task_path}" ]]; then
    echo "error: missing task file ${task_path}"; exit 1
  fi
  if [[ ! -f "${llm_config}" ]]; then
    echo "error: missing LLM config ${llm_config}"; exit 1
  fi

  cell_dir="${RESULTS_PARENT}/${task}/${llm}"
  mkdir -p "${cell_dir}/_manifest"

  echo "============================================="
  echo "Submitting cell task=${task} llm=${llm}"
  echo "  cell_dir=${cell_dir}"
  echo "  llm_config=${llm_config}"
  echo "  ollama_model=${ollama_model}"

  STAGE_A_JOBID=$(sbatch \
    --parsable \
    --array="${STAGE_A_ARRAY}" \
    --export=ALL,TASK_PATH="${task_path}",LLM_CONFIG_PATH="${llm_config}",OLLAMA_MODEL="${ollama_model}",CELL_DIR="${cell_dir}",SHARD_MAX_INDICES="${SHARD_MAX_INDICES}",SHARD_VALID_TARGET="${SHARD_VALID_TARGET}",CELL_VALID_TARGET="${CELL_VALID_TARGET}" \
    slurm/gen_codes_gpu.sbatch)

  STAGE_B_JOBID=$(sbatch \
    --parsable \
    --array="${STAGE_B_ARRAY}" \
    --dependency=afterany:"${STAGE_A_JOBID}" \
    --export=ALL,TASK_PATH="${task_path}",CELL_DIR="${cell_dir}",ARRAY_COUNT="${ARRAY_COUNT}" \
    slurm/sample_cpu.sbatch)

  echo "  Stage A job id: ${STAGE_A_JOBID}"
  echo "  Stage B job id: ${STAGE_B_JOBID} (depends on Stage A)"
  echo "${TIMESTAMP},${task},${llm},${STAGE_A_JOBID},${STAGE_B_JOBID},${cell_dir}" >> "${SUBMISSION_LOG}"
  CANCEL_LINE="${CANCEL_LINE} ${STAGE_A_JOBID} ${STAGE_B_JOBID}"
done

echo
echo "All submissions recorded in ${SUBMISSION_LOG}"
echo "Abort everything with: scancel${CANCEL_LINE}"
echo "Watch live with: bash scripts/watch_progress.sh ${RESULTS_PARENT}"
