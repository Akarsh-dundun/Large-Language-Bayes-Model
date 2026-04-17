#!/usr/bin/env bash
# Small real-LLM smoke run over every FEMA task plus the coin task.
#
# Starts a local Ollama server bound to a private port, runs
# generate_and_sample.py (5 models per task, reduced MCMC) and evaluate.py
# against each task, and writes artifacts into experiment_results/smoke/.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

SCRATCH="/scratch3/workspace/edmondcunnin_umass_edu-siple"
OLLAMA_BIN="${SCRATCH}/ollama/bin/ollama"
export OLLAMA_MODELS="${SCRATCH}/ollama_models"
export LD_LIBRARY_PATH="${SCRATCH}/ollama/lib/ollama:${LD_LIBRARY_PATH:-}"

OLLAMA_PORT="${OLLAMA_PORT:-11555}"
export OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}"
API_URL="http://127.0.0.1:${OLLAMA_PORT}/api/generate"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5-coder:latest}"
LLM_CONFIG="${LLM_CONFIG:-llm_configs/qwen25_coder.json}"
LLM_NAME="${LLM_NAME:-qwen25_coder}"
SHARD_COUNT="${SHARD_COUNT:-5}"

SMOKE_DIR="${REPO_ROOT}/experiment_results/smoke"
mkdir -p "${SMOKE_DIR}"

TASKS=(
  "tasks/hurricane_eal_counties.json"
  "tasks/tornado_counts_plains.json"
  "tasks/earthquake_frequency_west.json"
  "tasks/wildfire_eal_west.json"
  "tasks/inland_flood_eal.json"
)

echo "Starting Ollama on ${OLLAMA_HOST}"
"${OLLAMA_BIN}" serve > "${SMOKE_DIR}/ollama.log" 2>&1 &
OLLAMA_PID=$!

cleanup() {
  echo "Stopping Ollama (pid ${OLLAMA_PID})"
  kill "${OLLAMA_PID}" 2>/dev/null || true
  wait "${OLLAMA_PID}" 2>/dev/null || true
}
trap cleanup EXIT

for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${OLLAMA_PORT}/api/tags" > /dev/null 2>&1; then
    echo "Ollama ready after ${i}s"
    break
  fi
  sleep 1
done

"${OLLAMA_BIN}" pull "${OLLAMA_MODEL}"

for task_path in "${TASKS[@]}"; do
  task_name="$(basename "${task_path}" .json)"
  out_dir="${SMOKE_DIR}/${task_name}/${LLM_NAME}"
  mkdir -p "${out_dir}"

  echo
  echo "=============================================="
  echo "Task: ${task_name}"
  echo "=============================================="

  uv run python -m generate_and_sample \
    --task "${task_path}" \
    --llm-config "${LLM_CONFIG}" \
    --api-url "${API_URL}" \
    --output-dir "${out_dir}" \
    --shard-start 0 \
    --shard-count "${SHARD_COUNT}" \
    --seed 42 \
    --mcmc-num-warmup 100 \
    --mcmc-num-samples 200 \
    --log-marginal-num-inner 3 \
    --log-marginal-num-outer 16 \
    --loo-num-warmup 20 \
    --loo-num-samples 40 \
    --loo-num-inner 8

  uv run python -m evaluate \
    --task "${task_path}" \
    --sample-dir "${out_dir}" \
    --llm-name "${LLM_NAME}" \
    --ks 3 5 \
    --output-jsonl "${SMOKE_DIR}/${task_name}/metrics.jsonl" \
    --output-full-json "${SMOKE_DIR}/${task_name}/metrics_full.json"
done

echo
echo "Smoke run complete. Artifacts in ${SMOKE_DIR}"
