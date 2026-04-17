#!/bin/bash
#SBATCH --job-name=llb-gen
#SBATCH --partition=gpu-preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-9
#SBATCH --output=slurm_logs/gen_%A_%a.out
#SBATCH --error=slurm_logs/gen_%A_%a.err
#SBATCH --requeue

set -euo pipefail
mkdir -p slurm_logs

echo "Host: $(hostname)"
echo "Array job ${SLURM_ARRAY_JOB_ID}, task ${SLURM_ARRAY_TASK_ID}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

# ─── Configuration ───────────────────────────────────────────────
# Set these before submitting or override via environment variables.
CONFIG_JSON="${CONFIG_JSON:-generation_config.json}"
CACHE_DIR="${CACHE_DIR:-model_cache}"
N_TASKS="${N_TASKS:-${SLURM_ARRAY_TASK_COUNT:-10}}"
OLLAMA_MODEL="${OLLAMA_MODEL:-gemma4:e4b}"

cd /home/edmondcunnin_umass_edu/Large-Language-Bayes-Model

# ─── Start Ollama on this node's GPU ─────────────────────────────
OLLAMA_PORT=$((11434 + SLURM_ARRAY_TASK_ID))
export OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}"

ollama serve &
OLLAMA_PID=$!

cleanup() {
  echo "Stopping Ollama (pid ${OLLAMA_PID})"
  kill "${OLLAMA_PID}" 2>/dev/null || true
  wait "${OLLAMA_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for Ollama to become ready on port ${OLLAMA_PORT}..."
for i in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:${OLLAMA_PORT}/api/tags" > /dev/null 2>&1; then
    echo "Ollama ready after ${i}s"
    break
  fi
  sleep 1
done

ollama pull "${OLLAMA_MODEL}"

# ─── Update config to point at this node's Ollama ────────────────
# The generation config JSON has api_url pointing to localhost:11434.
# Override it to use the per-task port via an environment variable
# that generate_models.py can pick up, OR just rewrite the config
# with the correct port. We use a temp config per task.
TASK_CONFIG="/tmp/gen_config_task${SLURM_ARRAY_TASK_ID}.json"
python3 -c "
import json, sys
with open('${CONFIG_JSON}') as f:
    cfg = json.load(f)
cfg['api_url'] = 'http://127.0.0.1:${OLLAMA_PORT}/api/generate'
with open('${TASK_CONFIG}', 'w') as f:
    json.dump(cfg, f, indent=2)
"

# ─── Generate this chunk ─────────────────────────────────────────
echo "Generating chunk: task_id=${SLURM_ARRAY_TASK_ID}, n_tasks=${N_TASKS}"
uv run python -m generate_models \
  --config "${TASK_CONFIG}" \
  --cache-dir "${CACHE_DIR}" \
  generate-chunk \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --n-tasks "${N_TASKS}"

echo "Task ${SLURM_ARRAY_TASK_ID} complete"
