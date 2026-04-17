#!/bin/bash
#SBATCH --job-name=llb-gemma4
#SBATCH --partition=gpu-preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --array=0-99
#SBATCH --output=slurm_logs/gemma4_%A_%a.out
#SBATCH --error=slurm_logs/gemma4_%A_%a.err
#SBATCH --requeue

set -euo pipefail
mkdir -p slurm_logs

echo "Host: $(hostname)"
echo "Array job ${SLURM_ARRAY_JOB_ID}, task ${SLURM_ARRAY_TASK_ID}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

SCRATCH=/scratch3/workspace/edmondcunnin_umass_edu-siple
export OLLAMA_MODELS="${SCRATCH}/ollama_models"
export LD_LIBRARY_PATH="${SCRATCH}/ollama/lib/ollama:${LD_LIBRARY_PATH:-}"
OLLAMA="${SCRATCH}/ollama/bin/ollama"

OLLAMA_MODEL="gemma4:e4b"
CACHE_DIR="model_cache/gemma4_e4b"
N_TASKS=100
BASE_CONFIG="${CACHE_DIR}/config.json"

cd /home/edmondcunnin_umass_edu/Large-Language-Bayes-Model

mkdir -p "${CACHE_DIR}"

if [ ! -f "${BASE_CONFIG}" ]; then
  cat > "${BASE_CONFIG}" <<'CONFIGEOF'
{
  "text": "I have a bunch of coin flips. What's the bias?",
  "data": {"flips": [0, 1, 0, 1, 1, 0]},
  "targets": ["true_bias"],
  "api_url": "http://127.0.0.1:11434/api/generate",
  "api_model": "gemma4:e4b",
  "n_models": 10000,
  "random_seed": 42
}
CONFIGEOF
  echo "Wrote base config to ${BASE_CONFIG}"
fi

OLLAMA_PORT=$((11434 + SLURM_ARRAY_TASK_ID))
export OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}"

"${OLLAMA}" serve &
OLLAMA_PID=$!

cleanup() {
  echo "Stopping Ollama (pid ${OLLAMA_PID})"
  kill "${OLLAMA_PID}" 2>/dev/null || true
  wait "${OLLAMA_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for Ollama to become ready on port ${OLLAMA_PORT}..."
for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${OLLAMA_PORT}/api/tags" > /dev/null 2>&1; then
    echo "Ollama ready after ${i}s"
    break
  fi
  sleep 1
done

"${OLLAMA}" pull "${OLLAMA_MODEL}"

TASK_CONFIG="/tmp/gen_gemma4_task${SLURM_ARRAY_TASK_ID}.json"
python3 -c "
import json
with open('${BASE_CONFIG}') as f:
    cfg = json.load(f)
cfg['api_url'] = 'http://127.0.0.1:${OLLAMA_PORT}/api/generate'
with open('${TASK_CONFIG}', 'w') as f:
    json.dump(cfg, f, indent=2)
"

echo "Generating chunk: task_id=${SLURM_ARRAY_TASK_ID}, n_tasks=${N_TASKS}"
uv run python -m generate_models \
  --config "${TASK_CONFIG}" \
  --cache-dir "${CACHE_DIR}" \
  generate-chunk \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --n-tasks "${N_TASKS}"

echo "Task ${SLURM_ARRAY_TASK_ID} complete"
