#!/bin/bash
#SBATCH --job-name=llb-codes
#SBATCH --partition=gpu-preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --array=0-99
#SBATCH --output=slurm_logs/codes_%A_%a.out
#SBATCH --error=slurm_logs/codes_%A_%a.err
#SBATCH --requeue

# Stage A: distinct-valid LLM code generation for one (task, LLM) cell.
#
# Each array task starts a private Ollama on a unique port, generates up to
# SHARD_MAX_INDICES codes from its slot range, and stops as soon as either
# this shard has SHARD_VALID_TARGET new (non-duplicate) hashes or the cell
# as a whole has CELL_VALID_TARGET distinct codes. Resume on preemption is
# automatic via the hash registry under <CELL_DIR>/codes/.
#
# Required environment variables (set at submission time):
#   TASK_PATH            tasks/<name>.json
#   LLM_CONFIG_PATH      llm_configs/<name>.json
#   OLLAMA_MODEL         model tag for `ollama pull`
#   CELL_DIR             experiment_results/paper/<task>/<llm>/
# Optional:
#   SHARD_MAX_INDICES    default 150
#   SHARD_VALID_TARGET   default 100
#   CELL_VALID_TARGET    default 10000
#   SEED                 default 42

set -euo pipefail
mkdir -p slurm_logs

echo "Host: $(hostname)"
echo "Array job ${SLURM_ARRAY_JOB_ID}, task ${SLURM_ARRAY_TASK_ID}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

: "${TASK_PATH:?TASK_PATH must be set}"
: "${LLM_CONFIG_PATH:?LLM_CONFIG_PATH must be set}"
: "${OLLAMA_MODEL:?OLLAMA_MODEL must be set}"
: "${CELL_DIR:?CELL_DIR must be set}"

SHARD_MAX_INDICES="${SHARD_MAX_INDICES:-150}"
SHARD_VALID_TARGET="${SHARD_VALID_TARGET:-100}"
CELL_VALID_TARGET="${CELL_VALID_TARGET:-10000}"
SEED="${SEED:-42}"

SCRATCH=/scratch3/workspace/edmondcunnin_umass_edu-siple
export OLLAMA_MODELS="${SCRATCH}/ollama_models"
export LD_LIBRARY_PATH="${SCRATCH}/ollama/lib/ollama:${LD_LIBRARY_PATH:-}"
OLLAMA="${SCRATCH}/ollama/bin/ollama"

cd /home/edmondcunnin_umass_edu/Large-Language-Bayes-Model

mkdir -p "${CELL_DIR}"

SHARD_START=$((SLURM_ARRAY_TASK_ID * SHARD_MAX_INDICES))

OLLAMA_PORT=$((11434 + SLURM_ARRAY_TASK_ID))
export OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}"

"${OLLAMA}" serve > "${CELL_DIR}/_manifest/ollama_${SLURM_ARRAY_TASK_ID}.log" 2>&1 &
OLLAMA_PID=$!

cleanup() {
  echo "Stopping Ollama (pid ${OLLAMA_PID})"
  kill "${OLLAMA_PID}" 2>/dev/null || true
  wait "${OLLAMA_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for Ollama on port ${OLLAMA_PORT}..."
for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${OLLAMA_PORT}/api/tags" > /dev/null 2>&1; then
    echo "Ollama ready after ${i}s"
    break
  fi
  sleep 1
done

"${OLLAMA}" pull "${OLLAMA_MODEL}"

API_URL="http://127.0.0.1:${OLLAMA_PORT}/api/generate"

echo "Generate codes: shard_start=${SHARD_START} shard_max=${SHARD_MAX_INDICES} \
shard_target=${SHARD_VALID_TARGET} cell_target=${CELL_VALID_TARGET} task=${TASK_PATH} llm=${LLM_CONFIG_PATH}"

uv run python -m generate_codes \
  --task "${TASK_PATH}" \
  --llm-config "${LLM_CONFIG_PATH}" \
  --cell-dir "${CELL_DIR}" \
  --shard-start "${SHARD_START}" \
  --shard-max-indices "${SHARD_MAX_INDICES}" \
  --shard-valid-target "${SHARD_VALID_TARGET}" \
  --cell-valid-target "${CELL_VALID_TARGET}" \
  --seed "${SEED}" \
  --api-url "${API_URL}"

echo "Stage A task ${SLURM_ARRAY_TASK_ID} complete"
