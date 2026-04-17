#!/bin/bash
#SBATCH --job-name=llb-gensample
#SBATCH --partition=gpu-preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --array=0-99
#SBATCH --output=slurm_logs/gensample_%A_%a.out
#SBATCH --error=slurm_logs/gensample_%A_%a.err
#SBATCH --requeue

# Generation plus PPL sampling for one sharded (task, LLM) pair.
# Each array task owns SHARD_SIZE contiguous model indices. Per-model artifacts
# are written to OUTPUT_DIR. Resume on preemption is handled inside
# generate_and_sample.py by skipping indices whose meta.json already has a
# populated status.
#
# Required environment variables (set at submission time):
#   TASK_PATH          path to task spec JSON (e.g. tasks/coin_flip.json)
#   LLM_CONFIG_PATH    path to llm config JSON (e.g. llm_configs/gemma4.json)
#   OLLAMA_MODEL       ollama model tag to pull (e.g. gemma4:e4b)
#   OUTPUT_DIR         directory for per-model artifacts
# Optional:
#   SHARD_SIZE         number of model indices per array task (default 100)
#   SEED               base seed (default 42)
#   MCMC_NUM_WARMUP    default 500
#   MCMC_NUM_SAMPLES   default 1000
#   USE_TRUE_LOO       0 or 1 (default 1)
#   LOO_NUM_WARMUP     default 50
#   LOO_NUM_SAMPLES    default 100
#   LOO_NUM_INNER      default 25
#   IWAE_NUM_INNER     default 5
#   IWAE_NUM_OUTER     default 80

set -euo pipefail
mkdir -p slurm_logs

echo "Host: $(hostname)"
echo "Array job ${SLURM_ARRAY_JOB_ID}, task ${SLURM_ARRAY_TASK_ID}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

: "${TASK_PATH:?TASK_PATH must be set}"
: "${LLM_CONFIG_PATH:?LLM_CONFIG_PATH must be set}"
: "${OLLAMA_MODEL:?OLLAMA_MODEL must be set}"
: "${OUTPUT_DIR:?OUTPUT_DIR must be set}"

SHARD_SIZE="${SHARD_SIZE:-100}"
SEED="${SEED:-42}"
MCMC_NUM_WARMUP="${MCMC_NUM_WARMUP:-500}"
MCMC_NUM_SAMPLES="${MCMC_NUM_SAMPLES:-1000}"
LOO_NUM_WARMUP="${LOO_NUM_WARMUP:-50}"
LOO_NUM_SAMPLES="${LOO_NUM_SAMPLES:-100}"
LOO_NUM_INNER="${LOO_NUM_INNER:-25}"
IWAE_NUM_INNER="${IWAE_NUM_INNER:-5}"
IWAE_NUM_OUTER="${IWAE_NUM_OUTER:-80}"
USE_TRUE_LOO="${USE_TRUE_LOO:-1}"

SCRATCH=/scratch3/workspace/edmondcunnin_umass_edu-siple
export OLLAMA_MODELS="${SCRATCH}/ollama_models"
export LD_LIBRARY_PATH="${SCRATCH}/ollama/lib/ollama:${LD_LIBRARY_PATH:-}"
OLLAMA="${SCRATCH}/ollama/bin/ollama"

cd /home/edmondcunnin_umass_edu/Large-Language-Bayes-Model

mkdir -p "${OUTPUT_DIR}"

SHARD_START=$((SLURM_ARRAY_TASK_ID * SHARD_SIZE))
SHARD_COUNT="${SHARD_SIZE}"

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

API_URL="http://127.0.0.1:${OLLAMA_PORT}/api/generate"

TRUE_LOO_FLAG="--use-true-loo"
if [ "${USE_TRUE_LOO}" = "0" ]; then
  TRUE_LOO_FLAG="--no-true-loo"
fi

echo "Running shard: start=${SHARD_START} count=${SHARD_COUNT} task=${TASK_PATH} llm=${LLM_CONFIG_PATH}"
uv run python -m generate_and_sample \
  --task "${TASK_PATH}" \
  --llm-config "${LLM_CONFIG_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --shard-start "${SHARD_START}" \
  --shard-count "${SHARD_COUNT}" \
  --seed "${SEED}" \
  --api-url "${API_URL}" \
  --mcmc-num-warmup "${MCMC_NUM_WARMUP}" \
  --mcmc-num-samples "${MCMC_NUM_SAMPLES}" \
  --log-marginal-num-inner "${IWAE_NUM_INNER}" \
  --log-marginal-num-outer "${IWAE_NUM_OUTER}" \
  --loo-num-inner "${LOO_NUM_INNER}" \
  --loo-num-warmup "${LOO_NUM_WARMUP}" \
  --loo-num-samples "${LOO_NUM_SAMPLES}" \
  ${TRUE_LOO_FLAG}

echo "Task ${SLURM_ARRAY_TASK_ID} complete"
