#!/bin/bash
#SBATCH --job-name=llb-sample
#SBATCH --partition=cpu-preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-99
#SBATCH --output=slurm_logs/sample_%A_%a.out
#SBATCH --error=slurm_logs/sample_%A_%a.err
#SBATCH --requeue

# Stage B: posterior inference, IWAE, true-LOO, and held-out test scoring for
# one (task, LLM) cell. Pure CPU. Reads codes/code_<sha>.code.json files
# written by Stage A, partitions the sorted hash list across array tasks, and
# writes samples/sample_<sha>.{npz,meta.json}. Idempotent resume via the
# meta.json existence check.
#
# Required environment variables:
#   TASK_PATH       tasks/<name>.json
#   CELL_DIR        experiment_results/paper/<task>/<llm>/
# Optional:
#   ARRAY_COUNT     default 100 (set to match #SBATCH --array)
#   SEED            default 42
#   LOO_WORKERS     default 8 (multiprocess pool size for true-LOO)
#   MCMC_NUM_WARMUP    default 500
#   MCMC_NUM_SAMPLES   default 1000
#   IWAE_NUM_INNER     default 5
#   IWAE_NUM_OUTER     default 80
#   LOO_NUM_INNER      default 25
#   LOO_NUM_WARMUP     default 50
#   LOO_NUM_SAMPLES    default 100

set -euo pipefail
mkdir -p slurm_logs

echo "Host: $(hostname)"
echo "Array job ${SLURM_ARRAY_JOB_ID}, task ${SLURM_ARRAY_TASK_ID}"

: "${TASK_PATH:?TASK_PATH must be set}"
: "${CELL_DIR:?CELL_DIR must be set}"

ARRAY_COUNT="${ARRAY_COUNT:-100}"
SEED="${SEED:-42}"
LOO_WORKERS="${LOO_WORKERS:-8}"
MCMC_NUM_WARMUP="${MCMC_NUM_WARMUP:-500}"
MCMC_NUM_SAMPLES="${MCMC_NUM_SAMPLES:-1000}"
IWAE_NUM_INNER="${IWAE_NUM_INNER:-5}"
IWAE_NUM_OUTER="${IWAE_NUM_OUTER:-80}"
LOO_NUM_INNER="${LOO_NUM_INNER:-25}"
LOO_NUM_WARMUP="${LOO_NUM_WARMUP:-50}"
LOO_NUM_SAMPLES="${LOO_NUM_SAMPLES:-100}"

cd /home/edmondcunnin_umass_edu/Large-Language-Bayes-Model

# Force JAX onto CPU and respect SLURM cpu allocation for BLAS threads.
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export XLA_FLAGS="--xla_force_host_platform_device_count=${SLURM_CPUS_PER_TASK:-16}"

mkdir -p "${CELL_DIR}/samples"

uv run python -m sample_from_codes \
  --task "${TASK_PATH}" \
  --cell-dir "${CELL_DIR}" \
  --array-task-id "${SLURM_ARRAY_TASK_ID}" \
  --array-task-count "${ARRAY_COUNT}" \
  --seed "${SEED}" \
  --mcmc-num-warmup "${MCMC_NUM_WARMUP}" \
  --mcmc-num-samples "${MCMC_NUM_SAMPLES}" \
  --log-marginal-num-inner "${IWAE_NUM_INNER}" \
  --log-marginal-num-outer "${IWAE_NUM_OUTER}" \
  --loo-num-inner "${LOO_NUM_INNER}" \
  --loo-num-warmup "${LOO_NUM_WARMUP}" \
  --loo-num-samples "${LOO_NUM_SAMPLES}" \
  --loo-workers "${LOO_WORKERS}"

echo "Stage B task ${SLURM_ARRAY_TASK_ID} complete"
