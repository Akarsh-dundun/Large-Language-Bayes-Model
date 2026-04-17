#!/bin/bash
#SBATCH --job-name=llb-sample-topup
#SBATCH --partition=cpu-preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-19
#SBATCH --output=slurm_logs/sample_topup_%A_%a.out
#SBATCH --error=slurm_logs/sample_topup_%A_%a.err
#SBATCH --requeue

# Stage B top-up: re-runs sample_from_codes after Stage A grew. Identical to
# sample_cpu.sbatch except for a smaller default array size (the assumption
# is that only a few hundred new hashes need processing). Idempotent resume
# means existing ok samples are skipped automatically.

set -euo pipefail
mkdir -p slurm_logs

: "${TASK_PATH:?TASK_PATH must be set}"
: "${CELL_DIR:?CELL_DIR must be set}"

ARRAY_COUNT="${ARRAY_COUNT:-20}"
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

export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

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

echo "Top-up task ${SLURM_ARRAY_TASK_ID} complete"
