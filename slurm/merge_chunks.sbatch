#!/bin/bash
#SBATCH --job-name=llb-merge
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=slurm_logs/merge_%j.out
#SBATCH --error=slurm_logs/merge_%j.err

set -euo pipefail
mkdir -p slurm_logs

CONFIG_JSON="${CONFIG_JSON:-generation_config.json}"
CACHE_DIR="${CACHE_DIR:-model_cache}"

cd /home/edmondcunnin_umass_edu/Large-Language-Bayes-Model

echo "Merging chunks from ${CACHE_DIR}"
uv run python -m generate_models \
  --config "${CONFIG_JSON}" \
  --cache-dir "${CACHE_DIR}" \
  merge

echo "Merge complete"
