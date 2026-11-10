#!/bin/bash
#SBATCH --job-name=f187n-overfit
#SBATCH --partition=gpu1a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=slurm-f187n-overfit-%j.log
set -euo pipefail
: "${AGN_ENV_PREFIX:?Set the conda environment prefix}"
: "${AGN_SNAPSHOT_ROOT:?Set a fresh frozen diagnostic run directory}"
module load anaconda3/2024.10-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$AGN_ENV_PREFIX"
export MPLBACKEND=Agg
python scripts/maintenance/launch_f187n_pipeline.py --verify-snapshot "$AGN_SNAPSHOT_ROOT"
python scripts/maintenance/check_f187n_environment.py --require-cuda
python scripts/cloudy/overfit_f187n.py --snapshot-run "$AGN_SNAPSHOT_ROOT" --device cuda
python scripts/maintenance/export_f187n_review.py --run-root "$AGN_SNAPSHOT_ROOT" \
  --output "$AGN_SNAPSHOT_ROOT/review.tar.gz" --include-selected-checkpoint
