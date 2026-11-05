#!/bin/bash
#SBATCH --job-name=f187n-generate
#SBATCH --partition=compute1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-f187n-generate-%j.log
set -euo pipefail
: "${AGN_ENV_PREFIX:?Set the absolute conda environment path}"
: "${AGN_DATASET:?Set the absolute planned dataset path}"
module load anaconda3/2024.10-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$AGN_ENV_PREFIX"
export MPLBACKEND=Agg
if [[ -n "${AGN_SNAPSHOT_ROOT:-}" ]]; then
  python scripts/maintenance/launch_f187n_pipeline.py --verify-snapshot "$AGN_SNAPSHOT_ROOT"
fi
python scripts/maintenance/check_f187n_environment.py --inputs
if [[ ! -e "$AGN_DATASET" ]]; then
  python scripts/cloudy/prepare_f187n_dataset.py plan --dataset "$AGN_DATASET" \
    --config "${AGN_DATASET_CONFIG:-configs/cloudy/f187n_dataset_pilot_v2.json}"
elif [[ "${AGN_RESUME_GENERATION:-0}" != 1 ]]; then
  echo 'Dataset already exists. Use the launcher --resume to verify and continue it.' >&2
  exit 1
fi
python scripts/cloudy/prepare_f187n_dataset.py render --dataset "$AGN_DATASET" --all
python scripts/cloudy/prepare_f187n_dataset.py finalize --dataset "$AGN_DATASET"
python scripts/cloudy/prepare_f187n_dataset.py validate --dataset "$AGN_DATASET"
