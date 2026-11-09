#!/bin/bash
#SBATCH --job-name=f187n-train
#SBATCH --partition=gpu1a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=slurm-f187n-train-%j.log
set -euo pipefail
: "${AGN_ENV_PREFIX:?Set the absolute conda environment path}"
: "${AGN_DATASET:?Set the absolute finalized dataset path}"
: "${AGN_RUN:?Set a fresh absolute training output path}"
module load anaconda3/2024.10-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$AGN_ENV_PREFIX"
export MPLBACKEND=Agg
if [[ -n "${AGN_SNAPSHOT_ROOT:-}" ]]; then
  python scripts/maintenance/launch_f187n_pipeline.py --verify-snapshot "$AGN_SNAPSHOT_ROOT"
fi
python scripts/cloudy/prepare_f187n_dataset.py validate --dataset "$AGN_DATASET"
python scripts/maintenance/check_f187n_environment.py --require-cuda
F187N_MANIFEST_ARGS=()
if [[ -n "${AGN_DATASET_MANIFEST_SHA256:-}" ]]; then
  F187N_MANIFEST_ARGS=(--expected-manifest-sha256 "$AGN_DATASET_MANIFEST_SHA256")
fi
python scripts/cloudy/train_f187n_pilot.py --dataset "$AGN_DATASET" \
  --output "${AGN_RUN}_smoke" --device cuda --variant "${AGN_VARIANT:-full}" \
  --config "${AGN_TRAIN_CONFIG:-configs/cloudy/train_f187n_pilot_v2.json}" "${F187N_MANIFEST_ARGS[@]}" --smoke-test
if [[ "${AGN_SMOKE_ONLY:-0}" == 1 ]]; then exit 0; fi
python scripts/cloudy/train_f187n_pilot.py --dataset "$AGN_DATASET" \
  --output "$AGN_RUN" --device cuda --variant "${AGN_VARIANT:-full}" \
  --config "${AGN_TRAIN_CONFIG:-configs/cloudy/train_f187n_pilot_v2.json}" "${F187N_MANIFEST_ARGS[@]}"
python scripts/cloudy/summarize_f187n_population.py --dataset "$AGN_DATASET" \
  --training-report "$AGN_RUN/test_report.json" --output "$AGN_RUN/complexity_diagnostics"
