# F187N controlled pilot on UTSA ARC

Prepared 2026-11-02 from the user's ARC module and partition listings. Use this
workflow for **one-channel 256 × 256 F187N images**, with the nominal projected
geometric mask and `circular_cone_envelope_v2`. It does not launch the older
F502N or spectral-cube experiment. The environment and GPU have not yet been
tested on ARC; the complete small workflow has passed locally on CPU.

## 1. Create the environment on ARC

Start in the cloned repository, after bringing across the source update. ARC
lists `anaconda3/2024.10-1`, `compute1`, and `gpu1a100`. Seeing no `conda` before
loading its module is expected. See the official [UTSA module instructions](https://hpcsupport.utsa.edu/foswiki/bin/view/WebDocumentation/ModuleEnvironments)
and [Anaconda instructions](https://hpcsupport.utsa.edu/foswiki/bin/view/WebDocumentation/PythonVMsInAnaconda).

Request a CPU allocation for installation and checks; these commands assume
the user's account `ppi246` and its `/work/ppi246` directory:

```bash
srun --partition=compute1 --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=01:00:00 --pty bash -l
module load anaconda3/2024.10-1
source "$(conda info --base)/etc/profile.d/conda.sh"
export AGN_ENV_PREFIX=/work/ppi246/agn/envs/f187n
conda create --prefix "$AGN_ENV_PREFIX" python=3.12 pip -y
conda activate "$AGN_ENV_PREFIX"
python -m pip install -r requirements-f187n.txt
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu126
python scripts/maintenance/check_f187n_environment.py
```

The CUDA 12.6 wheel for torch 2.11.0 is listed in the official
[PyTorch version table](https://pytorch.org/get-started/previous-versions/).
Only PyTorch is needed, not torchvision or torchaudio. A successful CPU import
does not prove driver/GPU compatibility. The GPU job below checks actual CUDA
forward/backward execution and stops if it fails. If installation reports a
glibc, Python, or wheel incompatibility, retain the error and resolve it before
submitting training. Do not reuse the system Python 3.6 environment.

Re-entering a shell requires loading the module, sourcing the conda hook, and
activating the prefix again. Job scripts do this explicitly. Submit jobs from
the repository root so relative script paths resolve. Resource requests are
starting requests; the partition listing does not prove account access.

## 2. Transfer the four scientific inputs separately

Git intentionally excludes these files:

```text
data/cloudy/ngc1068_f187n_pilot_v1/grid.npz
data/cloudy/hii_f187n_calculated_v2/grid.npz
results/cloudy/f187n_reference/nucleus_cutout.npz
results/cloudy/f187n_psf_nrcb4/kernel.npz
```

A local bundle was prepared at `results/transfers/f187n_pilot_inputs_v1.tar.gz`
(about 2.6 MB), containing only these inputs and their SHA-256 checklist.
From the **local computer**, in this repository, transfer it:

```bash
scp results/transfers/f187n_pilot_inputs_v1.tar.gz ppi246@arc.utsa.edu:~/f187n_pilot_inputs_v1.tar.gz
```

Then on **ARC**, in the cloned repository:

```bash
tar --keep-old-files -xzvf ~/f187n_pilot_inputs_v1.tar.gz
sha256sum -c data/cloudy/F187N_INPUTS.sha256
python scripts/maintenance/check_f187n_environment.py --inputs
```

`--keep-old-files` protects existing research files and may report that some
already exist. Every checksum must still pass. If a file differs, preserve it
and resolve the input version before proceeding. The bundle does not contain
the full observational archive or new CLOUDY runs. These are the existing
spectral grids used by the corrected image; their documented convergence and
calibration limitations still apply.

## 3. Plan and generate the controlled dataset

With the environment active, in the repository root:

```bash
export AGN_ENV_PREFIX=/work/ppi246/agn/envs/f187n
export AGN_DATASET=/work/ppi246/agn/data/f187n_pilot_v1
python scripts/cloudy/prepare_f187n_dataset.py plan --dataset "$AGN_DATASET"
sbatch scripts/slurm/generate_f187n.slurm
```

The plan creates a new directory and records source/input hashes before any
rendering. The CPU job renders sequentially and finalizes the manifest only
after all groups complete. Check its `slurm-f187n-generate-JOBID.log`; once
finished, verify:

```bash
python scripts/cloudy/prepare_f187n_dataset.py validate --dataset "$AGN_DATASET"
```

The default population has **64 independent realization seeds**, assigned to
40 training, 12 validation, and 12 test groups. Each group has three images:

| Variant | Input meaning |
| --- | --- |
| `smooth_control` | Analytic smooth illumination, matching PSF and cone flux; no spectra, dust, contaminants, or noise |
| `clumpy_cone` | Saved cone-associated gas, including its excitation/extinction and PSF |
| `full` | Cone plus modeled other components, empirical baseline, and noise |

All three share the same geometric label and split. There are 192 images, but
training on one variant uses 64 images, not 192 independent galaxies.
`configs/cloudy/f187n_dataset_pilot_v1.json` records uniform experimental ranges:
intrinsic half-opening 20–40 degrees, inclination from the sky plane −35–35
degrees, undirected sky PA 0–180 degrees, and size/dust/component-amplitude
variation. These are controlled ranges, not inferred population distributions.
No cone-free population or independent spectral conditions are included.

Inputs use invertible signed asinh normalization; metadata keeps the scale.
Component ratios use linear noiseless post-PSF maps, with baseline/noise
excluded, in reference-valid pixels at 0.3–3.3 arcsec. Per-image complexity,
resolution, configurations, component maps, hashes, and renderer logs remain
with the dataset. Noise significance inherits the documented generator noise
model and is not a measured correlated-noise significance for real data.

For a cluster pipeline check, use a different fresh dataset path and add
`--groups 3` to `plan`. This gives one group per split and no useful accuracy
estimate. Do not combine that check with the 64-group results.

On a failed render, preserve the attempt and log. Retry only incomplete groups
with `render --dataset "$AGN_DATASET" --group-index N`, where N is zero-based,
then `finalize --dataset "$AGN_DATASET"`. `--all` intentionally refuses already
completed groups. A killed job can leave `locks/gNNNN`; verify the job is dead
before clearing that empty lock. Source/input changes require a new versioned
plan instead of mixing versions. Existing finalized datasets remain usable.

## 4. Verify the GPU, then train

After dataset validation, submit a **smoke-only** GPU job:

```bash
export AGN_RUN=/work/ppi246/agn/runs/f187n_gpu_check_v1
export AGN_SMOKE_ONLY=1
sbatch scripts/slurm/train_f187n.slurm
```

This checks CUDA, then performs one real F187N forward/backward/optimizer step.
Read the Slurm log and `${AGN_RUN}_smoke/smoke.json`. Once that passes, use a
new run name for the controlled full-image pilot:

```bash
export AGN_RUN=/work/ppi246/agn/runs/f187n_full_v1
export AGN_SMOKE_ONLY=0
export AGN_VARIANT=full
sbatch scripts/slurm/train_f187n.slurm
```

The default is 30 epochs, batch size 4, encoder widths [16,32,64,128], Adam
at 1e-4, and BCE/Dice loss with positive weight 1. See
`configs/cloudy/train_f187n_pilot_v1.json`. It uses validation loss for selection
and evaluates the test split afterward. Every epoch checkpoint is retained;
there is no automatic training resume. Fresh output names protect old runs.
To assess recoverability, subsequently compare separate `smooth_control`,
`clumpy_cone`, and `full` runs using the same manifest and training settings.

The report includes mask overlap, fitted projected opening/axis errors, fit
failure counts, a photometric angular-excess baseline, and per-case complexity.
The mask fit uses a fixed 0.3–1.5 arcsec annulus and 3-degree search steps; the
photometric baseline uses 0.7–3.3 arcsec. Both overlap scores also use that
latter common annulus. Fit failures remain undefined, not zero-error results.
These descriptive estimates are not 3D torus angles or observational validation.

Use `squeue -u "$USER"` to monitor jobs. See the official
[UTSA Slurm instructions](https://hpcsupport.utsa.edu/foswiki/bin/view/WebDocumentation/RunningApplicationsInteractivelyAndSubmittingBatchJobs).
`/work` is scratch storage, is not backed up, and has a retention policy;
archive important manifests, logs, inputs, and selected outputs elsewhere per
[UTSA storage guidance](https://hpcsupport.utsa.edu/foswiki/bin/view/WebDocumentation/WebHome).

## Local verification, 2026-11-02

Three groups/nine variants were generated and their manifest validated under
`data/cloudy/f187n_pipeline_check_v1`. A CPU smoke check and a one-epoch run
completed, including checkpoint reload and baseline/test reporting, under
`results/cloudy/f187n_pipeline_check_{smoke,epoch}_v1`. These are ignored local
artifacts. The local environment was Python 3.11.2 with torch 2.11.0+cu130,
executed on CPU; Python 3.12 with the CUDA 12.6 wheel is the proposed ARC setup.
These tests demonstrate executable plumbing only. ARC environment setup,
input transfer, a GPU smoke check, and the 64-group experiment remain pending.
