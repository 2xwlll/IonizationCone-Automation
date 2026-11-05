# F187N generation and training on UTSA ARC

Updated 2026-11-05. This workflow uses one-channel 256 × 256 F187N images and
nominal projected cone masks with `circular_cone_envelope_v2`. It includes
paired cone-free controls. The [model/loss audit](../configs/cloudy/F187N_MODEL_AUDIT.md)
records what was checked locally. ARC package installation, account access,
and GPU/driver execution still require verification on ARC.

## One launch command, after initial setup

From the updated repository on ARC:

```bash
python3 scripts/maintenance/launch_f187n_pipeline.py \
  --env-prefix /work/ppi246/agn/envs/f187n \
  --run-root /work/ppi246/agn/experiments/f187n_full_v3
```

The system Python can run this standard-library launcher, including Python 3.6.
Choose a fresh experiment directory. The launcher copies source, settings, and
six scientific input files into that directory, records their hashes, and submits:

1. A `compute1` CPU job to plan, generate, finalize, and validate the dataset.
2. A `gpu1a100` job with a success dependency on the CPU job. It validates the
   frozen files and dataset, checks CUDA execution, runs a positive/negative
   optimizer smoke test, and trains only if those checks pass.

The GPU job is canceled if its generation dependency cannot succeed. This uses
Slurm's documented [`afterok` dependency](https://slurm.schedmd.com/sbatch.html).
The login node only copies the small source/input snapshot and submits jobs.

Defaults are 64 independent realization groups, 30 epochs, and the `full`
positive-image variant. Each group contributes a full positive and a matched
cone-free negative to this training experiment: **128 selected images**, split
80/24/24 images from 40/12/12 groups. The generator also saves two positive
controls, giving **256 total saved images**. Versions and paired positives/
negatives of one realization always stay together in one split.

For the first ARC check, use a separate name and smoke-only mode:

```bash
python3 scripts/maintenance/launch_f187n_pipeline.py \
  --env-prefix /work/ppi246/agn/envs/f187n \
  --run-root /work/ppi246/agn/experiments/f187n_gpu_check_v3 \
  --groups 3 --smoke-only
```

It generates three groups and checks real GPU forward/backward/optimizer work,
then stops. This does not estimate accuracy. After success, run the normal
launch command with its own fresh directory. To change the bounded pilot size
or duration, supply `--groups N --epochs N`. `--variant smooth_control` or
`--variant clumpy_cone` selects that positive control plus the same negatives.
These control runs are separate experiments, not extra independent galaxies.

## Initial environment setup (once)

The user confirmed `anaconda3/2024.10-1`, `compute1`, and `gpu1a100` in ARC's
listings. `conda` normally appears after its module is loaded; see official
[UTSA modules](https://hpcsupport.utsa.edu/foswiki/bin/view/WebDocumentation/ModuleEnvironments)
and [Anaconda](https://hpcsupport.utsa.edu/foswiki/bin/view/WebDocumentation/PythonVMsInAnaconda)
instructions. Start in the repository and request a CPU allocation:

```bash
srun --partition=compute1 --nodes=1 --ntasks=1 --cpus-per-task=4 --time=01:00:00 --pty bash -l
module load anaconda3/2024.10-1
source "$(conda info --base)/etc/profile.d/conda.sh"
export AGN_ENV_PREFIX=/work/ppi246/agn/envs/f187n
conda create --prefix "$AGN_ENV_PREFIX" python=3.12 pip -y
conda activate "$AGN_ENV_PREFIX"
python -m pip install -r requirements-f187n.txt
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu126
python scripts/maintenance/check_f187n_environment.py
```

Wait for the allocated compute-node shell before loading Anaconda. ARC's Lmod
explicitly rejects this module on login nodes; `hostname` should show a compute
node such as `c002` before the `module load` step.

Use the dedicated environment's Python for scientific commands. Job scripts
activate it automatically. The CUDA wheel is listed in the official
[PyTorch version table](https://pytorch.org/get-started/previous-versions/).
CPU imports alone do not establish GPU/driver compatibility. Retain any Python,
glibc, CUDA, or package error and resolve it before the main training run.
The local CPU tests used Python 3.11.2 and torch 2.11.0+cu130. The user has now
verified ARC CPU imports on `c002`: Python 3.12.16, glibc 2.28, the pinned
scientific packages, and torch 2.11.0+cu126. CUDA availability was false on that
CPU node, as expected; GPU kernel execution remains pending. The scientific
inputs were not present at that check and require the separate transfer below.

ARC's live scheduler output supplied on 2026-11-05 advertises only 1 MiB per
node in compute1/2/3 and gpu1a100. `compute1` reports `DefMemPerNode=UNLIMITED`,
`MaxMemPerNode=UNLIMITED`, and `SelectTypeParameters=NONE`. That configured
value does not describe physical RAM, but explicit positive multi-GB requests
are rejected as unsatisfiable. The interactive command and both batch scripts
therefore omit memory flags and use the site's defaults. Do not change them
to `--mem=1M` based on this placeholder. See Slurm's
[memory configuration documentation](https://slurm.schedmd.com/slurm.conf.html).
If the scheduler configuration changes, inspect its live settings before
adding memory requests. This correction has been checked locally for script
syntax; acceptance and execution still require the user's ARC session.

## Scientific input transfer (once)

Git excludes the six required input files:

```text
data/cloudy/ngc1068_f187n_pilot_v1/grid.npz
data/cloudy/ngc1068_f187n_pilot_v1/manifest.json
data/cloudy/hii_f187n_calculated_v2/grid.npz
data/cloudy/hii_f187n_calculated_v2/manifest.json
results/cloudy/f187n_reference/nucleus_cutout.npz
results/cloudy/f187n_psf_nrcb4/kernel.npz
```

The local `results/transfers/f187n_pilot_inputs_v2.tar.gz` bundle (about 2.6 MB)
contains the four unchanged arrays, both grid manifests, and their checksums.
The earlier v1 bundle omitted the H II completion manifest required by the renderer;
use this v2 bundle even if the arrays have already been transferred. From the **local computer**,
in this repository:

```bash
scp results/transfers/f187n_pilot_inputs_v2.tar.gz ppi246@arc.utsa.edu:/work/ppi246/agn/f187n_pilot_inputs_v2.tar.gz
```

Then on **ARC**, in the cloned repository, with the environment active:

```bash
ls -lh /work/ppi246/agn/f187n_pilot_inputs_v2.tar.gz
tar --keep-old-files -xzvf /work/ppi246/agn/f187n_pilot_inputs_v2.tar.gz
sha256sum -c data/cloudy/F187N_INPUTS_V2.sha256
python scripts/maintenance/check_f187n_environment.py --inputs
```

Existing files are protected by `--keep-old-files`; every checksum must pass.
The `/work/ppi246/agn` parent already exists after environment creation. Use
this explicit work path: a successful upload to `~/...` was not visible at the
same home path on `c002` in the user's session. The exact cause of that home-path
discrepancy is unresolved; do not infer a specific mount failure. UTSA documents
[`/work` as available on all nodes](https://hpcsupport.utsa.edu/foswiki/bin/view/WebDocumentation/AboutArcCluster).
Preserve any differing versions and resolve them before launching. The bundle
contains the existing spectral grids, reference cutout, and PSF; it does not
replace older spectra with converged grids or supply the full raw archive.

## What the generated images mean

`configs/cloudy/f187n_dataset_pilot_v3.json` records controlled uniform ranges:
intrinsic half-opening 20–40 degrees, inclination from the sky plane −35–35
degrees, undirected sky PA 0–180 degrees, and size/amplitude variation. The v3 recipe also varies foreground
dust depth, coverage and spatial scale, host/H II strength, and cone/host texture.
See [generation controls and measured descriptors](../configs/cloudy/F187N_VARIATION_AND_DESCRIPTORS.md).
These are experimental ranges, not inferred galaxy-population priors.

| Saved variant | Meaning |
| --- | --- |
| `smooth_control` | Analytic smooth illumination with matching PSF and cone flux; no spectra, dust, contaminants, or noise |
| `clumpy_cone` | Cone-associated post-PSF gas with its excitation and extinction |
| `full` | All modeled components, empirical baseline, and noise |
| `cone_free` | Full image minus the entire saved post-PSF cone component, including off-axis tails, with an empty mask |

The negative retains other emission (including possible AGN-powered host and
nuclear light), the empirical amplitude/baseline, and the identical additive
noise realization. It is a **paired component-removal control**, not an
independent inactive-galaxy simulation or a new photon-noise calculation.
The 50% negative fraction in each selected experiment is a controlled choice,
not an estimate of the prevalence of cones in observed galaxies.

All images use signed-asinh normalization with a stored inverse scale. Component
ratios use linear noiseless post-PSF maps in reference-valid pixels at
0.3–3.3 arcsec, excluding baseline/noise. For cone-free cases, C is zero,
M=B/C is undefined, cone geometry/length are undefined, and the status is
`zero_cone_signal`. Other-component membership and patchiness remain recorded.
V3 additionally measures the nominal-cone intersection of this annulus, records
foreground-screen transmission/coverage, and assigns explicitly provisional
descriptive categories. There is no messiness input or validated difficulty score.
The baseline/noise inherited from the parent are explicitly retained separately.

The prescribed geometric label can be faint inside its boundary and have gas
emission beyond it. Shared old spectra, empirical calibration, simplified
projected radial structure, and shared observational inputs limit conclusions
to recovery within this generator. See [geometry](../configs/cloudy/F187N_CONE_GEOMETRY.md)
and [spectral constraints](../configs/cloudy/NGC1068_SPECTRAL_CONSTRAINTS.md).

## Logs, results, inspection, and retries

Each experiment directory contains:

- `snapshot.json` and `code/`: frozen source, configs and scientific inputs.
- `submissions/*.json`: submission commands and job IDs, including partial
  submission failures. These describe submission, not final job status.
- `logs/generate-JOBID.log` and `logs/train-JOBID.log`: persistent job output.
- `dataset/`: plan, attempted renders, component archives, tensors, completion
  records, the validated final manifest, and automatic descriptor tables/plots.
- `training/model_smoke/smoke.json`: positive/negative optimizer check.
- `training/model/`: config/provenance, epoch checkpoints and statistics,
  validation selection, test report, prediction previews, and plots of held-out
  errors versus measured contrast/obscuration.

Use `squeue -u "$USER"` to monitor jobs and the logs to inspect failures.
The resource requests are CPU: 1 core/12 hours; GPU: 1 A100/4 cores/4 hours,
with memory left to the site defaults for the scheduler configuration above.
Requests do not prove account access or sufficient time for every
future population size. See [UTSA Slurm](https://hpcsupport.utsa.edu/foswiki/bin/view/WebDocumentation/RunningApplicationsInteractivelyAndSubmittingBatchJobs).

To prepare a frozen, reviewable workflow without submitting jobs, add
`--dry-run`. Submit it later with:

```bash
python3 scripts/maintenance/launch_f187n_pipeline.py \
  --run-root /work/ppi246/agn/experiments/f187n_full_v3 --resume
```

The same command can retry failed generation after its previous jobs have
ended. It uses the original frozen settings, verifies completed groups before
skipping them, and keeps failed render attempts. Active job IDs block duplicate
submission. A killed process may leave `.submission.lock` or a dataset group
lock; verify the job is dead before removing that empty lock. Changed source,
inputs, completion records, or corrupted arrays stop execution; preserve them
and use a fresh experiment directory. Dataset finalization is repeatable after
verification. Training attempts are preserved and refuse automatic overwrite;
there is no automatic training-checkpoint resume.

Do not edit frozen files while jobs are queued. Source changes belong in a new
experiment. Work storage is not backed up and has a retention policy; archive
important inputs, manifests, logs, and selected checkpoints per [UTSA storage
guidance](https://hpcsupport.utsa.edu/foswiki/bin/view/WebDocumentation/WebHome).
