# ARC diagnostic: can the current U-Net fit sharp masks on known examples?

Prepared 2026-11-10 after inspecting completed ARC job 859150. This is the first
step before deciding whether to increase dataset size, training duration or
network width. It is a training-example memorization experiment, not a new
estimate of held-out accuracy or physical realism.

## Submit on ARC

From an ARC login terminal:

```bash
cd ~/projects/IonizationCone-Automation &&
git pull --ff-only origin master &&
python3 scripts/maintenance/launch_f187n_overfit.py \
  --source-run /work/ppi246/agn/experiments/f187n_full_v3_retry_v1 \
  --run-root /work/ppi246/agn/experiments/f187n_overfit_v1
```

The standard-library launcher works with login-node Python 3.6. It freezes the
current source and original training/scientific settings into a fresh directory
and submits **one A100 job**. Anaconda is loaded inside the allocated job. No
positive memory request is made on ARC's placeholder-memory Slurm configuration.
The existing population is reused; no generation job or dataset modification is
performed. An existing output directory is protected against reuse; choose a
new suffix for a deliberate new attempt. `--dry-run` prepares a snapshot without
submission; it also requires a new directory.

The launcher prints `diagnostic_job`. Monitor that ID with `squeue -j JOBID` and,
after it leaves the queue, `sacct -j JOBID --format=JobID,State,ExitCode,Elapsed`.
After starting, its log is `.../f187n_overfit_v1/logs/overfit-JOBID.log`.

## Experiment and fixed settings

The versioned recipe is `configs/cloudy/f187n_overfit_diagnostic_v1.json`:

- Four deterministic **training** groups, shared by all three runs. No validation
  or test image/mask arrays are loaded. Metadata for other splits is retained
  solely to audit the original grouping and dataset provenance.
- Three fresh models, fitted serially: `smooth_control`, `clumpy_cone`, `full`.
  Each receives four positive images and the same four cone-free controls, for
  eight images per model and sixteen distinct input images overall.
- Identical initial seed, architecture `[16,32,64,128]`, original learning rate
  (1e-4), batch size 4, BCE/positive-only Dice loss and 0.5 prediction threshold.
  The original GradScaler overflow correction is retained.
- Up to **300 epochs per model** (600 attempted updates each). Evaluation on
  those same eight images occurs at epoch 1, every 10 epochs and the final epoch.
  Fitting can stop after epoch 30 once all declared recovery checks are met.

The smooth control has analytic illumination with matching PSF and total cone
flux, without the full image's dust, contaminants or noise. Clumpy cone-only
retains its saved post-PSF gas/extinction/excitation. The full image includes all
components, baseline and noise. These are existing control constructions, not
strict single-parameter ablations. In the smooth/clumpy runs the negatives are
the original full-image cone-removal controls, not counterfactual removals of
each separately normalized positive control. Original geometric masks remain
identical across the three positive representations.

## What sharpness means here

Scores use the full training stamp, matching the loss domain. Alongside positive
Dice, the diagnostic measures boundary F1: predicted/target inner boundaries
are extracted using 8-neighbor erosion. A boundary pixel matches if a boundary
of the other mask is within a Euclidean distance of **two pixels**. Symmetric
boundary precision and recall form F1. The corresponding angular tolerance is
recorded in provenance (about 0.0615 arcsec on this grid). Empty positive
predictions score zero boundary F1. Negative masks have no boundary score.

Predeclared recovery checks require mean positive Dice >= 0.95, minimum
individual positive Dice >= 0.90, mean positive boundary F1 >= 0.90 and maximum
negative predicted area <= 1% of the full stamp. These are diagnostic goals,
not calibrated scientific acceptance or detection thresholds. They differ in
aperture from the original held-out common-annulus reporting.

Selection uses the first checked epoch meeting all goals after the minimum
epoch count, otherwise the lowest fit-set loss. Only the selected checkpoint
per representation is retained. Logs, evaluation history, overflow events,
metrics, probability maps and previews are preserved. A `budget_exhausted`
result is a completed computation with unmet recovery goals, not a Slurm error.

## Inspect and transfer

Outputs are in `.../f187n_overfit_v1/diagnostics/`, with a `summary.json` and
subdirectories for all three representations. Each contains `report.json`,
`history.jsonl`, `fit_curves.png`, `fit_predictions.png`, `fit_probabilities.npz`
and `selected.pth`. The GPU job automatically creates and verifies
`.../f187n_overfit_v1/review.tar.gz` and its `.sha256` sidecar. The export contains
only fitted training inputs/masks, the selected models, reports, logs and frozen
source/scientific inputs. It contains no held-out input arrays. Export completion
is part of the batch script; check its log if the final archive is absent.

After successful completion, run on the **local computer outside SSH**:

```bash
cd /home/wlll2x/Programming/AGN/IonizationCone-Automation &&
mkdir -p results/transfers/f187n_overfit_v1 &&
scp 'ppi246@arc.utsa.edu:/work/ppi246/agn/experiments/f187n_overfit_v1/review.tar.gz*' results/transfers/f187n_overfit_v1/ &&
venv/bin/python scripts/maintenance/export_f187n_review.py \
  --verify results/transfers/f187n_overfit_v1/review.tar.gz
```

Keep exports under ignored `results/`; source alone belongs in GitHub.

## Interpretation and next decision

Success on full images establishes that this model/optimizer can memorize
their sharp targets under these settings. It does not establish that new images
contain enough information to recover the same boundary. Success on smooth
controls with weaker full-image fitting motivates examining complexity,
optimization and emission–label correspondence. Failure within this finite
budget does not prove the architecture incapable or the labels unrecoverable.

Use these results to select the next experiment: an expanded independent
population, a longer optimization budget, or a controlled width comparison.
Avoid changing all three together. The earlier test set has already informed
diagnosis; reserve new realizations for a later final assessment. No expensive
population expansion or wider-model run is automatically submitted here.

Local preparation checks include boundary scoring, train-only selection,
single-job submission and export role/integrity tests. A one-epoch CPU workflow
exercises all three representations and checks that only selected training
arrays and the shared validity map are read. This does not establish ARC GPU
execution or achievement of the memorization targets.
