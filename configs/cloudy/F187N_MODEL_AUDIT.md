# F187N model, loss, and evaluation audit

Update 2026-11-12: user-supplied ARC job 859951 logs report successful fit-set
memorization, including full-image Dice 0.9516 and boundary F1 0.9100 on four
training realizations. This supports sharp-mask fitting capacity, not held-out
accuracy. The pixel trainer now supports opt-in validation stopping with best
checkpoint restoration. See the [constrained geometry proposal and safeguards](../../docs/F187N_CONSTRAINED_GEOMETRY_PROPOSAL.md).
The geometry parameter head is proposed and not yet implemented or trained.

Update 2026-11-09: the original ARC full run failed during epoch 18 after
17 completed epochs. Scaled-gradient overflow handling and smoke-test precision
coverage have been corrected. See [the failure review and retry](../../docs/F187N_TRAINING_FAILURE_857188.md).
The strict-gradient behavior described below is a dated initial implementation:
the corrected trainer skips scaler-detected FP16 overflow updates and lowers
the scale, while retaining explicit failure for unscaled nonfinite gradients
or a nonfinite norm of finite gradients. It records actual update/skip counts.

Checked 2026-11-04 for one-channel 256 × 256 F187N images and nominal projected
cone masks, including paired cone-free images. This is a computation and
controlled-pilot audit, not certification of scientific segmentation accuracy.

## Architecture and loss

The existing 2D U-Net returns one raw logit per pixel. There is no final sigmoid
in the model. This is correct for PyTorch's numerically stable
[`BCEWithLogitsLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html).
The architecture needed no change. The pilot uses encoder widths
[16,32,64,128], preserving the historical model defaults for other experiments.
BatchNorm operates over spatial pixels even with a one-image batch, but the
pilot's normal batch size is four. Real GPU memory/throughput remain untested.

The historical shared BCE/Dice loss defaults to positive weight 20 and computes
Dice on every image. On an empty target its Dice numerator is only epsilon;
when predictions are wrong, the Dice loss is nearly constant with tiny gradient.
BCE still supplies a valid correction, so the historical loss is not mathematically
invalid; its empty-target Dice term is unhelpful for the new experiment.

The F187N-specific `F187NLoss` leaves historical trainers unchanged and uses:

- Mean BCE over all pixels/images, including empty targets.
- Mean soft Dice over images with nonempty targets only; zero Dice contribution
  for an entirely empty batch.
- Explicit weights BCE=1, Dice=1, positive class weight=1. These are pilot
  choices, not optimized hyperparameters or inferred class priors.

The old F187N configuration used Dice weight 3. The new explicit equal weights
are versioned in `train_f187n_pilot_v2.json`; old and new losses are not silently
interchangeable. Loss is computed in float32 even when the U-Net uses CUDA
mixed precision. Gradients are unscaled before clipping at norm 5, with
nonfinite outputs/loss/gradients treated as failed runs. Adam uses learning
rate 1e-4; default duration is 30 epochs. No claim is made that these are optimal.

## Checks and evaluation rules

Unit tests check finite losses and gradients for positive, empty, mixed and
extreme-logit cases, and a corrective gradient on false-positive empty masks.
The production-width model also reduces the fixed mixed-batch loss by at least
20% over 12 Adam steps on a small 64 × 64 computation test. This is an optimizer
sanity test. The real-image smoke check separately uses 256 × 256 inputs with
both a positive and its cone-free counterpart.

The trainer validates the dataset schema, normalization, binary labels,
cone-presence flags, hashes, split grouping and matched masks before fitting.
It logs image/negative counts, package/code versions, and the manifest hash.
Validation loss chooses an epoch; the test split is then evaluated once.
The fixed probability threshold is 0.5, not tuned on test data.

Report positive-target IoU and Dice separately from negative-target false-positive
area. Empty/empty overlap is defined as 1 for per-case accounting, but negative
cases do not inflate the positive-target averages. Opening and axis errors
apply only to positive targets. Failed geometric fits stay undefined and their
counts are reported; cone-free cases have no true opening angle.

For negatives, the main statistic is predicted area / valid aperture area;
report an additional count exceeding 1% as an explicit experimental flag.
This flag is not a calibrated cone-detection classifier. The same statistics
are computed for the angular-excess baseline. Mask-angle fitting uses the fixed
0.3–1.5 arcsec annulus and 3-degree search steps; baseline extraction and the
common overlap/false-positive aperture use 0.7–3.3 arcsec. Differences in these
estimators remain a limitation. Inspect the saved per-case results/previews
and errors versus contrast, patchiness, resolution, dust and inclination.

## Local evidence

The complete 84-test unit suite passed, including loss/model, empty-target evaluation,
manifest, and mocked Slurm submission/dependency/failure tests. Three fresh
realizations, each with four variants, are saved locally under
`data/cloudy/f187n_pipeline_check_v2`. Local CPU smoke and one-epoch evaluation
outputs are under `results/cloudy/f187n_pipeline_check_{smoke,epoch}_v2`.
They demonstrate computation and report generation only. A frozen-launcher dry
run is under `results/cloudy/f187n_launcher_check_v2`. The final isolated
workflow under `results/cloudy/f187n_pipeline_frozen_check_v2b` passed fresh
generation, validation, verified skipping of completed groups, mixed-target
smoke checking, one CPU epoch, checkpoint reload, and test/preview generation.
The initial isolated attempt under `f187n_pipeline_frozen_check_v2` failed
because the transfer set omitted the required H II grid completion manifest;
its inputs and log are preserved. The corrected v2 bundle and preflight now
include both grid manifests, with a regression test for missing sidecars.
Binary previews now use fixed 0–1 limits so all-positive predictions appear
white. The first preview with automatic limits is preserved; its corrected
version is `f187n_pipeline_check_epoch_v2/test_predictions_fixed_scale.png`.
No real Slurm jobs were submitted from this workstation. All these are ignored
research artifacts; a one-epoch model still predicts excessive positive area.

The [ARC instructions](../../docs/F187N_ARC_SETUP.md) distinguish setup, the
GPU smoke check, and the controlled training experiment. Shared reference data,
CLOUDY grids and generator assumptions remain shared across the grouped split;
held-out seeds do not establish independent real-galaxy generalization.
