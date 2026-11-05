# F187N diagnostic-pilot readiness

Checked 2026-11-04. This status concerns one-channel F187N images and their
nominal projected geometric masks using `circular_cone_envelope_v2`.

**The explicit F187N pipeline now passes a local end-to-end check. ARC setup,
GPU verification, and the controlled 64-group pilot remain pending.** Follow
[F187N ARC setup](../../docs/F187N_ARC_SETUP.md). Existing optical/spectral
experiments must not silently stand in for this dataset.

## Latest local check

- `prepare_f187n_dataset.py` generated three independent groups with four
  matched variants each, including paired cone-free controls, one group per split. Source/input hashes, component
  diagnostics, geometry, and grouped train/validation/test assignments are saved.
- `train_f187n_pilot.py` passed a CPU forward/backward/optimizer check and one
  epoch through validation, checkpoint selection/reload, and held-out reporting
  against the angular-excess baseline. This is a pipeline demonstration, not
  evidence of segmentation accuracy.
- The pilot uses one 256 × 256 channel, invertible signed-asinh normalization,
  nominal projected masks, empty masks for negatives, and a smaller U-Net with
  widths [16,32,64,128]. The new explicit loss applies BCE to all images and
  Dice to positive targets only. See [model audit](F187N_MODEL_AUDIT.md).
- Reports are in ignored `results/cloudy/f187n_pipeline_check_smoke_v2` and
  `results/cloudy/f187n_pipeline_check_epoch_v2`; the dataset is in ignored
  `data/cloudy/f187n_pipeline_check_v2`. Unit tests passed (84 tests).
- The one-epoch computation demonstration still predicts excessive area,
  including on the negative test image. This is not a trained accuracy result.
- The default dataset configuration specifies 64 groups, with 40/12/12 grouped
  splits. A full-variant run selects 64 positives and 64 paired negatives.
  This larger dataset and its 30-epoch training have **not** been run.
- The standard-library launcher freezes source/configs/inputs and chains CPU
  generation/validation to GPU checking/training through a Slurm success
  dependency. Dry-run and mocked submission/failure tests passed locally.
- The isolated local workflow in ignored
  `results/cloudy/f187n_pipeline_frozen_check_v2b` also completed generation,
  validation, verified generation skips, mixed-target smoke testing, and a
  one-epoch CPU train/evaluate run. The v2 transfer bundle includes both grid
  manifests; the earlier four-array bundle missed a required H II sidecar.
- The ARC recipe targets the user-confirmed Anaconda module and A100 partition.
  Local tests do not establish remote package/driver compatibility or access.

## Earlier architecture-only check (historical)

### Verified at that earlier stage

- The corrected config is `ngc1068_f187n_soft_gas_projection_v2.json`.
- Its saved pair is a finite 256 x 256 F187N image and binary geometric mask.
- The existing `UNet(in_channels=1, out_channels=1)` with default encoder
  widths [64,128,256,512] produces finite logits of shape (1,1,256,256) from
  that image on CPU.
- This was a randomly initialized forward check only. No optimizer step,
  training, accuracy evaluation, checkpoint fitting or SSH GPU check was run.
  Per-image standardization was used for the compatibility check, not adopted
  as the final training normalization.
- The local report and input/model hashes are saved in
  `results/cloudy/f187n_training_readiness_v1/readiness.json` (ignored output).

### Requirements identified at that earlier stage

Items 1–3 now have explicit implementations and a three-group local check;
population generation at the planned scale and item 4 remain pending on ARC.

1. Generate a modest corrected population with smooth, clumpy and contaminated
   cases, explicit angle ranges and retained components. Currently the corrected
   example supplies one realization, not a training population.
2. Prepare a deterministic manifest grouping variants of each physical
   realization together across train/validation/test assignments. Record band,
   projection, labels, normalization and input hashes.
3. Add an explicit F187N training entry point/configuration accepting one image
   channel and the exact manifest. Specify loss, optimizer, seed and evaluation
   against the geometric baseline, including angle errors and mask overlap.
4. Verify the required grids, reference, PSF, dataset and environment on the
   training host; then run a one-batch computation/optimizer check before an
   authorized diagnostic training pilot.

`scripts/2d/train.py` currently uses name-similarity selection for an old
`synthetic_oiii_realistic` dataset. It is not an explicit F187N launcher.
`scripts/cloudy/train_oiii_cube.py` defaults to 96 observed-wavelength channels
and validates a spectral-cube manifest. Its historical smoke test is not a
F187N readiness check. Removing old plot files changes neither of these facts.

The current F187N image still uses its recorded older spectral grids; geometry
correction does not replace their spectra or resolve convergence limitations.
Those limits must remain explicit in any pilot conclusions. Recoverability
of prescribed synthetic geometry is the first experimental question, not a
claim of validated real-galaxy segmentation.
