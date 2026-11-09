# ARC training job 857188: partial learning and numerical failure

Recorded 2026-11-09 from the user's scheduler output and pasted log excerpt.
The original run remains at `/work/ppi246/agn/experiments/f187n_full_v3`.
No original inputs, component archives, checkpoints, code snapshot or logs
should be edited or deleted for this correction.

## What the run did

Generation job 857187 completed in 13:05 with exit 0:0. GPU job 857188 passed
A100-PCIE-40GB CUDA forward/backward checks and its original full-precision
smoke check. It saved 17 complete training epochs, then failed during epoch 18
with a nonfinite total gradient norm at `clip_grad_norm_`. Scheduler state was
FAILED, exit 1:0, elapsed 2:11. The shown trainer evaluates the test set only
after all epochs and validation selection, so no completed test evaluation is
established by this excerpt.

The experiment is F187N, one 256 × 256 channel, nominal corrected projected
geometry masks, and paired empty-mask component-removal controls. The configured
64-group pilot selects 80/24/24 train/validation/test images. The shared source
spectra and calibration still limit generalization claims.

| Completed epoch | Train Dice, positives | Validation Dice, positives | Validation negative predicted area |
| --- | ---: | ---: | ---: |
| 1 | 0.375 | 0.238 | 100.0% |
| 8 | 0.546 | 0.531 | 16.1% |
| 12 | 0.619 | 0.341 | 2.76% |
| 16 | 0.667 | 0.490 | 3.44% |
| 17 | 0.694 | 0.520 | 4.62% |

The network learned to suppress many false-positive pixels and improved positive
validation overlap, but validation Dice is moderate and fluctuating. The later
train/validation gap warrants inspection. The minimum completed validation loss
was epoch 16 (1.1277); highest completed validation Dice was epoch 8 (0.531).
These select different epochs because loss includes both BCE and positive Dice,
including BCE on negatives. A lower combined loss does not guarantee higher
thresholded positive Dice. Negative percentages describe full-stamp predicted
pixel area, not the percentage of galaxies falsely detected. Epoch metrics use
the full stamp; final test reporting also provides a common valid annulus.

No opening-angle accuracy or improvement over the geometric baseline can be
claimed until a completed evaluation is available. The partial learning plot,
CSV transcribed from the user-supplied JSON lines, and source-qualified summary
are in ignored `results/cloudy/f187n_arc_857188_review_v1`. The remote log remains
the canonical record; this is not a downloaded copy of its full contents.

## Numerical correction

The previous full trainer enabled FP16 autocast and dynamic gradient scaling,
but called strict gradient clipping before allowing the scaler to skip an
overflowed update. Its smoke check did not exercise that mixed-precision path.
The log confirms the failing norm and enabled mixed precision; it does not
identify a particular layer or record whether every raw gradient was finite.
FP16 overflow is therefore a supported explanation, not a proven layer-level
root cause.

The corrected optimizer helper follows [PyTorch 2.11 AMP guidance](https://docs.pytorch.org/docs/2.11/notes/amp_examples.html):

- Unscale gradients before inspecting/clipping them.
- If scaled gradients are nonfinite, allow GradScaler to skip the optimizer
  update and lower its loss scale. Do not clip those gradients or update weights.
- Log the affected sample IDs, epoch/batch and scale change in
  `overflow_events.jsonl`. Per-epoch statistics record attempted overflows,
  successful optimizer updates and the current loss scale.
- Stop after eight consecutive overflowed batches, an epoch with no updates,
  nonfinite forward outputs/loss, or nonfinite gradients without scaling.
- Keep strict clipping for finite gradients. A nonfinite norm despite finite
  entries still raises, preserving that failure for further diagnosis.

The smoke check now exercises configured mixed precision and up to the configured
batch size, retaining positive and negative targets. Its disposable model may
retry scaling calibration; full training starts in a separate process with its
original seed. Loss weights, learning rate, masks and image normalization are
unchanged. Planned batches remain 600 for 30 epochs; the actual optimizer-update
count may be lower if overflowed batches are skipped. Epoch loss/Dice summaries
include evaluated batches even when their optimizer update was skipped.

Unit tests inject overflow with a CPU GradScaler, verify weights and Adam state
remain unchanged, verify the scale decreases, and verify the next finite batch
updates normally. FP32 nonfinite gradients are rejected. This tests the shared
scaler/update logic; a corrected ARC GPU run is still required.

## Training-only retry

From the repository on an ARC login node after pulling the source update:

```bash
python3 scripts/maintenance/retry_f187n_training.py \
  --source-run /work/ppi246/agn/experiments/f187n_full_v3 \
  --run-root /work/ppi246/agn/experiments/f187n_full_v3_retry_v1
```

The standard-library launcher parses on Python 3.6. It submits only a GPU job.
It verifies the original frozen source, copies the original scientific inputs
and training settings into a fresh snapshot with the corrected current source,
and points training at the existing dataset without regenerating it. The
original manifest/plan hashes are pinned and the trainer verifies the manifest
again before fitting. New logs/checkpoints go under the retry directory. Active
source jobs block submission, and existing retry directories are protected.

This is a fresh initialization, not an epoch-17 checkpoint continuation; the
old trainer did not save the complete RNG/dataloader state needed for exact
continuation. Model configuration, seed, split and dataset are retained. Original
checkpoints remain available for later inspection. No test data or validation
metric is used to alter geometry or emission lines.

For a reviewable preparation without submission, add `--dry-run`. Submit that
prepared snapshot later using the same paths plus `--submit-prepared`. A retry
that already has submission records requires a new directory rather than an
automatic overwrite or duplicate job.

Local evidence: optimizer overflow/recovery tests, retry submission/hash tests,
a local corrected smoke check, a frozen training-only dry run and one CPU epoch
through checkpoint reload/evaluation completed. No ARC job was submitted by the
local assistant; the user submits the command above. Retry success and scientific
performance remain to be established from that run's outputs.
