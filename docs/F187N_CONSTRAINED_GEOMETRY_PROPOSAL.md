# F187N: geometric predictions and validation stopping

Updated 2026-11-12 after mentor feedback and the user's ARC log excerpt for job
859951. The cone-parameter head below is a **proposal, not an implemented or
trained model**. Validation stopping is implemented separately in the existing
pixel trainer, with an opt-in versioned configuration. No ARC job was submitted
as part of this update. Existing frozen runs and labels are unchanged.

## New diagnostic evidence

ARC job 859951 completed, exit 0:0, in 3:30. The user supplied scheduler output
and a log excerpt reporting all three memorization goals met:

| Representation | Selected epoch | Mean positive Dice | Boundary F1 | Maximum negative full-stamp area |
| --- | ---: | ---: | ---: | ---: |
| Smooth control | 190 | 0.9594 | 0.9501 | 0.0122% |
| Clumpy cone-only | 110 | 0.9667 | 0.9679 | 0.7294% |
| Full image | 110 | 0.9516 | 0.9100 | 0.5493% |

These are four known training realizations and four cone-free controls per
representation. Boundary F1 uses the declared two-pixel tolerance. This supports
the current U-Net's ability to fit sharp masks on known examples; it does not
establish generalization. Full/clumpy fitting took 220 optimizer updates, whereas
the 30-epoch population run made 599 updates over a larger set. Epoch counts are
not directly comparable. The earlier training/validation divergence warrants
overfitting controls; it does not isolate duration as the sole cause.

The complete diagnostic archive remains on ARC at
`/work/ppi246/agn/experiments/f187n_overfit_v1/review.tar.gz`, with a reported size
35,178,611 bytes and SHA-256
`4f6c1e872137f71c44d5e8bce0d9a1fe9ba9432c2d952f29cd8b5428536d139a`.
This document records user-supplied output, not a local archive inspection.

## Recommended geometric output

The current scientific target is prescribed nominal projected bicone geometry.
For that target, a constrained output is well aligned with the intended
measurement. A proposed small head attached to learned U-Net features predicts:

| Output | Definition and rule |
| --- | --- |
| Cone-presence probability | Allows an empty output; never force a cone on every image. A sigmoid score needs calibration before being called a calibrated probability. |
| Axis PA | East of north, undirected modulo 180 deg. Predict cos(2 PA), sin(2 PA) to respect periodicity. |
| Projected full opening | Full angular width, 0 < width < 180 deg; explicitly distinct from intrinsic half-opening. |
| Projected radial cutoff | In pixels/arcsec on the documented grid; this is a circular radial cutoff, not deprojected cone length. |

Keep the apex fixed at the known generator nucleus for the first experiment;
adding center coordinates before there are offset nuclei would add unsupported
freedom. Use the saved pixel-to-sky transformation when constructing the mask.
Render a two-lobed straight-sided sector mask with the current radial cutoff
and versioned center/rebinning rules. Two components of PA plus width, radius
and presence amount to five raw outputs for four physical/geometric quantities.

Train presence on all images and geometry only on cone-positive images. Use a
periodic axis loss, scale-normalized opening/radius losses and a differentiable
mask-consistency term. The hard inference renderer guarantees the stated shape;
a numerical soft renderer for training is separate from the physical gas
illumination prescription. Version all bounds and loss weights before fitting.
These remain proposed choices requiring tests and a controlled comparison.

Retain the original free pixel mask as an evidence/residual diagnostic. Display
it beside the constrained mask and input: a neat cone can still have a wrong
axis, opening or radius, or be unsupported by the observed emission. Reducing
the output space is a shape prior, not a guarantee against overfitting. A fixed
symmetric bicone can also be an inadequate model of a real galaxy. Do not use
this prior to replace the observational definition of an emission boundary.
Intrinsic opening/inclination are not uniquely determined by a projected mask.

Predicting a restricted family of shapes has precedent, for example
[StarDist's polygon representation](https://arxiv.org/abs/1806.03535); that paper
does not validate this proposed AGN cone model.

## Implemented validation safeguards for the pixel-model comparison

`train_f187n_pilot_v3_guarded.json` is opt-in. It retains the same model, losses,
learning rate, threshold and grouped-data rules. Its 150 epochs are a cap:

- Stop after 12 validation evaluations without an improvement of more than
  0.001 in the declared objective, with at least 20 completed epochs.
- Select and reload the actual best validation checkpoint, including smaller
  improvements that do not reset patience. Evaluate the test set only afterward.
- Log objective, stale count, stop reason, completed epochs and selected epoch.
- Use `0.5 * (positive Dice + 1 - negative predicted area)` for selection in this
  provisional configuration, with both terms measured on full stamps. It gives
  equal weight to positive overlap and negative pixel specificity. It is not
  detection accuracy, a physical quantity or a messiness score. Continue to
  report the two terms and angular errors separately; this score can still trade
  positive errors against negative errors.

The controller also supports validation loss or positive Dice explicitly. The
legacy v2 configuration retains fixed-budget training and minimum-validation-
loss selection when no early-stopping settings are supplied. The intentional
memorization diagnostic is unchanged and must not use held-out stopping.
The standard [early-stopping principle](https://keras.io/api/callbacks/early_stopping/)
is implemented here in PyTorch without adding a Keras dependency.

The ARC launcher accepts `--train-config` and takes the epoch cap from that file
unless `--epochs` explicitly overrides it. Resumed snapshots retain their own
configuration. Merely choosing a new output directory does not create new
physical realizations: dataset seeds/configuration must also change to reserve
a genuinely fresh final assessment population. Do not treat the already
inspected 12 test realizations as a fresh final benchmark after model selection
has been guided by them.

Early stopping reduces needless fitting after validation plateaus. It cannot
guarantee generalization, repair ambiguous targets or replace more independent
data. Compare the free-mask and proposed geometric outputs under the same
predeclared validation policy, report failed/detected cone counts, and retain a
new final assessment set. Shape-head implementation and its ARC comparison
remain future work; no geometry-head performance claim is made here.
