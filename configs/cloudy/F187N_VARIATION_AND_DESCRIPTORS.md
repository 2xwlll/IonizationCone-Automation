# F187N generation controls and measured image descriptors

Added 2026-11-05. Recipe `f187n_controlled_variation_v3` treats scene complexity
as measured output. There is no input called messiness and no universal scalar
segmentation-difficulty score. The one-channel U-Net receives the image; these
measurements describe the dataset and support performance analysis.

## Generation controls

`f187n_dataset_pilot_v3.json` independently samples the following inputs, in
addition to the existing geometric angles, projected radius, stellar continuum,
and nuclear continuum amplitudes:

| Input | Controlled range | Meaning |
| --- | --- | --- |
| Foreground patch optical-depth increment at Pa alpha | 0.1–2.0 | Additional tau on covered intrinsic pixels; diffuse floor is zero |
| Foreground covering fraction | 0–0.8 | Fraction of the whole intrinsic stamp assigned to patches |
| Dust correlation sigma | 0.08–0.6 arcsec | Gaussian correlation scale of the field used to locate patches, not a measured cloud radius |
| Host-gas relative weight | 0.02–0.3 | Multiplier of the low-U host spatial weight |
| H II relative strength | 0.05–1.0 | H II gas p99 relative to the renderer's AGN-gas p99 convention |
| Cone texture correlation sigma | 0.04–0.12 arcsec | Small-scale cloud texture correlation |
| Host texture correlation sigma | 0.05–0.25 arcsec | Small-scale host texture correlation |
| Cone log-texture coefficient | 0.2–1.2 | Coefficient multiplying its standardized small-scale Gaussian field |
| Host log-texture coefficient | 0.2–1.0 | Corresponding host coefficient |

These are illustrative experimental ranges, not measured population priors.
Relative amplitudes still inherit gas-p99 normalization and the shared empirical
single-band scaling. Independently sampled knobs do not make component
luminosities physically independent or absolutely calibrated. The same spectral
grids and line handling remain in use; no individual emission lines are rescaled.
Hot nuclear dust temperature/mixture and grain prescriptions inside CLOUDY grids
are unchanged. This remains a controlled 2D experiment, not a new torus model.

## Foreground screen definition

`patchy_screen_v1` smooths an independent seeded Gaussian random field, then
selects its highest-valued pixels to obtain the requested whole-stamp coverage
(to the nearest intrinsic pixel). The optical depth is tau_floor + tau_patch
on covered pixels, and tau_floor elsewhere. A separate RNG stream ensures that
changing dust controls does not resample the gas clouds, noise, or geometry.
The actual covered area inside the cone can differ from whole-stamp coverage.

The renderer applies wavelength-dependent attenuation before filter integration
and PSF convolution to gas emission, and applies Pa-alpha-reference attenuation
to the assumed band-integrated stellar continuum. The nuclear continuum retains
its separate normalized SED/extinction prescription. This simplified foreground
screen does not compute dust mass, scattering, re-emission, or energy balance.
It does not derive the illumination opening angle or change the geometric label.

The component archive saves intrinsic tau and coverage maps, plus output-pixel
area averages of tau, coverage, and transmission. Transmission is averaged as
mean(exp(-tau)), not exp(-mean(tau)). These maps describe the screen before the
PSF; they are not emission-weighted cone flux loss or a fitted observational
extinction map. Dust mass inference would require grain opacity and geometry.
The `smooth_control` has no applied screen, although its total flux is still
matched to the saved cone component. Other variants inherit the screen effects
on their retained components. Cone-free controls remove the full cone component
and keep the same other emission, baseline and additive noise.

Configurations without `foreground_dust` retain the legacy gas-correlated screen
and original texture defaults. New physical-angle experiments explicitly select
the corrected projection. Versioned v2 configs are retained as dated recipes.
Use a fresh v3 experiment directory; existing frozen jobs retain their source.

## Measurements and descriptive categories

Each row records both `complexity_regions.whole_annulus` and
`complexity_regions.cone_annulus`. The common aperture is reference-valid
0.3 < radius < 3.3 arcsec; the cone aperture intersects the nominal mask.
Linear post-PSF C is cone-associated gas; B contains the explicitly listed
other components, including possible AGN-powered host/nuclear light. Component
ratios exclude the empirical baseline and noise. Archive reconstruction checks
remain mandatory. The old `complexity` field remains the whole-annulus alias.

Record M=sum(B)/sum(C), C/(C+B), area fraction B>C, structure at Gaussian sigmas
0.1/0.2/0.4 arcsec, cone signal/pixel-sigma, and projected extent/optical PSF core
FWHM separately. See [complexity definitions](F187N_COMPLEXITY.md). Noise remains
the prescribed reference-ERR-based correlated template, not a calibrated
integrated-significance model. Dust descriptors are area-mean tau, transmission,
and coverage, measured separately in the same apertures from pre-PSF maps.

`f187n_descriptive_categories_v1` uses these explicit, provisional cuts:

- Contrast inside the nominal mask: M<1 `cone_brighter`; 1<=M<3 `other_1_to_3`;
  M>=3 `other_ge3`. Empty/undefined cases stay separate.
- Area-mean transmission T: T>=0.8 `weak`; 0.5<=T<0.8 `moderate`; T<0.5
  `strong`. Use the cone aperture for positives and the whole annulus for
  negatives. The smooth control is `not_applied`.

These labels summarize measured conditions; they are not empirically validated
classes or a prediction of U-Net performance. Patchiness remains continuous.
For cone-free cases, M and cone geometry/extent are undefined, and the contrast
category is `cone_free`. Their obscuration descriptor refers to retained other
emission's foreground screen, not an absent cone.

## Automatic reports

Generation writes `dataset/diagnostics/generation-JOBID/` with:

- `descriptors.csv`: one row per variant with measured values, categories,
  split/group identifiers, and the sampled generation controls.
- `summary.json`: definitions, category counts by variant, ranges, manifest/code
  hashes, and plot/table hashes.
- `population.png`: contrast coverage, contrast versus other-light structure,
  foreground transmission versus contrast, and resolution versus signal/noise.
  It uses full positives once per independent group, not every matched variant.

After training, `training/model/complexity_diagnostics/` adds a plot of held-out
positive IoU versus measured contrast and transmission, and opening-angle error
versus contrast. Failed angle fits remain undefined; the plot displays counts.
Separate panels show negative false-positive area against retained-light
structure/transmission and its distribution, comparing U-Net and baseline.
Negative statistics also remain separate in the test report. The reporting
command checks the training manifest hash before combining results.

```bash
venv/bin/python scripts/cloudy/summarize_f187n_population.py \
  --dataset /path/to/dataset --output /fresh/report/directory
```

Add `--training-report /path/to/training/model/test_report.json` for performance
plots. Reports do not alter images, labels, spectra, or selected checkpoints.
A modest randomized pilot need not cover every combination; inspect measured
coverage before interpreting performance trends or claiming a balanced
population. Shared calibration/spectra still limit generalization claims.

## Local verification

Three groups/12 variants were generated in `data/cloudy/f187n_pipeline_check_v3`.
The CSV/figures in `results/cloudy/f187n_population_descriptors_v3` and
`results/cloudy/f187n_performance_descriptors_v3` were inspected. The one-epoch
model is a computation test with excessive positive area, not an accuracy
result. The physical checks in `results/cloudy/f187n_v3_physical_checks` confirm
unchanged masks/coverage when only foreground depth changes, unit transmission
for zero foreground depth, and bit-identical legacy input/mask reproduction.
Unit tests cover screen endpoints, independent controls, averaging transmission
after exponentiation, strict recipe paths, cone-local measurements, categories,
and undefined negative descriptors. Full population coverage and ARC GPU
execution remain pending. These research outputs remain ignored by Git.
