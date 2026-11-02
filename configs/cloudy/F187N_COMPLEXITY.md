# F187N cone contrast and spatial structure

Implemented 2026-10-29 as a diagnostic of **existing synthetic component maps**.
No image, mask, CLOUDY spectrum, or training checkpoint is changed. This does
not validate the underlying emission physics or establish a difficulty ranking.

The implementation is `src/simple_utils/image_complexity.py`; the saved-image
adapter/report command is `scripts/cloudy/measure_f187n_complexity.py`.
The adapter currently accepts F187N archives only. It is not connected to the
F502N or historical spectral-cube trainer.

## What is measured

Let C be the sum of components explicitly named `cone_*` in the archive's
`component_names`, and B be the sum of the remaining listed components. B can
include AGN-powered host gas and nuclear continuum; it does not mean non-AGN.
All measurements use linear maps after bandpass weighting and PSF convolution.
Aggregate maps such as `pa_alpha` are not counted again.

The default measurement aperture A is the nominal geometric cone mask,
intersected with 0.3 < radius < 3.7 arcsec and reference-valid pixels. The radius
uses the synthetic generator's image center, not a fitted nucleus. A second
set of measurements covers the entire valid annulus to characterize the wider
scene. The core exclusion is a reproducible pilot choice, not a universal rule.

| Output | Definition | Interpretation |
|---|---|---|
| `other_over_cone` (M) | sum_A(B dOmega) / sum_A(C dOmega) | Competing modeled light relative to cone gas |
| `cone_fraction` | sum_A(C dOmega) / sum_A((C+B) dOmega) | Cone fraction of modeled component light; 1/(1+M) when defined |
| `other_dominant_area_fraction` | Area in A where B > C, divided by area of A | Whether competing light is widespread |
| `structure` for C and B | sum_A(abs(I - G_sigma * I) dOmega) / sum_A(I dOmega) | Structure removed by smoothing at a specified scale |
| `median_cone_signal_over_pixel_sigma` | median_A(C / reference_error) over usable noise pixels | Typical cone signal relative to the saved pixel-noise amplitude |
| `extent_over_optical_psf_fwhm` | Maximum center-to-mask-pixel radius / mean optical PSF core FWHM | Approximate number of optical resolution widths along the projected radial extent |

The empirical baseline and noise are excluded from M and the component fraction.
An empirical baseline may contain unseparated galaxy light: excluding it does
not establish a fraction of all physically emitted light. Maps plus baseline
must reconstruct `clean` before analysis proceeds. Tiny FFT roundoff negatives
up to 1e-12 of each component's maximum absolute value are clipped to zero;
counts and removed absolute pixel sums are recorded. Material negative values,
duplicate components, or failed reconstruction raise an error.

Equal-area pixels cancel in ratios. The utility supports area weighting, while
the F187N adapter uses the documented constant pixel scale. Weighted sums have
empirical MJy units (MJy/sr times sr), not calibrated CLOUDY luminosities.

## Patchiness convention and limits

Gaussian smoothing sigmas are **0.1, 0.2, and 0.4 arcsec** by default; these are
sigma, not FWHM. Smoothing is applied to the full component map before selecting
the aperture, with reflective image boundaries and a four-sigma kernel cutoff.
This avoids manufacturing structure by zeroing pixels outside the cone mask.

The normalized absolute residual is invariant to multiplying a component by a
positive constant. Thus increasing B's amplitude raises M but leaves B's
normalized structure unchanged. Equal total contaminating light can still have
different structure values when distributed smoothly versus in knots.

This first estimator also responds to smooth gradients, the cone's own edges,
and central light smoothed into the aperture. It is not a pure measure of clouds,
is not necessarily bounded by one, and is not a cloud-size estimator. Finite
stamp boundaries can affect the largest smoothing scale; repeat with a larger
stamp or a more restricted aperture when that affects interpretation. Comparisons
require stated apertures, smoothing scales, and PSFs. Fixed angular smoothing
does not guarantee the same physical scale at different distances.

## Noise, resolution, and distance

The renderer uses reference ERR amplitudes in reference-valid pixels. The
reported median C/ERR is a per-pixel summary; it does not combine pixels into
a detection significance or assume independent noise. Invalid/nonpositive ERR
pixels are excluded from this statistic, and their coverage is recorded.
With `--valid-pixels all`, component ratios include all synthetic aperture
pixels, but the noise statistic still uses only usable reference-valid pixels.

PSF width is measured by linearly interpolating the first half-maximum crossings
in x and y through the peak of the saved optical kernel, then averaging those
two widths. Individual widths and the kernel hash are saved. The current kernel
gives a mean core width of approximately 0.0641 arcsec. This is not the full
detector-integrated/drizzled effective PSF, so the extent/PSF ratio is explicitly
approximate. The extent is one-sided radial extent, not tip-to-tip length or a
measured emission extent. Mask contact with the image boundary is flagged.

No distance is inferred automatically from z. To add a physical extent, supply
both an adopted angular-diameter distance and its provenance:
`--distance-mpc VALUE --distance-source 'reference and distance convention'`.
The conversion is L_pc = 4.8481368 D_A_Mpc theta_arcsec. Redshift remains
metadata. This option labels existing angular sizes; it does not simulate a
new observation at another distance.

## First saved report

`results/cloudy/f187n_complexity_v1` contains the soft-gas pilot and the four
population-v3 examples. These have different component recipes and multiple
parameters changing together; they are not a controlled experiment or five
independent observed galaxies.

| Synthetic sample | M | Cone fraction | Area where B > C | B structure at sigma=0.2 arcsec |
|---|---:|---:|---:|---:|
| soft_gas_v1 | 1.515 | 39.76% | 82.53% | 0.111 |
| population reference | 5.902 | 14.49% | 97.18% | 0.104 |
| variant_01 | 13.231 | 7.03% | 99.65% | 0.284 |
| variant_02 | 4.221 | 19.15% | 96.69% | 0.045 |
| variant_03 | 6.323 | 13.65% | 96.88% | 0.044 |

All entries use the default cone-annulus aperture above. The small difference
from the earlier read-only M=1.515038 calculation comes from excluding invalid
reference pixels by default. `--valid-pixels all` reproduces that earlier
aperture policy. These quantities describe the simulations, not NGC 1068's
observed component fractions, and do not establish which image a U-Net finds
hardest.

## Reproduce and verify

Use a fresh output directory; existing directories are refused. Source archives
are read-only. The report saves source/code/kernel SHA-256 hashes, source
configuration and limitations, component membership, aperture rules, and units.

```bash
venv/bin/python scripts/cloudy/measure_f187n_complexity.py \
  results/cloudy/ngc1068_f187n_soft_gas_v1/image_mask.npz \
  results/cloudy/f187n_population_pilot_v3/reference/image_mask.npz \
  results/cloudy/f187n_population_pilot_v3/variant_01/image_mask.npz \
  results/cloudy/f187n_population_pilot_v3/variant_02/image_mask.npz \
  results/cloudy/f187n_population_pilot_v3/variant_03/image_mask.npz \
  --output results/cloudy/f187n_complexity_v2

venv/bin/python -m unittest discover -s tests -p 'test_image_complexity.py' -v
```

`metrics.json` is the complete record; `metrics.csv` has one row per sample and
aperture. `comparison.png` compares M and spatial structure; `sample_XX.png`
shows the saved input, C, B, and C/(C+B). The component panels share a brightness
scale within each figure; the observed input uses its own display scaling.
Display transformations do not enter the measurements.

Checks cover analytic mixtures, scale invariance, equal-flux but different
structure, variable area weights, undefined/empty cases, invalid inputs,
Gaussian PSF widths, reconstruction, and aggregate double counting.

## Using these measurements in the next experiment

Keep geometry and cloud positions fixed while varying one factor at a time.
For an initial amplitude control, multiplying B by a positive factor k gives
M_new = k M_old, without changing normalized B structure. This is a controlled
image-component experiment, not a self-consistent change in AGN luminosity,
ionization, or photon noise. A future rendering control must state which
components change and how the noise model follows those changes.

Evaluate angle error and mask overlap against M, spatial structure, resolution,
and noise separately. No combined score or training thresholds are chosen by
this implementation. Empty masks and zero cone signal are explicit statuses;
undefined ratios are JSON null/empty CSV fields, never invented finite scores.
