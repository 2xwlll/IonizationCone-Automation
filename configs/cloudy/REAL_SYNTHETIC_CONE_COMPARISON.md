# NGC 1068 and one synthetic cone: matched F187N measurement

Measured 2026-10-31. This is a controlled image-analysis comparison of one
observed cutout and one existing synthetic realization. It measures candidate
emission geometry and tests the synthetic label; it does not measure the
intrinsic torus opening or establish neural-network performance.

Follow-up on 2026-11-01: see `NGC1068_CONE_LITERATURE_COMPARISON.md` for the
published kinematic benchmark and an analytic audit showing that the current
projection prescription is not an exact circular-3D-cone projection. The
measurements below still refer to the unchanged original 2D target.

## Inputs and reproducibility

- Observed: `data/2d/jwst_ngc1068/jw03707-o128_t002_nircam_clear-f187n_i2d.fits`,
  SCI extension, cached in `results/cloudy/f187n_reference/nucleus_cutout.npz`.
- Synthetic: `results/cloudy/ngc1068_f187n_soft_gas_v1/image_mask.npz`, generated
  from `configs/cloudy/ngc1068_f187n_soft_gas_v1.json`.
- Both main inputs are 256 x 256 total F187N images at 0.030742469 arcsec/pixel.
  The synthetic input includes its saved baseline and noise. This is not a
  comparison between an optical line map and an infrared image.
- The synthetic configuration uses `data/cloudy/ngc1068_f187n_pilot_v1/grid.npz`
  and `data/cloudy/hii_f187n_calculated_v2/grid.npz`. Those existing inputs were
  not replaced by newer spectral pilots or refitted here. Their previously
  documented limitations still apply.
- The third panel uses the saved, linear, PSF-convolved cone gas components
  alone as a diagnostic. These components are unavailable separately in the
  observed image. Component membership, reconstruction checks, configuration,
  centers, source SHA-256 values and analysis-code hashes are in the report.

Run from the repository root, choosing a fresh output directory:

```sh
venv/bin/python scripts/cloudy/compare_real_synthetic_cones.py \
  --output results/cloudy/real_synthetic_cone_comparison_NEW
venv/bin/python -m unittest tests.test_cone_comparison tests.test_observed_cone tests.test_image_complexity
```

The reviewed output is `results/cloudy/real_synthetic_cone_comparison_v2/`:

- `cone_comparison.png`: north-up overlays, with cyan measured angular support
  and dashed green nominal synthetic geometry.
- `orientation_audit.png`: original array display, a deliberately reversed-y
  display demonstration, and north-up display, with N/E arrows.
- `angular_profiles.png`: directional contrast and the thresholded profiles.
- `measurements.json`: numerical results and all 180 sensitivity trials per input.
- `comparison_arrays.npz`: native arrays, labels, measured masks and profiles.

Version 1 is retained. Version 2 corrects plot-title spacing, moves compass labels
to arrow tips, and makes total-image contrast easier to see; the numerical
measurement method is unchanged. Development logs are in
`results/cloudy/cone_comparison_development_audit/`. All 17 relevant tests pass.

## What is being measured

A cone-like feature should have preferential directions of emission around the
nucleus. We compare that directional excess using the same operator for both
total images, without giving the extractor the synthetic label or expected angle.

The nominal aperture is 0.7 < projected radius < 3.3 arcsec, on the reference-valid
footprint. Measurements use native pixels and their local sky-coordinate matrix;
only the displayed images are resampled north-up. The real center comes from the
cached adopted nucleus coordinate (RA 40.669621667 deg, Dec -0.013294444 deg),
and the synthetic center is the generator's pixel center (127.5, 127.5).

1. Smooth each image by an additional Gaussian sigma of 0.10 arcsec, with valid
   pixel normalization and exclusion of insufficiently supported pixels.
2. Divide the annulus into 12 radial bins and 120 position-angle bins (3 degrees).
   Divide each polar-cell median by its full-ring median and subtract one.
3. Take the median excess across radial bins, then smooth the angular profile
   with a 5-degree Gaussian sigma.
4. Identify contiguous sectors above 20% of the positive profile peak. Retain
   widths from 15 to 160 degrees; rank by summed positive profile. A second
   sector separated by at least 120 degrees can be a counterlobe. No missing
   lobe is supplied from a geometric prior.

Angles are measured east of north. The reported opening is the **full projected
width of a detected sector**. Position angle is its midpoint, not necessarily the
brightest pixel. An axis comparison is modulo 180 degrees; a primary lobe retains
its directed 0--360 degree PA. Inner/outer aperture cuts do not measure the
physical start or end of the cone.

Radial normalization is not continuum subtraction. Total F187N contains multiple
emission contributions, so these are candidate image-based envelopes, not
confirmed pure ionization-cone masks. The positive angular excess is not the
physical ionization parameter, a cone component fraction, or an S/N statistic.
The same extra smoothing does not guarantee identical effective PSFs.

## Results

| Quantity | Observed NGC 1068 | Synthetic total image |
| --- | ---: | ---: |
| Primary lobe PA, east of north | 30.0 deg | 226.5 deg |
| Primary axis, modulo 180 deg | 30.0 deg | 46.5 deg |
| Primary full opening | 84 deg | 117 deg |
| Counterlobe PA / full opening | 199.5 / 57 deg | Not identified at nominal settings |
| Peak fractional angular excess | 0.404 | 0.171 |
| Primary full-opening range across analysis trials | 45--96 deg | 24--126 deg |
| Axis offsets from nominal across trials, modulo 180 deg | -13.5 to +6 deg | -45 to +33 deg |

The trial family varies smoothing sigma (0.06, 0.10, 0.15 arcsec), radial aperture
(0.7--3.3, 1.0--3.3, 0.7--2.5 arcsec), center (nominal and +/-0.1 arcsec separately
in east and north), and relative threshold (0.1, 0.2, 0.3, 0.4). All 180 trials
find a candidate in each main image. These ranges describe method sensitivity;
they are not statistical confidence intervals, and a primary can switch lobes.

The observed image has a larger angular excess and a more stable recovered axis
under this extractor. The synthetic total image gives a 16.5-degree axis error
and a +27.6-degree full-width error relative to its prescribed geometry. This is
evidence of a more difficult geometry-recovery example for this method, not a
validated universal difficulty score or evidence that every real image is easier.

The cone-only diagnostic also fragments into narrow sectors (primary PA 187.5
deg, width 39 deg). Bright gas knots and the radial normalization can therefore
limit recovery even without other components. Contamination alone does not
explain all of the difference.

The earlier HST F502N minus scaled F547M analysis in
`NGC1068_OBSERVED_CONE.md` gave a nominal primary PA 27 deg and full width 120 deg.
That is an independent optical morphology reference, with a different tracer,
aperture and boundary rule. Its approximate directional agreement is useful;
the different widths must not be treated as interchangeable measurements.

## Does the synthetic mask match its gas?

The saved target is the generator's sharp nominal projected geometry. Its input
half angle is 35 degrees, inclination parameter 45 degrees, and axis PA 30
degrees. Under the renderer's existing approximation the projected full opening
is 89.44 degrees. This is not a newly verified exact 3D projection formula.

Within the valid 0.7--3.3 arcsec aperture:

- The stored label and recomputed nominal geometry have IoU = 1.000. There is
  no detected placement/orientation error in this annulus.
- The label encloses **73.8% of the modeled cone-component flux**; **26.2% lies
  outside**. These are linear post-PSF component sums, excluding baseline and
  noise. Soft angular boundaries, off-axis gas and convolution can all matter;
  this measurement does not isolate their individual contributions.
- Inside the label, C/(C+B) = 0.401 and other modeled light exceeds cone light
  in 82.1% of the area. C is cone-associated gas; B is host gas, star-forming gas,
  stellar continuum and nuclear continuum as listed in the report. Other light
  is not necessarily non-AGN-powered. These values are not an observed NGC 1068
  component decomposition.
- The envelope extracted from the synthetic total image has IoU = 0.415 and
  Dice = 0.586 against the nominal target, and encloses 60.4% of the cone flux.
- Thresholding the cone component at C/pixel-sigma >= 1, 3, and 5 gives label
  IoU values 0.534, 0.516, and 0.440. These illustrate label dependence on
  detectability; they are not integrated detection significances or calibrated
  confidence contours. Noise correlations are not modeled by these thresholds.

The mask is correct for the prescribed geometric target, but it does not enclose
all cone-associated light. Retain that distinction when choosing the training
objective. A detectable-emission target would need its own explicit, versioned
label rule. No original image or target was changed in this analysis.

## Why NGC 1068 can look flipped

The cached science cutout equals the original unflipped FITS SCI pixels exactly,
including NaN locations. Its saved coordinate matrix agrees with direct local
FITS WCS evaluation. Native positive y points at PA **63.51 degrees east of
north**, so the vertical array direction is not north.

The transform from a north-up/east-left grid to the native grid has determinant
+1.000: it is a rotation, not a reflection. The existing inspected preview code
uses `origin='lower'`; switching to `origin='upper'` would reverse displayed y.
The middle orientation panel deliberately demonstrates that possibility and is
not evidence that an existing image was actually plotted that way. East being
left in a north-up astronomical map is the stated sky-coordinate convention.

Thus there is no detected flip in the saved data. Native and sky-oriented views
can look different without altering those data. The exact cause in another
specific figure remains conditional on which figure the user means.

## Next scientific use

Review these candidate boundaries against the independent HST line morphology,
especially the faint edges and asymmetric counterlobe. Keep this measurement as
a geometric baseline and retain the sensitivity ranges. For a diagnostic training
pilot, score the model against the explicitly chosen synthetic label and examine
errors versus the separate contrast, patchiness, resolution and noise descriptors
in `F187N_COMPLEXITY.md`. Success on this one pair cannot validate a real-galaxy
segmentation model or establish an observational contamination ratio.
