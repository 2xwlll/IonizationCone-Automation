# Observed NGC 1068 morphology reference

## What was done

On 2026-10-29, `scripts/cloudy/extract_ngc1068_observed_cone.py` used the locally
downloaded **observed** HST/WFPC2 F502N and F547M images to form a signed line
emission proxy, then measured its apparent angular support. It also sampled the
observed JWST/NIRCam F187N image on the same north-up, east-left sky grid.
No U-Net prediction, synthetic cone, or literature opening angle supplied the
candidate orientation or boundaries.

The report is `results/cloudy/ngc1068_observed_cone_v1`. The comparison figure
overlays the candidate HST sectors on all three observed bands and the signed
HST subtraction. Cyan marks the primary northern sector; green marks a weaker
southern sector. These are threshold-defined envelopes, not pure-emission pixel
labels or direct measurements of the torus boundary.

## Measurement and sensitivity

F547M was interpolated onto the comparison WCS and scaled by the ratio of the
F502N and F547M median values in the valid 6–8 arcsec annulus. The resulting
scale is 0.0302442. The annulus contains galaxy light and potentially line
emission: it is not blank sky or an independently fitted continuum SED.
Signed residuals are retained, including oversubtracted regions.

The nominal analysis uses 0.7–3.7 arcsec, 12 radial bins, 3-degree angular bins,
additional Gaussian image smoothing sigma=0.1 arcsec, and angular smoothing
sigma=5 degrees. The radial-bin median residuals are collapsed by another median.
Contiguous angular support above 20% of the profile peak defines candidate
lobes; the primary has the largest sum of positive profile values. No lobe is
invented when no distinct candidate is found.

| Relative angular-profile threshold | Primary sky PA east of north | Apparent full opening |
|---|---:|---:|
| 10% | 33.0 degrees | 156 degrees |
| 20% (nominal) | 27.0 degrees | 120 degrees |
| 30% | 31.5 degrees | 99 degrees |
| 40% | 34.5 degrees | 75 degrees |

These rows hold the other nominal settings fixed. Including fainter angular
emission broadens the measured support. The profile's brightest direction and
the midpoint of the support envelope are different quantities.

Across 56 explicitly specified analysis choices, 55 yield a candidate. Varying
continuum scale (0.8/1/1.2 times the empirical scale), threshold, smoothing,
radial range, and center shifts of 0.1 arcsec gives primary PAs from 22.5 to
37.5 degrees and full widths from 72 to 156 degrees. These are **sensitivity
ranges, not confidence intervals, population priors, or an intrinsic cone-angle
distribution**. The nominal weaker southern sector has PA=192 degrees and full
width=30 degrees; it is not forced to be a symmetric copy of the north.

## What this means for the synthetic data

The existing `ngc1068_f187n_soft_gas_v1.json` uses sky PA=30 degrees, an input
half-angle of 35 degrees, and inclination=45 degrees. Its current projection
prescription produces an apparent full width of about 89.44 degrees. The PA
and projected width fall within the observed *analysis-sensitivity* ranges.
This comparison does not statistically validate that configuration, but it does
not establish a need to replace its angle with the nominal 120-degree envelope.

`ngc1068_observed_cone_reference_v1.json` records this candidate reference and
the comparison. It is an observational-reference record, **not a renderer
configuration**, and is not consumed automatically by the generator.

Use this reference to compare sky orientation, angular profiles, asymmetry,
and how detectable structure changes with radius and threshold. Apply the same
measurement procedure to simulated and observed images where possible. Do not
insert the apparent half-width into an intrinsic-angle field and project it a
second time. Do not paste the observed cone pixels into multiple samples and
treat those samples as independent physical realizations.

**No F187N cone/contaminant fraction or M was measured.** The F187N image here
remains total band light. An optical support envelope cannot establish which
fraction of near-IR flux inside it is cone gas: continuum and other gas can
overlap it. F187N component-amplitude calibration still requires suitable
near-IR continuum/spectral constraints. This is why the synthetic complexity
ratios must not be presented as measurements of NGC 1068.

## Limits and provenance

- The HST data were observed in 1995 and JWST data in 2024. Tracer, extinction,
  spatial response, and epoch differ. Equal-looking or different-looking shapes
  alone do not establish equal or different excitation fractions.
- Alignment uses FITS WCS only, not an empirical registration from matched field
  stars. Center shifts in the sweep are trials, not measured astrometric errors.
- Both optical images receive the same additional blur, not matched empirical
  PSFs. The core is excluded and smoothing is varied, but PSF systematics remain.
- Brightness thresholds are descriptive; neither HST weight maps nor residual
  galaxy structure have been converted into an independent-pixel S/N model.
- The chosen radial limits are analysis apertures, not measured cone endpoints.
- The scenario-frequency image is a stability visualization, not a probability.
- Once NGC 1068 is used to tune simulations, it is a calibration reference.
  Independent scientific validation requires data not used for that tuning.

The report includes source FITS SHA-256 hashes, source header values, code hash,
adopted center, WCS, continuum prescription, every sensitivity trial, and source
limitations. Existing images, old extraction results, and synthetic settings
are preserved.

Historical context: F547M was used to study optical continuum structure in
[Lynds et al. (1991)](https://authors.library.caltech.edu/records/jf1m7-83d33).
The complicated excitation and obscuration of the NLR are documented in
[Kraemer & Crenshaw (2000)](https://arxiv.org/abs/astro-ph/0007018).
These provide context, not measured angles for this extraction.

## Reproduce

Choose a fresh output directory:

```bash
venv/bin/python scripts/cloudy/extract_ngc1068_observed_cone.py \
  --output results/cloudy/ngc1068_observed_cone_v2
venv/bin/python -m unittest discover -s tests -p 'test_observed_cone.py' -v
```

`observed_reference.npz` preserves the resampled observations, signed residual,
validity maps, candidate sectors, angular profile, and analysis-frequency map.
FITS exports preserve the sky WCS. `measurements.json` and `sensitivity.csv`
contain numeric results; `comparison.png` and `sensitivity.png` are review plots.
