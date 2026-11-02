# Reference-constrained F187N population pilot

## Purpose and status

Use the real NGC 1068 F187N cutout to constrain coarse brightness properties,
then generate different cone/host geometries with known geometric labels.
This is a four-image exploratory pilot, not a validated physical population,
posterior distribution, training dataset, or train/validation/test split.

The current deliverable is `results/cloudy/f187n_population_pilot_v3`.
Its `population.png` shows the four images and masks on common display scales.
`reference_comparison.png` compares the real image, previous prototype and
reference scenario, with their mean radial brightness profiles.

## What is fitted and what is assumed

The reference template is the existing component-separated F187N pilot v2.
Convolve/integrate the gas and continuum as before. Form four band-image
populations: cone gas, low-U host gas, assumed stellar continuum, and assumed
unresolved nuclear continuum. Normalize templates by their fixed reference
99th percentile (maximum for the unresolved source) solely for numerical scale.
A nonnegative constant baseline is a fifth template.

Average valid pixels in 18 annuli from 0 to 3.7 arcsec. Missing/invalid central
pixels are excluded, not interpolated. Fit nonnegative amplitudes by minimizing
squared fractional residuals of these annular means. Every annulus is a fit
constraint: none is an independent validation observation. No pixel locations,
filaments, cone angle, orientation, cloud texture or segmentation mask are fitted.
No ERR-based chi-squared or statistical uncertainty is claimed. The baseline
can contain unseparated galaxy emission; it is not measured empty sky.

The unconstrained fit preferred zero cone amplitude. Its nuclear point source
also overpredicted peak brightness when extrapolated into the central region.
Those outcomes are retained in `calibration.json`; they do not demonstrate
that the real galaxy lacks an ionization cone or establish a nuclear flux.
They expose inadequacies and degeneracies of these templates and summaries.

For a **deliberate cone-containing scenario**, fix the cone coefficient at 20%
of the previous prototype amplitude. This is an illustrative assumption, not
a measured fraction or fitted lower bound. Fit the host, stellar and baseline
amplitudes while capping the reference point-source template peak at the largest
valid real SCI value. That cap is a descriptive guard against an excessively
bright point source, not an observational determination of the unresolved core.
The extended central emission remains poorly reproduced.

For the saved reference, fractional annular RMS is approximately 0.429 before
and 0.345 after this constrained adjustment. The unconstrained fit gives about
0.235, but is unsuitable as our cone-containing reference scenario. These are
in-sample descriptive residuals, not accuracy, confidence or segmentation scores.

## Population generation

Reuse the reference template normalizations, fitted coefficients and baseline
across samples. The renderer does **not** match each new sample to the real
image's percentiles. Change the random cloud realization, cone position angle,
inclination, opening angle and extent, disk orientation and flattening,
foreground dust and population amplitudes. Keep real F187N throughput, redshift,
PSF, pixel scale and ERR-derived noise-amplitude map fixed for this first pilot.
The same geometric cone label rule is used throughout.

The four parameter choices are recorded in `configs/*.json`, with a manifest
and per-sample configuration/metadata. They are hand-chosen exploratory scenarios,
not observationally justified parameter ranges or population draws. This tests
the generator's variation and data interface; it does not establish realism.
All samples share a calibration galaxy and cached CLOUDY grid, so random image
splits would not establish independent physical or observational generalization.
The real SCI image is used for calibration/comparison, not pasted into the
synthetic image or used as its segmentation label.

## Preserved diagnostic trials

`f187n_population_pilot_v1` records an initial radial-only fit excluding the
central 0.15 arcsec. It removed cone emission and extrapolated a bright point
source. `f187n_population_pilot_v2` includes valid core pixels and a fixed
nonzero cone, but leaves the point source unbounded. Both are diagnostic trials,
not recommended population datasets. Their outputs are preserved for audit.

## Reproduce and verify

```bash
venv/bin/python scripts/cloudy/build_f187n_population.py --render
venv/bin/python -m unittest discover -s tests -p 'test_f187n*.py'
```

The script refuses an existing output directory; use `--output` with a new path
for a fresh run. Without `--render`, it saves the fit, comparison and sample
configurations without running the image renderer. The renderer accepts optional
`calibration` and `population_amplitude_factors` keys, and preserves the older
percentile-matching behavior for configurations without `calibration`.

## Remaining scientific work

The next useful constraints are continuum-sensitive complementary bands and/or
spectroscopy, an appropriate extended central-light model, and a converged
infrared CLOUDY grid with a dedicated star-forming H II population. Existing
first-iteration warnings, censored weak lines, illustrative geometry, assumed
continuum spectra, empirical flux units, fixed reference PSF and approximate
noise remain unresolved. Broader independent models and galaxies are needed
before training/evaluating a segmentation model for scientific claims.
