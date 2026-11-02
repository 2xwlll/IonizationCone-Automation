# NGC 1068 spectral constraints: first observational anchor

## What was actually obtained

Published nuclear emission-line fluxes, equivalent widths and their reported
errors were transcribed from Martins et al. (2010), MNRAS 406, 2168, Table 1.
Selected lines in adjacent south Ap01 and north Ap09 apertures, retaining main
and secondary velocity-component identities, were transcribed from Tables 2-3.
These numerical measurements are in `data/spectra/ngc1068_martins2010_v1`, with
source URL/DOI, observation date, aperture/instrument information, caveats and
checksums. They are not pixels digitized from figures and are not a downloaded
sampled spectrum. Main and secondary components are not silently summed.

Source: https://academic.oup.com/mnras/article/406/4/2168/1017118
DOI: https://doi.org/10.1111/j.1365-2966.2010.17042.x

The observations used IRTF/SpeX SXD on 2007-10-30, a 0.8 x 15 arcsec north-south
slit, approximately 1.3-arcsec seeing and 360 km/s spectral resolution. The
nuclear extraction spans 1.6 arcsec along the slit; ordinary adjacent extractions
span 0.4 arcsec. The specially oversampled nuc_a--nuc_g subapertures are not used
here. Exact off-nuclear aperture-center registration has not been established.
Neither north nor south aperture is assumed to be a pure excitation class.

**Pa-alpha could not be measured reliably in this source.** The paper identifies
both an order gap and strong telluric absorption. No Pa-alpha measurement is
invented, inferred from a plotted feature, or used as an observed fit constraint.
The paper's nominal 0.8--2.4 micrometer coverage is not continuous reliable
coverage of the entire F187N passband.

## Archive assessment

`find_ngc1068_spectroscopy.py` queried public MAST metadata around the nucleus:

- No JWST/NIRSpec records returned within 0.05 deg with the saved criteria.
  This is a query result, not an exhaustive claim about every archive.
- 94 HST/NICMOS observations returned within 0.03 deg; none had a grism filter.
- 16 F187N/F190N image records are saved as candidate complementary line/
  continuum data. These include archive associations/exposures, not 16 established
  independent science images. Their raw/core quality, throughput, alignment and
  usable products have not been inspected in this step.
- The STScI AGN-atlas NGC 1068 template is a composite model with listed coverage
  only to 11000 Angstrom, not a measured Pa-alpha spectrum.
- Brown GALSEDATLAS uses model-filled gaps and matched galaxy photometry. It is
  not accepted here as a measured nuclear Pa-alpha spectrum.

Metadata and source assessments are preserved in
`results/cloudy/ngc1068_spectroscopy_audit_v1`.

## New converged AGN pilot

`build_ngc1068_spectral_grid.py` calculates three AGN-photoionized models with
log U=-3, -2.25, -1.5, fixed log n_H=3, solar abundance scaling, one ISM grain
prescription, no PAH and stop log column=21.5. They request iteration to
convergence and explicitly saved emergent infrared lines, including Pa-alpha,
He I/He II, Pa-beta, Br-gamma, Pa-gamma, Fe II, Si VI and optical diagnostics.
They use the restored official CLOUDY C25 final executable. Raw inputs/outputs,
executable hash, iteration counts, failure status and output hashes are retained.

A successful exit, final saved iteration >=2, finite nonnegative saved line
values and absence of explicit convergence-failure phrases are checked before
packing a compact grid. Inspect the final cautions as well. This is a small
fixed-density pilot, not a replacement for all 81 earlier models or a complete
physical parameter space. The earlier images remain unchanged and continue to
carry their recorded limitations.

Line identifiers are explicitly entered as air wavelengths. The compact grid
retains those identifiers; all photon calculations convert them to vacuum.
The saved continuum wavelengths already use vacuum and are not converted again.

## Fitting and prediction

`fit_ngc1068_spectral_constraints.py` compares the nuclear measurements with the
three new AGN models. It fits a nonnegative observer-flux scale and foreground
optical depth at Pa-alpha, with a lambda^-1.3 attenuation prescription, using
Pa-beta, Br-gamma, Pa-gamma, He II 1.012, Fe II 1.257/1.643 and Si VI 1.962.
It also compares each regional main kinematic component to the same scenarios,
with independent scales/attenuation and the subset of measured diagnostic lines.
Nuclear and regional table units differ by ten and are explicitly converted.

The He I 1.083 observed blend is excluded: the model target is not established
as the full measured blend. Molecular hydrogen is not fitted by this initial
AGN grid; the H2 S(3) feature is additionally blended with broad Si VI in the
source discussion. The nuclear aperture contains mixed light, not a clean
unresolved nuclear spectrum or pure single-zone AGN cloud.

Residuals are weighted by reported flux errors. Shared calibration covariance,
atomic/model systematics and multiple phases are not included. The weighted
residual sum is descriptive, not a validated chi-square likelihood. No posterior
probabilities or physical confidence intervals are claimed from three models.
A best available scenario can still be a poor fit.

For the continuum, F_line/EW estimates the local continuum underneath each
published feature. These are **derived continuum proxies**, not measured
spectrum pixels. The fit uses hot-dust Planck spectra, a fixed F_nu~nu^-0.5
power law, and a 4500 K Planck proxy for stellar light. This is not a stellar
population synthesis fit or unique decomposition. Flux/EW covariance is unknown;
illustrative propagated errors and a 30% floor supply descriptive weights only.
Published nominal rest-like wavelengths and rest EWs are assumed for this first
calculation; raw spectrum wavelength/EW frame metadata still needs confirmation.

The modeled continuum and selected predicted lines are photon-integrated through
the measured F187N curve. Pa-alpha in these calculations is a prediction, not a
measurement. Bulk velocities -600/0/+600 km/s with assumed sigma=85 km/s show
bandpass sensitivity, not fitted nuclear velocity errors. No full, continuously
observed near-IR spectrum has been reconstructed. No total stellar/AGN fraction,
star-formation rate, full luminosity or spatial map is inferred from this fit.

## Reproduction and outputs

```bash
venv/bin/python scripts/cloudy/build_ngc1068_spectral_grid.py
venv/bin/python scripts/cloudy/fit_ngc1068_spectral_constraints.py
venv/bin/python -m unittest discover -s tests -p 'test_spectral_constraints.py'
```

Use fresh `--output` directories for new scientific runs; existing outputs are
refused. The primary fit output is `results/cloudy/ngc1068_spectral_constraints_v1`:

- `spectral_constraints.png`: derived continuum proxies and nuclear line test.
- `regional_line_constraints.png`: nuclear and adjacent north/south line tests.
- `fit.json`: parameters, residual scores, regional fits, explicit assumptions,
  source hashes and conditional F187N predictions.
- `line_comparison.csv`: measured versus modeled nuclear line fluxes.
- `predicted_line_band_contributions.csv`: model line fluxes and F187N response.
- `continuum_proxies.csv` and `modeled_continuum.npz`: distinct inferred local
  continuum data products and modeled curves. They are not raw observed spectra.

## What this does not justify yet

Do not update the image amplitudes or claim a measured F187N decomposition from
this pilot. First inspect line mismatches, broaden the gas/stellar models, obtain
usable calibrated spectra or complementary Pa-alpha/continuum imaging, register
apertures and match PSFs. The 2007 spectra and 2024 NIRCam image also differ in
epoch, and the real NIRCam core has invalid pixels. Only then should a physical
spectral calibration inform a new image version. The work remains 1D spectral
constraints plus 2D projected images; no 3D galaxy model is introduced.

## Results of this saved first pass

All three default-dataset AGN runs completed after three iterations, with no
cautions in their end summaries. The separate Tayal18 control also completed
after three iterations with no final cautions. Each saves 21 high-precision
lines. These are four new calculations, not four independent observed spectra.

The lowest residual among the default scenarios occurs at the highest tested
log U=-1.5. It is not a successful full line fit: Fe II and Si VI are badly
underpredicted, with the largest discrepancy about 8.48 times the reported
measurement error. Shared calibration/model uncertainties prevent treating this
number as a formal detection significance or the grid minimum as an inferred
ionization parameter.

The Fe II issue has an atomic-data component. The supplied Smyth19 rates imply
an optically-thin intrinsic energy ratio I(1.257)/I(1.644) of 0.0664, while the
supplied Tayal18 rates imply 1.960. The lower levels are indices 1 and 6 and the
shared upper level is index 10, checked in both supplied energy files. Their
emergent ratios at identical cloud conditions are 0.0605 and 1.785, compared
with the measured nuclear ratio 1.264 (which includes attenuation and mixed gas).
The controlled rerun recalculates thermal balance; raw inputs and both datasets'
outputs remain preserved. Neither dataset is certified as correct by this test.
The absolute iron and coronal-line mismatch persists, so it cannot be hidden
by switching atomic data or arbitrarily scaling individual lines.

`atomic_dataset_control.png` and `feii_atomic_sensitivity.json` document the
comparison. `check_feii_atomic_sensitivity.py` reproduces it after the control
grid is available. The builder accepts `--log-u` and `--fe-dataset`; the default
three-model calculation remains reproducible without changing its parameters.

The best simple continuum-proxy curve uses a 900 K Planck component plus power
law and zero coefficient for the 4500 K stellar proxy. This is not a measured
dust temperature or evidence that stars are absent: the proxies, restricted
basis, stellar absorption and continuum-component degeneracy preclude that
interpretation. A proper SSP fit requires suitable sampled stellar features.

No synthetic image, segmentation mask or checkpoint was replaced or trained.
The next model refinement should explore additional ionization phases, density,
column, depletion/grain physics and appropriate atomic data, alongside better
continuum observations and a usable Pa-alpha/continuum constraint. A model/data
mismatch is recorded rather than edited out of the measurements.
