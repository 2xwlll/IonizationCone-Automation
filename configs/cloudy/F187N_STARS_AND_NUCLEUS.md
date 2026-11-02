# Stellar-ionized gas and spectral nuclear continuum

## Interpretation of the synthetic images

CLOUDY supplies gas line and nebular-continuum spectra. Our forward model places
those spectra in spatial gas populations, adds assumed continuum sources,
integrates the measured filter response, and applies a telescope PSF, sampling
and approximate noise. CLOUDY itself does not produce these galaxy morphologies.
A convincing appearance is not a validation of the gas model or decomposition.

Star formation belongs to a disk/ring/cloud population, not a Boolean complement
of the AGN cone. Distinct populations can overlap in projection or coexist with
mixed excitation. NGC 1068's prominent star-forming pseudo-ring is at roughly
15--20 arcsec radius, outside the present 7.9-arcsec-wide central cutout. A toy
central H II disk is not a reproduction of that ring.

Sources:
- https://academic.oup.com/mnras/article/300/2/388/1037547
- https://academic.oup.com/mnras/article/502/2/3021/6119921
- https://academic.oup.com/mnras/article/487/3/4153/5513463

## H II-region grid

`build_hii_f187n_grid.py` prepares a separate stellar-ionized grid rather than
relabeling low-U AGN models as star formation. The initial grid spans 35000 and
45000 K blackbody ionizing sources, log U -3.5/-2.5, log n_H 2/3, solar abundance
scaling and one grain prescription. These are exploratory choices, not measured
conditions in NGC 1068. A blackbody is a controlled first approximation to hot
stars, not a population synthesis spectrum with ages and an IMF.

Inputs request `iterate to convergence`, high-precision saved emergent Pa-alpha,
two nearby He I features, H-beta and [O III] diagnostics, and continuum. This
is still an incomplete faint-line list. The packer rejects missing completion
markers, nonzero exit codes, explicit convergence failures and final saved line
intensities from fewer than two iterations. Review raw cautions as well; the
checks do not establish every aspect of physical convergence or correctness.
No failed scientific calculation is automatically retried.

Prepared-only inputs are under `data/cloudy/hii_f187n_prepared_v1`. A directory's
`manifest.json` status and presence of `grid.npz` distinguish preparation from
completed calculations. A source spectral shape alone is not calculated gas.
The official C25 final release is restored under `data/tools/c25.00`, separately
from the old C25.00_rc2 cached AGN outputs. The download URL and SHA are recorded
in `data/tools/cloudy_download.json`. Raw outputs and executable SHA are retained
for new calculations. Mixing the new pilot with old AGN spectra does not resolve
the old AGN first-iteration warnings or release differences.

```bash
venv/bin/python scripts/cloudy/build_hii_f187n_grid.py \
  --cloudy data/tools/c25.00/source/sys_gcc/cloudy.exe \
  --output data/cloudy/hii_f187n_calculated_v1
```

The optional `star_forming` renderer configuration loads a completed grid, picks
an explicitly recorded model index, and adds its lines and nebular continuum
in an independent clumpy disk or ring. It uses its own rotation and line width,
a shared foreground screen, the F187N throughput, and the same PSF/pixels.
`relative_to_agn_gas_p99` is an assumed relative emission normalization, not a
star-formation rate, emitting area, physical covering factor or fitted flux.
The mask still identifies the AGN cone alone. A previous four-component
calibration is deliberately rejected when this new population is enabled.

## Nuclear spectrum

`nuclear_sed.py` combines normalized Planck B_lambda(T) and a power law with
F_nu proportional to nu**alpha, hence F_lambda proportional to lambda**(-alpha-2).
A power-law optical-depth screen is applied; the resulting shape is normalized
to one at the chosen rest wavelength. Absolute luminosity remains separate.
It is redshifted with the F_lambda density Jacobian and photon-weighted through
actual system throughput, including F187N. The source is unresolved and receives
the saved PSF. It is not a spatially resolved radiative-transfer torus model.

The initial demonstration uses T=1100 K, a 90% hot-dust fraction at 1.8751 microns
before the shared screen, alpha=-0.5, tau(reference)=1 and exponent 1.3. These
are explicit assumptions, not fitted measurements. In the normalization used,
the screen changes spectral color; absolute extinction is degenerate with the
free source amplitude. No broad Pa-alpha line is inserted automatically into
this obscured Seyfert-2 spectrum. Gas lines remain in their gas populations.

`preview_nuclear_sed.py` saves the spectrum, predictions for all 29 cached imaging
filters, and a diagnostic plot in `results/cloudy/nuclear_sed_pilot_v1`.
`ngc1068_f187n_nuclear_sed_v1.json` demonstrates the renderer integration without
reusing the old reference-template calibration. Its nuclear normalization is
still the old illustrative flat-reference-band amplitude multiplied by the
spectrum's bandpass color factor; this is not an independently measured flux.
Adding a spectral shape need not make the F187N nucleus dramatically brighter.

Primary motivation for obscured hot nuclear dust:
- https://doi.org/10.1051/0004-6361/201936255
- https://www.nature.com/articles/s41586-021-04311-7

CLOUDY command examples:
- https://data.nublado.org/workshop/14Belfast/QuickStart.pdf
- https://www.pa.uky.edu/~gary/CloudyWorkshop/CloudyWorkshop/18Thailand/day1/1_1_Cloudy_intro.pdf

## Checks

```bash
venv/bin/python -m unittest discover -s tests -p 'test_hii_population.py'
venv/bin/python -m unittest discover -s tests -p 'test_nuclear_sed.py'
```

These verify input source/iteration settings, rejection of first-iteration saved
fluxes, analytic photon-wavelength scaling for an independent H II population,
nuclear normalization, the flat-F_lambda redshift response and dust color.
The analytic fixtures are not outputs of actual CLOUDY runs.

## Wavelength correction and run state

The older pilots used CLOUDY printed air line wavelengths directly with the
JWST vacuum throughput curve. That convention was incorrect. New versioned
configurations set `convert_cached_air_lines_to_vacuum: true` for the cached AGN
lines, using the same Peck & Reeder two-step conversion as CLOUDY C25's
`source/cddefines.cpp`. Continuum grids already use vacuum and are not converted.
Pa-alpha 18751 air becomes 18756.11985 Angstrom vacuum. At the current redshift,
zero velocity and sigma=85 km/s, the corrected F187N photon response is 0.93231
times the previous value. This changes the line selection by about 6.8% for that
specific example; it is not a universal image-flux correction.

The new H II line list marks its requested wavelengths explicitly as `air`,
then packs vacuum wavelengths into `grid.npz`. Older output images and configs
are preserved for reproducibility; they are not silently relabeled corrected.
The nuclear-spectrum v2 image includes the cached-gas vacuum correction.

The first H II launch under `data/cloudy/hii_f187n_calculated_v1` failed because
safe archive extraction had omitted 30 internal symbolic links, including the
Fe II energy-file selector. The official data files were present. All links
were subsequently validated to remain inside the extraction root and restored;
the failed raw output remains intact. On review, the user clarified that the parent instruction restricting automatic
scientific-job restarts was intended for OpenClaw. The repository `AGENTS.md`
records that scope; Codex may repair and rerun within this authorized task,
preserving failed outputs and using a fresh output directory.

## Corrected pilot configuration

The two-model pilot run is under `data/cloudy/hii_f187n_calculated_v2`; it uses
35000 K stellar ionizing sources at log U=-3.5, log n_H=2 and 3. The wider
eight-input design remains prepared separately. Check the run manifest for
completion before using the grid.

`ngc1068_f187n_hii_nucleus_v1.json` enables the dedicated grid plus the explicit
nuclear spectrum and vacuum correction. The illustrative H II disk has a
1.6-arcsec exponential scale radius, axis ratio 0.65 and PA=100 degrees. Its
unblurred gas p99 is normalized to 0.3 times the AGN/host gas p99. Neither that
relative amplitude nor the placement is measured; this configuration does not
pretend that the outer starburst ring is inside our central field. The prior
population fit is not reused. It needs new observational constraints once the
extra population is included.

## Completed two-model result

Both models in `data/cloudy/hii_f187n_calculated_v2` completed with successful
CLOUDY exits after three iterations and no cautions reported in the end summaries.
They passed the saved-output checks. Pa-alpha/H-beta is 0.33512 and 0.33272;
these are outputs of the two calculations, not assumed input ratios. The compact
grid stores vacuum Pa-alpha at 18756.11985 Angstrom and the corresponding
line-free outward continua. Raw inputs, outputs, source/executable hashes,
configuration and iteration counts are retained. These results do not resolve
the old AGN grid's first-iteration warnings.

The renderer consumes model index 0 (35000 K, log U=-3.5, log n_H=2) for the
first hybrid image under `results/cloudy/ngc1068_f187n_hii_nucleus_v1`. Its H II
emission is present both inside and outside the projected cone. The bright
nuclear continuum now has an explicit wavelength dependence; neither its
normalization nor the extra H II amplitude has been fitted to observed fluxes.
