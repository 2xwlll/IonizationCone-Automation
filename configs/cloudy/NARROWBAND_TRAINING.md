# Active direction: 2D [O III] narrowband images

The U-Net input is one `(256,256)` F502N brightness image with a matching hard
geometric cone mask. Internal wavelength samples are used to calculate the
image, not passed to the model. The older cube-training experiment is retained
for reference.

## Wavelength integration

For the tabulated HST/WFPC2-PC F502N total-system response, calculate

```text
image(y,x) proportional to integral F_lambda(y,x) T(lambda) lambda/(h*c) d(lambda)
```

Wavelengths are observed-frame Angstrom. The source generator already applies
the galaxy redshift and gas velocity shifts, so the renderer does not apply a
second redshift. The filter response already represents telescope/instrument
throughput, so another detector-efficiency curve is not multiplied in.

The implementation integrates piecewise-linear source spectra and throughput
using two-point Gaussian quadrature on their merged wavelength breakpoints.
This integrates their product times wavelength exactly for that interpolated
representation, including the correct boundary weights and nonuniform grids.
Sampling a continuous line on a finite grid still introduces interpolation
error. The tests compare 0.5-Angstrom Gaussian line samples at velocities
-900, 0, and +900 km/s against a fine wavelength reference. Significant line
responses agree within 0.3%; negligible off-band tails use an absolute
efficiency error criterion.

The old internal 4965–5040-Angstrom grid omits about 0.197% of the tabulated
photon response for flat F_lambda and can truncate the redmost outflow line.
The new internal grid is 4960–5065 Angstrom with 211 samples (0.5-Angstrom
spacing). It covers the full saved throughput table, 4966–5060 Angstrom.
Default rendering rejects insufficient coverage rather than extrapolating
missing line flux. `--allow-partial-bandpass` is an explicit legacy diagnostic
option that reports and records the missing response. The saved filter table
has small nonzero edge values; coverage is relative to that finite table, not
a guarantee of coverage of every possible instrument-response tail.

The pilot uses spatial oversampling 1 to limit memory on this computer.
Supersampling can be increased on a machine with more memory; this changes
spatial discretization, not the passband formula or final image dimensions.

## Reproduce the current pilot

First resample the existing 81 Cloudy continuum outputs onto the wider internal
grid. Line luminosities and physical grid parameters remain the cached values;
no Cloudy physics calculation is rerun.

```bash
venv/bin/python scripts/cloudy/resample_cloudy_grid.py
```

The resampler refuses to overwrite an existing output. After its first run,
reuse `data/cloudy/ngc1068_f502n_v2/grid.npz` directly:

```bash
venv/bin/python scripts/cloudy/generate_oiii_cube.py \
  --config configs/cloudy/ngc1068_f502n_v2.json \
  --grid data/cloudy/ngc1068_f502n_v2/grid.npz \
  --output data/cloudy/ngc1068_f502n_v2/cube_1069.npz \
  --seed 1069
venv/bin/python scripts/cloudy/render_hst_f502n.py \
  data/cloudy/ngc1068_f502n_v2/cube_1069.npz \
  --output data/cloudy/ngc1068_f502n_v2/f502n_1069.npz \
  --seed 1069
venv/bin/python scripts/cloudy/validate_f502n_render.py \
  data/cloudy/ngc1068_f502n_v2/f502n_1069.npz
```

The renderer saves final image/mask `.npy` files in
`data/cloudy/ngc1068_f502n_v2/f502n_training_pair/`. Neither the geometric mask
nor training labels are convolved through the filter or spatial PSF.
These files still need batch generation and a fixed train/validation/test
split before a large training run.

## Spatial blur and remaining calibration

Wavelength weighting and spatial convolution are separate operations. Gas
already blurred in the source cube receives only the additional Gaussian width
needed to reach the requested target FWHM. Newly added continuum receives the
full target Gaussian width. The target must not be narrower than the source.
The PSF is a constant Gaussian approximation, not a measured WFPC2 PSF and not
a model of spatial/color/field dependence or every detail of drizzling.

The stellar/nuclear continuum is synthetic morphology with a configurable
local F_lambda power-law slope and a smooth foreground attenuation law. It is
integrated through the full filter response, but it is not yet a calibrated
stellar-population/dust-material spectrum. Cloudy continuum and dust mixed
inside the emitting clouds are distinct from that foreground screen.

The source flux has a shared linear normalization. Passband integration is
performed before any asinh training/display transformation. Output brightness
is empirically matched to the real reference array; raw response maps and
weights are saved for inspection. Absolutely calibrated electron counts still
require source luminosity/covering factors, collecting area, gain, and exposure
conventions. Reference-based Gaussian correlated noise is still an
approximation, not a complete Poisson/read-noise detector simulation.

## Checks

```bash
venv/bin/python -m unittest discover -s tests -p 'test_synthetic_photometry.py' -v
```

The analytic flat-continuum and monochromatic-photon checks, velocity-shifted
line checks, filter-coverage rejection, linearity, and saved-map reconstruction
verify wavelength integration. They do not establish full galaxy realism or
U-Net accuracy.

The photon-counting treatment follows the STScI synthetic-photometry approach:
https://synphot.readthedocs.io/en/stable/synphot/overview.html
