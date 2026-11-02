# Existing narrow wavelength cube → synthetic narrowband image

Use `scripts/cloudy/render_synthetic_filter.py` to weight an existing cube
through a custom passband and save one 2D image. This is a synthetic imaging
measurement, not a claim that this exact filter exists on JWST.

The default is a unit-throughput rectangular filter covering the entire cube.
Choose a narrower rectangle, a Gaussian passband, or a supplied response table.
All passband wavelengths are observed-frame Angstrom, matching the existing
Cloudy generator's `wavelength_angstrom` axis. Redshift and spatial blur already
in the cube are retained. No additional spatial convolution, galaxy components,
noise, continuum subtraction, or image normalization is applied.

## Example: select the observed [O III] 5007 region

```bash
venv/bin/python scripts/cloudy/render_synthetic_filter.py \
  data/cloudy/ngc1068_f502n_v2/cube_1069.npz \
  --tophat 5005 5047 \
  --output results/cloudy/synthetic_filters/oiii_5007_tophat.npz \
  --preview results/cloudy/synthetic_filters/oiii_5007_tophat.png
```

The example bounds target the longer-wavelength member of the [O III] doublet
in this already redshifted cube. These bounds are a custom choice, not a NIRCam
passband, and may exclude emission in other galaxies or at other velocities.

Replace `--tophat 5005 5047` with `--gaussian 5025.83 10` for a filter centered
at 5025.83 Angstrom with a 10-Angstrom FWHM. Gaussian filters have explicit
support at ±4 sigma (about 0.0063% of the ideal Gaussian area is omitted).
Omit all filter options to integrate the full cube, or use `--throughput file.csv`
for a two-column table: wavelength in Angstrom, throughput in [0,1]. CSV headers
must be commented with `#`, as in the project's throughput files.

## Calculation and outputs

The source is a `(wavelength,y,x)` array of linear F_lambda samples. Integration
uses `integral F_lambda T(lambda) lambda/(h*c) d(lambda)`, with piecewise-linear
interpolation and exact quadrature for that representation. A rectangular
filter's edges are integrated exactly even when they fall between source samples.
Source spectral resolution still limits the accuracy of unresolved lines.

The NPZ contains `image`, optional unchanged `mask`, wavelength-channel `weights`,
the source wavelength axis, the passband curve, and JSON `metadata`. A matching
JSON sidecar records the filter definition and source metadata. The optional
preview shows the spectrum, response and an asinh display of the image;
the saved numeric image remains linear. Existing outputs are not overwritten.

Normalized Cloudy flux produces relative photon response, not calibrated
JWST counts or MJy/sr. Continuum and neighboring lines contribute if they are
present inside the passband. No labels or geometric masks are filtered.

Passbands extending outside the source cube are rejected. A wider spectrum is
not needed for a passband fully inside the working narrow cube. Do not supply
per-bin integrated flux, independently normalized channels, or log/asinh
training tensors. Metadata is checked for nonlinear transforms and non-observed
wavelength frames; without metadata the wavelength axis is assumed observed.

```bash
venv/bin/python -m unittest discover -s tests -p 'test_synthetic_filter*.py'
```
