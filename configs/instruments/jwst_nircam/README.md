# NIRCam imaging response foundation

This catalog supports all 29 NIRCam imaging filters with configurable galaxy
redshift. Nominal imaging channels are 0.6–2.3 and 2.4–5.0 microns, sampled at
0.031 and 0.063 arcsec per native detector pixel. Actual coverage is the full
tabulated throughput curve, not a hard wavelength cutoff or rectangular filter.

Sources:

- [STScI NIRCam filters](https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-instrumentation/nircam-filters)
- [STScI NIRCam imaging](https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-observing-modes/nircam-imaging)
- [STScI NIRCam PSFs](https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-performance/nircam-point-spread-functions)
- [STPSF source weighting and monochromatic PSFs](https://stpsf.readthedocs.io/en/latest/usage.html)

`catalog.json` records retrieval time, table URLs, source hashes, pivot
wavelengths, effective bandwidths, and half-power edges. These edges mean 50%
of peak throughput, not zero response. The STScI page identifies the current
curves as version 7.0 (January 2025). Its archive filename still mentions v5;
the fetcher therefore uses the individual linked mean-system tables.

Original `*_source.txt` files are preserved byte-for-byte in microns. Converted
CSV files use Angstrom and nonnegative throughput. Some official tables contain
small negative out-of-band samples: these are clipped to zero only in the
integration copy, with the count and original minimum recorded in the catalog.
The curves include telescope optics, detector efficiency, and required
pupil/filter-wheel combinations; do not multiply the blocking filter again.

## Forward model

1. Supply an intrinsic `(wavelength,y,x)` F_lambda cube, adequately resolving
   continuum and emission-line profiles across every selected passband.
2. Stretch the rest-frame wavelengths by `(1+z)` and divide F_lambda by `(1+z)`.
   This accounts for the density Jacobian only. Physical luminosity normalization
   requires distance separately; angular geometry must already be configured.
3. Convolve each wavelength image with its spatial PSF, then sum using photon
   weights `T(lambda)*lambda/(h*c)`. Wavelength channels are not spatially mixed.
4. Keep all filter responses separately without independent peak normalization.

`src/jwst_utils/nircam.py` defaults to a **preliminary Gaussian diffraction-core
approximation**, FWHM = 1.028 lambda / 6.5 m. It is wavelength dependent, but
does not reproduce segmented-pupil spikes, wings, aberrations, pupil stops,
detector response, field dependence, or drizzling. The Python API accepts a
`kernel_provider(filter, observed_wavelength_angstrom, model_pixel_scale_arcsec)`
for centered monochromatic STPSF or empirical kernels. The provider must supply
odd-sized, nonnegative arrays on the model grid, with the correct sky orientation.
Kernels are normalized to unit sum; finite-kernel truncation losses must be
evaluated separately. STPSF and its reference data are not installed by this work.

The output remains on the common model grid. Use a sufficiently fine intrinsic
grid (e.g. 0.005 arcsec/pixel for the short-wavelength core), and perform sampling
convergence checks. Native detector pixel integration/rebinning, angular-size
scaling with redshift, collecting area, exposure time, calibration into MJy/sr,
and noise are **not implemented by this foundation**. Zero exterior boundaries
allow light to leave a cutout; use adequate spatial margins.

## CLI

Input NPZ must contain `cube` and `rest_wavelength_angstrom`. The input must not
already be convolved with a telescope PSF. An explicit input PSF declaration is
required, and existing source PSF metadata is checked when present.

```bash
venv/bin/python scripts/cloudy/render_nircam.py \
  --input intrinsic_full_spectrum.npz --output nircam_images.npz \
  --redshift 0.2 --pixel-scale 0.005 --input-psf-fwhm 0
```

Default selection is all 29 filters; add `--filters F164N F187N F405N` for a
subset. Output keys are filter names plus JSON `metadata`. Physical F_lambda
in erg/s/cm^2/Angstrom per model pixel gives electron-response rates per cm^2
of collecting area; normalized Cloudy cubes give relative responses only.

The existing NGC 1068 F502N cube covers only 4960–5065 Angstrom and already has
spatial blur. It cannot be treated as a complete intrinsic NIRCam spectrum, or
redshifted again using this interface. Broader intrinsic Cloudy output is needed
before producing science or training images through these filters.

Re-fetch with `venv/bin/python scripts/cloudy/fetch_nircam_throughputs.py`.
Verify with `venv/bin/python -m unittest discover -s tests -p 'test_nircam.py'`.

## Filter reference

All wavelengths below are in microns. Blue/red edges are half-power limits.

| Filter | Channel | Pivot | Blue edge | Red edge |
| --- | --- | ---: | ---: | ---: |
| F070W | short | 0.705 | 0.624 | 0.781 |
| F090W | short | 0.902 | 0.795 | 1.005 |
| F115W | short | 1.154 | 1.013 | 1.282 |
| F140M | short | 1.405 | 1.331 | 1.479 |
| F150W | short | 1.501 | 1.331 | 1.668 |
| F162M | short | 1.627 | 1.542 | 1.713 |
| F164N | short | 1.645 | 1.635 | 1.653 |
| F150W2 | short | 1.672 | 1.007 | 2.380 |
| F182M | short | 1.845 | 1.722 | 1.968 |
| F187N | short | 1.874 | 1.863 | 1.885 |
| F200W | short | 1.988 | 1.755 | 2.227 |
| F210M | short | 2.096 | 1.992 | 2.201 |
| F212N | short | 2.121 | 2.109 | 2.134 |
| F250M | long | 2.503 | 2.412 | 2.595 |
| F277W | long | 2.776 | 2.422 | 3.131 |
| F300M | long | 2.996 | 2.831 | 3.156 |
| F322W2 | long | 3.247 | 2.432 | 4.013 |
| F323N | long | 3.237 | 3.217 | 3.255 |
| F335M | long | 3.362 | 3.177 | 3.537 |
| F356W | long | 3.565 | 3.136 | 3.981 |
| F360M | long | 3.623 | 3.426 | 3.814 |
| F405N | long | 4.053 | 4.030 | 4.076 |
| F410M | long | 4.083 | 3.865 | 4.301 |
| F430M | long | 4.281 | 4.167 | 4.398 |
| F444W | long | 4.402 | 3.881 | 4.982 |
| F460M | long | 4.630 | 4.515 | 4.747 |
| F466N | long | 4.654 | 4.629 | 4.681 |
| F470N | long | 4.708 | 4.683 | 4.733 |
| F480M | long | 4.817 | 4.662 | 4.972 |
