# STPSF F187N spatial convolution

We use STScI's existing implementation, rather than reproducing telescope
optics ourselves. The authoritative starting points are:

- [STPSF usage examples](https://stpsf.readthedocs.io/en/latest/usage.html)
- [STPSF installation and reference data](https://stpsf.readthedocs.io/en/stable/installation.html)
- [NIRCam PSF characteristics and downloadable models](https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-performance/nircam-point-spread-functions)

The essential call adapted from their API is:

```python
import stpsf

nrc = stpsf.NIRCam()
nrc.filter = "F187N"
nrc.detector = "NRCA1"
psf = nrc.calc_psf(
    monochromatic=1.8822122543e-6,  # meters; Pa alpha at z=0.003793
    fov_pixels=129,
    detector_oversample=3,
    fft_oversample=2,
    add_distortion=False,
)
```

This is a monochromatic optical PSF at the line's observed wavelength,
configured for F187N. It approximates a line-dominated image; it is not a
full source-weighted integration across the band. The `monochromatic` argument
overrides spectral weighting: it does not apply absolute throughput. Our
existing filter integration applies the F187N throughput separately, once.

For broadband calculations STPSF accepts a `synphot.SourceSpectrum`, which it
weights by the instrument response. The default stellar source is not an
appropriate replacement for our modeled gas spectrum. More accurate treatment
of different spatial spectra uses monochromatic PSFs in the wavelength cube,
followed by the existing photon-weighted filter integration.

## Generated and verified

`results/cloudy/f187n_psf/` contains:

- `stpsf_f187n.fits`: original STPSF output, optical and detector-sampled extensions.
- `kernel.npz`: normalized optical PSF, pixel scale, wavelength and filter name.
- `diagnostic.npz`: intrinsic point sources and their spatial convolution.
- `preview.png`: PSF and convolution diagnostic.
- `metadata.json`: software/reference versions, OPD, detector, normalization and limitations.

The first calculation used STPSF 2.2.0, its compatible reference data (the
downloaded minimal archive identifies its internal data version as 2.1.0),
the NRCA1 detector center and the supplied 2022-07-30 example wavefront.
The kernel is 387 x 387 on a 0.0104086167 arcsec/pixel grid. The original stamp
sum is approximately 0.9821; normalizing it to one supports relative-image
convolution but excludes absolute finite-aperture losses. A centered delta
source reproduced the kernel and preserved its total response to numerical
precision. This is an instrument diagnostic, not a Pa-alpha galaxy simulation.

The optical `OVERSAMP` extension is used deliberately: it is suitable for
convolution on a fine model grid, with subsequent detector pixel integration.
The detector-sampled extension is a different product and must not be applied
on a finer grid as though it were an optical-only kernel. Detector effects,
distortion and drizzling are not included in this diagnostic.

## Use with intrinsic synthetic data

```python
from src.jwst_utils.psf import convolve_with_saved_psf

blurred = convolve_with_saved_psf(
    intrinsic_pa_alpha_image,
    "results/cloudy/f187n_psf/kernel.npz",
    pixel_scale_arcsec=0.010408616666666667,
    input_psf_fwhm_arcsec=0.0,
)
```

The input must be an unblurred image representing the intended emission,
sampled and oriented consistently with the kernel. The helper rejects different
pixel scales and a nonzero declared input PSF. The existing optical Cloudy cube
has a 0.075-arcsec Gaussian PSF and a 0.04-arcsec pixel scale: it does not qualify.
Do not use deconvolution or a Gaussian quadrature subtraction to pretend this
already-blurred image is intrinsic. Build the Pa-alpha spatial model without
the earlier telescope blur, then apply this PSF. Geometric masks remain unchanged.

## Reproduce

STPSF is in `data/tools/stpsf-runtime`, a separate environment that reads common
dependencies from the existing project environment. STPSF-specific dependencies
(including its required NumPy version) are installed only in that separate
environment. The original project environment was not upgraded or downgraded.

Reference data is stored under `data/tools/stpsf-reference/stpsf-data`.
The download URL and archive hash are recorded in its parent manifest.
The fetcher is `scripts/cloudy/fetch_stpsf_reference.py`.

```bash
data/tools/stpsf-runtime/bin/python scripts/cloudy/generate_f187n_psf.py \
  --reference-data data/tools/stpsf-reference/stpsf-data \
  --output results/cloudy/f187n_psf_new
```

The generator requires a new output location, disables multiprocessing and
limits numerical-library threads to one. Reference files and outputs stay
inside the project; temporary configuration/cache files stay in `/tmp`.

Matching a particular JWST observation still requires its detector location,
appropriate observation-date wavefront, sky orientation and sampling. The
selected NGC 1068 observation used module B; this initial module-A kernel is
a generic demonstration, not a claim of an exact match to that observation.
