"""Apply a saved optical PSF to intrinsic images on a matching spatial grid."""

import numpy as np

from src.jwst_utils.nircam import convolve_kernel


def convolve_with_saved_psf(image, kernel_path, *, pixel_scale_arcsec,
                            input_psf_fwhm_arcsec):
    """Convolve an intrinsic 2D image; refuse a different grid or existing blur.

    The caller must supply an image at the kernel's wavelength/filter, sky
    orientation and pixel scale. No resampling, throughput, detector pixel
    integration, deconvolution, or mask processing is done here. The kernel is
    normalized to unit sum; zero exterior boundaries permit cutout flux loss.
    """
    if not np.isfinite(input_psf_fwhm_arcsec) or input_psf_fwhm_arcsec != 0:
        raise ValueError('Input must be intrinsic (PSF FWHM=0); existing blur cannot be undone here')
    if not np.isfinite(pixel_scale_arcsec) or pixel_scale_arcsec <= 0:
        raise ValueError('Pixel scale must be finite and positive')
    array = np.asarray(image)
    if array.ndim != 2 or min(array.shape) < 1 or not np.isfinite(array).all():
        raise ValueError('Expected a nonempty finite 2D image')
    with np.load(kernel_path, allow_pickle=False) as saved:
        kernel = saved['kernel']
        kernel_scale = float(saved['pixel_scale_arcsec'])
    if not np.isclose(pixel_scale_arcsec, kernel_scale, rtol=1e-6, atol=0):
        raise ValueError(f'Image pixel scale {pixel_scale_arcsec} does not match PSF {kernel_scale}; '
                         'generate the intrinsic image on the PSF grid or explicitly resample first')
    return convolve_kernel(array, kernel)
