"""Generate a documented STPSF F187N optical kernel and a convolution diagnostic.

The default monochromatic PSF uses Pa alpha at NGC 1068's systemic redshift.
This is an instrument test, not a calibrated galaxy or a rendered Cloudy cube.
"""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-data', type=Path, required=True,
                   help='STPSF data directory containing version.txt')
    p.add_argument('--output', type=Path, default=ROOT/'results/cloudy/f187n_psf')
    p.add_argument('--detector', default='NRCA1', help='Generic reference detector, not an observation match')
    p.add_argument('--wavelength-um', type=float, default=1.87510*(1+0.003793))
    p.add_argument('--fov-pixels', type=int, default=129)
    p.add_argument('--oversample', type=int, default=3)
    args = p.parse_args()
    if not (args.reference_data/'version.txt').is_file():
        p.error('Reference data directory must contain version.txt')
    if args.fov_pixels <= 0 or args.oversample <= 0 or args.fov_pixels % 2 != 1 or args.oversample % 2 != 1:
        p.error('Use positive odd FOV pixels and oversampling for a pixel-centered convolution kernel')
    if not 1.7482 <= args.wavelength_um <= 2.0022:
        p.error('Wavelength must lie in the tabulated F187N support (microns)')
    paths = [args.output/name for name in ['stpsf_f187n.fits', 'kernel.npz', 'diagnostic.npz', 'preview.png', 'metadata.json']]
    if any(path.exists() for path in paths):
        p.error('Choose a new output directory; existing outputs are not overwritten')
    os.environ['STPSF_PATH'] = str(args.reference_data.resolve())
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/ioncone-stpsf-matplotlib')
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    import numpy as np
    from astropy.config.paths import set_temp_cache, set_temp_config
    for temporary in ['/tmp/ioncone-stpsf-cache', '/tmp/ioncone-stpsf-config']:
        Path(temporary).mkdir(parents=True, exist_ok=True)
    with set_temp_cache('/tmp/ioncone-stpsf-cache'), set_temp_config('/tmp/ioncone-stpsf-config'):
        import stpsf
        import poppy
        poppy.conf.use_multiprocessing = False
        nrc = stpsf.NIRCam()
        nrc.filter = 'F187N'
        nrc.detector = args.detector
        nrc.detector_position = (1024, 1024)
        nrc.options['parity'] = 'odd'
        psf = nrc.calc_psf(monochromatic=args.wavelength_um*1e-6,
                           fov_pixels=args.fov_pixels, detector_oversample=args.oversample,
                           fft_oversample=2,
                           add_distortion=False)
        args.output.mkdir(parents=True, exist_ok=True)
        psf.writeto(args.output/'stpsf_f187n.fits')
        optical = psf['OVERSAMP']
        raw = np.asarray(optical.data, dtype=float)
        if (not np.isfinite(raw).all() or np.any(raw < 0) or raw.sum() <= 0
                or any(n % 2 != 1 for n in raw.shape)):
            raise ValueError('Unexpected optical PSF; inspect the saved FITS file')
        kernel = raw/raw.sum()
        scale = float(optical.header['PIXELSCL'])
        from src.jwst_utils.nircam import convolve_kernel
        # A point source must produce exactly the centered supplied kernel.
        margin = 32
        size = kernel.shape[0]+2*margin
        points = np.zeros((size, size))
        points[size//2, size//2] = 1
        blurred_point = convolve_kernel(points, kernel)
        np.testing.assert_allclose(blurred_point[margin:-margin, margin:-margin], kernel,
                                   atol=1e-15, rtol=1e-10)
        if abs(blurred_point.sum()-1) > 1e-12:
            raise ValueError('Point-source flux-conservation check failed')
        # A small intrinsic source group demonstrates extended-image convolution.
        scene = points.copy()
        offset = max(1, int(round(0.25/scale)))
        scene[size//2-offset, size//2+offset] = 0.3
        scene[size//2+offset, size//2-offset] = 0.6
        convolved_scene = convolve_kernel(scene, kernel)
        np.savez_compressed(args.output/'kernel.npz', kernel=kernel, pixel_scale_arcsec=scale,
                            wavelength_um=args.wavelength_um, filter='F187N')
        np.savez_compressed(args.output/'diagnostic.npz', intrinsic=scene,
                            convolved=convolved_scene, point_response=blurred_point)
        metadata = {
            'tool': 'STPSF', 'version': stpsf.__version__, 'filter': 'F187N',
            'detector': nrc.detector, 'detector_position': list(nrc.detector_position),
            'wavelength_um': args.wavelength_um, 'spectral_mode': 'monochromatic Pa-alpha approximation',
            'opd': str(nrc.pupilopd), 'reference_data': str(args.reference_data.resolve()),
            'reference_version': (args.reference_data/'version.txt').read_text().strip(),
            'extension': 'OVERSAMP', 'kernel_shape': list(kernel.shape),
            'pixel_scale_arcsec': scale, 'finite_stamp_sum_before_normalization': float(raw.sum()),
            'kernel_sum': float(kernel.sum()), 'point_response_sum': float(blurred_point.sum()),
            'kernel_sha256': sha256(kernel.tobytes()).hexdigest(),
            'spatial_coordinates': 'STPSF ideal coordinates; sky rotation not fitted',
            'limitations': ['generic detector center and reference OPD, not observation-specific',
                            'monochromatic approximation; velocity-dependent PSFs not integrated yet',
                            'optical model; no detector pixel integration, distortion or drizzling',
                            'finite stamp renormalized to unit sum; absolute aperture losses not modeled',
                            'diagnostic sources only; no Pa-alpha galaxy cube rendered'],
            'documentation': 'https://stpsf.readthedocs.io/en/latest/usage.html',
        }
        (args.output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
        extent = np.array([-1, 1, -1, 1])*kernel.shape[0]*scale/2
        axes[0].imshow(kernel, origin='lower', cmap='magma', extent=extent,
                       norm=LogNorm(vmin=kernel.max()*1e-6, vmax=kernel.max()))
        axes[0].set(title=f'STPSF F187N optical PSF\nPa alpha at {args.wavelength_um:.5f} µm',
                    xlabel='arcsec', ylabel='arcsec')
        for axis, data, title in zip(axes[1:], [scene, convolved_scene],
                                      ['Intrinsic point-source test', 'After spatial convolution']):
            axis.imshow(np.maximum(data, 1e-15), origin='lower', cmap='magma',
                        norm=LogNorm(vmin=1e-6, vmax=1))
            axis.set(title=title)
            axis.set_axis_off()
        fig.suptitle('Instrument diagnostic — not a calibrated galaxy simulation')
        fig.savefig(args.output/'preview.png', dpi=160)
        plt.close(fig)
        print(f'Saved F187N kernel {kernel.shape}, {scale:.7f} arcsec/pixel in {args.output}')
        print(f'Point-source convolution check passed; total response = {blurred_point.sum():.12f}')


if __name__ == '__main__':
    main()
