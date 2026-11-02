"""Inspect the real F187N mosaic and save a small nucleus cutout and its errors."""

import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/ioncone-matplotlib')
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'data/2d/jwst_ngc1068/jw03707-o128_t002_nircam_clear-f187n_i2d.fits'
OUTPUT = ROOT/'results/cloudy/f187n_reference'


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import AsinhNorm
    from matplotlib.path import Path as Polygon

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with fits.open(SOURCE, memmap=True) as hdul:
        header = hdul[0].header
        ra, dec = float(header['TARG_RA']), float(header['TARG_DEC'])
        wcs = WCS(hdul['SCI'].header)
        x, y = map(float, wcs.world_to_pixel_values(ra, dec))
        cx, cy = int(round(x)), int(round(y))
        size = 256
        slices = (slice(cy-size//2, cy+size//2), slice(cx-size//2, cx+size//2))
        science = np.array(hdul['SCI'].data[slices])
        error = np.array(hdul['ERR'].data[slices])
        weight = np.array(hdul['WHT'].data[slices])
        valid = np.isfinite(science) & np.isfinite(error) & (error > 0) & (weight > 0)
        scale = float(np.mean(proj_plane_pixel_scales(wcs))*3600)
        yy, xx = np.indices(science.shape)
        radius = np.hypot(xx-(x-slices[1].start), yy-(y-slices[0].start))*scale
        outer = valid & (radius > 2.5) & (radius < 3.8)
        detectors = []
        for row in hdul['HDRTAB'].data:
            region = str(row['S_REGION']).split()
            vertices = np.array([float(v) for v in region[2:]]).reshape(-1, 2)
            if Polygon(vertices).contains_point((ra, dec)):
                detectors.append({'detector': str(row['DETECTOR']),
                                  'pa_v3_deg': float(row['PA_V3']),
                                  'roll_ref_deg': float(row['ROLL_REF'])})
        report = {
            'source_fits': str(SOURCE), 'filter': header['FILTER'],
            'date_obs': header['DATE-OBS'], 'exposure_seconds': float(header['EFFEXPTM']),
            'units': hdul['SCI'].header['BUNIT'], 'cutout_shape': list(science.shape),
            'pixel_scale_arcsec': scale, 'field_arcsec': scale*size,
            'target_ra_deg': ra, 'target_dec_deg': dec,
            'nucleus_cutout_xy': [float(x-slices[1].start), float(y-slices[0].start)],
            'valid_fraction': float(valid.mean()),
            'central_invalid_pixels_r_lt_0p3_arcsec': int(np.count_nonzero(~valid & (radius < .3))),
            'outer_median_science_mjysr': float(np.median(science[outer])),
            'outer_median_error_mjysr': float(np.median(error[outer])),
            'science_percentiles_mjysr': dict(zip(['p10','p50','p90','p99','max'],
                                                 [float(v) for v in np.percentile(science[valid], [10,50,90,99,100])])),
            'target_detector_footprints': detectors,
            'pixel_to_sky_matrix': wcs.pixel_scale_matrix.tolist(),
            'limitations': ['SCI invalid pixels are masked, not filled or treated as zero flux',
                            'No DQ extension in the stage-3 mosaic; cause of invalid core pixels not determined',
                            'Outer-cutout baseline includes galaxy emission, not an independently measured sky',
                            'No Pa-alpha/stellar/nuclear decomposition established by this single band'],
        }
        arrays = dict(science=science, error=error, weight=weight, valid=valid,
                      pixel_scale_arcsec=scale, metadata=json.dumps(report))
        for name in ['VAR_POISSON', 'VAR_RNOISE', 'VAR_FLAT']:
            arrays[name.lower()] = np.array(hdul[name].data[slices])
        np.savez_compressed(OUTPUT/'nucleus_cutout.npz', **arrays)
        (OUTPUT/'inspection.json').write_text(json.dumps(report, indent=2)+'\n')
        fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
        full = hdul['SCI'].data[::8, ::8]
        high = float(np.nanpercentile(full, 99.8))
        axes[0].imshow(np.ma.masked_invalid(full), origin='lower', cmap='magma',
                       norm=AsinhNorm(linear_width=max(high*.03, 1), vmin=0, vmax=high))
        axes[0].plot(x/8, y/8, '+', color='cyan')
        axes[0].set_title('Real NGC 1068 F187N mosaic')
        high = float(np.percentile(science[valid], 99.5))
        axes[1].imshow(np.ma.array(science, mask=~valid), origin='lower', cmap='magma',
                       norm=AsinhNorm(linear_width=high*.03, vmin=0, vmax=high))
        axes[1].set_title(f'Nuclear cutout: {scale*size:.2f} arcsec\nInvalid pixels masked; units MJy/sr')
        for axis in axes:
            axis.set_axis_off()
        fig.savefig(OUTPUT/'preview.png', dpi=150)
        plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
