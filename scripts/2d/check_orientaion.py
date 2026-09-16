#!/usr/bin/env python3
"""
Run this from your project root:
    python3 check_wcs.py

It will tell you:
1. Which direction is North in your image
2. Where the NGC 1068 cone axis should point in pixel angle
3. What the correct theta0_init should be for your fitter
4. The flux scale factor diagnostic
"""

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.ndimage import gaussian_filter

# ── change these to match your actual paths ──────────────────────────────────
O3_FILE   = "data/2d/MAST_2026-04-14T0111/HST/hst_5754_01_wfpc2_pc_f502n_u2m301/hst_5754_01_wfpc2_pc_f502n_u2m301_drz.fits"
CONT_FILE = "data/2d/MAST_continuum/mastDownload/HST/hst_5754_01_wfpc2_pc_f547m_u2m301/hst_5754_01_wfpc2_pc_f547m_u2m301_drz.fits"
# ─────────────────────────────────────────────────────────────────────────────

# ── load ─────────────────────────────────────────────────────────────────────
print("Loading...")
with fits.open(O3_FILE) as h:
    o3     = h[1].data.astype(np.float32)
    header = h[1].header
    wcs    = WCS(header)

with fits.open(CONT_FILE) as h:
    cont = h[1].data.astype(np.float32)

o3   = np.nan_to_num(o3,   nan=0.0, posinf=0.0, neginf=0.0)
cont = np.nan_to_num(cont, nan=0.0, posinf=0.0, neginf=0.0)

# ── find nucleus ─────────────────────────────────────────────────────────────
valid       = (o3 > 0) & (cont > 0)
cont_smooth = gaussian_filter(cont * valid, sigma=5)
edge = 50
cont_smooth[:edge,:]  = cont_smooth[-edge:,:] = 0
cont_smooth[:,:edge]  = cont_smooth[:,-edge:] = 0
ny, nx = np.unravel_index(np.argmax(cont_smooth), cont_smooth.shape)
print(f"\nNucleus found at pixel: col={nx}  row={ny}")

# ── WCS orientation ───────────────────────────────────────────────────────────
print("\n--- WCS ORIENTATION ---")
try:
    sky_center = wcs.pixel_to_world(nx, ny)
    print(f"Nucleus sky coords: RA={sky_center.ra.deg:.4f}  Dec={sky_center.dec.deg:.4f}")

    # step +0.01 deg North in Dec to find North direction in pixel space
    sky_north = SkyCoord(
        ra=sky_center.ra,
        dec=sky_center.dec + 0.01 * u.deg
    )
    px_north = wcs.world_to_pixel(sky_north)

    north_angle = np.degrees(np.arctan2(
        float(px_north[1]) - ny,
        float(px_north[0]) - nx
    ))
    print(f"North direction in image: {north_angle:.1f}° from +x axis")

    # NGC 1068 ionization cone PA = ~30° East of North (NE direction)
    # PA is measured East of North, so add to North angle
    cone_pa_from_north = 30.0   # degrees — well established for NGC 1068
    cone_angle_image   = north_angle + cone_pa_from_north
    # wrap to [-180, 180]
    cone_angle_image   = (cone_angle_image + 180) % 360 - 180

    print(f"NGC 1068 cone PA = {cone_pa_from_north}° from North")
    print(f"→ cone axis in image coords = {cone_angle_image:.1f}° from +x axis")
    print(f"→ theta0_init = np.radians({cone_angle_image:.1f})")
    print(f"→ use bounds: [{cone_angle_image-40:.1f}°, {cone_angle_image+40:.1f}°]")

except Exception as e:
    print(f"WCS failed: {e}")
    print("Check that the FITS header has valid WCS keywords (CRVAL, CRPIX, CD or CDELT)")

# ── flux scale diagnostic ────────────────────────────────────────────────────
print("\n--- FLUX SCALE DIAGNOSTIC ---")

CROP = 400
half = CROP // 2
r0 = max(0, ny - half);  r1 = min(o3.shape[0], ny + half)
c0 = max(0, nx - half);  c1 = min(o3.shape[1], nx + half)

o3c   = o3[r0:r1,   c0:c1]
contc = cont[r0:r1, c0:c1]
H, W  = o3c.shape
yy, xx = np.indices((H, W))
r_grid = np.sqrt((xx - (nx-c0))**2 + (yy - (ny-r0))**2) + 1e-6

# sky annulus — far from galaxy, zero emission expected
sky_mask = (r_grid > 130) & (r_grid < 180) & (o3c > 0) & (contc > 0)
print(f"Sky annulus pixels: {sky_mask.sum()}")

if sky_mask.sum() > 50:
    o3_sky   = np.median(o3c[sky_mask])
    cont_sky = np.median(contc[sky_mask])
    scale_sky = o3_sky / (cont_sky + 1e-10)
    print(f"F502N sky median:  {o3_sky:.6f}")
    print(f"F547M sky median:  {cont_sky:.6f}")
    print(f"Sky scale factor:  {scale_sky:.6f}")
    if abs(scale_sky) < 0.1 or abs(scale_sky) > 10:
        print("WARNING: scale factor far from 1 — units may differ between filters")
        print("         check BUNIT keyword in both FITS headers")
else:
    print("WARNING: too few sky pixels — crop or r_grid range may need adjusting")

# check BUNIT
print("\n--- FITS HEADER UNITS ---")
with fits.open(O3_FILE) as h:
    print(f"F502N BUNIT: {h[1].header.get('BUNIT', 'not found')}")
    print(f"F502N EXPTIME: {h[1].header.get('EXPTIME', 'not found')}")
    print(f"F502N PHOTFLAM: {h[1].header.get('PHOTFLAM', 'not found')}")
with fits.open(CONT_FILE) as h:
    print(f"F547M BUNIT: {h[1].header.get('BUNIT', 'not found')}")
    print(f"F547M EXPTIME: {h[1].header.get('EXPTIME', 'not found')}")
    print(f"F547M PHOTFLAM: {h[1].header.get('PHOTFLAM', 'not found')}")

print("\nDONE — use the values above to update two_component_fit.py")
