#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from pathlib import Path
import shutil
import argparse
import json

# =========================================================
# CONFIG
# =========================================================

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="synthetic_oiii_realistic")
args = parser.parse_args()

BASE_DIR    = Path("data/2d") / args.name
GRID        = 128
N_SAMPLES   = 1000
TRAIN_SPLIT = 0.8
VAL_SPLIT   = 0.1

# =========================================================
# RESET
# =========================================================

def reset():
    if BASE_DIR.exists():
        assert "2d" in str(BASE_DIR)
        print(f"Resetting: {BASE_DIR}")
        shutil.rmtree(BASE_DIR)
    for split in ["train", "val", "test"]:
        (BASE_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (BASE_DIR / split / "masks").mkdir(parents=True, exist_ok=True)

# =========================================================
# GEOMETRY
# =========================================================

def warped_cone(grid, phi, opening_angle, r_max, inclination):
    """
    Hollow cone with sigmoid ramp onset.
    Radial peak at r_peak (0.35-0.60 * r_max) — mid-lobe, not nucleus.
    Warp clamped to 0-8 deg to avoid arc artifacts.
    Inclination drives near/far lobe asymmetry.
    """
    c       = grid // 2
    phi_rad = np.radians(phi)
    inc_rad = np.radians(inclination)

    y, x = np.mgrid[0:grid, 0:grid]
    dx   = (x - c).astype(np.float32)
    dy   = (y - c).astype(np.float32)
    r    = np.sqrt(dx**2 + dy**2) + 1e-8

    warp_amplitude = np.random.uniform(0, 8)
    warp_scale     = np.random.uniform(grid * 0.2, grid * 0.6)
    warp_direction = np.random.uniform(0, 2 * np.pi)

    warp_angle    = np.radians(warp_amplitude) * (1 - np.exp(-r / warp_scale))
    effective_phi = (phi_rad
                     + warp_angle * np.cos(warp_direction) * (dx / r)
                     + warp_angle * np.sin(warp_direction) * (dy / r))

    local_axis_x  = np.cos(effective_phi)
    local_axis_y  = np.sin(effective_phi)
    vdir_x        = dx / r
    vdir_y        = dy / r

    cosang        = np.clip(
        vdir_x * local_axis_x + vdir_y * local_axis_y, -1, 1
    )
    angle         = np.arccos(cosang)
    local_opening = opening_angle * np.random.uniform(0.8, 1.2)
    angular_illum = np.exp(-(angle / np.radians(local_opening))**2)

    base_axis         = np.array(
        [np.cos(phi_rad), np.sin(phi_rad)], dtype=np.float32
    )
    along             = dx * base_axis[0] + dy * base_axis[1]
    foreshorten       = 1.0 + np.sin(inc_rad) * (along / r)
    r_eff             = r / np.clip(foreshorten, 0.2, 5.0)
    brightness_factor = np.clip(foreshorten, 0.1, 3.0)

    r_inner = np.random.uniform(grid * 0.04, grid * 0.18)
    hollow  = 1.0 / (1.0 + np.exp(-(r_eff - r_inner) / (r_inner * 0.4)))

    r_peak  = r_max * np.random.uniform(0.35, 0.60)
    radial  = hollow * np.exp(-((r_eff - r_peak) / (r_max * 0.35))**2)
    radial *= (r_eff < r_max).astype(np.float32)

    return (angular_illum * radial * brightness_factor).astype(np.float32)

# =========================================================
# PHYSICAL LAYERS
# =========================================================

def fractal_field(grid, beta=2.0, seed=None):
    """
    Power-law noise in Fourier space — ISM turbulence spectrum.
    beta=2: red noise. beta=1.5: Kolmogorov.
    Produces multi-scale granularity: large diffuse patches + fine knots.
    """
    rng  = np.random.RandomState(seed) if seed is not None else np.random
    raw  = rng.normal(0, 1, (grid, grid))
    fraw = np.fft.fft2(raw)

    fy, fx      = np.meshgrid(
        np.fft.fftfreq(grid), np.fft.fftfreq(grid), indexing="ij"
    )
    power       = (fx**2 + fy**2 + 1e-8) ** (-beta / 2.0)
    power[0, 0] = 0.0

    field  = np.real(np.fft.ifft2(fraw * power)).astype(np.float32)
    field -= field.min()
    field /= field.max() + 1e-8
    return field

def ragged_wedge_mask(grid, cone_phi, opening, r_max, cone_amp,
                      ragged_seed=None):
    """
    Wedge mask with fractal-perturbed boundary.

    The nominal opening angle is modulated pixel-by-pixel by a
    low-beta fractal field so the cone edge is irregular — dense
    clumps poke beyond the nominal boundary, sparse regions retreat.

    This directly produces the lumpy spiky boundary seen in real
    NLR emission (NGC 1068 soft mask) rather than a clean geometric
    wedge.

    ragged_strength: 0.3 = moderate irregularity (±30% of half-angle)
    Additional radial fractal smears the outer boundary so the far
    edge of the cone is also ragged, not a clean arc.
    """
    c       = grid // 2
    yy, xx  = np.mgrid[0:grid, 0:grid]
    r_grid  = np.sqrt((xx - c)**2 + (yy - c)**2).astype(np.float32) + 1e-8

    axis_angle = np.arctan2(yy - c, xx - c).astype(np.float32)
    cone_rad   = np.radians(cone_phi)
    delta      = np.abs(np.angle(
        np.exp(1j * (axis_angle - cone_rad))
    )).astype(np.float32)

    # angular boundary fractal — perturbs opening angle per pixel
    ang_fractal = fractal_field(
        grid, beta=1.5,
        seed=ragged_seed if ragged_seed is not None
             else np.random.randint(0, 99999)
    )
    ragged_strength = np.random.uniform(0.20, 0.45)
    local_opening   = np.radians(opening / 2) * (
        1.0 + ragged_strength * (ang_fractal - 0.5)
    )

    # radial boundary fractal — perturbs r_max per pixel
    rad_fractal  = fractal_field(
        grid, beta=1.8, seed=np.random.randint(0, 99999)
    )
    r_strength   = np.random.uniform(0.15, 0.35)
    local_r_max  = r_max * (1.0 + r_strength * (rad_fractal - 0.5))
    local_r_max  = np.clip(local_r_max, r_max * 0.5, r_max * 1.4)

    ragged  = (delta < local_opening).astype(np.float32)
    radial  = (r_grid > grid * 0.05).astype(np.float32)
    radial *= (r_grid < local_r_max).astype(np.float32)

    wedge   = ragged * radial
    return (wedge * cone_amp).astype(np.float32)

def nucleus(grid, sigma=None):
    """Bright central point source."""
    sigma = sigma or np.random.uniform(1.5, 3.5)
    c     = grid // 2
    y, x  = np.mgrid[0:grid, 0:grid]
    return np.exp(
        -((x - c)**2 + (y - c)**2) / (2 * sigma**2)
    ).astype(np.float32)

def isotropic_halo(grid, r0=None):
    """Faint isotropic gas — cone amplifies this."""
    r0   = r0 or np.random.uniform(grid * 0.15, grid * 0.35)
    c    = grid // 2
    y, x = np.mgrid[0:grid, 0:grid]
    r    = np.sqrt((x - c)**2 + (y - c)**2)
    return np.exp(-(r / r0)).astype(np.float32)

def stellar_field(grid):
    """
    Poisson-distributed faint background stars.
    Sub-resolution (sigma 0.4-1.2 px), amp 0.02-0.12.
    Breaks pure-black background. Teaches model that point
    sources outside cone are not cone features.
    """
    img     = np.zeros((grid, grid), dtype=np.float32)
    n_stars = int(np.clip(np.random.poisson(lam=18), 5, 40))
    yy, xx  = np.mgrid[0:grid, 0:grid]

    for _ in range(n_stars):
        cx    = np.random.uniform(2, grid - 2)
        cy    = np.random.uniform(2, grid - 2)
        sigma = np.random.uniform(0.4, 1.2)
        amp   = np.random.uniform(0.02, 0.12)
        img  += amp * np.exp(
            -((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2)
        )

    return img.astype(np.float32)

def host_disk(grid):
    """Faint elliptical disk. PA independent of cone. Added after PSF."""
    c      = grid // 2
    pa     = np.random.uniform(0, 180)
    q      = np.random.uniform(0.2, 0.7)
    r_disk = np.random.uniform(grid * 0.2, grid * 0.45)
    amp    = np.random.uniform(0.015, 0.06)

    pa_rad = np.radians(pa)
    y, x   = np.mgrid[0:grid, 0:grid]
    dx     = (x - c).astype(np.float32)
    dy     = (y - c).astype(np.float32)

    x_rot =  dx * np.cos(pa_rad) + dy * np.sin(pa_rad)
    y_rot = -dx * np.sin(pa_rad) + dy * np.cos(pa_rad)

    r_ell = np.sqrt(x_rot**2 + (y_rot / q)**2)
    return (amp * np.exp(-(r_ell / r_disk)**2)).astype(np.float32)

def nlr_texture(grid, illum, lobe_seed=None):
    """
    Three-layer NLR texture gated by cone illumination.

    Layer 1 — fractal background (beta 1.8-2.5, weight 0.5):
        ISM turbulence. Diffuse multi-scale granularity.

    Layer 2 — sparse isotropic knots (5-20, weight 1.0):
        Dense photoionized clumps. Circular, random positions,
        no preferred axis.

    Layer 3 — random-PA filaments (2-6, weight 0.7):
        Outflow shock filaments. Elongated, fully random
        orientation — NOT tied to cone axis.

    Final texture blended 60/40 with its own Gaussian smooth
    (sigma=2.5) so each feature has a soft diffuse halo
    connecting it to surrounding gas. No hard brightness cutoffs.
    """
    rng  = np.random.RandomState(lobe_seed) if lobe_seed is not None \
           else np.random.RandomState(np.random.randint(0, 99999))

    cone_mask = np.clip(illum / (illum.max() + 1e-8), 0, 1)

    # Layer 1: fractal turbulent background
    beta    = rng.uniform(1.8, 2.5)
    fractal = fractal_field(grid, beta=beta, seed=rng.randint(0, 99999))
    fractal = np.power(fractal, rng.uniform(0.5, 0.8))
    fractal /= fractal.max() + 1e-8
    fractal *= cone_mask

    # Layer 2: sparse bright knots
    knots       = np.zeros((grid, grid), dtype=np.float32)
    n_knots     = rng.randint(5, 20)
    yy, xx      = np.mgrid[0:grid, 0:grid]
    cone_pixels = np.argwhere(cone_mask > 0.15)

    if len(cone_pixels) > n_knots:
        chosen = cone_pixels[
            rng.choice(len(cone_pixels), n_knots, replace=False)
        ]
        for cy, cx in chosen:
            sigma  = rng.uniform(1.5, 4.5)
            amp    = rng.uniform(0.4, 1.0)
            knots += amp * np.exp(
                -((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2)
            )

    knots /= knots.max() + 1e-8
    knots *= cone_mask

    # Layer 3: short random-PA filaments
    filaments   = np.zeros((grid, grid), dtype=np.float32)
    n_filaments = rng.randint(2, 7)

    if len(cone_pixels) > n_filaments:
        chosen = cone_pixels[
            rng.choice(len(cone_pixels), n_filaments, replace=False)
        ]
        for cy, cx in chosen:
            pa      = rng.uniform(0, np.pi)
            sigma_r = rng.uniform(4.0, 14.0)
            sigma_t = rng.uniform(0.8,  2.5)
            amp     = rng.uniform(0.3,  0.8)

            ddx = (xx - cx).astype(np.float32)
            ddy = (yy - cy).astype(np.float32)
            dr  =  ddx * np.cos(pa) + ddy * np.sin(pa)
            dt  = -ddx * np.sin(pa) + ddy * np.cos(pa)

            filaments += amp * np.exp(
                -(dr**2 / (2 * sigma_r**2) + dt**2 / (2 * sigma_t**2))
            )

    filaments /= filaments.max() + 1e-8
    filaments *= cone_mask

    # Combine
    texture  = 0.5 * fractal + 1.0 * knots + 0.7 * filaments
    texture /= texture.max() + 1e-8
    texture  = np.power(texture, 0.7)
    texture /= texture.max() + 1e-8

    # Soft halo blend — connects features to background, no hard cutoffs
    texture_smooth  = gaussian_filter(texture, sigma=2.5)
    texture_smooth /= texture_smooth.max() + 1e-8
    texture_blended = 0.6 * texture + 0.4 * texture_smooth
    texture_blended /= texture_blended.max() + 1e-8

    return texture_blended.astype(np.float32)

def dust_clumps(grid, phi, opening_angle, r_max):
    """Clumpy large-scale dust absorption. Applied before PSF."""
    c       = grid // 2
    phi_rad = np.radians(phi)
    y, x    = np.mgrid[0:grid, 0:grid]
    dx      = (x - c).astype(np.float32)
    dy      = (y - c).astype(np.float32)
    r       = np.sqrt(dx**2 + dy**2) + 1e-8

    dust = np.zeros((grid, grid), dtype=np.float32)
    for sigma in [np.random.uniform(4, 10), np.random.uniform(10, 20)]:
        layer  = np.random.uniform(0, 1, (grid, grid)).astype(np.float32)
        layer  = gaussian_filter(layer, sigma=sigma)
        layer /= layer.max() + 1e-8
        dust  += layer
    dust /= dust.max() + 1e-8

    axis   = np.array([np.cos(phi_rad), np.sin(phi_rad)], dtype=np.float32)
    vdir   = np.stack([dx, dy], axis=-1) / r[..., None]
    cosang = np.clip(np.sum(vdir * axis, axis=-1), -1, 1)
    angle  = np.arccos(cosang)

    cone_region  = np.exp(-(angle / np.radians(opening_angle))**2)
    cone_region *= (r < r_max).astype(np.float32)

    depth      = np.random.uniform(0.2, 0.5)
    absorption = 1.0 - depth * cone_region * dust

    return absorption.astype(np.float32)

# =========================================================
# SAMPLE GENERATION
# =========================================================

def sample_params(grid):
    is_bicone   = np.random.rand() > 0.60
    counter_amp = np.random.uniform(0.1, 0.3) if is_bicone else 0.0

    return {
        "has_agn":        np.random.rand() > 0.15,
        "bicone":         is_bicone,
        "counter_amp":    counter_amp,
        "phi":            np.random.uniform(0, 360),
        "opening":        np.random.uniform(15, 55),
        "r_max":          np.random.uniform(grid * 0.25, grid * 0.48),
        "inclination":    np.random.uniform(0, 60),
        "boost_str":      np.random.uniform(1.0, 3.0),
        "nlr_amp":        np.random.uniform(0.8, 2.0),
        "halo_r0_frac":   np.random.uniform(0.08, 0.18),
        "nucleus_sigma":  np.random.uniform(1.5, 3.5),
        "psf_sigma":      np.random.uniform(0.4, 0.9),
        # post-blur grain: roughens the final image surface
        "grain_amp":      np.random.uniform(0.06, 0.15),
        "noise":          np.random.uniform(0.003, 0.025),
        "has_disk":       np.random.rand() > 0.3,
        "has_stars":      np.random.rand() > 0.1,
    }

def generate_sample():
    grid   = GRID
    params = sample_params(grid)

    # --- no AGN: faint noisy halo + stars ---
    if not params["has_agn"]:
        halo = isotropic_halo(grid, r0=grid * params["halo_r0_frac"])
        img  = 0.2 * halo
        img += np.random.normal(0, params["noise"], img.shape)
        if params["has_stars"]:
            img += stellar_field(grid)
        img  = np.clip(img, 0, None)
        img /= img.max() + 1e-8
        return img.astype(np.float32), np.zeros((grid, grid), dtype=np.float32)

    phi         = params["phi"]
    opening     = params["opening"]
    r_max       = params["r_max"]
    inclination = params["inclination"]

    halo     = isotropic_halo(grid, r0=grid * params["halo_r0_frac"])
    emission = np.zeros((grid, grid), dtype=np.float32)
    mask_acc = np.zeros((grid, grid), dtype=np.float32)

    cone_phis = [phi]
    cone_amps = [1.0]
    if params["bicone"]:
        cone_phis.append(phi + 180)
        cone_amps.append(params["counter_amp"])

    # ── STAGE 1: smooth cone base + multiplicative density field ───
    illum_store = {}
    for cone_phi, cone_amp in zip(cone_phis, cone_amps):

        illum = warped_cone(
            grid, cone_phi, opening, r_max, inclination
        )

        boost         = 1.0 + params["boost_str"] * illum
        cone_emission = halo * boost

        # multiplicative fractal density — soft gamma 0.6-0.8 so
        # troughs stay nonzero and transitions are gradual
        density       = fractal_field(
            grid, beta=np.random.uniform(1.6, 2.2),
            seed=np.random.randint(0, 99999)
        )
        density       = np.power(density, np.random.uniform(0.6, 0.8))
        density      /= density.max() + 1e-8
        cone_mask_d   = np.clip(illum / (illum.max() + 1e-8), 0, 1)
        density_field = 1.0 - cone_mask_d * (1.0 - density)
        cone_emission *= density_field

        dust          = dust_clumps(grid, cone_phi, opening, r_max)
        cone_emission *= dust

        cone_emission *= cone_amp
        emission      += cone_emission

        # ragged wedge mask — fractal-perturbed boundary
        # opening angle varies per-pixel so edges are lumpy/spiky
        wedge      = ragged_wedge_mask(
            grid, cone_phi, opening, r_max, cone_amp,
            ragged_seed=np.random.randint(0, 99999)
        )
        mask_acc  += wedge

        illum_store[cone_phi] = (illum, cone_amp)

    emission += 0.05 * halo
    emission += nucleus(grid, sigma=params["nucleus_sigma"])

    # ── STAGE 2: PSF blur ──────────────────────────────────────────
    emission = gaussian_filter(emission, sigma=params["psf_sigma"])

    # ── STAGE 3: NLR texture — sharp knots/filaments after blur ────
    for cone_phi, (illum, cone_amp) in illum_store.items():
        texture   = nlr_texture(
            grid, illum,
            lobe_seed=np.random.randint(0, 99999)
        )
        emission += params["nlr_amp"] * cone_amp * texture



    # ── STAGE 5: background stars + disk + shot noise ───────────────
    if params["has_stars"]:
        emission += stellar_field(grid)

    if params["has_disk"]:
        emission += host_disk(grid)

    shot      = np.random.normal(0, params["noise"], emission.shape)
    emission += shot * np.sqrt(np.abs(emission) + 1e-6)

    emission  = np.clip(emission, 0, None)
    emission /= emission.max() + 1e-8

    # grain added AFTER normalization so bright cone does not squash it
    # flat across whole image — consistent noise floor inside and outside cone
    grain     = fractal_field(
        grid, beta=1.2, seed=np.random.randint(0, 99999)
    )
    emission += params["grain_amp"] * grain
    emission  = np.clip(emission, 0, 1)

    # clip mask to 0-1
    mask_acc  = np.clip(mask_acc, 0, 1).astype(np.float32)

    return emission.astype(np.float32), mask_acc.astype(np.float32)

# =========================================================
# BUILD + SAVE
# =========================================================

def generate():
    samples = []
    for i in range(N_SAMPLES):
        if i % 100 == 0:
            print(f"  {i}/{N_SAMPLES}...")
        samples.append(generate_sample())
    return samples

def save(samples):
    for i, (img, mask) in enumerate(samples):
        if i < int(N_SAMPLES * TRAIN_SPLIT):
            split = "train"
        elif i < int(N_SAMPLES * (TRAIN_SPLIT + VAL_SPLIT)):
            split = "val"
        else:
            split = "test"
        # save as (H, W) — dataset class adds channel dim itself
        np.save(BASE_DIR / split / "images" / f"{i:05d}.npy", img)
        np.save(BASE_DIR / split / "masks"  / f"{i:05d}.npy", mask)

# =========================================================
# VIZ
# =========================================================

def viz(samples):
    fig, axes = plt.subplots(3, 6, figsize=(15, 8))
    for i in range(9):
        img, mask = samples[i]
        col = (i % 3) * 2
        row = i // 3
        axes[row][col].imshow(
            img, cmap="gray", origin="lower", vmin=0, vmax=1
        )
        axes[row][col].set_title(f"image {i}", fontsize=8)
        axes[row][col].axis("off")
        axes[row][col + 1].imshow(
            mask, cmap="inferno", origin="lower", vmin=0, vmax=1
        )
        axes[row][col + 1].set_title(f"mask {i}", fontsize=8)
        axes[row][col + 1].axis("off")
    plt.tight_layout()
    plt.show()

# =========================================================
# METADATA
# =========================================================

def save_metadata():
    meta = {
        "grid":           GRID,
        "samples":        N_SAMPLES,
        "model":          "nucleus + (halo*boost * fractal_density) + dust"
                          " + PSF + nlr_texture(blended) + grain + stars + disk",
        "radial_peak":    "Gaussian at r_peak (0.35-0.60 * r_max)",
        "warp":           "axis drifts 0-8 deg",
        "inclination":    "foreshortening drives lobe asymmetry",
        "bicone":         "40% bicone, counter-lobe 0.1-0.3x primary",
        "density_field":  "multiplicative fractal (beta 1.6-2.2, gamma 0.6-0.8)"
                          " — soft ISM modulation, gradual transitions",
        "mask":           "ragged wedge — angular + radial fractal perturbation"
                          " (strength 0.20-0.45 angular, 0.15-0.35 radial)"
                          " — lumpy spiky boundary follows gas not geometry",
        "nlr_texture":    "fractal(0.5) + knots(1.0) + filaments(0.7),"
                          " blended 60/40 with sigma=2.5 smooth",
        "post_blur_grain":"fine fractal (beta=1.2) weighted by emission"
                          " — re-introduces surface roughness after PSF",
        "stellar_field":  "Poisson(lam=18) stars, sigma 0.4-1.2px, 90% samples",
        "dust":           "clumpy absorption before PSF, depth 0.2-0.5",
        "host_disk":      "after PSF, 70% samples",
        "noise":          "Poisson-like shot noise sqrt(signal)",
        "opening_range":  "15-55 degrees",
    }
    with open(BASE_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    reset()
    samples = generate()
    viz(samples[:9])
    save(samples)
    save_metadata()
    print(f"\nDONE → {BASE_DIR}\n")
