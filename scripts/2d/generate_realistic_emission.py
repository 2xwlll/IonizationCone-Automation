#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from pathlib import Path
import shutil
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.simple_utils.agn_profile import apply_agn_profile
from src.simple_utils.cone_labels import connected_cone_mask

# =========================================================
# CONFIG
# =========================================================

parser = argparse.ArgumentParser(
    description="Generate a reproducible, parameter-tracked synthetic OIII dataset."
)
parser.add_argument("--name", type=str, default="synthetic_oiii_clumpy_v2")
parser.add_argument(
    "--config", type=Path,
    default=Path("configs/2d/synthetic_clumpy_v2.json"),
)
parser.add_argument("--samples", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--agn-profile", type=Path, default=None,
                    help="Observation-backed geometry profile JSON")
parser.add_argument("--negative-frac", type=float, default=None)
parser.add_argument("--obscured-counter-frac", type=float, default=None,
                    help="Training fraction of intrinsic bicones whose counter-lobe is hidden")
parser.add_argument(
    "--overwrite", action="store_true",
    help="Replace an existing dataset directory (never enabled implicitly).",
)
args = parser.parse_args()

with open(args.config) as f:
    CONFIG = json.load(f)
if args.agn_profile is not None:
    CONFIG = apply_agn_profile(CONFIG, args.agn_profile)

dataset_config = CONFIG["dataset"]
mixture_config = CONFIG["mixture"]
BASE_DIR    = Path("data/2d") / args.name
GRID        = int(dataset_config["grid"])
N_SAMPLES   = int(args.samples if args.samples is not None else dataset_config["samples"])
SEED        = int(args.seed if args.seed is not None else dataset_config["seed"])
TRAIN_SPLIT = float(dataset_config["train_fraction"])
VAL_SPLIT   = float(dataset_config["validation_fraction"])
NEGATIVE_FRAC = float(
    args.negative_frac
    if args.negative_frac is not None
    else mixture_config["negative_fraction"]
)
OBSCURED_COUNTER_FRAC = float(
    args.obscured_counter_frac
    if args.obscured_counter_frac is not None
    else mixture_config.get("obscured_counter_fraction", 0.0)
)


def sample_range(section, name):
    """Sample a float from a configured closed interval."""
    low, high = CONFIG[section][name]
    return float(np.random.uniform(low, high))


def sample_int_range(section, name):
    """Sample an integer from a configured [low, high) interval."""
    low, high = CONFIG[section][name]
    return int(np.random.randint(int(low), int(high)))


def validate_config():
    labels = CONFIG.get("labels", {})
    if labels.get("mode", "connected_geometry") not in ("connected_geometry", "visible_emission"):
        raise ValueError("labels.mode must be connected_geometry or visible_emission")
    for name in ("path_width_fraction", "pathway_strength", "path_width_tracking", "edge_emission_tracking"):
        low, high = CONFIG["geometry"][name]
        if not 0 <= low <= high <= 1:
            raise ValueError(f"geometry.{name} must be within [0, 1]")
    if labels.get("detection_snr", 2.5) <= 0:
        raise ValueError("labels.detection_snr must be positive")
    if not 0 <= labels.get("peak_fraction_floor", 0.0) < 1:
        raise ValueError("labels.peak_fraction_floor must be in [0, 1)")
    if "counter_transmission" in CONFIG["obscuration"]:
        low, high = CONFIG["obscuration"]["counter_transmission"]
        if not 0 <= low <= high <= 1:
            raise ValueError("obscuration.counter_transmission must be within [0, 1]")
    if "clouds" in CONFIG:
        cloud = CONFIG["clouds"]
        for name in ("count", "sigma_pixels", "cluster_scale_pixels", "structure_scale_pixels"):
            if min(cloud[name]) <= 0:
                raise ValueError(f"clouds.{name} must be positive")
        for name in ("axis_ratio", "cluster_fraction", "irregularity"):
            if not 0 <= cloud[name][0] <= cloud[name][1] <= 1:
                raise ValueError(f"clouds.{name} must be within [0, 1]")
        if cloud["axis_ratio"][0] == 0:
            raise ValueError("clouds.axis_ratio must be positive")
        if any(int(v) != v for v in cloud["count"]) or cloud["count"][0] >= cloud["count"][1]:
            raise ValueError("clouds.count must be an integer [low, high) range")
        for name in ("luminosity_scatter", "density_contrast", "cloud_weight", "diffuse_weight", "filament_weight", "bridge_weight"):
            if name not in cloud:
                continue
            if min(cloud[name]) < 0:
                raise ValueError(f"clouds.{name} cannot be negative")
        if cloud["cloud_weight"][0] <= 0:
            raise ValueError("clouds.cloud_weight must be positive")
    fractions = {
        "train_fraction": TRAIN_SPLIT,
        "validation_fraction": VAL_SPLIT,
        "negative_fraction": NEGATIVE_FRAC,
        "bicone_fraction": mixture_config["bicone_fraction"],
        "obscured_counter_fraction": OBSCURED_COUNTER_FRAC,
        "distractor_on_positive_fraction": mixture_config[
            "distractor_on_positive_fraction"
        ],
        "disk_fraction": CONFIG["background"]["disk_fraction"],
    }
    for name, value in fractions.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")
    if TRAIN_SPLIT + VAL_SPLIT >= 1.0:
        raise ValueError("train_fraction + validation_fraction must be less than 1")
    if not mixture_config["negative_types"]:
        raise ValueError("mixture.negative_types cannot be empty")
    if (
        mixture_config["distractor_on_positive_fraction"] > 0
        and not [kind for kind in mixture_config["negative_types"] if kind != "diffuse"]
    ):
        raise ValueError(
            "mixture.negative_types must include a non-diffuse type when "
            "distractors are enabled"
        )
    for section, values in CONFIG.items():
        if not isinstance(values, dict):
            continue
        for name, value in values.items():
            if isinstance(value, list) and len(value) == 2 and all(
                isinstance(item, (int, float)) for item in value
            ):
                if value[0] > value[1]:
                    raise ValueError(f"Invalid range {section}.{name}: {value}")
    for section, name in (("emission", "wisp_count"), ("emission", "grain_count")):
        low, high = CONFIG[section][name]
        if int(low) >= int(high):
            raise ValueError(
                f"Integer range {section}.{name} must have low < high: "
                f"{CONFIG[section][name]}"
            )

# =========================================================
# RESET
# =========================================================

def reset():
    if BASE_DIR.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{BASE_DIR} already exists; choose another --name or pass --overwrite"
            )
        assert BASE_DIR.parent == Path("data/2d")
        print(f"Resetting: {BASE_DIR}")
        shutil.rmtree(BASE_DIR)
    for split in ["train", "val", "test"]:
        (BASE_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (BASE_DIR / split / "masks").mkdir(parents=True, exist_ok=True)

# =========================================================
# GEOMETRY
# =========================================================

def warped_cone(grid, phi, opening_angle, r_max, inclination, params):
    """
    Hollow cone with sigmoid ramp onset.
    
    *** IMPORTANT***
    We might need to use SKIRTOR in order to generate some clumpiness 
    in the 3D model before suming it for brightness peaks.

    Hollow: emission suppressed near nucleus, peaks at r_inner,
    fades beyond. The sigmoid transition gives a gradual ramp-up
    rather than an abrupt bright edge.

    Warp: cone axis drifts smoothly with radius — mimics jet
    precession and asymmetric outflow geometry.

    Inclination: foreshortening makes near lobe brighter/longer,
    far lobe fainter/compressed. This is the dominant asymmetry
    between the two lobes — dust is secondary.
    """
    cx      = params["center_x"]
    cy      = params["center_y"]
    phi_rad = np.radians(phi)
    inc_rad = np.radians(inclination)

    y, x = np.mgrid[0:grid, 0:grid]
    dx   = (x - cx).astype(np.float32)
    dy   = (y - cy).astype(np.float32)
    r    = np.sqrt(dx**2 + dy**2) + 1e-8

    # smooth axis warp with radius
    warp_amplitude = params["warp_amplitude"]
    warp_scale     = params["warp_scale"]
    warp_direction = np.radians(params["warp_direction"])

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
    local_opening = opening_angle * params["opening_scale"]
    boundary_noise = np.random.normal(0, 1, (grid, grid)).astype(np.float32)
    boundary_noise = gaussian_filter(boundary_noise, params["boundary_noise_scale"])
    boundary_noise -= boundary_noise.mean()
    boundary_noise /= boundary_noise.std() + 1e-8
    local_opening_map = np.radians(local_opening) * np.clip(
        1.0 + params["angular_boundary_roughness"] * boundary_noise,
        0.35, 1.8,
    )
    angular_illum = np.exp(-(angle / local_opening_map) ** 2)

    # inclination foreshortening on base (unwarped) axis
    base_axis         = np.array(
        [np.cos(phi_rad), np.sin(phi_rad)], dtype=np.float32
    )
    along             = dx * base_axis[0] + dy * base_axis[1]
    foreshorten       = 1.0 + np.sin(inc_rad) * (along / r)
    r_eff             = r / np.clip(foreshorten, 0.2, 5.0)
    brightness_factor = np.clip(foreshorten, 0.1, 3.0)

    # sigmoid hollow ramp — gradual onset, not abrupt bright edge
    # transitions smoothly over ~r_inner pixels centered at r_inner
    r_inner = params["inner_radius"]
    hollow  = 1.0 / (1.0 + np.exp(-(r_eff - r_inner) / (r_inner * 0.4)))

    radial_limit = r_max * np.clip(
        1.0 + params["radial_boundary_roughness"] * boundary_noise,
        0.45, 1.5,
    )
    soft_edge = 1.0 / (
        1.0 + np.exp((r_eff - radial_limit) / params["radial_edge_width"])
    )
    radial = hollow * np.exp(-(r_eff / (r_max * 0.65)) ** 2) * soft_edge

    fragment_noise = np.random.uniform(0, 1, (grid, grid)).astype(np.float32)
    fragment_noise = gaussian_filter(
        fragment_noise, params["boundary_noise_scale"] * 0.45
    )
    fragment_noise -= fragment_noise.min()
    fragment_noise /= fragment_noise.max() + 1e-8
    fragmentation = (
        1.0 - params["fragmentation_strength"]
        + params["fragmentation_strength"] * fragment_noise
    )

    return (
        angular_illum * radial * brightness_factor * fragmentation
    ).astype(np.float32)

# =========================================================
# PHYSICAL LAYERS
# =========================================================

def nucleus(grid, sigma, amplitude=1.0, center_x=None, center_y=None):
    """Bright central point source. Not textured or obscured."""
    center_x = grid // 2 if center_x is None else center_x
    center_y = grid // 2 if center_y is None else center_y
    y, x  = np.mgrid[0:grid, 0:grid]
    return np.exp(
        -((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2)
    ).astype(np.float32) * amplitude

def isotropic_halo(grid, r0, center_x=None, center_y=None):
    """
    Faint gas in all directions from nucleus.
    The cone amplifies this — not a separate source.
    """
    center_x = grid // 2 if center_x is None else center_x
    center_y = grid // 2 if center_y is None else center_y
    y, x = np.mgrid[0:grid, 0:grid]
    r    = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    return np.exp(-(r / r0)).astype(np.float32)

def host_disk(grid, params):
    """
    Faint elliptical disk — host galaxy structure leaking into
    emission after imperfect continuum subtraction.

    PA is independent of cone axis — disk and cone are not
    necessarily aligned, teaching the model that elongated
    structure is not always a cone.

    Added AFTER PSF blur so it does not get double-smoothed.
    Amplitude kept tight so it never dominates the cone signal.
    """
    c      = grid // 2
    pa     = params["disk_pa"]
    q      = params["disk_axis_ratio"]
    r_disk = params["disk_radius"]
    amp    = params["disk_amplitude"]

    pa_rad = np.radians(pa)
    y, x   = np.mgrid[0:grid, 0:grid]
    dx     = (x - c).astype(np.float32)
    dy     = (y - c).astype(np.float32)

    x_rot =  dx * np.cos(pa_rad) + dy * np.sin(pa_rad)
    y_rot = -dx * np.sin(pa_rad) + dy * np.cos(pa_rad)

    r_ell = np.sqrt(x_rot**2 + (y_rot / q)**2)
    return (amp * np.exp(-(r_ell / r_disk)**2)).astype(np.float32)


def hard_negative(grid, kind, params):
    """Produce bright cone-like distractors with an intentionally empty mask."""
    c = grid // 2
    yy, xx = np.mgrid[0:grid, 0:grid]
    image = params["diffuse_fraction"] * isotropic_halo(
        grid, params["halo_radius"], params["center_x"], params["center_y"]
    )

    if kind == "disk":
        image += params["distractor_strength"] * host_disk(grid, params)
    elif kind == "jet":
        pa = np.radians(params["distractor_pa"])
        along = (xx - c) * np.cos(pa) + (yy - c) * np.sin(pa)
        across = -(xx - c) * np.sin(pa) + (yy - c) * np.cos(pa)
        width = params["distractor_width"]
        length = params["distractor_length"]
        image += params["distractor_strength"] * np.exp(
            -0.5 * (across / width) ** 2 - 0.5 * (along / length) ** 2
        )
    elif kind == "bipolar_blobs":
        pa = np.radians(params["distractor_pa"])
        radius = params["distractor_radius"]
        for sign in (-1, 1):
            cx = c + sign * radius * np.cos(pa)
            cy = c + sign * radius * np.sin(pa)
            sigma = params["distractor_width"] * 2.5
            image += params["distractor_strength"] * np.exp(
                -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)
            )
    elif kind == "psf_residual":
        radius = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
        scale = params["distractor_width"] * 2.0
        rings = np.square(np.sinc(radius / scale))
        image += params["distractor_strength"] * rings
    elif kind != "diffuse":
        raise ValueError(f"Unknown hard-negative kind: {kind}")

    image += np.random.normal(0, params["noise"], image.shape)
    image = gaussian_filter(np.clip(image, 0, None), params["psf_sigma"])
    image /= image.max() + 1e-8
    return image.astype(np.float32)

def wispy_gas(
    grid, phi, opening_angle, r_max, n_wisps=None, center_x=None, center_y=None
):
    """
    Three size classes per sample:
        small knots     (60%) — bright, compact
        medium filaments (25%) — moderate length and width
        large patches   (15%) — diffuse, dominate area

    axis_offset shifts the wisp cluster off the cone axis —
    breaks left/right symmetry within a single lobe.
    """
    n_wisps     = n_wisps or np.random.randint(15, 60)
    img         = np.zeros((grid, grid), dtype=np.float32)
    center_x    = grid // 2 if center_x is None else center_x
    center_y    = grid // 2 if center_y is None else center_y
    phi_rad     = np.radians(phi)
    spread      = opening_angle * 0.6
    axis_offset = np.random.normal(0, np.radians(opening_angle * 0.3))

    for _ in range(n_wisps):
        angle = phi_rad + axis_offset + np.random.normal(
            0, np.radians(spread)
        )
        r = np.random.uniform(3, min(r_max * 0.95, grid * 0.47))

        cx = center_x + r * np.cos(angle)
        cy = center_y + r * np.sin(angle)

        size_class = np.random.rand()
        if size_class < 0.60:
            sigma_r = np.random.uniform(1.5,  5.0)
            sigma_t = np.random.uniform(0.4,  1.5)
            amp     = np.random.uniform(0.2,  0.7)
        elif size_class < 0.85:
            sigma_r = np.random.uniform(5.0, 14.0)
            sigma_t = np.random.uniform(0.8,  2.5)
            amp     = np.random.uniform(0.1,  0.4)
        else:
            sigma_r = np.random.uniform(12.0, 22.0)
            sigma_t = np.random.uniform(3.0,   8.0)
            amp     = np.random.uniform(0.05,  0.2)

        yy, xx = np.mgrid[0:grid, 0:grid]
        ddx    = xx - cx
        ddy    = yy - cy

        dr =  ddx * np.cos(angle) + ddy * np.sin(angle)
        dt = -ddx * np.sin(angle) + ddy * np.cos(angle)

        img += amp * np.exp(
            -(dr**2 / (2 * sigma_r**2) + dt**2 / (2 * sigma_t**2))
        )

    return (img / (img.max() + 1e-8)).astype(np.float32)


def granular_knots(grid, count, sigma_min, sigma_max):
    """Create many compact, nonuniform emission knots efficiently."""
    grains = np.zeros((grid, grid), dtype=np.float32)
    groups = 4
    for group in range(groups):
        impulses = np.zeros_like(grains)
        n_group = count // groups + (group < count % groups)
        ys = np.random.randint(0, grid, n_group)
        xs = np.random.randint(0, grid, n_group)
        amplitudes = np.random.lognormal(mean=0.0, sigma=0.8, size=n_group)
        np.add.at(impulses, (ys, xs), amplitudes.astype(np.float32))
        fraction = group / max(groups - 1, 1)
        sigma = sigma_min * (sigma_max / sigma_min) ** fraction
        grains += gaussian_filter(impulses, sigma=max(sigma, 0.15))
    grains /= grains.max() + 1e-8
    return grains


def cloud_components(grid, illum, settings, seed):
    """Compact cloud complexes and an independent correlated diffuse field.

    Local RNG keeps changes to these controls out of geometry/dust/noise draws.
    Positions follow illumination continuously, without a radial cutoff.
    Sizes are intrinsic Gaussian major-axis sigmas, before the instrument PSF.
    """
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[:grid, :grid]
    probability = np.sqrt(np.maximum(illum, 0)).ravel().astype(float)
    probability /= probability.sum()
    count = settings["count"]
    indices = rng.choice(grid * grid, count, p=probability)
    centers = np.column_stack(np.unravel_index(indices, (grid, grid))).astype(float)
    # Draw all random arrays even when a weight/fraction is zero, for ablations.
    parent_indices = rng.randint(0, count, count)
    clustered = rng.uniform(size=count) < settings["cluster_fraction"]
    offsets = rng.normal(size=(count, 2)) * settings["cluster_scale_pixels"]
    centers[clustered] = (centers[parent_indices] + offsets)[clustered]
    sigma = np.exp(rng.uniform(np.log(settings["sigma_pixels"][0]),
                               np.log(settings["sigma_pixels"][1]), count))
    ratios = rng.uniform(*settings["axis_ratio"], count)
    angles = rng.uniform(0, np.pi, count)
    amplitudes = np.exp(np.clip(rng.normal(size=count) * settings["luminosity_scatter"], -10, 10))
    clouds = np.zeros((grid, grid), dtype=np.float64)
    for (cy, cx), size, ratio, angle, amplitude in zip(centers, sigma, ratios, angles, amplitudes):
        dx, dy = xx - cx, yy - cy
        major = dx * np.cos(angle) + dy * np.sin(angle)
        minor = -dx * np.sin(angle) + dy * np.cos(angle)
        clouds += amplitude * np.exp(-0.5 * ((major / size)**2 + (minor / (size * ratio))**2))
    structure = gaussian_filter(rng.normal(size=(grid, grid)), settings["structure_scale_pixels"])
    structure = (structure - structure.mean()) / (structure.std() + 1e-8)
    clouds *= np.exp(np.clip(settings["irregularity"] * structure, -10, 10))
    diffuse = np.exp(np.clip(settings["density_contrast"] * structure, -10, 10))
    return clouds.astype(np.float32), diffuse.astype(np.float32)


def normalize_component(field, illum):
    """Unit illumination-weighted mean: weights control flux, not peak counts."""
    mean = np.sum(field * illum, dtype=np.float64) / (np.sum(illum, dtype=np.float64) + 1e-12)
    return field / max(mean, 1e-12)


def sample_cloud_settings(sample_seed):
    """Sample cloud controls separately so existing sample geometry is preserved."""
    rng = np.random.RandomState((int(sample_seed) + 104729) % (2**32))
    result = {}
    for name, bounds in CONFIG["clouds"].items():
        if name in ("sigma_pixels", "axis_ratio"):
            result[name] = list(bounds)
        elif name == "count":
            result[name] = int(rng.randint(*bounds))
        else:
            result[name] = float(rng.uniform(*bounds))
    return result

def gas_texture(
    grid, phi, opening_angle, r_max, strength, gamma, lobe_seed=None,
    center_x=None, center_y=None,
):
    """
    Multiplicative clumpy texture inside the cone.
    Each lobe gets independent random texture via lobe_seed.

    Fine-scale noise is weighted most heavily — this produces
    the knot-like roughness that smooth Gaussian cones lack.
    Gamma correction sharpens contrast away from midtones.

    Weights: fine (1.5) > medium (1.0) > coarse (0.5)
    """
    rng   = np.random.RandomState(lobe_seed) if lobe_seed is not None \
            else np.random
    scale = strength
    img   = np.zeros((grid, grid), dtype=np.float32)

    for sigma, weight in [
        (rng.uniform(1.0, 2.5), 1.5),
        (rng.uniform(3.0, 7.0), 1.0),
        (rng.uniform(8.0, 16.0), 0.5),
    ]:
        layer  = rng.uniform(0, 1, (grid, grid)).astype(np.float32)
        layer  = gaussian_filter(layer, sigma=sigma)
        layer /= layer.max() + 1e-8
        img   += weight * layer

    img /= img.max() + 1e-8

    # gamma > 1 suppresses the smooth midtones and separates bright clouds.
    img  = np.power(img, gamma)
    img /= img.max() + 1e-8

    center_x = grid // 2 if center_x is None else center_x
    center_y = grid // 2 if center_y is None else center_y
    phi_rad = np.radians(phi)
    y, x    = np.mgrid[0:grid, 0:grid]
    dx      = (x - center_x).astype(np.float32)
    dy      = (y - center_y).astype(np.float32)
    r       = np.sqrt(dx**2 + dy**2) + 1e-8
    axis    = np.array([np.cos(phi_rad), np.sin(phi_rad)], dtype=np.float32)
    vdir    = np.stack([dx, dy], axis=-1) / r[..., None]
    cosang  = np.clip(np.sum(vdir * axis, axis=-1), -1, 1)
    angle   = np.arccos(cosang)

    cone_region  = np.exp(-(angle / np.radians(opening_angle * 0.8))**2)
    # A broad continuous envelope avoids an artificial circular arc at r_max.
    cone_region *= np.exp(-0.5 * (r / r_max) ** 2)

    density = (1.0 - scale) + scale * img
    return (density * cone_region).astype(np.float32)

def dust_clumps(
    grid, phi, opening_angle, r_max, depth, scales,
    center_x=None, center_y=None,
):
    """
    Irregular clumpy dust absorption inside the cone.
    Not geometric — no hard edges, no symmetric pattern.
    Mild depth only: secondary to inclination asymmetry.
    """
    center_x = grid // 2 if center_x is None else center_x
    center_y = grid // 2 if center_y is None else center_y
    phi_rad = np.radians(phi)
    y, x    = np.mgrid[0:grid, 0:grid]
    dx      = (x - center_x).astype(np.float32)
    dy      = (y - center_y).astype(np.float32)
    r       = np.sqrt(dx**2 + dy**2) + 1e-8

    dust = np.zeros((grid, grid), dtype=np.float32)
    for sigma in scales:
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
    # A broad continuous envelope avoids an artificial circular arc at r_max.
    cone_region *= np.exp(-0.5 * (r / r_max) ** 2)

    absorption = 1.0 - depth * cone_region * dust

    return absorption.astype(np.float32)

# =========================================================
# SAMPLE GENERATION
# =========================================================

def sample_params(grid, sample_seed):
    is_negative = np.random.rand() < NEGATIVE_FRAC
    negative_kind = (
        str(np.random.choice(mixture_config["negative_types"]))
        if is_negative else None
    )
    offset_radius = sample_range("geometry", "center_offset_pixels")
    offset_pa = np.random.uniform(0, 2 * np.pi)
    center_x = grid / 2 + offset_radius * np.cos(offset_pa)
    center_y = grid / 2 + offset_radius * np.sin(offset_pa)
    cone_boost = sample_range("emission", "cone_boost")
    noise = sample_range("instrument", "noise_sigma")
    contrast_proxy = cone_boost / max(noise, 1e-6)
    difficulty = "easy" if contrast_proxy > 1200 else "hard" if contrast_proxy < 300 else "medium"
    positive_distractor = (
        not is_negative
        and np.random.rand() < mixture_config["distractor_on_positive_fraction"]
    )
    positive_distractor_types = [
        kind for kind in mixture_config["negative_types"] if kind != "diffuse"
    ]
    intrinsic_bicone = bool(not is_negative and np.random.rand() < mixture_config["bicone_fraction"])
    counter_lobe_obscured = bool(
        intrinsic_bicone and np.random.rand() < OBSCURED_COUNTER_FRAC
    )
    counter_transmission = (
        sample_range("obscuration", "counter_transmission")
        if counter_lobe_obscured else 1.0
    )
    return {
        "sample_seed":   int(sample_seed),
        "is_negative":   is_negative,
        "negative_kind": negative_kind,
        "has_agn":       not is_negative,
        "difficulty":     difficulty,
        "intrinsic_bicone": intrinsic_bicone,
        "counter_lobe_obscured": counter_lobe_obscured,
        "counter_transmission": counter_transmission,
        "bicone":        intrinsic_bicone and not counter_lobe_obscured,
        "phi":           sample_range("geometry", "position_angle_deg"),
        "opening":       sample_range("geometry", "opening_angle_deg"),
        "opening_scale": float(np.random.uniform(0.8, 1.2)),
        "r_max":         grid * sample_range("geometry", "radius_fraction"),
        "inner_radius":  grid * sample_range("geometry", "inner_radius_fraction"),
        "inclination":   sample_range("geometry", "inclination_deg"),
        "center_x":      float(center_x),
        "center_y":      float(center_y),
        "center_offset": float(offset_radius),
        "warp_amplitude": sample_range("geometry", "warp_amplitude_deg"),
        "warp_scale":    grid * sample_range("geometry", "warp_scale_fraction"),
        "warp_direction": float(np.random.uniform(0, 360)),
        "path_width_fraction": sample_range("geometry", "path_width_fraction"),
        "pathway_strength": sample_range("geometry", "pathway_strength"),
        "path_width_tracking": sample_range("geometry", "path_width_tracking"),
        "edge_emission_tracking": sample_range("geometry", "edge_emission_tracking"),
        "lobe_ratio":     sample_range("geometry", "lobe_ratio"),
        "boost_str":      cone_boost,
        "wisp_amp":       sample_range("emission", "wisp_amplitude"),
        "wisp_count":     sample_int_range("emission", "wisp_count"),
        "halo_radius":    grid * sample_range("emission", "halo_radius_fraction"),
        "diffuse_fraction": sample_range("emission", "diffuse_fraction"),
        "texture_strength": sample_range("emission", "texture_strength"),
        "texture_gamma": sample_range("emission", "texture_gamma"),
        "smooth_cone_fraction": sample_range("emission", "smooth_cone_fraction"),
        "grain_count": sample_int_range("emission", "grain_count"),
        "grain_sigma_min": CONFIG["emission"]["grain_sigma_pixels"][0],
        "grain_sigma_max": CONFIG["emission"]["grain_sigma_pixels"][1],
        "angular_boundary_roughness": sample_range(
            "morphology", "angular_boundary_roughness"
        ),
        "radial_boundary_roughness": sample_range(
            "morphology", "radial_boundary_roughness"
        ),
        "boundary_noise_scale": sample_range(
            "morphology", "boundary_noise_scale_pixels"
        ),
        "radial_edge_width": sample_range(
            "morphology", "radial_edge_width_pixels"
        ),
        "fragmentation_strength": sample_range(
            "morphology", "fragmentation_strength"
        ),
        "dust_depth":     sample_range("obscuration", "dust_depth"),
        "dust_scale_small": sample_range("obscuration", "dust_scale_small"),
        "dust_scale_large": sample_range("obscuration", "dust_scale_large"),
        "nucleus_sigma":  sample_range("instrument", "nucleus_sigma_pixels"),
        "nucleus_amplitude": sample_range("instrument", "nucleus_amplitude"),
        "psf_sigma":      sample_range("instrument", "psf_sigma_pixels"),
        "noise":          noise,
        "intensity_gamma": sample_range("instrument", "intensity_gamma"),
        "has_disk":       np.random.rand() < CONFIG["background"]["disk_fraction"],
        "disk_pa":        float(np.random.uniform(0, 180)),
        "disk_axis_ratio": sample_range("background", "disk_axis_ratio"),
        "disk_radius":    grid * sample_range("background", "disk_radius_fraction"),
        "disk_amplitude": sample_range("background", "disk_amplitude"),
        "gradient_amplitude": sample_range("background", "gradient_amplitude"),
        "has_distractor": bool(positive_distractor),
        "distractor_kind": (
            str(np.random.choice(positive_distractor_types))
            if positive_distractor else None
        ),
        "distractor_pa": float(np.random.uniform(0, 360)),
        "distractor_strength": float(np.random.uniform(0.15, 0.65)),
        "distractor_width": float(np.random.uniform(0.8, 4.0)),
        "distractor_length": float(np.random.uniform(grid * 0.2, grid * 0.48)),
        "distractor_radius": float(np.random.uniform(grid * 0.12, grid * 0.35)),
    }

def generate_sample(sample_seed):
    grid   = GRID
    np.random.seed(sample_seed)
    params = sample_params(grid, sample_seed)

    if params["is_negative"]:
        img = hard_negative(grid, params["negative_kind"], params)
        return img, np.zeros((grid, grid), dtype=np.float32), params

    if "clouds" in CONFIG:
        params["clouds"] = sample_cloud_settings(sample_seed)

    phi         = params["phi"]
    opening     = params["opening"]
    r_max       = params["r_max"]
    inclination = params["inclination"]

    halo     = isotropic_halo(
        grid, params["halo_radius"], params["center_x"], params["center_y"]
    )
    emission = np.zeros((grid, grid), dtype=np.float32)
    cone_signal = np.zeros((grid, grid), dtype=np.float32)

    cone_phis = [phi, phi + 180] if params["intrinsic_bicone"] else [phi]

    for lobe_index, cone_phi in enumerate(cone_phis):

        illum = warped_cone(
            grid, cone_phi, opening, r_max, inclination, params
        )

        wisps          = wispy_gas(
            grid, cone_phi, opening, r_max, n_wisps=params["wisp_count"],
            center_x=params["center_x"], center_y=params["center_y"],
        )

        texture        = gas_texture(
            grid, cone_phi, opening, r_max, params["texture_strength"],
            params["texture_gamma"],
            lobe_seed=np.random.randint(0, 99999),
            center_x=params["center_x"], center_y=params["center_y"],
        )
        grains = granular_knots(
            grid, params["grain_count"], params["grain_sigma_min"],
            params["grain_sigma_max"],
        )
        cone_emission = halo * (
            params["smooth_cone_fraction"] * illum
            + params["boost_str"] * illum * texture
        )
        cone_emission += params["wisp_amp"] * wisps * illum
        cone_emission += params["wisp_amp"] * grains * illum

        if "clouds" in params:
            settings = params["clouds"]
            clouds, diffuse = cloud_components(
                grid, illum, settings,
                (int(sample_seed) + 13007 * (lobe_index + 1)) % (2**32),
            )
            yy, xx = np.mgrid[:grid, :grid]
            across = (-(xx - params["center_x"]) * np.sin(np.radians(cone_phi))
                      + (yy - params["center_y"]) * np.cos(np.radians(cone_phi)))
            radius = np.hypot(xx - params["center_x"], yy - params["center_y"])
            bridge_width = 1.5 + 0.08 * radius
            bridge = np.exp(-0.5 * (across / bridge_width) ** 2) * np.sqrt(diffuse)
            # Keep legacy random draws above for paired geometry comparisons.
            mixture = (
                settings["cloud_weight"] * normalize_component(clouds, illum)
                + settings["diffuse_weight"] * normalize_component(diffuse, illum)
                + settings["filament_weight"] * normalize_component(wisps, illum)
                + settings.get("bridge_weight", 0.0) * normalize_component(bridge, illum)
            )
            total_weight = sum(settings[key] for key in
                               ("cloud_weight", "diffuse_weight", "filament_weight"))
            total_weight += settings.get("bridge_weight", 0.0)
            cone_emission = params["boost_str"] * halo * illum * mixture / total_weight

        dust           = dust_clumps(
            grid, cone_phi, opening, r_max, params["dust_depth"],
            [params["dust_scale_small"], params["dust_scale_large"]],
            center_x=params["center_x"], center_y=params["center_y"],
        )
        cone_emission *= dust

        if lobe_index == 1:
            cone_emission *= params["lobe_ratio"] * params["counter_transmission"]

        emission += cone_emission

        # Preserve the cone-only flux for a detectability label. Dust and lobe
        # asymmetry have already been applied; contaminants are added later.
        cone_signal += cone_emission

    # faint isotropic background
    emission += params["diffuse_fraction"] * halo

    # nucleus — clean, not textured or obscured
    emission += nucleus(
        grid, params["nucleus_sigma"], params["nucleus_amplitude"],
        params["center_x"], params["center_y"],
    )

    # PSF blur — before disk so disk is not double-smoothed
    emission = gaussian_filter(emission, sigma=params["psf_sigma"])
    cone_signal = gaussian_filter(cone_signal, sigma=params["psf_sigma"])

    # host disk added after PSF — avoids double-smoothing
    if params["has_disk"]:
        emission += host_disk(grid, params)

    if params["has_distractor"]:
        emission += params["distractor_strength"] * hard_negative(
            grid, params["distractor_kind"], params
        )

    y, x = np.mgrid[0:grid, 0:grid]
    gradient_pa = np.radians(params["distractor_pa"])
    gradient = (
        (x - grid / 2) * np.cos(gradient_pa)
        + (y - grid / 2) * np.sin(gradient_pa)
    ) / grid
    emission += params["gradient_amplitude"] * gradient

    if CONFIG.get("labels", {}).get("mode", "connected_geometry") == "connected_geometry":
        target_mask = connected_cone_mask(grid, params)
    else:
        # Legacy label: detectable cone flux, potentially disconnected.
        snr_cut = CONFIG.get("labels", {}).get("detection_snr", 2.5)
        signal_noise = params["noise"] * np.sqrt(np.abs(emission) + 1e-6)
        peak_floor = CONFIG.get("labels", {}).get("peak_fraction_floor", 0.0)
        visible_mask = cone_signal >= np.maximum(
            snr_cut * np.maximum(signal_noise, 1e-8),
            peak_floor * cone_signal.max(),
        )
        yy, xx = np.mgrid[:grid, :grid]
        nuclear_radius = np.hypot(xx - params["center_x"], yy - params["center_y"])
        visible_mask &= nuclear_radius >= params["inner_radius"]
        target_mask = visible_mask.astype(np.float32)

    # Poisson-like shot noise
    shot      = np.random.normal(0, params["noise"], emission.shape)
    emission += shot * np.sqrt(np.abs(emission) + 1e-6)

    emission  = np.clip(emission, 0, None)
    emission /= emission.max() + 1e-8
    emission = np.power(emission, params["intensity_gamma"])

    return emission.astype(np.float32), target_mask, params

# =========================================================
# BUILD + SAVE
# =========================================================

def generate():
    seed_rng = np.random.RandomState(SEED)
    for i in range(N_SAMPLES):
        if i % 100 == 0:
            print(f"  {i}/{N_SAMPLES}...")
        sample_seed = int(seed_rng.randint(0, np.iinfo(np.int32).max))
        yield generate_sample(sample_seed)

def save(samples):
    sample_metadata = []
    preview_samples = []
    for i, (img, mask, params) in enumerate(samples):
        if len(preview_samples) < 9:
            preview_samples.append((img, mask, params))
        if i < int(N_SAMPLES * TRAIN_SPLIT):
            split = "train"
        elif i < int(N_SAMPLES * (TRAIN_SPLIT + VAL_SPLIT)):
            split = "val"
        else:
            split = "test"
        np.save(BASE_DIR / split / "images" / f"{i:05d}.npy", img)
        np.save(BASE_DIR / split / "masks"  / f"{i:05d}.npy", mask)
        sample_metadata.append({
            "sample_id": f"{i:05d}",
            "split": split,
            "foreground_fraction_at_0.5": float((mask > 0.5).mean()),
            **{key: (value.item() if isinstance(value, np.generic) else value)
               for key, value in params.items()},
        })
    with open(BASE_DIR / "samples.json", "w") as f:
        json.dump(sample_metadata, f, indent=2)
    return preview_samples, sample_metadata

# =========================================================
# VIZ
# =========================================================

def viz(samples):
    columns_per_sample = 2
    fig, axes = plt.subplots(3, 6, figsize=(15, 8))
    for i in range(9):
        img, mask, params = samples[i]
        col = (i % 3) * columns_per_sample
        row = i // 3
        axes[row][col].imshow(
            np.arcsinh(10 * img) / np.arcsinh(10),
            cmap="gray", origin="lower", vmin=0, vmax=1
        )
        label = (params["negative_kind"] or
                 ("counter hidden" if params["counter_lobe_obscured"] else
                  "bicone" if params["bicone"] else "cone"))
        axes[row][col].set_title(f"image {i}: {label}", fontsize=8)
        axes[row][col].axis("off")
        axes[row][col + 1].imshow(
            mask, cmap="inferno", origin="lower", vmin=0, vmax=1
        )
        axes[row][col + 1].set_title(f"mask {i}", fontsize=8)
        axes[row][col + 1].axis("off")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "dataset_preview.png", dpi=160)
    plt.close(fig)

# =========================================================
# METADATA
# =========================================================

def save_metadata(sample_metadata):
    split_counts = {
        split: sum(item["split"] == split for item in sample_metadata)
        for split in ("train", "val", "test")
    }
    negative_count = sum(item["is_negative"] for item in sample_metadata)
    positive_items = [item for item in sample_metadata if not item["is_negative"]]
    negative_type_counts = {
        kind: sum(item["negative_kind"] == kind for item in sample_metadata)
        for kind in mixture_config["negative_types"]
    }
    difficulty_counts = {
        level: sum(item["difficulty"] == level for item in positive_items)
        for level in ("easy", "medium", "hard")
    }
    foreground_fractions = np.asarray([
        item["foreground_fraction_at_0.5"] for item in positive_items
    ], dtype=np.float64)
    meta = {
        "grid":          GRID,
        "samples":       N_SAMPLES,
        "seed":          SEED,
        "config_path":   str(args.config),
        "config":        CONFIG,
        "negative_fraction": NEGATIVE_FRAC,
        "obscured_counter_fraction": OBSCURED_COUNTER_FRAC,
        "negative_types": list(mixture_config["negative_types"]),
        "actual_mixture": {
            "split_counts": split_counts,
            "negative_count": negative_count,
            "negative_fraction": negative_count / len(sample_metadata),
            "negative_type_counts": negative_type_counts,
            "positive_count": len(positive_items),
            "bicone_count": sum(item["bicone"] for item in positive_items),
            "intrinsic_bicone_count": sum(item["intrinsic_bicone"] for item in positive_items),
            "obscured_counter_count": sum(item["counter_lobe_obscured"] for item in positive_items),
            "positive_distractor_count": sum(
                item["has_distractor"] for item in positive_items
            ),
            "difficulty_counts": difficulty_counts,
            "positive_foreground_fraction_at_0.5": (
                {
                    "min": float(foreground_fractions.min()),
                    "mean": float(foreground_fractions.mean()),
                    "max": float(foreground_fractions.max()),
                }
                if foreground_fractions.size else None
            ),
        },
        "model":         ("illumination-weighted compact clouds + correlated diffuse gas + filaments"
                          if "clouds" in CONFIG else
                          "offset nucleus + halo*boost + wisps*illum + texture + dust")
                         + " + host disk + gradients + optional hard distractor",
        "clouds":        CONFIG.get("clouds"),
        "hollow":        "sigmoid ramp centered at r_inner (0.04-0.18*grid) "
                         "— gradual onset, no abrupt bright edge",
        "warp":          "configured radial axis drift",
        "inclination":   "foreshortening drives near/far lobe asymmetry",
        "texture":       "fine-weighted (1.5/1.0/0.5) + configured gamma contrast; "
                         "independent per lobe; "
                         "Gaussian radial envelope without a hard cutoff",
        "wisp_sizes":    "three classes: knots 60%, filaments 25%, "
                         "diffuse 15%",
        "dust":          "clumpy absorption per lobe with Gaussian radial envelope; "
                         "ranges stored in config",
        "host_disk":     "added after PSF; amplitude and frequency stored in config",
        "mask":          ("single connected projected cone or bicone including nucleus"
                          if CONFIG.get("labels", {}).get("mode", "connected_geometry") == "connected_geometry"
                          else "cone-only post-PSF emission above configured signal-to-noise cutoff"),
        "noise":         "Poisson-like shot noise sqrt(signal)",
        "opening_range_deg": CONFIG["geometry"]["opening_angle_deg"],
    }
    with open(BASE_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    validate_config()
    if N_SAMPLES < 10:
        parser.error("--samples must be at least 10")
    if not 0.0 <= NEGATIVE_FRAC < 1.0:
        parser.error("--negative-frac must be in [0, 1)")
    np.random.seed(SEED)
    reset()
    samples = generate()
    preview_samples, sample_metadata = save(samples)
    viz(preview_samples)
    save_metadata(sample_metadata)
    print(f"\nDONE → {BASE_DIR}\n")
