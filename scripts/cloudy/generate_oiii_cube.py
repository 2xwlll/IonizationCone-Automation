#!/usr/bin/env python3
"""Project a Cloudy emissivity grid into a wavelength-resolved 2D bicone.

The output input array has shape ``(wavelength, y, x)``.  A conventional 2D
U-Net treats wavelength bins as input channels and predicts one ``(y, x)``
geometric mask.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

C_KMS = 299792.458
OIII_LINES_A = {"OIII_4959": 4958.91, "OIII_5007": 5006.84}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=Path("configs/cloudy/ngc1068_oiii_v1.json"))
    parser.add_argument("--grid", type=Path,
                        default=Path("data/cloudy/ngc1068_oiii_v1/grid.npz"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/cloudy/ngc1068_oiii_v1/example_cube.npz"))
    parser.add_argument("--seed", type=int, default=1068)
    parser.add_argument("--spatial-pixels", type=int, default=None)
    return parser.parse_args()


def nearest_models(grid: dict, log_u: np.ndarray, log_nh: np.ndarray,
                   metallicity: np.ndarray, dust_scale: np.ndarray,
                   pah_scale: np.ndarray) -> np.ndarray:
    scales = {"log_u": 0.75, "log_nh_cm3": 1.0,
              "metallicity_solar": 0.4, "dust_scale": 0.5,
              "pah_scale": 0.1}
    distance = (
        ((log_u[..., None] - grid["log_u"]) / scales["log_u"]) ** 2
        + ((log_nh[..., None] - grid["log_nh_cm3"]) / scales["log_nh_cm3"]) ** 2
        + ((metallicity[..., None] - grid["metallicity_solar"]) /
           scales["metallicity_solar"]) ** 2
        + ((dust_scale[..., None] - grid["dust_scale"]) /
           scales["dust_scale"]) ** 2
        + ((pah_scale[..., None] - grid["pah_scale"]) /
           scales["pah_scale"]) ** 2
    )
    return np.argmin(distance, axis=-1)


def downsample_mean(array: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return array
    *lead, height, width = array.shape
    return array.reshape(*lead, height // factor, factor,
                         width // factor, factor).mean(axis=(-3, -1))


def sample_range(rng: np.random.Generator, values: list[float]) -> float:
    return float(rng.uniform(float(values[0]), float(values[1])))


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    cfg = config["cube"]
    sampling = config.get("sampling", {})
    rng = np.random.default_rng(args.seed)

    with np.load(args.grid) as loaded:
        grid = {key: loaded[key] for key in loaded.files}
    if len(grid["log_u"]) < 2:
        raise ValueError("The Cloudy grid needs at least two models for spatial variation")

    size = args.spatial_pixels or int(cfg["spatial_pixels"])
    oversample = int(cfg["oversample"])
    high_size = size * oversample
    yy, xx = np.indices((high_size, high_size), dtype=np.float64)
    x = (xx - (high_size - 1) / 2) / (high_size / 2)
    y = (yy - (high_size - 1) / 2) / (high_size / 2)

    pa_deg = (sample_range(rng, sampling["position_angle_deg"])
              if "position_angle_deg" in sampling else float(cfg["position_angle_deg"]))
    inclination_abs_deg = (sample_range(rng, sampling["inclination_abs_deg"])
                           if "inclination_abs_deg" in sampling
                           else abs(float(cfg["inclination_deg"])))
    near_side_sign = float(rng.choice([-1.0, 1.0]))
    inclination_deg = near_side_sign * inclination_abs_deg
    opening_deg = (sample_range(rng, sampling["opening_half_angle_deg"])
                   if "opening_half_angle_deg" in sampling
                   else float(cfg["opening_half_angle_deg"]))
    cone_radius = (sample_range(rng, sampling["cone_radius_fraction"])
                   if "cone_radius_fraction" in sampling else 0.92)
    pa = np.deg2rad(pa_deg)
    axial = x * np.sin(pa) + y * np.cos(pa)
    cross = x * np.cos(pa) - y * np.sin(pa)
    radius = np.hypot(axial, cross)
    half_angle = np.deg2rad(opening_deg)
    projected_half_angle = np.arctan(
        np.tan(half_angle) / max(np.cos(np.deg2rad(inclination_abs_deg)), 0.2)
    )
    mask_hi = ((np.abs(cross) <= np.abs(axial) * np.tan(projected_half_angle))
               & (radius <= cone_radius))
    visible_lobes = "both"
    if rng.random() < float(sampling.get("single_visible_lobe_fraction", 0.0)):
        visible_sign = float(rng.choice([-1.0, 1.0]))
        mask_hi &= axial * visible_sign >= 0
        visible_lobes = "positive" if visible_sign > 0 else "negative"
    # The nucleus joins both lobes into one exact geometric component.
    mask_hi |= radius <= (2.5 / high_size)

    small = gaussian_filter(rng.normal(size=mask_hi.shape), 2.0 * oversample)
    large = gaussian_filter(rng.normal(size=mask_hi.shape), 9.0 * oversample)
    texture = np.exp(1.15 * small / max(small.std(), 1e-8)
                     + 0.45 * large / max(large.std(), 1e-8))
    radial_falloff = np.exp(-2.0 * radius)
    diffuse = 0.16 + 0.28 * gaussian_filter(mask_hi.astype(float), 2.5 * oversample)
    gas_weight = mask_hi * radial_falloff * (diffuse + texture)

    density_field = gaussian_filter(rng.normal(size=mask_hi.shape), 5.0 * oversample)
    density_field /= max(density_field.std(), 1e-8)
    log_nh = np.clip(3.0 + 0.62 * density_field, 2.0, 4.0)
    # U decreases approximately as r^-2, with a floor at the nucleus.
    log_u = np.clip(-1.55 - 1.45 * np.log10(1.0 + 8.0 * radius), -3.0, -1.5)
    metallicity = np.clip(1.25 - 0.35 * radius + 0.12 * large / max(large.std(), 1e-8),
                          0.7, 1.5)
    dust_scale = np.where(large > np.median(large), 1.0, 0.5)
    # PAHs survive preferentially in dustier, lower-ionization cells.
    pah_scale = np.where((dust_scale >= 1.0) & (log_u < -2.2), 0.1, 0.0)
    model_index = nearest_models(grid, log_u, log_nh, metallicity,
                                 dust_scale, pah_scale)

    inclination = np.deg2rad(inclination_deg)
    vmax = float(cfg["maximum_outflow_kms"])
    velocity = (np.sign(axial) * vmax * np.sin(inclination)
                * np.clip(radius / 0.75, 0.0, 1.0))
    velocity += gaussian_filter(rng.normal(0, 35.0, mask_hi.shape),
                                2.0 * oversample)
    velocity *= mask_hi

    wavelengths = np.linspace(cfg["wavelength_min_angstrom"],
                              cfg["wavelength_max_angstrom"],
                              int(cfg["wavelength_channels"]), dtype=np.float64)
    cube_hi = np.zeros((len(wavelengths), high_size, high_size), dtype=np.float64)
    sigma_v = float(cfg["velocity_dispersion_kms"])
    z = float(config["redshift"])
    for name, rest_a in OIII_LINES_A.items():
        line_strength = grid[name][model_index]
        amplitude = gas_weight * line_strength
        center = rest_a * (1.0 + z) * (1.0 + velocity / C_KMS)
        sigma_a = rest_a * (1.0 + z) * sigma_v / C_KMS
        profile = np.exp(-0.5 * ((wavelengths[:, None, None]
                                 - center[None, :, :]) / sigma_a) ** 2)
        profile /= sigma_a * np.sqrt(2.0 * np.pi)
        cube_hi += amplitude[None, :, :] * profile

    # Non-cone [O III] prevents a trivial brightness-to-mask shortcut.  It
    # represents lower-ionization host/NLR clouds and has disk rotation rather
    # than the coherent radial bicone velocity field.
    host_fraction = sample_range(
        rng, sampling.get("host_oiii_fraction", [0.03, 0.18])
    )
    disk_axis_ratio = rng.uniform(0.28, 0.65)
    disk_pa = rng.uniform(0.0, 2.0 * np.pi)
    disk_x = x * np.cos(disk_pa) + y * np.sin(disk_pa)
    disk_y = -x * np.sin(disk_pa) + y * np.cos(disk_pa)
    disk_radius = np.sqrt(disk_x**2 + (disk_y / disk_axis_ratio)**2)
    host_field = gaussian_filter(rng.normal(size=mask_hi.shape),
                                 4.0 * oversample)
    host_texture = np.exp(0.85 * host_field / max(host_field.std(), 1e-8))
    host_weight = host_fraction * np.exp(-3.5 * disk_radius) * host_texture
    host_weight *= ~mask_hi
    host_log_u = np.full(mask_hi.shape, -3.0)
    host_log_nh = np.clip(2.4 + 0.35 * density_field, 2.0, 4.0)
    host_model_index = nearest_models(
        grid, host_log_u, host_log_nh, metallicity, dust_scale, pah_scale
    )
    host_velocity = 180.0 * disk_x * np.sin(rng.uniform(0.2, 1.2))
    for name, rest_a in OIII_LINES_A.items():
        host_amplitude = host_weight * grid[name][host_model_index]
        host_center = rest_a * (1.0 + z) * (1.0 + host_velocity / C_KMS)
        sigma_a = rest_a * (1.0 + z) * sigma_v / C_KMS
        host_profile = np.exp(-0.5 * ((wavelengths[:, None, None]
                                      - host_center[None, :, :]) / sigma_a) ** 2)
        host_profile /= sigma_a * np.sqrt(2.0 * np.pi)
        cube_hi += host_amplitude[None, :, :] * host_profile

    if "continuum_outward_flambda" not in grid:
        raise ValueError(
            "Cloudy grid has no emergent continuum; rebuild it with build_oiii_grid.py"
        )
    cloudy_continuum = grid["continuum_outward_flambda"]
    if cloudy_continuum.shape != (len(grid["log_u"]), len(wavelengths)):
        raise ValueError(
            f"Continuum grid shape {cloudy_continuum.shape} does not match "
            f"({len(grid['log_u'])}, {len(wavelengths)})"
        )
    for channel in range(len(wavelengths)):
        cube_hi[channel] += (
            gas_weight * cloudy_continuum[:, channel][model_index]
            + host_weight * cloudy_continuum[:, channel][host_model_index]
        )

    # Foreground host-galaxy dust acts along the final line of sight and is
    # separate from the grains mixed into each Cloudy cloud calculation.
    tau_5007 = sample_range(
        rng, sampling.get("foreground_dust_tau_5007", [0.0, 0.8])
    )
    dust_lane = gaussian_filter(rng.normal(size=mask_hi.shape), 14.0 * oversample)
    dust_lane = (dust_lane - dust_lane.min()) / max(np.ptp(dust_lane), 1e-8)
    attenuation = np.exp(-tau_5007 * dust_lane)
    cube_hi *= attenuation[None, :, :]

    cube = downsample_mean(cube_hi, oversample).astype(np.float32)
    velocity_out = downsample_mean(velocity, oversample).astype(np.float32)
    mask = downsample_mean(mask_hi.astype(float), oversample) > 0.5
    center_pixel = size // 2
    mask[center_pixel - 1:center_pixel + 2,
         center_pixel - 1:center_pixel + 2] = True
    psf_sigma_pixels = (float(cfg["psf_fwhm_arcsec"])
                        / float(cfg["pixel_scale_arcsec"]) / 2.35482)
    cube = gaussian_filter(cube, sigma=(0.0, psf_sigma_pixels, psf_sigma_pixels))
    scale = float(np.percentile(cube, 99.8))
    cube = (cube / max(scale, 1e-12)).astype(np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        input=cube,
        mask=mask.astype(np.float32),
        wavelength_angstrom=wavelengths,
        velocity_kms=velocity_out,
        rest_line_names=np.asarray(list(OIII_LINES_A)),
        rest_line_wavelength_angstrom=np.asarray(list(OIII_LINES_A.values())),
        channel_axis=np.asarray("wavelength_angstrom"),
    )
    # Direct U-Net pair: matching stems in separate image/mask directories.
    pair_root = args.output.parent / "training_pair"
    (pair_root / "images").mkdir(parents=True, exist_ok=True)
    (pair_root / "masks").mkdir(parents=True, exist_ok=True)
    np.save(pair_root / "images" / f"{args.output.stem}.npy", cube)
    np.save(pair_root / "masks" / f"{args.output.stem}.npy",
            mask.astype(np.float32))
    metadata = {
        "shape": list(cube.shape),
        "axis_order": ["wavelength", "y", "x"],
        "redshift": z,
        "pixel_scale_arcsec": cfg["pixel_scale_arcsec"],
        "distance_mpc": cfg["distance_mpc"],
        "parsec_per_pixel": (float(cfg["distance_mpc"]) * 1e6
                              * np.deg2rad(float(cfg["pixel_scale_arcsec"]) / 3600)),
        "cloudy_grid": str(args.grid),
        "continuum_source": "Cloudy outward emergent continuum with line bins subtracted",
        "cloudy_model_count": int(len(grid["log_u"])),
        "geometry": {
            "position_angle_deg": pa_deg,
            "inclination_deg_signed": inclination_deg,
            "opening_half_angle_deg": opening_deg,
            "cone_radius_fraction": cone_radius,
            "visible_lobes": visible_lobes,
        },
        "host_oiii_fraction": host_fraction,
        "foreground_dust_tau_5007": tau_5007,
        "seed": args.seed,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved cube {cube.shape} and mask {mask.shape} to {args.output}")


if __name__ == "__main__":
    main()
