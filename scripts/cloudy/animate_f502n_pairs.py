#!/usr/bin/env python3
"""Generate current F502N pairs and animate examples, components, and integration.

All frames are derived from generator outputs. Animations show independent
synthetic galaxies or numerical passband integration, never physical time.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/agn-f502n-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, label

from render_hst_f502n import load_curve, match_pixel_scale

ROOT = Path(__file__).resolve().parents[2]


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--generate", action="store_true",
                   help="Generate fresh cubes and render each through F502N first")
    p.add_argument("--count", type=int, default=6)
    p.add_argument("--seed", type=int, default=1068)
    p.add_argument("--config", type=Path,
                   default=Path("configs/cloudy/ngc1068_oiii_v1.json"))
    p.add_argument("--grid", type=Path,
                   default=Path("data/cloudy/ngc1068_oiii_full_v1/grid.npz"))
    p.add_argument("--data-dir", type=Path,
                   default=Path("data/cloudy/ngc1068_f502n_gif_v1"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/cloudy/ngc1068_f502n_gifs"))
    p.add_argument("--duration-ms", type=int, default=1500,
                   help="Duration per independent galaxy example")
    args = p.parse_args()
    if args.count < 1:
        p.error("--count must be positive")
    return args


def gif(paths: list[Path], output: Path, duration: int | list[int]) -> None:
    """Use one palette across frames to prevent changes in display colors."""
    thumbnails = []
    for path in paths:
        with Image.open(path) as im:
            thumb = im.convert("RGB")
            thumb.thumbnail((256, 192))
            thumbnails.append(thumb)
    sheet = Image.new("RGB", (256, 192*len(thumbnails)))
    for i, thumb in enumerate(thumbnails):
        sheet.paste(thumb, (0, 192*i))
    palette = sheet.quantize(colors=256)
    frames = []
    for path in paths:
        with Image.open(path) as im:
            frames.append(im.convert("RGB").quantize(
                palette=palette, dither=Image.Dither.NONE))
    frames[0].save(output, save_all=True, append_images=frames[1:],
                   duration=duration, loop=0, optimize=False, disposal=2)
    for frame in frames:
        frame.close()


def stretch(image: np.ndarray, ceiling: float) -> np.ndarray:
    return np.clip(np.arcsinh(12*np.clip(image, 0, None)/ceiling)
                   / np.arcsinh(12), 0, 1)


def panel(ax, image: np.ndarray, title: str, ceiling: float,
          mask: bool = False, overlay: np.ndarray | None = None) -> None:
    ax.imshow(image if mask else stretch(image, ceiling), origin="lower",
              cmap="gray" if mask else "magma", vmin=0, vmax=1)
    if overlay is not None:
        ax.contour(overlay, levels=[0.5], colors=["#4ae0ec"], linewidths=0.8)
    ax.set_title(title, fontsize=11)
    ax.set_axis_off()


def main() -> None:
    args = arguments()
    os.chdir(ROOT)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = args.output_dir/"frames"
    frame_dir.mkdir(exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    samples = []
    for seed in range(args.seed, args.seed+args.count):
        cube_path = args.data_dir/f"cube_{seed}.npz"
        pair_path = args.data_dir/f"f502n_{seed}.npz"
        if args.generate:
            subprocess.run([
                sys.executable, "scripts/cloudy/generate_oiii_cube.py",
                "--config", str(args.config), "--grid", str(args.grid),
                "--output", str(cube_path), "--seed", str(seed),
            ], check=True)
            subprocess.run([
                sys.executable, "scripts/cloudy/render_hst_f502n.py", str(cube_path),
                "--output", str(pair_path), "--seed", str(seed),
                "--input-pixel-scale", str(config["cube"]["pixel_scale_arcsec"]),
                "--input-psf-fwhm", str(config["cube"]["psf_fwhm_arcsec"]),
                "--preview", str(args.output_dir/f"preview_{seed}.png"),
            ], check=True)
        with np.load(pair_path) as d:
            arrays = {k: d[k] for k in d.files}
        meta = json.loads(pair_path.with_suffix(".json").read_text())
        cube_meta = json.loads(cube_path.with_suffix(".json").read_text())
        if not np.isfinite(arrays["input"]).all():
            raise ValueError(f"Nonfinite input: {pair_path}")
        if not np.isin(arrays["mask"], [0, 1]).all():
            raise ValueError(f"Nonbinary target: {pair_path}")
        components = label(arrays["mask"] > 0.5)[1]
        if components != 1:
            raise ValueError(f"Target has {components} components: {pair_path}")
        samples.append((seed, cube_path, pair_path, arrays, meta, cube_meta))
        print(f"Validated current F502N sample {seed}", flush=True)

    # One intensity scale across all samples and all flux panels; dust and
    # binary targets have explicitly different units and fixed 0..1 scales.
    ceiling = float(np.percentile(
        np.stack([s[3]["input"] for s in samples]), 99.7))
    if ceiling <= 0:
        raise ValueError("Expected a positive display scale")
    pair_frames, component_frames = [], []
    for seed, _, _, d, m, cm in samples:
        g = cm["geometry"]
        subtitle = (f"seed {seed} | PA {g['position_angle_deg']:.1f}° | "
                    f"inclination {g['inclination_deg_signed']:+.1f}° | "
                    f"half opening {g['opening_half_angle_deg']:.1f}° | "
                    f"lobes: {g['visible_lobes']}")
        fig, axes = plt.subplots(1, 3, figsize=(11, 4.4), constrained_layout=True)
        panel(axes[0], d["input"], "F502N training image", ceiling)
        panel(axes[1], d["input"], "Image + target boundary", ceiling,
              overlay=d["mask"])
        panel(axes[2], d["mask"], "Hard geometric target", ceiling, mask=True)
        fig.suptitle("Independent synthetic galaxies — current F502N pipeline\n"
                     +subtitle, fontsize=12)
        fig.supxlabel("Fixed asinh display scale; each frame is a different galaxy, not time",
                      fontsize=9)
        path = frame_dir/f"pair_{seed}.png"
        fig.savefig(path, dpi=105); plt.close(fig)
        pair_frames.append(path)

        # Gas already has the cube PSF. Apply only the renderer's remaining blur.
        extra_sigma = np.sqrt(max(m["target_psf_fwhm_arcsec"]**2-
                                 config["cube"]["psf_fwhm_arcsec"]**2, 0))
        extra_sigma /= m["pixel_scale_arcsec"]*2.35482
        gas_counts = gaussian_filter(d["filter_integrated_gas"]
                                     *m["gas_peak_counts"], extra_sigma)
        fig, axes = plt.subplots(2, 3, figsize=(10, 7.6), constrained_layout=True)
        panel(axes[0, 0], gas_counts, "Cloudy gas: cone + host [O III]", ceiling)
        panel(axes[0, 1], d["stellar_continuum"], "Attenuated stellar + nuclear continuum", ceiling)
        axes[0, 2].imshow(d["continuum_attenuation"], origin="lower",
                          cmap="gray", vmin=0, vmax=1)
        axes[0, 2].set_title("Continuum transmission (dark = more dust)", fontsize=10)
        axes[0, 2].set_axis_off()
        panel(axes[1, 0], d["clean"], "Combined, blurred F502N", ceiling)
        panel(axes[1, 1], d["input"], "With background noise", ceiling)
        panel(axes[1, 2], d["mask"], "Geometric target", ceiling, mask=True)
        fig.suptitle("Current generator components\n"+subtitle, fontsize=12)
        fig.supxlabel("Flux panels share one scale. Dust panel shows the continuum screen only.",
                      fontsize=9)
        path = frame_dir/f"components_{seed}.png"
        fig.savefig(path, dpi=105); plt.close(fig)
        component_frames.append(path)

    gif(pair_frames, args.output_dir/"f502n_image_mask_pairs.gif", args.duration_ms)
    gif(component_frames, args.output_dir/"f502n_components.gif", args.duration_ms)

    # Show how actual weighted wavelength contributions build the final 2D
    # observation. The same continuum, sky and noise remain in every frame.
    seed, cube_path, _, d, m, _ = samples[0]
    with np.load(cube_path) as source:
        cube = source["input"].astype(np.float64)
        wave = source["wavelength_angstrom"]
    throughput = d["filter_throughput"]
    weights = throughput*wave*np.gradient(wave)
    factor = m["input_pixel_scale_arcsec"]/m["pixel_scale_arcsec"]
    total = match_pixel_scale(np.tensordot(weights, cube, axes=(0, 0)), factor, 1)
    flux_scale = max(float(np.percentile(total, 99.8)), 1e-12)
    sigma = np.sqrt(max(m["target_psf_fwhm_arcsec"]**2-
                       config["cube"]["psf_fwhm_arcsec"]**2, 0))
    sigma /= m["pixel_scale_arcsec"]*2.35482
    contributions = np.stack([
        gaussian_filter(match_pixel_scale(layer, factor, 1), sigma)
        *(weight*m["gas_peak_counts"]/flux_scale)
        for layer, weight in zip(cube, weights)
    ])
    rendered_gas = gaussian_filter(
        d["filter_integrated_gas"].astype(np.float64)*m["gas_peak_counts"], sigma)
    base = d["input"].astype(np.float64)-rendered_gas
    accumulated = base.copy()
    contribution_ceiling = max(float(np.percentile(contributions, 99.9)), 1e-12)
    spectrum = contributions.sum(axis=(1, 2))
    filter_wave, filter_response = load_curve(Path(
        "configs/instruments/hst_wfpc2_pc_f502n.csv"))
    wavelength_frames = []
    for i, wavelength in enumerate(wave):
        accumulated += contributions[i]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), constrained_layout=True)
        panel(axes[0], contributions[i], f"F502N contribution at {wavelength:.2f} Å",
              contribution_ceiling)
        panel(axes[1], accumulated, "Accumulated flux + continuum + fixed noise", ceiling)
        ax = axes[2]
        ax.plot(wave, spectrum, color="tab:green", label="Detected gas per bin")
        ax.axvspan(wave[0], wavelength, color="tab:green", alpha=0.08)
        ax.axvline(wavelength, color="tab:red", linewidth=1.5)
        ax.set(xlabel="Observed wavelength (Å)", ylabel="Gas counts per spectral bin",
               xlim=(wave[0], wave[-1]), ylim=(0, spectrum.max()*1.1))
        ax.grid(alpha=0.2)
        ax2 = ax.twinx()
        ax2.plot(filter_wave, filter_response, color="tab:blue", linestyle="--")
        ax2.set(ylabel="F502N throughput", ylim=(0, filter_response.max()*1.15))
        ax.set_title("Spectrum × photon response; red cursor = current bin")
        fig.suptitle(f"F502N numerical integration — seed {seed} — bin {i+1}/{len(wave)}",
                     fontsize=12)
        fig.supxlabel("Integration order, not physical time. Fixed scales: contribution and cumulative panels differ.",
                      fontsize=9)
        path = frame_dir/f"integration_{i:03d}.png"
        fig.savefig(path, dpi=95); plt.close(fig)
        wavelength_frames.append(path)
    error = float(np.max(np.abs(accumulated-d["input"])))
    if error > 1e-5:
        raise ValueError(f"Integration did not reconstruct final image: max error {error}")
    gif(wavelength_frames, args.output_dir/"f502n_passband_integration.gif",
        [120]*(len(wave)-1)+[1800])

    # A compact overview for inspection without opening an animation.
    fig, axes = plt.subplots(2, int(np.ceil(args.count/2)),
                             figsize=(12, 7), squeeze=False, constrained_layout=True)
    for ax in axes.flat:
        ax.set_axis_off()
    for ax, (seed, _, _, d, _, cm) in zip(axes.flat, samples):
        panel(ax, d["input"], f"Seed {seed}: {cm['geometry']['visible_lobes']} lobes",
              ceiling, overlay=d["mask"])
    fig.suptitle("Fresh F502N synthetic images with geometric target boundaries")
    fig.savefig(args.output_dir/"f502n_contact_sheet.png", dpi=130)
    plt.close(fig)

    manifest = {
        "generator": "scripts/cloudy/generate_oiii_cube.py",
        "renderer": "scripts/cloudy/render_hst_f502n.py",
        "config": str(args.config), "grid": str(args.grid),
        "display": {"stretch": "asinh", "softening_fraction": 1/12,
                    "shared_flux_ceiling": ceiling},
        "integration_max_reconstruction_error": error,
        "interpretation": "Independent samples and numerical integration; no temporal evolution",
        "samples": [{"seed": s[0], "cube": str(s[1]), "pair": str(s[2]),
                     "geometry": s[5]["geometry"], "mask_components": 1}
                    for s in samples],
    }
    (args.output_dir/"manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.output_dir/"README.md").write_text(
        "# Current F502N diagnostics\n\n"
        "- `f502n_image_mask_pairs.gif`: independent galaxies, final input, target overlay, and hard mask.\n"
        "- `f502n_components.gif`: Cloudy cone + host gas, stellar/nuclear continuum, continuum dust screen, "
        "blurred image, noisy image, and target.\n"
        "- `f502n_passband_integration.gif`: actual wavelength contributions integrated through F502N. "
        "The last frame reconstructs the final input. The animation does not show physical time.\n\n"
        "Flux panels use a fixed asinh stretch; contribution panels use a separate fixed stretch "
        "because individual bins are much fainter than integrated images. "
        "Masks stay binary. Bright host gas outside the mask is included in the current generator. "
        "The main gas component is still restricted to the prescribed cone geometry. "
        "The displayed dust map is the continuum screen; gas attenuation is already baked into the cube.\n\n"
        "Regenerate from the project root:\n\n```bash\n"
        "venv/bin/python scripts/cloudy/animate_f502n_pairs.py --generate\n```\n"
        "Omit `--generate` to rebuild animations from the saved samples.\n")
    print(f"Saved three animations to {args.output_dir}; reconstruction error {error:.3g}", flush=True)


if __name__ == "__main__":
    main()
