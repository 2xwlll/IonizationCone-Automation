#!/usr/bin/env python3
"""
generate_realistic_bicone_training_v2.py

Generates 2000 synthetic ionized gas emission images with REALISTIC clumpy,
filamentary structure matching NGC 1068 observations.

Key improvements from v1:
- Clumpy emission uses fractal/multi-scale noise (Perlin-style)
- Filamentary structures are irregular and wispy, not straight lines
- Emission has "dirty gritty" texture with many small bright knots
- Cone MASK is smooth, but EMISSION inside is highly irregular
- Better noise modeling with proper correlation structure
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, rotate, zoom
from scipy import ndimage
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from typing import Tuple, List
import warnings

warnings.filterwarnings('ignore')

np.random.seed(42)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

IMG_SIZE = 256
N_SAMPLES = 2000
TRAIN_SPLIT = 0.8

OUT_DIR = Path("data/2d/synthetic_bicone_v3")

# ─────────────────────────────────────────────
# FRACTAL NOISE FOR REALISTIC TEXTURE
# ─────────────────────────────────────────────

def generate_perlin_noise(shape: Tuple[int, int], octaves: int = 4,
                          persistence: float = 0.5, lacunarity: float = 2.0) -> np.ndarray:
    """
    Generate Perlin-like noise for organic clumpy textures.
    Multiple octaves create fractal-like structure.
    """
    noise = np.zeros(shape)
    frequency = 1.0
    amplitude = 1.0

    for i in range(octaves):
        # Generate base noise at this scale
        base = np.random.randn(shape[0] // int(frequency) + 2,
                               shape[1] // int(frequency) + 2)

        # Upsample to full resolution
        from scipy.ndimage import zoom
        zoom_factor = [shape[0] / base.shape[0], shape[1] / base.shape[1]]
        upsampled = zoom(base, zoom_factor, order=1)
        upsampled = upsampled[:shape[0], :shape[1]]

        noise += upsampled * amplitude
        amplitude *= persistence
        frequency *= lacunarity

    return noise


def generate_fractal_clumps(shape: Tuple[int, int],
                            n_scales: int = 4,
                            base_scale: float = 30.0) -> np.ndarray:
    """
    Generate fractal-like clumpy emission structure.
    Creates the "dirty gritty" look of real ionized gas.
    """
    clumps = np.zeros(shape)

    # Multi-scale clump generation
    for scale_idx in range(n_scales):
        scale = base_scale / (2 ** scale_idx)
        n_clumps = int(50 * (2 ** scale_idx))

        for _ in range(n_clumps):
            cx = np.random.randint(0, shape[1])
            cy = np.random.randint(0, shape[0])

            # Irregular clump shapes (not perfect Gaussians)
            yy, xx = np.indices(shape)
            dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)

            # Vary amplitude and add irregularity
            amp = np.random.uniform(0.3, 1.5) / (scale_idx + 1)
            sigma = scale * np.random.uniform(0.5, 2.0)

            # Add some noise to the clump shape
            shape_noise = np.random.uniform(0.8, 1.2)
            clump = amp * np.exp(-(dist / sigma)**shape_noise)

            # Make clumps more "knotty" by thresholding
            if np.random.rand() < 0.3:
                clump = (clump > 0.3 * amp) * clump * 1.5

            clumps += clump

    return clumps


def generate_wispy_filaments(shape: Tuple[int, int], n_filaments: int = 15) -> np.ndarray:
    """
    Generate irregular, wispy filamentary structures.
    Real ionized gas has curved, branching filaments, not straight lines.
    """
    filaments = np.zeros(shape)

    for _ in range(n_filaments):
        # Random walk for filament path
        length = np.random.randint(50, 150)
        width = np.random.uniform(2, 6)
        amplitude = np.random.uniform(0.5, 2.0)

        # Starting point
        x = np.random.randint(shape[1] // 4, 3 * shape[1] // 4)
        y = np.random.randint(shape[0] // 4, 3 * shape[0] // 4)

        # Random walk with momentum (curved paths)
        angle = np.random.uniform(0, 2 * np.pi)
        angular_momentum = np.random.uniform(-0.1, 0.1)

        points = [(x, y)]
        for step in range(length):
            angle += angular_momentum + np.random.uniform(-0.15, 0.15)
            x += np.cos(angle) * 2
            y += np.sin(angle) * 2

            if 0 <= int(x) < shape[1] and 0 <= int(y) < shape[0]:
                points.append((int(x), int(y)))

        # Draw filament with variable width
        for i, (px, py) in enumerate(points):
            # Width varies along filament
            local_width = width * (1 + 0.3 * np.sin(i * 0.2))
            local_amp = amplitude * np.exp(-i / (length * 0.7))  # Fade at end

            yy, xx = np.indices(shape)
            dist = np.sqrt((xx - px)**2 + (yy - py)**2)
            filaments += local_amp * np.exp(-dist**2 / (2 * local_width**2))

    return filaments


def generate_bright_knots(shape: Tuple[int, int], n_knots: int = 30) -> np.ndarray:
    """
    Generate small bright knots of emission.
    Real ionized gas has many small bright spots.
    """
    knots = np.zeros(shape)

    for _ in range(n_knots):
        cx = np.random.randint(0, shape[1])
        cy = np.random.randint(0, shape[0])

        yy, xx = np.indices(shape)
        dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)

        # Small bright knots
        sigma = np.random.uniform(1, 3)
        amp = np.random.uniform(1.0, 3.0)

        knot = amp * np.exp(-dist**2 / (2 * sigma**2))

        # Add some sub-structure to knots
        if np.random.rand() < 0.5:
            sub_noise = np.random.randn(*shape) * 0.3
            sub_noise = gaussian_filter(sub_noise, sigma=1)
            knot = knot * (1 + sub_noise)

        knots += np.maximum(knot, 0)

    return knots


# ─────────────────────────────────────────────
# CONE MASK GENERATION (smooth geometric)
# ─────────────────────────────────────────────

@dataclass
class ConeParams:
    """Physical parameters for a 3D bicone"""
    # Center position (relative to image center)
    r_offset: float       # Radial offset from image center
    phi_offset: float     # Azimuthal angle around image center (0-360°)

    # 3D orientation angles
    theta: float          # Polar angle from line of sight (0°=face-on, 90°=edge-on)
    phi: float            # Azimuthal rotation around cone axis (0-360°)

    # Cone geometry
    opening_angle: float  # Half-opening angle of cone (degrees)
    cone_length: float    # Length of cone in pixels

    # Physical properties
    has_top_half: bool    # Is top half visible?
    has_bottom_half: bool # Is bottom half visible?
    is_obscured: bool     # Dust obscuration present?
    obscuration_frac: float  # How much of cone is obscured (0-1)

    # Classification
    is_realistic: bool    # Opening angle in realistic range?
    has_cone: bool        # Negative sample if False


def generate_cone_mask(shape: Tuple[int, int], params: ConeParams) -> np.ndarray:
    """
    Generate smooth binary mask for cone region.
    The mask is geometric, but emission inside will be clumpy.
    """
    if not params.has_cone:
        return np.zeros(shape, dtype=np.float32)

    # Image center with offset
    cx = shape[1] // 2 + int(params.r_offset * np.cos(np.deg2rad(params.phi_offset)))
    cy = shape[0] // 2 + int(params.r_offset * np.sin(np.deg2rad(params.phi_offset)))

    cx = np.clip(cx, shape[1] // 4, 3 * shape[1] // 4)
    cy = np.clip(cy, shape[0] // 4, 3 * shape[0] // 4)

    yy, xx = np.indices(shape)
    x = xx - cx
    y = yy - cy

    theta_rad = np.deg2rad(params.theta)
    phi_rad = np.deg2rad(params.phi)
    opening_rad = np.deg2rad(params.opening_angle)

    mask = np.zeros(shape, dtype=np.float32)

    # Rotate coordinates by phi
    xr = x * np.cos(phi_rad) + y * np.sin(phi_rad)
    yr = -x * np.sin(phi_rad) + y * np.cos(phi_rad)

    # Handle different viewing angles
    if params.theta < 15:  # Face-on
        r = np.sqrt(x**2 + y**2)
        r_max = params.cone_length * np.tan(opening_rad)

        for sign in [1, -1]:
            if (sign == 1 and not params.has_top_half) or \
               (sign == -1 and not params.has_bottom_half):
                continue

            cone_mask = (r < r_max) & (r > r_max * 0.05)
            mask[cone_mask] = 1.0

    elif params.theta > 75:  # Edge-on
        line_width = params.cone_length * np.tan(opening_rad) * 0.3

        for sign in [1, -1]:
            if (sign == 1 and not params.has_top_half) or \
               (sign == -1 and not params.has_bottom_half):
                continue

            line_mask = (np.abs(xr) < line_width) & \
                       (yr * sign > 0) & \
                       (np.abs(yr) < params.cone_length)
            mask[line_mask] = 1.0

    else:  # Intermediate angles
        projected_opening = np.arctan(np.tan(opening_rad) / np.cos(theta_rad))

        for sign in [1, -1]:
            if (sign == 1 and not params.has_top_half) or \
               (sign == -1 and not params.has_bottom_half):
                continue

            angle = np.arctan2(np.abs(xr), np.maximum(yr * sign, 0.01))
            cone_mask = (angle < projected_opening) & \
                       (yr * sign > 0) & \
                       (yr * sign < params.cone_length)
            mask[cone_mask] = 1.0

    # Soften edges
    mask = gaussian_filter(mask, sigma=2.0)

    # Apply obscuration
    if params.is_obscured and params.has_cone:
        obscuration = generate_obscuration_pattern(shape, params.obscuration_frac)
        mask *= (1 - 0.8 * obscuration)

    return mask


def generate_obscuration_pattern(shape: Tuple[int, int], frac: float) -> np.ndarray:
    """Create dust obscuration pattern"""
    pattern = np.zeros(shape, dtype=np.float32)

    n_patches = int(3 + frac * 8)
    for _ in range(n_patches):
        cx = np.random.randint(0, shape[1])
        cy = np.random.randint(0, shape[0])
        sigma = np.random.uniform(15, 50)

        yy, xx = np.indices(shape)
        dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        pattern += np.exp(-dist**2 / (2 * sigma**2)) * np.random.uniform(0.4, 1.0)

    return np.clip(pattern, 0, 1)


# ─────────────────────────────────────────────
# REALISTIC EMISSION GENERATION
# ─────────────────────────────────────────────

def generate_realistic_ionized_gas(shape: Tuple[int, int],
                                   cone_mask: np.ndarray,
                                   intensity: float) -> np.ndarray:
    """
    Generate realistic clumpy ionized gas emission inside cone region.
    """
    if intensity == 0:
        return np.zeros(shape, dtype=np.float32)

    # Multi-layer approach for realistic texture

    # Layer 1: Base diffuse glow
    base = generate_perlin_noise(shape, octaves=3, persistence=0.6)
    base = gaussian_filter(base, sigma=10)
    base = (base - base.min()) / (base.max() - base.min() + 1e-8)

    # Layer 2: Medium-scale clumps
    medium_clumps = generate_fractal_clumps(shape, n_scales=3, base_scale=25.0)
    medium_clumps = gaussian_filter(medium_clumps, sigma=1.5)

    # Layer 3: Fine filaments
    filaments = generate_wispy_filaments(shape, n_filaments=np.random.randint(10, 25))
    filaments = gaussian_filter(filaments, sigma=0.8)

    # Layer 4: Small bright knots
    knots = generate_bright_knots(shape, n_knots=np.random.randint(20, 50))

    # Combine layers with different weights
    emission = (
        0.2 * base +           # Diffuse background
        0.4 * medium_clumps +  # Medium clumps
        0.3 * filaments +      # Filaments
        0.3 * knots            # Bright knots
    )

    # Add high-frequency noise for "grit"
    high_freq = np.random.randn(*shape) * 0.1
    high_freq = gaussian_filter(high_freq, sigma=0.5)
    emission += high_freq

    # Mask to cone region
    emission *= cone_mask

    # Enhance contrast in masked region
    emission = np.maximum(emission, 0)
    emission = emission ** 0.7  # Gamma correction for more dynamic range

    # Scale by intensity
    emission *= intensity * 2.0

    # Final smoothing (PSF-like)
    emission = gaussian_filter(emission, sigma=1.2)

    return emission.astype(np.float32)


def generate_background_emission(shape: Tuple[int, int]) -> np.ndarray:
    """
    Generate background galaxy emission (outside cone region).
    Less structured than cone emission.
    """
    # Large-scale structure
    bg = generate_perlin_noise(shape, octaves=4, persistence=0.5)
    bg = gaussian_filter(bg, sigma=np.random.uniform(20, 40))

    # Add some faint clumps
    clumps = generate_fractal_clumps(shape, n_scales=2, base_scale=40.0)
    clumps = gaussian_filter(clumps, sigma=3.0) * 0.3

    # Continuum subtraction residuals (smooth gradients)
    x = np.linspace(-1, 1, shape[1])
    y = np.linspace(-1, 1, shape[0])
    xx, yy = np.meshgrid(x, y)
    grad = np.random.uniform(-0.3, 0.3) * xx + np.random.uniform(-0.3, 0.3) * yy

    emission = 0.4 * bg + 0.2 * clumps + 0.1 * grad

    # Normalize
    emission = emission - np.median(emission)
    emission = emission / (np.std(emission) + 1e-8) * 0.3

    return emission.astype(np.float32)


# ─────────────────────────────────────────────
# NOISE FUNCTIONS
# ─────────────────────────────────────────────

def add_photon_noise(image: np.ndarray, scale: float = 3000.0) -> np.ndarray:
    """Add Poisson photon noise"""
    lam = np.clip(image * scale, 0, None)
    noisy = np.random.poisson(lam.astype(np.float32)) / scale
    return noisy.astype(np.float32)


def add_read_noise(image: np.ndarray, sigma: float = 0.003) -> np.ndarray:
    """Add Gaussian read noise"""
    noise = np.random.normal(0, sigma, image.shape)
    return image + noise.astype(np.float32)


def convolve_psf(image: np.ndarray, fwhm: float = 2.0) -> np.ndarray:
    """Convolve with telescope PSF"""
    sigma = fwhm / 2.355
    return gaussian_filter(image, sigma=sigma)


def add_continuum_residuals(image: np.ndarray) -> np.ndarray:
    """Add smooth residuals from imperfect continuum subtraction"""
    # Large-scale smooth pattern
    residual = generate_perlin_noise(image.shape, octaves=2, persistence=0.7)
    residual = gaussian_filter(residual, sigma=np.random.uniform(30, 60))
    residual = residual * np.random.uniform(0.02, 0.06)

    return image + residual.astype(np.float32)


# ─────────────────────────────────────────────
# SAMPLE GENERATION
# ─────────────────────────────────────────────

def generate_cone_params(sample_idx: int) -> ConeParams:
    """Generate random cone parameters"""

    has_cone = np.random.rand() > 0.1  # 10% negative samples

    if not has_cone:
        return ConeParams(
            r_offset=np.random.uniform(0, 30),
            phi_offset=np.random.uniform(0, 360),
            theta=45.0,
            phi=0.0,
            opening_angle=35.0,
            cone_length=80.0,
            has_top_half=False,
            has_bottom_half=False,
            is_obscured=False,
            obscuration_frac=0.0,
            is_realistic=False,
            has_cone=False
        )

    # Viewing angle
    theta_choice = np.random.choice(['face', 'intermediate', 'edge'], p=[0.2, 0.5, 0.3])
    if theta_choice == 'face':
        theta = np.random.uniform(0, 25)
    elif theta_choice == 'intermediate':
        theta = np.random.uniform(25, 65)
    else:
        theta = np.random.uniform(65, 85)

    # Opening angle
    angle_type = np.random.choice(['narrow', 'realistic', 'wide'], p=[0.15, 0.7, 0.15])
    if angle_type == 'narrow':
        opening_angle = np.random.uniform(5, 20)
        is_realistic = False
    elif angle_type == 'realistic':
        opening_angle = np.random.uniform(20, 50)
        is_realistic = True
    else:
        opening_angle = np.random.uniform(50, 80)
        is_realistic = False

    phi = np.random.uniform(0, 360)
    phi_offset = np.random.uniform(0, 360)
    cone_length = np.random.uniform(60, 120)

    # Difficulty/intensity
    difficulty = np.random.choice(['easy', 'medium', 'hard'], p=[0.3, 0.5, 0.2])
    if difficulty == 'easy':
        intensity = np.random.uniform(1.2, 2.5)
    elif difficulty == 'medium':
        intensity = np.random.uniform(0.7, 1.2)
    else:
        intensity = np.random.uniform(0.3, 0.7)

    # Obscuration
    obscuration_pattern = np.random.choice(['none', 'partial', 'severe'], p=[0.6, 0.25, 0.15])
    is_obscured = obscuration_pattern != 'none'
    obscuration_frac = 0.0 if obscuration_pattern == 'none' else \
                       np.random.uniform(0.2, 0.5) if obscuration_pattern == 'partial' else \
                       np.random.uniform(0.5, 0.9)

    # Visibility
    visibility = np.random.choice([
        'both', 'top_only', 'bottom_only', 'top_partial', 'bottom_partial'
    ], p=[0.5, 0.15, 0.15, 0.1, 0.1])

    has_top_half = visibility in ['both', 'top_only', 'top_partial', 'bottom_partial']
    has_bottom_half = visibility in ['both', 'bottom_only', 'top_partial', 'bottom_partial']

    if visibility in ['top_partial', 'bottom_partial']:
        is_obscured = True
        obscuration_frac = max(obscuration_frac, 0.3)

    r_offset = np.random.uniform(0, 35)

    params = ConeParams(
        r_offset=r_offset,
        phi_offset=phi_offset,
        theta=theta,
        phi=phi,
        opening_angle=opening_angle,
        cone_length=cone_length,
        has_top_half=has_top_half,
        has_bottom_half=has_bottom_half,
        is_obscured=is_obscured,
        obscuration_frac=obscuration_frac,
        is_realistic=is_realistic,
        has_cone=True
    )

    # Store intensity separately
    params._intensity = intensity

    return params


def generate_sample(sample_idx: int) -> Tuple[np.ndarray, np.ndarray, ConeParams]:
    """Generate a single training sample"""

    params = generate_cone_params(sample_idx)

    # Generate smooth cone mask
    cone_mask = generate_cone_mask((IMG_SIZE, IMG_SIZE), params)

    # Generate realistic clumpy emission inside cone
    intensity = getattr(params, '_intensity', 1.0)
    cone_emission = generate_realistic_ionized_gas(
        (IMG_SIZE, IMG_SIZE),
        cone_mask,
        intensity
    )

    # Generate background emission
    background = generate_background_emission((IMG_SIZE, IMG_SIZE))

    # Combine
    image = background + cone_emission

    # Add noise in order
    image = add_photon_noise(image)
    image = convolve_psf(image)
    image = add_read_noise(image)
    image = add_continuum_residuals(image)

    # Normalize
    image = image - np.percentile(image, 1)
    image = image / (np.percentile(image, 99) + 1e-8)
    image = np.clip(image, 0, 1)

    # Binary mask for training
    binary_mask = (cone_mask > 0.1).astype(np.float32)

    return image, binary_mask, params


# ─────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("REALISTIC BICONE TRAINING DATA GENERATOR v2")
    print("(Clumpy, filamentary emission like NGC 1068)")
    print("=" * 60)
    print(f"\nGenerating {N_SAMPLES} samples ({IMG_SIZE}x{IMG_SIZE})")

    # Create directories
    for split in ['train', 'val']:
        for sub in ['images', 'masks']:
            (OUT_DIR / split / sub).mkdir(parents=True, exist_ok=True)

    n_train = int(N_SAMPLES * TRAIN_SPLIT)
    metadata = {}

    for i in range(N_SAMPLES):
        image, mask, params = generate_sample(i)

        is_train = i < n_train
        split = 'train' if is_train else 'val'

        sample_id = f"{i:05d}"
        np.save(OUT_DIR / split / "images" / f"X_{sample_id}.npy", image)
        np.save(OUT_DIR / split / "masks" / f"Y_{sample_id}.npy", mask)

        # Store metadata
        meta = asdict(params)
        meta['split'] = split
        meta['intensity'] = getattr(params, '_intensity', 1.0)
        metadata[sample_id] = meta

        if (i + 1) % 200 == 0:
            print(f"  Generated {i + 1}/{N_SAMPLES} samples...")

    # Save metadata
    with open(OUT_DIR / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Dataset saved to {OUT_DIR}")

    # Create visualizations
    create_visualizations(OUT_DIR, metadata, n_train)

    # Print statistics
    print_statistics(metadata)


def create_visualizations(out_dir: Path, metadata: dict, n_train: int):
    """Create diagnostic plots"""
    print("\nCreating visualizations...")

    # Sample visualization
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    fig.suptitle("Realistic Clumpy Emission Samples", fontsize=14, fontweight='bold')

    viz_indices = [0, n_train // 2, min(n_train + 200, len(metadata) - 1)]

    for row, idx in enumerate(viz_indices):
        sample_id = f"{idx:05d}"
        params = metadata[sample_id]
        split = params['split']

        img = np.load(out_dir / split / "images" / f"X_{sample_id}.npy")
        mask = np.load(out_dir / split / "masks" / f"Y_{sample_id}.npy")

        axes[row, 0].imshow(img, cmap='magma', vmin=0, vmax=1)
        axes[row, 0].set_title(f"Sample {sample_id}")
        axes[row, 0].axis('off')

        axes[row, 1].imshow(mask, cmap='gray')
        axes[row, 1].set_title("Cone Mask")
        axes[row, 1].axis('off')

        overlay = np.stack([img, img, img], axis=-1)
        overlay[..., 1] = np.clip(overlay[..., 1] + mask * 0.4, 0, 1)
        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title("Overlay")
        axes[row, 2].axis('off')

        # Parameters
        if params['has_cone']:
            text = (
                f"θ={params['theta']:.0f}° φ={params['phi']:.0f}°\n"
                f"Opening={params['opening_angle']:.0f}°\n"
                f"Intensity={params.get('intensity', 1.0):.2f}\n"
                f"Visible: {('Both' if params['has_top_half'] and params['has_bottom_half'] else 'Partial')}\n"
                f"Obscured: {params['is_obscured']}"
            )
        else:
            text = "NEGATIVE\n(No cone)"

        axes[row, 3].text(0.1, 0.5, text, fontsize=9, family='monospace',
                         verticalalignment='center', transform=axes[row, 3].transAxes)
        axes[row, 3].axis('off')

    plt.tight_layout()
    plt.savefig(out_dir / "diagnostics.png", dpi=200)
    plt.close()

    # Parameter distributions
    create_distribution_plots(out_dir, metadata)


def create_distribution_plots(out_dir: Path, metadata: dict):
    """Create parameter distribution plots"""

    # Extract data
    thetas = [p['theta'] for p in metadata.values() if p['has_cone']]
    phis = [p['phi'] for p in metadata.values() if p['has_cone']]
    openings = [p['opening_angle'] for p in metadata.values() if p['has_cone']]
    n_negative = sum(1 for p in metadata.values() if not p['has_cone'])
    n_realistic = sum(1 for p in metadata.values() if p['is_realistic'] and p['has_cone'])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Dataset Parameter Distributions", fontsize=14, fontweight='bold')

    axes[0, 0].hist(thetas, bins=30, color='steelblue', edgecolor='white')
    axes[0, 0].set_xlabel('Theta (viewing angle) [degrees]')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Viewing Angle Distribution')

    axes[0, 1].hist(openings, bins=30, color='forestgreen', edgecolor='white')
    axes[0, 1].axvline(20, color='red', linestyle='--', alpha=0.7)
    axes[0, 1].axvline(50, color='red', linestyle='--', alpha=0.7)
    axes[0, 1].set_xlabel('Opening Angle [degrees]')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Opening Angle Distribution')

    axes[1, 0].hist(phis, bins=30, color='coral', edgecolor='white')
    axes[1, 0].set_xlabel('Phi (azimuthal) [degrees]')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Azimuthal Orientation')

    labels = ['Negative', 'Realistic (20-50°)', 'Unrealistic']
    sizes = [n_negative, n_realistic, len(metadata) - n_negative - n_realistic]
    colors = ['lightgray', 'forestgreen', 'coral']
    axes[1, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[1, 1].set_title('Sample Types')

    plt.tight_layout()
    plt.savefig(out_dir / "parameter_distributions.png", dpi=200)
    plt.close()


def print_statistics(metadata: dict):
    """Print dataset statistics"""
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    n_negative = sum(1 for p in metadata.values() if not p['has_cone'])
    n_realistic = sum(1 for p in metadata.values() if p['is_realistic'] and p['has_cone'])

    print(f"\nTotal samples: {len(metadata)}")
    print(f"  Negative (no cone): {n_negative} ({100*n_negative/len(metadata):.1f}%)")
    print(f"  Realistic opening angles: {n_realistic}")

    thetas = [p['theta'] for p in metadata.values() if p['has_cone']]
    openings = [p['opening_angle'] for p in metadata.values() if p['has_cone']]

    print(f"\nTheta range: {min(thetas):.1f}° - {max(thetas):.1f}° (mean: {np.mean(thetas):.1f}°)")
    print(f"Opening angle range: {min(openings):.1f}° - {max(openings):.1f}° (mean: {np.mean(openings):.1f}°)")

    print("\n✓ Dataset generation complete!")
    print(f"\nTo train: update train.py with:")
    print(f'  DATASET_CONFIG["name_hint"] = "synthetic_bicone_v3"')


if __name__ == "__main__":
    main()
