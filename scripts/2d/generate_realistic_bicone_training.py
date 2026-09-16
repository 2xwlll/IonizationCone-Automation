#!/usr/bin/env python3
"""
generate_realistic_bicone_training.py

Generates 2000 synthetic ionized gas emission images with realistic noise patterns
for training a U-Net to detect AGN ionization cones.

Key Features:
- Clumpy ionized gas emission centered around varying (r, phi, theta) orientations
- Realistic bicone opening angles (15-50°) + unrealistic variations (5-15°, 50-80°)
- 3D cone projection with proper phi (azimuthal) and theta (polar) angle variation
- Random dust obscuration of top halves
- Completely missing cone negative samples
- Realistic noise: Poisson photon noise, read noise, PSF convolution,
  sky background, continuum subtraction residuals
- Outputs train/val split compatible with IonizationConeDataset2D

Output Structure:
    data/2d/synthetic_bicone_v2/
        train/
            images/  -> X_00000.npy ... X_01599.npy (1600 samples)
            masks/   -> Y_00000.npy ... Y_01599.npy
        val/
            images/  -> X_01600.npy ... X_01999.npy (400 samples)
            masks/   -> Y_01600.npy ... Y_01999.npy
        metadata.json  -> Contains all parameters for each sample
        diagnostics.png -> Visualization of 3 samples with masks
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, rotate
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from typing import Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

# Image dimensions - must be divisible by 16 for U-Net pooling layers
IMG_SIZE = 256
N_SAMPLES = 2000
TRAIN_SPLIT = 0.8  # 1600 train, 400 val

# Output paths
OUT_DIR = Path("data/2d/synthetic_bicone_v2")
TRAIN_IMG_DIR = OUT_DIR / "train" / "images"
TRAIN_MASK_DIR = OUT_DIR / "train" / "masks"
VAL_IMG_DIR = OUT_DIR / "val" / "images"
VAL_MASK_DIR = OUT_DIR / "val" / "masks"

# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────
# 3D CONE GEOMETRY PARAMETERS
# ─────────────────────────────────────────────

@dataclass
class ConeParams:
    """Physical parameters for a 3D bicone"""
    # Center position (relative to image center)
    r_offset: float       # Radial offset from image center (0-50 pixels)
    phi_offset: float     # Azimuthal angle around image center (0-360°)

    # 3D orientation angles
    theta: float          # Polar angle from line of sight (0°=face-on, 90°=edge-on)
    phi: float            # Azimuthal rotation around cone axis (0-360°)

    # Cone geometry
    opening_angle: float  # Half-opening angle of cone (degrees)
    cone_length: float    # Length of cone in pixels
    cone_intensity: float # Relative brightness of cone emission

    # Physical properties
    has_top_half: bool    # Is top half visible?
    has_bottom_half: bool # Is bottom half visible?
    is_obscured: bool     # Dust obscuration present?
    obscuration_frac: float  # How much of cone is obscured (0-1)

    # Classification
    is_realistic: bool    # Opening angle in realistic range?
    has_cone: bool        # Negative sample if False

# ─────────────────────────────────────────────
# NOISE AND INSTRUMENT PARAMETERS
# ─────────────────────────────────────────────

class NoiseParams:
    """Realistic noise parameters for HST-like observations"""
    # Photon noise (Poisson)
    PHOTON_SCALE = 5000.0      # Peak counts for bright emission
    SKY_LEVEL = 50.0           # Background sky counts

    # Read noise (Gaussian)
    READ_NOISE_SIGMA = 5.0     # electrons

    # PSF (HST/WFC3-like)
    PSF_FWHM_PX = 2.5          # Full width at half maximum in pixels

    # Continuum subtraction residuals
    RESIDUAL_GRADIENT_AMP = 0.02  # Amplitude of smooth residual structure
    RESIDUAL_CLUMP_SCALE = 30.0   # Scale of clumpy residuals

    # Clumpy gas properties
    N_CLUMPS_RANGE = (15, 40)     # Number of emission clumps
    CLUMP_SIGMA_RANGE = (3, 12)   # Size of clumps
    CLUMP_AMP_RANGE = (0.1, 0.8)  # Relative amplitude of clumps

# ─────────────────────────────────────────────
# GEOMETRY FUNCTIONS
# ─────────────────────────────────────────────

def spherical_to_cartesian(r: float, theta: float, phi: float) -> Tuple[float, float, float]:
    """
    Convert spherical to cartesian coordinates.
    theta: polar angle from z-axis (0 to pi)
    phi: azimuthal angle in xy-plane (0 to 2*pi)
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def project_3d_cone_to_2d(
    img_size: int,
    cone_params: ConeParams
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project a 3D bicone onto 2D image plane.

    Returns:
        emission_map: 2D array of cone emission
        mask: Binary mask of cone region
    """
    if not cone_params.has_cone:
        return np.zeros((img_size, img_size)), np.zeros((img_size, img_size))

    # Image center
    cx = img_size // 2 + int(cone_params.r_offset * np.cos(np.deg2rad(cone_params.phi_offset)))
    cy = img_size // 2 + int(cone_params.r_offset * np.sin(np.deg2rad(cone_params.phi_offset)))

    # Ensure center is within bounds
    cx = np.clip(cx, img_size // 4, 3 * img_size // 4)
    cy = np.clip(cy, img_size // 4, 3 * img_size // 4)

    # Create coordinate grids
    yy, xx = np.indices((img_size, img_size))
    x = xx - cx
    y = yy - cy

    # Convert theta/phi to radians
    theta_rad = np.deg2rad(cone_params.theta)
    phi_rad = np.deg2rad(cone_params.phi)
    opening_rad = np.deg2rad(cone_params.opening_angle)

    # Project 3D cone to 2D based on viewing angle
    if cone_params.theta < 15:  # Face-on: circular symmetry
        r = np.sqrt(x**2 + y**2)
        angle_from_pole = np.arctan2(r, 1e-6)  # Nearly face-on

        # Create cone emission (bicone appears as ring/annulus)
        emission = np.zeros_like(x, dtype=np.float32)
        mask = np.zeros_like(x, dtype=np.float32)

        # Opening angle projects to radius
        r_max = cone_params.cone_length * np.tan(opening_rad)

        for sign in [-1, 1]:  # Both cones of bicone
            if (sign == 1 and not cone_params.has_top_half) or \
               (sign == -1 and not cone_params.has_bottom_half):
                continue

            cone_mask = (r < r_max) & (r > r_max * 0.1)
            emission[cone_mask] += cone_params.cone_intensity * np.exp(-r[cone_mask] / (r_max * 0.5))
            mask[cone_mask] = 1.0

    elif cone_params.theta > 75:  # Edge-on: thin line
        # Rotate coordinates by phi
        xr = x * np.cos(phi_rad) + y * np.sin(phi_rad)
        yr = -x * np.sin(phi_rad) + y * np.cos(phi_rad)

        emission = np.zeros_like(x, dtype=np.float32)
        mask = np.zeros_like(x, dtype=np.float32)

        # Edge-on cones appear as lines extending from center
        for sign in [-1, 1]:
            if (sign == 1 and not cone_params.has_top_half) or \
               (sign == -1 and not cone_params.has_bottom_half):
                continue

            # Line along rotated y-axis
            line_width = cone_params.cone_length * np.tan(opening_rad) * 0.5
            line_mask = (np.abs(xr) < line_width) & \
                       (yr * sign > 0) & \
                       (np.abs(yr) < cone_params.cone_length)

            emission[line_mask] += cone_params.cone_intensity * \
                                   np.exp(-np.abs(yr[line_mask]) / (cone_params.cone_length * 0.3))
            mask[line_mask] = 1.0

    else:  # Intermediate angles: elliptical cones
        # Rotate by phi
        xr = x * np.cos(phi_rad) + y * np.sin(phi_rad)
        yr = -x * np.sin(phi_rad) + y * np.cos(phi_rad)

        # Project opening angle based on inclination
        projected_opening = np.arctan(np.tan(opening_rad) / np.cos(theta_rad))

        emission = np.zeros_like(x, dtype=np.float32)
        mask = np.zeros_like(x, dtype=np.float32)

        for sign in [-1, 1]:
            if (sign == 1 and not cone_params.has_top_half) or \
               (sign == -1 and not cone_params.has_bottom_half):
                continue

            # Elliptical cone projection
            r_proj = np.sqrt(xr**2 + (yr * np.cos(theta_rad))**2)
            angle = np.arctan2(np.abs(xr), np.maximum(yr * sign, 0.01))

            cone_mask = (angle < projected_opening) & \
                       (yr * sign > 0) & \
                       (yr * sign < cone_params.cone_length)

            # Intensity falls with distance
            dist_factor = np.exp(-np.abs(yr) / (cone_params.cone_length * 0.4))
            emission[cone_mask] += cone_params.cone_intensity * dist_factor[cone_mask]
            mask[cone_mask] = 1.0

    # Apply obscuration (dust)
    if cone_params.is_obscured and cone_params.has_cone:
        obscuration_mask = create_obscuration_pattern(img_size, cone_params.obscuration_frac)
        emission *= (1 - 0.7 * obscuration_mask)

    # Soften edges
    emission = gaussian_filter(emission, sigma=1.5)
    mask = gaussian_filter(mask, sigma=1.0)
    mask = (mask > 0.1).astype(np.float32)

    return emission, mask


def create_obscuration_pattern(img_size: int, frac: float) -> np.ndarray:
    """Create random dust obscuration pattern"""
    pattern = np.zeros((img_size, img_size), dtype=np.float32)

    n_dust_patches = int(5 + frac * 10)
    for _ in range(n_dust_patches):
        cx = np.random.randint(0, img_size)
        cy = np.random.randint(0, img_size)
        sigma = np.random.uniform(10, 40)

        yy, xx = np.indices((img_size, img_size))
        dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        pattern += np.exp(-dist**2 / (2 * sigma**2)) * np.random.uniform(0.3, 1.0)

    return np.clip(pattern, 0, 1)

# ─────────────────────────────────────────────
# BACKGROUND EMISSION GENERATION
# ─────────────────────────────────────────────

def generate_clumpy_background(img_size: int) -> np.ndarray:
    """
    Generate clumpy ionized gas background similar to continuum-subtracted emission.
    """
    # Start with correlated noise (smooth structure)
    bg = np.random.normal(0, 1, (img_size, img_size))
    bg = gaussian_filter(bg, sigma=np.random.uniform(20, 40))

    # Add filamentary structure (ionized gas tends to form filaments)
    n_filaments = np.random.randint(3, 8)
    for _ in range(n_filaments):
        # Random filament orientation
        angle = np.random.uniform(0, 360)
        length = np.random.uniform(img_size * 0.3, img_size * 0.8)
        width = np.random.uniform(3, 10)
        amplitude = np.random.uniform(0.3, 1.2)

        # Create filament
        filament = create_filament(img_size, angle, length, width, amplitude)
        bg += filament

    # Add emission clumps (ionized gas clouds)
    n_clumps = np.random.randint(*NoiseParams.N_CLUMPS_RANGE)
    for _ in range(n_clumps):
        cx = np.random.randint(0, img_size)
        cy = np.random.randint(0, img_size)
        sigma = np.random.uniform(*NoiseParams.CLUMP_SIGMA_RANGE)
        amp = np.random.uniform(*NoiseParams.CLUMP_AMP_RANGE)

        yy, xx = np.indices((img_size, img_size))
        dist_sq = (xx - cx)**2 + (yy - cy)**2
        bg += amp * np.exp(-dist_sq / (2 * sigma**2))

    # Add gradient (imperfect continuum subtraction)
    x = np.linspace(-1, 1, img_size)
    xx, yy = np.meshgrid(x, x)
    grad = np.random.uniform(-0.5, 0.5) * xx + np.random.uniform(-0.5, 0.5) * yy
    bg += NoiseParams.RESIDUAL_GRADIENT_AMP * grad

    # Normalize
    bg = bg - np.median(bg)
    bg = bg / (np.std(bg) + 1e-8) * 0.5

    return bg.astype(np.float32)


def create_filament(img_size: int, angle: float, length: float,
                    width: float, amplitude: float) -> np.ndarray:
    """Create a filamentary structure"""
    rad = np.deg2rad(angle)
    filament = np.zeros((img_size, img_size), dtype=np.float32)

    # Random start position
    cx = np.random.randint(img_size // 4, 3 * img_size // 4)
    cy = np.random.randint(img_size // 4, 3 * img_size // 4)

    # Create line
    t = np.linspace(-length/2, length/2, int(length))
    x_line = cx + t * np.cos(rad)
    y_line = cy + t * np.sin(rad)

    # Rasterize
    for x, y in zip(x_line, y_line):
        if 0 <= int(y) < img_size and 0 <= int(x) < img_size:
            yy, xx = np.indices((img_size, img_size))
            dist = np.sqrt((xx - x)**2 + (yy - y)**2)
            filament += amplitude * np.exp(-dist**2 / (2 * width**2))

    return filament

# ─────────────────────────────────────────────
# NOISE FUNCTIONS
# ─────────────────────────────────────────────

def add_photon_noise(image: np.ndarray, scale: float = None) -> np.ndarray:
    """Add Poisson photon noise"""
    if scale is None:
        scale = NoiseParams.PHOTON_SCALE

    # Ensure non-negative for Poisson
    lam = np.clip(image * scale, 0, None)
    noisy = np.random.poisson(lam) / scale
    return noisy.astype(np.float32)


def add_read_noise(image: np.ndarray, sigma: float = None) -> np.ndarray:
    """Add Gaussian read noise"""
    if sigma is None:
        sigma = NoiseParams.READ_NOISE_SIGMA / NoiseParams.PHOTON_SCALE

    noise = np.random.normal(0, sigma, image.shape)
    return image + noise.astype(np.float32)


def convolve_psf(image: np.ndarray, fwhm: float = None) -> np.ndarray:
    """Convolve with HST-like PSF"""
    if fwhm is None:
        fwhm = NoiseParams.PSF_FWHM_PX

    sigma = fwhm / 2.355  # Convert FWHM to sigma
    return gaussian_filter(image, sigma=sigma)


def add_continuum_residuals(image: np.ndarray) -> np.ndarray:
    """Add smooth residuals from imperfect continuum subtraction"""
    # Large scale smooth residuals
    residual = np.random.normal(0, 0.1, image.shape)
    residual = gaussian_filter(residual, sigma=np.random.uniform(40, 80))

    # Clumpy residuals
    n_clumps = np.random.randint(3, 8)
    for _ in range(n_clumps):
        cx = np.random.randint(0, image.shape[0])
        cy = np.random.randint(0, image.shape[1])
        sigma = np.random.uniform(20, 50)

        yy, xx = np.indices(image.shape)
        dist_sq = (xx - cx)**2 + (yy - cy)**2
        residual += np.random.uniform(-0.1, 0.1) * np.exp(-dist_sq / (2 * sigma**2))

    return image + residual.astype(np.float32)


def add_sky_background(image: np.ndarray) -> np.ndarray:
    """Add sky background with spatial variation"""
    sky = NoiseParams.SKY_LEVEL / NoiseParams.PHOTON_SCALE

    # Spatial variation (few percent)
    variation = np.random.normal(0, 0.02, image.shape)
    variation = gaussian_filter(variation, sigma=50)

    return image + sky * (1 + variation)


# ─────────────────────────────────────────────
# SAMPLE GENERATION
# ─────────────────────────────────────────────

def generate_cone_params(sample_idx: int) -> ConeParams:
    """Generate random cone parameters for a sample"""

    # Negative samples: 10% have no cone
    has_cone = np.random.rand() > 0.1

    if not has_cone:
        return ConeParams(
            r_offset=np.random.uniform(0, 30),
            phi_offset=np.random.uniform(0, 360),
            theta=45.0,  # Default values for negative samples
            phi=0.0,
            opening_angle=35.0,
            cone_length=80.0,
            cone_intensity=0.0,
            has_top_half=False,
            has_bottom_half=False,
            is_obscured=False,
            obscuration_frac=0.0,
            is_realistic=False,
            has_cone=False
        )

    # Determine viewing angle distribution
    # AGN are randomly oriented, so theta should be sin-weighted toward edge-on
    # But we also want some face-on examples for training diversity
    theta_choice = np.random.choice(['face', 'intermediate', 'edge'], p=[0.2, 0.5, 0.3])
    if theta_choice == 'face':
        theta = np.random.uniform(0, 25)
    elif theta_choice == 'intermediate':
        theta = np.random.uniform(25, 65)
    else:  # edge
        theta = np.random.uniform(65, 85)

    # Opening angle - mix realistic and unrealistic
    angle_type = np.random.choice(['narrow', 'realistic', 'wide'], p=[0.15, 0.7, 0.15])
    if angle_type == 'narrow':
        opening_angle = np.random.uniform(5, 15)
        is_realistic = False
    elif angle_type == 'realistic':
        opening_angle = np.random.uniform(20, 50)
        is_realistic = True
    else:  # wide
        opening_angle = np.random.uniform(50, 80)
        is_realistic = False

    # Random azimuthal orientation
    phi = np.random.uniform(0, 360)
    phi_offset = np.random.uniform(0, 360)

    # Cone length varies
    cone_length = np.random.uniform(60, 120)

    # Intensity - make some easy to detect, some hard
    difficulty = np.random.choice(['easy', 'medium', 'hard'], p=[0.3, 0.5, 0.2])
    if difficulty == 'easy':
        cone_intensity = np.random.uniform(1.5, 3.0)
    elif difficulty == 'medium':
        cone_intensity = np.random.uniform(0.8, 1.5)
    else:  # hard
        cone_intensity = np.random.uniform(0.4, 0.8)

    # Random obscuration patterns
    obscuration_pattern = np.random.choice(['none', 'partial', 'severe'], p=[0.6, 0.25, 0.15])
    is_obscured = obscuration_pattern != 'none'
    if obscuration_pattern == 'none':
        obscuration_frac = 0.0
    elif obscuration_pattern == 'partial':
        obscuration_frac = np.random.uniform(0.2, 0.5)
    else:
        obscuration_frac = np.random.uniform(0.5, 0.9)

    # Top/bottom half visibility
    # Randomly decide which halves are visible
    visibility = np.random.choice([
        'both',           # Both cones visible
        'top_only',       # Only approaching cone
        'bottom_only',    # Only receding cone
        'top_partial',    # Top cone partially obscured
        'bottom_partial'  # Bottom cone partially obscured
    ], p=[0.5, 0.15, 0.15, 0.1, 0.1])

    if visibility == 'both':
        has_top_half = True
        has_bottom_half = True
    elif visibility == 'top_only':
        has_top_half = True
        has_bottom_half = False
    elif visibility == 'bottom_only':
        has_top_half = False
        has_bottom_half = True
    elif visibility == 'top_partial':
        has_top_half = True
        has_bottom_half = True
        is_obscured = True
        obscuration_frac = max(obscuration_frac, 0.3)
    else:  # bottom_partial
        has_top_half = True
        has_bottom_half = True
        is_obscured = True
        obscuration_frac = max(obscuration_frac, 0.3)

    # Radial offset from center
    r_offset = np.random.uniform(0, 40)

    return ConeParams(
        r_offset=r_offset,
        phi_offset=phi_offset,
        theta=theta,
        phi=phi,
        opening_angle=opening_angle,
        cone_length=cone_length,
        cone_intensity=cone_intensity,
        has_top_half=has_top_half,
        has_bottom_half=has_bottom_half,
        is_obscured=is_obscured,
        obscuration_frac=obscuration_frac,
        is_realistic=is_realistic,
        has_cone=True
    )


def generate_sample(sample_idx: int) -> Tuple[np.ndarray, np.ndarray, ConeParams]:
    """Generate a single training sample"""

    # Get cone parameters
    params = generate_cone_params(sample_idx)

    # Generate cone emission and mask
    cone_emission, cone_mask = project_3d_cone_to_2d(IMG_SIZE, params)

    # Generate clumpy background
    background = generate_clumpy_background(IMG_SIZE)

    # Combine
    image = background + cone_emission

    # Add noise in realistic order
    # 1. Sky background
    image = add_sky_background(image)

    # 2. Convolve with PSF (before photon noise for realism)
    image = convolve_psf(image)

    # 3. Add photon noise
    image = add_photon_noise(image)

    # 4. Add read noise
    image = add_read_noise(image)

    # 5. Add continuum subtraction residuals
    image = add_continuum_residuals(image)

    # Normalize to 0-1 range for ML
    image = image - np.percentile(image, 1)
    image = image / (np.percentile(image, 99) + 1e-8)
    image = np.clip(image, 0, 1)

    return image, cone_mask, params


# ─────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("SYNTHETIC BICONE TRAINING DATA GENERATOR")
    print("=" * 60)
    print(f"\nGenerating {N_SAMPLES} samples ({IMG_SIZE}x{IMG_SIZE})")
    print(f"Train: {int(N_SAMPLES * TRAIN_SPLIT)}, Val: {int(N_SAMPLES * (1 - TRAIN_SPLIT))}")
    print()

    # Create directories
    for d in [TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Generate all samples
    metadata = {}
    n_train = int(N_SAMPLES * TRAIN_SPLIT)

    for i in range(N_SAMPLES):
        # Generate sample
        image, mask, params = generate_sample(i)

        # Determine split
        is_train = i < n_train
        if is_train:
            img_dir = TRAIN_IMG_DIR
            mask_dir = TRAIN_MASK_DIR
        else:
            img_dir = VAL_IMG_DIR
            mask_dir = VAL_MASK_DIR

        # Save
        sample_id = f"{i:05d}"
        np.save(img_dir / f"X_{sample_id}.npy", image.astype(np.float32))
        np.save(mask_dir / f"Y_{sample_id}.npy", mask.astype(np.float32))

        # Store metadata
        metadata[sample_id] = {
            **asdict(params),
            'split': 'train' if is_train else 'val',
            'sample_idx': i
        }

        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{N_SAMPLES} samples...")

    # Save metadata
    metadata_path = OUT_DIR / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Saved {N_SAMPLES} samples to {OUT_DIR}")
    print(f"✓ Metadata saved to {metadata_path}")

    # ─────────────────────────────────────────────
    # VISUALIZATION
    # ─────────────────────────────────────────────
    print("\nCreating diagnostics visualization...")

    # Select 3 interesting samples
    viz_indices = [0, n_train // 2, n_train + 100]  # First, middle train, val

    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    fig.suptitle("Synthetic Bicone Training Samples", fontsize=14, fontweight='bold')

    for row, idx in enumerate(viz_indices):
        sample_id = f"{idx:05d}"
        params = metadata[sample_id]

        # Load sample
        split_dir = 'train' if params['split'] == 'train' else 'val'
        img_path = OUT_DIR / split_dir / "images" / f"X_{sample_id}.npy"
        mask_path = OUT_DIR / split_dir / "masks" / f"Y_{sample_id}.npy"

        image = np.load(img_path)
        mask = np.load(mask_path)

        # Plot
        axes[row, 0].imshow(image, cmap='magma', vmin=0, vmax=1)
        axes[row, 0].set_title(f"Sample {sample_id} ({params['split']})")
        axes[row, 0].axis('off')

        axes[row, 1].imshow(mask, cmap='gray', vmin=0, vmax=1)
        axes[row, 1].set_title("Cone Mask")
        axes[row, 1].axis('off')

        # Overlay
        overlay = np.stack([image, image, image], axis=-1)
        overlay[..., 1] = np.clip(overlay[..., 1] + mask * 0.5, 0, 1)
        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title("Overlay")
        axes[row, 2].axis('off')

        # Parameters text
        if params['has_cone']:
            param_text = (
                f"θ={params['theta']:.1f}° (viewing)\n"
                f"φ={params['phi']:.1f}° (azimuth)\n"
                f"Opening={params['opening_angle']:.1f}° "
                f"({'real' if params['is_realistic'] else 'unreal'})\n"
                f"Intensity={params['cone_intensity']:.2f}\n"
                f"Visible: "
            )
            if params['has_top_half'] and params['has_bottom_half']:
                param_text += "Both halves"
            elif params['has_top_half']:
                param_text += "Top only"
            elif params['has_bottom_half']:
                param_text += "Bottom only"
            else:
                param_text += "None"

            if params['is_obscured']:
                param_text += f"\nObscured: {params['obscuration_frac']*100:.0f}%"
        else:
            param_text = "NEGATIVE SAMPLE\n(No cone)"

        axes[row, 3].text(0.1, 0.5, param_text, fontsize=9, family='monospace',
                         verticalalignment='center', transform=axes[row, 3].transAxes)
        axes[row, 3].set_xlim(0, 1)
        axes[row, 3].set_ylim(0, 1)
        axes[row, 3].axis('off')

    plt.tight_layout()
    viz_path = OUT_DIR / "diagnostics.png"
    plt.savefig(viz_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved diagnostics to {viz_path}")

    # ─────────────────────────────────────────────
    # DATASET STATISTICS
    # ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    n_negative = sum(1 for p in metadata.values() if not p['has_cone'])
    n_realistic = sum(1 for p in metadata.values() if p['is_realistic'] and p['has_cone'])
    n_unrealistic = sum(1 for p in metadata.values() if not p['is_realistic'] and p['has_cone'])
    n_obscured = sum(1 for p in metadata.values() if p['is_obscured'] and p['has_cone'])
    n_partial = sum(1 for p in metadata.values()
                    if p['has_cone'] and (not p['has_top_half'] or not p['has_bottom_half']))

    # Theta distribution
    thetas = [p['theta'] for p in metadata.values() if p['has_cone']]
    phis = [p['phi'] for p in metadata.values() if p['has_cone']]
    openings = [p['opening_angle'] for p in metadata.values() if p['has_cone']]

    print(f"\nSample Distribution:")
    print(f"  Total: {N_SAMPLES}")
    print(f"  Negative samples (no cone): {n_negative} ({100*n_negative/N_SAMPLES:.1f}%)")
    print(f"  Positive samples: {N_SAMPLES - n_negative}")
    print(f"    - Realistic opening angles (20-50°): {n_realistic}")
    print(f"    - Unrealistic opening angles: {n_unrealistic}")
    print(f"    - Partially visible cones: {n_partial}")
    print(f"    - Dust obscured: {n_obscured}")

    print(f"\nGeometry Statistics:")
    print(f"  Theta (viewing angle): {np.min(thetas):.1f}° - {np.max(thetas):.1f}° (mean: {np.mean(thetas):.1f}°)")
    print(f"  Phi (azimuthal): {np.min(phis):.1f}° - {np.max(phis):.1f}°")
    print(f"  Opening angles: {np.min(openings):.1f}° - {np.max(openings):.1f}° (mean: {np.mean(openings):.1f}°)")

    # Create distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Dataset Parameter Distributions", fontsize=14, fontweight='bold')

    # Theta distribution
    axes[0, 0].hist(thetas, bins=30, color='steelblue', edgecolor='white')
    axes[0, 0].axvline(25, color='red', linestyle='--', label='Face/Int boundary')
    axes[0, 0].axvline(65, color='red', linestyle='--', label='Int/Edge boundary')
    axes[0, 0].set_xlabel('Theta (viewing angle) [degrees]')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Viewing Angle Distribution\n(0°=face-on, 90°=edge-on)')
    axes[0, 0].legend()

    # Opening angle distribution
    axes[0, 1].hist(openings, bins=30, color='forestgreen', edgecolor='white')
    axes[0, 1].axvline(20, color='red', linestyle='--', label='Realistic lower bound')
    axes[0, 1].axvline(50, color='red', linestyle='--', label='Realistic upper bound')
    axes[0, 1].set_xlabel('Opening Angle [degrees]')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Opening Angle Distribution')
    axes[0, 1].legend()

    # Phi distribution
    axes[1, 0].hist(phis, bins=30, color='coral', edgecolor='white')
    axes[1, 0].set_xlabel('Phi (azimuthal) [degrees]')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Azimuthal Orientation Distribution')

    # Sample type pie chart
    labels = ['Negative\n(no cone)', 'Realistic\n(20-50°)', 'Unrealistic narrow\n(<20°)', 'Unrealistic wide\n(>50°)']
    n_narrow = sum(1 for p in metadata.values() if p['has_cone'] and p['opening_angle'] < 20)
    n_wide = sum(1 for p in metadata.values() if p['has_cone'] and p['opening_angle'] > 50)
    sizes = [n_negative, n_realistic, n_narrow, n_wide]
    colors = ['lightgray', 'forestgreen', 'gold', 'coral']
    axes[1, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[1, 1].set_title('Sample Type Distribution')

    plt.tight_layout()
    stats_path = OUT_DIR / "parameter_distributions.png"
    plt.savefig(stats_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved parameter distributions to {stats_path}")

    print("\n" + "=" * 60)
    print("GENERATION COMPLETE!")
    print("=" * 60)
    print(f"\nTo train your model, update your training script:")
    print(f'  DATASET_CONFIG["name_hint"] = "synthetic_bicone_v2"')
    print(f"\nDataset location: {OUT_DIR}")


if __name__ == "__main__":
    main()
