#!/usr/bin/env python3
"""
Synthetic NGC 1068-like Emission Map
Mimics the continuum-subtracted [O III] residual / ionization cone
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from pathlib import Path

# ========================= CONFIG =========================
GRID = 256
PA_DEG = 132.8          # Cone axis from your fit
OPENING_DEG = 11.5
RADIUS = 95             # Length of each lobe in pixels

BASE_DIR = Path("data/2d/synthetic_ngc1068_emission")
BASE_DIR.mkdir(parents=True, exist_ok=True)

# ====================== GENERATE ======================
def generate_emission_map():
    c = GRID // 2
    y, x = np.mgrid[0:GRID, 0:GRID]
    dy = y - c
    dx = x - c
    dist = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)

    # Rotate to desired PA
    pa_rad = np.radians(PA_DEG - 90)   # -90 because matplotlib y increases downward
    theta_rot = theta - pa_rad

    # Narrow bicone mask
    half_open = np.radians(OPENING_DEG) / 2
    in_cone = np.abs(np.sin(theta_rot)) < np.sin(half_open)
    in_length = dist < RADIUS

    # Two lobes with different brightness
    lobe1 = in_cone & in_length & (theta_rot > 0)      # one direction
    lobe2 = in_cone & in_length & (theta_rot < 0)      # opposite

    # Base emission
    emission = np.zeros((GRID, GRID), dtype=np.float32)
    emission[lobe1] = 1.0
    emission[lobe2] = 0.85   # slightly dimmer opposite lobe

    # Radial brightness gradient (brighter near center)
    radial = np.exp(-dist / (RADIUS * 0.55))
    emission *= radial

    # Add central bright core
    core = np.exp(-dist / 12)
    emission += core * 2.2

    # Smooth
    emission = gaussian_filter(emission, sigma=2.8)

    # Add some faint extended emission / noise-like structure
    noise = np.random.normal(0, 0.08, (GRID, GRID))
    noise = gaussian_filter(noise, sigma=6)
    emission += noise * 0.25

    # Normalize
    emission = np.clip(emission, 0, None)
    emission /= emission.max()

    return emission


# ====================== PLOT & SAVE ======================
if __name__ == "__main__":
    emission = generate_emission_map()

    plt.figure(figsize=(10, 8))
    
    # Use asinh stretch like real astronomical images
    stretched = np.arcsinh(emission * 8)   # asinh stretch

    plt.imshow(stretched, cmap='inferno', origin='lower')
    
    # Add cone axis line
    c = GRID // 2
    length = RADIUS * 0.95
    dx = length * np.cos(np.radians(PA_DEG))
    dy = length * np.sin(np.radians(PA_DEG))
    
    plt.plot([c - dx, c + dx], [c - dy, c + dy], color='yellow', linewidth=2.5, alpha=0.9)
    plt.plot([c - dx*0.6, c + dx*0.6], [c - dy*0.6, c + dy*0.6], 
             color='cyan', linewidth=1.2, linestyle='--', alpha=0.7)

    # Nucleus marker
    plt.plot(c, c, 'o', color='black', markersize=18, markeredgecolor='white', markeredgewidth=2)

    plt.title("Synthetic Continuum-Subtracted Emission Map\n"
              f"NGC 1068-like | Cone PA = {PA_DEG}° | Opening ≈ {OPENING_DEG}°", 
              color='white', fontsize=14)
    
    plt.axis('off')
    plt.tight_layout()

    plt.savefig(BASE_DIR / "synthetic_emission.png", dpi=400, bbox_inches='tight', facecolor='black')
    np.save(BASE_DIR / "synthetic_emission.npy", emission)

    plt.show()

    print(f"Done! Image saved to: {BASE_DIR / 'synthetic_emission.png'}")
