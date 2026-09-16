#!/usr/bin/env python3
"""
Synthetic Narrow Bicone Generator
Updated parameters:
- PA = 139.32° counter-clockwise
- θ = 33.86° toward us
- Opening angle = 12.134°
- ONLY bottom cone clipped by 10%
- Dimmer + fade on the near side (top)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter

# ========================= CONFIG =========================
GRID = 128
PA_DEG = 129.32        # Position angle CCW
THETA_DEG = 33.86      # Toward us angle
OPENING_DEG = 12.134   # Opening angle
RADIUS = GRID // 4     # Half of normal radius
BOTTOM_CLIP = 0.90     # 10% shorter — applied ONLY to bottom cone

BASE_DIR = Path("data/2d/synthetic_cone_exact")
BASE_DIR.mkdir(parents=True, exist_ok=True)

# ====================== GEOMETRY ======================
def get_axis(phi_deg, theta_deg):
    phi = np.radians(phi_deg)
    theta = np.radians(theta_deg)
    return np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ], dtype=np.float32)


def make_bicone():
    ax1 = get_axis(PA_DEG, THETA_DEG)   # Near cone  → Top
    ax2 = -ax1                          # Far cone   → Bottom

    c = GRID // 2
    z, y, x = np.mgrid[0:GRID, 0:GRID, 0:GRID]
    v = np.stack([x - c, y - c, z - c], axis=-1)
    dist = np.linalg.norm(v, axis=-1) + 1e-8
    v_unit = v / dist[..., None]

    # Inside cone condition
    cosang1 = np.sum(v_unit * ax1, axis=-1)
    cosang2 = np.sum(v_unit * ax2, axis=-1)
   
    # === ONLY BOTTOM CONE IS CLIPPED ===
    cone1 = (cosang1 >= np.cos(np.radians(OPENING_DEG))) & (dist <= RADIUS)           # Top = full length
    cone2 = (cosang2 >= np.cos(np.radians(OPENING_DEG))) & (dist <= RADIUS * BOTTOM_CLIP)  # Bottom = clipped

    vol = cone1 | cone2

    # Projection
    image = vol.max(axis=0).astype(np.float32)
   
    # Distance from center for fading
    dist_proj = np.sqrt((y - c)**2 + (x - c)**2).max(axis=0)
   
    # Near side (top lobe facing us)
    near_side = cosang1.max(axis=0) > 0.0

    # === Brightness with fade ===
    radial_fade = np.exp(-dist_proj / (RADIUS * 0.78))
    near_fade = np.where(near_side, 0.68, 1.0)           # dim top/near side
    length_fade = np.exp(-dist_proj / (RADIUS * 1.65))

    image = image * radial_fade * near_fade * length_fade

    # Smooth edges
    image = gaussian_filter(image, sigma=1.5)

    # Normalize
    image = image / (image.max() + 1e-8)

    # Binary mask
    mask = (image > 0.03).astype(np.float32)

    return image, mask


# ====================== GENERATE ======================
if __name__ == "__main__":
    print(f"Generating bicone: PA={PA_DEG:.3f}°, θ={THETA_DEG:.2f}° toward us, opening={OPENING_DEG:.3f}°")
    print(f"Bottom cone clipped by 10% (top cone full length)")

    image, mask = make_bicone()

    # Save
    np.save(BASE_DIR / "synthetic_cone.npy", image)
    np.save(BASE_DIR / "mask.npy", mask)

    # Plot
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='inferno', origin='lower')
    plt.title(f"Synthetic Bicone\nPA={PA_DEG:.3f}° CCW | θ={THETA_DEG:.2f}° toward us\n"
              f"Opening={OPENING_DEG:.3f}° | Bottom clipped 10% | Top full length")
    plt.colorbar(label='Normalized Intensity')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray', origin='lower')
    plt.title('Binary Mask')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(BASE_DIR / "synthetic_cone.png", dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Done! Files saved in: {BASE_DIR}")
