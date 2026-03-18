

import numpy as np
import matplotlib.pyplot as plt
import os

# ── 3D cone builder ───

def build_3d_cone(grid_size, r, phi_deg, theta_deg, opening_deg, clip_fraction=None):
    """
    Mark voxels inside a cone in 3D space.

    Parameters
    ----------
    grid_size      : int — cube is (grid_size, grid_size, grid_size)
    r              : max radius of cone in voxels
    phi_deg        : rotation in the image plane (degrees)
    theta_deg      : inclination toward viewer (degrees)
    opening_deg    : half-opening angle (degrees)
    clip_fraction  : if set (0.0–1.0), cone is clipped at this fraction of r
                     e.g. 0.5 = cone only extends to half its radius

    Returns
    -------
    volume : 3D numpy array, 1 inside cone 0 outside
    """
    phi         = np.radians(phi_deg)
    theta       = np.radians(theta_deg)
    opening_rad = np.radians(opening_deg)

    # cone axis unit vector
    ax = np.sin(phi) * np.cos(theta)
    ay = np.cos(phi) * np.cos(theta)
    az = np.sin(theta)

    cx = cy = cz = grid_size // 2

    zz, yy, xx = np.mgrid[0:grid_size, 0:grid_size, 0:grid_size]
    dx = xx - cx
    dy = yy - cy
    dz = zz - cz

    dist = np.sqrt(dx**2 + dy**2 + dz**2)

    dot       = dx*ax + dy*ay + dz*az
    cos_angle = np.where(dist > 0, dot / (dist + 1e-12), 1.0)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle     = np.arccos(cos_angle)

    effective_r = r * clip_fraction if clip_fraction is not None else r

    volume = ((angle <= opening_rad) & (dist <= effective_r)).astype(np.float32)
    return volume


def project_to_2d(volume):
    """Collapse 3D volume onto 2D by summing along z-axis. Normalize to max=1."""
    image = volume.sum(axis=0)
    if image.max() > 0:
        image = image / image.max()
    return image


# ── sample generator ──

def generate_sample(
    n_samples=100,
    grid_size=64,
    cone_probability=0.7,
    single_cone_probability=0.3,
    obscuration_probability=0.2,
    seed=42
):
    """"
    Generate a batch of 2D projected cone images from 3D voxel cones.

    Obscuration (dust torus blocking one cone):
      - only applies to bicones
      - obscuration_probability chance of occurring (default 20%)
      - 70% partial (random clip 20-80% of radius), 30% full cutoff

    Returns list of dicts:
        image      - soft 2D projection (0 to 1)
        mask       - binary label for UNet training
        has_cone   - bool
        is_bicone  - bool
        obscured   - None | 'partial' | 'full'
        params     - geometry dict
    """
    rng = np.random.default_rng(seed)
    samples = []

    for i in range(n_samples):
        has_cone  = rng.random() < cone_probability
        is_bicone = rng.random() >= single_cone_probability

        r           = rng.uniform(8, grid_size // 2 - 2)
        opening_deg = rng.uniform(15, 60)
        phi_deg     = rng.uniform(0, 360)
        theta_deg   = rng.uniform(0, 75)

        params = dict(
            r=round(r, 2),
            opening_deg=round(opening_deg, 2),
            phi_deg=round(phi_deg, 2),
            theta_deg=round(theta_deg, 2),
        )

        obscured = None

        if not has_cone:
            image = np.zeros((grid_size, grid_size), dtype=np.float32)
            mask  = np.zeros((grid_size, grid_size), dtype=np.float32)

        else:
            # build first cone
            vol = build_3d_cone(grid_size, r, phi_deg, theta_deg, opening_deg)

            if is_bicone:
                # check for obscuration on the opposite cone
                if rng.random() < obscuration_probability:
                    if rng.random() < 0.7:
                        # partial — clip at random depth between 20% and 80%
                        clip = rng.uniform(0.2, 0.8)
                        vol2 = build_3d_cone(grid_size, r, phi_deg + 180,
                                             theta_deg, opening_deg,
                                             clip_fraction=clip)
                        obscured = 'partial'
                    else:
                        # full — opposite cone completely missing
                        vol2     = np.zeros_like(vol)
                        obscured = 'full'
                else:
                    vol2 = build_3d_cone(grid_size, r, phi_deg + 180,
                                         theta_deg, opening_deg)

                vol = np.clip(vol + vol2, 0, 1)

            image = project_to_2d(vol)
            mask  = (image > 0).astype(np.float32)

        samples.append({
            'image'    : image,
            'mask'     : mask,
            'has_cone' : has_cone,
            'is_bicone': is_bicone,
            'obscured' : obscured,
            'params'   : params,
        })

    return samples


# ── visualization ───
def plot_samples(samples, n_show=9, save_path=None):
    n_show = min(n_show, len(samples))
    ncols  = 3
    nrows  = (n_show + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(9, nrows * 3))
    axes = axes.flatten()

    for i in range(n_show):
        s  = samples[i]
        im = axes[i].imshow(s['image'], cmap='inferno', origin='lower', vmin=0, vmax=1)
        p  = s['params']
        kind  = "bicone" if s['is_bicone'] else "single"
        obs   = f" [{s['obscured']}]" if s['obscured'] else ""
        title = f"{'CONE' if s['has_cone'] else 'BLANK'} | {kind}{obs}\n"
        if s['has_cone']:
            title += f"phi={p['phi_deg']:.0f} theta={p['theta_deg']:.0f} alpha={p['opening_deg']:.0f}"
        axes[i].set_title(title, fontsize=8)
        axes[i].axis('off')
        fig.colorbar(im, ax=axes[i], fraction=0.046)

    for j in range(n_show, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()


# ── run ───
if __name__ == "__main__":
    samples = generate_sample(n_samples=200, seed=0)

    n_cone    = sum(s['has_cone'] for s in samples)
    n_bicone  = sum(s['is_bicone'] and s['has_cone'] for s in samples)
    n_single  = sum(not s['is_bicone'] and s['has_cone'] for s in samples)
    n_partial = sum(s['obscured'] == 'partial' for s in samples)
    n_full    = sum(s['obscured'] == 'full' for s in samples)

    print(f"Total:   {len(samples)}")
    print(f"Cone:    {n_cone} | Bicone: {n_bicone} | Single: {n_single}")
    print(f"Obscured: partial={n_partial} | full={n_full}")

    os.makedirs("results", exist_ok=True)

    # show a mix including some obscured ones
    obscured_samples = [s for s in samples if s['obscured']]
    clean_samples    = [s for s in samples if s['has_cone'] and not s['obscured']]
    blank_samples    = [s for s in samples if not s['has_cone']]
    showcase = (clean_samples[:4] + obscured_samples[:3] + blank_samples[:2])[:9]

    plot_samples(showcase, n_show=9, save_path="results/sample_cones.png")
