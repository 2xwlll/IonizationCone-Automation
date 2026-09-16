import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
from scipy.ndimage import zoom, gaussian_filter

# =========================================================
# CONFIG
# =========================================================
NAME = "ngc1068_high_fidelity"
BASE_DIR = Path("data/2d") / NAME
GRID = 128
N_SAMPLES = 1000

# =========================================================
# CORE GENERATOR COMPONENTS
# =========================================================

def generate_turbulence(grid, strength=4.0):
    """Generates high-contrast log-normal gas density."""
    final_noise = np.zeros((grid, grid))
    # Multi-scale octaves for filamentary structure
    for i in range(6):
        freq = 2.0 ** i
        low_res = max(4, int(grid // (10 / freq)))
        noise = np.random.uniform(0, 1, (low_res, low_res))
        upscaled = zoom(noise, grid / low_res, order=1)[:grid, :grid]
        final_noise += (0.6 ** i) * upscaled
    
    # Log-normal creates the 'clumps' and 'voids'
    gas = np.exp(final_noise * strength)
    return gas / (gas.max() + 1e-8)

def get_shadowed_illumination(grid, opening_deg, gas_density):
    """
    Project light from center. Gas density acts as an 
    attenuation medium, creating shadowed lanes.
    """
    c = grid // 2
    y, x = np.mgrid[0:grid, 0:grid]
    r = np.sqrt((x-c)**2 + (y-c)**2)
    
    # 1. Base Geometry
    phi = np.random.uniform(0, 360)
    theta = np.random.uniform(45, 135) # Inclination
    p, t = np.radians(phi), np.radians(theta)
    axis_vec = np.array([np.sin(t)*np.cos(p), np.sin(t)*np.sin(p), np.cos(t)])
    
    v = np.stack([x-c, y-c, np.full_like(x, c)], axis=-1)
    v_unit = v / (np.linalg.norm(v, axis=-1)[..., None] + 1e-8)
    
    cosang = np.sum(v_unit * axis_vec, axis=-1)
    cone_mask = np.exp(-(np.arccos(np.clip(cosang, -1, 1)) / np.radians(opening_deg))**2)

    # 2. Shadowing (Simplified Ray Integration)
    # Gas blocks light. We use the radial gradient of the gas to simulate shadows.
    # Higher density near center = longer shadows.
    shadow_depth = gaussian_filter(gas_density, sigma=1.5)
    attenuation = np.exp(-shadow_depth * (r / (grid*0.2)))
    
    return cone_mask * attenuation / (r**0.5 + 1)

def make_sample():
    # 1. Stellar Continuum (The 'Fitted smooth continuum' panel)
    c = GRID // 2
    y, x = np.mgrid[0:GRID, 0:GRID]
    r = np.sqrt((x-c)**2 + (y-c)**2)
    continuum = np.exp(-(r / (GRID * 0.15))) * np.random.uniform(0.4, 0.7)

    # 2. The Gas Medium
    gas = generate_turbulence(GRID)

    # 3. The Ionization (Asymmetric Bicones with Shadows)
    opening = np.random.uniform(10, 20)
    # Generate two cones separately to allow for the 'asym' in your image
    side_a = get_shadowed_illumination(GRID, opening, gas)
    side_b = get_shadowed_illumination(GRID, opening, gas)
    
    # Random asymmetry multiplier
    asym_val = np.random.uniform(0.1, 0.5) 
    ionization_field = side_a + (side_b * asym_val)

    # 4. Final Image (Emission + Continuum)
    # The 'Emission' is gas being hit by light
    emission = (gas * ionization_field)
    
    # Final composite
    full_image = emission + (continuum * 0.3)
    
    # 5. Realistic Post-processing
    full_image = gaussian_filter(full_image, sigma=0.7) # PSF
    full_image += np.random.normal(0, 0.005, full_image.shape) # Shot noise
    
    # Use asinh scaling as in your panel
    full_image = np.arcsinh(full_image * 100) / np.arcsinh(100)

    # The Mask (What the UNet needs to find: the cone axis/region)
    mask = (ionization_field > 0.05).astype(np.float32)

    return np.clip(full_image, 0, 1).astype(np.float32), mask

# =========================================================
# RUNNER
# =========================================================

def run():
    if BASE_DIR.exists(): shutil.rmtree(BASE_DIR)
    for s in ["train", "val", "test"]:
        (BASE_DIR / s / "images").mkdir(parents=True)
        (BASE_DIR / s / "masks").mkdir(parents=True)

    print("Generating NGC 1068 style samples...")
    for i in range(N_SAMPLES):
        img, mask = make_sample()
        split = "train" if i < 800 else "val" if i < 900 else "test"
        np.save(BASE_DIR / split / "images" / f"{i:05d}.npy", img)
        np.save(BASE_DIR / split / "masks" / f"{i:05d}.npy", mask)
        
    # Visualization check
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img, cmap='magma')
    ax[0].set_title("Synthetic Data (asinh)")
    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title("Ground Truth Mask")
    plt.show()

if __name__ == "__main__":
    run()

if __name__ == "__main__":
    run()
