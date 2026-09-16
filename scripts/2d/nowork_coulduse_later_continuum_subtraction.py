from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from astropy.visualization import simple_norm

# -----------------------------
# LOAD DATA
# -----------------------------
file = "data/2d/MAST_2026-04-14T0111/HST/hst_5754_01_wfpc2_pc_f502n_u2m301/hst_5754_01_wfpc2_pc_f502n_u2m301_drz.fits"

data = fits.open(file)[1].data.astype(float)

# -----------------------------
# CLEAN BASIC ISSUES
# -----------------------------
data = np.nan_to_num(data)

# -----------------------------
# ESTIMATE CONTINUUM (SMOOTH COMPONENT)
# -----------------------------
# big sigma = removes small-scale emission knots
continuum_est = gaussian_filter(data, sigma=10)

# -----------------------------
# SUBTRACT
# -----------------------------
sub = data - continuum_est

# clip noise floor
sub[sub < 0] = 0

# -----------------------------
# NORMALIZE FOR VISUAL + ML
# -----------------------------
sub /= np.max(sub)

# -----------------------------
# VISUALIZE
# -----------------------------
norm = simple_norm(sub, 'sqrt', percent=99.5)

plt.figure(figsize=(8, 8))
plt.imshow(sub, origin='lower', cmap='inferno', norm=norm)
plt.colorbar(label="Pseudo line emission")
plt.title("NGC 1068 - Smoothed Continuum Subtraction (Pseudo)")
plt.show()

# -----------------------------
# SAVE
# -----------------------------
fits.writeto("ngc1068_pseudo_contsub.fits", sub, overwrite=True)
