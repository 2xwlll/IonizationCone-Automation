#!/usr/bin/env python3
"""
Debug script: Check if angular wrap-around at pi/-pi is affecting cone detection.

In combined_realcone_pipeline.py, we bin theta values from -pi to pi.
If the cone axis is near pi (or -pi), the histogram splits the peak across
both edges, and np.argmax picks the wrong side.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Simulate a bipolar cone with true axis near pi (the problematic region)
true_axis = 3.0  # very close to pi (~3.14159)
n_points = 1000

# Generate random angles concentrated around true_axis and opposite
theta_vals = np.concatenate([
    np.random.normal(true_axis, 0.3, n_points//2),
    np.random.normal(true_axis - np.pi, 0.3, n_points//2)  # opposite side
])

# Wrap to [-pi, pi]
theta_vals = ((theta_vals + np.pi) % (2*np.pi)) - np.pi

# Method 1: Simple histogram (what combined_realcone_pipeline.py does)
bins = np.linspace(-np.pi, np.pi, 180)
centers = 0.5 * (bins[:-1] + bins[1:])
hist, _ = np.histogram(theta_vals, bins=bins)

peak_idx = np.argmax(hist)
peak_simple = centers[peak_idx]

print(f"True axis: {true_axis:.3f} rad ({np.degrees(true_axis):.1f} deg)")
print(f"Simple histogram peak: {peak_simple:.3f} rad ({np.degrees(peak_simple):.1f} deg)")

# Method 2: Circular histogram (correct way)
# Double the data and shift to handle wrap-around
theta_doubled = np.concatenate([theta_vals, theta_vals + 2*np.pi])
bins_doubled = np.linspace(-np.pi, 3*np.pi, 360)
hist_doubled, _ = np.histogram(theta_doubled, bins=bins_doubled)
centers_doubled = 0.5 * (bins_doubled[:-1] + bins_doubled[1:])

# Find peak in doubled histogram, map back to [-pi, pi]
peak_idx_doubled = np.argmax(hist_doubled)
peak_corrected = centers_doubled[peak_idx_doubled]
if peak_corrected > np.pi:
    peak_corrected -= 2*np.pi

print(f"Corrected peak: {peak_corrected:.3f} rad ({np.degrees(peak_corrected):.1f} deg)")

# Plot
fig, axes = plt.subplots(2, 1, figsize=(10, 6))

axes[0].bar(np.degrees(centers), hist, width=np.degrees(centers[1]-centers[0]),
            color='steelblue', edgecolor='black', alpha=0.7)
axes[0].axvline(np.degrees(true_axis), color='red', linestyle='--', linewidth=2, label=f'True axis: {np.degrees(true_axis):.1f}°')
axes[0].axvline(np.degrees(peak_simple), color='orange', linestyle='-', linewidth=2, label=f'Simple method: {np.degrees(peak_simple):.1f}°')
axes[0].set_xlabel('Angle (degrees)')
axes[0].set_ylabel('Count')
axes[0].set_title('Simple Histogram (WRAPS at ±180°) - Problematic!')
axes[0].legend()
axes[0].set_xlim(-180, 180)

# Show doubled histogram
axes[1].bar(np.degrees(centers_doubled), hist_doubled, width=np.degrees(centers_doubled[1]-centers_doubled[0]),
            color='green', edgecolor='black', alpha=0.5)
axes[1].axvline(np.degrees(true_axis), color='red', linestyle='--', linewidth=2, label=f'True axis: {np.degrees(true_axis):.1f}°')
axes[1].axvline(np.degrees(peak_corrected), color='lime', linestyle='-', linewidth=2, label=f'Corrected: {np.degrees(peak_corrected):.1f}°')
axes[1].set_xlabel('Angle (degrees)')
axes[1].set_ylabel('Count')
axes[1].set_title('Doubled Histogram (handles wrap-around correctly)')
axes[1].legend()
axes[1].set_xlim(-180, 540)

plt.tight_layout()
plt.savefig('angle_wrap_debug.png', dpi=150)
print(f"\nSaved plot: angle_wrap_debug.png")
print(f"\nError from simple method: {np.degrees(np.abs(peak_simple - true_axis)):.1f} degrees")
print(f"Error from corrected method: {np.degrees(np.abs(peak_corrected - true_axis)):.1f} degrees")
