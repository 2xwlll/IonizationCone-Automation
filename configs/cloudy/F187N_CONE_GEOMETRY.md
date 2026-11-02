# Versioned F187N cone geometry

Implemented and checked 2026-11-01. The current corrected experiment is
`ngc1068_f187n_soft_gas_projection_v2.json`. This is the existing 2D soft-gas
example with corrected angular projection, not an observational fit to NGC 1068
and not a full 3D emissivity/velocity integration.

## Meaning and equations

The opening describes the directions occupied by the nominal cone. Available
gas and its excitation determine which parts emit. A geometric mask therefore
need not coincide with a brightness contour, even with correct projection.

New physical-angle configurations must explicitly set:

```json
{
  "projection": "circular_cone_envelope_v2",
  "inclination_convention": "from_sky_plane_positive_axis_receding",
  "position_angle_east_of_north_deg": 30.0,
  "opening_half_angle_deg": 35.0,
  "inclination_deg": 45.0,
  "radius_fraction": 0.85
}
```

- `opening_half_angle_deg` is intrinsic half-opening alpha, measured from the
  circular cone's axis to its wall. Full intrinsic opening is 2 alpha.
- `inclination_deg` is signed tilt i from the plane of the sky. Zero places
  the axis in the sky plane. Positive i makes the positive PA-axis lobe recede,
  consistent with the existing renderer's Doppler sign; negative i reverses it.
- Position angle is east of north. The opposite lobe is at PA + 180 degrees.
  Changing inclination sign leaves the projected angular envelope unchanged.
- For a sightline outside the cone, the exact orthographic angular envelope is
  `beta = asin(sin(alpha)/cos(i))`. Full projected opening is 2 beta.
- The supported domain is 0 < alpha < 90 deg and alpha + abs(i) < 90 deg.
  Sightlines on/inside the cone are rejected rather than clipped into a wedge.
  They require a different label representation. Nonfinite/ambiguous angles
  and unknown projection versions also fail explicitly.

`src/simple_utils/cone_geometry.py` supplies this calculation to both the
renderer and the comparison script. Resolved geometry and conventions are saved
in image metadata and checked by the comparison script. New metadata cannot
silently disagree with the geometry used by analysis.

The radius remains a **projected circular radial cutoff**, not a deprojected
cone length. The connected central mask pixels and majority rebinning are
retained. Soft illumination, cloud texture, foreground extinction and spectral
grid selection remain empirical prescriptions. The existing axial velocity
law scales with sin(i), but does not integrate the velocities of individual
cone walls. Correcting the angular envelope does not make the whole renderer
a 3D physical solution.

## Historical reproducibility

`legacy_tangent_v1` retains `atan(tan(alpha)/max(cos(i),0.2))` exactly. A missing
projection field is interpreted as this legacy version for old configurations,
and the renderer emits a warning. The original no-argument renderer default
still names a historical pilot; use the explicit command below for new work.
Existing population configurations and historical F502N/cube branches have
not been migrated or certified by this change. Some old sampled angles put the
sightline inside the cone and are not suitable for the corrected sector model.

Original soft-gas v1 config and image hashes are unchanged. Original analysis
outputs are retained; new analysis reads each input's own projection version.
The previous analysis code and renderer were snapshotted before editing under
`results/cloudy/cone_projection_correction_audit_v1/`.

## Controlled rerender and results

The new configuration keeps the old 35-degree half-angle, 45-degree inclination,
30-degree PA, seed 1069, spectral grids, gas prescription, PSF and noise recipe.
It changes the projection declaration and writes a fresh output. It does not
adopt the literature's 5-degree NGC 1068 tilt, because that would be a separate
physical-parameter change.

The existing empirical amplitude normalization is recomputed during rendering:
3.2541700e-8 before versus 2.7109884e-8 afterward. Thus this is a comparison of
the corrected recipe, not an experiment holding absolute component amplitudes
fixed. The wider envelope changes the gas image as well as the mask. Its
stochastic fields use the same seed. Old cached spectral limitations remain.

Within valid pixels at 0.7 < radius < 3.3 arcsec:

| Quantity | Legacy example | Corrected example |
| --- | ---: | ---: |
| Prescribed projected full opening | 89.44 deg | 108.42 deg |
| Nominal label vs analytic wedge IoU | 1.000 | 1.000 |
| Cone component flux inside nominal mask | 73.8% | 82.4% |
| Measured primary full opening from total image | 117 deg | 120 deg |
| Absolute axis error modulo 180 deg | 16.5 deg | 18.0 deg |
| Measured full width minus nominal width | +27.56 deg | +11.58 deg |
| Measured envelope vs nominal mask IoU | 0.415 | 0.400 |

The observed NGC 1068 result remains exactly unchanged: primary PA 30 deg,
full width 84 deg. The corrected synthetic cone-only diagnostic still fragments
under the radial-profile estimator. Improved flux enclosure and width error do
not demonstrate better overall recoverability; overlap and axis error did not
improve. Both the mask and rendered scene changed, so these are not scores
against a fixed benchmark label. This does not establish U-Net success or failure.

## Reproduce and verify

The following commands produced the preserved outputs; choose a new output
path in a copied config before rerunning, as overwriting is refused.

```sh
venv/bin/python scripts/cloudy/render_f187n_pilot.py \
  --config configs/cloudy/ngc1068_f187n_soft_gas_projection_v2.json
venv/bin/python scripts/cloudy/compare_real_synthetic_cones.py \
  --synthetic results/cloudy/ngc1068_f187n_soft_gas_projection_v2/image_mask.npz \
  --output results/cloudy/real_synthetic_cone_projection_v3
venv/bin/python -m unittest tests.test_cone_geometry tests.test_cone_comparison \
  tests.test_observed_cone tests.test_image_complexity tests.test_f187n \
  tests.test_gas_structure tests.test_f187n_spectral_export
```

Validation: 32 tests passed. The geometry test independently builds circular
cone directions in 3D, rotates them, and measures their projected extrema,
covering zero tilt, both signs, broad cones and near-boundary sightlines. Tests
also check legacy numerical compatibility and rejected configurations. The
fresh saved image is finite, its mask binary, and its components plus baseline
reconstruct the clean image. Source/input hashes, raw run logs, preservation
checks and before/after numbers are in
`results/cloudy/cone_projection_correction_audit_v1/`.

Reviewed comparison: `results/cloudy/real_synthetic_cone_projection_v3/cone_comparison.png`.
Numerical report: `results/cloudy/real_synthetic_cone_projection_v3/measurements.json`.

The angular mathematics is resolved for this stated model. The next separate
question is recoverability: compare smooth cone-only, clumpy cone-only, and
full images with fixed conventions and explicit labels before interpreting a
diagnostic training pilot as a physical measurement.
