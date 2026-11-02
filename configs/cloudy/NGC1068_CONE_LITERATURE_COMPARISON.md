# NGC 1068 cone literature comparison and synthetic-target audit

Date: 2026-11-01. Scope: the existing 2D F187N soft-gas realization and the
measurements in `REAL_SYNTHETIC_CONE_COMPARISON.md`. No renderer, configuration,
training mask, or observational pixel was changed; no training was run.

Subsequent implementation: `F187N_CONE_GEOMETRY.md` documents the versioned
projection correction and fresh rerender. This literature-audit record retains
the original example's numbers below for provenance.

## Literature benchmark and what can be compared

[Das et al. (2006), Section 4 and Table 2](https://arxiv.org/pdf/astro-ph/0603803)
fit HST/STIS [O III] velocities with a hollow biconical outflow. Their quoted
parameters are axis PA 30 +/- 2 deg east of north, inner half-opening 20 +/- 2
deg, outer half-opening 40 +/- 2 deg, and inclination 5 +/- 2 deg out of the
sky plane, northeast nearer. Thus the intrinsic full outer opening is 80 deg.
These are model parameters, not F187N isophotal measurements. Their spatial
coverage extends beyond our 0.7--3.3 arcsec aperture.

[Crenshaw & Kraemer (2000), Section 3 and Table 1](https://arxiv.org/pdf/astro-ph/0002438)
also use an outer half-opening of 40 deg and inclination of 5 deg, with an
inner half-opening of 26 deg and an adopted axis PA of 15 deg. The multi-slit
Das study revised the axis to 30 deg. Literature estimates depend on the data
and model, so these are not independent measurements to average blindly.

For a circular cone tilted 5 deg from the sky plane, our analytic projection
of the 40-degree outer half-angle gives a full envelope of 80.37 deg. This
projection is our calculation, not an additional measured value in the paper.

| Quantity | Axis PA modulo 180 deg | Full angular width | Definition |
| --- | ---: | ---: | --- |
| Das benchmark projected by our circular-cone calculation | 30 | 80.37 | Outer geometric envelope of fitted 3D outflow |
| Our observed F187N candidate | 30 | 84 | 20%-of-peak angular excess, 0.7--3.3 arcsec |
| Our HST F502N-minus-scaled-F547M candidate | 27 | 120 | 20%-of-peak residual support, 0.7--3.7 arcsec |
| Current synthetic nominal mask | 30 | 89.44 | Prescribed 2D wedge using renderer approximation |
| Candidate recovered from synthetic total F187N | 46.5 | 117 | Same operator as observed F187N |

The F187N candidate differs from the projected benchmark by 0 deg in nominal
axis and +3.63 deg in full width (about 4.5%). This is numerical proximity,
not 4.5% measurement accuracy: the tracers, physical targets, apertures and
methods differ. Our 45--96 deg opening sensitivity range is not a statistical
confidence interval. The HST proxy is nominally 39.63 deg wider, and has a
72--156 deg method-sensitivity range; these ranges cannot validate a common
physical boundary merely by overlapping the benchmark.

[Raban et al. (2009), Section 5.3](https://academic.oup.com/mnras/article/394/3/1325/1065996)
explicitly discuss differences between visible HST morphology and kinematically
inferred geometry, considering partial illumination by obscuring clouds. This
supports comparing like definitions instead of treating every published
"opening angle" as one observable.

## A demonstrated projection limitation

The renderer sets its half-width using

```
beta_renderer = atan(tan(alpha) / max(cos(i), 0.2))
```

This defines the current 2D target exactly, but is not the exact projected
envelope of a circular 3D cone. If i is interpreted as tilt out of the sky plane
(consistent with the renderer's sin(i) axial velocity scaling), the exact
directional envelope for alpha + abs(i) < 90 deg is

```
beta_circular_cone = asin(sin(alpha) / cos(i))
```

This is derived for orthographic projection of a circular cone. It is not a
brightness boundary or a finite-aperture emission prediction. With the projected
axis along x, a unit cone generator has projected coordinates

```
x = cos(alpha)*cos(i) - sin(alpha)*sin(i)*cos(phi)
y = sin(alpha)*sin(phi)
```

Maximizing abs(atan2(y,x)) gives beta above. A numerical check with 200001
azimuth samples agrees to better than 1e-6 deg for both audited parameter sets.
Results, assumptions, and hashes are saved in
`results/cloudy/cone_literature_audit_v1/projection_check.json`.

For the current alpha=35 deg and i=45 deg, the renderer gives full width 89.44
deg, whereas the circular-cone envelope is 108.42 deg: a difference of 18.98
deg. This does not retroactively change the existing mask or explain away its
recovery error. It establishes that the input angle should not be interpreted
as an exact 3D angle. The synthetic inclination parameter also differs from
the literature's near-sky-plane NGC 1068 configuration. That is acceptable for
a generic controlled realization but not an observationally fitted NGC 1068.

For a future version, either parameterize the intended projected 2D opening
directly, or explicitly implement and verify a chosen 3D projection with clear
inclination conventions and compatible kinematics. Do not silently replace the
formula or overwrite old data/labels.

## How the current synthetic cone is determined

1. The configuration supplies axis, half-angle, inclination parameter and extent.
   The renderer constructs a sharp geometric wedge from them.
2. `soft_agn_emission_weight` supplies a separate empirical gas-light weight:
   a sigmoid angular transition, irregular boundary perturbation, off-axis floor,
   radial taper, random cloud texture and compact central gas. It is not a
   solution for angular radiation escape through a torus.
3. Prescribed local density, ionization parameter and other grid coordinates
   select CLOUDY spectra. Those spectra, velocities, extinction and F187N
   throughput determine the rendered light, with additional continua and gas
   populations followed by PSF convolution and noise.

Thus the accurate description is **prescribed geometry with simulated emission**.
The wavelengths determine which emission the filter transmits; they do not
generate the physical opening angle. The current geometry is not recovered
from CLOUDY, a torus calculation, or a fitted observed gas distribution.

Physically, directional escape of ionizing radiation and the available gas
jointly determine the ionized structure. Its visible footprint further depends
on excitation, obscuration, projection and sensitivity. Spatially resolved
line ratios and kinematics constrain different aspects; polarization and
infrared dust observations provide complementary information. For NGC 1068,
[Kraemer & Crenshaw (2000)](https://arxiv.org/abs/astro-ph/0007018) find complex
excitation, predominantly central photoionization with additional local effects.
One total F187N image cannot uniquely separate these mechanisms or its emitting
populations. In this renderer the band includes multiple lines and continua,
and both AGN gas and stellar-ionized gas can contribute hydrogen emission.

## What the failed recovery means for U-Net

We know and can compute the nominal synthetic target. The current angular-profile
extractor does not reliably recover it from this realization: synthetic axis
error 16.5 deg, full-width error +27.56 deg and IoU 0.415. Even the cone-only
diagnostic fragments into narrow sectors under this estimator. This flags both
the estimator's limitations and possible label ambiguity; it does not prove
that the image contains no usable information.

A U-Net could use spatial relationships discarded by radial-profile reduction.
However, success cannot be assumed. If distinct physical targets produce
observationally indistinguishable images, no image-only estimator can uniquely
identify them without prior assumptions. A network may infer a likely target
from training priors, including generator artifacts. That is not automatically
a measurement of this galaxy's physical boundary.

The next bounded diagnostic should:

1. Settle projected-geometry versus detectable-emission labels and resolve the
   angle convention/projection issue for a new version.
2. Compare smooth cone-only, clumpy cone-only, and full contaminated/noisy images
   from matched realizations, retaining the same known target and saved components.
   Vary angles and independent cloud realizations to avoid a fixed-angle shortcut.
3. Run a modest authorized pilot against a geometric baseline on held-out
   physical realizations, including unfamiliar orientations and contrasts. Report
   overlap and angle errors separately. Use independent observations for broader
   scientific claims after any NGC 1068-based tuning.

This is a recoverability experiment, not a requirement to perfect every aspect
of the galaxy simulation before any pilot training.
