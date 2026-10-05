# Synthetic cloud controls

The default generator now uses `synthetic_clumpy_v2.json`. The previous
`synthetic_expanded_v1.json` remains usable through `--config` and retains the
legacy wispy morphology, with the radial arc fix.

Our working visual definition of clumpiness is compact bright patches, darker
inter-cloud gaps, a range of sizes and brightnesses, and irregular groups of
clouds. The reference is the **emission panel** in
`results/visualizations/ngc1068_prediction.png`, not the model prediction or
classical mask. That panel uses an asinh stretch; the generated previews use
linear intensity. Apparent contrast cannot be calibrated directly between them.

These are phenomenological image controls, not inferred gas densities or physical
cloud sizes. The initial ranges are exploratory, not fitted to NGC 1068.

## Parameters in the `clouds` section

All pairs are per-sample uniform ranges, except `count` (integer, upper bound
excluded), `sigma_pixels` (per-cloud log-uniform size range), and `axis_ratio`
(per-cloud uniform range). Equal floating-point endpoints fix a control.

| Control | Meaning | Initial range |
| --- | --- | --- |
| `count` | Number of cloud components per lobe, before overlap/PSF | 18–45 |
| `sigma_pixels` | Intrinsic major-axis Gaussian sigma, before PSF | 1.2–4.5 px |
| `axis_ratio` | Minor/major sigma; near 1 gives compact round clouds | 0.65–1 |
| `cluster_fraction` | Fraction of centers moved around another sampled center | 0.3–0.65 |
| `cluster_scale_pixels` | Standard deviation of those 2D offsets | 3–8 px |
| `luminosity_scatter` | Log scatter of cloud peak amplitudes | 0.4–1 |
| `irregularity` | Log-amplitude of correlated cloud-shape modulation | 0.25–0.7 |
| `structure_scale_pixels` | Smoothing sigma of the modulation field | 0.8–2 px |
| `density_contrast` | Log contrast of the diffuse correlated field | 0.6–1.4 |
| `cloud_weight` | Relative compact-cloud emission weight | 1 |
| `diffuse_weight` | Relative diffuse emission weight; fills gaps | 0.12–0.35 |
| `filament_weight` | Relative weight of legacy radial wisps | 0.02–0.12 |
| `bridge_weight` | Faint clumpy gas along the cone axis, connecting inner clouds | 0.1–0.25 |

Each component is normalized to unit illumination-weighted mean before mixing.
Weights therefore specify relative contributions **before** the halo, dust,
PSF and final image normalization; they are not final observed flux fractions.
Increasing cloud count does not automatically increase total intrinsic emission.
More/larger clouds overlap more and can actually make the image smoother.
Clustering controls grouping, not a guaranteed minimum separation.

The existing `instrument.psf_sigma_pixels` controls whether knots survive blur.
For an unmodulated Gaussian cloud, sigma after Gaussian PSF convolution is
approximately `sqrt(cloud_sigma**2 + psf_sigma**2)` on each axis. Tiny clouds
cannot remain resolved under a broad PSF. Keep the instrument control separate
from intrinsic cloud size when tuning.

With `clouds` enabled, the old `smooth_cone_fraction`, `texture_strength`,
`texture_gamma`, `grain_count`, `grain_sigma_pixels` and `wisp_amplitude` do not
set the emission mixture. `wisp_count` still controls the filament component.
Legacy random draws are retained so matched seeds preserve geometry, dust and
noise. Old configurations without `clouds` keep the old emission path.

Sampled settings are stored under `clouds` in each `samples.json` record;
`metadata.json` stores the configuration. Per-lobe cloud seeds are derived from
the saved sample seed. Negative samples keep their existing generation path.

## Review and generate

From the repository root:

```sh
MPLBACKEND=Agg venv/bin/python scripts/2d/preview_cloud_controls.py
MPLBACKEND=Agg venv/bin/python scripts/2d/check_emission_boundaries.py
MPLBACKEND=Agg venv/bin/python scripts/2d/generate_realistic_emission.py --name clumpy_review --samples 30
```

Previews are `results/synthetic_ionization_cone/cloud_comparison.png` and
`cloud_controls.png`. The latter isolates one parameter at a time with a fixed
1-pixel PSF; count changes also change the random placement realization.

The binary target now follows a single connected geometric cone or bicone,
including the nucleus. The clumpy [O III] emission and dust shape the image;
they do not cut holes in this geometric target. The optional
`labels.mode: visible_emission` retains the earlier disconnected
brightness-threshold label for comparisons.
Review masks alongside images before replacing training data. Matching real
data next requires matched pixel scale, PSF, preprocessing, crop and display
stretch, then comparison of knot sizes, brightness contrasts and spatial
correlations across multiple observations. A single processed display cannot
determine those physical parameters uniquely.
