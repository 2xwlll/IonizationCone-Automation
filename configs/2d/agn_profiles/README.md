# AGN profiles for synthetic [O III] data

Run the current generator from the repository root with an AGN profile:

```sh
MPLBACKEND=Agg venv/bin/python scripts/2d/generate_realistic_emission.py \
  --agn-profile configs/2d/agn_profiles/ngc1068.json \
  --name ngc1068_profile_review --samples 30
```

The profile reads the saved NGC 1068 `cone_params.json` at generation time.
Its pathway reference is `results/visualizations/ngc1068_prediction.png`.
Its measured opening half-angle (39°), inner radius (15 px), and outer radius
(110 px) feed the synthetic geometry. The saved mask S/N cutoff (2.5) is
retained for the optional `visible_emission` label mode.
Radii are divided by the 400-pixel source crop size before scaling to the
synthetic grid. The profile records the observed axis (132.5°), while sample
angles rotate through 360° as augmentation. The configured spreads are
exploratory and allow variety around this one object.

The default target is a **single connected projected cone or bicone** that
includes the nucleus. Its edge has smooth variation, while cloud knots and
dust affect the image within that region. Bicone lobes meet at the nucleus.
This NGC 1068 review profile begins with two intrinsic lobes. In 20% of
positive training draws, strong obscuration reduces the counter-lobe to
0–3% transmission and the target becomes a single cone. The other draws
have joined bicone targets. The 20% is a **training coverage choice**, not
an estimate of how often this occurs in galaxies. Change
`mixture.obscured_counter_fraction` or pass
`--obscured-counter-frac` to test another training mix.
The label is a hard geometric sector defined only by the sampled nucleus,
axis, opening angle, radial extent, and visible-lobe state. Gas emission, dust,
clouds, noise, PSF, and the reference prediction never modify its pixels. The
reference UNet prediction itself has detached regions; the geometric target
deliberately stays connected through the nucleus. `labels.mode:
visible_emission` retains the earlier brightness-threshold mask only for
comparisons; that mode can be disconnected.

The profile's `overrides` increase the number of smaller clouds, give some
knots higher brightness, add diffuse and axial bridge emission, strengthen the
central point source, and favor one dominant lobe. `emission.cone_boost`,
`obscuration.dust_depth`, and `instrument.nucleus_amplitude` are sampled per
image and recorded in `samples.json`. Change their ranges in the profile to
create brighter, fainter, dustier, or clearer runs. The image is normalized
per sample, so these change **relative contrast**, not an absolute flux scale.
These are phenomenological controls, not measured gas or dust parameters.
Every sample saves exactly one cold binary target in `masks/`. There is no
soft boundary target and no emission-guided boundary variation.
The saved `dataset_preview.png` uses an asinh display stretch to reveal faint
connected emission. The `.npy` images retain their original generated values.

Cloud shapes, dust, host light, PSF, noise, and flux distributions still use
exploratory settings from the base config. Matching those requires comparing
images at a common pixel scale, PSF, preprocessing, crop, and intensity
stretch, ideally across several AGN. This first profile anchors geometry; it
does not claim physical gas parameters or validated survey realism.

`metadata.json` embeds the effective config and profile measurements. Each
`samples.json` entry records drawn parameters, `intrinsic_bicone`,
`counter_lobe_obscured`, `counter_transmission`, and foreground fraction.
`metadata.json` also counts visible bicones and hidden counter-cones. New
AGN profiles can point to another compatible geometry fit and specify their
crop size, geometry spreads, and base-config control overrides.
