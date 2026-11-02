# F187N pilot v2: separate emitting populations

The v2 renderer assigns geometry to physical populations, rather than imposing
one cone shape on every wavelength. This remains an illustrative relative model,
not a fit to NGC 1068 or a validated segmentation training dataset.

## Populations

- **AGN cone gas:** the existing clumpy bicone, radially varying ionization and
  density, and outflow velocity. Its CLOUDY line spectrum and nebular continuum
  share the emitting gas geometry; different lines need not have identical
  surface brightness because their local emissivities and filter responses vary.
- **Host gas:** a separate clumpy projected exponential disk. Its position angle,
  axis ratio, radial falloff, ionization, density normalization and rotation are
  configurable under `host_geometry`. Its random fields and local CLOUDY model
  choices are separate from those of the cone. It is allowed both inside and
  outside the cone's projected footprint. The current spectrum is from the
  existing low-ionization AGN grid, **not a star-forming H II-region calculation**.
- **Stellar continuum:** the existing assumed band-integrated bulge profile.
- **Nuclear continuum:** the existing assumed unresolved band-integrated source.

A common wavelength-dependent foreground dust screen attenuates both gas
populations. This shared screen does not require their intrinsic emission to
have the same shape. Line attenuation uses each population's own Doppler shift.
All populations retain the existing F187N throughput, monochromatic PSF,
reference pixel scale, empirical amplitude matching and approximate noise model.
The stellar and nuclear components are band-integrated assumptions; this change
does not introduce stellar spectral templates.

The disk position angle (100 degrees east of north), axis ratio (0.55), gas
normalization and other v2 parameters are illustrative, not observational fits.
They intentionally remain explicit in `ngc1068_f187n_pilot_v2.json`.

## Outputs

Run from the repository root:

```bash
venv/bin/python scripts/cloudy/render_f187n_pilot.py \
  --config configs/cloudy/ngc1068_f187n_pilot_v2.json
```

The new output directory is `results/cloudy/ngc1068_f187n_pilot_v2`.
Existing v1 outputs are preserved. Explicitly selecting the v1 configuration
retains its original host-exclusion recipe for reproducibility; the renderer
refuses existing output directories.

`preview.png` compares the real observation, synthetic image, label and spectral
components. `components.png` shows cone gas, host gas, their sum and the target
mask. The three gas panels use a common display scale.

`image_mask.npz` saves six separate PSF-convolved gas maps: cone and host versions
of `pa_alpha`, `other_lines` and `nebular_continuum`. Stellar and nuclear continuum
maps are separate too. Sum only the eight names listed in `component_names`,
then add `galaxy_plus_background_baseline_mjysr`, to reconstruct `clean`.
The three `aggregate_names` are sums of their cone and host versions and must
not be added again. Those rendered maps have the empirical MJy/sr scale.

`intrinsic_cone_gas` and `intrinsic_host_gas` are unblurred, detector-grid averages
in relative photon-weighted units, before empirical normalization or baseline.
They allow direct checks of projected overlap and geometric independence.
The label is the original unblurred AGN cone geometry, even where host gas
projects inside it; brightness, PSF, continuum and noise do not change the label.
The training-pair files are an interface example; no U-Net was trained here.

## Scientific limits

This correction does not resolve the existing cached CLOUDY first-iteration
warnings, incomplete printed weak-line lists, uncalibrated emitting areas,
assumed continuum amplitudes, illustrative geometry, reference rather than
observation-specific wavefront, or approximate noise. The baseline includes
unseparated galaxy light. Dedicated converged star-forming H II-region spectra
and independent observational constraints are still needed before interpreting
this as a realistic component decomposition of NGC 1068.
