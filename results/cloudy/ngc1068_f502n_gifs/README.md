# Current F502N diagnostics

- `f502n_image_mask_pairs.gif`: independent galaxies, final input, target overlay, and hard mask.
- `f502n_components.gif`: Cloudy cone + host gas, stellar/nuclear continuum, continuum dust screen, blurred image, noisy image, and target.
- `f502n_passband_integration.gif`: actual wavelength contributions integrated through F502N. The last frame reconstructs the final input. The animation does not show physical time.

Flux panels use a fixed asinh stretch; contribution panels use a separate fixed stretch because individual bins are much fainter than integrated images. Masks stay binary. Bright host gas outside the mask is included in the current generator. The main gas component is still restricted to the prescribed cone geometry. The displayed dust map is the continuum screen; gas attenuation is already baked into the cube.

Regenerate from the project root:

```bash
venv/bin/python scripts/cloudy/animate_f502n_pairs.py --generate
```
Omit `--generate` to rebuild animations from the saved samples.
