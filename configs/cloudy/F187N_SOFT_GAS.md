# Soft intrinsic AGN emission with a sharp cone label

The earlier hybrid prototype multiplied all AGN gas emission by the binary
cone mask. This imposed a sharp angular boundary and a circular outer edge
before the telescope PSF. The new optional `gas_structure` separates those
emission assumptions from the geometric target.

The new illustration uses a logistic angular illumination transition of
10 degrees, a boundary perturbation up to 7 degrees following the existing
large-scale cloud field, a 2.5% off-axis emissivity floor, and a smooth radial
fade of width 0.10 times the image half-width. It retains the clumpy gas and
local CLOUDY model choices. An additional compact central gas envelope has
sigma=0.23 arcsec, allowing central ionized emission in all directions.
These are phenomenological emissivity weights, not fitted escape fractions,
3D radiative transfer, or newly computed angular CLOUDY solutions. The floor
is an emissivity prescription, not a statement that 2.5% of ionizing photons
escape. Gas and continuum spectral filtering are otherwise unchanged.

The nucleus retains its explicit hot-dust/power-law spectrum. Its spatial
profile now combines 65% unresolved flux and 35% compact Gaussian surroundings
with sigma=0.16 arcsec. Both are assigned the same assumed continuum color for
this first illustration; this is not an inferred torus or scattering model.
The spatial profile sums to one before the PSF. The illustrative nominal nuclear
normalization is increased from 20 to 35 relative to gas p99, so it is not
claimed that the brighter center follows from morphology alone. Neither that
amplitude nor the extended light fraction is an observational measurement.

The image is not globally blurred to hide a boundary. The intrinsic gas
emissivity itself is continuous before the existing telescope convolution.
The old binary cone computation and detector-grid label are unchanged. The
mask is **bitwise identical** to the previous hybrid image's mask, remains 0/1,
and is never convolved. It represents the nominal projected geometry, not the
full support of every AGN photon or a surface-brightness contour. Some AGN
emission outside the label is intentional; the boundary perturbations do not
redefine the label. This labeling convention needs assessment before training
and does not imply perfect recoverability from a single band.

Dedicated H II gas remains a separate population and the old low-U host gas
remains separate. The existing CLOUDY spectra, vacuum correction, F187N response,
PSF, pixel scale and approximate noise are retained. The old AGN convergence
warnings, incomplete faint-line inventories, assumed amplitudes and lack of
observational fitting remain unresolved.

Reproduce with a fresh output directory in a copied configuration:

```bash
venv/bin/python scripts/cloudy/render_f187n_pilot.py \
  --config configs/cloudy/ngc1068_f187n_soft_gas_v1.json
venv/bin/python -m unittest discover -s tests -p 'test_gas_structure.py'
```

The renderer refuses existing outputs. Configurations without `gas_structure`
and `nuclear_structure` preserve the previous rendering behavior. The new
version is `results/cloudy/ngc1068_f187n_soft_gas_v1`. `before_after.png` displays
both images, an overlay, both intrinsic AGN gas maps, and the unchanged mask.
The gas-map pair uses a common relative scale; the total images use the same
empirical MJy/sr display scale. Each image still has its own old-style percentile
normalization, not a joint physical luminosity fit.

Three focused numerical checks verify smooth intrinsic emission at the nominal
angular and radial boundaries, directional contrast with nonzero extended
emission, and conservation of nuclear source flux as its extent changes.
The saved-array checks verify the unchanged mask, finite image, component
reconstruction and exact training-mask match. These checks establish the
implementation behavior, not astrophysical validity.
