# F187N synthetic-image physics checklist

Audited 2026-10-23 against `ngc1068_f187n_soft_gas_v1.json` and its saved image,
CLOUDY grids, manifests and component maps. Scope: a 2D NGC 1068-like F187N pilot.

**A checked box means included in the current pipeline, not scientifically
validated or measured in NGC 1068.** The source/status column distinguishes
computed gas spectra, assumed components and observational reference data.
We have not imported a catalog of individual stars or other objects. Gas
populations emit several lines; multiple populations can emit the same line.

## Physical emitting populations

| Included | Population / source of light | Source/status in the current image | Still needed |
|---|---|---|---|
| [x] | AGN-photoionized gas, including central gas and extended cone-associated emission | Cached CLOUDY AGN grid; 81 models. Soft 2D illumination, cloud distribution and central envelope are prescribed. | Replace first-iteration spectra with converged calculations; constrain gas parameters, spatial scales and amplitudes. |
| [x] | Low-ionization host gas | Independent disk using the low-U portion of the same AGN grid. | This is not star formation; its gas spectra share the AGN convergence limitation. |
| [x] | Stellar-ionized H II-region gas | Dedicated CLOUDY C25 final pilot. Two models completed after three iterations; no cautions reported in their final summaries. Latest image uses model index 0 only. | Expand the stellar source, ionization, density and abundance range; constrain placement and luminosity. |
| [x] | H II-region ionizing stars, as an input radiation field | 35000 K blackbody in the two completed models; eight broader grid inputs are prepared separately. | Replace/compare with stellar atmosphere or population spectra. This blackbody is not a directly rendered young-stellar population. |
| [x] | Stellar bulge continuum | Assumed band-integrated 2D brightness profile, with foreground attenuation. | An actual stellar population spectrum filtered through F187N; ages/metallicities and observationally constrained normalization. |
| [x] | Compact nuclear continuum | Explicit assumed hot-dust Planck component plus power-law spectrum and obscuration, integrated through F187N. Unresolved source plus compact Gaussian surroundings. | Fit nuclear spectral shape and amplitude; distinguish stellar, scattered and dust light where data allow. This is not a CLOUDY torus calculation. |
| [x] | Nebular continuum associated with each gas population | CLOUDY line-free outward continuum, integrated through F187N separately for cone, low-U host and H II gas. | Converged AGN replacement and verification of completeness/normalization. No independent fitted dust/stellar decomposition is inferred. |
| [ ] | Shock-excited gas | No dedicated shock spectrum/grid in this image. | Assess its contribution with observations; add a justified component if material to the band. |
| [ ] | Direct young-cluster stellar continuum | Not a separate stellar population component. H II gas emission is included, but this does not supply all the young stars' near-IR light. | Suitable stellar spectra and relative normalization. |
| [ ] | Individually identified foreground/background objects | No object catalog, measured positions or individual source spectra imported. | Include only where they affect the chosen field or intended robustness tests; do not invent an observational source inventory. |

The compact nuclear model has an explicit spectral shape, but all its parameters
and amplitude remain assumptions. The geometric mask does not identify a unique
excitation source in the observed image: emission components can overlap.

## Emission lines already in use

- [x] **H I Pa-alpha:** in both AGN/host and dedicated H II grids. Rest vacuum
  wavelength 1.875612 micrometers; systemic observed wavelength 1.882726 at
  z=0.003793. AGN grid: finite entries 81/81. H II grid: 2/2; latest image uses
  one H II model. This is the explicitly saved `pa_alpha` map for each population.
- [x] **Two He I features:** rest vacuum 1.869050 and 1.870231 micrometers;
  observed 1.876140 and 1.877324. Present in both grids (81/81 AGN and 2/2 H II).
  Rendered within each population's `other_lines` map, not individually saved maps.
- [x] **He II features:** rest vacuum 1.864159, 1.874822 and 1.876212 micrometers;
  observed 1.871230, 1.881933 and 1.883329. Included from the AGN grid, not the
  current dedicated H II line list. The first two have 81/81 finite AGN entries;
  the last has 45/81. Missing entries are unknown/censored, not measured zeros.
- [x] **Fe II features, including CLOUDY's `Fe 2b` entry:** in the AGN inventory.
  The 1.889516-micrometer rest feature reaches observed 1.896683 in the red
  response wing. Three bluer Fe II entries have very weak F187N transmission at
  this redshift. Their inclusion does not establish significance or detections.
- [x] **Additional H I / He I / Si VI inventory entries:** passed through the
  response from the cached AGN grid. Several have extremely weak transmission
  here and should not be advertised as major F187N tracers. See the full table.
- [x] **H-beta and [O III] 5007 diagnostics:** calculated/saved in the dedicated
  H II runs (and available in the optical work). **They are not included in this
  nearby galaxy's F187N image**, because they lie outside this band.
- [ ] Complete high-precision F187N-relevant line inventory across the new AGN
  and stellar-ionized model ranges, with an explicit omitted-flux assessment.

### Full cached AGN line inventory

The current AGN table contains 15 near-IR features. The broader response table
includes tails outside the main passband; table membership is not equivalent to
strong transmission. All wavelengths below are converted to vacuum before
redshifting, while the original grid retains its air-wavelength identifiers.

“Relative filter response” is photon-averaged system throughput for a Gaussian
line with sigma=85 km/s, divided by the peak throughput. It uses zero bulk
velocity and z=0.003793. It is **not** a line's fraction of galaxy flux, an observed
line detection, or proof that the line contributes in the selected pixels.
Local velocities and emissivities change each line's actual contribution.

| CLOUDY label | Rest vacuum (µm) | Observed at z=0.003793 (µm) | Available AGN models | Relative filter response at v=0 |
|---|---:|---:|---:|---:|
| Fe 2 | 1.797601 | 1.804419 | 3/81 | 3.12e-06 |
| Fe 2 | 1.800512 | 1.807341 | 12/81 | 3.1e-06 |
| Fe 2 | 1.809894 | 1.816759 | 75/81 | 4.69e-06 |
| H 1 | 1.817906 | 1.824802 | 81/81 | 3.27e-05 |
| He 2 | 1.864159 | 1.871230 | 81/81 | 0.997 |
| He 1 | 1.869050 | 1.876140 | 81/81 | 0.989 |
| He 1 | 1.870231 | 1.877324 | 81/81 | 0.977 |
| He 2 | 1.874822 | 1.881933 | 81/81 | 0.769 |
| H 1 | 1.875612 | 1.882726 | 81/81 | 0.692 |
| He 2 | 1.876212 | 1.883329 | 45/81 | 0.63 |
| Fe 2b | 1.889516 | 1.896683 | 81/81 | 0.0657 |
| He 1 | 1.909461 | 1.916704 | 30/81 | 0.000121 |
| H 1 | 1.945091 | 1.952469 | 81/81 | 2.67e-06 |
| He 1 | 1.954844 | 1.962258 | 45/81 | 1.87e-06 |
| Si 6 | 1.963006 | 1.970451 | 27/81 | 2.46e-06 |

For example, Pa-alpha has relative response about 0.692 at zero velocity in this
calculation; that does not mean 69.2% of the image is Pa-alpha. The Fe 2b entry
has response about 0.0657, while the listed Si VI feature is approximately
2.46e-6. Assess significance using intrinsic line flux times response, not
wavelength alone. The two-model H II grid saves only the three in-band features
listed above plus its optical diagnostics; no complete faint-line claim is made.

The original AGN table has 330 missing line/model entries. They remain NaN in
the grid; the renderer currently fills them with zero for a lower-bound estimate.
The AGN output warning remains unresolved regardless of finite entry counts.

## Image formation already implemented

- [x] Cached measured F187N system-throughput table and photon-weighted integration.
- [x] Configurable redshift; latest version converts cached gas air lines to vacuum.
- [x] Doppler-shifted lines and explicit Gaussian line widths for the gas populations.
- [x] ISM grains within the gas calculations and an additional prescribed foreground
  attenuation screen. These are approximate dust treatments, not a fitted dust map.
- [x] Independent 2D emitting populations, allowing projected overlap.
- [x] Intrinsically soft AGN emission edges and a compact central gas component.
- [x] Sharp binary nominal cone mask, independent of brightness and PSF; unchanged
  from the previous hybrid label. Soft AGN emission can extend outside the mask.
- [x] NRCB4 STPSF kernel, reference pixel scale and detector-grid averaging.
- [x] Approximate correlated Gaussian noise using real ERR amplitudes.
- [x] Saved synthetic input, separate component maps, mask, configuration and checks.
- [x] Real NGC 1068 F187N cutout for comparison and empirical intensity scaling.
  Its SCI morphology is not pasted into the synthetic image.
- [ ] Observation-specific wavefront, detector position, sky rotation and drizzle matching.
- [ ] Physically calibrated fluxes or observationally constrained component amplitudes.
- [ ] Source-dependent detector noise, core-flag assessment and saturation treatment.
- [ ] Separate measured background from galaxy continuum: the current empirical
  baseline contains unseparated galaxy light and is not an empty-sky measurement.

## Work order before scientific training

1. [ ] Replace the cached AGN/low-U spectra with converged, traceable calculations
   and an explicit F187N line list; retain the current outputs as older pilots.
2. [ ] Expand the dedicated H II grid beyond the two completed models and introduce
   suitable stellar ionizing spectra. Eight prepared inputs are not eight results.
3. [ ] Supply direct stellar continuum spectra and constrain the nucleus, stellar
   light, gas and attenuation with complementary observations/spectroscopy.
4. [ ] Quantify which additional lines and excitation mechanisms materially affect
   F187N, including a shock-contribution assessment; do not add every conceivable
   source solely to fill a checklist.
5. [ ] Constrain plausible 2D spatial scales and component strengths; retain varied
   orientations/clump distributions, soft emission and explicitly defined masks.
   No 3D galaxy model is required by the current project scope.
6. [ ] Generate independent physical realizations with documented sampling ranges,
   then assign train/validation/test groups by underlying physical realization.
7. [ ] Run a pilot segmentation experiment and assess faint/obscured cone regions,
   host contamination and label recoverability. A pipeline score is not real-data
   validation; independent galaxies/observational labels are still needed.

## Evidence and audit files

- Active example: `results/cloudy/ngc1068_f187n_soft_gas_v1/image_mask.npz`.
- Active configuration: `configs/cloudy/ngc1068_f187n_soft_gas_v1.json`.
- Cached AGN provenance: `data/cloudy/ngc1068_f187n_pilot_v1/manifest.json`.
- Dedicated H II provenance: `data/cloudy/hii_f187n_calculated_v2/manifest.json`.
- Machine-readable line table: `results/cloudy/f187n_physics_inventory_v1/line_inventory.csv`.
- Audit parameters, response definition and source hashes:
  `results/cloudy/f187n_physics_inventory_v1/inventory.json`.

This is an implementation/provenance checklist. It is not a catalog of detections
in the real galaxy and does not certify that the synthetic image is physically
complete or that every listed line is appreciable in F187N.


## Subsequent spectral-reference work (2026-10-23)

Published nuclear and selected adjacent north/south line fluxes and equivalent
widths have now been imported with source/aperture metadata. Three new AGN
spectral-reference calculations completed after three iterations with no final
cautions reported. These are a separate fixed-density pilot; the image audited
above has NOT been switched to that grid and still uses the cached 81-model
spectra. Initial line constraints reveal substantial mismatches, and an Fe II
atomic-data sensitivity check is being documented rather than forcing an image
fit. The published Pa-alpha measurement is unusable because of an order gap and
telluric absorption, so it is not treated as an observed constraint. See
`NGC1068_SPECTRAL_CONSTRAINTS.md` and the source/audit directories it identifies.
