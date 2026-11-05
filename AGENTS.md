# Repository instructions for Codex

Within an already authorized research task, Codex may diagnose a failed run,
make a safe correction, and rerun it without asking for repeated confirmation.
Preserve failed inputs, raw outputs and logs; use a fresh output directory so
that the change and rerun remain reproducible. This clarification does not
authorize unrelated jobs, destructive changes, or external publication.

## Research scope and communication

Updated from the research discussion on 2026-10-29. These instructions preserve
the working direction; they do not certify a dataset or authorize new jobs.

- Develop physically motivated synthetic emission images for ionization-cone
  segmentation and projected-geometry recovery. Keep assumptions, computed
  quantities, observational constraints, and validation results distinguishable.
- Explain the physical meaning before introducing equations or implementation
  details. Avoid saying only "ready to train": specify pipeline demonstration,
  controlled pilot, or independently validated scientific application.
- The recent discussion concerns the F187N branch. The repository also has an
  F502N/[O III] branch and a historical spectral-cube trainer. Identify the
  actual configuration, band, input representation, grid, and label definition
  before changing or training a pipeline; do not silently mix these branches.
- A full torus simulation or 3D galaxy model is not required for the current
  2D pilot. Prioritize corrections that affect the intended measurement.

## Opening angle and label meaning

- Emission wavelength does not by itself set the physical cone opening.
  Circumnuclear obscuration and angular radiation escape shape illumination;
  the gas distribution and excitation determine where line emission occurs.
  Projection, extinction, filter response, PSF, and sensitivity affect the
  apparent extent measured in an image.
- Distinguish intrinsic illumination opening, projected geometric opening, and
  measured emission opening. Do not equate the brightest gas with the boundary,
  or a projected segmentation result with a direct torus-opening measurement.
- In the current generators, opening angle is prescribed or sampled. CLOUDY
  spectra and filter integration determine emission appearance; they do not
  derive that angle. Describe this as prescribed geometry with simulated emission.
- Record half versus full opening angle, axis/position-angle convention,
  inclination convention, projection prescription, and sampling range/source.
  A Gaussian angular width is not automatically a geometric half-opening angle.
  Audit the existing projection approximation before interpreting its parameter
  as an exact 3D cone angle.
- The proposed first pilot target is the generator's nominal projected cone
  geometry. Current soft gas emission may extend outside that hard label, while
  parts inside it may be faint or obscured. Test label recoverability explicitly.
  If targeting detectable emission instead, define and version that label rule
  separately rather than silently replacing the geometric target.
- Choosing angles as inputs is acceptable for a controlled experiment. Justify
  their ranges with relevant observations/literature, retaining tracer, aperture,
  projection, and angle conventions. Do not treat a chosen value as a measurement.

## Characterizing cone contrast and image complexity

The user wants a general description of how "messy" each synthetic image is
relative to its cone. Implement and assess separate descriptors before claiming
a universal scalar difficulty score. These are proposed diagnostics, not yet a
validated predictor of segmentation performance.

- Reserve "ionization parameter" for its physical meaning; use cone contrast,
  contamination, or image complexity for the descriptors below. Other light can
  itself be AGN-powered, so do not label all non-cone components "non-AGN".
- Define C as modeled cone-associated gas light and B as the explicitly listed
  other modeled components. Start with M = sum_A(B) / sum_A(C), using the same
  documented aperture A. Also report sum_A(C) / sum_A(C+B) and the fraction of
  aperture area where B > C. M is an overlapping-light ratio, not clumpiness.
- Compute these ratios from linear, noiseless component maps after bandpass
  integration and PSF convolution on a common pixel grid. Do not use log/asinh
  display arrays or add aggregate maps to their constituent components twice.
  Verify that the components plus any baseline reconstruct the clean image.
- Record aperture/radial limits, core exclusions, valid-pixel rules, component
  membership, band, PSF, and baseline treatment. Keep empirical baseline and
  noise separate from the initial component ratio. Use solid-angle weighting
  for surface-brightness maps if pixel areas vary. Flag zero cone signal as an
  undefined ratio; retain cone-free cases as a separate category.
- Add a spatial patchiness descriptor with an explicit smoothing scale and
  normalization, measured separately for cone and other emission. Its precise
  estimator still needs to be chosen and checked; smooth and knotty contaminants
  with equal M need not cause equal difficulty.
- Record projected cone length / PSF FWHM (both in the same angular units),
  pixel sampling, and a clearly defined cone signal-to-noise measure. Account
  for correlated noise when reporting an integrated significance.
- Synthetic component ratios are known from the generator. They are not an
  observational decomposition of a real galaxy or proof of physical realism.

As a discussion reference only, the existing
`results/cloudy/ngc1068_f187n_soft_gas_v1/image_mask.npz` gave M approximately
1.52 and a cone component fraction approximately 0.40 inside its nominal mask
intersected with 0.3 < radius < 3.7 arcsec, excluding baseline and noise. This was
a read-only calculation from saved component maps, not an implemented metric
pipeline, a population statistic, or a target calibration. Recompute if inputs
or definitions change.

## Angular size, physical size, and redshift

- Store pixels, arcseconds, and physical lengths with explicit units. Arcseconds
  describe apparent size; parsecs/kiloparsecs describe physical extent. Neither
  alone measures contamination or image difficulty.
- Convert transverse size using angular-diameter distance:
  L_pc = 4.8481368 * D_A_Mpc * theta_arcsec. Do not divide a size directly by z.
  Record the adopted cosmology or independently adopted distance and its source;
  for nearby objects, consider peculiar-velocity uncertainty in redshift distances.
- Use dimensionless ratios such as cone length / PSF FWHM and clump size / cone
  length for comparisons across scales. State whether lengths are projected.
- Changing redshift is not merely relabeling physical distances. A new rendered
  observation must consistently handle angular sampling, spectral shifts through
  the filter, flux/surface-brightness conventions, PSF, and noise. Current empirical
  image scaling does not establish an absolutely calibrated distance simulation.

## Training readiness and next work

An opening-angle question alone does not prohibit training. A pilot can test
recovery under stated assumptions; scientific use needs evidence beyond success
on synthetic images from the same generator.

1. Settle and audit the angle and label definitions for the selected band.
2. Document parameter ranges and their evidence, and check the actual spectra
   used in the rendered images. Inspect convergence and missing-line handling;
   new converged pilot grids do not repair older images unless incorporated.
3. Characterize contrast, spatial complexity, resolution, and noise. Preserve
   component maps and generation metadata so difficulty can be varied and measured.
4. Prepare a modest dataset with reproducible train/validation/test groups.
   Keep noise, filter, and other versions of one physical realization together.
   Shared calibration galaxies/grids limit what a split demonstrates; reserve
   independent physical conditions and observations for broader claims.
5. Run an authorized diagnostic pilot, comparing mask overlap and opening-angle
   error against a simple geometric baseline. Examine errors versus contrast,
   resolution, dust, clumpiness, inclination, and faint boundaries. Check for
   learned generator assumptions and poorly recoverable labels.

Do not turn this into an unlimited realism checklist. Fix demonstrated issues
material to the selected band and task, report limitations, and distinguish
pilot conclusions from real-galaxy validation. Do not hide model/data mismatch
by arbitrarily rescaling individual emission lines or changing observations.

## Repository evidence to consult

Geometry correction on 2026-11-01: new F187N physical-angle experiments should
explicitly select `circular_cone_envelope_v2` with inclination measured from the
sky plane. Use `configs/cloudy/F187N_CONE_GEOMETRY.md` and the versioned
`ngc1068_f187n_soft_gas_projection_v2.json` example. Missing projection fields
retain the legacy approximation for reproducibility, not for new physical-angle
interpretation. This corrects the angular envelope only; the emitting gas,
projected radial cutoff and velocity prescription remain simplified. Existing
F502N/cube and historical population configurations were not migrated.

Treat these as dated records; inspect newer configurations and outputs before
repeating a status claim. As of this discussion, the documented F187N population
was four exploratory scenarios, not a validated training population. Older image
grids had recorded iteration warnings; newer converged spectral pilots were
separate and did not automatically replace those image inputs.

- `configs/cloudy/F187N_PHYSICS_CHECKLIST.md`: included physics and known limits.
- `configs/cloudy/F187N_CONE_GEOMETRY.md`: versioned projection conventions,
  independent geometric tests, corrected rerender and remaining recovery errors.
- `configs/cloudy/F187N_REFERENCE_POPULATION.md`: illustrative population,
  calibration assumptions, and unresolved issues.
- `configs/cloudy/F187N_COMPLEXITY.md`: implemented component diagnostics,
  aperture/patchiness definitions, saved reports, and interpretation limits.
- `configs/cloudy/F187N_MODEL_AUDIT.md`: F187N-specific BCE/positive-only Dice,
  mixed-target checks and separate negative false-positive evaluation.
- `configs/cloudy/F187N_VARIATION_AND_DESCRIPTORS.md`: v3 independent foreground
  screen/texture/source controls and measured whole/cone-annulus descriptors.
  Category cuts are provisional descriptions, not a validated difficulty
  score. Dust coverage/optical depth do not imply dust mass or torus recovery.
- `docs/F187N_ARC_SETUP.md`: v2 paired-removal negatives and the frozen-source
  CPU generation/validation to GPU training launcher. Local checks do not
  establish ARC execution or real-galaxy accuracy; no cluster jobs were launched.
- `configs/cloudy/NGC1068_OBSERVED_CONE.md`: candidate observed HST emission
  geometry, sky-coordinate conventions, threshold sensitivity, and comparison
  to F187N; not a measured F187N component decomposition.
- `configs/cloudy/NGC1068_SPECTRAL_CONSTRAINTS.md`: spectral provenance,
  converged pilots, observational caveats, and remaining mismatches.
- `configs/cloudy/NARROWBAND_TRAINING.md`: 2D F502N input and rendering checks.
- `configs/cloudy/SPECTRAL_TRAINING.md`: historical spectral-cube experiment.

Use `venv/bin/python` for this repository's Python commands when available.
Preserve existing research artifacts and unrelated working-tree changes.
