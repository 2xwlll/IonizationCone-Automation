# Train on wavelength cubes

**Historical experiment:** the active direction is now a single 2D narrowband
image. See [NARROWBAND_TRAINING.md](NARROWBAND_TRAINING.md) for the current
passband integration pipeline. The cube trainer below remains available as an
experimental reference.

The learning input is a full channel-first cube `(96,256,256)`, not an
integrated F502N image. The output is one projected geometric cone mask
`(1,256,256)`. A batch has shape `(batch,96,256,256)`.

The 96 samples cover observed 4965–5040 Angstrom at approximately 0.789 Angstrom
spacing. These are adjacent spectral bins containing [O III] 4959/5007 and
Cloudy nebular continuum, not separate filters or different chemical species.
The current generator also includes host [O III], gas attenuation, and spatial
blur. Stellar/nuclear continuum and observation noise added by the F502N
renderer are not automatically present in these source cubes. Extending those
components into calibrated spectral cubes remains a physics/instrument task.

`render_hst_f502n.py` and `animate_f502n_pairs.py` create photometric diagnostics.
Their 2D products are not used by the spectral training entry point below.
`scripts/2d/train.py` retains its legacy dataset settings. Use
`scripts/cloudy/train_oiii_cube.py` for the cube experiment.

## Prepare the existing fresh cubes

From the project root:

```bash
venv/bin/python scripts/cloudy/prepare_cube_dataset.py
```

This selects `cube_*.npz` from `data/cloudy/ngc1068_f502n_gif_v1`, preserving
the wavelength axis and all 96 channels. Source archives remain unchanged.
The current six samples split into four training, one validation, and one test
sample using a fixed seed; this is a pipeline demonstration, not a final
scientific dataset. Each generation seed is unique and duplicate cubes are
rejected. All splits currently use one shared Cloudy physics grid, so this
does not test generalization to unseen physical conditions or real galaxies.

Training tensors use `asinh(cube/softening)`, with one softening parameter across
the entire cube. No channel is independently scaled. Each sample's metadata
stores the shared scale and inverse transform, and the dataset stores the
wavelength axis. Line ratios should be computed from the source arrays, or
after undoing the transform; division of transformed values is not a physical
line ratio. The sources currently contain normalized generator flux rather
than absolutely calibrated scientific flux.

Default storage is float32. Optional float16 storage uses `--dtype float16`
and a new output directory; it does not require Cloudy to run again. The loader
casts stored tensors to float32 and CUDA training uses automatic mixed
precision when available.

## Verify the path, then train

```bash
venv/bin/python scripts/cloudy/train_oiii_cube.py --smoke-test
venv/bin/python scripts/cloudy/train_oiii_cube.py
```

The first command checks forward/backward computation and an optimizer update
on one real 96-channel sample using a small U-Net. It verifies that every input
channel has a gradient. This does not prove the model learns kinematics.
The normal trainer uses the architecture and settings in
`configs/cloudy/train_oiii_cube_v1.json`. Input channels must be 96, and a 2D
integrated sample is rejected. There is no fuzzy dataset selection. Model
checkpoints include the wavelength axis and normalization description.

The baseline is a spatial 2D U-Net with all wavelengths as input channels. Its
first convolution combines spectral information at each location while later
layers learn spatial structure. Wavelength position is encoded by fixed channel
order, not a coordinate provided to the network. An explicit spectral encoder
and channel-ablation experiments can be added after this baseline works.

## Generate more cubes without filter integration

Reuse the Cloudy grid and select a new seed/output for each galaxy:

```bash
venv/bin/python scripts/cloudy/generate_oiii_cube.py \
  --config configs/cloudy/ngc1068_oiii_v1.json \
  --grid data/cloudy/ngc1068_oiii_full_v1/grid.npz \
  --output data/cloudy/oiii_cube_batch_v2/cube_2000.npz \
  --seed 2000
```

After generating at least three independent samples, prepare a new dataset:

```bash
venv/bin/python scripts/cloudy/prepare_cube_dataset.py \
  --source-dir data/cloudy/oiii_cube_batch_v2 \
  --output-dir data/cloudy/oiii_spectral_training_v2
venv/bin/python scripts/cloudy/train_oiii_cube.py \
  --dataset data/cloudy/oiii_spectral_training_v2
```

Do not call the F502N renderer between cube generation and cube training.
