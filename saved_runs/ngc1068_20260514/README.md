# Saved NGC 1068 results

Captured on 2026-09-10 from the local results dated 2026-05-14. Both candidate
"ok" results are preserved because the intended preview was not identified.

| Result | Axis (degrees) | Full opening (degrees) | Preview | Saved measurements |
| --- | --- | --- | --- | --- |
| Classical polar extraction | 132.5 | 78.0 | [Image](two_component_result.png) / [polar diagnostic](polar_diagnostic.png) | [Parameters](cone_params.json) |
| U-Net prediction | 142.5 | 80.0 | [Image](prediction.png) | [Parameters](prediction_params.json) |

Angles above are values reported by the scripts, not independently verified sky
position angles.

## Settings and provenance

[settings.json](settings.json) captures the current scripts' top-level settings.
The [source](source/) folder preserves their full contents, including parameters
embedded inside processing steps. These are reference copies, not standalone
runnable entry points. Saved result JSON files contain the measured output values.
Current source settings are not proof of the exact historical execution.

The classical settings include radii 15–110 pixels, 40 radial bins, 360 angular
bins, angular smoothing of 7 bins, edge threshold 0.20, mask S/N cutoff 2.5,
a 400-pixel crop, and continuum PSF smoothing sigma 0.5 pixels.
The saved continuum scale factor is 0.030811479315161705.
Prediction settings include a 128-pixel grid, threshold 0.5, and nucleus crop
half-width 110 pixels.

[manifest.json](manifest.json) records original repository-relative paths,
SHA-256 hashes, file sizes, and the Git revision at capture. The working tree
contained uncommitted changes, so the revision alone does not describe these files.

The original prediction did not save its checkpoint identity. Both available
`best.pth` files are linked as candidates; current latest-run selection would
choose `run_20260513_064046`. This association is unverified. The copied training
source and [dataset metadata](training_dataset_metadata.json) are supporting
context, not a verified historical training configuration.

## Data links without recursive copies

[data_links](data_links/) contains relative symlinks to individual local files:
the F502N and F547M input FITS files, processed maps and masks, and candidate
model weights. No directory is copied or linked, and no link points back into
this snapshot. Model links end in `.pth.link` so Git's `*.pth` ignore rule does
not hide them.

These links work while the repository retains its current data layout. The data
and model bytes are not backed up here or included by committing the links.
The manifest preserves their identities if they later move or change. To copy
this snapshot elsewhere without copying data, preserve symlinks (for example,
`cp -a`) and avoid dereferencing them (`cp -L`).

Treat this folder as a fixed snapshot; save future results in sibling folders.
Do not recursively copy the project or its `saved_runs` directory into a snapshot.
