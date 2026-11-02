# Source control and SSH copies

Git holds reusable source, tests, configurations, small instrument response
tables with provenance, scientific documentation, and authored LaTeX sources.
Generated research products remain local or are transferred separately.

New files under `data/`, `results/`, `saved_runs/`, and compiled documents under
`output/` are ignored. On 2026-11-02, 482 previously tracked generated images,
animations, and array/FITS links were removed from the current Git tree. Small
historical provenance records and source snapshots are retained. Original local
outputs were preserved and backed up; Git history was not rewritten. Old images
remain accessible in earlier commits, so a fresh clone still downloads that
history. This cleanup does not reduce past Git object storage.

## Current F187N work

Use `configs/cloudy/F187N_CONE_GEOMETRY.md` and the explicit corrected config
`configs/cloudy/ngc1068_f187n_soft_gas_projection_v2.json`. Older configurations
and the F502N/cube branch are retained for provenance. Presence in Git does not
mean an exploratory configuration is scientifically validated or ready for a
production training run.

## Before committing

```sh
git status --short
git diff --cached --stat
```

Stage named source paths. To undo accidental staging without deleting files:

```sh
git restore --staged -- results output/pdf/ionization_cone_science_traceability.pdf
```

Do not run `git clean` to remove the data. Git only transfers committed files.
Avoid committing credentials, local environments, arrays, checkpoints, movies,
animation frames, or repeated render outputs.

## Update the SSH clone

The GitHub integration branch is `master`. The local development branch was
named `main`; use an explicit branch in synchronization commands to avoid
assuming those names match.

**For the first update that removes old tracked outputs, back up the SSH copies
before pulling.** Fetching alone does not change the working files. From the
SSH repository directory:

```sh
git status --short
git fetch origin
git show origin/master:scripts/maintenance/backup_removed_outputs.py > /tmp/agn-backup-removed-outputs.py
python3 /tmp/agn-backup-removed-outputs.py --revision origin/master
git pull --ff-only origin master
```

The helper identifies outputs deleted by the pending update, creates a verified
archive under `.local_backups/`, and preserves symlinks as links without copying
their underlying datasets. Check its success message before pulling. It does
not delete or reset anything. The pull will remove the previously tracked
output paths from the SSH working tree; their contents remain in the backup.
Ignored datasets and other untracked outputs are not removed by this cleanup.
The helper reports no removals on subsequent updates that do not delete outputs.
The helper uses Python 3.6-compatible standard-library APIs so it can run on
older login nodes; this does not imply the scientific training stack supports
Python 3.6. If an earlier downloaded helper fails on `text=True`, fetch again
and repeat the `git show` command above before rerunning it.

This updates the currently checked-out branch only if a fast-forward is possible.
If Git reports local changes or diverging commits, preserve that work and resolve
the divergence before proceeding. Do not discard it with `reset --hard`.

## Inputs needed separately for the corrected example

The corrected renderer still needs these ignored local inputs:

```
data/cloudy/ngc1068_f187n_pilot_v1/grid.npz
data/cloudy/hii_f187n_calculated_v2/grid.npz
results/cloudy/f187n_reference/nucleus_cutout.npz
results/cloudy/f187n_psf_nrcb4/kernel.npz
```

The observed/synthetic comparison additionally needs the original JWST FITS
file recorded by the reference metadata, normally
`data/2d/jwst_ngc1068/jw03707-o128_t002_nircam_clear-f187n_i2d.fits`, and the
rendered `image_mask.npz`. Existing cached metadata can contain an absolute
`source_fits` path: verify that it resolves on the SSH host before running the
comparison. Regenerate the reference with the correct local source path when
necessary, retaining its original provenance.

Transfer required files with their relative directory structure and provenance
using your normal research-storage or SSH transfer workflow. A code push does
not transfer them, install Python packages, repair old spectral-grid convergence
limitations, or create a training dataset. Do not copy a local virtual environment
between machines; create/install the environment appropriate to the host.

For source checks after the environment is installed:

```sh
venv/bin/python -m unittest discover -s tests
```

The telluric integration-test class skips when its separately downloaded ESO
fixture is unavailable; other source tests can run without the research outputs.
