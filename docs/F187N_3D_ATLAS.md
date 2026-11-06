# F187N 3D descriptor atlas

Created 2026-11-06. `scripts/cloudy/plot_f187n_3d_atlas.py` makes read-only
scientific figures from an existing v3 grouped dataset. It generates six PNGs,
a six-page PDF, an HTML gallery, and `atlas.json` with source hashes, numerical
cross-checks, display conventions, and example-selection rules.

## What the plots show

1. **Competing light:** height is the local bounded fraction B/(C+B). The
   summed aperture ratio M=sum(B)/sum(C), cone light fraction and area B>C are
   annotated alongside each synthetic image. The average local fraction is
   not the summed ratio M.
2. **Spatial structure:** separate cone and other-light surfaces show
   log10(1+R_I), where R_I=abs(I-Gaussian_sigma(I))/mean_A(I). Mean_A(R_I)
   equals the saved patchiness P_I. Sigma is 0.2 arcsec; smoothing precedes
   aperture selection. This responds to gradients/edges as well as knots.
3. **Dust:** surfaces show pre-PSF area averages of tau and exp(-tau) in each
   output pixel. Mean screen depth, transmission and covered area are
   annotated. These are not a dust mass or emission-weighted flux loss.
4. **Visibility:** height is log10(1+C/pixel_sigma). The median is a per-pixel
   descriptor, not integrated significance. Projected extent/optical PSF core
   FWHM and optical PSF sampling are annotated as global scalars.
5. **Descriptor space:** one point per full positive realization, with axes
   M, other-light patchiness and mean transmission. Numbered thumbnails match
   the selected points. This is a scatter plot, not a fitted response function.
6. **Resolution space:** one point per realization, with projected extent/PSF,
   median cone/pixel_sigma and cone patchiness as separate coordinates.

The cyan region is the nominal cone mask intersected with reference-valid
0.3<r<3.3 arcsec pixels. Component quantities exclude empirical baseline/noise;
matched input images include both. Images use common asinh display limits,
with the top 0.5% of selected input values saturated for display only.
Coordinates use the saved sky matrix, with east left/north up in the image
panels. The central +/-3.6 arcsec is displayed.

Examples are selected by low/middle/high measured rank per page; multiple
physical controls differ between examples. The plots do not establish causal
single-factor effects or U-Net accuracy. Cone-free cases have undefined M and
extent and are excluded from these positive-cone figures.

## Local example artifact

The deliverable is `output/pdf/F187N_3D_Descriptor_Atlas.pdf`. Its companion
figures/gallery are in `results/cloudy/f187n_3d_descriptor_atlas_v3/`.
This local atlas contains **three independent v3 pipeline-check galaxies**,
not the 64-galaxy ARC population. Surfaces use every sixth pixel for display;
annotations and recomputation checks use full-resolution arrays. Pixel maps,
patchiness averages, dust measurements and source archives are checked against
the dataset metadata. Three numerical unit tests pass. All six PDF pages were
rendered with Poppler and visually inspected.

The first plotting failure and the later PC-crash interruption are preserved
in sibling `_v1` and `_v2` output directories, including source/input records.
The completed `_v3` export renders each figure once, then assembles raster PDF
pages at 135 dpi, reducing the cost of repeated 3D vector drawing. Completed
PNGs remain available if a later page is interrupted; the PDF's final name is
created only after successful document closure. These outputs are Git-ignored.

## Use the full ARC population

The plotter needs the dataset manifest, saved image/mask arrays and selected
`source_archive` component NPZs, including v3 foreground-screen maps. It does
not read a trained checkpoint, rerender galaxies or modify frozen jobs.
It uses the existing numpy/scipy/matplotlib/Pillow environment.

After pulling the source update on ARC, request a CPU shell (no explicit memory
flag for the site's current scheduler configuration), load Anaconda and run:

```bash
srun --partition=compute1 --nodes=1 --ntasks=1 --cpus-per-task=2 --time=00:30:00 --pty bash -l
module load anaconda3/2024.10-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /work/ppi246/agn/envs/f187n
cd ~/projects/IonizationCone-Automation
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python scripts/cloudy/plot_f187n_3d_atlas.py \
  --dataset /work/ppi246/agn/experiments/f187n_full_v3/dataset \
  --output /work/ppi246/agn/experiments/f187n_full_v3/review/3d_atlas_v1 \
  --stride 6 --examples 3 \
  --label 'ARC v3 synthetic population: 64 independent groups'
```

Use a fresh output directory. Both global scatter plots use all valid full
positive cases; each surface page shows representative low/middle/high examples.
Do not interpret three example images as the full population distribution.
