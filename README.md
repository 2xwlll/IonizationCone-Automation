# IonizationCone-Automization

Automated machine learning pipeline to identify and segment ionization cones in active galactic nuclei (AGN) using 2D [O III] emission maps.

---

## Background

Ionization cones are key signatures of AGN activity, traced via [O III] lines. They reveal the geometry of the narrow-line region and the extent of radiation escaping from the central engine.  
Manual identification is slow, subjective, and unscalable, motivating this automated approach.  

> “Imaging observation is more efficient with a large field of view, which means the method developed based on images will have wide applications.”  
> — Mentor insight

---

## Goals

- Create a reproducible pipeline for ionization cone identification and segmentation.
- Train ML models on synthetic and real AGN datasets.
- Validate models with known AGN cone geometries.
- Produce reliable predictions with visualizations and metrics.

---

## What It Does

- Trains a custom 2D UNet on 2D [O III] emission maps.
- Predicts binary masks of ionization cones from FITS or NumPy images.
- Visualizes predictions and evaluates performance (IoU, Dice, Precision, Recall).
- Designed for reproducibility and modular extension.

**Workflow:**

Cube (FITS) → Slice / Integrate → 2D [O III] Map → UNet → Cone Mask → Visualization & Metrics


---

## Supplementary / Legacy Projects

Some scripts are experimental or supplementary for learning purposes:  

- `_default_ionization_cone_calculation.py` — preliminary cone computation.
- `broadband_cones/` — miniature project for transforming cubes to images (integrated over slices for better results).  

Everything else is part of the main IonizationCone-Automization project and under active development.

---

## Project Structure
```Bash
IonizationCone-Automization/
├── config.py
├── data/
│ ├── cubes/
│ │ ├── raw/ # Raw FITS cubes from MAST or synthetic generators
│ │ ├── processed/ # Normalized cubes ready for analysis/training
│ │ ├── masks/ # Segmentation masks (real & synthetic)
│ │ ├── predicted/ # Model outputs from cube-based predictions
│ │ └── sorted/ # Organized copies of cubes for batch processing
│ ├── 2d/
│ │ ├── raw/ # 2D emission map slices derived from cubes
│ │ ├── processed/ # Normalized and resized 2D datasets
│ │ ├── masks/ # 2D segmentation masks
│ │ └── predict/ # Raw + predicted images from 2D model
│ ├── raw_sliced/ # Temporary storage for intermediate cube slices
│ ├── external/ # External datasets or imports
│ └── interim/ # Intermediate processing outputs
├── results/
│ ├── synthetic/ # Generated synthetic test outputs
│ ├── forward_vis/ # Forward pass & prediction visualizations
│ ├── evaluation_slides/ # Dice histograms, example comparisons
│ ├── cone_analysis_.png # Cone fitting or broadband projection plots
│ ├── loss_curve.png # Model training curve plots
│ ├── unet_best_2d.pth # Best trained 2D UNet model
│ └── test_cone.fits # Example FITS output
├── scripts/
│ ├── 2d/
│ │ ├── train.py # Train UNet on 2D [O III] maps
│ │ ├── predict.py # Run predictions on new 2D data
│ │ ├── process_synthetic.py # Generate & preprocess synthetic data
│ │ ├── process_real_fits.py # Real MAST FITS → 2D workflow
│ │ └── visualize_*.py # Visualization utilities for masks & predictions
│ ├── cubes/
│ │ ├── generate_noisy_cubes.py
│ │ ├── generate_synthetic_cone_masks.py
│ │ ├── train_cubes.py
│ │ └── evaluate_cubes.py
│ ├── broadband_cones/
│ │ ├── extract_2d_slice.py
│ │ └── test_pipeline.py
│ ├── organize/
│ │ ├── create_data_dirs.py
│ │ └── reorganize_src.py
│ ├── process_real_mast_data.py
│ └── sort_fits.py
├── src/
│ ├── broadband_cones/
│ │ ├── fit_cone.py
│ │ ├── project_to_image.py
│ │ └── pipeline.py
│ ├── machine_learning/
│ │ ├── datasets/
│ │ │ └── ionization_dataset.py
│ │ ├── models/
│ │ │ ├── model_2d.py
│ │ │ └── model_cube.py # Planned 3D model
│ │ ├── losses/
│ │ │ └── dice_loss.py
│ │ └── ionization/preprocess.py
│ ├── utils/
│ │ ├── normalize.py
│ │ ├── plot.py
│ │ └── metrics.py
│ └── paths.py
├── legacy/ # Older versions & archived scripts
├── requirements.txt
├── README.md
└── pyproject.toml
```

---

## Setup

```bash
git clone https://github.com/yourusername/IonizationCone-Automization.git
cd IonizationCone-Automization
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src

Usage

Train a 2D UNet:

PYTHONPATH=src python scripts/2d/train.py

Predict on new data:

PYTHONPATH=src python scripts/2d/predict.py --input <fits_file>

Visualize predictions:

PYTHONPATH=src python scripts/2d/visualize_masks.py --pred <prediction_file>

Always activate your virtual environment and ensure PYTHONPATH is set to src.

Acknowledgments

    Mentors: Dr. Chris Packham, John Schneider, Lulu Zhang

    Data sources: SDSS, MaNGA, JWST mock catalogs

    UTSA MAE Group for project support
