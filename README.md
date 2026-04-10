### IonizationCone-Automization

Automated machine learning pipeline for detecting and segmenting ionization cones in active galactic nuclei (AGN) using 2D [O III] emission maps.

## Overview

Ionization cones are anisotropic emission structures associated with active galactic nuclei and trace the geometry and orientation of the narrow-line region. They provide insight into radiation escape, AGN feedback, and host galaxy interaction.

Manual identification of these structures is time-consuming and subjective. This project implements a machine learning pipeline to automate segmentation of ionization cones from imaging data.

## Objectives
Develop a reproducible pipeline for ionization cone segmentation in AGN emission maps
Train a 2D U-Net model using synthetic and real [O III] datasets
Evaluate model performance using standard segmentation metrics
Generate qualitative visualizations of predictions for analysis

## Methodology

# The pipeline consists of the following stages:

FITS / Synthetic Data
→ preprocessing (scripts/) (no current preprocessing needed right now)
→ dataset construction (src/machine_learning/datasets)
→ training (scripts/2d/train.py)
→ inference (scripts/2d/predict.py)
→ visualization (scripts/2d/visualize_*.py)

Synthetic data generation is used to provide controlled geometries for initial model validation, while real data supports generalization to observational regimes.

## Functionality

# The system supports:

Training of a 2D U-Net segmentation model on AGN emission maps
Binary segmentation of ionization cone regions
Inference on FITS or NumPy-based inputs
Visualization of predictions and training performance
Evaluation using metrics such as Dice coefficient and IoU
Results

# Model outputs include:

Segmentation masks of ionization cones
Training and validation loss curves
Qualitative prediction overlays comparing input maps and model outputs

# Representative outputs are stored in:

results/visualizations/
results/training_curves.png
results/sample_cones.png

## Repository Structure

```bash
IonizationCone-Automization/
├── scripts/        # Training, inference, preprocessing pipelines
├── src/            # Core models, datasets, losses, utilities
├── results/        # Model outputs and visualizations
├── data/           # Local datasets (not fully tracked)
├── archive/        # Legacy and experimental code
├── README.md
```

## Installation
Clone repository
git clone https://github.com/yourusername/IonizationCone-Automization.git
cd IonizationCone-Automization
Create environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
Set Python path
export PYTHONPATH=src
Training

Train a 2D U-Net model:

PYTHONPATH=src python scripts/2d/train.py
Inference

Run predictions on new data:

PYTHONPATH=src python scripts/2d/predict.py --input <fits_file>
Visualization

Generate prediction visualizations:

PYTHONPATH=src python scripts/2d/visualize_predictions.py
Data Notes
Synthetic and real datasets are supported
Local data directories are not fully tracked in the repository
Example outputs are provided in the results/ directory
Legacy Code

The archive/ directory contains experimental and deprecated implementations. These are not part of the primary training or inference pipeline but are retained for reference.

## Acknowledgments
Dr. Chris Packham
John Schneider
Lulu Zhang
UTSA MAE Group
Data sources: SDSS, MaNGA, JWST mock catalogs
Project Status

This project is actively developed. Structure and components may evolve as new datasets and experiments are incorporated.
