# DyNaTwin3D

DyNaTwin3D is a 3D digital twin framework for glioblastoma MRI analysis. The framework integrates multi-task 3D tumour sub-region segmentation, PINN-calibrated Fisher-KPP tumour growth modelling, uncertainty-quantified progression/survival estimation, and conditional adaptive post-processing for necrotic core recovery.

The source code supports experiments with multiple 3D U-Net style model variants, including ResUNet, attention U-Net, ASPP/deep-supervision models, and lightweight M3Plus-style variants. The project focuses especially on the necrotic core (NCR) bottleneck in BraTS-style glioblastoma segmentation and on downstream patient-specific tumour growth simulation.

## Repository Contents

- `dynatwin/`: core package for configuration, data loading, model definitions, losses, training, evaluation, statistics, visualization, and digital twin simulation.
- `postprocess_conditional.py`: conditional adaptive post-processing workflow for NCR/ET refinement.
- `predict_validation.py`: validation prediction utilities.
- `visualize_dt_pipeline.py`: digital twin pipeline visualization utilities.
- `visualize_postprocess.py`: post-processing visualization utilities.

This repository contains source code only. Datasets, trained model weights, generated outputs, figures, logs, and manuscript files are not included.

## Data Availability

The datasets used in this work are publicly available from the official BraTS release pages:

- BraTS 2020: https://www.med.upenn.edu/cbica/brats2020/data.html
- BraTS 2021: https://www.cancerimagingarchive.net/analysis-result/rsna-asnr-miccai-brats-2021/

Users should download the datasets directly from the official sources and comply with the corresponding data access, citation, and usage terms. No BraTS imaging data are redistributed in this repository.

## Citation

If you use this code, please cite the associated DyNaTwin3D manuscript:

**DyNaTwin3D: A Digital Twin Framework Integrating PINN-Calibrated Tumour Growth, Adaptive Necrotic Core Recovery, and Conditional Adaptive Post-Processing**

## Disclaimer

This code is provided for research purposes only and is not intended for clinical decision-making without independent validation and regulatory approval.
