# Satellite Land Cover Classification using Transfer Learning

Fine-tuned ResNet on the [EuroSAT](https://github.com/phelber/EuroSAT) dataset to classify satellite imagery into 10 land cover categories. Built an end-to-end training pipeline in PyTorch including custom data loaders, augmentation, validation metrics, and error analysis.

---

## Overview

This project applies transfer learning to remote sensing imagery classification. A pre-trained ResNet model is adapted to the EuroSAT benchmark, which contains 27,000 labeled Sentinel-2 satellite image patches across 10 land use and land cover (LULC) classes.

## Dataset

**EuroSAT** — based on Sentinel-2 satellite imagery covering 13 spectral bands at 10m resolution.

| Class | Description |
|-------|-------------|
| AnnualCrop | Annual crop fields |
| Forest | Forest areas |
| HerbaceousVegetation | Grasslands and shrublands |
| Highway | Roads and highways |
| Industrial | Industrial zones |
| Pasture | Pasture land |
| PermanentCrop | Vineyards, orchards, etc. |
| Residential | Residential areas |
| River | Rivers and waterways |
| SeaLake | Sea and lakes |

- **Total images:** 27,000 (64×64 px patches)
- **Classes:** 10
- **Split:** Train / Validation / Test

## Model Architecture

- **Backbone:** ResNet (pre-trained on ImageNet)
- **Fine-tuning strategy:** Replace the final fully connected layer with a new classifier head matching the 10 EuroSAT classes; optionally unfreeze deeper layers for end-to-end fine-tuning
- **Framework:** PyTorch + torchvision

## Pipeline

```
EuroSAT Dataset
      │
      ▼
Custom DataLoader
  ├── Train/Val/Test split
  └── Data Augmentation (random flips, rotations, color jitter, normalization)
      │
      ▼
Pre-trained ResNet
  └── Fine-tuned classification head (10 classes)
      │
      ▼
Training Loop
  ├── Cross-entropy loss
  ├── SGD / Adam optimizer with learning rate scheduler
  └── Validation metrics per epoch (accuracy, loss)
      │
      ▼
Error Analysis
  ├── Confusion matrix
  ├── Per-class accuracy
  └── Misclassified sample visualization
```

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | — |
| Macro F1 Score | — |

*Results will be updated after training runs.*

## Requirements

```
torch
torchvision
numpy
pandas
matplotlib
scikit-learn
Pillow
tqdm
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Open the notebooks in Jupyter and run the cells in order:

```bash
jupyter notebook eurosat.ipynb
```

## Project Structure

```
├── eurosat.ipynb           # Main notebook: data loading, model training, evaluation, and error analysis
├── 1.ipynb                 # Exploratory / supplementary notebook
└── README.md
```

## References

- Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019). [EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification](https://doi.org/10.1109/JSTARS.2019.2918242). *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing.*
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385). *CVPR.*