# 🛰️ Satellite Land Cover Classifier with Uncertainty Estimation

> Temporal-aware satellite image classification using transfer learning and 
> Monte Carlo Dropout uncertainty estimation on the EuroSAT dataset.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-96.63%25-green)
![Dataset](https://img.shields.io/badge/Dataset-EuroSAT-lightgrey)

---

## 🌍 Problem

Satellite imagery classification faces a fundamental limitation —
**single image snapshots cannot distinguish visually similar land cover types.**

Key failure modes discovered through error analysis:
- **River ↔ Highway** — 51 confusions, both are thin linear structures
- **AnnualCrop ↔ PermanentCrop** — 40 confusions, indistinguishable in one image
- **29% of errors** made with >80% confidence — silent failures with no warning signal

This project addresses the confidence problem directly with uncertainty estimation,
and motivates temporal change detection as the path to solving structural confusions.

---

## 🎯 Solution

An end-to-end satellite image classification pipeline with:
1. **Transfer learning** on ResNet18 pretrained on ImageNet
2. **Staged fine-tuning** — layer-by-layer unfreezing strategy
3. **MC Dropout uncertainty estimation** — flags unreliable predictions
4. **Human-in-the-loop system** — auto-approves confident predictions, escalates uncertain ones

---

## 📊 Results

### Classification Performance

| Configuration | Val Accuracy | Delta |
|---|---|---|
| Frozen backbone | 85.30% | baseline |
| Frozen + augmentation | 83.02% | -2.28% |
| Layer4 fine-tuned | 94.04% | +8.74% |
| Layer4 + Dropout(0.3) | 94.41% | +9.11% |
| Layer4 + LR Scheduler | 94.89% | +9.59% |
| Full fine-tuned | **96.63%** | +11.33% |

### Uncertainty Estimation Performance

| Metric | Value |
|---|---|
| Uncertainty ratio (wrong/correct) | **6.4x** |
| Auto-approved predictions | 4,824 / 5,400 (89.3%) |
| Auto-approved accuracy | **97.86%** |
| Errors caught by flagging | 64.73% |
| Predictions flagged for review | 576 (10.7%) |

### Per Class Performance (Best Model)

| Class | F1 Score | Error Rate |
|---|---|---|
| SeaLake | 0.99 | 1.25% |
| Forest | 0.98 | 0.86% |
| Residential | 0.97 | 1.49% |
| Industrial | 0.97 | 2.27% |
| Pasture | 0.96 | 5.30% |
| AnnualCrop | 0.95 | 6.81% |
| HerbaceousVeg | 0.95 | 4.58% |
| Highway | 0.94 | 6.48% |
| River | 0.95 | 4.23% |
| PermanentCrop | 0.92 | 8.50% |

---

## 🔑 Key Findings

1. **Fine-tuning is everything** — contributed 87% of total accuracy gain
2. **Layer4 matters most** — unfreezing just layer4 gave +8.74%, early layers added only +2.59%
3. **Augmentation needs fine-tuning** — augmentation alone hurt frozen backbone by 2.28%
4. **Uncertainty is a real signal** — wrong predictions are 6.4x more uncertain than correct ones
5. **Structural confusions need temporal data** — River/Highway and Crop confusions are 
   fundamentally unsolvable with single-image classification

---

## 🏗️ Architecture
Input (3 × 64 × 64)
↓
ResNet18 Backbone (pretrained ImageNet)
conv1 → layer1 → layer2 → layer3 → layer4
↓
AdaptiveAvgPool2d
↓
Dropout(0.3)
↓
Linear(512 → 10)
↓
Output (10 classes)

---

## 📁 Project Structure
satellite-classifier/
├── config.py          ← all hyperparameters
├── dataset.py         ← EuroSAT loading + transforms
├── model.py           ← ResNetWithDropout
├── train.py           ← training loop + early stopping
├── evaluate.py        ← metrics + error analysis
├── uncertainty.py     ← MC Dropout inference
├── utils.py           ← plotting + checkpointing
└── README.md

---

## 🚀 Quickstart

```bash
# Clone
git clone https://github.com/yourusername/satellite-classifier
cd satellite-classifier

# Install
pip install -r requirements.txt

# Train
python train.py

# Evaluate
python evaluate.py

# Uncertainty analysis
python uncertainty.py
```

---

## 🧪 Experiments

### Ablation Study
Fine-tuning strategy was tested systematically:
- Frozen backbone establishes baseline
- Layer4-only unfreezing captures domain-specific high-level features
- Full fine-tuning with low LR (1e-5) prevents catastrophic forgetting

### Optimizer Comparison
- Adam (1e-4) outperformed SGD (1e-3) by 1.44% in 5 epochs
- StepLR scheduler gave free +0.48% by reducing LR at epoch 6

### Regularization
- Dropout(0.3) improved accuracy by +0.52%
- Weight decay showed no consistent benefit — model not in overfitting regime

---

## ⚠️ Limitations

- 64×64 resolution limits fine-grained discrimination
- Single timestamp — cannot distinguish seasonal land cover changes
- 29% of errors made with >80% confidence before uncertainty estimation
- MC Dropout approximation — not true Bayesian uncertainty

---

## 🔭 Future Work

- **Temporal change detection** — multi-date Sentinel-2 sequences to resolve
  crop and linear structure confusions
- **Higher resolution** — EfficientNet backbone with 224×224 inputs
- **True Bayesian uncertainty** — Deep Ensembles or SNGP
- **Deployment** — Gradio demo for interactive inference with uncertainty display

---

## 📚 Dataset

**EuroSAT** — Sentinel-2 satellite imagery
- 27,000 labeled images across 10 land cover classes
- 64×64 pixels, 3 spectral bands (RGB)
- Balanced — 2,700 images per class

---

## 🛠️ Tech Stack

- PyTorch 2.1
- torchvision
- scikit-learn
- matplotlib
- tqdm
- numpy

---

