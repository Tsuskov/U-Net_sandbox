# U-Net: Convolutional Networks for Biomedical Image Segmentation

A complete PyTorch implementation of the U-Net architecture from:
**"U-Net: Convolutional Networks for Biomedical Image Segmentation"** (Ronneberger et al., 2015)

## Overview

This repository provides:
- **Complete U-Net architecture** matching paper specifications
- **Training infrastructure** with weighted loss for cell separation
- **Data augmentation** including elastic deformations
- **Evaluation metrics** (IoU, Dice, Rand error, warping error)
- **Experiment scripts** for reproducing paper results
- **Inference utilities** with overlap-tile strategy for large images

## Key Features

✓ **31M parameters** - Matches the architecture in the paper
✓ **Unpadded convolutions** - Preserves valid region only (572×572 → 388×388)
✓ **Skip connections** - Combines multi-scale features
✓ **Weighted loss** - Emphasizes cell boundaries
✓ **Elastic deformation** - Key augmentation for small datasets
✓ **Overlap-tile strategy** - Seamless segmentation of large images
✓ **High momentum SGD** - Training with momentum=0.99

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Verify Installation

```bash
python3 verify.py
```

### 2. Run Demo

```bash
python3 experiments/demo.py
```

Trains a model on synthetic data to verify the pipeline works.

### 3. Run Experiments

```bash
# EM Segmentation
python3 experiments/em_segmentation.py --epochs 100

# Cell Tracking - PhC-U373
python3 experiments/cell_tracking.py --dataset phc_u373 --epochs 100

# Cell Tracking - DIC-HeLa
python3 experiments/cell_tracking.py --dataset dic_hela --epochs 100
```

## Architecture Details

**Contracting Path (Encoder):**
- 4 downsampling levels
- Each: 2×(3×3 conv + ReLU) + 2×2 max pool
- Channels: 64 → 128 → 256 → 512

**Bottleneck:**
- 2×(3×3 conv + ReLU) at 1024 channels

**Expanding Path (Decoder):**
- 4 upsampling levels
- Each: up-conv (2×2) + crop & concat + 2×(3×3 conv + ReLU)
- Skip connections from contracting path

**Output:**
- 1×1 convolution to map to classes
- 388×388 from 572×572 (valid region only)

## Training Details

**Hyperparameters (from paper):**
- Optimizer: SGD with momentum=0.99
- Learning rate: 0.001
- Batch size: 1
- Weight init: Gaussian σ=√(2/N)

**Loss Function:**
- Weighted pixel-wise cross-entropy
- w(x) = wc(x) + w0·exp(-(d1+d2)²/2σ²)
- w0=10, σ≈5 pixels

**Data Augmentation:**
- Elastic deformations (σ=10 pixels on 3×3 grid)
- Random rotations
- Random shifts
- Gaussian noise

## Experiments

### EM Segmentation (ISBI)
Neuronal structure segmentation in electron microscopy

Expected: Warping error 0.000353, Rand error 0.0382
Dataset: 30 training images (512×512)

### PhC-U373 Cell Tracking
Glioblastoma-astrocytoma cells (phase contrast)

Expected: IoU ~92% (vs 2nd place: 83%)
Dataset: 35 training images

### DIC-HeLa Cell Tracking
HeLa cells (differential interference contrast)

Expected: IoU ~77.5% (vs 2nd place: 46%)
Dataset: 20 training images

## API Examples

### Inference

```python
from src.model import UNet
from src.inference import UNetInference

model = UNet(in_channels=1, num_classes=2)
inference = UNetInference(model)
prediction = inference.predict(image)
```

### Training

```python
from src.trainer import UNetTrainer

trainer = UNetTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    use_weighted_loss=True
)
trainer.train()
```

### Metrics

```python
from src.metrics import SegmentationMetrics

iou = SegmentationMetrics.iou_score(pred, target)
dice = SegmentationMetrics.dice_score(pred, target)
```

## File Structure

```
src/
├── __init__.py
├── model.py           # U-Net architecture
├── trainer.py         # Training loop
├── datasets.py        # Dataset classes
├── losses_and_augmentation.py
├── metrics.py         # Evaluation metrics
└── inference.py       # Inference utilities

experiments/
├── em_segmentation.py
├── cell_tracking.py
└── demo.py

verify.py
requirements.txt
README.md
```

## Dataset Preparation

Expected directory structure:
```
data/
├── em_segmentation/
│   ├── train/{images,masks}/
│   └── test/{images,masks}/
└── cell_tracking/
    ├── phc_u373/{train,test}/{images,masks}/
    └── dic_hela/{train,test}/{images,masks}/
```

## References

```bibtex
@inproceedings{ronneberger2015u,
  title={U-net: Convolutional networks for biomedical image segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  booktitle={Medical Image Computing and Computer-Assisted Intervention},
  pages={234--241},
  year={2015},
  organization={Springer}
}
```
