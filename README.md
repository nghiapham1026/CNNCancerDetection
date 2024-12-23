# Histopathologic Cancer Detection

This repository contains a mini-project for detecting metastatic tissue in histopathologic scans of lymph node sections. The task is a **binary classification** problem (tumor vs. no tumor) adapted from the [PatchCamelyon (PCam) benchmark dataset](https://github.com/basveeling/pcam).

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Training and Comparison](#training-and-comparison)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
- [Methodology](#methodology)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

**Goal**:  
Create a robust classifier to identify whether a 96×96 pathology image patch contains tumor tissue in its central region. Achieving high accuracy on this task has clinical importance for accelerating histopathologic diagnostics and potentially reducing human error.

**Key Points**:  
- **Data**: Thousands of labeled images (RGB `.tif` files) representing lymph node sections.  
- **Models**: 
  1. A **Simple CNN** built from scratch (baseline).  
  2. Pretrained **ResNet** models (transfer learning).  
- **Performance Metric**: Primary focus on **Accuracy**, though medical contexts might also consider ROC AUC or F1-score.

---

## Project Structure

```
histopathology-cancer-detection/
├── README.md                <- You're reading this file
├── requirements.txt         <- Dependencies
├── dataset.py               <- Custom PyTorch Dataset class
├── transforms.py            <- Torchvision transforms (train & val)
├── models.py                <- CNN and ResNet-based architectures
├── train.py                 <- Main training script (comparison of architectures)
├── hyperparam_tuning.py     <- Script for hyperparameter tuning
└── (train_labels.csv)       <- Provided labels file (not in repo by default)
└── (train/)                 <- Directory of .tif images (not in repo by default)
```

- **`dataset.py`**: Defines the `HistopathologyDataset` class for loading images and labels.  
- **`transforms.py`**: Contains train/validation transforms (resizing, normalization, data augmentation).  
- **`models.py`**: Houses both the **Simple CNN** and the **ResNet** architectures.  
- **`train.py`**: Trains and compares multiple architectures (from-scratch vs. ResNet18). Plots learning curves.  
- **`hyperparam_tuning.py`**: Automates experiments for different hyperparameters (learning rate, weight decay, architecture). Displays results and a bar chart.  
- **`requirements.txt`**: Lists required Python packages.

---

## Dataset

1. **Source**  
   The dataset is derived from the [PatchCamelyon (PCam)](https://github.com/basveeling/pcam) benchmark.  
   Alternatively, one can reference the [Kaggle Histopathologic Cancer Detection Competition](https://www.kaggle.com/c/histopathologic-cancer-detection) which provides a very similar dataset.

2. **Structure**  
   - **train_labels.csv**: CSV with two columns: `id` (file name prefix) and `label` (0 or 1).  
   - **train/**: Directory of images (`<id>.tif`) that correspond to entries in the CSV.  

3. **Important**  
   - Place `train_labels.csv` and the `train/` folder in the repository root (or update paths in `train.py` and `hyperparam_tuning.py`).  
   - Each `.tif` is a 96×96 color patch; the label is determined by the presence of tumor tissue in the **center** region.

---

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/<your-username>/histopathology-cancer-detection.git
   cd histopathology-cancer-detection
   ```
2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   Alternatively, you can create and activate a virtual environment or Conda environment before installing.

3. **Data Setup**  
   - Download and extract the dataset (either from Kaggle or PCam).  
   - Copy/move `train/` (with all `.tif` files) and `train_labels.csv` to the project root.  
   - Adjust file paths in the `.py` scripts if your directory structure differs.

---

## Quick Start

### Training and Comparison

1. **Train the Models**
   ```bash
   python train.py
   ```
   - This script:
     - Loads and splits the data (by default, 80% train / 20% validation).
     - Trains both the **Simple CNN** and **ResNet18**.
     - Logs accuracy and loss per epoch.
     - Displays side-by-side training vs. validation curves for both architectures.

2. **Expected Outputs**
   - Terminal printout showing epoch-by-epoch updates (loss, accuracy).  
   - A Matplotlib window (or saved image) comparing training vs. validation curves.

### Hyperparameter Tuning

1. **Run Experiments**
   ```bash
   python hyperparam_tuning.py
   ```
   - Automatically tries different combinations of:
     - Architectures (e.g., ResNet18, ResNet34)  
     - Learning rates (1e-3, 1e-4, etc.)  
     - Weight decays (0, 1e-5, etc.)  
   - Prints the best validation accuracy per combination and optionally plots a bar chart of final results.

2. **Insights**
   - See which model+hyperparams yield the highest accuracy.  
   - Decide on a final configuration to use in production or further refine.

---

## Methodology

1. **CNN from Scratch**  
   - 3 Convolution + Pooling blocks, followed by fully connected layers.  
   - Serves as a **baseline** to assess if deeper networks truly add value.

2. **Transfer Learning**  
   - **ResNet18** / **ResNet34** pretrained on ImageNet.  
   - Final fully connected layer is replaced for a 2-class output (tumor vs. not tumor).  
   - Typically requires fewer epochs to converge and handles complex patterns better.

3. **Data Augmentation**  
   - Random flips, rotations, and normalization in the training set.  
   - Helps reduce overfitting and improves generalization.

4. **Loss & Optimization**  
   - **CrossEntropyLoss** with **Adam** optimizer.  
   - Could integrate learning rate schedulers or other optimizers (SGD, RMSprop) in future experiments.

---

## Results

1. **Baseline (Simple CNN)**  
   - Achieves around **85–90%** validation accuracy (depending on hyperparams and epoch count).

2. **ResNet Models**  
   - Consistently achieve **90–95%** validation accuracy with proper fine-tuning.  
   - Lower learning rates (1e-4) and slight weight decay (1e-5) often yield the best performance.

3. **Hyperparameter Tuning Example**  
   - ResNet34 + **LR=1e-4** + **WD=1e-5** has reached **~93–94%** in sample runs with 10k training images.

*(Note: Actual numbers can vary depending on random seeds, subsets used, and epoch counts.)*

---

## Future Improvements

- **Advanced Augmentations**: MixUp, CutMix, or color jitter for more robust training.  
- **Additional Architectures**: EfficientNet, DenseNet, or Vision Transformers if compute resources allow.  
- **Other Metrics**: AUC, F1, Sensitivity, and Specificity for more clinically relevant evaluation.  
- **Inference Pipeline**: Extend to entire Whole-Slide Imaging (WSI) for a real clinical workflow.

---

## Contributing

Contributions and suggestions are welcome! Here’s how you can help:

1. **Fork** the repo.
2. **Create** a new branch:  
   ```bash
   git checkout -b feature/awesome-improvement
   ```
3. **Commit** your changes:  
   ```bash
   git commit -m "Add improved data augmentation"
   ```
4. **Push** to the branch:  
   ```bash
   git push origin feature/awesome-improvement
   ```
5. **Open** a Pull Request explaining what you changed.

---

## License

This project is licensed under the **[MIT License](https://opensource.org/licenses/MIT)** — you are free to use, modify, and distribute the code under the conditions described there.

---

## Acknowledgments

- [Kaggle Histopathologic Cancer Detection Competition](https://www.kaggle.com/c/histopathologic-cancer-detection)  
- [PatchCamelyon (PCam) Dataset](https://github.com/basveeling/pcam)  
- **PyTorch** and **TorchVision** teams for powerful deep learning frameworks.

> **Disclaimer**: This code and project are intended for educational and research purposes. They are **not** intended for direct clinical use. Always consult with medical professionals for diagnostic decisions.