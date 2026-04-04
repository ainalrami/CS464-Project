# EuroSAT ML/DL Comparison Study

A course project comparing 3 classical machine learning models and 1 deep learning model on the **EuroSAT RGB dataset** (10-class land cover classification from satellite images), with **robustness analysis** under image degradations.

## Project Structure

```
ML_DEMO/
├── configs/
│   ├── ml.yaml              # Classical ML configuration
│   ├── dl.yaml              # Deep learning configuration
│   └── robustness.yaml      # Robustness analysis configuration
├── src/
│   ├── data/
│   │   └── dataset.py       # Data loading, splitting, PyTorch Dataset
│   ├── features/
│   │   └── extractors.py    # HOG, Color Histogram, LBP extractors
│   ├── ml/
│   │   ├── train.py         # ML model training with hyperparameter tuning
│   │   └── evaluate.py      # ML model evaluation
│   ├── dl/
│   │   ├── model.py         # ResNet18 model builder
│   │   ├── train.py         # DL training loop
│   │   └── evaluate.py      # DL model evaluation
│   ├── robustness/
│   │   ├── degradations.py  # Image degradation functions
│   │   └── evaluate.py      # Robustness evaluation
│   └── evaluation/
│       ├── metrics.py       # Metrics computation
│       └── plots.py         # Visualization utilities
├── run_ml.py                # CLI: Run classical ML pipeline
├── run_dl.py                # CLI: Run deep learning pipeline
├── run_robustness.py        # CLI: Run robustness analysis
├── summarize_results.py     # CLI: Generate summary tables & plots
├── requirements.txt
├── README.md
└── report_notes.md
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Optional**: For XGBoost support, also install:
```bash
pip install xgboost
```
If XGBoost is not installed, Logistic Regression is used as a fallback.

### 2. Download EuroSAT RGB Dataset

Download the EuroSAT RGB dataset and place it at `./data/EuroSAT_RGB/`:

```
data/
└── EuroSAT_RGB/
    ├── AnnualCrop/
    ├── Forest/
    ├── HerbaceousVegetation/
    ├── Highway/
    ├── Industrial/
    ├── Pasture/
    ├── PermanentCrop/
    ├── Residential/
    ├── River/
    └── SeaLake/
```

You can customize the dataset path in the YAML config files.

## Step-by-Step Run Sequence

### Step 1: Run Classical ML Pipeline

```bash
python run_ml.py --config configs/ml.yaml
```

This will:
- Create a stratified train/val/test split (70/15/15)
- Extract features for 3 ablation modes (HOG, HOG+Color, HOG+Color+LBP)
- Train SVM, Random Forest, and XGBoost (or Logistic Regression) with hyperparameter tuning
- Evaluate on validation and test sets
- Save models, metrics, and confusion matrices

### Step 2: Run Deep Learning Pipeline

```bash
python run_dl.py --config configs/dl.yaml
```

This will:
- Train a ResNet18 model with transfer learning (with and without augmentation)
- Use early stopping and learning rate scheduling
- Evaluate on validation and test sets
- Save training curves, model checkpoints, and metrics

### Step 3: Run Robustness Analysis

```bash
python run_robustness.py --config configs/robustness.yaml
```

This will:
- Load all trained models (ML + DL)
- Evaluate on degraded test sets (Gaussian blur, noise, downsampling)
- Generate comparison tables and plots

### Step 4: Generate Summary

```bash
python summarize_results.py
```

This will:
- Aggregate all metrics into summary tables
- Generate publication-style comparison plots
- Print results to the terminal

## Models

| Model | Type | Details |
|-------|------|---------|
| SVM | Classical ML | RBF/Linear kernel, C tuning |
| Random Forest | Classical ML | n_estimators, max_depth tuning |
| XGBoost / LogReg | Classical ML | XGBoost preferred, LogReg fallback |
| ResNet18 | Deep Learning | Pre-trained on ImageNet, fine-tuned |

## Feature Ablation Modes

| Mode | Features |
|------|----------|
| `hog` | HOG only |
| `hog_color` | HOG + HSV Color Histogram |
| `hog_color_texture` | HOG + HSV Color Histogram + LBP |

## Robustness Degradations

| Degradation | Severity Levels |
|-------------|----------------|
| Gaussian Blur | σ = 1.0 (low), σ = 3.0 (high) |
| Gaussian Noise | σ = 0.05 (low), σ = 0.15 (high) |
| Downsampling | 2× (low), 4× (high) |

## Output Structure

```
results/
├── metrics/                    # CSV files with accuracy, P/R/F1
│   ├── ml_summary.csv
│   ├── dl_summary.csv
│   ├── overall_test_summary.csv
│   ├── robustness_results.csv
│   ├── robustness_pivot.csv
│   └── *_report.csv, *_metrics.csv
├── confusion_matrices/         # PNG heatmaps
├── plots/                      # Training curves, comparison charts
│   ├── *_training_curves.png
│   ├── model_comparison.png
│   ├── per_class_f1_comparison.png
│   └── robustness_comparison.png
├── models/                     # Saved model files (.pkl, .pt)
└── logs/                       # Split metadata, training logs
```

## Configuration

All experiment settings are controlled via YAML config files in `configs/`. Key settings:

- **Dataset path**: `dataset.root`
- **Random seed**: `random_seed` (default: 42)
- **Split ratios**: 70% train, 15% val, 15% test
- **Image sizes**: 64×64 for ML features, 224×224 for DL

## Reproducibility

- All random seeds are set globally (NumPy, PyTorch)
- Split metadata is saved as CSV and reused across runs
- Model checkpoints and hyperparameters are saved
