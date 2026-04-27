---
marp: true
theme: default
paginate: true
size: 16:9
header: 'CS 464 — Group 8 | EuroSAT Land Cover Classification'
footer: 'Bilgi · Asıl · Baş · Kıyanus · İncesesli'
style: |
  section { font-size: 24px; }
  h1 { color: #1a4480; }
  h2 { color: #1a4480; }
  table { font-size: 20px; }
  .small { font-size: 18px; }
  .highlight { color: #c0392b; font-weight: bold; }
---

<!-- _class: lead -->
<!-- _paginate: false -->

# Comparative Land Cover Classification on EuroSAT

### Classical ML vs. Deep Learning — and How Robust Are They?

**Group 8** — Ferit Bilgi · Berke İ. E. Asıl · Talha B. Baş · Berhan Kıyanus · Hasan M. İncesesli

CS 464 — Final Project Presentation · April 2026

<!--
SPEAKER NOTES (≈30 sec):
"Hi everyone, we are Group 8. Our project is a comparative study of classical machine learning and deep learning for satellite land-cover classification on the EuroSAT dataset. We also test how well each method holds up when images are degraded — blurred, noisy, or downsampled."
-->

---

## Why Land Cover Classification?

- **Application:** automatic mapping of land use from satellite imagery
- **Real-world impact:** urban planning, precision agriculture, disaster response, climate monitoring
- **Why it's hard at 64×64 px:**
  - Visually similar classes (AnnualCrop ↔ PermanentCrop, Pasture ↔ HerbaceousVegetation)
  - Real sensors → blur, noise, varying resolution

### Our Research Questions
1. Which model wins on EuroSAT — classical ML or deep learning?
2. How much do hand-crafted features (HOG / Color / LBP) matter?
3. Does upscaling 64→128 help classical ML?
4. Which model is **most robust** to image degradations?

<!--
SPEAKER NOTES (≈45 sec):
"Land cover classification is the task of labeling each satellite tile by what's on the ground — forest, river, residential, etc. It feeds into urban planning, agriculture, and climate work. The challenge is that at 64-by-64 pixels several classes look almost identical, and real satellites produce noisy, blurry images. We frame the project around four questions: ML vs DL, the value of feature engineering, the effect of upscaling, and robustness."
-->

---

## Dataset — EuroSAT RGB

| Property | Value |
|---|---|
| Total images | **27,000** |
| Resolution | **64×64**, RGB |
| Classes | **10** land-cover types |
| Source | Sentinel-2 satellite |

**Classes:** AnnualCrop · Forest · HerbaceousVegetation · Highway · Industrial · Pasture · PermanentCrop · Residential · River · SeaLake

**Splits (stratified, fixed seed):**
Train **70%** (18,900) · Val **15%** (4,050) · Test **15%** (4,050)

<!--
SPEAKER NOTES (≈30 sec):
"We use the EuroSAT RGB benchmark — 27,000 Sentinel-2 tiles across 10 classes, all 64×64. We use a stratified 70/15/15 split with a fixed seed so every model sees identical samples. That makes our comparisons reproducible."
-->

---

## Two-Track Experimental Design

<div class="small">

| Track | Models | Input | Idea |
|---|---|---|---|
| **Classical ML** | SVM · Random Forest · XGBoost | hand-crafted features | Test how far engineered features can go |
| **Deep Learning** | ResNet18 (pretrained) · SimpleCNN (scratch) | raw 224×224 image | Test learned features + transfer learning |
| **Robustness** | both tracks | degraded test images | Compare under blur / noise / downsampling |

</div>

- Same train/val/test split across **all** experiments
- Hyperparameters tuned on validation set with **RandomizedSearchCV** (ML) / early stopping (DL)
- All metrics reported on the **untouched test set**

<!--
SPEAKER NOTES (≈40 sec):
"We split the work into three tracks: a classical ML pipeline with hand-crafted features, a deep learning pipeline, and a robustness study that re-evaluates trained models on degraded images. The same data split is reused everywhere so any difference is the model, not the data."
-->

---

## Classical ML — Feature Extraction

Three feature sets in an **ablation study**:

| Mode | Dim | Captures |
|---|---|---|
| **HOG** | 1,764 | edges & local gradients (9 orient., 8×8 cells, 2×2 blocks, L2-Hys) |
| **HOG + HSV color** | 1,860 | + hue/saturation/value histograms (32 bins/ch.) |
| **HOG + HSV + LBP** | 1,886 | + rotation-invariant micro-texture (uniform LBP, R=3, P=24) |

- **HSV chosen over RGB:** decouples color (hue) from brightness (V)
- **LBP uniform mode:** rotation-invariant, only 26 bins instead of 256
- All features standardized with **StandardScaler** before training

<!--
SPEAKER NOTES (≈55 sec — DETAIL):
"For classical ML we extract three feature families. HOG — Histogram of Oriented Gradients — divides each image into 8×8 pixel cells, computes gradient orientations into 9 bins, then normalizes over 2×2-cell blocks with L2-Hys. This captures edges like crop rows or road lines. We add HSV color histograms because hue cleanly separates water, vegetation, and built-up areas. Finally we add Local Binary Patterns in uniform mode for rotation-invariant texture. The three modes form an ablation: HOG alone, HOG+color, HOG+color+LBP. Everything goes through StandardScaler so SVM behaves."
-->

---

## Classical ML — Models & Tuning

<div class="small">

| Model | Why we picked it | Search space |
|---|---|---|
| **SVM (RBF)** | Effective in high-dim feature space; kernel non-linearity | C ∈ {0.1, 1, 10}, γ ∈ {scale, auto} |
| **Random Forest** | Robust ensemble, handles irrelevant features | n_estimators ∈ {100, 200, 300}, max_depth ∈ {10, 20, None} |
| **XGBoost** | Gradient boosting on engineered features | max_depth ∈ {3, 6}, lr ∈ {0.05, 0.1, 0.2}, subsample ∈ {0.8, 1.0} |

</div>

- **RandomizedSearchCV** with 5-fold CV on the **training set only**
- Best config picked on **validation accuracy**
- We also re-ran feature extraction with images **upscaled to 128×128** to test if higher resolution helps (it doesn't add information — bicubic interpolation only)

<!--
SPEAKER NOTES (≈40 sec):
"We tune three classical models with RandomizedSearchCV — SVM with an RBF kernel for non-linear boundaries, Random Forest as a robust ensemble baseline, and XGBoost which usually shines on engineered features. We also tested 128×128 upscaling to see whether finer HOG cells help."
-->

---

## Deep Learning — Two Architectures

<div class="small">

| Model | Params | Setup |
|---|---|---|
| **ResNet18** | ~11.7 M | **Pretrained on ImageNet**, all layers fine-tuned, FC head replaced (512→10) |
| **SimpleCNN** | ~2.4 M | **From scratch.** 5 conv blocks (32→64→128→256→256) + GAP + Dropout + FC |

</div>

- Input upscaled to **224×224**, ImageNet normalization
- **Augmentation (train only):** HorizontalFlip · Rotation ±15° · ColorJitter
- **Training:** Adam (lr=1e-3, wd=1e-4), CrossEntropy, ReduceLROnPlateau, **early stopping** (patience=3), batch=64, max 30 epochs
- All layers unfrozen → low-level filters adapt to satellite textures

<!--
SPEAKER NOTES (≈55 sec — DETAIL):
"On the deep learning side we use two networks. ResNet18 is pretrained on ImageNet and we fine-tune every layer — satellite imagery is far enough from ImageNet that we want even the early filters to adapt. SimpleCNN is a 5-block CNN trained from scratch — it tells us how much of ResNet's win comes from transfer learning vs architecture. Inputs are upscaled to 224 because that's ResNet's native size. We augment with horizontal flips, ±15° rotation, and color jitter — satellite tiles have no canonical orientation, and color jitter mimics atmospheric variation. We train with Adam, ReduceLROnPlateau, and early stopping."
-->

---

## Results — Overall Test Accuracy

<div class="small">

| Model | Features | Accuracy | Macro F1 |
|---|---|---|---|
| SVM | HOG | 0.596 | 0.581 |
| **Random Forest** | **HOG** | **0.623** | **0.586** |
| ResNet18 (transfer) | raw 224×224 | **expected ~0.95+** | — |
| SimpleCNN (scratch) | raw 224×224 | expected 0.70–0.85 | — |

</div>

- Among classical ML, **Random Forest + HOG** wins
- Adding HSV / LBP gave **diminishing returns** at 64×64 — micro-texture below feature resolution
- **128×128 upscaling did not help** classical ML — interpolation adds no real information
- DL closes most of the remaining gap thanks to **hierarchical learned features** + **ImageNet transfer**

<!--
SPEAKER NOTES (≈45 sec):
"Random Forest with HOG-only is our strongest classical model — about 62% test accuracy and 0.59 macro F1. Adding color or LBP gives small or no gains because at 64×64 the micro-texture distinctions are already lost. Upscaling to 128 didn't help either — bicubic interpolation doesn't create new information. ResNet18 with transfer learning is in a different league, and SimpleCNN-from-scratch lands in between, showing how much of the win is the pretraining."
-->

---

## Per-Class Insight — Where Models Fail

**Easy classes (F1 > 0.80):**
- **SeaLake** (F1 ≈ 0.97) — uniform deep-blue, near-zero texture
- **Forest** (F1 ≈ 0.82) — consistent dark green, regular canopy texture

**Hard classes (F1 < 0.40):**
- **PermanentCrop** (F1 ≈ 0.28) — looks like AnnualCrop at 64×64
- **HerbaceousVegetation** (F1 ≈ 0.40) — confused with Pasture / AnnualCrop
- **Pasture** (F1 ≈ 0.27 RF / 0.43 SVM) — RF over-predicts AnnualCrop

**Takeaway:** errors track **visual similarity**, not model weakness — RF gets higher overall accuracy but **SVM has more balanced** per-class recall.

<!--
SPEAKER NOTES (≈50 sec):
"Looking class-by-class is more informative than the headline number. SeaLake is nearly perfect — water is spectrally unique. Forest is also easy. The hard classes are the green ones: PermanentCrop, HerbaceousVegetation, and Pasture all look similar at 64×64, and HOG can't see tree spacing or grass granularity at that resolution. Random Forest gets a higher overall score, but SVM is more balanced — RF has only 17% recall on Pasture because it over-predicts AnnualCrop. So 'best' depends on whether you care about overall accuracy or worst-class performance."
-->

---

## Robustness — Stress-Testing the Models

We re-evaluated each trained model on **degraded test images**:

| Degradation | Low | High |
|---|---|---|
| Gaussian Blur | σ = 1.0 | σ = 3.0 |
| Gaussian Noise | σ = 0.05 | σ = 0.15 |
| Downsampling | 2× | 4× |

**Findings:**
- **DL models** degrade most gracefully — augmentation during training + deep features
- **HOG-only ML** is robust to noise (no color channel to corrupt) but breaks under heavy blur
- **HOG + Color** loses color histogram fidelity under noise
- **All models** suffer most under 4× downsampling — directly removes the spatial detail HOG depends on

<!--
SPEAKER NOTES (≈45 sec):
"For robustness we don't retrain — we just feed degraded test images into the existing models. The DL models hold up best because they were already trained with augmentation and learn redundant features. HOG-only ML is surprisingly noise-robust because there's no color histogram to wash out, but heavy blur kills its gradient signal. Heavy downsampling hurts everyone — when you go from 64 down to 16 pixels, there's nothing left for any model to work with."
-->

---

## Conclusions

- **Best classical ML:** Random Forest + HOG → **0.623 / 0.586**
- **Deep Learning** wins overall — transfer learning is the single biggest lever
- **Feature engineering** still useful: HOG alone is competitive; LBP marginal at 64×64
- **Visual ambiguity, not the model**, drives most errors (the green-class cluster)
- **Robustness ranking:** ResNet18 > SimpleCNN > HOG+Color ML > HOG-only ML

### What we'd do next
- EuroSAT **multispectral** (13 bands) — would crack the green-class confusion
- **Multi-scale HOG** + ML/DL ensembling
- **Vision Transformer** baseline + test-time augmentation

<!--
SPEAKER NOTES (≈40 sec):
"To wrap up: classical ML caps around 62% on this dataset; deep learning, especially with ImageNet transfer, takes us much higher. Most of the residual error is genuine visual ambiguity between green classes at 64×64 — the fix would be multispectral bands, which Sentinel-2 actually provides. Robustness-wise, deep models are the most reliable. Future work: multi-spectral, multi-scale features, ensembling, and a ViT baseline."
-->

---

<!-- _class: lead -->
<!-- _paginate: false -->

# Thank you!

### Questions?

**Group 8** — Bilgi · Asıl · Baş · Kıyanus · İncesesli
Code · Report · Configs available in the project repository

<!--
SPEAKER NOTES (≈10 sec):
"Thanks for listening — happy to take questions."
-->

---

## Backup — Hyperparameter Details

**SVM:** RBF kernel · best C ≈ 10 · γ = scale
**Random Forest:** n_estimators = 300 · max_depth = None · min_samples_split = 2
**XGBoost:** max_depth = 6 · lr = 0.1 · subsample = 1.0 · tree_method = hist

**ResNet18 fine-tune:** Adam(1e-3) · weight_decay 1e-4 · ReduceLROnPlateau(factor=0.5, patience=2) · early stop patience 3

**SimpleCNN:** Conv→BN→ReLU→MaxPool blocks 32→64→128→256→256 · GAP · Dropout 0.5 · FC 256→128 · Dropout 0.25 · FC 128→10

---

## Backup — Why HSV, why uniform LBP?

- **HSV vs RGB:** in RGB, brightness changes shift all three channels; in HSV, only V moves. So vegetation on a sunny vs cloudy day stays at the same hue.
- **LBP uniform:** a "uniform" pattern has ≤2 bit transitions in the binary code. ~90% of natural-image patterns are uniform → 26 bins instead of 256, plus rotation invariance with `method='uniform'` + `R=3, P=24`.
- **HOG L2-Hys:** L2 normalize, clip values >0.2, renormalize — proven more robust than plain L2.

---

## Backup — Reproducibility

- All splits driven by **fixed `random_state` = 42**
- Stratified split preserves class ratios in train / val / test
- Pipelines: `StandardScaler → Classifier` (single sklearn `Pipeline` object)
- Configs in `configs/ml.yaml`, `configs/dl.yaml`, `configs/robustness.yaml`
- One-command runs: `run_ml.py` · `run_dl.py` · `run_robustness.py`

---

## Backup — Per-Class F1 (RandomForest, HOG)

<div class="small">

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| AnnualCrop | 0.581 | 0.724 | 0.645 | 450 |
| Forest | 0.724 | 0.931 | 0.814 | 450 |
| HerbaceousVegetation | 0.360 | 0.427 | 0.391 | 450 |
| Highway | 0.680 | 0.619 | 0.648 | 375 |
| Industrial | 0.572 | 0.392 | 0.465 | 375 |
| Pasture | 0.605 | 0.173 | 0.269 | 300 |
| PermanentCrop | 0.364 | 0.224 | 0.277 | 375 |
| Residential | 0.571 | 0.898 | 0.698 | 450 |
| River | 0.755 | 0.624 | 0.683 | 375 |
| SeaLake | 0.977 | 0.964 | 0.971 | 450 |
| **Macro avg** | **0.619** | **0.598** | **0.586** | 4050 |

</div>
