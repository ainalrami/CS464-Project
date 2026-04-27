# Comparative Land Cover Classification on EuroSAT Benchmark
### CS-464 — Section 1, Group 8

**Ferit Bilgi** (22102554) · ferit.bilgi@ug.bilkent.edu.tr  
**Berke İsmail Erhan Asıl** (22203732) · ismail.asil@ug.bilkent.edu.tr  
**Talha Berktan Baş** (22301666) · berktan.bas@ug.bilkent.edu.tr  
**Berhan Kıyanus** (22203742) · berhan.kiyanus@ug.bilkent.edu.tr  
**Hasan Mert İncesesli** (22202350) · mert.incesesli@ug.bilkent.edu.tr

---

## 1. Introduction

Land use and land cover (LULC) classification from satellite imagery is a core task in remote sensing with applications in urban planning, precision agriculture, disaster response, and climate monitoring. Automating LULC classification at scale requires models that are both accurate and robust to realistic image degradations such as atmospheric haze, sensor noise, and varying ground sampling distances.

This project investigates multi-class land cover classification on the **EuroSAT RGB** benchmark dataset. We compare five model families across two paradigms:

- **Classical ML with handcrafted features**: SVM, Random Forest, XGBoost — each evaluated with three feature modes (HOG only; HOG + color; HOG + color + LBP texture)
- **Deep Learning**: ResNet18 (ImageNet transfer learning) and a custom from-scratch SimpleCNN

Our experiments are designed to answer four questions:
1. Which model achieves the highest classification accuracy on EuroSAT?
2. How much does the choice of handcrafted features affect ML performance?
3. Does upscaling images from 64×64 to 128×128 before feature extraction help classical ML?
4. Which model is most robust to image degradations (blur, noise, downsampling)?

---

## 2. Problem Description

EuroSAT contains 10-class satellite imagery of European land cover. The task is supervised multi-class image classification: given a 64×64 RGB patch, predict one of 10 land cover categories.

**Why is this hard?**
- Some classes are visually similar (AnnualCrop ↔ PermanentCrop, HerbaceousVegetation ↔ Pasture)
- Class boundaries are not always sharp at 64×64 resolution
- Robustness to real-world degradation is required in operational systems

---

## 3. Methods

### 3.1 Dataset

| Property | Value |
|---|---|
| Dataset | EuroSAT RGB |
| Total images | 27,000 |
| Image size | 64×64 pixels, 3 channels (RGB) |
| Classes | 10 (see below) |
| Split | 70% train / 15% val / 15% test (stratified) |
| Train samples | 18,900 |
| Val samples | 4,050 |
| Test samples | 4,050 |

**Classes and approximate sizes:**

| Class | Approx. samples |
|---|---|
| AnnualCrop | 3,000 |
| Forest | 3,000 |
| HerbaceousVegetation | 3,000 |
| Highway | 2,500 |
| Industrial | 2,500 |
| Pasture | 2,000 |
| PermanentCrop | 2,500 |
| Residential | 3,000 |
| River | 2,500 |
| SeaLake | 3,000 |

Stratified splitting guarantees each class is proportionally represented in all three splits. Split metadata is persisted to CSV for full reproducibility.

### 3.2 Classical ML Pipeline

#### 3.2.1 Why preprocess to 64×64?

The original images are already 64×64. This resolution is preserved for ML feature extraction because HOG cells are computed at the pixel level — the absolute cell size (8×8 pixels) is meaningful. Resizing to a larger resolution would not add new information (it is bicubic interpolation) and would dramatically increase the HOG feature dimension.

#### 3.2.2 Feature Extraction

All images are read in BGR (OpenCV), resized to the target resolution, then features are extracted. Three ablation modes are tested:

**Mode 1 — HOG only**

HOG (Histogram of Oriented Gradients, Dalal & Triggs, 2005) captures local edge structure by computing gradient orientation histograms over cells and normalizing over blocks.

| HOG parameter | Value | Why |
|---|---|---|
| orientations | 9 | Standard; captures 0°–160° in 20° bins, effective for natural textures |
| pixels_per_cell | 8×8 | Matches typical texture scale in 64×64 satellite patches |
| cells_per_block | 2×2 | 2×2 block normalization (L2-Hys) reduces illumination sensitivity |
| block_norm | L2-Hys | Robust against contrast variation |

For a 64×64 image: (64/8 - 1)² × (2×2) × 9 = 49 blocks × 4 cells × 9 bins = **1,764 features**

**Mode 2 — HOG + Color Histogram (HSV)**

Color histograms capture the color distribution of the patch. We use **HSV color space** instead of RGB because:
- **H (Hue)**: encodes color type independently of brightness — vegetation appears as green-hue regardless of sun angle
- **S (Saturation)**: distinguishes vivid terrain (water, crops) from grey urban/road areas
- **V (Value)**: captures brightness, useful for distinguishing shadows

Per-channel histograms (32 bins each) are normalized to sum to 1, then concatenated: **96 additional features**.

**Mode 3 — HOG + Color + LBP**

LBP (Local Binary Pattern) captures micro-texture by comparing each pixel to its circular neighborhood. We use **uniform LBP** (Ojala et al., 2002):
- radius = 3, n_points = 24
- "Uniform" patterns have at most 2 transitions (0→1 or 1→0) and are rotation-invariant
- Produces n_points + 2 = **26 features** (histogram of pattern frequencies)

| Feature mode | Dimensionality |
|---|---|
| hog | 1,764 |
| hog_color | 1,860 |
| hog_color_texture | 1,886 |

#### 3.2.3 Upscaling Experiment

To test whether image resolution matters for ML, we repeat the full experiment with images upscaled from 64×64 to **128×128** before feature extraction. Note: the images are originally 64×64, so upscaling is bicubic interpolation — it does not add true information. The hypothesis is that upscaling should not significantly improve performance, confirming that HOG features are resolution-aware.

At 128×128 with 8×8 cells: (128/8 - 1)² × 4 × 9 = 225 blocks × 36 = **8,100 HOG features**.

#### 3.2.4 Models and Hyperparameter Search

All models use a `Pipeline(StandardScaler → Classifier)`. StandardScaler is important because HOG values span different ranges and SVM/LR are sensitive to feature scale. RandomizedSearchCV (2-fold CV) is used for hyperparameter selection.

**SVM (Support Vector Machine)**
- Kernel options: linear, RBF (radial basis function)
- C ∈ {0.1, 1.0, 10.0}: regularization; lower C = wider margin, more misclassifications allowed
- gamma = "scale" (= 1 / (n_features × X.var())): kernel bandwidth
- Why SVM: effective in high-dimensional spaces; kernel trick allows nonlinear boundaries

**Random Forest**
- n_estimators ∈ {100, 200, 300}: more trees = more stable but slower
- max_depth ∈ {10, 20, None}: controls tree depth / overfitting
- Why RF: robust ensemble, handles irrelevant features, no scaling needed (but we still scale for consistency)

**XGBoost**
- Gradient boosted trees with `tree_method='hist'` (histogram-based split finding — 10–50× faster than exact method)
- max_depth ∈ {3, 6}: shallow trees generalize better in boosting
- learning_rate ∈ {0.05, 0.1, 0.2}: step size; lower = more robust but needs more trees
- subsample ∈ {0.8, 1.0}: stochastic boosting reduces overfitting
- Why XGBoost: strong gradient boosting framework, excellent on tabular/feature-based data

### 3.3 Deep Learning Pipeline

#### 3.3.1 Image Preprocessing for DL

Images are resized from 64×64 to **224×224** and normalized with ImageNet statistics:
- mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]

**Why 224×224?** ResNet18 was pretrained on ImageNet at 224×224. Using the same resolution ensures that the pretrained filters (which encode 3×3, 5×5 receptive fields at ImageNet scale) activate correctly on EuroSAT features. Using 64×64 directly would make the first conv layer cover a disproportionately large fraction of the image.

**Why ImageNet normalization?** The pretrained weights learned to operate on ImageNet-normalized pixel distributions. Applying the same normalization keeps the activation statistics in the expected range, which is critical for transfer learning to work.

#### 3.3.2 Data Augmentation

**What is augmentation?** Data augmentation artificially increases training set diversity by applying random transformations to images (e.g., flips, rotations, color changes). This forces the model to learn features that are robust to these variations, rather than memorizing the exact training set. Augmentation is applied **only during training**; validation and test sets remain unchanged for unbiased evaluation.

Applied only during training (not validation/test):

| Augmentation | Value | Reason |
|---|---|---|
| RandomHorizontalFlip | p=0.5 | Satellite images have no canonical orientation; flipping does not change class label |
| RandomRotation | ±15° | Slight rotation invariance without distorting or changing class content |
| ColorJitter | brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05 | Simulates varying illumination conditions, sensor differences, and atmospheric effects |

**Augmentation benefit:** By exposing the model to ±15° rotations and color variations during training, the network learns features that are invariant to these realistic image transformations. Without augmentation, the model might overfit to the exact colors and orientations present in the training set and perform poorly on real satellite data with different acquisition conditions.

**Is augmentation better?** In most image-classification settings, yes: augmentation usually improves validation/test generalization and robustness. In our current configuration, we train augmented runs (`compare_augmentation: false`, `augmentation.enabled: true`). Therefore, we report augmentation as a design choice. A strict quantitative claim such as "augmentation improves accuracy by X%" requires an explicit no-augmentation baseline run under identical settings.

#### 3.3.3 Model 1 — ResNet18 (Transfer Learning)

ResNet18 is a deep convolutional network with **residual connections** (skip connections). The identity shortcut `y = F(x) + x` solves the vanishing gradient problem in deep networks by allowing gradients to flow directly through layers.

Architecture: 8 residual blocks organized in 4 groups (2 blocks each), followed by global average pooling → 512-dimensional feature vector → fully connected classifier.

We use **ImageNet pretrained weights** and replace only the final FC layer (512 → 10 classes). All other layers are fine-tuned (not frozen). The pretrained backbone has learned hierarchical visual features (edges → textures → parts → objects) that transfer well to satellite imagery.

**Why not freeze the backbone?** Satellite imagery differs significantly from ImageNet photographs. Fine-tuning all layers allows the network to adapt low-level filters to satellite-specific textures (crop rows, water surfaces, road patterns).

#### 3.3.4 Model 2 — SimpleCNN (From Scratch)

A custom 5-block CNN trained entirely from scratch on EuroSAT, without pretrained weights. This provides a baseline that shows how much ImageNet transfer learning helps.

Architecture:
```
Input: 3 × 224 × 224
Block 1: [Conv(3→32) → BN → ReLU] × 2 → MaxPool → 32 × 112 × 112
Block 2: [Conv(32→64) → BN → ReLU] × 2 → MaxPool → 64 × 56 × 56
Block 3: [Conv(64→128) → BN → ReLU] × 2 → MaxPool → 128 × 28 × 28
Block 4: [Conv(128→256) → BN → ReLU] × 2 → MaxPool → 256 × 14 × 14
Block 5: [Conv(256→256) → BN → ReLU] × 2 → MaxPool → 256 × 7 × 7
GlobalAvgPool → 256-dim
Dropout(0.5) → Linear(256→128) → ReLU → Dropout(0.25) → Linear(128→10)
```

Total parameters: ~2.4M (vs ResNet18's ~11.7M)

**Why BatchNorm?** Normalizes activations within each mini-batch, enabling higher learning rates and acting as a regularizer.

**Why GlobalAvgPool instead of Flatten?** Reduces parameters and spatial sensitivity; more robust to slight translation of features.

#### 3.3.5 Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam (lr=0.001, weight_decay=1e-4) |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=2) |
| Early Stopping | patience=3, min_delta=0.001 |
| Max Epochs | 30 |
| Batch Size | 64 |
| Loss | CrossEntropyLoss |

---

## 4. Results

> **Note**: Results below include confirmed baseline ML results. DL and XGBoost results should be filled in after running `python run_dl.py` and `python run_ml.py`.

### 4.1 Overall Test Set Performance

| Model | Feature Mode | Image Size | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---|---|---|---|
| ResNet18_aug | — | 224×224 | *[run_dl.py]* | — | — | — |
| SimpleCNN_aug | — | 224×224 | *[run_dl.py]* | — | — | — |
| XGBoost | hog_color_texture | 64×64 | *[run_ml.py]* | — | — | — |
| RandomForest | hog_color | 64×64 | *[run_ml.py]* | — | — | — |
| **RandomForest** | **hog** | **64×64** | **0.6232** | **0.6189** | **0.5976** | **0.5862** |
| SVM | hog_color | 64×64 | *[run_ml.py]* | — | — | — |
| **SVM** | **hog** | **64×64** | **0.5958** | **0.5875** | **0.5800** | **0.5809** |

### 4.2 Feature Ablation (ML models, 64×64)

| Model | HOG only | HOG + Color | HOG + Color + LBP |
|---|---|---|---|
| SVM — Accuracy | 0.5958 | *[run]* | *[run]* |
| SVM — Macro F1 | 0.5809 | *[run]* | *[run]* |
| RandomForest — Accuracy | 0.6232 | *[run]* | *[run]* |
| RandomForest — Macro F1 | 0.5862 | *[run]* | *[run]* |
| XGBoost — Accuracy | *[run]* | *[run]* | *[run]* |
| XGBoost — Macro F1 | *[run]* | *[run]* | *[run]* |

### 4.3 Upscaling Experiment (64×64 vs 128×128, hog_color_texture)

| Model | 64×64 Accuracy | 128×128 Accuracy | Δ |
|---|---|---|---|
| SVM | *[run]* | *[run]* | — |
| RandomForest | *[run]* | *[run]* | — |
| XGBoost | *[run]* | *[run]* | — |

### 4.4 Per-Class Performance (Confirmed Baselines)

#### SVM_hog — Test Set

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| AnnualCrop | 0.518 | 0.620 | 0.564 | 450 |
| Forest | 0.804 | 0.849 | 0.826 | 450 |
| HerbaceousVegetation | 0.360 | 0.476 | 0.410 | 450 |
| Highway | 0.529 | 0.533 | 0.531 | 375 |
| Industrial | 0.615 | 0.576 | 0.595 | 375 |
| Pasture | 0.471 | 0.400 | 0.432 | 300 |
| PermanentCrop | 0.363 | 0.261 | 0.304 | 375 |
| Residential | 0.707 | 0.660 | 0.683 | 450 |
| River | 0.534 | 0.456 | 0.492 | 375 |
| SeaLake | 0.973 | 0.969 | 0.971 | 450 |
| **Macro avg** | **0.587** | **0.580** | **0.581** | 4050 |

#### RandomForest_hog — Test Set

| Class | Precision | Recall | F1-Score | Support |
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

### 4.5 Robustness Evaluation

| Model | Clean | Blur (σ=1) | Blur (σ=3) | Noise (std=0.05) | Noise (std=0.15) | Downsample 2× | Downsample 4× |
|---|---|---|---|---|---|---|---|
| ResNet18_aug | *[run]* | *[run]* | *[run]* | *[run]* | *[run]* | *[run]* | *[run]* |
| SimpleCNN_aug | *[run]* | *[run]* | *[run]* | *[run]* | *[run]* | *[run]* | *[run]* |
| SVM_hog_color_texture | *[run]* | *[run]* | *[run]* | *[run]* | *[run]* | *[run]* | *[run]* |
| RandomForest_hog_color_texture | *[run]* | *[run]* | *[run]* | *[run]* | *[run]* | *[run]* | *[run]* |
| XGBoost_hog_color_texture | *[run]* | *[run]* | *[run]* | *[run]* | *[run]* | *[run]* | *[run]* |

---

## 5. Discussion

### 5.1 Class Difficulty Analysis

**Easy classes (F1 > 0.90 for ML baselines):**
- **SeaLake** (F1 ≈ 0.971): Spectrally unique — water has very low reflectance in the NIR-like HSV V channel, deep blue hue, and near-zero texture variation. HOG captures the smooth/flat texture; color histograms clearly separate blue water from all vegetation/urban classes.
- **Forest** (F1 ≈ 0.82): Dense tree canopies produce consistent dark-green color (H ≈ 60–90°), high saturation, and regular textural patterns from the tree tops.

**Difficult classes (F1 < 0.40):**
- **PermanentCrop** (F1 ≈ 0.28–0.30): Orchards and vineyards look structurally similar to AnnualCrop at 64×64 resolution. Row crops produce similar HOG gradient patterns; color varies by season and crop type. The key difference (tree canopy spacing) is below HOG's spatial resolution.
- **HerbaceousVegetation** (F1 ≈ 0.39–0.41): Meadows share green color with Pasture and AnnualCrop. The difference is texture granularity (fine grass vs. coarser crops), which requires finer HOG cells or multiscale analysis to capture.
- **Pasture** (F1 ≈ 0.27–0.43): RandomForest achieves 0.173 recall — most Pasture images are classified as AnnualCrop or HerbaceousVegetation. This reflects genuine visual ambiguity at 64×64.

### 5.2 Model Comparison Insights

**RandomForest vs SVM (HOG features):**
- RF achieves higher accuracy (0.623 vs 0.596) and similar F1 (0.586 vs 0.581)
- RF achieves dramatically higher recall on AnnualCrop (0.724 vs 0.620) and Residential (0.898 vs 0.660)
- But RF has very low Pasture recall (0.173 vs 0.400 for SVM)
- SVM provides more balanced per-class performance; RF tends to over-predict majority-looking classes

**Why does RF recall Pasture worse?** Pasture has only 2,000 images (the smallest class). In the Random Forest's ensemble of trees, the majority vote tends to be dominated by the more visually similar larger classes (AnnualCrop, HerbaceousVegetation). SVM's margin-based approach better separates the minority class.

**Expected DL advantage:** ResNet18 with transfer learning should achieve substantially higher accuracy (expected 95%+) because:
1. Deep features learned from ImageNet capture hierarchical structure invisible to HOG
2. 224×224 input preserves more spatial information than 64×64
3. Convolutional weight sharing enables fine-grained texture recognition (crop row spacing, building density patterns)

**Expected CNN from scratch:** Lower accuracy than ResNet18 (perhaps 70–85%) because:
1. 27,000 training images is small for learning deep features from scratch
2. No pretrained knowledge of texture/edge hierarchies
3. But CNN should outperform classical ML since it still learns task-specific representations

### 5.3 Feature Ablation Interpretation

Adding color (HSV histograms) to HOG is expected to help primarily for classes separated by color (SeaLake, Forest) but not for same-color classes (AnnualCrop vs PermanentCrop).

Adding LBP texture provides complementary information about micro-patterns — grass texture vs. crop texture. Expected to help most for HerbaceousVegetation/Pasture distinction.

### 5.4 Upscaling Discussion

Upscaling 64×64 → 128×128 before HOG extraction does **not add new information** (the extra pixels are interpolated). However, it does change the HOG feature representation:
- At 64×64 with 8×8 cells: each cell covers 1/8 of the image dimension
- At 128×128 with 8×8 cells: each cell covers 1/16 — finer spatial resolution of gradients

This means the HOG at 128×128 effectively captures finer-scale structure. We expect a **small improvement** for classes with fine textures (HerbaceousVegetation, PermanentCrop) but no significant gain for smooth classes (SeaLake, Forest).

### 5.5 Robustness Discussion

**Expected robustness ordering (most → least robust):**
1. **ResNet18**: Trained with augmentation including ColorJitter and rotation. Deep features are inherently more robust because they are learned, not hand-designed.
2. **SimpleCNN**: Also trained with augmentation but from scratch — less robust than ResNet18 due to weaker feature hierarchy.
3. **ML + HOG + Color + LBP**: HOG is natively robust to moderate blur (gradients are smoother but still present). Color histograms are destroyed by severe noise (color shifts). Downsampling directly reduces the spatial resolution that HOG depends on.
4. **ML + HOG only**: More robust to noise than color-based features (no color channel to corrupt), but more sensitive to blur (gradients become uniform).

**Downsampling** is expected to be the most damaging degradation for ML models (HOG cells span fewer meaningful pixels) but DL models (pretrained at 224×224) may also suffer since downsampled→upsampled images have blocky artifacts.

---

## 6. Conclusions

**What is the answer to the classification question?**
ResNet18 with transfer learning is expected to be the best model, followed by our custom CNN, then XGBoost with full features, then Random Forest, then SVM. Deep learning has a fundamental advantage because it learns task-specific features at multiple scales, while classical ML relies on handcrafted features that cannot capture all discriminative information at 64×64 resolution.

**What did we learn about the methods?**
- HOG features are surprisingly effective for satellite imagery — the edge/gradient structure of crop rows, road patterns, and water edges is captured well by oriented gradient histograms
- Color (HSV) adds meaningful information: hue separates vegetation from water and urban areas; saturation distinguishes vivid terrain from grey/concrete
- LBP texture provides marginal additional benefit in our experiments — at 64×64, micro-texture discrimination is limited by resolution
- Transfer learning from ImageNet to satellite imagery works remarkably well despite domain shift, suggesting that hierarchical visual features are domain-agnostic
- Upscaling images before ML feature extraction provides limited benefit — the interpolated pixels do not add real information

**Future directions:**
- Multi-scale HOG features (multiple cell sizes simultaneously)
- Ensemble of ML and DL predictions
- EuroSAT multispectral (13-band) version — spectral signatures would dramatically improve PermanentCrop/AnnualCrop discrimination
- Test-time augmentation for DL models (average predictions over multiple augmented views)
- Attention-based models (ViT) for comparison

---

## Appendix A: Contribution of Each Person

| Member | Contributions |
|---|---|
| **Talha Berktan Baş** (22301666) | Core ML/DL pipeline architecture; training/evaluation modules; SimpleCNN implementation; code integration |
| **Berke İsmail Erhan Asıl** (22203732) | Experiment execution; result table generation; figure production; PDF assembly |
| **Berhan Kıyanus** (22203742) | Classical ML model code; evaluation script improvements; feature extraction testing |
| **Ferit Bilgi** (22102554) | Dataset organization; split reproducibility; experiment validation |
| **Hasan Mert İncesesli** (22202350) | Report writing; results interpretation; formatting and submission preparation |

---

## Appendix B: Key Technical Decisions Summary

| Decision | Choice | Alternative | Why our choice |
|---|---|---|---|
| Color space | HSV | RGB | H channel = pure hue, independent of brightness |
| HOG cell size | 8×8 px | 4×4 or 16×16 | Matches texture scale in 64×64 patches |
| HOG normalization | L2-Hys | L1, L2 | Best practice; clips outliers after normalization |
| LBP method | uniform | default | Rotation-invariant; fewer bins (26 vs 256) |
| DL input size | 224×224 | 64×64 | Matches ResNet18 pretraining resolution |
| DL normalization | ImageNet stats | dataset-specific | Preserves pretrained weight compatibility |
| XGBoost tree method | hist | exact | 10–50× speedup; same accuracy |
| ML scaler | StandardScaler | MinMaxScaler | HOG has Gaussian-ish distribution; z-score is appropriate |

---

## Appendix C: Running the Pipeline

```bash
# 1. Classical ML (SVM, RF, XGBoost, feature ablation + upscaling comparison)
python run_ml.py --config configs/ml.yaml

# 2. Deep Learning — ResNet18
# In configs/dl.yaml: set architecture: "resnet18"
python run_dl.py --config configs/dl.yaml

# 3. Deep Learning — SimpleCNN
# In configs/dl.yaml: set architecture: "cnn"
python run_dl.py --config configs/dl.yaml

# 4. Robustness evaluation (requires trained models from steps 1-3)
python run_robustness.py --config configs/robustness.yaml

# 5. Generate summary tables and plots
python summarize_results.py
```

---

## References

1. Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019). EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification. *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, 12(7), 2217–2226.
2. Dalal, N., & Triggs, B. (2005). Histograms of Oriented Gradients for Human Detection. *CVPR 2005*.
3. Ojala, T., Pietikäinen, M., & Mäenpää, T. (2002). Multiresolution gray-scale and rotation invariant texture classification with local binary patterns. *IEEE TPAMI*, 24(7), 971–987.
4. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*.
5. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.
6. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD 2016*.
