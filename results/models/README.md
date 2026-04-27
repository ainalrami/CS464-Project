# Trained Models

This folder contains trained classical ML model checkpoints.

XGBoost models are committed directly to the repository (each ~2.3 MB).
RandomForest models are too large for GitHub (186-291 MB each) — download
them from Google Drive.

## In the repository

| File | Size | What it is |
|------|------|------------|
| `XGBoost_hog.pkl` | 2.3 MB | XGB on HOG features (64×64) — test acc 0.714 |
| `XGBoost_hog_color.pkl` | 2.3 MB | XGB on HOG + HSV color (64×64) — test acc 0.894 |
| `XGBoost_hog_color_texture.pkl` | 2.2 MB | **Best classical model** — XGB on HOG + Color + LBP (64×64) — test acc **0.910** |

## RandomForest — download from Google Drive

Cloud storage link: **<https://drive.google.com/drive/u/0/folders/1LtoNuMKAFzCKIcYD7IX31L44J7hw256i>**

After downloading, place the `.pkl` files directly into this folder.

| File | Size | What it is |
|------|------|------------|
| `RandomForest_hog.pkl` | 291 MB | RF on HOG features (64×64) — test acc 0.631 |
| `RandomForest_hog_color.pkl` | 211 MB | RF on HOG + HSV color (64×64) — test acc 0.823 |
| `RandomForest_hog_color_texture.pkl` | 186 MB | RF on HOG + Color + LBP (64×64) — test acc 0.849 |
| `RandomForest_hog_color_texture_upscaled.pkl` | 203 MB | RF on HOG + Color + LBP at 128×128 — test acc 0.849 |

## Reproducing

```bash
python download_data.py
python run_ml.py --config configs/ml.yaml --model RandomForest
python run_ml.py --config configs/ml.yaml --model XGBoost
```

RF takes ~10 minutes (4 modes), XGBoost ~25 minutes (3 native modes only —
the upscaled mode is too slow without parallelism).
