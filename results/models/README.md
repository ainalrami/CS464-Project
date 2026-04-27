# Trained Models

The trained model files (`.pkl`) are not included in this repository because each
exceeds GitHub's 100 MB single-file limit.

## Download

Cloud storage link: **<https://drive.google.com/drive/u/0/folders/1LtoNuMKAFzCKIcYD7IX31L44J7hw256i>**

After downloading, place all `.pkl` files directly into this folder
(`results/models/`).

## Files

| File | Size | What it is |
|------|------|------------|
| `RandomForest_hog.pkl` | 291 MB | RF trained on HOG features (64×64) |
| `RandomForest_hog_color.pkl` | 211 MB | RF trained on HOG + HSV color (64×64) |
| `RandomForest_hog_color_texture.pkl` | 186 MB | **Best model** — RF on HOG + Color + LBP (64×64) |
| `RandomForest_hog_color_texture_upscaled.pkl` | 203 MB | RF on HOG + Color + LBP at 128×128 |

## Reproducing

If you'd rather regenerate the models locally:

```bash
python download_data.py
python run_ml.py --config configs/ml.yaml --model RandomForest
```

This will retrain all four RF variants on the EuroSAT dataset (~10 minutes
total on a modern CPU). Output goes to `results/models/`.
