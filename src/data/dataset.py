"""
Data loading, splitting, and PyTorch Dataset for EuroSAT RGB.

EuroSAT RGB is organized as:
    root/
        AnnualCrop/
            AnnualCrop_00001.jpg
            ...
        Forest/
            ...
        ...
"""

import json
import logging
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

# Canonical class ordering for reproducibility
CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake"
]


def load_dataset(root: str):
    """
    Scan the dataset root directory and return a list of (image_path, label_index) tuples.

    Args:
        root: Path to the EuroSAT_RGB folder with per-class subdirectories.

    Returns:
        data: List of (path_str, label_int) tuples.
        class_names: Sorted list of class names found.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {root}\n"
            "Please download EuroSAT RGB and place it at the configured path."
        )

    # Discover classes from subdirectories
    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if len(class_dirs) == 0:
        raise ValueError(f"No class subdirectories found in {root}")

    found_classes = [d.name for d in class_dirs]
    logger.info(f"Found {len(found_classes)} classes: {found_classes}")

    # Build class-to-index mapping using canonical order if possible
    if set(found_classes) == set(CLASS_NAMES):
        class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
        class_names = CLASS_NAMES
    else:
        class_names = found_classes
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    # Collect samples
    data = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    for class_name, idx in class_to_idx.items():
        class_dir = root / class_name
        if not class_dir.exists():
            logger.warning(f"Class directory missing: {class_dir}")
            continue
        for img_file in sorted(class_dir.iterdir()):
            if img_file.suffix.lower() in valid_extensions:
                data.append((str(img_file), idx))

    logger.info(f"Total samples: {len(data)}")
    return data, class_names


def create_splits(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Perform a stratified train/val/test split.

    Args:
        data: List of (path, label) tuples.
        train_ratio, val_ratio, test_ratio: Split proportions (must sum to 1).
        random_seed: Random seed for reproducibility.

    Returns:
        dict with keys 'train', 'val', 'test', each mapping to a list of (path, label).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"

    paths = [d[0] for d in data]
    labels = [d[1] for d in data]

    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    train_paths, valtest_paths, train_labels, valtest_labels = train_test_split(
        paths, labels,
        test_size=val_test_ratio,
        stratify=labels,
        random_state=random_seed
    )

    # Second split: val vs test
    relative_test_ratio = test_ratio / val_test_ratio
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        valtest_paths, valtest_labels,
        test_size=relative_test_ratio,
        stratify=valtest_labels,
        random_state=random_seed
    )

    splits = {
        "train": list(zip(train_paths, train_labels)),
        "val": list(zip(val_paths, val_labels)),
        "test": list(zip(test_paths, test_labels)),
    }

    for name, subset in splits.items():
        logger.info(f"  {name}: {len(subset)} samples")

    return splits


def save_split_metadata(splits, output_dir, class_names):
    """
    Save split metadata as CSV and class names as JSON for reproducibility.

    Args:
        splits: Dict with train/val/test lists of (path, label) tuples.
        output_dir: Directory to write the metadata files.
        class_names: List of class name strings.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, samples in splits.items():
        df = pd.DataFrame(samples, columns=["image_path", "label"])
        csv_path = output_dir / f"split_{split_name}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"  Saved {csv_path}")

    # Save class names mapping
    class_map = {name: idx for idx, name in enumerate(class_names)}
    json_path = output_dir / "class_names.json"
    with open(json_path, "w") as f:
        json.dump(class_map, f, indent=2)
    logger.info(f"  Saved {json_path}")


def load_split_metadata(output_dir):
    """
    Load previously saved split metadata.

    Returns:
        splits: Dict with train/val/test lists of (path, label) tuples.
        class_names: List of class name strings.
    """
    output_dir = Path(output_dir)

    splits = {}
    for split_name in ["train", "val", "test"]:
        csv_path = output_dir / f"split_{split_name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Split file not found: {csv_path}")
        df = pd.read_csv(csv_path)
        splits[split_name] = list(zip(df["image_path"].tolist(), df["label"].tolist()))

    with open(output_dir / "class_names.json", "r") as f:
        class_map = json.load(f)
    class_names = sorted(class_map.keys(), key=lambda k: class_map[k])

    return splits, class_names


# ---------------------------------------------------------------------------
# PyTorch Dataset for DL pipeline
# ---------------------------------------------------------------------------

class EuroSATDataset(Dataset):
    """
    PyTorch Dataset for EuroSAT images.

    Args:
        samples: List of (image_path, label) tuples.
        transform: torchvision transform to apply to each image.
    """

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def get_dl_transforms(image_size=(224, 224), augmentation_cfg=None, is_train=False):
    """
    Build torchvision transforms for the DL pipeline.

    Args:
        image_size: (H, W) target size.
        augmentation_cfg: Augmentation config dict (from dl.yaml).
        is_train: Whether to include training augmentations.

    Returns:
        A torchvision.transforms.Compose object.
    """
    # ImageNet normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform_list = []

    if is_train and augmentation_cfg and augmentation_cfg.get("enabled", False):
        transform_list.append(transforms.Resize(image_size))
        if augmentation_cfg.get("horizontal_flip", False):
            transform_list.append(transforms.RandomHorizontalFlip())
        if augmentation_cfg.get("vertical_flip", False):
            transform_list.append(transforms.RandomVerticalFlip())
        rotation = augmentation_cfg.get("rotation_degrees", 0)
        if rotation > 0:
            transform_list.append(transforms.RandomRotation(rotation))
        cj = augmentation_cfg.get("color_jitter", {})
        if cj:
            transform_list.append(transforms.ColorJitter(
                brightness=cj.get("brightness", 0),
                contrast=cj.get("contrast", 0),
                saturation=cj.get("saturation", 0),
                hue=cj.get("hue", 0),
            ))
    else:
        transform_list.append(transforms.Resize(image_size))

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return transforms.Compose(transform_list)
