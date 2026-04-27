"""
Deep learning model definitions.

  - ResNet18: ImageNet pre-trained backbone with a custom classification head.
  - SimpleCNN: Lightweight from-scratch CNN for EuroSAT classification.
"""

import logging

import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom CNN
# ---------------------------------------------------------------------------

class _ConvBlock(nn.Sequential):
    """Conv2d → BatchNorm → ReLU."""
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class SimpleCNN(nn.Module):
    """
    Five-block CNN designed for 224×224 RGB inputs (EuroSAT).

    Spatial sizes (224 input):
        Block 1:  → 112×112   (MaxPool)
        Block 2:  →  56×56    (MaxPool)
        Block 3:  →  28×28    (MaxPool)
        Block 4:  →  14×14    (MaxPool)
        Block 5:  →   7×7     (MaxPool)
        GAP:      →   1×1
    """

    def __init__(self, num_classes=10, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 3 → 32
            _ConvBlock(3, 32),
            _ConvBlock(32, 32),
            nn.MaxPool2d(2, 2),        # 224 → 112

            # Block 2: 32 → 64
            _ConvBlock(32, 64),
            _ConvBlock(64, 64),
            nn.MaxPool2d(2, 2),        # 112 → 56

            # Block 3: 64 → 128
            _ConvBlock(64, 128),
            _ConvBlock(128, 128),
            nn.MaxPool2d(2, 2),        # 56 → 28

            # Block 4: 128 → 256
            _ConvBlock(128, 256),
            _ConvBlock(256, 256),
            nn.MaxPool2d(2, 2),        # 28 → 14

            # Block 5: 256 → 256
            _ConvBlock(256, 256),
            _ConvBlock(256, 256),
            nn.MaxPool2d(2, 2),        # 14 → 7
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_cnn(num_classes=10, dropout=0.5):
    """Build a SimpleCNN from scratch (no pretrained weights)."""
    model = SimpleCNN(num_classes=num_classes, dropout=dropout)
    logger.info(f"Built SimpleCNN (num_classes={num_classes}, dropout={dropout})")
    return model


# ---------------------------------------------------------------------------
# ResNet18
# ---------------------------------------------------------------------------

def build_model(num_classes=10, pretrained=True, freeze_backbone=False):
    """
    Build a ResNet18 model with a custom classification head.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to load ImageNet pre-trained weights.
        freeze_backbone: If True, freeze all layers except the final FC layer.

    Returns:
        PyTorch model ready for training.
    """
    if pretrained:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
    else:
        weights = None

    model = models.resnet18(weights=weights)

    # Freeze backbone if requested
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        logger.info("Backbone layers frozen. Only FC head will be trained.")

    # Replace the final fully-connected layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    logger.info(f"Built ResNet18 model (pretrained={pretrained}, "
                f"num_classes={num_classes}, freeze_backbone={freeze_backbone})")

    return model
