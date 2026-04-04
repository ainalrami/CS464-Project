"""
Deep learning model definition.

ResNet18 with a replaced fully-connected head for EuroSAT classification.
"""

import logging

import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)


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
