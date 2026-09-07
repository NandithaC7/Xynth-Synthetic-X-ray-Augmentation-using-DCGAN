"""Lightweight CNN classifier for chest X-ray class prediction."""

from __future__ import annotations

import torch
from torch import nn


class XrayCNN(nn.Module):
    """
    Small convolutional classifier for multi-class X-ray diagnosis.

    Designed to be fast to train so baseline vs augmented comparisons
    remain practical on CPU.
    """

    def __init__(self, num_classes: int = 4, in_channels: int = 1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits for a batch of images."""
        return self.classifier(self.features(x))


def build_cnn(num_classes: int = 4, device: torch.device | None = None) -> XrayCNN:
    """Construct the classifier and move it to ``device``."""
    device = device or torch.device("cpu")
    return XrayCNN(num_classes=num_classes).to(device)
