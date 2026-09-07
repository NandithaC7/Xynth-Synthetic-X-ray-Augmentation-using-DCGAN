"""Torchvision transforms for DCGAN and classifier pipelines."""

from __future__ import annotations

from torchvision import transforms


def get_dcgan_transforms(image_size: int = 64) -> transforms.Compose:
    """
    Preprocess X-rays for DCGAN training.

    Converts to grayscale, resizes to ``image_size``, and normalizes to [-1, 1].
    """
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ]
    )


def get_classifier_transforms(image_size: int = 64) -> transforms.Compose:
    """
    Preprocess X-rays for the CNN classifier.

    Same spatial pipeline as DCGAN; ImageNet-style single-channel normalize.
    """
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ]
    )
