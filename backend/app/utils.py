"""Shared filesystem and serialization helpers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid, save_image


def ensure_dirs(*paths: Path) -> None:
    """Create directories if they do not exist."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    """Select CUDA if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_json(data: dict[str, Any], path: Path) -> None:
    """Write a JSON file with indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path) -> dict[str, Any] | None:
    """Load JSON if the file exists, otherwise return None."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_loss_plot(
    g_losses: list[float],
    d_losses: list[float],
    out_path: Path,
    title: str = "DCGAN Training Loss",
) -> None:
    """Plot and save generator / discriminator loss curves."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(g_losses, color="#1E5EFF", label="Generator", linewidth=1.5)
    ax.plot(d_losses, color="#0A3D91", label="Discriminator", linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("BCE Loss")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_image_grid(
    tensor: torch.Tensor,
    path: Path,
    nrow: int = 8,
    normalize: bool = True,
) -> None:
    """Save a batch of images as a grid PNG (expects [-1, 1] range)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = make_grid(tensor, nrow=nrow, normalize=normalize, value_range=(-1, 1))
    save_image(grid, path)


def denormalize_to_uint8(tensor: torch.Tensor) -> torch.Tensor:
    """Map a [-1, 1] tensor to [0, 255] uint8."""
    img = (tensor.clamp(-1, 1) + 1) / 2.0
    return (img * 255).round().to(torch.uint8)


def copy_tree_files(src: Path, dst: Path, extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg")) -> int:
    """Copy image files from src into dst. Returns number of files copied."""
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    if not src.exists():
        return 0
    for path in src.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            shutil.copy2(path, dst / path.name)
            count += 1
    return count
