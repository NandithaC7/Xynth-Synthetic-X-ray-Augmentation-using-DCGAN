"""Train baseline and augmented CNN classifiers."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import get_settings
from backend.app.utils import ensure_dirs, get_device, save_json
from backend.data.dataset_loader import XrayFolderDataset
from backend.data.preprocessing import get_classifier_transforms
from backend.models.cnn import build_cnn
from backend.training.evaluate import evaluate_model, save_confusion_matrix_plot


def _labels_from_dataset(dataset: XrayFolderDataset) -> list[int]:
    """Extract integer labels for stratified splitting."""
    return [label for _, label in dataset.samples]


def make_split_indices(
    labels: list[int],
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Create a stratified train/test index split."""
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    return train_idx.tolist(), test_idx.tolist()


def train_one_experiment(
    name: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_classes: int,
    epochs: int,
    device: torch.device,
    class_names: list[str],
    metrics_dir: Path,
    plots_dir: Path,
    checkpoints_dir: Path,
) -> dict:
    """Train a CNN and evaluate on the held-out test set."""
    model = build_cnn(num_classes=num_classes, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    history: list[dict] = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_acc = correct / max(total, 1)
        train_loss = running_loss / max(total, 1)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc})
        print(f"[{name}] Epoch {epoch + 1}/{epochs} loss={train_loss:.4f} acc={train_acc:.4f}")

    metrics = evaluate_model(model, test_loader, device, class_names)
    metrics["experiment"] = name
    metrics["history"] = history

    ckpt = checkpoints_dir / f"cnn_{name}.pt"
    torch.save({"model": model.state_dict(), "class_names": class_names}, ckpt)
    metrics["checkpoint"] = str(ckpt)

    cm_path = plots_dir / f"confusion_{name}.png"
    save_confusion_matrix_plot(metrics["confusion_matrix"], class_names, cm_path, title=f"{name} Confusion Matrix")
    metrics["confusion_matrix_plot"] = str(cm_path)

    save_json(metrics, metrics_dir / f"classifier_{name}.json")
    return metrics


def train_classifiers(
    epochs: int | None = None,
    max_per_class: int | None = None,
) -> dict:
    """
    Run baseline (original) and augmented classifier experiments.

    Uses an identical test split derived from the original dataset indices
    so only the training distribution changes between experiments.
    """
    settings = get_settings()
    epochs = epochs or settings.classifier_epochs
    device = get_device()
    ensure_dirs(settings.metrics_dir, settings.plots_dir, settings.checkpoints_dir)

    transform = get_classifier_transforms(settings.image_size)
    original = XrayFolderDataset(
        root=settings.dataset_dir,
        class_names=settings.class_names,
        transform=transform,
    )
    if len(original) == 0:
        raise FileNotFoundError(f"No images under {settings.dataset_dir}")

    # Optional downsampling for faster local demos
    if max_per_class is not None:
        kept: list[tuple] = []
        counts: dict[int, int] = {}
        for path, label in original.samples:
            if counts.get(label, 0) < max_per_class:
                kept.append((path, label))
                counts[label] = counts.get(label, 0) + 1
        original.samples = kept

    labels = _labels_from_dataset(original)
    train_idx, test_idx = make_split_indices(labels)
    # Persist split for reproducibility / API
    save_json(
        {"train_idx": train_idx, "test_idx": test_idx, "seed": 42},
        settings.metrics_dir / "split_indices.json",
    )

    train_loader_base = DataLoader(
        Subset(original, train_idx),
        batch_size=settings.batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        Subset(original, test_idx),
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=0,
    )

    class_names = list(settings.class_names)
    baseline = train_one_experiment(
        name="baseline",
        train_loader=train_loader_base,
        test_loader=test_loader,
        num_classes=len(class_names),
        epochs=epochs,
        device=device,
        class_names=class_names,
        metrics_dir=settings.metrics_dir,
        plots_dir=settings.plots_dir,
        checkpoints_dir=settings.checkpoints_dir,
    )

    # Augmented experiment: same real images as baseline + synthetic target-class samples.
    # Keeps the identical test indices; only training sees extra generated images.
    from backend.data.dataset_loader import list_image_paths
    from backend.training.generate_images import build_augmented_dataset, generate_images

    samples_dir = settings.generated_dir / "samples"
    if not samples_dir.exists() or not any(samples_dir.glob("*.png")):
        try:
            generate_images()
        except FileNotFoundError:
            raise FileNotFoundError(
                "No synthetic samples found. Train DCGAN and run generation first."
            ) from None
    # Also materialize on-disk augmented_dataset/ for inspection / reproducibility
    if not settings.augmented_dir.exists() or not any(settings.augmented_dir.iterdir()):
        build_augmented_dataset()

    target_label = class_names.index(settings.target_class)
    synthetic_paths = list_image_paths(samples_dir)
    augmented_ds = XrayFolderDataset(
        root=settings.dataset_dir,
        class_names=settings.class_names,
        transform=transform,
    )
    # Start from the same (possibly downsampled) real pool as baseline
    augmented_ds.samples = list(original.samples)
    for path in synthetic_paths:
        augmented_ds.samples.append((path, target_label))

    # Locked test set = same indices into the shared real prefix
    real_count = len(original.samples)
    aug_test_idx = list(test_idx)
    aug_train_idx = [i for i in train_idx] + list(range(real_count, len(augmented_ds)))

    train_loader_aug = DataLoader(
        Subset(augmented_ds, aug_train_idx),
        batch_size=settings.batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_loader_aug = DataLoader(
        Subset(augmented_ds, aug_test_idx),
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=0,
    )

    augmented = train_one_experiment(
        name="augmented",
        train_loader=train_loader_aug,
        test_loader=test_loader_aug,
        num_classes=len(class_names),
        epochs=epochs,
        device=device,
        class_names=class_names,
        metrics_dir=settings.metrics_dir,
        plots_dir=settings.plots_dir,
        checkpoints_dir=settings.checkpoints_dir,
    )

    comparison = {
        "baseline": {
            "accuracy": baseline["accuracy"],
            "precision": baseline["precision"],
            "recall": baseline["recall"],
            "f1": baseline["f1"],
        },
        "augmented": {
            "accuracy": augmented["accuracy"],
            "precision": augmented["precision"],
            "recall": augmented["recall"],
            "f1": augmented["f1"],
        },
        "delta_accuracy": augmented["accuracy"] - baseline["accuracy"],
        "class_names": class_names,
        "epochs": epochs,
        "device": str(device),
    }
    save_json(comparison, settings.metrics_dir / "comparison.json")

    # CSV summary
    csv_path = settings.metrics_dir / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "accuracy", "precision", "recall", "f1"])
        for exp_name, m in (("baseline", baseline), ("augmented", augmented)):
            writer.writerow([exp_name, m["accuracy"], m["precision"], m["recall"], m["f1"]])

    print(f"Comparison saved → {csv_path}")
    return comparison


def main() -> None:
    """CLI entry point for classifier experiments."""
    parser = argparse.ArgumentParser(description="Train baseline & augmented CNNs")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Limit samples per class (useful for quick demos)",
    )
    args = parser.parse_args()
    train_classifiers(epochs=args.epochs, max_per_class=args.max_per_class)


if __name__ == "__main__":
    main()
