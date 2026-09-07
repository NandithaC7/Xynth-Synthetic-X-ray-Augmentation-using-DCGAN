"""Dataset discovery and PyTorch DataLoader helpers."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from backend.app.config import Settings, get_settings
from backend.data.preprocessing import get_dcgan_transforms


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def resolve_class_image_dir(dataset_root: Path, class_name: str) -> Path:
    """
    Resolve the folder that contains images for a class.

    Supports both flat layouts (``Class/*.png``) and the Kaggle radiography
    layout (``Class/images/*.png``).
    """
    class_dir = dataset_root / class_name
    images_subdir = class_dir / "images"
    if images_subdir.is_dir():
        return images_subdir
    return class_dir


def list_image_paths(directory: Path) -> list[Path]:
    """Return sorted image paths under ``directory`` (non-recursive)."""
    if not directory.exists():
        return []
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def get_dataset_stats(settings: Settings | None = None) -> dict:
    """Return per-class image counts and totals for the configured dataset."""
    settings = settings or get_settings()
    root = settings.dataset_dir
    counts: dict[str, int] = {}
    for name in settings.class_names:
        paths = list_image_paths(resolve_class_image_dir(root, name))
        counts[name] = len(paths)
    return {
        "dataset_path": str(root),
        "exists": root.exists(),
        "classes": counts,
        "total": sum(counts.values()),
        "target_class": settings.target_class,
        "target_count": counts.get(settings.target_class, 0),
    }


class XrayFolderDataset(Dataset):
    """Single-class or multi-class folder dataset of chest X-ray images."""

    def __init__(
        self,
        root: Path,
        class_names: tuple[str, ...] | list[str],
        transform: Callable | None = None,
        target_class: str | None = None,
    ) -> None:
        """
        Build an index of image paths.

        If ``target_class`` is set, only that class is loaded (label 0).
        Otherwise labels follow the order of ``class_names``.
        """
        self.root = Path(root)
        self.class_names = list(class_names)
        self.transform = transform
        self.target_class = target_class
        self.samples: list[tuple[Path, int]] = []

        if target_class is not None:
            img_dir = resolve_class_image_dir(self.root, target_class)
            for path in list_image_paths(img_dir):
                self.samples.append((path, 0))
            self.label_names = [target_class]
        else:
            for label, name in enumerate(self.class_names):
                img_dir = resolve_class_image_dir(self.root, name)
                for path in list_image_paths(img_dir):
                    self.samples.append((path, label))
            self.label_names = self.class_names

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def class_counts(self) -> dict[str, int]:
        """Return sample counts keyed by class name."""
        counter = Counter(label for _, label in self.samples)
        return {self.label_names[i]: counter.get(i, 0) for i in range(len(self.label_names))}


def create_dcgan_dataloader(
    settings: Settings | None = None,
    shuffle: bool = True,
) -> DataLoader:
    """DataLoader of target-class images for DCGAN training."""
    settings = settings or get_settings()
    transform = get_dcgan_transforms(settings.image_size)
    dataset = XrayFolderDataset(
        root=settings.dataset_dir,
        class_names=settings.class_names,
        transform=transform,
        target_class=settings.target_class,
    )
    if len(dataset) == 0:
        raise FileNotFoundError(
            f"No images found for class '{settings.target_class}' under {settings.dataset_dir}"
        )
    return DataLoader(
        dataset,
        batch_size=settings.batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=True,
    )


def create_classifier_datasets(
    root: Path,
    class_names: tuple[str, ...] | list[str],
    image_size: int,
    train_indices: list[int] | None = None,
    test_indices: list[int] | None = None,
    transform: transforms.Compose | None = None,
) -> tuple[Dataset, Dataset | None]:
    """
    Build full (or train/test subset) classifier datasets from a folder root.

    When indices are provided, returns train and test Subsets sharing the same
    underlying sample order for reproducible baseline vs augmented splits.
    """
    from backend.data.preprocessing import get_classifier_transforms

    transform = transform or get_classifier_transforms(image_size)
    full = XrayFolderDataset(root=root, class_names=class_names, transform=transform)
    if train_indices is None:
        return full, None
    train_ds = Subset(full, train_indices)
    test_ds = Subset(full, test_indices or [])
    return train_ds, test_ds
