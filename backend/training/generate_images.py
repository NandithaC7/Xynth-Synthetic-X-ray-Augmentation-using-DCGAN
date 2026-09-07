"""Generate synthetic X-ray images from a trained DCGAN generator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.utils import save_image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import get_settings
from backend.app.utils import ensure_dirs, get_device, save_image_grid, save_json
from backend.models.dcgan import Generator


def load_generator(checkpoint_path: Path, latent_dim: int, device: torch.device) -> Generator:
    """Load generator weights from a DCGAN checkpoint."""
    net_g = Generator(latent_dim=latent_dim).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    net_g.load_state_dict(state["generator"])
    net_g.eval()
    return net_g


def generate_images(
    num_images: int | None = None,
    checkpoint: str | None = None,
) -> dict:
    """
    Sample synthetic PNGs, save individual files and preview grids.

    Returns a summary with output paths and counts.
    """
    settings = get_settings()
    num_images = num_images or settings.num_generated
    device = get_device()

    ckpt_path = Path(checkpoint) if checkpoint else settings.checkpoints_dir / "dcgan_latest.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. Train DCGAN first."
        )

    out_dir = settings.generated_dir / "samples"
    ensure_dirs(out_dir, settings.generated_dir)

    net_g = load_generator(ckpt_path, settings.latent_dim, device)

    saved_paths: list[str] = []
    batch_size = min(64, num_images)
    remaining = num_images
    index = 0

    with torch.no_grad():
        while remaining > 0:
            n = min(batch_size, remaining)
            noise = torch.randn(n, settings.latent_dim, 1, 1, device=device)
            fakes = net_g(noise).cpu()
            for i in range(n):
                # Map [-1, 1] → [0, 1] for PNG
                img = (fakes[i].clamp(-1, 1) + 1) / 2.0
                path = out_dir / f"synthetic_{settings.target_class}_{index:05d}.png"
                save_image(img, path)
                saved_paths.append(str(path))
                index += 1
            remaining -= n

        # Preview collage of up to 64 samples
        preview_noise = torch.randn(
            min(64, num_images), settings.latent_dim, 1, 1, device=device
        )
        preview = net_g(preview_noise).cpu()
        collage_path = settings.generated_dir / "preview_collage.png"
        save_image_grid(preview, collage_path, nrow=8)

        # Also keep a smaller 4x4 grid for the UI
        small = preview[:16]
        small_path = settings.generated_dir / "preview_4x4.png"
        save_image_grid(small, small_path, nrow=4)

    summary = {
        "status": "completed",
        "num_generated": num_images,
        "target_class": settings.target_class,
        "checkpoint": str(ckpt_path),
        "samples_dir": str(out_dir),
        "collage": str(collage_path),
        "preview_4x4": str(small_path),
        "sample_paths": saved_paths[:32],
    }
    save_json(summary, settings.metrics_dir / "generation.json")
    print(f"Generated {num_images} images → {out_dir}")
    return summary


def build_augmented_dataset(
    num_synthetic: int | None = None,
) -> dict:
    """
    Merge real images with generated samples for the target class only.

    Writes to ``augmented_dataset/`` without modifying the original dataset.
    """
    settings = get_settings()
    from backend.app.utils import copy_tree_files
    from backend.data.dataset_loader import list_image_paths, resolve_class_image_dir

    num_synthetic = num_synthetic or settings.num_generated
    samples_dir = settings.generated_dir / "samples"
    if not samples_dir.exists() or not any(samples_dir.iterdir()):
        generate_images(num_images=num_synthetic)

    aug_root = settings.augmented_dir
    ensure_dirs(aug_root)

    copied: dict[str, int] = {}
    for class_name in settings.class_names:
        src = resolve_class_image_dir(settings.dataset_dir, class_name)
        dst = aug_root / class_name
        # Clear previous augmented class folder
        if dst.exists():
            for f in dst.iterdir():
                if f.is_file():
                    f.unlink()
        count = copy_tree_files(src, dst)
        copied[class_name] = count

    # Append generated images only to the target class
    target_dst = aug_root / settings.target_class
    ensure_dirs(target_dst)
    synthetic_paths = list_image_paths(samples_dir)[:num_synthetic]
    added = 0
    for path in synthetic_paths:
        dest = target_dst / f"gen_{path.name}"
        Image.open(path).convert("L").save(dest)
        added += 1
    copied[settings.target_class] = copied.get(settings.target_class, 0) + added

    summary = {
        "status": "completed",
        "augmented_path": str(aug_root),
        "counts": copied,
        "synthetic_added": added,
        "target_class": settings.target_class,
    }
    save_json(summary, settings.metrics_dir / "augmented_dataset.json")
    print(f"Augmented dataset ready at {aug_root}")
    return summary


def main() -> None:
    """CLI for image generation and optional augmented-dataset build."""
    parser = argparse.ArgumentParser(description="Generate synthetic X-rays")
    parser.add_argument("--num", type=int, default=None, help="Number of images")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--build-augmented",
        action="store_true",
        help="Also build augmented_dataset/",
    )
    args = parser.parse_args()
    generate_images(num_images=args.num, checkpoint=args.checkpoint)
    if args.build_augmented:
        build_augmented_dataset(num_synthetic=args.num)


if __name__ == "__main__":
    main()
