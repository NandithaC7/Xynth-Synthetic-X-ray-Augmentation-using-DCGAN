"""DCGAN training loop for synthetic chest X-ray generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn, optim

# Ensure project root is on sys.path when run as a script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import get_settings
from backend.app.utils import (
    ensure_dirs,
    get_device,
    save_image_grid,
    save_json,
    save_loss_plot,
)
from backend.data.dataset_loader import create_dcgan_dataloader
from backend.models.dcgan import build_dcgan


def train_dcgan(
    epochs: int | None = None,
    batch_size: int | None = None,
    resume: bool = False,
) -> dict:
    """
    Train DCGAN on the target minority class.

    Returns a summary dict with loss history paths and checkpoint location.
    """
    settings = get_settings()
    epochs = epochs or settings.epochs
    device = get_device()

    ensure_dirs(
        settings.checkpoints_dir,
        settings.plots_dir,
        settings.metrics_dir,
        settings.generated_dir,
    )

    # Temporarily override batch size if provided
    if batch_size is not None:
        object.__setattr__(settings, "batch_size", batch_size)

    dataloader = create_dcgan_dataloader(settings)
    net_g, net_d = build_dcgan(latent_dim=settings.latent_dim, device=device)

    criterion = nn.BCELoss()
    optimizer_d = optim.Adam(
        net_d.parameters(),
        lr=settings.learning_rate,
        betas=(0.5, 0.999),
    )
    optimizer_g = optim.Adam(
        net_g.parameters(),
        lr=settings.learning_rate,
        betas=(0.5, 0.999),
    )

    start_epoch = 0
    g_losses: list[float] = []
    d_losses: list[float] = []
    fixed_noise = torch.randn(64, settings.latent_dim, 1, 1, device=device)

    latest_ckpt = settings.checkpoints_dir / "dcgan_latest.pt"
    if resume and latest_ckpt.exists():
        state = torch.load(latest_ckpt, map_location=device, weights_only=False)
        net_g.load_state_dict(state["generator"])
        net_d.load_state_dict(state["discriminator"])
        optimizer_g.load_state_dict(state["optimizer_g"])
        optimizer_d.load_state_dict(state["optimizer_d"])
        start_epoch = state.get("epoch", 0) + 1
        g_losses = state.get("g_losses", [])
        d_losses = state.get("d_losses", [])

    real_label = 1.0
    fake_label = 0.0

    net_g.train()
    net_d.train()

    for epoch in range(start_epoch, epochs):
        for i, (real_images, _) in enumerate(dataloader):
            real_images = real_images.to(device)
            b_size = real_images.size(0)

            # --- Update Discriminator ---
            net_d.zero_grad()
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = net_d(real_images)
            err_d_real = criterion(output, label)
            err_d_real.backward()

            noise = torch.randn(b_size, settings.latent_dim, 1, 1, device=device)
            fake = net_g(noise)
            label.fill_(fake_label)
            output = net_d(fake.detach())
            err_d_fake = criterion(output, label)
            err_d_fake.backward()
            err_d = err_d_real + err_d_fake
            optimizer_d.step()

            # --- Update Generator ---
            net_g.zero_grad()
            label.fill_(real_label)
            output = net_d(fake)
            err_g = criterion(output, label)
            err_g.backward()
            optimizer_g.step()

            g_losses.append(err_g.item())
            d_losses.append(err_d.item())

            if i % 50 == 0:
                print(
                    f"[{epoch + 1}/{epochs}][{i}/{len(dataloader)}] "
                    f"Loss_D: {err_d.item():.4f} Loss_G: {err_g.item():.4f}"
                )

        # Epoch-end sample grid
        with torch.no_grad():
            samples = net_g(fixed_noise).detach().cpu()
        save_image_grid(
            samples,
            settings.generated_dir / f"epoch_{epoch + 1:03d}_grid.png",
        )

        checkpoint = {
            "epoch": epoch,
            "generator": net_g.state_dict(),
            "discriminator": net_d.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "g_losses": g_losses,
            "d_losses": d_losses,
            "latent_dim": settings.latent_dim,
            "target_class": settings.target_class,
        }
        torch.save(checkpoint, latest_ckpt)

        if (epoch + 1) % settings.checkpoint_every == 0 or (epoch + 1) == epochs:
            torch.save(
                checkpoint,
                settings.checkpoints_dir / f"dcgan_epoch_{epoch + 1:03d}.pt",
            )

    loss_plot = settings.plots_dir / "dcgan_loss.png"
    save_loss_plot(g_losses, d_losses, loss_plot)

    summary = {
        "status": "completed",
        "epochs": epochs,
        "device": str(device),
        "target_class": settings.target_class,
        "num_batches": len(dataloader),
        "final_g_loss": g_losses[-1] if g_losses else None,
        "final_d_loss": d_losses[-1] if d_losses else None,
        "checkpoint": str(latest_ckpt),
        "loss_plot": str(loss_plot),
        "g_losses_tail": g_losses[-100:],
        "d_losses_tail": d_losses[-100:],
    }
    save_json(summary, settings.metrics_dir / "dcgan_training.json")
    # Persist full loss curves for the API / frontend
    save_json(
        {"g_losses": g_losses, "d_losses": d_losses},
        settings.metrics_dir / "dcgan_losses.json",
    )
    print(f"Training complete. Checkpoint: {latest_ckpt}")
    return summary


def main() -> None:
    """CLI entry point for DCGAN training."""
    parser = argparse.ArgumentParser(description="Train DCGAN on target X-ray class")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train_dcgan(epochs=args.epochs, batch_size=args.batch_size, resume=args.resume)


if __name__ == "__main__":
    main()
