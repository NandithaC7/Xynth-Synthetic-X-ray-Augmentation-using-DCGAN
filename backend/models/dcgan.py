"""DCGAN Generator and Discriminator (Radford et al., 2016)."""

from __future__ import annotations

import torch
from torch import nn


def weights_init(module: nn.Module) -> None:
    """Initialize Conv / BatchNorm layers as recommended by the DCGAN paper."""
    classname = module.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)


class Generator(nn.Module):
    """
    DCGAN generator: latent vector z → 1×64×64 grayscale image.

    Architecture follows Radford, Metz & Chintala (2016) for 64×64 output.
    """

    def __init__(self, latent_dim: int = 100, ngf: int = 64, nc: int = 1) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.main = nn.Sequential(
            # input: Z latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state: (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state: (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state: (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
            # output: nc x 64 x 64
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """Map latent noise to a synthetic image batch."""
        return self.main(noise)


class Discriminator(nn.Module):
    """
    DCGAN discriminator: 1×64×64 image → real/fake probability.

    Uses Conv2d + BatchNorm + LeakyReLU, ending with Sigmoid.
    """

    def __init__(self, ndf: int = 64, nc: int = 1) -> None:
        super().__init__()
        self.main = nn.Sequential(
            # input: nc x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state: (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state: (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state: (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state: (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Score a batch of images as real (1) or fake (0)."""
        return self.main(image).view(-1, 1).squeeze(1)


def build_dcgan(latent_dim: int = 100, device: torch.device | None = None) -> tuple[Generator, Discriminator]:
    """Construct and initialize Generator and Discriminator on ``device``."""
    device = device or torch.device("cpu")
    net_g = Generator(latent_dim=latent_dim).to(device)
    net_d = Discriminator().to(device)
    net_g.apply(weights_init)
    net_d.apply(weights_init)
    return net_g, net_d
