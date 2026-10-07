from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


@dataclass
class BatchTargets:
    instances: torch.Tensor  # [B,H,W] integer instance ids
    objectness: torch.Tensor  # [B,H,W] center-weighted object probability target in [0,1]
    rays: torch.Tensor  # [B,R,H,W] radial distances in pixels

    def to(self, device: torch.device | str) -> "BatchTargets":
        return BatchTargets(
            instances=self.instances.to(device),
            objectness=self.objectness.to(device),
            rays=self.rays.to(device),
        )


class SyntheticInstancesDataset(Dataset):
    """Deterministic synthetic touching-ellipse dataset with StarDist-style targets."""

    def __init__(
        self,
        length: int = 2048,
        image_size: int = 32,
        min_instances: int = 1,
        max_instances: int = 5,
        n_rays: int = 16,
        seed: int = 0,
    ) -> None:
        self.length = length
        self.image_size = image_size
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.n_rays = n_rays
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        g = torch.Generator().manual_seed(self.seed + index)
        h = w = self.image_size
        yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        instances = torch.zeros(h, w, dtype=torch.long)
        image = torch.rand(h, w, generator=g) * 0.06
        ellipses: list[tuple[int, float, float, float, float]] = []

        target_n = int(torch.randint(self.min_instances, self.max_instances + 1, (1,), generator=g))
        made = 0
        attempts = 0
        while made < target_n and attempts < target_n * 40:
            attempts += 1
            ry = int(torch.randint(3, 7, (1,), generator=g))
            rx = int(torch.randint(3, 7, (1,), generator=g))
            cy = int(torch.randint(ry + 1, h - ry - 1, (1,), generator=g))
            cx = int(torch.randint(rx + 1, w - rx - 1, (1,), generator=g))
            shape = ((yy - cy).float() / ry).square() + ((xx - cx).float() / rx).square() <= 1.0
            overlap = (shape & (instances > 0)).sum().float()
            if overlap / shape.sum().clamp_min(1) > 0.08:
                continue

            made += 1
            visible = shape & (instances == 0)
            instances[visible] = made
            ellipses.append((made, float(cy), float(cx), float(ry), float(rx)))
            intensity = 0.55 + 0.4 * torch.rand((), generator=g)
            texture = 0.07 * torch.randn(h, w, generator=g)
            image[visible] = (intensity + texture[visible]).clamp(0, 1)

        image = F.avg_pool2d(image[None, None], 3, stride=1, padding=1)[0, 0]
        image = (image + 0.025 * torch.randn(h, w, generator=g)).clamp(0, 1)
        objectness, rays = stardist_targets_from_ellipses(instances, ellipses, self.n_rays)
        return image.unsqueeze(0), instances, objectness, rays


def stardist_targets_from_ellipses(
    instances: torch.Tensor,
    ellipses: list[tuple[int, float, float, float, float]],
    n_rays: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create center-weighted objectness and analytic ellipse-boundary rays."""
    h, w = instances.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )
    objectness = torch.zeros(h, w, dtype=torch.float32)
    rays = torch.zeros(n_rays, h, w, dtype=torch.float32)
    angles = torch.arange(n_rays, dtype=torch.float32) * (2.0 * math.pi / n_rays)
    dy = angles.sin()
    dx = angles.cos()

    for idx, cy, cx, ry, rx in ellipses:
        mask = instances == idx
        if not mask.any():
            continue
        y = yy[mask]
        x = xx[mask]
        y0 = y - cy
        x0 = x - cx

        # StarDist-style object probability: 1 near the center, 0 at the boundary.
        normalized_radius = torch.sqrt((y0 / ry).square() + (x0 / rx).square())
        objectness[mask] = (1.0 - normalized_radius).clamp(0.0, 1.0)

        # Positive intersection of p + t*u with the ellipse boundary for every ray u.
        a = (dy / ry).square() + (dx / rx).square()
        b = 2.0 * (
            y0[:, None] * dy[None, :] / (ry * ry)
            + x0[:, None] * dx[None, :] / (rx * rx)
        )
        c = (y0 / ry).square()[:, None] + (x0 / rx).square()[:, None] - 1.0
        disc = (b.square() - 4.0 * a[None, :] * c).clamp_min(0.0)
        distance = (-b + torch.sqrt(disc)) / (2.0 * a[None, :]).clamp_min(1e-8)
        rays[:, mask] = distance.transpose(0, 1)

    return objectness, rays


def collate_batch(batch) -> tuple[torch.Tensor, BatchTargets]:
    images, instances, objectness, rays = zip(*batch)
    return torch.stack(images), BatchTargets(
        instances=torch.stack(instances),
        objectness=torch.stack(objectness),
        rays=torch.stack(rays),
    )
