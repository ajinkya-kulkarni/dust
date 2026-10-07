from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


@dataclass
class BatchTargets:
    instances: torch.Tensor  # [B,H,W] integer instance ids
    foreground: torch.Tensor  # [B,H,W] float
    offsets: torch.Tensor  # [B,2,H,W], (dy, dx) normalized by image extent

    def to(self, device: torch.device | str) -> "BatchTargets":
        return BatchTargets(
            instances=self.instances.to(device),
            foreground=self.foreground.to(device),
            offsets=self.offsets.to(device),
        )


class SyntheticInstancesDataset(Dataset):
    """Deterministic synthetic touching-ellipse instance segmentation dataset."""

    def __init__(
        self,
        length: int = 2048,
        image_size: int = 32,
        min_instances: int = 1,
        max_instances: int = 5,
        seed: int = 0,
    ) -> None:
        self.length = length
        self.image_size = image_size
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        g = torch.Generator().manual_seed(self.seed + index)
        h = w = self.image_size
        yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        instances = torch.zeros(h, w, dtype=torch.long)
        image = torch.rand(h, w, generator=g) * 0.06

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
            intensity = 0.55 + 0.4 * torch.rand((), generator=g)
            texture = 0.07 * torch.randn(h, w, generator=g)
            image[visible] = (intensity + texture[visible]).clamp(0, 1)

        image = F.avg_pool2d(image[None, None], 3, stride=1, padding=1)[0, 0]
        image = (image + 0.025 * torch.randn(h, w, generator=g)).clamp(0, 1)
        foreground, offsets = targets_from_instances(instances)
        return image.unsqueeze(0), instances, foreground, offsets


def targets_from_instances(instances: torch.Tensor):
    h, w = instances.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )
    foreground = (instances > 0).float()
    offsets = torch.zeros(2, h, w, dtype=torch.float32)
    for idx in range(1, int(instances.max()) + 1):
        mask = instances == idx
        if not mask.any():
            continue
        cy = yy[mask].mean()
        cx = xx[mask].mean()
        offsets[0, mask] = (cy - yy[mask]) / max(h - 1, 1)
        offsets[1, mask] = (cx - xx[mask]) / max(w - 1, 1)
    return foreground, offsets


def collate_batch(batch) -> tuple[torch.Tensor, BatchTargets]:
    images, instances, foreground, offsets = zip(*batch)
    return torch.stack(images), BatchTargets(
        instances=torch.stack(instances),
        foreground=torch.stack(foreground),
        offsets=torch.stack(offsets),
    )
