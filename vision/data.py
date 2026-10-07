from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
import torch
from scipy.ndimage import distance_transform_edt
from torch.utils.data import Dataset


@dataclass
class BatchTargets:
    instances: torch.Tensor  # [B,256,256] contiguous instance ids
    objectness: torch.Tensor  # [B,64,64] center-weighted target in [0,1]
    rays: torch.Tensor  # [B,R,64,64] radial distances normalized by ray_scale

    def to(self, device: torch.device | str) -> "BatchTargets":
        return BatchTargets(
            instances=self.instances.to(device),
            objectness=self.objectness.to(device),
            rays=self.rays.to(device),
        )


def relabel_instances(labels: np.ndarray) -> np.ndarray:
    """Relabel positive, possibly sparse/global ids to contiguous 1..N per patch."""
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError(f"Expected 2-D instance mask, got shape={labels.shape}")
    values = np.unique(labels)
    values = values[values > 0]
    out = np.zeros(labels.shape, dtype=np.int32)
    for new_id, old_id in enumerate(values.tolist(), start=1):
        out[labels == old_id] = new_id
    return out


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Convert TIFF data to one robustly normalized grayscale channel."""
    x = np.asarray(image)
    if x.ndim == 3:
        if x.shape[-1] <= 4:
            x = x[..., :3].mean(axis=-1)
        elif x.shape[0] <= 4:
            x = x[:3].mean(axis=0)
        else:
            raise ValueError(f"Cannot infer channel axis for TIFF shape={x.shape}")
    if x.ndim != 2:
        raise ValueError(f"Expected 2-D image after channel reduction, got shape={x.shape}")

    x = x.astype(np.float32, copy=False)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.float32)
    x = np.where(finite, x, 0.0)
    lo = float(np.percentile(x, 1.0))
    hi = float(np.percentile(x, 99.8))
    if hi <= lo + 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def normalized_instance_distance(labels: np.ndarray) -> np.ndarray:
    """Per-instance EDT normalized so each object's interior maximum is 1."""
    out = np.zeros(labels.shape, dtype=np.float32)
    for idx in range(1, int(labels.max()) + 1):
        ys, xs = np.nonzero(labels == idx)
        if ys.size == 0:
            continue
        y0, y1 = max(int(ys.min()) - 1, 0), min(int(ys.max()) + 2, labels.shape[0])
        x0, x1 = max(int(xs.min()) - 1, 0), min(int(xs.max()) + 2, labels.shape[1])
        mask = labels[y0:y1, x0:x1] == idx
        padded = np.pad(mask, 1, mode="constant", constant_values=False)
        distance = distance_transform_edt(padded)[1:-1, 1:-1].astype(np.float32)
        peak = float(distance.max())
        if peak > 0:
            distance /= peak
        patch = out[y0:y1, x0:x1]
        patch[mask] = distance[mask]
    return out


def _grid_coordinates(size: int, stride: int, offset: int) -> np.ndarray:
    coords = offset + np.arange(size // stride, dtype=np.int64) * stride
    if coords[-1] >= size:
        raise ValueError(
            f"Grid coordinate {coords[-1]} outside image size {size}; "
            f"check stride={stride}, offset={offset}"
        )
    return coords


def radial_targets(
    labels: np.ndarray,
    n_rays: int,
    output_stride: int,
    grid_offset: int,
    ray_scale: float,
    step_px: float = 0.5,
) -> np.ndarray:
    """Ray-march from foreground output-grid points until the instance id changes."""
    h, w = labels.shape
    gy = _grid_coordinates(h, output_stride, grid_offset)
    gx = _grid_coordinates(w, output_stride, grid_offset)
    sampled = labels[np.ix_(gy, gx)]
    fy, fx = np.nonzero(sampled > 0)
    rays = np.zeros((n_rays, gy.size, gx.size), dtype=np.float32)
    if fy.size == 0:
        return rays

    oy = gy[fy].astype(np.float32)
    ox = gx[fx].astype(np.float32)
    instance_ids = sampled[fy, fx]
    max_dist = math.hypot(h, w) + step_px

    for ray_idx in range(n_rays):
        theta = 2.0 * math.pi * ray_idx / n_rays
        dy = math.sin(theta)
        dx = math.cos(theta)
        active = np.ones(fy.size, dtype=bool)
        distance = np.full(fy.size, max_dist, dtype=np.float32)

        t = step_px
        while active.any() and t <= max_dist:
            yi = np.rint(oy + dy * t).astype(np.int64)
            xi = np.rint(ox + dx * t).astype(np.int64)
            inside = (yi >= 0) & (yi < h) & (xi >= 0) & (xi < w)

            same = np.zeros(fy.size, dtype=bool)
            valid = active & inside
            same[valid] = labels[yi[valid], xi[valid]] == instance_ids[valid]

            exited = active & ~same
            if exited.any():
                distance[exited] = max(step_px * 0.5, t - step_px * 0.5)
            active &= same
            t += step_px

        rays[ray_idx, fy, fx] = distance / ray_scale

    return rays


def stardist_targets_from_instances(
    labels: np.ndarray,
    n_rays: int = 32,
    output_stride: int = 4,
    grid_offset: int = 2,
    ray_scale: float = 16.0,
) -> tuple[np.ndarray, np.ndarray]:
    probability_full = normalized_instance_distance(labels)
    coords_y = _grid_coordinates(labels.shape[0], output_stride, grid_offset)
    coords_x = _grid_coordinates(labels.shape[1], output_stride, grid_offset)
    objectness = probability_full[np.ix_(coords_y, coords_x)].astype(np.float32)
    rays = radial_targets(
        labels,
        n_rays=n_rays,
        output_stride=output_stride,
        grid_offset=grid_offset,
        ray_scale=ray_scale,
    )
    return objectness, rays


class DSB2018Dataset(Dataset):
    """Paired TIFF + NPY DSB2018 patches with cached StarDist-style targets."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        image_size: int = 256,
        n_rays: int = 32,
        output_stride: int = 4,
        grid_offset: int = 2,
        ray_scale: float = 16.0,
        cache_targets: bool = True,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.n_rays = n_rays
        self.output_stride = output_stride
        self.grid_offset = grid_offset
        self.ray_scale = ray_scale
        self.cache_targets = cache_targets

        split_dir = self.root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")

        masks = sorted(split_dir.glob("*.npy"))
        if not masks:
            raise FileNotFoundError(f"No .npy masks found in {split_dir}")

        self.samples: list[tuple[Path, Path]] = []
        missing: list[str] = []
        for mask_path in masks:
            image_path = mask_path.with_suffix(".tif")
            if not image_path.exists():
                missing.append(image_path.name)
            else:
                self.samples.append((image_path, mask_path))
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} TIFF pair(s) missing in {split_dir}; first={missing[0]}"
            )

        cache_tag = (
            f"v2_size{image_size}_s{output_stride}_o{grid_offset}"
            f"_r{n_rays}_scale{ray_scale:g}"
        )
        self.cache_dir = self.root / ".dust_stardist_cache" / cache_tag / split
        if self.cache_targets:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_mask(self, path: Path) -> np.ndarray:
        labels = relabel_instances(np.load(path, allow_pickle=False))
        if labels.shape != (self.image_size, self.image_size):
            raise ValueError(
                f"{path.name}: expected {(self.image_size, self.image_size)}, got {labels.shape}"
            )
        return labels

    def _target_path(self, mask_path: Path) -> Path:
        return self.cache_dir / f"{mask_path.stem}.npz"

    def _targets(self, labels: np.ndarray, mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
        target_path = self._target_path(mask_path)
        if self.cache_targets and target_path.exists():
            with np.load(target_path, allow_pickle=False) as cached:
                return (
                    cached["objectness"].astype(np.float32),
                    cached["rays"].astype(np.float32),
                )

        objectness, rays = stardist_targets_from_instances(
            labels,
            n_rays=self.n_rays,
            output_stride=self.output_stride,
            grid_offset=self.grid_offset,
            ray_scale=self.ray_scale,
        )

        if self.cache_targets:
            tmp_path = target_path.with_suffix(".tmp")
            with tmp_path.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    objectness=objectness.astype(np.float16),
                    rays=rays.astype(np.float16),
                )
            os.replace(tmp_path, target_path)

        return objectness, rays

    def __getitem__(self, index: int):
        image_path, mask_path = self.samples[index]
        image = normalize_image(tifffile.imread(image_path))
        if image.shape != (self.image_size, self.image_size):
            raise ValueError(
                f"{image_path.name}: expected {(self.image_size, self.image_size)}, got {image.shape}"
            )
        labels = self._load_mask(mask_path)
        objectness, rays = self._targets(labels, mask_path)
        return (
            torch.from_numpy(image[None]).float(),
            torch.from_numpy(labels.astype(np.int64, copy=False)),
            torch.from_numpy(objectness).float(),
            torch.from_numpy(rays).float(),
        )


def collate_batch(batch) -> tuple[torch.Tensor, BatchTargets]:
    images, instances, objectness, rays = zip(*batch)
    return torch.stack(images), BatchTargets(
        instances=torch.stack(instances),
        objectness=torch.stack(objectness),
        rays=torch.stack(rays),
    )
