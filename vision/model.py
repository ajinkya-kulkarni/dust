from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def rms_norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.shape[-1],))


def fixed_2d_position_embedding(grid: int, dim: int) -> torch.Tensor:
    if dim % 4:
        raise ValueError("dim must be divisible by 4")
    y, x = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    quarter = dim // 4
    freq = torch.exp(-math.log(10000.0) * torch.arange(quarter) / max(quarter - 1, 1))
    x = x.reshape(-1, 1).float() * freq.reshape(1, -1)
    y = y.reshape(-1, 1).float() * freq.reshape(1, -1)
    return torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, t, d)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.attn = Attention(dim, heads)
        self.fc1 = nn.Linear(dim, mlp_ratio * dim, bias=False)
        self.fc2 = nn.Linear(mlp_ratio * dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(rms_norm(x))
        x = x + self.fc2(F.gelu(self.fc1(rms_norm(x))))
        return x


class TinyInstanceTransformer(nn.Module):
    """Tiny all-Linear transformer with a StarDist-like dense output head."""

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        dim: int = 64,
        depth: int = 2,
        heads: int = 4,
        n_rays: int = 16,
    ) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_rays = n_rays
        self.grid = image_size // patch_size
        self.patch_area = patch_size * patch_size
        self.patch_embed = nn.Linear(self.patch_area, dim, bias=False)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.head = nn.Linear(dim, self.patch_area * (1 + n_rays), bias=False)
        self.register_buffer("pos", fixed_2d_position_embedding(self.grid, dim), persistent=False)

    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        patches = F.unfold(images, kernel_size=self.patch_size, stride=self.patch_size)
        return patches.transpose(1, 2)

    def unpatchify(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, _ = raw.shape
        p = self.patch_size
        g = self.grid
        channels = 1 + self.n_rays
        raw = raw.reshape(b, g, g, p, p, channels)
        raw = raw.permute(0, 1, 3, 2, 4, 5).reshape(
            b, self.image_size, self.image_size, channels
        )
        obj_logits = raw[..., 0]
        rays = F.softplus(raw[..., 1:]).permute(0, 3, 1, 2)
        return obj_logits, rays

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.patch_embed(self.patchify(images))
        x = x + self.pos.to(dtype=x.dtype, device=x.device).unsqueeze(0)
        for block in self.blocks:
            x = block(x)
        raw = self.head(rms_norm(x))
        obj_logits, rays = self.unpatchify(raw)
        return {"obj_logits": obj_logits, "rays": rays, "token_raw": raw}
