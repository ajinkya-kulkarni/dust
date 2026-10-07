from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import DSB2018Dataset, collate_batch
from dust_vision import ForwardOnlyDUST
from model import TinyInstanceTransformer
from task import segmentation_loss_per_sample


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def norm_ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.norm() / b.norm().clamp_min(1e-12)).item()


def split_head_rows(
    grad: torch.Tensor,
    output_positions_per_token: int,
    n_rays: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    grad = grad.reshape(output_positions_per_token, 1 + n_rays, grad.shape[-1])
    return grad[:, 0], grad[:, 1:]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--init", type=Path, default=None, help="Optional BP checkpoint to evaluate gradients at.")
    p.add_argument("--split", default="train")
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--populations", type=int, nargs="+", default=[16, 64, 256])
    p.add_argument("--site", default="head")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--draw-chunk", type=int, default=8)
    p.add_argument("--head-credit", choices=["split", "global"], default="split")
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--output-stride", type=int, default=4)
    p.add_argument("--grid-offset", type=int, default=2)
    p.add_argument("--dim", type=int, default=96)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--n-rays", type=int, default=32)
    p.add_argument("--ray-scale", type=float, default=16.0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print(f"device={device} head_credit={args.head_credit}")

    ds = DSB2018Dataset(
        args.data_dir,
        split=args.split,
        image_size=args.image_size,
        n_rays=args.n_rays,
        output_stride=args.output_stride,
        grid_offset=args.grid_offset,
        ray_scale=args.ray_scale,
        cache_targets=True,
    )
    if args.sample_index < 0 or args.sample_index >= len(ds):
        raise IndexError(f"sample-index {args.sample_index} outside 0..{len(ds)-1}")
    indices = [(args.sample_index + i) % len(ds) for i in range(args.batch_size)]
    batch = [ds[i] for i in indices]
    images, targets = collate_batch(batch)
    images, targets = images.to(device), targets.to(device)

    model = TinyInstanceTransformer(
        image_size=args.image_size,
        patch_size=args.patch_size,
        output_stride=args.output_stride,
        grid_offset=args.grid_offset,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        n_rays=args.n_rays,
        ray_scale=args.ray_scale,
    ).to(device)

    if args.init is not None:
        state = torch.load(args.init, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        print(f"loaded checkpoint={args.init}")

    model.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    loss = segmentation_loss_per_sample(model(images), targets).mean()
    loss.backward()
    module = dict(model.named_modules())[args.site]
    true_grad = module.weight.grad.detach().float().clone()
    model.zero_grad(set_to_none=True)
    model.requires_grad_(False)

    print(
        f"site={args.site} exact_grad_norm={true_grad.norm().item():.6f} "
        f"sample_index={args.sample_index}"
    )
    if args.site == "head":
        true_obj, true_rays = split_head_rows(
            true_grad, model.output_positions_per_token, model.n_rays
        )
        print(
            f"  exact_obj_norm={true_obj.norm().item():.6f} "
            f"exact_rays_norm={true_rays.norm().item():.6f}"
        )

    for population in args.populations:
        dust = ForwardOnlyDUST(
            model,
            sites=[args.site],
            sigma=args.sigma,
            population=population,
            draw_chunk=args.draw_chunk,
            seed=args.seed + 1000,
            split_head_credit=args.head_credit == "split",
        )
        try:
            est, _ = dust.estimate_site_gradient(args.site, images, targets, capture=True)
        finally:
            dust.close()

        line = (
            f"population={population:5d} cosine={cosine(est, true_grad):.4f} "
            f"norm_ratio={norm_ratio(est, true_grad):.4f}"
        )
        if args.site == "head":
            est_obj, est_rays = split_head_rows(
                est, model.output_positions_per_token, model.n_rays
            )
            line += (
                f" obj_cos={cosine(est_obj, true_obj):.4f}"
                f" obj_norm={norm_ratio(est_obj, true_obj):.4f}"
                f" ray_cos={cosine(est_rays, true_rays):.4f}"
                f" ray_norm={norm_ratio(est_rays, true_rays):.4f}"
            )
        print(line)


if __name__ == "__main__":
    main()
