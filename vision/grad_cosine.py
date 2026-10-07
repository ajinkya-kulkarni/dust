from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from data import SyntheticInstancesDataset, collate_batch
from model import TinyInstanceTransformer
from task import segmentation_loss_per_sample
from dust_vision import ForwardOnlyDUST


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def norm_ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.norm() / b.norm().clamp_min(1e-12)).item()


def split_head_rows(
    grad: torch.Tensor,
    patch_area: int,
    n_rays: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split flattened dense-head rows into objectness and radial-output rows."""
    grad = grad.reshape(patch_area, 1 + n_rays, grad.shape[-1])
    return grad[:, 0], grad[:, 1:]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--populations", type=int, nargs="+", default=[16, 64, 256, 1024])
    p.add_argument("--site", default="head")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--draw-chunk", type=int, default=8)
    p.add_argument("--head-credit", choices=["split", "global"], default="split")
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print(f"device={device} head_credit={args.head_credit}")
    loader = DataLoader(
        SyntheticInstancesDataset(length=args.batch_size, seed=424242),
        batch_size=args.batch_size,
        collate_fn=collate_batch,
    )
    images, targets = next(iter(loader))
    images, targets = images.to(device), targets.to(device)
    model = TinyInstanceTransformer().to(device)

    model.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    loss = segmentation_loss_per_sample(model(images), targets).mean()
    loss.backward()
    module = dict(model.named_modules())[args.site]
    true_grad = module.weight.grad.detach().float().clone()
    model.zero_grad(set_to_none=True)
    model.requires_grad_(False)

    print(f"site={args.site} exact_grad_norm={true_grad.norm().item():.6f}")
    if args.site == "head":
        true_obj, true_rays = split_head_rows(true_grad, model.patch_area, model.n_rays)
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
            est_obj, est_rays = split_head_rows(est, model.patch_area, model.n_rays)
            line += (
                f" obj_cos={cosine(est_obj, true_obj):.4f}"
                f" obj_norm={norm_ratio(est_obj, true_obj):.4f}"
                f" ray_cos={cosine(est_rays, true_rays):.4f}"
                f" ray_norm={norm_ratio(est_rays, true_rays):.4f}"
            )
        print(line)


if __name__ == "__main__":
    main()
