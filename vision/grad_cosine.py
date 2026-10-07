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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--populations", type=int, nargs="+", default=[16, 64, 256, 1024])
    p.add_argument("--site", default="head")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--draw-chunk", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
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
    for population in args.populations:
        dust = ForwardOnlyDUST(
            model,
            sites=[args.site],
            sigma=args.sigma,
            population=population,
            draw_chunk=args.draw_chunk,
            seed=args.seed + 1000,
        )
        try:
            est, _ = dust.estimate_site_gradient(args.site, images, targets, capture=True)
        finally:
            dust.close()
        print(
            f"population={population:5d} cosine={cosine(est, true_grad):.4f} "
            f"norm_ratio={(est.norm() / true_grad.norm().clamp_min(1e-12)).item():.4f}"
        )


if __name__ == "__main__":
    main()
