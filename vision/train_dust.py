from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import DSB2018Dataset, collate_batch
from dust_vision import ForwardOnlyDUST
from model import TinyInstanceTransformer
from task import evaluate


def add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--output-stride", type=int, default=4)
    p.add_argument("--grid-offset", type=int, default=2)
    p.add_argument("--dim", type=int, default=96)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--n-rays", type=int, default=32)
    p.add_argument("--ray-scale", type=float, default=16.0)


def build_model(args) -> TinyInstanceTransformer:
    return TinyInstanceTransformer(
        image_size=args.image_size,
        patch_size=args.patch_size,
        output_stride=args.output_stride,
        grid_offset=args.grid_offset,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        n_rays=args.n_rays,
        ray_scale=args.ray_scale,
    )


def build_dataset(args, split: str) -> DSB2018Dataset:
    return DSB2018Dataset(
        args.data_dir,
        split=split,
        image_size=args.image_size,
        n_rays=args.n_rays,
        output_stride=args.output_stride,
        grid_offset=args.grid_offset,
        ray_scale=args.ray_scale,
        cache_targets=not args.no_target_cache,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--val-batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--population", type=int, default=64)
    p.add_argument("--draw-chunk", type=int, default=8)
    p.add_argument("--head-credit", choices=["split", "global"], default="split")
    p.add_argument(
        "--spatial-credit",
        choices=["local", "global"],
        default="local",
        help="Use exact token-local loss credit at eligible late sites.",
    )
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--mode", choices=["full", "head"], default="full")
    p.add_argument("--init", type=Path)
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--output", type=Path, default=Path("runs/dsb2018-stardist-dust.pt"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-target-cache", action="store_true")
    add_model_args(p)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print(
        f"device={device} head_credit={args.head_credit} "
        f"spatial_credit={args.spatial_credit} image={args.image_size} "
        f"patch={args.patch_size} stride={args.output_stride} rays={args.n_rays}"
    )

    train_ds = build_dataset(args, "train")
    val_ds = build_dataset(args, "val")
    generator = torch.Generator().manual_seed(args.seed + 17)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
    )

    model = build_model(args).to(device)
    if args.init:
        state = torch.load(args.init, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
    model.requires_grad_(False)

    sites = ["head"] if args.mode == "head" else None
    dust = ForwardOnlyDUST(
        model,
        sites=sites,
        sigma=args.sigma,
        population=args.population,
        draw_chunk=args.draw_chunk,
        seed=args.seed + 123,
        split_head_credit=args.head_credit == "split",
        token_local_credit=args.spatial_credit == "local",
    )
    params = [p for module in dust.modules.values() for p in module.parameters()]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"train={len(train_ds)} val={len(val_ds)} tokens={model.grid ** 2} "
        f"output_grid={model.output_grid}x{model.output_grid} "
        f"head_width={model.head.out_features} params={n_params:,} "
        f"sites={len(dust.modules)}"
    )

    iterator = iter(train_loader)
    try:
        for step in range(1, args.steps + 1):
            try:
                images, targets = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                images, targets = next(iterator)

            images, targets = images.to(device), targets.to(device)
            model.train()
            loss = dust.step(images, targets, optimizer)

            if step == 1 or step % args.eval_every == 0 or step == args.steps:
                metrics = evaluate(model, val_loader, device)
                print(
                    f"step={step:5d} train={loss:.4f} val={metrics['loss']:.4f} "
                    f"dice={metrics['dice']:.3f} pq={metrics['pq']:.3f} "
                    f"ray_mae_px={metrics['ray_mae']:.3f}"
                )
    finally:
        dust.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    torch.save(
        {
            "model": model.state_dict(),
            "args": metadata,
            "model_config": {
                "image_size": args.image_size,
                "patch_size": args.patch_size,
                "output_stride": args.output_stride,
                "grid_offset": args.grid_offset,
                "dim": args.dim,
                "depth": args.depth,
                "heads": args.heads,
                "n_rays": args.n_rays,
                "ray_scale": args.ray_scale,
            },
        },
        args.output,
    )
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
