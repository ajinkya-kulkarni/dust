from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import SyntheticInstancesDataset, collate_batch
from model import TinyInstanceTransformer
from task import evaluate
from dust_vision import ForwardOnlyDUST


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--population", type=int, default=64)
    p.add_argument("--draw-chunk", type=int, default=8)
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--mode", choices=["full", "head"], default="full")
    p.add_argument("--init", type=Path, help="Optional checkpoint from train_bp.py; recommended for --mode head")
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
    p.add_argument("--output", type=Path, default=Path("runs/vision-dust.pt"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print(f"device={device}")
    train_ds = SyntheticInstancesDataset(length=max(args.steps * args.batch_size, 1024), seed=2000)
    val_ds = SyntheticInstancesDataset(length=64, seed=900000)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_batch)

    model = TinyInstanceTransformer().to(device)
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
    )
    params = [p for m in dust.modules.values() for p in m.parameters()]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

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
            if step == 1 or step % 10 == 0 or step == args.steps:
                metrics = evaluate(model, val_loader, device)
                print(f"step={step:4d} train={loss:.4f} val={metrics['loss']:.4f} dice={metrics['dice']:.3f} pq={metrics['pq']:.3f}")
    finally:
        dust.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    torch.save({"model": model.state_dict(), "args": metadata}, args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
