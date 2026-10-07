from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import SyntheticInstancesDataset, collate_batch
from model import TinyInstanceTransformer
from task import evaluate, segmentation_loss_per_sample


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--output", type=Path, default=Path("runs/vision-stardist-bp.pt"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print(f"device={device}")
    train_ds = SyntheticInstancesDataset(length=max(args.steps * args.batch_size, 4096), seed=1000)
    val_ds = SyntheticInstancesDataset(length=128, seed=900000)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_batch)

    model = TinyInstanceTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    model.train()
    iterator = iter(train_loader)
    for step in range(1, args.steps + 1):
        try:
            images, targets = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            images, targets = next(iterator)
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(images)
        loss = segmentation_loss_per_sample(pred, targets).mean()
        loss.backward()
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == args.steps:
            metrics = evaluate(model, val_loader, device)
            print(f"step={step:4d} train={loss.item():.4f} val={metrics['loss']:.4f} dice={metrics['dice']:.3f} pq={metrics['pq']:.3f} ray_mae={metrics['ray_mae']:.3f}")
            model.train()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    torch.save({"model": model.state_dict(), "args": metadata}, args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
