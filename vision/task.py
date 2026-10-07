from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import BatchTargets


def segmentation_loss_per_sample(
    prediction: dict[str, torch.Tensor],
    targets: BatchTargets,
    offset_weight: float = 2.0,
) -> torch.Tensor:
    fg_logits = prediction["fg_logits"]
    offsets = prediction["offsets"]
    fg_loss = F.binary_cross_entropy_with_logits(fg_logits, targets.foreground, reduction="none")
    fg_loss = fg_loss.flatten(1).mean(1)
    off = F.smooth_l1_loss(offsets, targets.offsets, reduction="none").sum(1)
    mask = targets.foreground
    off_loss = (off * mask).flatten(1).sum(1) / mask.flatten(1).sum(1).clamp_min(1)
    return fg_loss + offset_weight * off_loss


def foreground_dice(prediction: dict[str, torch.Tensor], targets: BatchTargets) -> torch.Tensor:
    pred = prediction["fg_logits"].sigmoid() > 0.5
    gt = targets.foreground > 0.5
    inter = (pred & gt).flatten(1).sum(1).float()
    denom = pred.flatten(1).sum(1) + gt.flatten(1).sum(1)
    return ((2 * inter + 1e-6) / (denom.float() + 1e-6)).mean()


def decode_instances(
    prediction: dict[str, torch.Tensor],
    fg_threshold: float = 0.5,
    vote_threshold: float = 0.75,
    min_center_distance: float = 4.0,
    max_instances: int = 16,
) -> torch.Tensor:
    """Center voting + greedy NMS. Returns [B,H,W] integer instance labels."""
    fg = prediction["fg_logits"].sigmoid() > fg_threshold
    offsets = prediction["offsets"]
    b, h, w = fg.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=fg.device),
        torch.arange(w, device=fg.device),
        indexing="ij",
    )
    out = torch.zeros(b, h, w, dtype=torch.long, device=fg.device)
    for bi in range(b):
        mask = fg[bi]
        if not mask.any():
            continue
        cy = yy.float() + offsets[bi, 0] * max(h - 1, 1)
        cx = xx.float() + offsets[bi, 1] * max(w - 1, 1)
        iy = cy[mask].round().long().clamp(0, h - 1)
        ix = cx[mask].round().long().clamp(0, w - 1)
        votes = torch.zeros(h * w, device=fg.device)
        votes.scatter_add_(0, iy * w + ix, torch.ones_like(iy, dtype=votes.dtype))
        votes = votes.reshape(h, w)
        smooth = F.avg_pool2d(votes[None, None], 3, stride=1, padding=1)[0, 0]
        candidates = torch.nonzero(smooth >= vote_threshold, as_tuple=False)
        if candidates.numel() == 0:
            continue
        scores = smooth[candidates[:, 0], candidates[:, 1]]
        order = torch.argsort(scores, descending=True)
        centers: list[torch.Tensor] = []
        for idx in order:
            c = candidates[idx].float()
            if all(torch.linalg.vector_norm(c - kept).item() >= min_center_distance for kept in centers):
                centers.append(c)
                if len(centers) >= max_instances:
                    break
        if not centers:
            continue
        centers_t = torch.stack(centers)
        pixels = torch.nonzero(mask, as_tuple=False).float()
        predicted_centers = torch.stack((cy[mask], cx[mask]), dim=1)
        nearest = torch.cdist(predicted_centers, centers_t).argmin(1) + 1
        out[bi, pixels[:, 0].long(), pixels[:, 1].long()] = nearest
    return out


def pq_single(pred: torch.Tensor, gt: torch.Tensor, iou_threshold: float = 0.5) -> float:
    pred_ids = torch.unique(pred)
    pred_ids = pred_ids[pred_ids > 0]
    gt_ids = torch.unique(gt)
    gt_ids = gt_ids[gt_ids > 0]
    if len(pred_ids) == 0 and len(gt_ids) == 0:
        return 1.0
    pairs = []
    for pi in pred_ids.tolist():
        pm = pred == pi
        for gi in gt_ids.tolist():
            gm = gt == gi
            inter = (pm & gm).sum().item()
            if inter == 0:
                continue
            union = (pm | gm).sum().item()
            iou = inter / union
            if iou > iou_threshold:
                pairs.append((iou, pi, gi))
    pairs.sort(reverse=True)
    used_p, used_g, matched = set(), set(), []
    for iou, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        matched.append(iou)
    tp = len(matched)
    fp = len(pred_ids) - tp
    fn = len(gt_ids) - tp
    dq = tp / max(tp + 0.5 * fp + 0.5 * fn, 1e-8)
    sq = sum(matched) / tp if tp else 0.0
    return dq * sq


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device | str) -> dict[str, float]:
    model.eval()
    losses, dices, pqs = [], [], []
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        pred = model(images)
        losses.append(segmentation_loss_per_sample(pred, targets).mean().item())
        dices.append(foreground_dice(pred, targets).item())
        decoded = decode_instances(pred)
        for bi in range(images.shape[0]):
            pqs.append(pq_single(decoded[bi].cpu(), targets.instances[bi].cpu()))
    return {
        "loss": sum(losses) / max(len(losses), 1),
        "dice": sum(dices) / max(len(dices), 1),
        "pq": sum(pqs) / max(len(pqs), 1),
    }
