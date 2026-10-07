from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import BatchTargets


def segmentation_loss_components_per_sample(
    prediction: dict[str, torch.Tensor],
    targets: BatchTargets,
    obj_pos_weight: float = 4.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unweighted objectness and radial losses for each sample."""
    obj_logits = prediction["obj_logits"]
    rays = prediction["rays"]
    pos_weight = torch.as_tensor(
        obj_pos_weight, device=obj_logits.device, dtype=obj_logits.dtype
    )
    obj_loss = F.binary_cross_entropy_with_logits(
        obj_logits,
        targets.objectness,
        reduction="none",
        pos_weight=pos_weight,
    ).flatten(1).mean(1)

    ray = F.smooth_l1_loss(rays, targets.rays, reduction="none").mean(1)
    mask = targets.instances > 0
    ray_loss = (ray * mask).flatten(1).sum(1) / mask.flatten(1).sum(1).clamp_min(1)
    return obj_loss, ray_loss


def segmentation_loss_per_sample(
    prediction: dict[str, torch.Tensor],
    targets: BatchTargets,
    ray_weight: float = 0.5,
    obj_pos_weight: float = 4.0,
) -> torch.Tensor:
    obj_loss, ray_loss = segmentation_loss_components_per_sample(
        prediction, targets, obj_pos_weight=obj_pos_weight
    )
    return obj_loss + ray_weight * ray_loss


def ray_mae(prediction: dict[str, torch.Tensor], targets: BatchTargets) -> torch.Tensor:
    err = (prediction["rays"] - targets.rays).abs().mean(1)
    mask = targets.instances > 0
    return (err * mask).sum() / mask.sum().clamp_min(1)


def _star_mask(
    cy: int,
    cx: int,
    rays: torch.Tensor,
    yy: torch.Tensor,
    xx: torch.Tensor,
) -> torch.Tensor:
    """Rasterize a star-convex shape by interpolating adjacent radial predictions."""
    n_rays = rays.numel()
    dy = yy - float(cy)
    dx = xx - float(cx)
    distance = torch.sqrt(dy.square() + dx.square())
    angle = torch.remainder(torch.atan2(dy, dx), 2.0 * math.pi)
    ray_pos = angle * (n_rays / (2.0 * math.pi))
    i0 = torch.floor(ray_pos).long() % n_rays
    i1 = (i0 + 1) % n_rays
    frac = ray_pos - torch.floor(ray_pos)
    allowed = rays[i0] * (1.0 - frac) + rays[i1] * frac
    return distance <= allowed


def decode_instances(
    prediction: dict[str, torch.Tensor],
    obj_threshold: float = 0.30,
    nms_iou: float = 0.30,
    max_candidates: int = 64,
    max_instances: int = 16,
) -> torch.Tensor:
    """StarDist-like local maxima, star rasterization, and overlap NMS."""
    prob = prediction["obj_logits"].sigmoid()
    rays = prediction["rays"]
    b, h, w = prob.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=prob.device, dtype=torch.float32),
        torch.arange(w, device=prob.device, dtype=torch.float32),
        indexing="ij",
    )
    out = torch.zeros(b, h, w, dtype=torch.long, device=prob.device)
    max_radius = math.sqrt(h * h + w * w)

    for bi in range(b):
        pooled = F.max_pool2d(prob[bi][None, None], 3, stride=1, padding=1)[0, 0]
        is_peak = (prob[bi] >= pooled - 1e-7) & (prob[bi] >= obj_threshold)
        candidates = torch.nonzero(is_peak, as_tuple=False)
        if candidates.numel() == 0:
            continue
        scores = prob[bi, candidates[:, 0], candidates[:, 1]]
        order = torch.argsort(scores, descending=True)[:max_candidates]
        kept_masks: list[torch.Tensor] = []

        for idx in order:
            cy = int(candidates[idx, 0])
            cx = int(candidates[idx, 1])
            radial = rays[bi, :, cy, cx].clamp(0.0, max_radius)
            mask = _star_mask(cy, cx, radial, yy, xx)
            if mask.sum().item() < 4:
                continue

            suppress = False
            for kept in kept_masks:
                inter = (mask & kept).sum().float()
                union = (mask | kept).sum().float().clamp_min(1)
                if (inter / union).item() > nms_iou:
                    suppress = True
                    break
            if suppress:
                continue

            kept_masks.append(mask)
            if len(kept_masks) >= max_instances:
                break

        # Candidates are score-sorted, so higher-confidence stars win overlaps.
        for label, mask in enumerate(kept_masks, start=1):
            out[bi][mask & (out[bi] == 0)] = label
    return out


def foreground_dice(decoded: torch.Tensor, targets: BatchTargets) -> torch.Tensor:
    pred = decoded > 0
    gt = targets.instances > 0
    inter = (pred & gt).flatten(1).sum(1).float()
    denom = pred.flatten(1).sum(1) + gt.flatten(1).sum(1)
    return ((2 * inter + 1e-6) / (denom.float() + 1e-6)).mean()


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
    losses, dices, pqs, ray_errors = [], [], [], []
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        pred = model(images)
        losses.append(segmentation_loss_per_sample(pred, targets).mean().item())
        ray_errors.append(ray_mae(pred, targets).item())
        decoded = decode_instances(pred)
        dices.append(foreground_dice(decoded, targets).item())
        for bi in range(images.shape[0]):
            pqs.append(pq_single(decoded[bi].cpu(), targets.instances[bi].cpu()))
    return {
        "loss": sum(losses) / max(len(losses), 1),
        "dice": sum(dices) / max(len(dices), 1),
        "pq": sum(pqs) / max(len(pqs), 1),
        "ray_mae": sum(ray_errors) / max(len(ray_errors), 1),
    }
