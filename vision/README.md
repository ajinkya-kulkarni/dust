# DUST on synthetic StarDist-style instance segmentation

A controlled follow-up to the centroid-offset toy experiment. The synthetic 32x32 touching-ellipse images and tiny bidirectional transformer stay the same; only the instance representation and decoder change.

## Question

Can forward-only activation-perturbation credit assignment learn a StarDist-like dense instance representation?

Each foreground location receives:

- a center-weighted objectness target in [0,1]
- 16 radial distances to the ellipse boundary, at fixed angles

The model predicts the same quantities. At inference, local objectness maxima seed star-convex masks, and greedy overlap NMS produces instance labels.

## Model

- 32x32 grayscale input
- 4x4 patches -> 64 spatial tokens
- width 64
- two bidirectional transformer blocks, 4 attention heads
- fixed 2-D sinusoidal positional encoding
- linear dense head
- 16 rays
- 16 pixels per token x (1 objectness + 16 rays) = 272 outputs per token

All trainable tensors remain inside nn.Linear modules, so the same DUST estimator is used without introducing a Conv2d estimator yet.

## Loss

The per-sample objective is:

- weighted BCE on the center-weighted objectness map
- masked SmoothL1 on the 16 radial distances inside instances

Predicted radii are kept positive with softplus.

## Decoder and metrics

The decoder:

1. finds local maxima of predicted objectness;
2. reads the 16 rays at each candidate;
3. rasterizes a star-convex mask by interpolating adjacent rays;
4. applies greedy mask-IoU NMS;
5. assigns score-ordered instance labels.

Evaluation prints:

- validation loss
- foreground Dice from decoded instance masks
- instance PQ at IoU > 0.5
- radial-distance MAE in pixels

## Why CPU is the default on Mac

The earlier DUST experiment produced invalid numerical behavior on MPS, while CPU was stable. CUDA remains the default when available; otherwise these scripts use CPU. MPS can still be requested explicitly, but it is not the recommended reference path yet.

## Run

First make sure you are on this branch:

```bash
git fetch origin
git checkout experiment/vision-stardist-synthetic
```

### 1. Backprop baseline

```bash
uv run python vision/train_bp.py \
  --device cpu \
  --steps 500 \
  --output runs/vision-stardist-bp.pt
```

### 2. Gradient-cosine test

```bash
uv run python vision/grad_cosine.py \
  --device cpu \
  --populations 16 64 256 1024
```

Test deeper sites too:

```bash
uv run python vision/grad_cosine.py --device cpu --site blocks.1.fc2 --populations 16 64 256 1024
uv run python vision/grad_cosine.py --device cpu --site blocks.0.attn.qkv --populations 16 64 256 1024
uv run python vision/grad_cosine.py --device cpu --site patch_embed --populations 16 64 256 1024
```

### 3. Full forward-only training

```bash
uv run python vision/train_dust.py \
  --device cpu \
  --population 256 \
  --draw-chunk 8 \
  --steps 100 \
  --lr 3e-4 \
  --output runs/vision-stardist-dust-p256-100.pt
```

A population-64 comparison:

```bash
uv run python vision/train_dust.py \
  --device cpu \
  --population 64 \
  --draw-chunk 8 \
  --steps 100 \
  --lr 3e-4 \
  --output runs/vision-stardist-dust-p64-100.pt
```

## Local implementation sanity checks

Before pushing this branch, the StarDist-like variant was checked locally on CPU:

- the model/data/loss/decoder path runs end to end;
- a BP run learns the task, reaching about Dice 0.82 and PQ 0.40 by 300 steps in the local smoke test;
- head-gradient cosine improved with population: roughly 0.17, 0.40, 0.71, 0.91 for populations 16, 64, 256, 1024;
- at population 1024 the head-gradient norm ratio was about 1.07.

These are smoke-test observations, not final benchmark claims.

## Files

- `data.py`: synthetic ellipses, center-weighted objectness, analytic 16-ray targets
- `model.py`: tiny all-Linear transformer with StarDist-like output head
- `task.py`: loss, star rasterizer, overlap NMS, Dice/PQ/ray-MAE
- `dust_vision.py`: forward-only activation perturbation estimator
- `train_bp.py`: ordinary AdamW/backprop baseline
- `train_dust.py`: end-to-end DUST training
- `grad_cosine.py`: DUST-vs-backprop gradient alignment
