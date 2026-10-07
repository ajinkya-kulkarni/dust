# DUST on toy instance segmentation

A deliberately small end-to-end test of forward-only DUST-style credit assignment on instance segmentation. The original language-model code at repository root is untouched.

## Question

Can activation-perturbation credit assignment learn a spatial instance-segmentation task without calling `backward()`?

This experiment uses synthetic 32x32 grayscale images containing 1-5 touching ellipses. The model predicts foreground probability plus a normalized `(dy, dx)` vector from each foreground pixel to its instance centroid. At inference, pixels vote for centers and are assigned to the nearest detected center.

## Model

The default network is intentionally tiny and DUST-friendly:

- 4x4 patches -> 64 tokens
- linear patch embedding, width 64
- two bidirectional transformer blocks, 4 heads
- linear dense-prediction head
- fixed 2-D sinusoidal positional encoding
- no trainable normalization parameters

All trainable tensors live inside `nn.Linear` modules.

## What differs from Q Labs' language DUST

This is a research adaptation, not a claim that the original algorithm transfers unchanged. The LM implementation exploits causal token structure and specialized attention/head estimators. Here we use the simplest general test:

1. perturb one selected Linear output with Gaussian activation noise;
2. use antithetic `+noise/-noise` forward evaluations;
3. reward each sample using its own segmentation-loss change;
4. estimate the activation gradient;
5. convert it to a Linear weight gradient by the clean input/output-error outer product;
6. hand the estimated gradient to AdamW.

The DUST training path sets `requires_grad_(False)` and never calls autograd/backward.

## Run the backprop baseline

```bash
python vision/train_bp.py --steps 500 --output runs/vision-bp.pt
```

## Gradient-cosine sanity check

```bash
python vision/grad_cosine.py --populations 16 64 256 1024
```

The first thing to look for is whether alignment improves as population increases. Another Linear site can be checked with:

```bash
python vision/grad_cosine.py --site blocks.1.fc2 --populations 16 64 256 1024
```

## DUST-train the full toy segmenter

Start small:

```bash
python vision/train_dust.py \
  --population 64 \
  --draw-chunk 8 \
  --steps 100 \
  --output runs/vision-dust-p64.pt
```

Then increase population only if the cosine experiment supports it:

```bash
python vision/train_dust.py --population 256 --draw-chunk 8 --steps 100 --output runs/vision-dust-p256.pt
```

`--draw-chunk` controls memory, not estimator population.

## Optional: DUST only the head of a backprop checkpoint

```bash
python vision/train_dust.py \
  --mode head \
  --init runs/vision-bp.pt \
  --population 256 \
  --steps 100 \
  --lr 5e-4 \
  --output runs/vision-dust-head.pt
```

## Metrics

Training/evaluation prints dense segmentation loss, foreground Dice, and instance PQ using IoU > 0.5 matching.

If cosine similarity does not improve with population on this tiny problem, scaling to a real microscopy dataset is not justified yet.

## Files

- `data.py`: synthetic touching-ellipse data and centroid-offset targets
- `model.py`: tiny all-Linear transformer segmenter
- `task.py`: loss, center-voting instance decoder, Dice and PQ
- `dust_vision.py`: antithetic forward-only estimator for `nn.Linear` sites
- `train_bp.py`: AdamW + backprop baseline
- `train_dust.py`: end-to-end forward-only training
- `grad_cosine.py`: DUST-vs-backprop gradient alignment sweep
