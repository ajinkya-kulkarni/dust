# Forward-only StarDist-style instance segmentation with DUST

This branch tests whether a StarDist-like instance-segmentation model can be trained end to end with forward-only, DUST-style credit assignment instead of backpropagation.

It is a controlled synthetic experiment, not a reimplementation of the official StarDist network and not a literal port of the Q Labs language-model DUST code.

## What is being tested

The task uses deterministic 32x32 grayscale images containing 1-5 touching ellipses.

For every pixel inside an instance, the target contains:

- a center-weighted objectness value in [0,1];
- 16 radial distances from that pixel to the instance boundary at fixed angles.

The model predicts the same representation. At inference, objectness peaks seed star-convex polygons reconstructed from the 16 rays, followed by greedy overlap NMS.

The experiment asks:

> Can forward-only activation perturbations recover useful credit signals well enough to train this radial instance representation without calling backward()?

## Model

The network is deliberately small and DUST-friendly:

- input: 32x32 grayscale;
- patch size: 4x4;
- 64 spatial tokens;
- width: 64;
- two bidirectional transformer blocks;
- four attention heads;
- fixed 2-D sinusoidal positional encoding;
- one linear dense prediction head;
- 16 radial outputs per pixel.

Each 4x4 token predicts 16 pixels, and each pixel has 17 outputs:

```text
1 objectness + 16 rays = 17 values/pixel

16 pixels/token x 17 = 272 outputs/token
```

All trainable tensors live inside `nn.Linear` modules.

## StarDist-style targets and loss

The objectness target is highest near an instance center and approaches zero near its boundary.

For an ellipse, each radial target is computed analytically by intersecting a ray from the current foreground pixel with the ellipse boundary.

The training loss is:

```text
L = L_objectness + 0.5 * L_rays
```

where:

- `L_objectness` is weighted binary cross entropy;
- `L_rays` is masked SmoothL1 over the 16 radial distances inside instances.

Predicted radial distances are constrained positive with `softplus`.

## Decoder

At evaluation time:

1. sigmoid objectness is computed;
2. local objectness maxima above threshold become candidates;
3. the 16 rays at each candidate are converted into a star-convex mask;
4. overlapping candidate masks are suppressed with greedy mask-IoU NMS;
5. surviving masks become instance labels.

Reported metrics are:

- validation loss;
- foreground Dice from decoded instance masks;
- PQ with IoU > 0.5 matching;
- radial MAE in pixels.

## Forward-only credit assignment

For a selected linear layer with clean input `x` and output activation `y`, DUST perturbs the activation rather than differentiating through the network.

For Gaussian perturbation `eps`:

```text
y+ = y + sigma * eps
y- = y - sigma * eps
```

Both perturbed networks are evaluated only with forward passes.

The directional loss derivative is estimated with the antithetic finite difference:

```text
d = (L(y+) - L(y-)) / (2 * sigma)
```

A population of random perturbations estimates the activation error:

```text
delta_y ~= mean(d * eps)
```

For an `nn.Linear` layer, the weight gradient is reconstructed locally from the clean layer input:

```text
grad_W ~= delta_y^T x
```

The resulting tensor is assigned to `parameter.grad` and AdamW performs the parameter update.

The DUST training path sets model parameters to `requires_grad_(False)` and does not call `backward()`.

### Important distinction from original DUST

The Q Labs implementation is designed around causal language transformers and includes LM-specific estimators.

This branch tests the underlying forward-only activation-perturbation idea on a bidirectional vision transformer. It should therefore be described as **DUST-style** or **DUST-inspired forward-only credit assignment**, not as an unchanged application of the original LM algorithm.

## Why split head credit is needed

The first naive StarDist experiment perturbed the complete 272-dimensional head and scored every perturbation with the full scalar loss.

That produced a misleadingly good overall head gradient because the true head gradient was dominated by objectness:

```text
exact objectness gradient norm: 0.695
exact ray gradient norm:        0.094
```

With the original global estimator at population 256:

```text
objectness cosine = 0.976
ray cosine        = 0.122
ray norm ratio    = 6.64
```

The model could lower the total loss while barely learning the radial geometry.

The default estimator on this branch therefore separates the final head into two credit channels:

```text
objectness outputs
    -> perturb objectness only
    -> score with objectness loss
    -> objectness error estimate

ray outputs
    -> perturb rays only
    -> score with radial loss
    -> radial error estimate

combine both error estimates
    -> reconstruct head weight gradient
```

Shared transformer layers continue to use the full StarDist loss.

This change dramatically improves radial credit quality. At population 256:

```text
overall head cosine = 0.951
objectness cosine   = 0.977
ray cosine          = 0.452
ray norm ratio      = 2.05
```

At population 1024, the ray cosine reaches 0.666, but training at that population is substantially more expensive and is not required to establish the synthetic result.

## Results

### Backprop baseline

Batch size 32, 500 steps:

| Step | Val loss | Dice | PQ | Ray MAE |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 1.344 | 0.680 | 0.000 | 2.545 |
| 150 | 1.128 | 0.781 | 0.044 | 2.204 |
| 200 | 0.974 | 0.826 | 0.161 | 1.941 |
| 300 | 0.751 | 0.824 | 0.398 | 1.523 |
| 500 | 0.565 | 0.811 | 0.534 | 1.153 |

A matched batch-size-4 BP run reached at 100 steps:

```text
val loss = 1.354
Dice     = 0.672
PQ       = 0.000
ray MAE  = 2.549
```

### Naive global-credit DUST

Population 256, batch size 4, 100 steps:

```text
val loss = 1.580
Dice     = 0.065
PQ       = 0.000
ray MAE  = 3.046
```

The total loss decreased, but the radial representation barely learned.

### Split-head DUST

Population 256, batch size 4, 100 steps:

```text
val loss = 1.502
Dice     = 0.474
PQ       = 0.000
ray MAE  = 2.870
```

Population 256, batch size 4, 300 steps:

```text
val loss = 1.062
Dice     = 0.772
PQ       = 0.068
ray MAE  = 2.176
```

PQ first becomes non-zero around step 220 and continues to improve through step 300.

This demonstrates that the forward-only model is learning instance geometry, not only foreground/objectness.

A useful comparison is:

```text
BP, step 150:
Dice     0.781
PQ       0.044
ray MAE  2.204

DUST split K=256, step 300:
Dice     0.772
PQ       0.068
ray MAE  2.176
```

The forward-only optimizer is substantially more compute-intensive and learns more slowly than backprop, but it does train the StarDist-style representation end to end.

## Gradient-alignment results

With split head credit:

| Population | Overall cosine | Objectness cosine | Ray cosine | Ray norm ratio |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 0.347 | 0.659 | 0.170 | 9.60 |
| 64 | 0.790 | 0.879 | 0.275 | 4.11 |
| 256 | 0.951 | 0.977 | 0.452 | 2.05 |
| 1024 | 0.986 | 0.996 | 0.666 | 1.40 |

Internal layers also show population-dependent convergence toward the exact backprop gradient. Before introducing split head credit, representative population-1024 cosine similarities were:

```text
blocks.1.fc2  0.982
patch_embed   0.947
```

## Conclusion

On this synthetic problem, the answer to the experimental question is **yes**:

> A small StarDist-like instance-segmentation model can be trained end to end with DUST-style forward-only activation credit assignment, without backpropagation.

The result does **not** show that DUST is competitive with backpropagation in efficiency. Population-based forward evaluations are much more expensive, and radial outputs require careful credit decomposition to control estimator variance.

The next meaningful experiment is to keep the validated StarDist representation and forward-only training mechanism and test them on a real microscopy dataset such as DSB2018.

## Reproduce

CPU is the recommended reference path for the current experiment. CUDA is used automatically when available. An earlier MPS run produced invalid numerical behavior, so MPS is not used as the default reference device.

### Backprop baseline

```bash
uv run python vision/train_bp.py \
  --device cpu \
  --steps 500 \
  --output runs/vision-stardist-bp.pt
```

Matched batch-size-4 control:

```bash
uv run python vision/train_bp.py \
  --device cpu \
  --batch-size 4 \
  --steps 100 \
  --output runs/vision-stardist-bp-b4-100.pt
```

### Gradient alignment

Split head credit is the default:

```bash
uv run python vision/grad_cosine.py \
  --device cpu \
  --populations 16 64 256 1024
```

The previous global estimator can be reproduced with:

```bash
uv run python vision/grad_cosine.py \
  --device cpu \
  --head-credit global \
  --populations 16 64 256 1024
```

### Forward-only training

```bash
uv run python vision/train_dust.py \
  --device cpu \
  --population 256 \
  --draw-chunk 8 \
  --steps 300 \
  --lr 3e-4 \
  --head-credit split \
  --output runs/vision-stardist-dust-split-p256-300.pt
```

## Files

- `data.py` - synthetic touching ellipses and analytic StarDist-style targets
- `model.py` - tiny all-Linear bidirectional transformer
- `task.py` - StarDist losses, polygon rasterization, NMS, Dice, PQ and ray MAE
- `dust_vision.py` - forward-only activation perturbation and local Linear gradient reconstruction
- `train_bp.py` - ordinary AdamW/backprop baseline
- `train_dust.py` - full forward-only training
- `grad_cosine.py` - exact-BP versus DUST gradient alignment diagnostics
