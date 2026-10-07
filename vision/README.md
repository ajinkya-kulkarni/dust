# DSB2018 StarDist-style training with forward-only DUST

This branch moves the validated synthetic StarDist experiment to real 256x256 DSB2018 patches.

It keeps the same research question:

> Can a StarDist-like instance-segmentation model be trained end to end with DUST-style forward-only activation credit assignment instead of backpropagation?

This is not the canonical StarDist CNN and not an unchanged port of the Q Labs language-model DUST implementation. The representation is StarDist-like; the encoder is a small all-Linear bidirectional transformer chosen so the forward-only estimator remains easy to inspect.

## Expected dataset layout

The loader expects paired TIFF images and NPY instance masks with identical stems:

```text
FluoFuse_dsb2018_stratified/
├── data.yaml
├── metadata.csv
├── train/
│   ├── sample.tif
│   ├── sample.npy
│   └── ...
├── val/
│   ├── sample.tif
│   ├── sample.npy
│   └── ...
└── test/
    ├── sample.tif
    ├── sample.npy
    └── ...
```

The NPY masks may contain sparse/global instance ids. Every patch is relabeled locally to contiguous ids 1..N before target generation.

## Locked first-pass geometry

```text
input                 256x256
encoder patch         16x16
encoder grid          16x16
transformer tokens    256

dim                   96
depth                 2
attention heads       4

StarDist stride       4 px
StarDist grid         64x64
grid offset           2 px
rays                  32
ray scale             16 px

output positions/token 4x4 = 16
channels/position      1 objectness + 32 rays = 33
head width             16 x 33 = 528
```

The stride-4 output grid places prediction centers at image coordinates:

```text
2, 6, 10, ..., 254
```

in both axes.

## Image preprocessing

TIFFs are reduced to one grayscale channel if necessary and normalized per patch using the 1st and 99.8th intensity percentiles, then clipped to [0,1].

## StarDist-style targets

### Objectness

For each instance, a Euclidean distance transform is computed independently and normalized by that instance's maximum interior distance.

The resulting center-weighted probability map is sampled on the 64x64 stride-4 grid.

### Rays

At every foreground output-grid location, 32 fixed-angle rays are marched through the full-resolution instance mask until the instance id changes or the ray leaves the image.

Distances are measured in pixels and stored normalized by:

```text
ray_scale = 16 px
```

so a typical 11 px radius is represented as roughly 0.69.

Targets are cached under:

```text
<data-dir>/.dust_stardist_cache/
```

The cache key includes image size, output stride, grid offset, ray count, and ray scale.

## Loss

```text
L = L_objectness + 0.5 * L_rays
```

where:

- objectness uses weighted BCE with default positive weight 4;
- rays use masked SmoothL1 at foreground output-grid locations;
- ray MAE is reported back in pixels.

## Decoder

Evaluation converts the 64x64 predictions back to 256x256 instances:

1. local maxima in objectness become candidates;
2. a candidate at output-grid coordinate `(gy,gx)` maps to full-resolution center `(2+4*gy, 2+4*gx)`;
3. its 32 normalized rays are multiplied by 16 px;
4. a star-convex mask is rasterized at full resolution;
5. greedy mask-IoU NMS removes duplicates;
6. up to 160 instances are retained.

Default candidate limit is 512, rather than the synthetic experiment's much smaller limits.

## Forward-only DUST

For a selected Linear activation `y`, Gaussian perturbations are evaluated with antithetic forward passes:

```text
y+ = y + sigma * eps
y- = y - sigma * eps

directional derivative
d = (L(y+) - L(y-)) / (2*sigma)

estimated activation error
delta_y ~= mean(d * eps)

Linear weight gradient
grad_W ~= delta_y^T x
```

No `backward()` call is used in the DUST training path. Estimated gradients are assigned manually to `.grad`, then AdamW applies the update.

### Split StarDist head credit

The successful synthetic experiment showed that a single global scalar loss badly contaminates the radial-head estimate.

Therefore split credit remains the default:

```text
objectness outputs
  -> perturb objectness only
  -> score with objectness loss

ray outputs
  -> perturb rays only
  -> score with radial loss

combine both activation-error estimates
  -> reconstruct head weight gradient
```

### Exact token-local spatial credit

On real 256x256 DSB2018 patches, the old image-global scalar reward becomes too noisy: one scalar loss change was being used to assign credit to all 256 transformer tokens at once.

For sites after the final cross-token mixing operation, the downstream StarDist loss decomposes exactly by encoder token. This branch therefore uses per-token loss contributions at these sites:

```text
head
blocks.1.attn.proj
blocks.1.fc1
blocks.1.fc2
```

Each encoder token owns a 4x4 block of the 64x64 StarDist grid. Objectness and ray loss contributions are computed for that block using the same global normalization as the ordinary loss, so summing all 256 token contributions reproduces the original per-sample loss.

The perturbation population is still evaluated in parallel across all tokens, but token `t` receives only the loss change from token `t`'s output block:

```text
global credit (old):
all token perturbations -> one image loss scalar -> credit every token

token-local credit (new):
token t perturbation -> token t's exact additive loss contribution -> credit token t
```

No approximation to the training objective is introduced at these eligible sites. Earlier sites such as `patch_embed`, block 0, and `blocks.1.attn.qkv` still require image-global credit because later attention mixes tokens.

Use `--spatial-credit global` to reproduce the old estimator for comparison. The default is `--spatial-credit local`.

## Setup

Checkout the branch:

```bash
git fetch origin
git checkout experiment/dsb2018-stardist
```

Install/update the environment:

```bash
uv sync
```

The DSB experiment adds NumPy, SciPy and tifffile for image loading, distance transforms and target generation.

## 1. Validate and cache targets

If the dataset directory is a sibling of the `dust` repo:

```bash
uv run python vision/prepare_dsb2018.py \
  --data-dir ../FluoFuse_dsb2018_stratified \
  --limit 10
```

This checks ten samples in each split and prints the image/target shapes.

Expected target shapes:

```text
image       (1, 256, 256)
instances   (256, 256)
objectness  (64, 64)
rays        (32, 64, 64)
```

If that is clean, precompute all targets:

```bash
uv run python vision/prepare_dsb2018.py \
  --data-dir ../FluoFuse_dsb2018_stratified
```

Training can also create missing cache entries automatically.

## 2. Backprop baseline first

Do not start DUST until ordinary training shows the model/targets are viable.

First real-data baseline:

```bash
uv run python vision/train_bp.py \
  --data-dir ../FluoFuse_dsb2018_stratified \
  --device cpu \
  --batch-size 4 \
  --steps 500 \
  --eval-every 100 \
  --output runs/dsb2018-stardist-bp-500.pt
```

The default model in this command is already the locked 256 / patch16 / stride4 / 32-ray configuration.

## 3. Gradient-alignment diagnostic

Once BP learns:

```bash
uv run python vision/grad_cosine.py \
  --data-dir ../FluoFuse_dsb2018_stratified \
  --init runs/dsb2018-stardist-bp-500.pt \
  --device cpu \
  --spatial-credit local \
  --populations 16 64 256
```

For the head this reports:

- complete gradient cosine and norm ratio;
- objectness-only cosine and norm ratio;
- ray-only cosine and norm ratio.

Internal sites can also be checked:

```bash
uv run python vision/grad_cosine.py \
  --data-dir ../FluoFuse_dsb2018_stratified \
  --init runs/dsb2018-stardist-bp-500.pt \
  --device cpu \
  --spatial-credit local \
  --site blocks.1.fc2 \
  --populations 16 64 256

uv run python vision/grad_cosine.py \
  --data-dir ../FluoFuse_dsb2018_stratified \
  --init runs/dsb2018-stardist-bp-500.pt \
  --device cpu \
  --spatial-credit local \
  --site patch_embed \
  --populations 16 64 256
```

## 4. DUST smoke run

Only after the gradient diagnostic looks sensible:

```bash
uv run python vision/train_dust.py \
  --data-dir ../FluoFuse_dsb2018_stratified \
  --device cpu \
  --population 64 \
  --draw-chunk 8 \
  --batch-size 1 \
  --steps 10 \
  --lr 3e-4 \
  --head-credit split \
  --spatial-credit local \
  --output runs/dsb2018-stardist-dust-p64-smoke.pt
```

Then population 256 is the intended full experiment if the cosine results justify it.

## Files

- `data.py` - TIFF/NPY loading, per-patch relabeling, objectness EDT, 32-ray target generation and cache
- `prepare_dsb2018.py` - validates pairs and precomputes target cache
- `model.py` - 256x256 patch16 transformer with 64x64 StarDist output grid
- `task.py` - StarDist loss, full-resolution polygon decoder, Dice, PQ and ray MAE
- `dust_vision.py` - forward-only activation perturbation with split head credit
- `train_bp.py` - ordinary AdamW/backprop baseline
- `grad_cosine.py` - exact-BP versus DUST gradient alignment
- `train_dust.py` - full forward-only training
