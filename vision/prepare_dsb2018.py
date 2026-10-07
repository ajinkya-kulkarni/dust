from __future__ import annotations

import argparse
from pathlib import Path

from data import DSB2018Dataset


def main() -> None:
    p = argparse.ArgumentParser(description="Validate DSB2018 pairs and precompute StarDist targets.")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--output-stride", type=int, default=4)
    p.add_argument("--grid-offset", type=int, default=2)
    p.add_argument("--n-rays", type=int, default=32)
    p.add_argument("--ray-scale", type=float, default=16.0)
    args = p.parse_args()

    for split in args.splits:
        ds = DSB2018Dataset(
            args.data_dir,
            split=split,
            image_size=args.image_size,
            n_rays=args.n_rays,
            output_stride=args.output_stride,
            grid_offset=args.grid_offset,
            ray_scale=args.ray_scale,
            cache_targets=True,
        )
        count = len(ds) if args.limit is None else min(len(ds), args.limit)
        nuclei = 0
        nonempty = 0
        for i in range(count):
            image, instances, objectness, rays = ds[i]
            n = int(instances.max().item())
            nuclei += n
            nonempty += int(n > 0)
            if i == 0:
                print(
                    f"{split}: sample image={tuple(image.shape)} instances={tuple(instances.shape)} "
                    f"objectness={tuple(objectness.shape)} rays={tuple(rays.shape)}"
                )
        print(
            f"{split}: prepared={count}/{len(ds)} nonempty={nonempty} nuclei={nuclei} "
            f"cache={ds.cache_dir}"
        )


if __name__ == "__main__":
    main()
