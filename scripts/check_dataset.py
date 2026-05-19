from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def inspect_split(root: Path, split: str) -> None:
    image_dir = root / "data" / "frames" / split / "images"
    mask_dir = root / "data" / "frames" / split / "masks"
    images = {p.name: p for p in image_dir.glob("*.png")}
    masks = {p.name: p for p in mask_dir.glob("*.png")}
    missing_masks = sorted(set(images) - set(masks))
    missing_images = sorted(set(masks) - set(images))
    if missing_masks or missing_images:
        raise ValueError(
            f"{split}: {len(missing_masks)} images without masks, "
            f"{len(missing_images)} masks without images."
        )
    samples = [(images[name], masks[name]) for name in sorted(images)]
    values: set[int] = set()
    shapes: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}
    for image_path, mask_path in samples:
        image = Image.open(image_path)
        mask = Image.open(mask_path)
        key = (image.size, mask.size)
        shapes[key] = shapes.get(key, 0) + 1
        values.update(np.unique(np.asarray(mask)).astype(int).tolist())

    print(f"{split}: {len(samples)} paired PNG samples")
    print(f"{split}: mask values {sorted(values)}")
    for (image_size, mask_size), count in sorted(shapes.items(), key=lambda item: item[1], reverse=True):
        print(f"{split}: image={image_size}, mask={mask_size}, count={count}")
    print(f"{split}: first 10 files {[image_path.name for image_path, _ in samples[:10]]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", type=Path)
    args = parser.parse_args()
    inspect_split(args.root, "train")
    inspect_split(args.root, "test")


if __name__ == "__main__":
    main()
