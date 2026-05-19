from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


@dataclass(frozen=True)
class Sample:
    image: Path
    mask: Path


def list_paired_samples(image_dir: str | Path, mask_dir: str | Path) -> list[Sample]:
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    images = {p.name: p for p in image_dir.glob("*.png")}
    masks = {p.name: p for p in mask_dir.glob("*.png")}

    missing_masks = sorted(set(images) - set(masks))
    missing_images = sorted(set(masks) - set(images))
    if missing_masks or missing_images:
        msg = [
            f"Unpaired dataset: {len(missing_masks)} images without masks, "
            f"{len(missing_images)} masks without images."
        ]
        if missing_masks:
            msg.append(f"First missing masks: {missing_masks[:5]}")
        if missing_images:
            msg.append(f"First missing images: {missing_images[:5]}")
        raise ValueError(" ".join(msg))

    return [Sample(images[name], masks[name]) for name in sorted(images)]


def split_samples(
    samples: Iterable[Sample],
    validation_fraction: float = 0.2,
    seed: int = 47,
) -> tuple[list[Sample], list[Sample]]:
    samples = list(samples)
    rng = random.Random(seed)
    rng.shuffle(samples)
    n_valid = int(round(len(samples) * validation_fraction))
    return samples[n_valid:], samples[:n_valid]


class LUSSegmentationDataset(Dataset):
    """Lung ultrasound semantic segmentation dataset.

    Images are grayscale PNGs. Masks are single-channel PNGs with integer labels:
    0 background, 1 ribs, 2 pleural line, 3 A-line, 4 B-line, 5 B-line confluence.
    """

    def __init__(
        self,
        samples: Iterable[Sample],
        crop: tuple[int, int, int, int] | None = (100, 50, 850, 460),
        size: tuple[int, int] = (256, 256),
        augment: bool = False,
        rotation_degrees: float = 30.0,
        brightness_delta: float = 0.25,
        contrast_range: tuple[float, float] = (0.75, 1.25),
        use_depth_augmentation: bool = False,
        depth_crop_min: float = 0.75,
        depth_zoom_min: float = 0.75,
        use_tgc_augmentation: bool = False,
        tgc_darkness_sigma: float = 0.1,
        tgc_n_lines: int = 8,
        hflip_prob: float = 0.5,
    ) -> None:
        self.samples = list(samples)
        self.crop = crop
        self.size = size
        self.augment = augment
        self.rotation_degrees = rotation_degrees
        self.brightness_delta = brightness_delta
        self.contrast_range = contrast_range
        self.use_depth_augmentation = use_depth_augmentation
        self.depth_crop_min = depth_crop_min
        self.depth_zoom_min = depth_zoom_min
        self.use_tgc_augmentation = use_tgc_augmentation
        self.tgc_darkness_sigma = tgc_darkness_sigma
        self.tgc_n_lines = tgc_n_lines
        self.hflip_prob = hflip_prob

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[idx]
        image = Image.open(sample.image).convert("L")
        mask = Image.open(sample.mask).convert("L")

        if self.crop is not None:
            image = image.crop(self.crop)
            mask = mask.crop(self.crop)

        image = TF.resize(image, self.size, interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.size, interpolation=InterpolationMode.NEAREST)

        if self.augment:
            image, mask = self._augment_geometric(image, mask)

        image_tensor = TF.to_tensor(image)
        if self.augment:
            image_tensor = self._augment_intensity(image_tensor)

        mask_array = np.asarray(mask, dtype=np.int64)
        mask_tensor = torch.from_numpy(mask_array).long()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "name": sample.image.name,
        }

    def _augment_geometric(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        if random.random() < self.hflip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
        image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR, fill=0)
        mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST, fill=0)

        if self.use_depth_augmentation:
            image, mask = self._augment_depth(image, mask)

        return image, mask

    def _augment_depth(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        width, height = image.size
        min_side = min(width, height)
        min_crop = max(1, int(round(min_side * self.depth_crop_min)))
        crop_side = random.randint(min_crop, min_side)
        left = random.randint(0, max(width - crop_side, 0))
        top = random.randint(0, max(height - crop_side, 0))
        box = (left, top, left + crop_side, top + crop_side)

        image = image.crop(box)
        mask = mask.crop(box)

        scale = max(1, int(round(height * random.uniform(self.depth_zoom_min, 1.0))))
        image = image.resize((scale, scale), resample=Image.Resampling.BILINEAR)
        mask = mask.resize((scale, scale), resample=Image.Resampling.NEAREST)

        pad_left = max((width - scale) // 2, 0)
        padded_image = Image.new("L", (width, height), color=0)
        padded_mask = Image.new("L", (width, height), color=0)
        padded_image.paste(image, (pad_left, 0))
        padded_mask.paste(mask, (pad_left, 0))
        return padded_image, padded_mask

    def _augment_intensity(self, image: torch.Tensor) -> torch.Tensor:
        if self.use_tgc_augmentation:
            image = image + self._tgc_filter(image.shape[-2], image.shape[-1]).unsqueeze(0)

        if self.brightness_delta > 0:
            image = image + random.uniform(-self.brightness_delta, self.brightness_delta)

        lower, upper = self.contrast_range
        factor = random.uniform(lower, upper)
        mean = image.mean(dim=(-2, -1), keepdim=True)
        return (image - mean) * factor + mean

    def _tgc_filter(self, height: int, width: int) -> torch.Tensor:
        sigma = height / max(self.tgc_n_lines - 1, 1)
        y = torch.arange(height, dtype=torch.float32).view(height, 1)
        filt = torch.zeros((height, width), dtype=torch.float32)
        for idx in range(self.tgc_n_lines):
            darkness = abs(random.gauss(0.0, self.tgc_darkness_sigma))
            line = torch.exp(-((y - idx * sigma) ** 2) / (2.0 * (sigma / 2.0) ** 2))
            filt += line.repeat(1, width) * darkness
        return filt
