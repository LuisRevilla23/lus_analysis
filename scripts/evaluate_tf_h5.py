from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import Models as models
from src_torch.blas import calc_blas


CLASS_NAMES = {
    0: "Background",
    1: "Ribs",
    2: "Pleural line",
    3: "A-line",
    4: "B-line",
    5: "B-line confluence",
}

CLASS_COLORS = np.array(
    [
        [0, 0, 0],
        [230, 25, 75],
        [60, 180, 75],
        [255, 225, 25],
        [0, 130, 200],
        [245, 130, 48],
    ],
    dtype=np.uint8,
)


def list_pairs(image_dir: Path, mask_dir: Path) -> list[tuple[Path, Path]]:
    images = {p.name: p for p in image_dir.glob("*.png")}
    masks = {p.name: p for p in mask_dir.glob("*.png")}
    missing_masks = sorted(set(images) - set(masks))
    missing_images = sorted(set(masks) - set(images))
    if missing_masks or missing_images:
        raise ValueError(
            f"Unpaired data: {len(missing_masks)} missing masks, "
            f"{len(missing_images)} missing images."
        )
    return [(images[name], masks[name]) for name in sorted(images)]


def resize_nearest(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(array.astype(np.uint8), mode="L")
    return np.asarray(image.resize(size[::-1], resample=Image.Resampling.NEAREST), dtype=np.uint8)


def preprocess_image(path: Path, crop: tuple[int, int, int, int], size: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("L").crop(crop)
    image = image.resize(size[::-1], resample=Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return arr[None, :, :, None]


def preprocess_mask(path: Path, crop: tuple[int, int, int, int], size: tuple[int, int]) -> np.ndarray:
    mask = Image.open(path).convert("L").crop(crop)
    mask = mask.resize(size[::-1], resample=Image.Resampling.NEAREST)
    return np.asarray(mask, dtype=np.uint8)


def update_confusion(cm: np.ndarray, pred: np.ndarray, target: np.ndarray, num_classes: int) -> None:
    valid = (target >= 0) & (target < num_classes)
    idx = target[valid].astype(np.int64) * num_classes + pred[valid].astype(np.int64)
    cm += np.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def metrics_from_confusion(cm: np.ndarray) -> dict:
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    support = cm.sum(axis=1)

    dice = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) > 0)
    iou = np.divide(tp, tp + fp + fn, out=np.zeros_like(tp), where=(tp + fp + fn) > 0)
    pixel_accuracy = float(tp.sum() / max(cm.sum(), 1.0))

    return {
        "pixel_accuracy": pixel_accuracy,
        "mean_dice_excluding_background": float(dice[1:].mean()),
        "classes": {
            str(i): {
                "name": CLASS_NAMES[i],
                "dice": float(dice[i]),
                "iou": float(iou[i]),
                "support_pixels": int(support[i]),
            }
            for i in range(len(CLASS_NAMES))
        },
    }


def save_mask_png(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8), mode="L").save(path)


def save_colored_mask_png(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(mask, 0, len(CLASS_COLORS) - 1)
    Image.fromarray(CLASS_COLORS[clipped], mode="RGB").save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", type=Path)
    parser.add_argument("--weights", default="model_lus.h5", type=Path)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--output-dir", default="outputs/tf_h5_eval", type=Path)
    parser.add_argument("--limit-samples", default=None, type=int)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--crop", default="100,50,850,460",
                        help="x1,y1,x2,y2 crop used by the paper notebook for frame data.")
    parser.add_argument("--size", default="256,256", help="height,width model input size.")
    args = parser.parse_args()

    crop = tuple(int(v) for v in args.crop.split(","))
    size = tuple(int(v) for v in args.size.split(","))
    if len(crop) != 4 or len(size) != 2:
        raise ValueError("Invalid --crop or --size.")

    import tensorflow as tf

    pairs = list_pairs(
        args.root / "data" / "frames" / args.split / "images",
        args.root / "data" / "frames" / args.split / "masks",
    )
    if args.limit_samples is not None:
        pairs = pairs[:args.limit_samples]

    model = models.unet((size[0], size[1], 1), len(CLASS_NAMES), filters=[32, 64, 128, 256, 512])
    model.load_weights(str(args.weights))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cm = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    rows = []

    for image_path, mask_path in pairs:
        x = preprocess_image(image_path, crop, size)
        target = preprocess_mask(mask_path, crop, size)
        logits = model.predict(x, verbose=0)
        pred = logits[0].argmax(axis=-1).astype(np.uint8)
        update_confusion(cm, pred, target, len(CLASS_NAMES))

        row = {
            "name": image_path.name,
            "blas_target": calc_blas(target),
            "blas_pred": calc_blas(pred),
        }
        row["blas_abs_error"] = abs(row["blas_pred"] - row["blas_target"])
        rows.append(row)

        if args.save_predictions:
            save_mask_png(pred, args.output_dir / "pred_masks" / image_path.name)
            save_colored_mask_png(pred, args.output_dir / "pred_masks_color" / image_path.name)

    metrics = metrics_from_confusion(cm)
    metrics["n_samples"] = len(pairs)
    metrics["weights"] = str(args.weights)
    metrics["split"] = args.split
    metrics["crop"] = list(crop)
    metrics["size"] = list(size)

    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (args.output_dir / "blas.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "blas_target", "blas_pred", "blas_abs_error"])
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(metrics, indent=2))
    print(f"Wrote {args.output_dir / 'metrics.json'}")
    print(f"Wrote {args.output_dir / 'blas.csv'}")


if __name__ == "__main__":
    main()
