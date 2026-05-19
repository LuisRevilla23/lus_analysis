from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src_torch.blas import calc_blas
from src_torch.dataset import LUSSegmentationDataset, list_paired_samples
from src_torch.metrics import confusion_matrix, metrics_from_confusion
from src_torch.model import build_model, count_parameters, normalize_architecture, scaled_filters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", type=Path)
    parser.add_argument("--checkpoint", default="outputs/torch_unet/best.pt", type=Path)
    parser.add_argument("--output", default="outputs/torch_unet/test_metrics.json", type=Path)
    parser.add_argument("--architecture", default=None,
                        help="Override checkpoint architecture. Defaults to checkpoint config or light_unet.")
    parser.add_argument("--width-multiplier", default=None, type=float,
                        help="Override checkpoint width multiplier. Defaults to checkpoint config or 1.0.")
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--limit-samples", default=None, type=int,
                        help="Use only the first N test samples for a quick smoke evaluation.")
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--device", default="cpu",
                        help="Use cpu for safe smoke tests; pass cuda explicitly for GPU evaluation.")
    args = parser.parse_args()

    samples = list_paired_samples(
        args.root / "data" / "frames" / "test" / "images",
        args.root / "data" / "frames" / "test" / "masks",
    )
    if args.limit_samples is not None:
        samples = samples[:args.limit_samples]
    dataset = LUSSegmentationDataset(samples, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get("config", {})
    architecture = normalize_architecture(args.architecture or config.get("architecture", "light_unet"))
    if args.width_multiplier is not None:
        filters = scaled_filters(args.width_multiplier)
    elif "filters" in config:
        filters = tuple(int(x) for x in config["filters"])
    else:
        filters = scaled_filters(float(config.get("width_multiplier", 1.0)))
    model = build_model(architecture, in_channels=1, num_classes=6, filters=filters).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    cm = torch.zeros((6, 6), dtype=torch.int64, device=device)
    blas_rows = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            cm += confusion_matrix(preds, masks, num_classes=6)

            for name, pred, target in zip(batch["name"], preds.cpu().numpy(), masks.cpu().numpy()):
                blas_rows.append({
                    "name": name,
                    "blas_pred": calc_blas(pred.astype(np.uint8)),
                    "blas_target": calc_blas(target.astype(np.uint8)),
                })

    metrics = metrics_from_confusion(cm.cpu())
    metrics["checkpoint"] = str(args.checkpoint)
    metrics["architecture"] = architecture
    metrics["filters"] = list(filters)
    metrics["parameter_count"] = count_parameters(model)
    metrics["blas"] = blas_rows
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
