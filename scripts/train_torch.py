from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src_torch.dataset import LUSSegmentationDataset, list_paired_samples, split_samples
from src_torch.losses import CombinedCrossEntropyDiceLoss
from src_torch.metrics import confusion_matrix, metrics_from_confusion
from src_torch.model import ARCHITECTURES, build_model, count_parameters, normalize_architecture, scaled_filters


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None) -> tuple[float, dict]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_seen = 0
    cm = torch.zeros((6, 6), dtype=torch.int64, device=device)

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            with autocast(enabled=scaler is not None):
                logits = model(images)
                loss = criterion(logits, masks)

            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        batch_size = images.shape[0]
        total_loss += float(loss.detach().item()) * batch_size
        total_seen += batch_size
        preds = logits.detach().argmax(dim=1)
        cm += confusion_matrix(preds, masks, num_classes=6)

    return total_loss / max(total_seen, 1), metrics_from_confusion(cm.detach().cpu())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", type=Path)
    parser.add_argument("--output-dir", default="outputs/torch_unet", type=Path)
    parser.add_argument("--architecture", default="light_unet",
                        help=f"Model architecture. Default is the paper lightweight U-Net. Options: {', '.join(ARCHITECTURES)}.")
    parser.add_argument("--width-multiplier", default=1.0, type=float,
                        help="Scale the base U-Net channel widths; 1.0 matches the paper model.")
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr-factor", default=0.1, type=float)
    parser.add_argument("--lr-patience", default=10, type=int)
    parser.add_argument("--min-lr", default=1e-6, type=float)
    parser.add_argument("--early-stop-patience", default=15, type=int,
                        help="Set to 0 to disable early stopping.")
    parser.add_argument("--early-stop-min-delta", default=0.0, type=float)
    parser.add_argument("--seed", default=47, type=int)
    parser.add_argument("--validation-fraction", default=0.2, type=float)
    parser.add_argument("--limit-samples", default=None, type=int,
                        help="Use only the first N training samples before train/validation split.")
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--device", default="cpu",
                        help="Use cpu for safe smoke tests; pass cuda explicitly for GPU training.")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--paper-augmentations", action="store_true",
                        help="Enable the depth and TGC augmentations used by the paper notebook.")
    parser.add_argument("--depth-crop-min", default=0.75, type=float)
    parser.add_argument("--depth-zoom-min", default=0.75, type=float)
    parser.add_argument("--tgc-darkness-sigma", default=0.1, type=float)
    parser.add_argument("--tgc-n-lines", default=8, type=int)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = list_paired_samples(
        args.root / "data" / "frames" / "train" / "images",
        args.root / "data" / "frames" / "train" / "masks",
    )
    if args.limit_samples is not None:
        samples = samples[:args.limit_samples]
        if len(samples) < 2:
            raise ValueError("--limit-samples must leave at least 2 samples.")
    train_samples, valid_samples = split_samples(samples, args.validation_fraction, args.seed)

    train_ds = LUSSegmentationDataset(
        train_samples,
        augment=True,
        use_depth_augmentation=args.paper_augmentations,
        depth_crop_min=args.depth_crop_min,
        depth_zoom_min=args.depth_zoom_min,
        use_tgc_augmentation=args.paper_augmentations,
        tgc_darkness_sigma=args.tgc_darkness_sigma,
        tgc_n_lines=args.tgc_n_lines,
    )
    valid_ds = LUSSegmentationDataset(valid_samples, augment=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    args.architecture = normalize_architecture(args.architecture)
    filters = scaled_filters(args.width_multiplier)
    device = torch.device(args.device)
    model = build_model(args.architecture, in_channels=1, num_classes=6, filters=filters).to(device)
    criterion = CombinedCrossEntropyDiceLoss(num_classes=6)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = GradScaler(enabled=use_amp) if use_amp else None

    config = vars(args).copy()
    config["root"] = str(config["root"])
    config["output_dir"] = str(config["output_dir"])
    config["n_train"] = len(train_samples)
    config["n_valid"] = len(valid_samples)
    config["architecture"] = args.architecture
    config["filters"] = filters
    config["parameter_count"] = count_parameters(model)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    best_val = float("inf")
    epochs_without_improvement = 0
    history_path = args.output_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "lr",
                "train_loss",
                "valid_loss",
                "train_mean_dice_ex_bg",
                "valid_mean_dice_ex_bg",
                "valid_pixel_accuracy",
            ],
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            train_loss, train_metrics = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
            with torch.no_grad():
                valid_loss, valid_metrics = run_epoch(model, valid_loader, criterion, device)
            scheduler.step(valid_loss)

            row = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "train_mean_dice_ex_bg": train_metrics["mean_dice_excluding_background"],
                "valid_mean_dice_ex_bg": valid_metrics["mean_dice_excluding_background"],
                "valid_pixel_accuracy": valid_metrics["pixel_accuracy"],
            }
            writer.writerow(row)
            f.flush()
            print(row)

            improved = valid_loss < best_val - args.early_stop_min_delta
            if improved:
                best_val = valid_loss
                epochs_without_improvement = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "valid_loss": valid_loss,
                        "valid_metrics": valid_metrics,
                        "config": config,
                    },
                    args.output_dir / "best.pt",
                )
            else:
                epochs_without_improvement += 1

            if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
                break


if __name__ == "__main__":
    main()
