from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src_torch.blas import calc_blas
from src_torch.dataset import LUSSegmentationDataset, Sample, list_paired_samples
from src_torch.losses import CombinedCrossEntropyDiceLoss
from src_torch.metrics import confusion_matrix, metrics_from_confusion
from src_torch.model import build_model, count_parameters, normalize_architecture, scaled_filters


RESULT_FIELDS = [
    "fold_index",
    "fold_group",
    "run_id",
    "architecture",
    "status",
    "timestamp_utc",
    "seed",
    "data_scope",
    "n_train",
    "n_valid",
    "n_test",
    "width_multiplier",
    "filters",
    "parameter_count",
    "epochs_requested",
    "epochs_completed",
    "best_epoch",
    "best_valid_loss",
    "best_valid_mean_dice_ex_bg",
    "test_pixel_accuracy",
    "test_mean_dice_ex_bg",
    "test_blas_mae",
    "test_blas_corr",
    "elapsed_sec",
    "output_dir",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_group(sample: Sample) -> str:
    stem = sample.image.stem
    return stem.split("-F", 1)[0]


def grouped_samples(samples: list[Sample]) -> dict[str, list[Sample]]:
    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[sample_group(sample)].append(sample)
    return dict(groups)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def build_validation_split(
    train_groups: dict[str, list[Sample]],
    validation_fraction: float,
    seed: int,
) -> tuple[list[Sample], list[Sample]]:
    group_names = sorted(train_groups)
    rng = random.Random(seed)
    rng.shuffle(group_names)
    total = sum(len(train_groups[group]) for group in group_names)
    target_valid_size = max(1, int(round(total * validation_fraction)))

    valid_names: list[str] = []
    valid_count = 0
    while group_names and valid_count < target_valid_size:
        group = group_names.pop(0)
        valid_names.append(group)
        valid_count += len(train_groups[group])

    train_samples = [sample for group in group_names for sample in train_groups[group]]
    valid_samples = [sample for group in valid_names for sample in train_groups[group]]
    if not train_samples or not valid_samples:
        raise ValueError("Could not build a non-empty train/validation split for LOOCV.")
    return train_samples, valid_samples


def completed_run_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {row["run_id"] for row in csv.DictReader(handle) if row.get("status") == "completed"}


def append_result(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})


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


def evaluate_test(model, loader, device) -> dict:
    model.eval()
    cm = torch.zeros((6, 6), dtype=torch.int64, device=device)
    blas_pred: list[float] = []
    blas_target: list[float] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            logits = model(images)
            preds = logits.argmax(dim=1)
            cm += confusion_matrix(preds, masks, num_classes=6)

            for pred, target in zip(preds.cpu().numpy(), masks.cpu().numpy()):
                blas_pred.append(calc_blas(pred.astype(np.uint8)))
                blas_target.append(calc_blas(target.astype(np.uint8)))

    metrics = metrics_from_confusion(cm.cpu())
    pred_arr = np.asarray(blas_pred, dtype=np.float64)
    target_arr = np.asarray(blas_target, dtype=np.float64)
    metrics["blas_mae"] = float(np.mean(np.abs(pred_arr - target_arr))) if pred_arr.size else 0.0
    if pred_arr.size > 1 and np.std(pred_arr) > 0 and np.std(target_arr) > 0:
        metrics["blas_corr"] = float(np.corrcoef(pred_arr, target_arr)[0, 1])
    else:
        metrics["blas_corr"] = 0.0
    return metrics


def train_one_fold(args, fold_index: int, fold_group: str, train_samples: list[Sample], valid_samples: list[Sample], test_samples: list[Sample]) -> dict:
    seed = args.seed + fold_index
    set_seed(seed)
    architecture = normalize_architecture(args.architecture)
    filters = scaled_filters(args.width_multiplier)
    run_id = f"fold{fold_index:04d}_{safe_name(fold_group)}"
    run_dir = args.output_dir / "folds" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ds = LUSSegmentationDataset(
        train_samples,
        augment=True,
        use_depth_augmentation=not args.disable_depth_aug,
        depth_crop_min=args.depth_crop_min,
        depth_zoom_min=args.depth_zoom_min,
        use_tgc_augmentation=not args.disable_tgc_aug,
        tgc_darkness_sigma=args.tgc_darkness_sigma,
        tgc_n_lines=args.tgc_n_lines,
    )
    valid_ds = LUSSegmentationDataset(valid_samples, augment=False)
    test_ds = LUSSegmentationDataset(test_samples, augment=False)

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
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    device = torch.device(args.device)
    model = build_model(architecture, in_channels=1, num_classes=6, filters=filters).to(device)
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

    config = {
        **vars(args),
        "root": str(args.root),
        "output_dir": str(args.output_dir),
        "run_dir": str(run_dir),
        "fold_index": fold_index,
        "fold_group": fold_group,
        "seed": seed,
        "architecture": architecture,
        "filters": filters,
        "parameter_count": count_parameters(model),
        "n_train": len(train_samples),
        "n_valid": len(valid_samples),
        "n_test": len(test_samples),
        "train_names": [sample.image.name for sample in train_samples],
        "valid_names": [sample.image.name for sample in valid_samples],
        "test_names": [sample.image.name for sample in test_samples],
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")

    best_valid_loss = float("inf")
    best_epoch = 0
    best_valid_metrics: dict = {}
    epochs_without_improvement = 0
    history_path = run_dir / "history.csv"
    start = time.time()

    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
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

        epoch = 0
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
            handle.flush()
            print(json.dumps({"run_id": run_id, **row}), flush=True)

            improved = valid_loss < best_valid_loss - args.early_stop_min_delta
            if improved:
                best_valid_loss = valid_loss
                best_epoch = epoch
                best_valid_metrics = valid_metrics
                epochs_without_improvement = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "valid_loss": valid_loss,
                        "valid_metrics": valid_metrics,
                        "config": config,
                    },
                    run_dir / "best.pt",
                )
            else:
                epochs_without_improvement += 1

            if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
                break

    checkpoint = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    test_metrics = evaluate_test(model, test_loader, device)
    (run_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")

    return {
        "fold_index": fold_index,
        "fold_group": fold_group,
        "run_id": run_id,
        "architecture": architecture,
        "status": "completed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "data_scope": args.data_scope,
        "n_train": len(train_samples),
        "n_valid": len(valid_samples),
        "n_test": len(test_samples),
        "width_multiplier": args.width_multiplier,
        "filters": "x".join(str(x) for x in filters),
        "parameter_count": config["parameter_count"],
        "epochs_requested": args.epochs,
        "epochs_completed": epoch,
        "best_epoch": best_epoch,
        "best_valid_loss": best_valid_loss,
        "best_valid_mean_dice_ex_bg": best_valid_metrics["mean_dice_excluding_background"],
        "test_pixel_accuracy": test_metrics["pixel_accuracy"],
        "test_mean_dice_ex_bg": test_metrics["mean_dice_excluding_background"],
        "test_blas_mae": test_metrics["blas_mae"],
        "test_blas_corr": test_metrics["blas_corr"],
        "elapsed_sec": round(time.time() - start, 3),
        "output_dir": str(run_dir),
    }


def write_summary(results_path: Path, summary_path: Path) -> None:
    if not results_path.exists():
        return
    rows = []
    with results_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") == "completed":
                rows.append(row)
    if not rows:
        return

    dice = np.asarray([float(row["test_mean_dice_ex_bg"]) for row in rows], dtype=np.float64)
    pixel = np.asarray([float(row["test_pixel_accuracy"]) for row in rows], dtype=np.float64)
    blas = np.asarray([float(row["test_blas_mae"]) for row in rows], dtype=np.float64)
    summary = {
        "completed_folds": len(rows),
        "architecture": rows[0].get("architecture", "light_unet"),
        "data_scope": rows[0].get("data_scope", ""),
        "mean_test_dice_ex_bg": float(np.mean(dice)),
        "std_test_dice_ex_bg": float(np.std(dice)),
        "mean_test_pixel_accuracy": float(np.mean(pixel)),
        "std_test_pixel_accuracy": float(np.std(pixel)),
        "mean_test_blas_mae": float(np.mean(blas)),
        "std_test_blas_mae": float(np.std(blas)),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run resumable leave-one-group-out CV for LUS segmentation.")
    parser.add_argument("--root", default=".", type=Path)
    parser.add_argument("--output-dir", default=Path("outputs/loocv/light_unet"), type=Path)
    parser.add_argument("--architecture", default="light_unet",
                        help="Architecture name. Default is the paper lightweight U-Net.")
    parser.add_argument("--width-multiplier", default=1.0, type=float)
    parser.add_argument("--data-scope", default="train_test", choices=["train", "train_test"],
                        help="train_test uses every labelled frame but is cross-validation, not the paper hold-out test.")
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr-factor", default=0.5, type=float)
    parser.add_argument("--lr-patience", default=8, type=int)
    parser.add_argument("--min-lr", default=1e-7, type=float)
    parser.add_argument("--early-stop-patience", default=25, type=int)
    parser.add_argument("--early-stop-min-delta", default=1e-4, type=float)
    parser.add_argument("--validation-fraction", default=0.2, type=float)
    parser.add_argument("--seed", default=47, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--disable-depth-aug", action="store_true")
    parser.add_argument("--disable-tgc-aug", action="store_true")
    parser.add_argument("--depth-crop-min", default=0.75, type=float)
    parser.add_argument("--depth-zoom-min", default=0.75, type=float)
    parser.add_argument("--tgc-darkness-sigma", default=0.1, type=float)
    parser.add_argument("--tgc-n-lines", default=8, type=int)
    parser.add_argument("--max-folds", default=None, type=int,
                        help="Optional cap for one scheduler allocation; rerun to resume.")
    parser.add_argument("--fold-start", default=0, type=int)
    parser.add_argument("--fold-stop", default=None, type=int,
                        help="Exclusive fold index stop; useful for chunking jobs.")
    args = parser.parse_args()
    args.architecture = normalize_architecture(args.architecture)

    samples = list_paired_samples(
        args.root / "data" / "frames" / "train" / "images",
        args.root / "data" / "frames" / "train" / "masks",
    )
    if args.data_scope == "train_test":
        samples += list_paired_samples(
            args.root / "data" / "frames" / "test" / "images",
            args.root / "data" / "frames" / "test" / "masks",
        )

    groups = grouped_samples(samples)
    fold_groups = sorted(groups)
    fold_stop = len(fold_groups) if args.fold_stop is None else min(args.fold_stop, len(fold_groups))
    selected = list(enumerate(fold_groups))[args.fold_start:fold_stop]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "architecture": args.architecture,
        "data_scope": args.data_scope,
        "n_images": len(samples),
        "n_groups": len(groups),
        "fold_start": args.fold_start,
        "fold_stop": fold_stop,
        "validation_fraction": args.validation_fraction,
        "depth_augmentation": not args.disable_depth_aug,
        "tgc_augmentation": not args.disable_tgc_aug,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)

    results_path = args.output_dir / "loocv_results.csv"
    done = completed_run_ids(results_path)
    folds_started = 0

    for fold_index, fold_group in selected:
        run_id = f"fold{fold_index:04d}_{safe_name(fold_group)}"
        if run_id in done:
            print(f"Skipping completed fold: {run_id}", flush=True)
            continue
        if args.max_folds is not None and folds_started >= args.max_folds:
            write_summary(results_path, args.output_dir / "loocv_summary.json")
            return

        test_samples = list(groups[fold_group])
        train_groups = {group: group_samples for group, group_samples in groups.items() if group != fold_group}
        train_samples, valid_samples = build_validation_split(train_groups, args.validation_fraction, args.seed + fold_index)
        row = train_one_fold(args, fold_index, fold_group, train_samples, valid_samples, test_samples)
        append_result(results_path, row)
        done.add(run_id)
        folds_started += 1
        write_summary(results_path, args.output_dir / "loocv_summary.json")

    write_summary(results_path, args.output_dir / "loocv_summary.json")


if __name__ == "__main__":
    main()
