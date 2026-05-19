from __future__ import annotations

import argparse
import csv
import json
import math
import random
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
    "run_id",
    "architecture",
    "status",
    "timestamp_utc",
    "seed",
    "data_size_target",
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


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_architecture_list(value: str) -> list[str]:
    return [normalize_architecture(x.strip()) for x in value.split(",") if x.strip()]


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


def grouped_scaling_split(
    samples: list[Sample],
    target_train_size: int,
    validation_fraction: float,
    seed: int,
) -> tuple[list[Sample], list[Sample]]:
    """Build nested, sequence-disjoint train/validation splits.

    The dataset has multiple frames per sequence/patient-like ID. Selecting by
    group avoids the frame-level leakage risk while preserving deterministic,
    nested training pools across scaling sizes for each seed.
    """
    groups = grouped_samples(samples)
    group_names = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(group_names)

    target_valid_size = max(1, int(round(len(samples) * validation_fraction)))
    valid_groups: list[str] = []
    valid_count = 0
    while group_names and valid_count < target_valid_size:
        group = group_names.pop(0)
        valid_groups.append(group)
        valid_count += len(groups[group])

    train_samples: list[Sample] = []
    for group in group_names:
        if len(train_samples) >= target_train_size:
            break
        train_samples.extend(groups[group])

    valid_samples = [sample for group in valid_groups for sample in groups[group]]
    if not train_samples or not valid_samples:
        raise ValueError("Could not build a non-empty grouped train/validation split.")
    return train_samples, valid_samples


def run_identifier(architecture: str, target_size: int, seed: int, width: float) -> str:
    suffix = f"n{target_size:04d}_seed{seed}_w{width:g}"
    if architecture == "light_unet":
        return suffix
    return f"{architecture}_{suffix}"


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


def append_result(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    if exists:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            existing_fields = reader.fieldnames or []
            if existing_fields != RESULT_FIELDS:
                old_rows = list(reader)
                with path.open("w", newline="", encoding="utf-8") as rewrite:
                    writer = csv.DictWriter(rewrite, fieldnames=RESULT_FIELDS)
                    writer.writeheader()
                    for old_row in old_rows:
                        migrated = {field: old_row.get(field, "") for field in RESULT_FIELDS}
                        if not migrated["architecture"]:
                            migrated["architecture"] = "light_unet"
                        writer.writerow(migrated)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})


def completed_run_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {row["run_id"] for row in csv.DictReader(handle) if row.get("status") == "completed"}


def train_one_run(args, train_samples, valid_samples, test_samples, architecture, seed, target_size, width) -> dict:
    set_seed(seed)
    filters = scaled_filters(width)
    run_id = run_identifier(architecture, target_size, seed, width)
    run_dir = args.output_dir / "runs" / run_id
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
        "architecture": architecture,
        "seed": seed,
        "data_size_target": target_size,
        "n_train": len(train_samples),
        "n_valid": len(valid_samples),
        "n_test": len(test_samples),
        "width_multiplier": width,
        "filters": filters,
        "parameter_count": count_parameters(model),
        "train_names": [sample.image.name for sample in train_samples],
        "valid_names": [sample.image.name for sample in valid_samples],
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
        "run_id": run_id,
        "architecture": architecture,
        "status": "completed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "data_size_target": target_size,
        "n_train": len(train_samples),
        "n_valid": len(valid_samples),
        "n_test": len(test_samples),
        "width_multiplier": width,
        "filters": "x".join(str(x) for x in filters),
        "parameter_count": config["parameter_count"],
        "epochs_requested": args.epochs,
        "epochs_completed": checkpoint["epoch"] if checkpoint["epoch"] > best_epoch else epoch,
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


def fit_scaling_curves(results_path: Path, output_path: Path) -> None:
    if not results_path.exists():
        return

    rows = []
    with results_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") == "completed":
                rows.append(row)

    grouped: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        architecture = row.get("architecture") or "light_unet"
        width = row["width_multiplier"]
        n_train = int(row["n_train"])
        error = 1.0 - float(row["test_mean_dice_ex_bg"])
        grouped[(architecture, width)][n_train].append(error)

    output_rows = []
    for (architecture, width), values_by_n in grouped.items():
        xs = np.asarray(sorted(values_by_n), dtype=np.float64)
        ys = np.asarray([np.mean(values_by_n[int(x)]) for x in xs], dtype=np.float64)
        if len(xs) < 3 or np.any(ys <= 0):
            continue

        best = None
        max_l_inf = max(0.0, float(np.min(ys) * 0.95))
        for l_inf in np.linspace(0.0, max_l_inf, 200):
            adjusted = ys - l_inf
            if np.any(adjusted <= 0):
                continue
            slope, intercept = np.polyfit(np.log(xs), np.log(adjusted), 1)
            alpha = -float(slope)
            a = float(math.exp(intercept))
            pred = l_inf + a * np.power(xs, -alpha)
            sse = float(np.sum((ys - pred) ** 2))
            if best is None or sse < best["sse"]:
                best = {"l_inf": float(l_inf), "a": a, "alpha": alpha, "sse": sse}

        if best is None:
            continue
        output_rows.append({
            "architecture": architecture,
            "width_multiplier": width,
            "n_points": len(xs),
            "min_n_train": int(xs.min()),
            "max_n_train": int(xs.max()),
            "metric": "1 - test_mean_dice_ex_bg",
            **best,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["architecture", "width_multiplier", "n_points", "min_n_train", "max_n_train", "metric", "l_inf", "a", "alpha", "sse"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run resumable LUS U-Net scaling-law experiments.")
    parser.add_argument("--root", default=".", type=Path)
    parser.add_argument("--output-dir", default=Path("outputs/scaling_law"), type=Path)
    parser.add_argument("--architecture", default="light_unet",
                        help="Architecture name or comma-separated names. Default is the paper lightweight U-Net.")
    parser.add_argument("--sizes", default="32,64,128,192,256,320,370")
    parser.add_argument("--seeds", default="47,48,49,50,51")
    parser.add_argument("--width-multipliers", default="0.5,1.0,1.5")
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr-factor", default=0.5, type=float)
    parser.add_argument("--lr-patience", default=8, type=int)
    parser.add_argument("--min-lr", default=1e-7, type=float)
    parser.add_argument("--early-stop-patience", default=25, type=int)
    parser.add_argument("--early-stop-min-delta", default=1e-4, type=float)
    parser.add_argument("--validation-fraction", default=0.2, type=float)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--disable-depth-aug", action="store_true")
    parser.add_argument("--disable-tgc-aug", action="store_true")
    parser.add_argument("--depth-crop-min", default=0.75, type=float)
    parser.add_argument("--depth-zoom-min", default=0.75, type=float)
    parser.add_argument("--tgc-darkness-sigma", default=0.1, type=float)
    parser.add_argument("--tgc-n-lines", default=8, type=int)
    parser.add_argument("--max-runs", default=None, type=int, help="Optional cap for one scheduler allocation; rerun to resume.")
    args = parser.parse_args()

    train_all = list_paired_samples(
        args.root / "data" / "frames" / "train" / "images",
        args.root / "data" / "frames" / "train" / "masks",
    )
    test_samples = list_paired_samples(
        args.root / "data" / "frames" / "test" / "images",
        args.root / "data" / "frames" / "test" / "masks",
    )
    sizes = [min(size, len(train_all)) for size in parse_int_list(args.sizes)]
    seeds = parse_int_list(args.seeds)
    widths = parse_float_list(args.width_multipliers)
    architectures = parse_architecture_list(args.architecture)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "train_images": len(train_all),
        "test_images": len(test_samples),
        "train_groups": len(grouped_samples(train_all)),
        "test_groups": len(grouped_samples(test_samples)),
        "architectures": architectures,
        "sizes": sizes,
        "seeds": seeds,
        "width_multipliers": widths,
        "depth_augmentation": not args.disable_depth_aug,
        "tgc_augmentation": not args.disable_tgc_aug,
        "depth_crop_min": args.depth_crop_min,
        "depth_zoom_min": args.depth_zoom_min,
        "tgc_darkness_sigma": args.tgc_darkness_sigma,
        "tgc_n_lines": args.tgc_n_lines,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)

    results_path = args.output_dir / "results.csv"
    done = completed_run_ids(results_path)
    runs_started = 0

    for architecture in architectures:
        for width in widths:
            for seed in seeds:
                for target_size in sizes:
                    run_id = run_identifier(architecture, target_size, seed, width)
                    if run_id in done:
                        print(f"Skipping completed run: {run_id}", flush=True)
                        continue
                    if args.max_runs is not None and runs_started >= args.max_runs:
                        fit_scaling_curves(results_path, args.output_dir / "scaling_fit.csv")
                        return

                    train_samples, valid_samples = grouped_scaling_split(
                        train_all,
                        target_train_size=target_size,
                        validation_fraction=args.validation_fraction,
                        seed=seed,
                    )
                    row = train_one_run(args, train_samples, valid_samples, test_samples, architecture, seed, target_size, width)
                    append_result(results_path, row)
                    done.add(run_id)
                    runs_started += 1
                    fit_scaling_curves(results_path, args.output_dir / "scaling_fit.csv")

    fit_scaling_curves(results_path, args.output_dir / "scaling_fit.csv")


if __name__ == "__main__":
    main()
