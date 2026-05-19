from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loocv_torch import (
    build_validation_split,
    completed_run_ids,
    grouped_samples,
    list_paired_samples,
    safe_name,
    train_one_fold,
    write_summary,
)
from src_torch.model import normalize_architecture


def make_group_folds(groups: dict, n_folds: int, seed: int) -> list[list[str]]:
    group_names = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(group_names)
    folds = [[] for _ in range(n_folds)]
    fold_counts = [0 for _ in range(n_folds)]

    for group in sorted(group_names, key=lambda g: len(groups[g]), reverse=True):
        fold_idx = min(range(n_folds), key=lambda idx: fold_counts[idx])
        folds[fold_idx].append(group)
        fold_counts[fold_idx] += len(groups[group])

    return [sorted(fold) for fold in folds]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run grouped K-fold CV for LUS segmentation.")
    parser.add_argument("--root", default=".", type=Path)
    parser.add_argument("--output-dir", default=Path("outputs/grouped_kfold/light_unet"), type=Path)
    parser.add_argument("--architecture", default="light_unet")
    parser.add_argument("--width-multiplier", default=1.0, type=float)
    parser.add_argument("--data-scope", default="train_test", choices=["train", "train_test"])
    parser.add_argument("--n-folds", default=5, type=int)
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
    parser.add_argument("--fold-start", default=0, type=int)
    parser.add_argument("--fold-stop", default=None, type=int)
    parser.add_argument("--max-folds", default=None, type=int)
    args = parser.parse_args()
    args.architecture = normalize_architecture(args.architecture)

    if args.n_folds < 2:
        raise ValueError("--n-folds must be at least 2.")

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
    folds = make_group_folds(groups, args.n_folds, args.seed)
    fold_stop = args.n_folds if args.fold_stop is None else min(args.fold_stop, args.n_folds)
    selected = list(enumerate(folds))[args.fold_start:fold_stop]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "architecture": args.architecture,
        "cv_mode": "grouped_kfold",
        "n_folds": args.n_folds,
        "data_scope": args.data_scope,
        "n_images": len(samples),
        "n_groups": len(groups),
        "fold_start": args.fold_start,
        "fold_stop": fold_stop,
        "fold_group_counts": [len(fold) for fold in folds],
        "fold_image_counts": [sum(len(groups[group]) for group in fold) for fold in folds],
        "validation_fraction": args.validation_fraction,
        "depth_augmentation": not args.disable_depth_aug,
        "tgc_augmentation": not args.disable_tgc_aug,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)

    results_path = args.output_dir / "kfold_results.csv"
    done = completed_run_ids(results_path)
    folds_started = 0

    for fold_index, test_group_names in selected:
        fold_label = f"kfold_{fold_index:02d}_{len(test_group_names)}groups"
        run_id = f"fold{fold_index:04d}_{safe_name(fold_label)}"
        if run_id in done:
            print(f"Skipping completed fold: {run_id}", flush=True)
            continue
        if args.max_folds is not None and folds_started >= args.max_folds:
            write_summary(results_path, args.output_dir / "kfold_summary.json")
            return

        test_samples = [sample for group in test_group_names for sample in groups[group]]
        train_group_dict = {group: group_samples for group, group_samples in groups.items() if group not in set(test_group_names)}
        train_samples, valid_samples = build_validation_split(train_group_dict, args.validation_fraction, args.seed + fold_index)
        row = train_one_fold(args, fold_index, fold_label, train_samples, valid_samples, test_samples)
        row["fold_group"] = ",".join(test_group_names)
        row["run_id"] = run_id
        row["data_scope"] = f"grouped_{args.n_folds}fold_{args.data_scope}"
        from loocv_torch import append_result

        append_result(results_path, row)
        done.add(run_id)
        folds_started += 1
        write_summary(results_path, args.output_dir / "kfold_summary.json")

    write_summary(results_path, args.output_dir / "kfold_summary.json")


if __name__ == "__main__":
    main()
