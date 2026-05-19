from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_metric(path: Path) -> dict:
    metrics = load_json(path)
    config_path = path.parent / "config.json"
    config = load_json(config_path) if config_path.exists() else {}
    return {
        "architecture": metrics.get("architecture") or config.get("architecture") or path.parent.parent.name,
        "seed": config.get("seed", path.parent.name.replace("seed_", "")),
        "output_dir": str(path.parent),
        "checkpoint": metrics.get("checkpoint", ""),
        "parameter_count": metrics.get("parameter_count", config.get("parameter_count", "")),
        "filters": "x".join(str(x) for x in metrics.get("filters", config.get("filters", []))),
        "pixel_accuracy": metrics.get("pixel_accuracy", ""),
        "mean_dice_excluding_background": metrics.get("mean_dice_excluding_background", ""),
        "blas_mae": _mean_abs_blas(metrics.get("blas", [])),
        "blas_corr": _blas_corr(metrics.get("blas", [])),
    }


def _mean_abs_blas(rows: list[dict]) -> float | str:
    if not rows:
        return ""
    errors = [abs(float(row["blas_pred"]) - float(row["blas_target"])) for row in rows]
    return float(np.mean(errors))


def _blas_corr(rows: list[dict]) -> float | str:
    if len(rows) < 2:
        return ""
    pred = np.asarray([float(row["blas_pred"]) for row in rows], dtype=np.float64)
    target = np.asarray([float(row["blas_target"]) for row in rows], dtype=np.float64)
    if np.std(pred) == 0 or np.std(target) == 0:
        return 0.0
    return float(np.corrcoef(pred, target)[0, 1])


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["architecture"])].append(row)

    summary = []
    for architecture, arch_rows in sorted(grouped.items()):
        dice = _numeric(arch_rows, "mean_dice_excluding_background")
        pixel = _numeric(arch_rows, "pixel_accuracy")
        blas = _numeric(arch_rows, "blas_mae")
        params = _numeric(arch_rows, "parameter_count")
        summary.append(
            {
                "architecture": architecture,
                "n_repeats": len(arch_rows),
                "mean_dice_ex_bg_mean": float(np.mean(dice)) if dice.size else "",
                "mean_dice_ex_bg_std": float(np.std(dice)) if dice.size else "",
                "pixel_accuracy_mean": float(np.mean(pixel)) if pixel.size else "",
                "pixel_accuracy_std": float(np.std(pixel)) if pixel.size else "",
                "blas_mae_mean": float(np.mean(blas)) if blas.size else "",
                "blas_mae_std": float(np.std(blas)) if blas.size else "",
                "parameter_count": int(params[0]) if params.size else "",
            }
        )
    return summary


def _numeric(rows: list[dict], key: str) -> np.ndarray:
    vals = []
    for row in rows:
        value = row.get(key, "")
        if value != "":
            vals.append(float(value))
    return np.asarray(vals, dtype=np.float64)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize architecture benchmark test_metrics.json files.")
    parser.add_argument("--root", default=Path("outputs/architecture_benchmark"), type=Path)
    parser.add_argument("--output", default=Path("outputs/architecture_benchmark/architecture_results.csv"), type=Path)
    parser.add_argument("--summary-output", default=Path("outputs/architecture_benchmark/architecture_summary.csv"), type=Path)
    args = parser.parse_args()

    rows = [flatten_metric(path) for path in sorted(args.root.glob("*/seed_*/test_metrics.json"))]
    write_csv(args.output, rows)
    write_csv(args.summary_output, summarize(rows))
    print(f"Wrote {args.output}")
    print(f"Wrote {args.summary_output}")


if __name__ == "__main__":
    main()
