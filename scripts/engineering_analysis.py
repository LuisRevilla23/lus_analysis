from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src_torch.blas import calc_blas, roi_bbox


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
    names = sorted(set(images) & set(masks))
    return [(images[name], masks[name]) for name in names]


def parse_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(v) for v in text.split(","))


def load_mask(path: Path, crop: tuple[int, int, int, int], size: tuple[int, int]) -> np.ndarray:
    mask = Image.open(path).convert("L").crop(crop)
    mask = mask.resize(size[::-1], resample=Image.Resampling.NEAREST)
    return np.asarray(mask, dtype=np.uint8)


def load_image(path: Path, crop: tuple[int, int, int, int], size: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("L").crop(crop)
    image = image.resize(size[::-1], resample=Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def preprocess_image(path: Path, crop: tuple[int, int, int, int], size: tuple[int, int]) -> np.ndarray:
    arr = load_image(path, crop, size).astype(np.float32) / 255.0
    return arr[None, :, :, None]


def colorize(mask: np.ndarray) -> np.ndarray:
    return CLASS_COLORS[np.clip(mask, 0, len(CLASS_COLORS) - 1)]


def category(value: float) -> str:
    if value < 0.5:
        return "low"
    if value <= 0.9:
        return "intermediate"
    return "high"


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    def ranks(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a)
        r = np.empty_like(order, dtype=np.float64)
        r[order] = np.arange(len(a), dtype=np.float64)
        return r

    return pearson(ranks(x), ranks(y))


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_rows(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_blas_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def blas_agreement(blas_csv: Path, output_dir: Path) -> list[dict]:
    import matplotlib.pyplot as plt

    rows = load_blas_rows(blas_csv)
    y_true = np.array([float(r["blas_target"]) for r in rows])
    y_pred = np.array([float(r["blas_pred"]) for r in rows])
    err = y_pred - y_true
    abs_err = np.abs(err)

    categories_true = [category(v) for v in y_true]
    categories_pred = [category(v) for v in y_pred]
    category_errors = sum(a != b for a, b in zip(categories_true, categories_pred))

    summary = {
        "n": len(rows),
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias_pred_minus_target": float(err.mean()),
        "pearson_r": pearson(y_true, y_pred),
        "spearman_r": spearman(y_true, y_pred),
        "category_disagreement_count": int(category_errors),
        "category_disagreement_fraction": float(category_errors / max(len(rows), 1)),
    }
    save_json(summary, output_dir / "blas_agreement_summary.json")

    enriched = []
    for r, diff, ae, ct, cp in zip(rows, err, abs_err, categories_true, categories_pred):
        enriched.append(
            {
                "name": r["name"],
                "blas_target": float(r["blas_target"]),
                "blas_pred": float(r["blas_pred"]),
                "blas_error": float(diff),
                "blas_abs_error": float(ae),
                "target_category": ct,
                "pred_category": cp,
                "category_error": ct != cp,
            }
        )
    write_rows(enriched, output_dir / "blas_agreement_cases.csv")

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.scatter(y_true, y_pred, s=28, alpha=0.75, edgecolors="none")
    ax.plot([0, 1], [0, 1], color="black", linewidth=1)
    ax.set(xlabel="Manual-label BLAS", ylabel="Predicted-mask BLAS", xlim=(-0.03, 1.03), ylim=(-0.03, 1.03))
    ax.set_title(f"BLAS agreement (r={summary['pearson_r']:.2f}, MAE={summary['mae']:.2f})")
    fig.tight_layout()
    fig.savefig(output_dir / "blas_scatter.png", dpi=300)
    plt.close(fig)

    mean = (y_true + y_pred) / 2
    loa = 1.96 * err.std(ddof=1)
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    ax.scatter(mean, err, s=28, alpha=0.75, edgecolors="none")
    ax.axhline(err.mean(), color="black", linewidth=1, label="Bias")
    ax.axhline(err.mean() + loa, color="tab:red", linestyle="--", linewidth=1, label="95% limits")
    ax.axhline(err.mean() - loa, color="tab:red", linestyle="--", linewidth=1)
    ax.set(xlabel="Mean BLAS", ylabel="Predicted - manual BLAS")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "blas_bland_altman.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.hist(abs_err, bins=np.linspace(0, 1, 21), color="tab:blue", alpha=0.85)
    ax.set(xlabel="Absolute BLAS error", ylabel="Number of frames")
    fig.tight_layout()
    fig.savefig(output_dir / "blas_abs_error_hist.png", dpi=300)
    plt.close(fig)

    return enriched


def make_failure_panels(
    cases: list[dict],
    root: Path,
    pred_dir: Path,
    output_dir: Path,
    crop: tuple[int, int, int, int],
    size: tuple[int, int],
    n_cases: int,
) -> None:
    import matplotlib.pyplot as plt

    panel_dir = output_dir / "failure_cases"
    panel_dir.mkdir(parents=True, exist_ok=True)
    worst = sorted(cases, key=lambda r: r["blas_abs_error"], reverse=True)[:n_cases]
    write_rows(worst, panel_dir / "top_failure_cases.csv")

    for rank, row in enumerate(worst, start=1):
        name = row["name"]
        image = load_image(root / "data" / "frames" / "test" / "images" / name, crop, size)
        target = load_mask(root / "data" / "frames" / "test" / "masks" / name, crop, size)
        pred = np.asarray(Image.open(pred_dir / name).convert("L"), dtype=np.uint8)
        error = np.zeros((*target.shape, 3), dtype=np.uint8)
        error[(target == pred) & (target > 0)] = [210, 210, 210]
        error[(target != pred) & (pred > 0)] = [230, 25, 75]
        error[(target != pred) & (target > 0)] = [0, 130, 200]

        fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.2))
        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("Image")
        axes[1].imshow(colorize(target))
        axes[1].set_title("Manual mask")
        axes[2].imshow(colorize(pred))
        axes[2].set_title("Predicted mask")
        axes[3].imshow(error)
        axes[3].set_title("Error map")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(
            f"{rank}. {name} | target={row['blas_target']:.2f}, pred={row['blas_pred']:.2f}, "
            f"|error|={row['blas_abs_error']:.2f}",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(panel_dir / f"{rank:02d}_{Path(name).stem}.png", dpi=250)
        plt.close(fig)


def perturb_remove_blines(mask: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    out = mask.copy()
    ys, xs = np.where((mask == 4) | (mask == 5))
    n = int(round(len(ys) * fraction))
    if n > 0:
        idx = rng.choice(len(ys), size=n, replace=False)
        out[ys[idx], xs[idx]] = 0
    return out


def perturb_add_blines(mask: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    out = mask.copy()
    bbox = roi_bbox(mask)
    if bbox is None:
        return out
    top, bottom, left, right = bbox
    roi_area = max((bottom - top) * (right - left), 1)
    n = int(round(roi_area * fraction))
    if n <= 0:
        return out
    ys = rng.integers(top, bottom, size=n)
    xs = rng.integers(left, right, size=n)
    out[ys, xs] = 4
    return out


def morphology(mask: np.ndarray, iterations: int, mode: str) -> np.ndarray:
    try:
        from scipy import ndimage
    except Exception:
        return mask.copy()

    out = mask.copy()
    b = (mask == 4) | (mask == 5)
    if mode == "dilate":
        changed = ndimage.binary_dilation(b, iterations=iterations)
    else:
        changed = ndimage.binary_erosion(b, iterations=iterations)
    out[b] = 0
    out[changed] = 4
    out[mask == 5] = np.where(changed[mask == 5], 5, out[mask == 5])
    return out


def sensitivity_analysis(
    root: Path,
    output_dir: Path,
    crop: tuple[int, int, int, int],
    size: tuple[int, int],
    repeats: int,
    seed: int,
) -> None:
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    pairs = list_pairs(root / "data" / "frames" / "test" / "images", root / "data" / "frames" / "test" / "masks")
    masks = [(mask_path.name, load_mask(mask_path, crop, size)) for _, mask_path in pairs]

    rows = []
    fractions = np.linspace(0, 0.8, 9)
    for name, mask in masks:
        base = calc_blas(mask)
        for frac in fractions:
            for rep in range(repeats):
                removed = perturb_remove_blines(mask, float(frac), rng)
                added = perturb_add_blines(mask, float(frac) * 0.08, rng)
                rows.append(
                    {
                        "name": name,
                        "perturbation": "remove_bline_pixels",
                        "level": float(frac),
                        "repeat": rep,
                        "baseline_blas": base,
                        "perturbed_blas": calc_blas(removed),
                        "delta_blas": calc_blas(removed) - base,
                    }
                )
                rows.append(
                    {
                        "name": name,
                        "perturbation": "add_false_positive_bline_pixels",
                        "level": float(frac) * 0.08,
                        "repeat": rep,
                        "baseline_blas": base,
                        "perturbed_blas": calc_blas(added),
                        "delta_blas": calc_blas(added) - base,
                    }
                )

        for mode in ["erode", "dilate"]:
            for it in range(1, 6):
                perturbed = morphology(mask, it, mode)
                rows.append(
                    {
                        "name": name,
                        "perturbation": f"{mode}_bline_regions",
                        "level": it,
                        "repeat": 0,
                        "baseline_blas": base,
                        "perturbed_blas": calc_blas(perturbed),
                        "delta_blas": calc_blas(perturbed) - base,
                    }
                )

    write_rows(rows, output_dir / "sensitivity_results.csv")

    summary_rows = []
    for perturbation in sorted({r["perturbation"] for r in rows}):
        levels = sorted({float(r["level"]) for r in rows if r["perturbation"] == perturbation})
        for level in levels:
            vals = np.array(
                [float(r["delta_blas"]) for r in rows if r["perturbation"] == perturbation and float(r["level"]) == level]
            )
            summary_rows.append(
                {
                    "perturbation": perturbation,
                    "level": level,
                    "mean_delta_blas": float(vals.mean()),
                    "median_delta_blas": float(np.median(vals)),
                    "p05_delta_blas": float(np.percentile(vals, 5)),
                    "p95_delta_blas": float(np.percentile(vals, 95)),
                }
            )
    write_rows(summary_rows, output_dir / "sensitivity_summary.csv")

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
    axes = axes.ravel()
    for ax, perturbation in zip(axes, sorted({r["perturbation"] for r in rows})):
        sub = [r for r in summary_rows if r["perturbation"] == perturbation]
        x = np.array([float(r["level"]) for r in sub])
        y = np.array([float(r["mean_delta_blas"]) for r in sub])
        lo = np.array([float(r["p05_delta_blas"]) for r in sub])
        hi = np.array([float(r["p95_delta_blas"]) for r in sub])
        ax.plot(x, y, marker="o")
        ax.fill_between(x, lo, hi, alpha=0.2)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(perturbation.replace("_", " "))
        ax.set_xlabel("Perturbation level")
        ax.set_ylabel("Delta BLAS")
    fig.tight_layout()
    fig.savefig(output_dir / "sensitivity_curves.png", dpi=300)
    plt.close(fig)


def load_or_predict_probs(
    root: Path,
    weights: Path,
    output_dir: Path,
    crop: tuple[int, int, int, int],
    size: tuple[int, int],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    cache = output_dir / "test_probabilities.npz"
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        return list(data["names"]), data["probs"], data["targets"]

    import tensorflow as tf  # noqa: F401
    from src import Models as models

    pairs = list_pairs(root / "data" / "frames" / "test" / "images", root / "data" / "frames" / "test" / "masks")
    model = models.unet((size[0], size[1], 1), len(CLASS_NAMES), filters=[32, 64, 128, 256, 512])
    model.load_weights(str(weights))

    names, probs, targets = [], [], []
    for image_path, mask_path in pairs:
        p = model.predict(preprocess_image(image_path, crop, size), verbose=0)[0].astype(np.float32)
        names.append(image_path.name)
        probs.append(p)
        targets.append(load_mask(mask_path, crop, size))
    probs_arr = np.stack(probs)
    targets_arr = np.stack(targets)
    np.savez_compressed(cache, names=np.array(names), probs=probs_arr, targets=targets_arr)
    return names, probs_arr, targets_arr


def conformal_uncertainty(
    root: Path,
    weights: Path,
    blas_cases: list[dict],
    output_dir: Path,
    crop: tuple[int, int, int, int],
    size: tuple[int, int],
    alpha: float,
    seed: int,
) -> None:
    names, probs, targets = load_or_predict_probs(root, weights, output_dir, crop, size)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(names))
    rng.shuffle(idx)
    split = max(1, len(idx) // 2)
    cal_idx, eval_idx = idx[:split], idx[split:]

    cal_probs = probs[cal_idx]
    cal_targets = targets[cal_idx]
    flat_probs = cal_probs.reshape(-1, cal_probs.shape[-1])
    flat_targets = cal_targets.reshape(-1)
    true_probs = flat_probs[np.arange(len(flat_targets)), flat_targets]
    scores = 1.0 - true_probs
    q = float(np.quantile(scores, min(1.0, np.ceil((len(scores) + 1) * (1 - alpha)) / len(scores)), method="higher"))
    prob_threshold = 1.0 - q

    blas_by_name = {r["name"]: r for r in blas_cases}
    rows = []
    for i in eval_idx:
        p = probs[i]
        t = targets[i]
        pred = p.argmax(axis=-1)
        sets = p >= prob_threshold
        true_in_set = sets.reshape(-1, sets.shape[-1])[np.arange(t.size), t.reshape(-1)]
        max_prob = p.max(axis=-1)
        entropy = -(p * np.log(np.clip(p, 1e-8, 1))).sum(axis=-1)
        case = blas_by_name[names[i]]
        rows.append(
            {
                "name": names[i],
                "split": "heldout_eval",
                "conformal_alpha": alpha,
                "probability_threshold": prob_threshold,
                "pixel_coverage": float(true_in_set.mean()),
                "mean_prediction_set_size": float(sets.sum(axis=-1).mean()),
                "fraction_ambiguous_pixels": float((sets.sum(axis=-1) > 1).mean()),
                "mean_max_probability": float(max_prob.mean()),
                "mean_entropy": float(entropy.mean()),
                "pixel_accuracy": float((pred == t).mean()),
                "blas_abs_error": float(case["blas_abs_error"]),
                "category_error": bool(case["category_error"]),
            }
        )

    write_rows(rows, output_dir / "conformal_uncertainty_cases.csv")
    coverage = np.array([r["pixel_coverage"] for r in rows])
    set_size = np.array([r["mean_prediction_set_size"] for r in rows])
    entropy = np.array([r["mean_entropy"] for r in rows])
    blas_error = np.array([r["blas_abs_error"] for r in rows])
    summary = {
        "alpha": alpha,
        "calibration_frames": int(len(cal_idx)),
        "heldout_eval_frames": int(len(eval_idx)),
        "probability_threshold": prob_threshold,
        "mean_pixel_coverage": float(coverage.mean()),
        "mean_prediction_set_size": float(set_size.mean()),
        "pearson_entropy_vs_blas_abs_error": pearson(entropy, blas_error),
        "spearman_entropy_vs_blas_abs_error": spearman(entropy, blas_error),
    }
    save_json(summary, output_dir / "conformal_uncertainty_summary.json")

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5.8, 4.4))
        ax.scatter(entropy, blas_error, s=34, alpha=0.8, edgecolors="none")
        ax.set_xlabel("Mean predictive entropy")
        ax.set_ylabel("Absolute BLAS error")
        ax.set_title(f"Uncertainty vs BLAS error (Spearman={summary['spearman_entropy_vs_blas_abs_error']:.2f})")
        fig.tight_layout()
        fig.savefig(output_dir / "uncertainty_vs_blas_error.png", dpi=300)
        plt.close(fig)
    except ModuleNotFoundError:
        pass


def mask_from_bbox(shape: tuple[int, int], bbox: tuple[int, int, int, int] | None) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    if bbox is None:
        return out
    top, bottom, left, right = bbox
    out[top:bottom, left:right] = True
    return out


def conformal_blas_refinement(
    root: Path,
    weights: Path,
    blas_cases: list[dict],
    output_dir: Path,
    crop: tuple[int, int, int, int],
    size: tuple[int, int],
    alpha: float,
    seed: int,
) -> None:
    import matplotlib.pyplot as plt

    names, probs, targets = load_or_predict_probs(root, weights, output_dir, crop, size)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(names))
    rng.shuffle(idx)
    split = max(1, len(idx) // 2)
    cal_idx, eval_idx = idx[:split], idx[split:]

    subsets = {
        "all_pixels": lambda t: np.ones(t.shape, dtype=bool),
        "foreground_pixels": lambda t: t > 0,
        "manual_blas_roi": lambda t: mask_from_bbox(t.shape, roi_bbox(t)),
        "manual_bline_or_confluence": lambda t: (t == 4) | (t == 5),
    }

    thresholds = {}
    for subset_name, subset_fn in subsets.items():
        scores = []
        for i in cal_idx:
            p = probs[i]
            t = targets[i]
            m = subset_fn(t)
            if not m.any():
                continue
            true_probs = p[m][np.arange(int(m.sum())), t[m]]
            scores.append(1.0 - true_probs)
        if scores:
            scores_arr = np.concatenate(scores)
            q = float(
                np.quantile(
                    scores_arr,
                    min(1.0, np.ceil((len(scores_arr) + 1) * (1 - alpha)) / len(scores_arr)),
                    method="higher",
                )
            )
            thresholds[subset_name] = 1.0 - q
        else:
            thresholds[subset_name] = float("nan")

    blas_by_name = {r["name"]: r for r in blas_cases}
    rows = []
    for i in eval_idx:
        p = probs[i]
        t = targets[i]
        max_prob = p.max(axis=-1)
        entropy = -(p * np.log(np.clip(p, 1e-8, 1))).sum(axis=-1)
        pred = p.argmax(axis=-1)
        case = blas_by_name[names[i]]

        row = {
            "name": names[i],
            "blas_abs_error": float(case["blas_abs_error"]),
            "category_error": bool(case["category_error"]),
            "pixel_accuracy_all": float((pred == t).mean()),
            "mean_entropy_all": float(entropy.mean()),
            "mean_max_probability_all": float(max_prob.mean()),
        }
        for subset_name, subset_fn in subsets.items():
            m = subset_fn(t)
            row[f"{subset_name}_n_pixels"] = int(m.sum())
            if not m.any() or np.isnan(thresholds[subset_name]):
                row[f"{subset_name}_coverage"] = float("nan")
                row[f"{subset_name}_mean_set_size"] = float("nan")
                row[f"{subset_name}_mean_entropy"] = float("nan")
                row[f"{subset_name}_mean_max_probability"] = float("nan")
                continue
            sets = p[m] >= thresholds[subset_name]
            true_in_set = sets[np.arange(int(m.sum())), t[m]]
            row[f"{subset_name}_coverage"] = float(true_in_set.mean())
            row[f"{subset_name}_mean_set_size"] = float(sets.sum(axis=-1).mean())
            row[f"{subset_name}_mean_entropy"] = float(entropy[m].mean())
            row[f"{subset_name}_mean_max_probability"] = float(max_prob[m].mean())
        rows.append(row)

    write_rows(rows, output_dir / "conformal_blas_refined_cases.csv")

    summary_rows = []
    blas_error = np.array([float(r["blas_abs_error"]) for r in rows])
    for subset_name in subsets:
        ent = np.array([float(r[f"{subset_name}_mean_entropy"]) for r in rows])
        cov = np.array([float(r[f"{subset_name}_coverage"]) for r in rows])
        valid_ent = ~np.isnan(ent)
        valid_cov = ~np.isnan(cov)
        summary_rows.append(
            {
                "subset": subset_name,
                "probability_threshold": thresholds[subset_name],
                "mean_coverage": float(np.nanmean(cov)) if valid_cov.any() else float("nan"),
                "mean_entropy": float(np.nanmean(ent)) if valid_ent.any() else float("nan"),
                "pearson_entropy_vs_blas_abs_error": pearson(ent[valid_ent], blas_error[valid_ent]) if valid_ent.any() else float("nan"),
                "spearman_entropy_vs_blas_abs_error": spearman(ent[valid_ent], blas_error[valid_ent]) if valid_ent.any() else float("nan"),
            }
        )
    write_rows(summary_rows, output_dir / "conformal_blas_refined_summary.csv")

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    labels = []
    values = []
    for s in summary_rows:
        labels.append(s["subset"].replace("manual_", "").replace("_", "\n"))
        values.append(float(s["spearman_entropy_vs_blas_abs_error"]))
    ax.bar(labels, values, color=["0.55", "tab:green", "tab:blue", "tab:orange"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Spearman rho: entropy vs |BLAS error|")
    ax.set_title("Does localized uncertainty explain BLAS error?")
    fig.tight_layout()
    fig.savefig(output_dir / "conformal_blas_refined_correlations.png", dpi=300)
    plt.close(fig)


def summarize_classes(mask: np.ndarray) -> str:
    labels = [int(v) for v in np.unique(mask) if int(v) != 0]
    if not labels:
        return "none"
    return "; ".join(f"{label}:{CLASS_NAMES[label]}" for label in labels)


def conformal_set_case_panels(
    root: Path,
    weights: Path,
    blas_cases: list[dict],
    output_dir: Path,
    crop: tuple[int, int, int, int],
    size: tuple[int, int],
    alpha: float,
    seed: int,
    n_cases: int,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    names, probs, targets = load_or_predict_probs(root, weights, output_dir, crop, size)
    name_to_idx = {name: i for i, name in enumerate(names)}

    rng = np.random.default_rng(seed)
    idx = np.arange(len(names))
    rng.shuffle(idx)
    cal_idx = idx[: max(1, len(idx) // 2)]

    scores = []
    for i in cal_idx:
        p = probs[i]
        t = targets[i]
        roi = mask_from_bbox(t.shape, roi_bbox(t))
        if not roi.any():
            continue
        true_probs = p[roi][np.arange(int(roi.sum())), t[roi]]
        scores.append(1.0 - true_probs)
    scores_arr = np.concatenate(scores)
    q = float(
        np.quantile(
            scores_arr,
            min(1.0, np.ceil((len(scores_arr) + 1) * (1 - alpha)) / len(scores_arr)),
            method="higher",
        )
    )
    threshold = 1.0 - q

    worst = sorted(blas_cases, key=lambda r: float(r["blas_abs_error"]), reverse=True)[:n_cases]
    panel_dir = output_dir / "conformal_prediction_set_cases"
    panel_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    cmap = ListedColormap(["black", "#009E73", "#0072B2", "#E69F00", "#CC79A7"])
    for rank, row in enumerate(worst, start=1):
        name = row["name"]
        i = name_to_idx[name]
        p = probs[i]
        target = targets[i]
        pred = p.argmax(axis=-1).astype(np.uint8)
        image = load_image(root / "data" / "frames" / "test" / "images" / name, crop, size)
        manual_roi = mask_from_bbox(target.shape, roi_bbox(target))
        pred_roi = mask_from_bbox(pred.shape, roi_bbox(pred))
        roi = manual_roi | pred_roi
        sets = p >= threshold
        set_size = sets.sum(axis=-1)
        includes_bline = sets[..., 4]
        includes_confluence = sets[..., 5]
        bset = np.zeros(target.shape, dtype=np.uint8)
        bset[includes_bline] = 1
        bset[includes_confluence] = 2
        bset[includes_bline & includes_confluence] = 3
        bset[roi & (bset == 0)] = 4

        roi_pixels = max(int(roi.sum()), 1)
        bline_target = (target == 4) | (target == 5)
        bline_pred = (pred == 4) | (pred == 5)
        rows.append(
            {
                "rank": rank,
                "name": name,
                "blas_target": float(row["blas_target"]),
                "blas_pred": float(row["blas_pred"]),
                "blas_abs_error": float(row["blas_abs_error"]),
                "target_category": row["target_category"],
                "pred_category": row["pred_category"],
                "manual_classes_present": summarize_classes(target),
                "pred_classes_present": summarize_classes(pred),
                "roi_probability_threshold": threshold,
                "manual_roi_pixels": int(manual_roi.sum()),
                "pred_roi_pixels": int(pred_roi.sum()),
                "union_roi_pixels": int(roi.sum()),
                "roi_mean_set_size": float(set_size[roi].mean()) if roi.any() else float("nan"),
                "roi_fraction_empty_set": float((set_size[roi] == 0).mean()) if roi.any() else float("nan"),
                "roi_fraction_including_bline": float(includes_bline[roi].mean()) if roi.any() else float("nan"),
                "roi_fraction_including_confluence": float(includes_confluence[roi].mean()) if roi.any() else float("nan"),
                "target_bline_confluence_pixels": int(bline_target.sum()),
                "pred_bline_confluence_pixels": int(bline_pred.sum()),
            }
        )

        fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.8))
        axes = axes.ravel()
        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("Image")
        axes[1].imshow(colorize(target))
        axes[1].set_title("Manual mask")
        axes[2].imshow(colorize(pred))
        axes[2].set_title("Argmax prediction")
        im = axes[3].imshow(set_size, cmap="magma", vmin=0, vmax=6)
        axes[3].set_title("Prediction-set size")
        fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
        axes[4].imshow(bset, cmap=cmap, vmin=0, vmax=4)
        axes[4].set_title("B-line/confluence in set")
        axes[5].imshow(roi, cmap="gray")
        axes[5].set_title("Manual or predicted BLAS ROI")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(
            f"{rank}. {name} | BLAS target={float(row['blas_target']):.2f}, "
            f"pred={float(row['blas_pred']):.2f}, |err|={float(row['blas_abs_error']):.2f}",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(panel_dir / f"{rank:02d}_{Path(name).stem}_prediction_sets.png", dpi=250)
        plt.close(fig)

    write_rows(rows, panel_dir / "prediction_set_case_summary.csv")


def sequence_id(path: Path) -> str:
    stem = path.stem
    if "-F" in stem:
        return stem.split("-F", 1)[0]
    return stem


def image_feature(path: Path, crop: tuple[int, int, int, int], size: int = 64) -> np.ndarray:
    img = Image.open(path).convert("L").crop(crop)
    img = img.resize((size, size), resample=Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = arr - arr.mean()
    denom = np.linalg.norm(arr)
    if denom > 0:
        arr = arr / denom
    return arr.reshape(-1)


def train_test_similarity_analysis(root: Path, output_dir: Path, crop: tuple[int, int, int, int]) -> None:
    import matplotlib.pyplot as plt

    train_paths = sorted((root / "data" / "frames" / "train" / "images").glob("*.png"))
    test_paths = sorted((root / "data" / "frames" / "test" / "images").glob("*.png"))
    train_features = np.stack([image_feature(p, crop) for p in train_paths])
    test_features = np.stack([image_feature(p, crop) for p in test_paths])
    sim = test_features @ train_features.T

    rows = []
    for i, test_path in enumerate(test_paths):
        order = np.argsort(-sim[i])[:5]
        for rank, j in enumerate(order, start=1):
            rows.append(
                {
                    "test_image": test_path.name,
                    "test_sequence": sequence_id(test_path),
                    "train_image": train_paths[j].name,
                    "train_sequence": sequence_id(train_paths[j]),
                    "rank": rank,
                    "cosine_similarity": float(sim[i, j]),
                    "same_sequence_id": sequence_id(test_path) == sequence_id(train_paths[j]),
                }
            )
    write_rows(rows, output_dir / "train_test_nearest_neighbors.csv")

    best = [r for r in rows if int(r["rank"]) == 1]
    train_seq = {sequence_id(p) for p in train_paths}
    test_seq = {sequence_id(p) for p in test_paths}
    shared_seq = sorted(train_seq & test_seq)
    best_sims = np.array([float(r["cosine_similarity"]) for r in best])
    summary = {
        "n_train_images": len(train_paths),
        "n_test_images": len(test_paths),
        "n_train_sequences": len(train_seq),
        "n_test_sequences": len(test_seq),
        "shared_sequence_ids_count": len(shared_seq),
        "shared_sequence_ids": shared_seq[:50],
        "max_nearest_neighbor_cosine_similarity": float(best_sims.max()),
        "median_nearest_neighbor_cosine_similarity": float(np.median(best_sims)),
        "n_test_images_with_nn_similarity_ge_0_99": int((best_sims >= 0.99).sum()),
        "n_test_images_with_nn_similarity_ge_0_97": int((best_sims >= 0.97).sum()),
        "n_test_images_with_nn_similarity_ge_0_95": int((best_sims >= 0.95).sum()),
    }
    save_json(summary, output_dir / "train_test_similarity_summary.json")

    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    ax.hist(best_sims, bins=20, color="tab:purple", alpha=0.85)
    ax.set_xlabel("Nearest train-image cosine similarity")
    ax.set_ylabel("Number of test images")
    ax.set_title("Train-test visual similarity")
    fig.tight_layout()
    fig.savefig(output_dir / "train_test_similarity_hist.png", dpi=300)
    plt.close(fig)

    panel_dir = output_dir / "train_test_similarity_examples"
    panel_dir.mkdir(parents=True, exist_ok=True)
    for row in sorted(best, key=lambda r: float(r["cosine_similarity"]), reverse=True)[:10]:
        test_path = root / "data" / "frames" / "test" / "images" / row["test_image"]
        train_path = root / "data" / "frames" / "train" / "images" / row["train_image"]
        test_img = Image.open(test_path).convert("L").crop(crop)
        train_img = Image.open(train_path).convert("L").crop(crop)
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.4))
        axes[0].imshow(test_img, cmap="gray")
        axes[0].set_title(f"Test: {row['test_image']}")
        axes[1].imshow(train_img, cmap="gray")
        axes[1].set_title(f"Nearest train: {row['train_image']}")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"Cosine similarity = {float(row['cosine_similarity']):.3f}", fontsize=10)
        fig.tight_layout()
        fig.savefig(panel_dir / f"{Path(row['test_image']).stem}_nearest_train.png", dpi=250)
        plt.close(fig)


def estimate_model_macs(model) -> int:
    macs = 0
    for layer in model.layers:
        class_name = layer.__class__.__name__
        if class_name not in {"Conv2D", "Conv2DTranspose"}:
            continue
        try:
            _, h, w, cout = layer.output_shape
            kh, kw = layer.kernel_size
            cin = int(layer.input_shape[-1])
            macs += int(h) * int(w) * int(cout) * int(kh) * int(kw) * int(cin)
        except Exception:
            continue
    return int(macs)


def energy_analysis(
    root: Path,
    weights: Path,
    output_dir: Path,
    crop: tuple[int, int, int, int],
    size: tuple[int, int],
    benchmark_image: Path | None,
) -> None:
    import tensorflow as tf  # noqa: F401
    from src import Models as models

    model = models.unet((size[0], size[1], 1), len(CLASS_NAMES), filters=[32, 64, 128, 256, 512])
    model.load_weights(str(weights))
    params = int(model.count_params())
    macs = estimate_model_macs(model)
    gmacs = macs / 1e9

    if benchmark_image is None:
        benchmark_image = sorted((root / "data" / "frames" / "test" / "images").glob("*.png"))[0]
    x = preprocess_image(benchmark_image, crop, size)
    for _ in range(5):
        model.predict(x, verbose=0)
    n = 30
    start = time.perf_counter()
    for _ in range(n):
        model.predict(x, verbose=0)
    elapsed = time.perf_counter() - start
    seconds_per_frame = elapsed / n

    hardware = [
        {"platform": "Laptop/desktop CPU measured here", "power_w": 25.0, "seconds_per_frame": seconds_per_frame},
        {"platform": "NVIDIA Jetson Orin Nano 15 W, assuming measured latency", "power_w": 15.0, "seconds_per_frame": seconds_per_frame},
        {"platform": "Edge accelerator 5 W, assuming 10 FPS", "power_w": 5.0, "seconds_per_frame": 0.1},
        {"platform": "Portable CPU 3 W, assuming 1 FPS", "power_w": 3.0, "seconds_per_frame": 1.0},
    ]

    rows = []
    for h in hardware:
        energy_j = h["power_w"] * h["seconds_per_frame"]
        rows.append(
            {
                "platform": h["platform"],
                "assumed_power_w": h["power_w"],
                "seconds_per_frame": h["seconds_per_frame"],
                "fps": 1.0 / h["seconds_per_frame"],
                "energy_j_per_frame": energy_j,
                "energy_wh_per_hour_continuous": h["power_w"],
                "runtime_hours_on_50wh_battery": 50.0 / h["power_w"],
            }
        )
    write_rows(rows, output_dir / "portable_energy_estimates.csv")
    save_json(
        {
            "model_parameters": params,
            "estimated_macs_per_frame": macs,
            "estimated_gmacs_per_frame": gmacs,
            "measured_seconds_per_frame_local_tensorflow": seconds_per_frame,
            "measured_fps_local_tensorflow": 1.0 / seconds_per_frame,
            "benchmark_image": str(benchmark_image),
            "notes": [
                "MAC estimate covers Conv2D and Conv2DTranspose layers only.",
                "Portable-platform rows are engineering estimates; replace power/latency with hardware measurements if available.",
            ],
        },
        output_dir / "portable_energy_summary.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", type=Path)
    parser.add_argument("--tf-output-dir", default="outputs/tf_h5_eval_test", type=Path)
    parser.add_argument("--weights", default="model_lus.h5", type=Path)
    parser.add_argument("--output-dir", default="outputs/engineering_analysis", type=Path)
    parser.add_argument("--crop", default="100,50,850,460")
    parser.add_argument("--size", default="256,256")
    parser.add_argument("--failure-cases", default=10, type=int)
    parser.add_argument("--sensitivity-repeats", default=20, type=int)
    parser.add_argument("--conformal-alpha", default=0.1, type=float)
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--skip-agreement", action="store_true")
    parser.add_argument("--skip-failures", action="store_true")
    parser.add_argument("--skip-sensitivity", action="store_true")
    parser.add_argument("--skip-conformal", action="store_true")
    parser.add_argument("--skip-refined-conformal", action="store_true")
    parser.add_argument("--skip-conformal-case-panels", action="store_true")
    parser.add_argument("--skip-energy", action="store_true")
    parser.add_argument("--skip-similarity", action="store_true")
    args = parser.parse_args()

    crop = parse_tuple(args.crop)
    size = parse_tuple(args.size)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cases_path = args.output_dir / "blas_agreement_cases.csv"
    cases = [] if args.skip_agreement and not cases_path.exists() else None
    if not args.skip_agreement:
        cases = blas_agreement(args.tf_output_dir / "blas.csv", args.output_dir)
    elif cases_path.exists():
        cases = load_blas_rows(cases_path)
        for row in cases:
            row["blas_target"] = float(row["blas_target"])
            row["blas_pred"] = float(row["blas_pred"])
            row["blas_error"] = float(row["blas_error"])
            row["blas_abs_error"] = float(row["blas_abs_error"])
            row["category_error"] = str(row["category_error"]).lower() == "true"

    if not args.skip_failures:
        make_failure_panels(
            cases,  # type: ignore[arg-type]
            args.root,
            args.tf_output_dir / "pred_masks",
            args.output_dir,
            crop,  # type: ignore[arg-type]
            size,  # type: ignore[arg-type]
            args.failure_cases,
        )
    if not args.skip_sensitivity:
        sensitivity_analysis(args.root, args.output_dir, crop, size, args.sensitivity_repeats, args.seed)  # type: ignore[arg-type]
    if not args.skip_conformal:
        conformal_uncertainty(
            args.root,
            args.weights,
            cases,  # type: ignore[arg-type]
            args.output_dir,
            crop,  # type: ignore[arg-type]
            size,  # type: ignore[arg-type]
            args.conformal_alpha,
            args.seed,
        )
    if not args.skip_refined_conformal:
        conformal_blas_refinement(
            args.root,
            args.weights,
            cases,  # type: ignore[arg-type]
            args.output_dir,
            crop,  # type: ignore[arg-type]
            size,  # type: ignore[arg-type]
            args.conformal_alpha,
            args.seed,
        )
    if not args.skip_conformal_case_panels:
        conformal_set_case_panels(
            args.root,
            args.weights,
            cases,  # type: ignore[arg-type]
            args.output_dir,
            crop,  # type: ignore[arg-type]
            size,  # type: ignore[arg-type]
            args.conformal_alpha,
            args.seed,
            args.failure_cases,
        )
    if not args.skip_energy:
        energy_analysis(args.root, args.weights, args.output_dir, crop, size, None)  # type: ignore[arg-type]
    if not args.skip_similarity:
        train_test_similarity_analysis(args.root, args.output_dir, crop)  # type: ignore[arg-type]
    print(f"Wrote engineering analysis outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
