from __future__ import annotations

import numpy as np
import torch


CLASS_NAMES = {
    0: "Background",
    1: "Ribs",
    2: "Pleural line",
    3: "A-line",
    4: "B-line",
    5: "B-line confluence",
}


def confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    pred = pred.view(-1).to(torch.int64)
    target = target.view(-1).to(torch.int64)
    valid = (target >= 0) & (target < num_classes)
    idx = target[valid] * num_classes + pred[valid]
    return torch.bincount(idx, minlength=num_classes**2).reshape(num_classes, num_classes)


def metrics_from_confusion(cm: torch.Tensor, class_names: dict[int, str] | None = None) -> dict:
    class_names = class_names or CLASS_NAMES
    cm = cm.double()
    tp = torch.diag(cm)
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp
    support = cm.sum(dim=1)

    dice = (2 * tp) / (2 * tp + fp + fn).clamp_min(1.0)
    iou = tp / (tp + fp + fn).clamp_min(1.0)
    accuracy = tp.sum() / cm.sum().clamp_min(1.0)

    out = {"pixel_accuracy": float(accuracy.item()), "classes": {}}
    for idx in range(len(dice)):
        out["classes"][str(idx)] = {
            "name": class_names.get(idx, str(idx)),
            "dice": float(dice[idx].item()),
            "iou": float(iou[idx].item()),
            "support_pixels": int(support[idx].item()),
        }

    foreground = dice[1:] if len(dice) > 1 else dice
    out["mean_dice_excluding_background"] = float(foreground.mean().item())
    return out


def summarize_metric_values(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if arr.size else 0.0,
        "std": float(arr.std(ddof=0)) if arr.size else 0.0,
    }

