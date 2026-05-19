from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def multiclass_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    include_background: bool = True,
    eps: float = 1e-7,
) -> torch.Tensor:
    probs = logits.softmax(dim=1)
    target_1h = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = (probs * target_1h).sum(dims)
    denominator = probs.sum(dims) + target_1h.sum(dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    if not include_background:
        dice = dice[1:]
    return 1.0 - dice.mean()


class CombinedCrossEntropyDiceLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 6,
        ce_weight: float = 1.0 / 3.0,
        include_background: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.include_background = include_background

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target)
        dice = multiclass_dice_loss(
            logits,
            target,
            num_classes=self.num_classes,
            include_background=self.include_background,
        )
        return self.ce_weight * ce + (1.0 - self.ce_weight) * dice

