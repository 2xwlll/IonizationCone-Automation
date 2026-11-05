"""Explicit binary loss for the F187N pilot, including empty targets."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class F187NLoss(nn.Module):
    """Mean BCE on every pixel plus mean soft Dice on positive images only.

    BCE teaches empty masks and remains informative when predictions are wrong.
    Empty images do not add a nearly constant, gradient-starved Dice penalty.
    No sigmoid belongs in the model; BCE consumes logits directly.
    """
    def __init__(self, bce_weight=1., dice_weight=1., pos_weight=1.):
        super().__init__()
        if not all(math.isfinite(v) for v in (bce_weight, dice_weight, pos_weight)):
            raise ValueError('Loss weights must be finite')
        if bce_weight <= 0 or dice_weight < 0 or pos_weight <= 0:
            raise ValueError('Require BCE > 0, Dice >= 0, and positive class weight > 0')
        self.bce_weight, self.dice_weight = bce_weight, dice_weight
        self.register_buffer('pos_weight', torch.tensor(float(pos_weight)))

    def components(self, logits, target):
        if logits.shape != target.shape or logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError('Expected matching N,1,H,W logits and targets')
        logits, target = logits.float(), target.float()
        bce = F.binary_cross_entropy_with_logits(logits, target,
                                                pos_weight=self.pos_weight)
        axes = (1, 2, 3)
        positive = target.sum(axes) > 0
        probability = logits.sigmoid()
        dice = 1 - (2*(probability*target).sum(axes)+1e-6) / (
            probability.sum(axes)+target.sum(axes)+1e-6)
        # Differentiable zero also handles batches containing only negatives.
        positive_dice = dice[positive].mean() if positive.any() else logits.sum()*0
        return bce, positive_dice

    def forward(self, logits, target):
        bce, dice = self.components(logits, target)
        return self.bce_weight*bce + self.dice_weight*dice
