import torch

def dice_score(preds, targets, smooth=1e-6):
    preds = preds > 0.5
    targets = targets > 0.5
    intersection = (preds & targets).float().sum((1, 2, 3))
    union = preds.float().sum((1, 2, 3)) + targets.float().sum((1, 2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean().item()

def pixel_accuracy(preds, targets):
    preds = preds > 0.5
    targets = targets > 0.5
    correct = (preds == targets).float().sum()
    total = torch.numel(preds)
    return (correct / total).item()

