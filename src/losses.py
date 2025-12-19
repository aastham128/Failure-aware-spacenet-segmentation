import torch

def safe_dice_loss(pred, target, eps=1e-6):
    pred = pred.contiguous().view(pred.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)

    dice = (2 * intersection + eps) / (union + eps)

    empty_gt = target.sum(dim=1) == 0
    empty_pred = pred.sum(dim=1) == 0

    # Redefine correctness for empty GT
    dice[empty_gt & empty_pred] = 1.0   # correct absence
    dice[empty_gt & ~empty_pred] = 0.0  # hallucination

    return 1 - dice.mean()
