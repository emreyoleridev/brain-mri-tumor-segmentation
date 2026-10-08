"""Segmentation losses and metrics."""
import cv2
import numpy as np
import torch
import torch.nn.functional as F


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """BCE for per-pixel calibration + soft Dice to counter the heavy foreground/background imbalance."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits).flatten(1)
    t = target.flatten(1)
    dice = (2 * (p * t).sum(1) + smooth) / (p.sum(1) + t.sum(1) + smooth)
    return bce + (1 - dice.mean())


def remove_small(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Drop connected components smaller than `min_area` pixels (suppresses speckle false positives)."""
    if min_area <= 0 or not mask.any():
        return mask
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[lab]


def seg_scores(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Per-slice scores. Dice/IoU are defined as 1.0 when both masks are empty (correct 'no tumor')."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    tn = pred.size - tp - fp - fn
    both_empty = (tp + fp + fn) == 0
    return {
        "dice": 1.0 if both_empty else 2 * tp / (2 * tp + fp + fn),
        "iou": 1.0 if both_empty else tp / (tp + fp + fn),
        "precision": tp / (tp + fp) if tp + fp else float(fn == 0),
        "recall": tp / (tp + fn) if tp + fn else float(fp == 0),
        "pixel_acc": (tp + tn) / pred.size,
        "tp": int(tp), "fp": int(fp), "fn": int(fn),
        "pred_area": int(pred.sum()), "gt_area": int(gt.sum()),
    }
