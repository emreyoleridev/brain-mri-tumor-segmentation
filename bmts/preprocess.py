"""Image preprocessing shared by training, evaluation and the web app."""
import cv2
import numpy as np
import torch


def normalize(img: np.ndarray) -> np.ndarray:
    """Per-image, per-channel z-score over brain (non-background) pixels.

    MRI intensities are not calibrated across scanners, so a fixed /255 scale leaves large patient-to-patient
    brightness shifts. Statistics are taken over non-zero pixels so the black background does not dominate.
    """
    img = img.astype(np.float32)
    out = np.zeros_like(img)
    brain = img.max(axis=2) > 5
    for c in range(img.shape[2]):
        vals = img[..., c][brain] if brain.sum() > 100 else img[..., c].ravel()
        out[..., c] = (img[..., c] - vals.mean()) / (vals.std() + 1e-6)
    out[~brain] = out[brain].min() if brain.any() else 0
    return out


def to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img.transpose(2, 0, 1).copy()).float()


def prepare_upload(img: np.ndarray, size: int = 256) -> np.ndarray:
    """Turn any user image (gray / RGB / RGBA, any size, any dtype) into a 3-channel uint8 RGB `size` x `size` slice."""
    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        img = (255 * (img - img.min()) / (np.ptp(img) + 1e-6)).astype(np.uint8)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    elif img.shape[2] == 4:
        img = img[..., :3]
    elif img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    if img.shape[:2] != (size, size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


def overlay(img: np.ndarray, mask: np.ndarray, color=(255, 40, 40), alpha=0.45, contour=True) -> np.ndarray:
    """Blend a binary mask on an RGB image, optionally drawing its outline."""
    out = img.copy()
    m = mask.astype(bool)
    out[m] = ((1 - alpha) * out[m] + alpha * np.array(color)).astype(np.uint8)
    if contour:
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cnts, -1, color, 1)
    return out
