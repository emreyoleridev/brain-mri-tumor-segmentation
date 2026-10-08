"""Dataset indexing, cleaning, patient-level splitting and the PyTorch Dataset for the LGG brain MRI data.

Raw layout (Kaggle `mateuszbuda/lgg-mri-segmentation`, folder `kaggle_3m`):
    <root>/<PATIENT_ID>/<PATIENT_ID>_<slice>.tif        3-channel MRI slice (pre-contrast, FLAIR, post-contrast)
    <root>/<PATIENT_ID>/<PATIENT_ID>_<slice>_mask.tif   binary FLAIR-abnormality mask
"""
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .preprocess import normalize, to_tensor

SLICE_RE = re.compile(r"^(?P<patient>.+)_(?P<slice>\d+)\.tif$")


def find_root(data_dir: Path) -> Path:
    """Return the folder that directly contains the patient folders (the zip nests it at varying depth)."""
    for p in sorted(Path(data_dir).rglob("*_mask.tif")):
        return p.parent.parent
    raise FileNotFoundError(f"No *_mask.tif files under {data_dir}")


def build_index(root: Path) -> pd.DataFrame:
    """One row per slice with paths, mask statistics and basic quality checks."""
    rows = []
    for img_path in sorted(Path(root).glob("*/*.tif")):
        m = SLICE_RE.match(img_path.name)
        if m is None or img_path.name.endswith("_mask.tif"):
            continue
        mask_path = img_path.with_name(img_path.stem + "_mask.tif")
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
        row = {
            "patient": m["patient"],
            "slice": int(m["slice"]),
            "image_path": str(img_path),
            "mask_path": str(mask_path) if mask is not None else None,
            "readable": img is not None and mask is not None,
        }
        if row["readable"]:
            fg = mask > 127
            row.update(
                height=img.shape[0],
                width=img.shape[1],
                mask_unique=len(np.unique(mask)),
                tumor_pixels=int(fg.sum()),
                tumor_frac=float(fg.mean()),
                img_mean=float(img.mean()),
                img_std=float(img.std()),
                blank_image=bool(img.max() == 0),
                **{f"mean_{c}": float(img[..., i].mean()) for i, c in enumerate(("post", "flair", "pre"))},
            )
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["patient", "slice"]).reset_index(drop=True)
    df["has_tumor"] = df["tumor_pixels"].fillna(0) > 0
    df["n_slices"] = df.groupby("patient")["slice"].transform("max")
    df["slice_pos"] = df["slice"] / df["n_slices"]
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop unreadable, mask-less, wrong-size and fully black slices. Returns (clean_df, report)."""
    report = {"total_slices": len(df)}
    checks = {
        "unreadable_or_missing_mask": ~df["readable"],
        "wrong_size": df["readable"] & ((df["height"] != 256) | (df["width"] != 256)),
        "blank_image": df["readable"] & df["blank_image"].fillna(False).astype(bool),
    }
    drop = pd.Series(False, index=df.index)
    for name, m in checks.items():
        report[name] = int(m.sum())
        drop |= m
    out = df[~drop].reset_index(drop=True)
    report["kept_slices"] = len(out)
    report["patients"] = out["patient"].nunique()
    return out, report


def patient_split(df: pd.DataFrame, val=0.15, test=0.15, seed=42) -> pd.DataFrame:
    """Split by patient (no slice leakage), stratified on whether a patient has many tumor slices."""
    per_patient = df.groupby("patient")["has_tumor"].mean()
    rng = np.random.default_rng(seed)
    split = {}
    # stratify on tumor burden tercile so splits get a similar mix of small and large tumors
    for _, grp in per_patient.groupby(pd.qcut(per_patient.rank(method="first"), 3, labels=False)):
        ids = grp.index.to_numpy()
        rng.shuffle(ids)
        n_test, n_val = round(len(ids) * test), round(len(ids) * val)
        for i, pid in enumerate(ids):
            split[pid] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    df = df.copy()
    df["split"] = df["patient"].map(split)
    return df


def load_pair(image_path: str, mask_path: str, size: int) -> tuple[np.ndarray, np.ndarray]:
    img = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img.shape[0] != size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return img, (mask > 127).astype(np.float32)


def augment(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Geometric + intensity augmentation that keeps image and mask aligned."""
    if rng.random() < 0.5:
        img, mask = img[:, ::-1], mask[:, ::-1]
    if rng.random() < 0.7:
        h, w = mask.shape
        angle = rng.uniform(-15, 15)
        scale = rng.uniform(0.9, 1.1)
        tx, ty = rng.uniform(-0.06, 0.06, size=2) * (w, h)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        M[:, 2] += (tx, ty)
        img = cv2.warpAffine(np.ascontiguousarray(img), M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
        mask = cv2.warpAffine(np.ascontiguousarray(mask), M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    if rng.random() < 0.5:
        img = img.astype(np.float32) * rng.uniform(0.85, 1.15) + rng.uniform(-15, 15)
        img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        gamma = rng.uniform(0.8, 1.25)
        img = (255 * (img / 255.0) ** gamma).astype(np.uint8)
    return np.ascontiguousarray(img), np.ascontiguousarray(mask)


class MRISegDataset(Dataset):
    def __init__(self, df: pd.DataFrame, size: int = 256, train: bool = False, seed: int = 0):
        self.items = df[["image_path", "mask_path"]].to_numpy()
        self.size, self.train, self.seed = size, train, seed
        self.rng = None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img, mask = load_pair(*self.items[i], self.size)
        if self.train:
            if self.rng is None:  # one generator per DataLoader worker, otherwise workers repeat augmentations
                self.rng = np.random.default_rng(torch.initial_seed() + self.seed)
            img, mask = augment(img, mask, self.rng)
        return to_tensor(normalize(img)), torch.from_numpy(mask)[None]
