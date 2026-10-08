"""Streamlit web app: brain MRI tumor segmentation with a U-Net.

Run:  streamlit run app.py
"""
import io
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from bmts import load_model, overlay, predict_proba, prepare_upload, remove_small, seg_scores

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models" / "unet_best.pt"
SAMPLES = ROOT / "samples"
RESULTS = ROOT / "results"

st.set_page_config(page_title="Brain MRI Tumor Segmentation", page_icon="🧠", layout="wide")
torch.set_num_threads(4)


@st.cache_resource
def get_model():
    return load_model(MODEL, torch.device("cpu"))


@st.cache_data
def sample_list():
    meta = SAMPLES / "samples.json"
    return json.loads(meta.read_text()) if meta.exists() else []


def read_image(data: bytes, gray=False) -> np.ndarray:
    img = np.array(Image.open(io.BytesIO(data)))
    if gray:
        return (img if img.ndim == 2 else img[..., 0]) > 127
    return img


def to_png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


model, meta = get_model()

# ---- sidebar ------------------------------------------------------------------------------------
st.sidebar.title("🧠 Brain MRI Tumor Segmentation")
st.sidebar.caption("U-Net trained on the LGG brain MRI dataset (Kaggle, TCGA/TCIA, 110 patients).")
source = st.sidebar.radio("Input", ["Test-set sample", "Upload image"], horizontal=True)
gt = None
if source == "Test-set sample":
    samples = sample_list()
    if not samples:
        st.error("No samples found. Run `python scripts/export_samples.py` first.")
        st.stop()
    labels = [f"{s['patient']} · slice {s['slice']} · {'tumor' if s['has_tumor'] else 'no tumor'}" for s in samples]
    idx = st.sidebar.selectbox("Sample (held-out test patients)", range(len(samples)), format_func=lambda i: labels[i])
    s = samples[idx]
    image = np.array(Image.open(SAMPLES / s["image"]).convert("RGB"))
    gt = np.array(Image.open(SAMPLES / s["mask"])) > 127
else:
    up = st.sidebar.file_uploader("MRI slice (tif / png / jpg)", type=["tif", "tiff", "png", "jpg", "jpeg"])
    up_mask = st.sidebar.file_uploader("Ground-truth mask (optional)", type=["tif", "tiff", "png", "jpg", "jpeg"])
    if up is None:
        st.title("Brain MRI Tumor Segmentation")
        st.info("Upload a brain MRI slice in the sidebar, or switch to **Test-set sample**. "
                "The model expects axial slices like the LGG dataset: 3 channels (pre-contrast, FLAIR, post-contrast). "
                "Grayscale images are accepted (copied to 3 channels) but accuracy is lower.")
        st.stop()
    image = prepare_upload(read_image(up.getvalue()))
    if up_mask is not None:
        gt = cv2.resize(read_image(up_mask.getvalue(), gray=True).astype(np.uint8), (256, 256),
                        interpolation=cv2.INTER_NEAREST).astype(bool)

st.sidebar.divider()
st.sidebar.subheader("Post-processing")
defaults = json.loads((RESULTS / "test_metrics.json").read_text()) if (RESULTS / "test_metrics.json").exists() else {}
thr = st.sidebar.slider("Probability threshold", 0.05, 0.95, float(defaults.get("threshold", 0.5)), 0.05)
min_area = st.sidebar.slider("Min. region area (px)", 0, 300, int(defaults.get("min_area", 0)), 5)
tta = st.sidebar.toggle("Flip test-time augmentation", value=True)
alpha = st.sidebar.slider("Overlay opacity", 0.1, 0.9, 0.45, 0.05)

# ---- inference ----------------------------------------------------------------------------------
prob = predict_proba(model, image, tta=tta)
pred = remove_small(prob > thr, min_area)
area = int(pred.sum())
brain_px = int((image.max(axis=2) > 5).sum())
n_regions = cv2.connectedComponents(pred.astype(np.uint8))[0] - 1

st.title("Brain MRI Tumor Segmentation")
c = st.columns(4)
c[0].metric("Prediction", "Tumor" if area else "No tumor")
c[1].metric("Tumor area", f"{area} px", f"{100 * area / max(brain_px, 1):.2f} % of brain", delta_color="off")
c[2].metric("Regions", n_regions)
c[3].metric("Max probability", f"{prob.max():.3f}")

heat = cv2.applyColorMap((prob * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1]
cols = st.columns(4 if gt is not None else 3)
cols[0].image(image, caption="Input (RGB = pre / FLAIR / post)", width="stretch")
cols[1].image(heat, caption="Tumor probability", width="stretch")
cols[2].image(overlay(image, pred, alpha=alpha), caption="Predicted mask", width="stretch")
if gt is not None:
    cols[3].image(overlay(image, gt, color=(40, 200, 40), alpha=alpha), caption="Ground truth", width="stretch")

with st.expander("MRI channels"):
    ch = st.columns(3)
    for col, i, name in zip(ch, range(3), ("pre-contrast", "FLAIR", "post-contrast")):
        col.image(image[..., i], caption=name, width="stretch", clamp=True)

if gt is not None:
    st.subheader("Comparison with ground truth")
    sc = seg_scores(pred, gt)
    m = st.columns(5)
    for col, k in zip(m, ("dice", "iou", "precision", "recall", "pixel_acc")):
        col.metric(k.replace("_", " ").title(), f"{sc[k]:.3f}")
    cmp = image.copy()
    cmp[gt & pred] = (60, 200, 60)
    cmp[gt & ~pred] = (240, 200, 0)
    cmp[~gt & pred] = (230, 40, 40)
    l, r = st.columns([1, 2])
    l.image(cmp, caption="green = TP · yellow = FN (missed) · red = FP", width="stretch")
    r.markdown("**Dice** = 2·|P∩G| / (|P|+|G|). If both masks are empty the slice counts as perfect (Dice 1). "
               "Missed pixels (yellow) usually sit on the faint tumor border, where FLAIR contrast fades.")

st.download_button("Download mask (PNG)", to_png(pred.astype(np.uint8) * 255), "tumor_mask.png", "image/png")

# ---- model card ---------------------------------------------------------------------------------
with st.expander("Model and test-set results"):
    st.markdown(f"U-Net, {meta.get('params', 0) / 1e6:.1f} M parameters, best validation Dice "
                f"{meta.get('val_dice', float('nan')):.3f} at epoch {meta.get('epoch', '?')}.")
    if (RESULTS / "test_summary.csv").exists():
        st.dataframe(pd.read_csv(RESULTS / "test_summary.csv", index_col=0), width="stretch")
    for f in ("training_curves.png", "test_predictions.png"):
        if (RESULTS / "figures" / f).exists():
            st.image(str(RESULTS / "figures" / f))

st.caption("Research / educational demo. Not a medical device and not for clinical decisions.")
