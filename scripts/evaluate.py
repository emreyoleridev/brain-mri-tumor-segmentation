"""Evaluate the trained U-Net on the held-out test patients.

1. tune the probability threshold and minimum component area on the *validation* split,
2. apply them unchanged to the test split,
3. write per-slice / per-patient CSVs, a summary table and figures.
"""
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bmts import load_model, overlay, pick_device, predict_proba, remove_small, seg_scores  # noqa: E402
from bmts.data import load_pair  # noqa: E402

RES, FIG = ROOT / "results", ROOT / "results" / "figures"
sns.set_theme(style="whitegrid")


def predict_split(model, df):
    probs, gts, imgs = [], [], []
    for _, r in df.iterrows():
        img, gt = load_pair(r.image_path, r.mask_path, 256)
        probs.append(predict_proba(model, img).astype(np.float16))
        gts.append(gt.astype(bool))
        imgs.append(img)
    return np.stack(probs), np.stack(gts), imgs


def score(probs, gts, thr, min_area, df):
    rows = []
    for p, g, (_, r) in zip(probs, gts, df.iterrows()):
        pred = remove_small(p > thr, min_area)
        rows.append({"patient": r.patient, "slice": r.slice, "has_tumor": bool(g.any()),
                     "pred_tumor": bool(pred.any()), **seg_scores(pred, g)})
    return pd.DataFrame(rows)


def summarize(s: pd.DataFrame) -> dict:
    tp, fp, fn = s.tp.sum(), s.fp.sum(), s.fn.sum()
    t = s[s.has_tumor]
    cm = confusion_matrix(s.has_tumor, s.pred_tumor, labels=[False, True])
    (tn_, fp_), (fn_, tp_) = cm
    patient_dice = s.groupby("patient").apply(
        lambda g: 2 * g.tp.sum() / max(2 * g.tp.sum() + g.fp.sum() + g.fn.sum(), 1), include_groups=False)
    return {
        "global_dice": 2 * tp / (2 * tp + fp + fn),
        "global_iou": tp / (tp + fp + fn),
        "mean_dice_all_slices": s.dice.mean(),
        "mean_dice_tumor_slices": t.dice.mean(),
        "mean_iou_tumor_slices": t.iou.mean(),
        "mean_patient_dice": patient_dice.mean(),
        "pixel_precision": tp / (tp + fp),
        "pixel_recall": tp / (tp + fn),
        "pixel_accuracy": s.pixel_acc.mean(),
        "detect_accuracy": (tp_ + tn_) / cm.sum(),
        "detect_sensitivity": tp_ / (tp_ + fn_),
        "detect_specificity": tn_ / (tn_ + fp_),
        "detect_precision": tp_ / max(tp_ + fp_, 1),
        "detect_cm": cm.tolist(),
        "slices": len(s),
        "tumor_slices": int(s.has_tumor.sum()),
        "patients": s.patient.nunique(),
    }


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    model, meta = load_model(ROOT / "models" / "unet_best.pt", device)
    df = pd.read_csv(ROOT / "data" / "index.csv")
    val_df, test_df = df[df.split == "val"], df[df.split == "test"]

    # ---- post-processing tuned on validation only --------------------------------------------
    vp, vg, _ = predict_split(model, val_df)
    grid = []
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
        for min_area in (0, 10, 25, 50, 100):
            sm = summarize(score(vp, vg, thr, min_area, val_df))
            grid.append({"threshold": thr, "min_area": min_area, "global_dice": sm["global_dice"],
                         "mean_dice_all_slices": sm["mean_dice_all_slices"], "detect_accuracy": sm["detect_accuracy"]})
    grid = pd.DataFrame(grid)
    grid.to_csv(RES / "val_postprocess_grid.csv", index=False)
    # select on per-slice mean Dice: it rewards both good masks and correctly empty predictions
    best = grid.sort_values(["mean_dice_all_slices", "global_dice"], ascending=False).iloc[0]
    thr, min_area = float(best.threshold), int(best.min_area)
    print(f"Chosen on val: threshold={thr} min_area={min_area}")

    # ---- test ----------------------------------------------------------------------------------
    tp_, tg, timgs = predict_split(model, test_df)
    raw = summarize(score(tp_, tg, 0.5, 0, test_df))
    s = score(tp_, tg, thr, min_area, test_df)
    final = summarize(s)
    s.to_csv(RES / "test_per_slice.csv", index=False)

    table = pd.DataFrame({"U-Net (thr 0.5, no post-proc.)": raw, f"U-Net + post-proc. (thr {thr}, min area {min_area})": final})
    keys = ["global_dice", "global_iou", "mean_dice_tumor_slices", "mean_iou_tumor_slices", "mean_dice_all_slices",
            "mean_patient_dice", "pixel_precision", "pixel_recall", "detect_accuracy", "detect_sensitivity",
            "detect_specificity", "detect_precision"]
    table.loc[keys].astype(float).round(4).to_csv(RES / "test_summary.csv")
    (RES / "test_summary.md").write_text(table.loc[keys].astype(float).round(4).to_markdown())
    (RES / "test_metrics.json").write_text(json.dumps(
        {"threshold": thr, "min_area": min_area, "val_selected": best.to_dict(), "checkpoint": meta,
         "raw": raw, "final": final}, indent=2, default=float))
    print(table.loc[keys].astype(float).round(4).to_string())

    # per-patient
    pp = s.groupby("patient").apply(lambda g: pd.Series({
        "slices": len(g), "tumor_slices": int(g.has_tumor.sum()), "gt_volume_px": int(g.gt_area.sum()),
        "pred_volume_px": int(g.pred_area.sum()),
        "dice": 2 * g.tp.sum() / max(2 * g.tp.sum() + g.fp.sum() + g.fn.sum(), 1)}), include_groups=False)
    pp.to_csv(RES / "test_per_patient.csv")

    # ---- figures -------------------------------------------------------------------------------
    hist = pd.read_csv(RES / "training_history.csv")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist.epoch, hist.train_loss, label="train")
    ax[0].plot(hist.epoch, hist.val_loss, label="val")
    ax[0].set(title="BCE + Dice loss", xlabel="epoch")
    ax[0].legend()
    ax[1].plot(hist.epoch, hist.val_dice, label="val Dice", color="#d35f5f")
    ax[1].plot(hist.epoch, hist.val_iou, label="val IoU", color="#82b366")
    b = hist.val_dice.idxmax()
    ax[1].scatter(hist.epoch[b], hist.val_dice[b], color="k", zorder=3, label=f"best {hist.val_dice[b]:.3f} @ {hist.epoch[b]}")
    ax[1].set(title="Validation overlap", xlabel="epoch", ylim=(0, 1))
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(FIG / "training_curves.png", dpi=130)
    plt.close(fig)

    t = s[s.has_tumor]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    sns.histplot(t.dice, bins=25, ax=ax[0], color="#d35f5f")
    ax[0].set(title="Dice per tumor slice (test)", xlabel="Dice")
    ax[1].scatter(t.gt_area, t.dice, s=10, alpha=0.6, color="#6c8ebf")
    ax[1].set(xscale="log", title="Dice vs. tumor size", xlabel="ground-truth tumor pixels (log)", ylabel="Dice")
    ax[2].scatter(pp.gt_volume_px, pp.pred_volume_px, color="#82b366")
    lim = max(pp.gt_volume_px.max(), pp.pred_volume_px.max()) * 1.05
    ax[2].plot([0, lim], [0, lim], "k--", lw=1)
    ax[2].set(title="Tumor volume per patient", xlabel="ground truth (px)", ylabel="predicted (px)")
    fig.tight_layout()
    fig.savefig(FIG / "test_dice_analysis.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(np.array(final["detect_cm"]), annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["no tumor", "tumor"], yticklabels=["no tumor", "tumor"])
    ax.set(title="Slice-level tumor detection (test)", xlabel="predicted", ylabel="ground truth")
    fig.tight_layout()
    fig.savefig(FIG / "test_confusion_matrix.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    piv = grid.pivot(index="min_area", columns="threshold", values="mean_dice_all_slices")
    sns.heatmap(piv, annot=True, fmt=".3f", cmap="viridis", ax=ax)
    ax.set_title("Val mean slice Dice by post-processing")
    fig.tight_layout()
    fig.savefig(FIG / "val_postprocess_grid.png", dpi=130)
    plt.close(fig)

    # qualitative: best, median, worst tumor slices + a false positive if any
    t_sorted = t.sort_values("dice")
    picks = [t_sorted.index[-1], t_sorted.index[-len(t) // 4], t_sorted.index[len(t) // 2],
             t_sorted.index[len(t) // 4], t_sorted.index[0]]
    fps = s[~s.has_tumor & s.pred_tumor]
    if len(fps):
        picks.append(fps.sort_values("pred_area").index[-1])
    fig, ax = plt.subplots(len(picks), 4, figsize=(12, 3 * len(picks)))
    for row, i in zip(ax, picks):
        img, g, p = timgs[i], tg[i], tp_[i].astype(np.float32)
        pred = remove_small(p > thr, min_area)
        cmp = img.copy()
        cmp[g & pred] = (60, 200, 60)
        cmp[g & ~pred] = (240, 200, 0)
        cmp[~g & pred] = (230, 40, 40)
        for a, (im, title) in zip(row, [(img, f"{s.patient[i]} #{s.slice[i]}"), (overlay(img, g, (40, 200, 40)), "ground truth"),
                                        (p, "probability"), (cmp, f"TP green / FN yellow / FP red\nDice {s.dice[i]:.3f}")]):
            a.imshow(im, cmap="magma" if im.ndim == 2 else None, vmin=0, vmax=1 if im.ndim == 2 else None)
            a.set_title(title, fontsize=9)
            a.axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "test_predictions.png", dpi=100)
    plt.close(fig)
    print(f"Results in {RES}")


if __name__ == "__main__":
    main()
