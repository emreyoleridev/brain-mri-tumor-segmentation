"""Exploratory data analysis: Pandas statistics + Matplotlib/Seaborn figures in results/figures/eda_*.png."""
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bmts.preprocess import overlay  # noqa: E402

FIG = ROOT / "results" / "figures"
sns.set_theme(style="whitegrid", context="notebook")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "data" / "index.csv")
    tumor = df[df["has_tumor"]]

    # ---- tables -------------------------------------------------------------------------------
    per_patient = df.groupby("patient").agg(
        slices=("slice", "size"), tumor_slices=("has_tumor", "sum"), max_tumor_frac=("tumor_frac", "max")
    )
    stats = {
        "patients": df["patient"].nunique(),
        "slices": len(df),
        "tumor_slices": int(df["has_tumor"].sum()),
        "tumor_slice_ratio": round(df["has_tumor"].mean(), 4),
        "slices_per_patient_min/median/max": "/".join(map(str, per_patient["slices"].describe()[["min", "50%", "max"]].astype(int))),
        "tumor_area_px_median": int(tumor["tumor_pixels"].median()),
        "tumor_area_px_p10/p90": "/".join(map(str, tumor["tumor_pixels"].quantile([0.1, 0.9]).astype(int))),
        "tumor_area_frac_mean_(tumor_slices)": round(tumor["tumor_frac"].mean(), 4),
        "foreground_pixel_ratio_(all_slices)": round(df["tumor_frac"].mean(), 4),
        "mask_values_unique_max": int(df["mask_unique"].max()),
    }
    pd.Series(stats).to_csv(ROOT / "results" / "eda_stats.csv", header=["value"])
    per_patient.to_csv(ROOT / "results" / "eda_per_patient.csv")
    print(pd.Series(stats).to_string())

    # ---- 1. class balance + split ---------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.countplot(data=df, x="has_tumor", ax=ax[0], hue="has_tumor", palette=["#6c8ebf", "#d35f5f"], legend=False)
    ax[0].set(title="Slices with / without tumor", xlabel="has tumor", ylabel="slices")
    for c in ax[0].containers:
        ax[0].bar_label(c)
    sp = df.groupby(["split", "has_tumor"]).size().unstack().loc[["train", "val", "test"]]
    sp.plot.bar(stacked=True, ax=ax[1], color=["#6c8ebf", "#d35f5f"], rot=0)
    ax[1].set(title="Patient-level split (slices)", xlabel="", ylabel="slices")
    ax[1].legend(["no tumor", "tumor"])
    fig.tight_layout()
    fig.savefig(FIG / "eda_class_balance.png", dpi=130)
    plt.close(fig)

    # ---- 2. tumor size distribution ------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(tumor["tumor_pixels"], bins=50, log_scale=True, ax=ax[0], color="#d35f5f")
    ax[0].set(title="Tumor area per slice (tumor slices only)", xlabel="tumor pixels (log)", ylabel="slices")
    sns.histplot(per_patient["tumor_slices"] / per_patient["slices"], bins=20, ax=ax[1], color="#82b366")
    ax[1].set(title="Share of tumor slices per patient", xlabel="tumor slices / slices", ylabel="patients")
    fig.tight_layout()
    fig.savefig(FIG / "eda_tumor_size.png", dpi=130)
    plt.close(fig)

    # ---- 3. where along the volume are tumors ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 21)
    df["pos_bin"] = pd.cut(df["slice_pos"], bins, include_lowest=True)
    prob = df.groupby("pos_bin", observed=True)["has_tumor"].mean()
    ax.bar((bins[:-1] + bins[1:]) / 2, prob.values, width=0.045, color="#d35f5f")
    ax.set(title="P(tumor) by relative slice position", xlabel="slice index / slices in volume", ylabel="P(tumor)")
    fig.tight_layout()
    fig.savefig(FIG / "eda_slice_position.png", dpi=130)
    plt.close(fig)

    # ---- 4. intensity per MRI channel, tumor vs non-tumor pixels -------------------------------
    sample = tumor.sample(min(150, len(tumor)), random_state=0)
    rows = []
    for _, r in sample.iterrows():
        img = cv2.imread(r["image_path"])  # BGR = (post-contrast, FLAIR, pre-contrast)
        m = cv2.imread(r["mask_path"], cv2.IMREAD_GRAYSCALE) > 127
        brain = (img.max(2) > 5) & ~m
        for i, ch in enumerate(("post-contrast", "FLAIR", "pre-contrast")):
            for region, sel in (("tumor", m), ("healthy brain", brain)):
                v = img[..., i][sel]
                rows.append({"channel": ch, "region": region, "intensity": v[:: max(1, len(v) // 200)]})
    long = pd.DataFrame(rows).explode("intensity")
    long["intensity"] = long["intensity"].astype(float)
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.violinplot(data=long, x="channel", y="intensity", hue="region", split=True, inner="quart",
                   palette={"tumor": "#d35f5f", "healthy brain": "#6c8ebf"}, ax=ax, cut=0)
    ax.set(title="Pixel intensity: tumor vs. healthy brain (150 tumor slices)", xlabel="")
    fig.tight_layout()
    fig.savefig(FIG / "eda_intensity.png", dpi=130)
    plt.close(fig)

    # ---- 5. correlation of slice-level features ------------------------------------------------
    cols = ["tumor_frac", "img_mean", "img_std", "mean_post", "mean_flair", "mean_pre", "slice_pos"]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax, square=True)
    ax.set_title("Slice-level feature correlation")
    fig.tight_layout()
    fig.savefig(FIG / "eda_correlation.png", dpi=130)
    plt.close(fig)

    # ---- 6. sample grid: channels + mask overlay ------------------------------------------------
    picks = tumor.sort_values("tumor_pixels").iloc[np.linspace(0, len(tumor) - 1, 4).astype(int)]
    fig, ax = plt.subplots(4, 5, figsize=(13, 11))
    for row, (_, r) in zip(ax, picks.iterrows()):
        bgr = cv2.imread(r["image_path"])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        m = cv2.imread(r["mask_path"], cv2.IMREAD_GRAYSCALE) > 127
        panels = [(rgb, "RGB composite"), (bgr[..., 2], "pre-contrast"), (bgr[..., 1], "FLAIR"),
                  (bgr[..., 0], "post-contrast"), (overlay(rgb, m), f"mask ({int(r.tumor_pixels)} px)")]
        for a, (im, t) in zip(row, panels):
            a.imshow(im, cmap="gray")
            a.set_title(t, fontsize=9)
            a.axis("off")
        row[0].set_ylabel(r["patient"], fontsize=8)
    fig.suptitle("Samples ordered by tumor size", y=0.995)
    fig.tight_layout()
    fig.savefig(FIG / "eda_samples.png", dpi=110)
    plt.close(fig)
    print(f"Figures written to {FIG}")


if __name__ == "__main__":
    main()
