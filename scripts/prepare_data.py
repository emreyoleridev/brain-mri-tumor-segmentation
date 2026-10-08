"""Download the LGG brain MRI dataset (Kaggle `mateuszbuda/lgg-mri-segmentation`), then index, clean and split it.

Uses the Kaggle API when credentials exist (~/.kaggle/kaggle.json), otherwise a public Hugging Face mirror of the
same archive. Writes data/index.csv (one row per slice, with split) and results/cleaning_report.json.
"""
import argparse
import json
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bmts.data import build_index, clean, find_root, patient_split  # noqa: E402

KAGGLE_SLUG = "mateuszbuda/lgg-mri-segmentation"
HF_URL = ("https://huggingface.co/datasets/gymprathap/Brain-MRI-LGG-Segmentation/resolve/main/"
          "Brain-MRI-LGG-Segmentation.zip")


def download(raw: Path) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    if any(raw.rglob("*_mask.tif")):
        return
    zips = list(raw.glob("*.zip"))
    if not zips:
        if (Path.home() / ".kaggle" / "kaggle.json").exists():
            subprocess.run(["kaggle", "datasets", "download", "-d", KAGGLE_SLUG, "-p", str(raw)], check=True)
        else:
            print("No Kaggle credentials, using Hugging Face mirror ...")
            urllib.request.urlretrieve(HF_URL, raw / "lgg.zip")
        zips = list(raw.glob("*.zip"))
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            zf.extractall(raw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw = args.data / "raw"
    download(raw)
    root = find_root(raw)
    print(f"Dataset root: {root}")

    df = build_index(root)
    # some archives ship the dataset twice (kaggle_3m and lgg-mri-segmentation/kaggle_3m)
    df = df.drop_duplicates(subset=["patient", "slice"]).reset_index(drop=True)
    df, report = clean(df)
    df = patient_split(df, seed=args.seed)
    report["split_patients"] = df.groupby("split")["patient"].nunique().to_dict()
    report["split_slices"] = df.groupby("split").size().to_dict()
    report["split_tumor_slices"] = df.groupby("split")["has_tumor"].sum().astype(int).to_dict()

    df.to_csv(args.data / "index.csv", index=False)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "cleaning_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
