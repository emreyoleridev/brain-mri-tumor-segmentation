"""Copy a small set of held-out test slices into samples/ as PNG so the web app works without the full dataset."""
import json
import shutil
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main(n_tumor: int = 12, n_empty: int = 4) -> None:
    df = pd.read_csv(ROOT / "data" / "index.csv")
    test = df[df.split == "test"]
    tumor = test[test.has_tumor].sort_values("tumor_pixels")
    # spread across tumor sizes, plus some empty slices
    picks = pd.concat([tumor.iloc[[round(i * (len(tumor) - 1) / (n_tumor - 1)) for i in range(n_tumor)]],
                       test[~test.has_tumor & (test.slice_pos.between(0.35, 0.65))].sample(n_empty, random_state=0)])
    out = ROOT / "samples"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()
    meta = []
    for _, r in picks.sort_values(["patient", "slice"]).iterrows():
        stem = f"{r.patient}_{r.slice}"
        cv2.imwrite(str(out / f"{stem}.png"), cv2.imread(r.image_path))
        cv2.imwrite(str(out / f"{stem}_mask.png"), cv2.imread(r.mask_path, cv2.IMREAD_GRAYSCALE))
        meta.append({"patient": r.patient, "slice": int(r.slice), "has_tumor": bool(r.has_tumor),
                     "image": f"{stem}.png", "mask": f"{stem}_mask.png"})
    (out / "samples.json").write_text(json.dumps(meta, indent=2))
    print(f"Exported {len(meta)} samples to {out}")


if __name__ == "__main__":
    main()
