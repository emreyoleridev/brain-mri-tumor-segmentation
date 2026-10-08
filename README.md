# Brain MRI Tumor Segmentation

A PyTorch U-Net that segments lower-grade glioma (FLAIR abnormality) in brain MRI slices, trained on the Kaggle
**LGG MRI Segmentation** dataset (110 patients, 3,929 slices). It includes Pandas/Seaborn EDA, a patient-level
evaluation and a Streamlit web app.

**Live demo:** _STREAMLIT_URL_

Full analysis: **[REPORT.md](REPORT.md)** · test results: [results/test_summary.md](results/test_summary.md)

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/prepare_data.py     # download (Kaggle API or HF mirror), clean, patient-level split -> data/index.csv
python scripts/eda.py              # statistics + figures in results/
python scripts/train.py            # U-Net training, best checkpoint -> models/unet_best.pt
python scripts/evaluate.py         # val-tuned post-processing, test metrics + figures
python scripts/export_samples.py   # 16 test slices for the app
streamlit run app.py               # web app at http://localhost:8501
```

The trained model (`models/unet_best.pt`) and samples (`samples/`) are included, so the app runs without the dataset:
`pip install -r requirements.txt && streamlit run app.py`. That is also all Streamlit Community Cloud needs.

## Web app

* pick a held-out test slice or upload your own (tif / png / jpg; gray images are expanded to 3 channels)
* probability heatmap, predicted mask overlay, tumor area and region count
* threshold, minimum region area and flip-TTA controls
* with a ground-truth mask: Dice / IoU / precision / recall and a TP / FN / FP error map
* download the predicted mask as PNG

## Project layout

```
bmts/data.py            indexing, cleaning, patient-level split, augmentation, Dataset
bmts/preprocess.py      brain-masked z-score normalization, upload handling, overlays
bmts/model.py           U-Net, checkpoint loading, inference with flip TTA
bmts/metrics.py         BCE + Dice loss, Dice / IoU / precision / recall, small-component removal
scripts/                prepare_data, eda, train, evaluate, export_samples
app.py                  Streamlit app
results/                CSV / JSON metrics, figures/
REPORT.md               project report
```

Research / educational project. Not a medical device.
