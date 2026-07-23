# Alzheimer Dataset Predictive Analytics

A GitHub-ready, reproducible predictive analytics project built from `Alzheimer_Dataset_Details.csv`.

## Project goal

Train and evaluate a multiclass model that predicts the dataset's `Disease` label using the metadata available in the CSV.

> **Medical disclaimer:** This is an educational metadata-only baseline. It does not inspect MRI image pixels and is not appropriate for diagnosis or any clinical use.

## Dataset summary

- 6,336 records
- 3 target classes
- Existing train, validation, and test splits
- No missing values
- All images are 128×128 grayscale JPG files with one channel

Class totals:

| Class | Rows |
|---|---:|
| Non_Demented | 3,200 |
| Very_Mild_Demented | 2,240 |
| Mild_Demented | 896 |

## Leakage-safe feature policy

The filenames and paths contain disease-related folder or naming information. Using them would let the model infer the answer from the label text rather than learn a meaningful pattern. They are intentionally excluded.

Because width, height, channels, color mode, and extension are constant, the baseline uses only:

```text
File Size (KB)
```

This limitation is documented rather than hidden. A meaningful medical-imaging model would require the actual image files and a properly validated computer-vision workflow.

## Baseline results

Random Forest on the provided test split:

| Metric | Score |
|---|---:|
| Accuracy | 0.419 |
| Balanced accuracy | 0.369 |
| Macro F1 | 0.340 |

The most-frequent-class baseline reaches about 0.505 accuracy but only 0.333 balanced accuracy and 0.224 macro F1. This illustrates why accuracy alone is misleading for an imbalanced dataset.

## Repository structure

```text
.
├── data/
│   ├── raw/Alzheimer_Dataset_Details.csv
│   └── processed/
├── models/
├── notebooks/01_exploratory_analysis.ipynb
├── reports/
│   ├── figures/
│   └── metrics.json
├── src/alzheimer_analytics/
│   ├── data.py
│   ├── eda.py
│   └── modeling.py
├── tests/test_pipeline.py
├── train.py
├── make_report.py
├── MODEL_CARD.md
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Run the project

```bash
PYTHONPATH=src python make_report.py
PYTHONPATH=src python train.py
PYTHONPATH=src pytest -q
```

Outputs:

- `models/random_forest.joblib`
- `reports/metrics.json`
- `reports/figures/*.png`
- `data/processed/test_predictions.csv`

## Recommended next step

For stronger predictive analytics, add the underlying MRI image files and build an image-classification pipeline with patient-level splitting, augmentation restricted to training data, external validation, calibration, and explainability checks.
