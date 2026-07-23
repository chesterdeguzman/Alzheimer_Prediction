# Model Card

## Purpose
Educational predictive analytics using image-file metadata to predict one of three dataset labels: `Mild_Demented`, `Very_Mild_Demented`, or `Non_Demented`.

## Important limitation
This repository does **not** analyze MRI pixels. The supplied CSV contains metadata only. Therefore, this model is a weak technical baseline and must not be used for medical diagnosis, screening, treatment, or clinical decision-making.

## Features
The model intentionally uses only `File Size (KB)`.

- `Filename` and `Full Path` are excluded because they directly contain class names and would cause target leakage.
- Width, height, channels, color mode, and extension are constant across all rows and provide no predictive variation.

## Evaluation
The original train/validation/test split is respected. Metrics include accuracy, balanced accuracy, macro F1, per-class precision/recall/F1, and confusion matrix.

## Ethical considerations
Predicted labels can be misleading because metadata does not represent clinical evidence. Use the project only for learning data validation, leakage prevention, reproducible modeling, and model evaluation.
