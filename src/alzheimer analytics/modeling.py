from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report, confusion_matrix

from .data import SAFE_FEATURES, TARGET, load_dataset, validate_dataset, split_dataset

@dataclass
class TrainingResult:
    model: object
    metrics: dict
    predictions: pd.DataFrame


def build_model(random_state: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )


def evaluate(model, frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    y_true = frame[TARGET]
    y_pred = model.predict(frame[SAFE_FEATURES])
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        "labels": list(model.classes_),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=model.classes_).tolist(),
    }
    out = frame[["Split", "Filename", TARGET, *SAFE_FEATURES]].copy()
    out["Predicted Disease"] = y_pred
    probabilities = model.predict_proba(frame[SAFE_FEATURES])
    for idx, label in enumerate(model.classes_):
        out[f"Probability - {label}"] = probabilities[:, idx]
    return metrics, out


def train(input_path: str | Path, model_path: str | Path, metrics_path: str | Path, predictions_path: str | Path) -> TrainingResult:
    df = load_dataset(input_path)
    validate_dataset(df)
    train_df, val_df, test_df = split_dataset(df)

    model = build_model()
    model.fit(train_df[SAFE_FEATURES], train_df[TARGET])

    val_metrics, _ = evaluate(model, val_df)
    test_metrics, predictions = evaluate(model, test_df)

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(train_df[SAFE_FEATURES], train_df[TARGET])
    baseline_metrics, _ = evaluate(baseline, test_df)

    metrics = {
        "feature_policy": {
            "used": SAFE_FEATURES,
            "excluded_for_leakage": ["Filename", "Full Path"],
            "excluded_as_constant": ["Extension", "Width", "Height", "Channels", "Color Mode"],
        },
        "validation": val_metrics,
        "test": test_metrics,
        "baseline_test": baseline_metrics,
        "warning": "Metadata-only model. Not suitable for diagnosis or clinical decision-making.",
    }

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    Path(predictions_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    Path(metrics_path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    predictions.to_csv(predictions_path, index=False)
    return TrainingResult(model=model, metrics=metrics, predictions=predictions)
