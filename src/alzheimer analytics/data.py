from __future__ import annotations

from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = {
    "Split", "Disease", "Filename", "Extension", "Width", "Height",
    "Channels", "Color Mode", "File Size (KB)", "File Size (MB)", "Full Path",
}

SAFE_FEATURES = ["File Size (KB)"]
TARGET = "Disease"


def load_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Dataset is empty.")
    return df


def validate_dataset(df: pd.DataFrame) -> None:
    allowed_splits = {"train", "val", "test"}
    actual = set(df["Split"].dropna().astype(str).str.lower())
    if not actual.issubset(allowed_splits):
        raise ValueError(f"Unexpected split values: {sorted(actual - allowed_splits)}")
    if df[TARGET].isna().any():
        raise ValueError("Target column contains missing values.")
    if df[SAFE_FEATURES].isna().any().any():
        raise ValueError("Safe model features contain missing values.")


def split_dataset(df: pd.DataFrame):
    normalized = df.assign(Split=df["Split"].astype(str).str.lower())
    return tuple(normalized[normalized["Split"] == name].copy() for name in ("train", "val", "test"))
