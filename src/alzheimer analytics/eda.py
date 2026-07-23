from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def create_figures(df: pd.DataFrame, output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    counts = df["Disease"].value_counts().sort_values(ascending=False)
    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")
    plt.title("Class Distribution")
    plt.xlabel("Disease Class")
    plt.ylabel("Images")
    plt.tight_layout()
    plt.savefig(out / "class_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    for label, group in df.groupby("Disease"):
        plt.hist(group["File Size (KB)"], bins=30, alpha=0.45, label=label)
    plt.title("File Size Distribution by Disease Class")
    plt.xlabel("File Size (KB)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "file_size_distribution.png", dpi=160)
    plt.close()
