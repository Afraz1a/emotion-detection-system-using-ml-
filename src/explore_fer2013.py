"""Stage 1a: Explore FER-2013 dataset."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.labels import EMOTIONS, ID_TO_LABEL
from utils.helpers import set_seed, DATA_DIR


def decode_pixels(s):
    return np.array(s.split(), dtype=np.uint8).reshape(48, 48)


def main():
    set_seed(42)
    csv_path = DATA_DIR / "fer2013.csv"
    assert csv_path.exists(), f"Missing {csv_path}. Download fer2013.csv from Kaggle."

    df = pd.read_csv(csv_path)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nUsage split:\n{df['Usage'].value_counts()}")

    df["emotion_name"] = df["emotion"].map(ID_TO_LABEL)
    counts = df["emotion_name"].value_counts().reindex(EMOTIONS)
    print(f"\nPer-class counts:\n{counts}")
    print(f"\nImbalance ratio (max/min): {counts.max() / counts.min():.1f}x")

    # Class balance plot
    fig, ax = plt.subplots(figsize=(8, 4))
    counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
    ax.set_title("FER-2013 - samples per emotion")
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 200, str(int(v)), ha="center", fontsize=9)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(DATA_DIR.parent / "fer2013_class_balance.png", dpi=120)
    print("Saved fer2013_class_balance.png")

    # Sample grid: one row per emotion
    n_per_class = 8
    fig, axes = plt.subplots(7, n_per_class, figsize=(n_per_class * 1.4, 7 * 1.4))
    for row, emotion in enumerate(EMOTIONS):
        subset = df[df["emotion_name"] == emotion].sample(
            min(n_per_class, (df["emotion_name"] == emotion).sum()),
            random_state=42,
        )
        for col, (_, sample) in enumerate(subset.iterrows()):
            ax = axes[row, col]
            ax.imshow(decode_pixels(sample["pixels"]), cmap="gray")
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(emotion, rotation=0, ha="right", va="center", fontsize=11)
    plt.suptitle("FER-2013 - random samples per emotion", y=1.01)
    plt.tight_layout()
    plt.savefig(DATA_DIR.parent / "fer2013_samples.png", dpi=120, bbox_inches="tight")
    print("Saved fer2013_samples.png")

    # Sanity checks
    bad = df["pixels"].apply(lambda s: len(s.split()) != 2304).sum()
    print(f"\nRows with wrong pixel count: {bad}")
    print(f"Rows with invalid label: {(~df['emotion'].isin(range(7))).sum()}")
    print(f"Missing values: {df.isna().sum().sum()}")


if __name__ == "__main__":
    main()
