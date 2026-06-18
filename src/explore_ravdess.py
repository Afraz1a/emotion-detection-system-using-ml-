"""Stage 1b: Explore RAVDESS dataset."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display

from utils.labels import EMOTIONS, ID_TO_LABEL, parse_ravdess_filename
from utils.helpers import set_seed, DATA_DIR


def main():
    set_seed(42)
    ravdess_dir = DATA_DIR / "ravdess"
    assert ravdess_dir.exists(), f"Missing {ravdess_dir}. Unzip RAVDESS Audio_Speech here."

    wav_files = sorted(ravdess_dir.rglob("*.wav"))
    print(f"Found {len(wav_files)} .wav files")
    assert len(wav_files) > 0, "No WAV files found."

    # Build index
    rows = []
    for p in wav_files:
        parts = p.stem.split("-")
        if len(parts) != 7:
            continue
        eid = parse_ravdess_filename(p.name)
        if eid is None:
            continue
        rows.append({
            "path": str(p),
            "actor": int(parts[6]),
            "gender": "male" if int(parts[6]) % 2 == 1 else "female",
            "emotion_id": eid,
            "emotion_name": ID_TO_LABEL[eid],
        })
    idx = pd.DataFrame(rows)
    print(f"Indexed {len(idx)} files")

    counts = idx["emotion_name"].value_counts().reindex(EMOTIONS, fill_value=0)
    print(f"\nPer-class counts:\n{counts}")
    print(f"\nGender:\n{idx['gender'].value_counts()}")

    # Class balance plot
    fig, ax = plt.subplots(figsize=(8, 4))
    counts.plot(kind="bar", ax=ax, color="darkorange", edgecolor="black")
    ax.set_title("RAVDESS - clips per emotion (calm merged into neutral)")
    ax.set_ylabel("Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 5, str(int(v)), ha="center", fontsize=9)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(DATA_DIR.parent / "ravdess_class_balance.png", dpi=120)
    print("Saved ravdess_class_balance.png")

    # Duration stats on a sample
    sample = idx.sample(min(200, len(idx)), random_state=42)
    durations = []
    sr_values = set()
    for _, row in sample.iterrows():
        y, sr = librosa.load(row["path"], sr=None)
        durations.append(len(y) / sr)
        sr_values.add(sr)
    durations = np.array(durations)
    print(f"\nSample rates: {sr_values}")
    print(f"Duration: mean={durations.mean():.2f}s std={durations.std():.2f}s "
          f"min={durations.min():.2f}s max={durations.max():.2f}s")

    # MFCC example per emotion
    SR = 22050
    fig, axes = plt.subplots(7, 2, figsize=(11, 14))
    fig.suptitle("RAVDESS - example per emotion (waveform + MFCC)", y=1.0)
    for row, emotion in enumerate(EMOTIONS):
        sub = idx[idx["emotion_name"] == emotion]
        if len(sub) == 0:
            axes[row, 0].set_axis_off()
            axes[row, 1].set_axis_off()
            continue
        path = sub.sample(1, random_state=42).iloc[0]["path"]
        y, _ = librosa.load(path, sr=SR)
        t = np.linspace(0, len(y) / SR, len(y))
        axes[row, 0].plot(t, y, color="steelblue", linewidth=0.6)
        axes[row, 0].set_title(f"{emotion} - waveform")
        mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=40, n_fft=2048, hop_length=512)
        img = librosa.display.specshow(mfcc, sr=SR, hop_length=512, x_axis="time", ax=axes[row, 1])
        axes[row, 1].set_title(f"{emotion} - MFCC")
        fig.colorbar(img, ax=axes[row, 1], format="%+0.1f")
    plt.tight_layout()
    plt.savefig(DATA_DIR.parent / "ravdess_examples.png", dpi=110, bbox_inches="tight")
    print("Saved ravdess_examples.png")


if __name__ == "__main__":
    main()
