"""Comprehensive class imbalance and dataset statistics analysis."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import entropy

from utils.labels import EMOTIONS, ID_TO_LABEL, parse_ravdess_filename
from utils.helpers import set_seed, DATA_DIR

FERPLUS_TO_UNIFIED = {
    "anger": 0, "disgust": 1, "fear": 2, "happiness": 3,
    "sadness": 4, "surprise": 5, "neutral": 6,
}
FERPLUS_EMOTION_COLS = list(FERPLUS_TO_UNIFIED.keys())


# ── helpers ──────────────────────────────────────────────────────────────────

def imbalance_stats(counts: pd.Series, name: str) -> dict:
    total = counts.sum()
    uniform = np.ones(len(counts)) / len(counts)
    actual = counts.values / total
    kl = float(entropy(actual, uniform))
    return {
        "dataset": name,
        "total": int(total),
        "max_class": counts.idxmax(),
        "max_count": int(counts.max()),
        "min_class": counts.idxmin(),
        "min_count": int(counts.min()),
        "ratio_max_min": round(counts.max() / counts.min(), 2),
        "std_dev": round(counts.std(), 1),
        "cv_%": round(counts.std() / counts.mean() * 100, 1),
        "kl_div_from_uniform": round(kl, 4),
    }


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ── FER-2013 ──────────────────────────────────────────────────────────────────

def analyze_fer2013():
    print_section("FER-2013")
    csv = DATA_DIR / "fer2013.csv"
    df = pd.read_csv(csv)
    df["emotion_name"] = df["emotion"].map(ID_TO_LABEL)

    print(f"Total rows: {len(df):,}")
    print(f"\nSplit breakdown:")
    print(df["Usage"].value_counts().to_string())

    overall_counts = df["emotion_name"].value_counts().reindex(EMOTIONS)
    print(f"\nOverall per-class counts:")
    pct = overall_counts / overall_counts.sum() * 100
    for em in EMOTIONS:
        print(f"  {em:<10} {int(overall_counts[em]):>6,}  ({pct[em]:5.1f}%)")
    stats = imbalance_stats(overall_counts, "FER-2013 (all)")
    _print_stats(stats)

    print(f"\nPer-split breakdown:")
    for split in ["Training", "PublicTest", "PrivateTest"]:
        sub = df[df["Usage"] == split]
        counts = sub["emotion_name"].value_counts().reindex(EMOTIONS, fill_value=0)
        s = imbalance_stats(counts, split)
        print(f"\n  [{split}]  n={s['total']:,}  ratio={s['ratio_max_min']}x  "
              f"CV={s['cv_%']}%  KL={s['kl_div_from_uniform']}")
        for em in EMOTIONS:
            print(f"    {em:<10} {int(counts[em]):>5,}  ({counts[em]/s['total']*100:5.1f}%)")

    return df, overall_counts


# ── FER+ ─────────────────────────────────────────────────────────────────────

def analyze_ferplus(fer_df):
    print_section("FER+ (improved labels)")
    csv = DATA_DIR / "fer2013new.csv"
    plus_df = pd.read_csv(csv)

    emotion_cols = FERPLUS_EMOTION_COLS + ["contempt", "unknown", "NF"]
    # FER+ has a 'Usage' column matching FER-2013
    plus_df.columns = plus_df.columns.str.strip()

    # Hard label = column with highest vote among our 7 classes
    vote_cols = [c for c in FERPLUS_EMOTION_COLS if c in plus_df.columns]
    all_cols  = vote_cols + [c for c in ["contempt", "unknown", "NF"] if c in plus_df.columns]

    # Drop junk rows
    mask_nf      = plus_df.get("NF", pd.Series(0, index=plus_df.index)) == plus_df[all_cols].max(axis=1)
    mask_unknown = plus_df.get("unknown", pd.Series(0, index=plus_df.index)) == plus_df[all_cols].max(axis=1)
    mask_contempt= plus_df.get("contempt", pd.Series(0, index=plus_df.index)) == plus_df[all_cols].max(axis=1)
    keep = ~(mask_nf | mask_unknown | mask_contempt)
    clean = plus_df[keep].copy()

    # Assign hard label
    clean["emotion_name"] = clean[vote_cols].idxmax(axis=1).map(FERPLUS_TO_UNIFIED).map(ID_TO_LABEL)
    clean = clean.dropna(subset=["emotion_name"])

    print(f"Total rows (raw): {len(plus_df):,}")
    print(f"After dropping NF/unknown/contempt majority: {len(clean):,}  "
          f"(dropped {len(plus_df)-len(clean):,}  = {(len(plus_df)-len(clean))/len(plus_df)*100:.1f}%)")

    counts = clean["emotion_name"].value_counts().reindex(EMOTIONS, fill_value=0)
    print(f"\nFER+ per-class counts (hard labels):")
    pct = counts / counts.sum() * 100
    for em in EMOTIONS:
        print(f"  {em:<10} {int(counts[em]):>6,}  ({pct[em]:5.1f}%)")
    stats = imbalance_stats(counts, "FER+")
    _print_stats(stats)

    # Compare with FER-2013
    fer_counts = fer_df["emotion_name"].value_counts().reindex(EMOTIONS, fill_value=0)
    diff = counts - fer_counts
    print(f"\nLabel shift FER+ vs FER-2013 (positive = more in FER+):")
    for em in EMOTIONS:
        print(f"  {em:<10} {int(diff[em]):>+6,}")

    return counts


# ── RAVDESS ───────────────────────────────────────────────────────────────────

def analyze_ravdess():
    print_section("RAVDESS")
    ravdess_dir = DATA_DIR / "ravdess"
    wav_files = sorted(ravdess_dir.rglob("*.wav"))
    print(f"Total WAV files: {len(wav_files)}")

    rows = []
    for p in wav_files:
        parts = p.stem.split("-")
        if len(parts) != 7:
            continue
        eid = parse_ravdess_filename(p.name)
        if eid is None:
            continue
        actor_id = int(parts[6])
        rows.append({
            "path": str(p),
            "modality": parts[0],   # 03=audio-only
            "vocal_channel": parts[1],
            "emotion_raw": parts[2],
            "intensity": parts[3],
            "statement": parts[4],
            "repetition": parts[5],
            "actor": actor_id,
            "gender": "male" if actor_id % 2 == 1 else "female",
            "emotion_id": eid,
            "emotion_name": ID_TO_LABEL[eid],
        })
    df = pd.DataFrame(rows)
    print(f"Valid files indexed: {len(df)}")

    counts = df["emotion_name"].value_counts().reindex(EMOTIONS, fill_value=0)
    print(f"\nPer-class counts (calm merged -> neutral):")
    pct = counts / counts.sum() * 100
    for em in EMOTIONS:
        print(f"  {em:<10} {int(counts[em]):>4,}  ({pct[em]:5.1f}%)")
    stats = imbalance_stats(counts, "RAVDESS")
    _print_stats(stats)

    print(f"\nGender split: {df['gender'].value_counts().to_dict()}")
    print(f"Actors: {df['actor'].nunique()}  (IDs {df['actor'].min()}–{df['actor'].max()})")
    print(f"Intensities: {df['intensity'].value_counts().to_dict()}")

    print(f"\nPer-gender per-class breakdown:")
    for gender in ["male", "female"]:
        sub = df[df["gender"] == gender]
        gc = sub["emotion_name"].value_counts().reindex(EMOTIONS, fill_value=0)
        print(f"  [{gender}]")
        for em in EMOTIONS:
            print(f"    {em:<10} {int(gc[em]):>3,}")

    return df, counts


# ── CREMAD ────────────────────────────────────────────────────────────────────

def analyze_cremad():
    print_section("CREMA-D")
    cremad_dir = DATA_DIR / "cremad"
    wav_files = sorted((cremad_dir / "AudioWAV").rglob("*.wav"))
    print(f"Total WAV files: {len(wav_files)}")

    # CREMAD filename: ACTORID_SENTENCE_EMOTION_LEVEL.wav
    # Emotions: ANG, DIS, FEA, HAP, NEU, SAD
    cremad_map = {"ANG": 0, "DIS": 1, "FEA": 2, "HAP": 3, "NEU": 6, "SAD": 4}

    rows = []
    for p in wav_files:
        parts = p.stem.split("_")
        if len(parts) < 3:
            continue
        emo_code = parts[2]
        eid = cremad_map.get(emo_code)
        if eid is None:
            continue
        actor_id = int(parts[0]) if parts[0].isdigit() else None
        rows.append({
            "path": str(p),
            "actor": actor_id,
            "sentence": parts[1] if len(parts) > 1 else None,
            "emotion_code": emo_code,
            "level": parts[3] if len(parts) > 3 else None,
            "emotion_id": eid,
            "emotion_name": ID_TO_LABEL[eid],
        })
    df = pd.DataFrame(rows)
    print(f"Valid files indexed: {len(df)}")

    counts = df["emotion_name"].value_counts().reindex(EMOTIONS, fill_value=0)
    print(f"\nPer-class counts:")
    pct = counts / counts.sum() * 100
    for em in EMOTIONS:
        print(f"  {em:<10} {int(counts[em]):>5,}  ({pct[em]:5.1f}%)")
    stats = imbalance_stats(counts, "CREMA-D")
    _print_stats(stats)

    if "actor" in df.columns and df["actor"].notna().any():
        print(f"\nActors: {df['actor'].nunique()}")

    if "level" in df.columns:
        print(f"\nIntensity levels: {df['level'].value_counts().to_dict()}")

    return df, counts


# ── Combined audio ────────────────────────────────────────────────────────────

def analyze_combined_audio(ravdess_counts, cremad_counts):
    print_section("Combined audio (RAVDESS + CREMA-D)")
    combined = ravdess_counts.add(cremad_counts, fill_value=0).reindex(EMOTIONS, fill_value=0)
    total = combined.sum()
    pct = combined / total * 100
    print(f"Total clips: {int(total):,}")
    for em in EMOTIONS:
        print(f"  {em:<10} {int(combined[em]):>5,}  ({pct[em]:5.1f}%)")
    stats = imbalance_stats(combined, "Audio combined")
    _print_stats(stats)
    return combined


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_all(fer_counts, ferplus_counts, ravdess_counts, cremad_counts, combined_audio):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Class Imbalance Analysis — All Datasets", fontsize=14, fontweight="bold")

    datasets = [
        ("FER-2013", fer_counts, "steelblue"),
        ("FER+", ferplus_counts, "royalblue"),
        ("RAVDESS", ravdess_counts, "darkorange"),
        ("CREMA-D", cremad_counts, "tomato"),
        ("Audio Combined", combined_audio, "purple"),
    ]

    for idx, (title, counts, color) in enumerate(datasets):
        ax = axes[idx // 3][idx % 3]
        counts_reindexed = counts.reindex(EMOTIONS, fill_value=0)
        bars = ax.bar(EMOTIONS, counts_reindexed.values, color=color, edgecolor="black", alpha=0.8)
        ax.set_title(f"{title}  (n={int(counts_reindexed.sum()):,})", fontsize=11)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=35)
        ratio = counts_reindexed.max() / max(counts_reindexed.min(), 1)
        ax.set_xlabel(f"imbalance ratio {ratio:.1f}x", fontsize=9, color="gray")
        for bar, val in zip(bars, counts_reindexed.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + counts_reindexed.max() * 0.01,
                    str(int(val)), ha="center", va="bottom", fontsize=8)

    # 6th panel: normalised comparison across all 5
    ax = axes[1][2]
    x = np.arange(len(EMOTIONS))
    width = 0.15
    for i, (title, counts, color) in enumerate(datasets):
        norm = counts.reindex(EMOTIONS, fill_value=0) / counts.sum()
        ax.bar(x + i * width, norm.values, width, label=title, color=color, alpha=0.8, edgecolor="black")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(EMOTIONS, rotation=35)
    ax.set_ylabel("Fraction of dataset")
    ax.set_title("Normalised comparison", fontsize=11)
    ax.legend(fontsize=7)

    plt.tight_layout()
    out = DATA_DIR.parent / "class_imbalance_analysis.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved plot -> {out}")
    plt.close()


# ── util ──────────────────────────────────────────────────────────────────────

def _print_stats(s: dict):
    print(f"\n  Imbalance stats:")
    print(f"    Max class : {s['max_class']} ({s['max_count']:,})")
    print(f"    Min class : {s['min_class']} ({s['min_count']:,})")
    print(f"    Ratio     : {s['ratio_max_min']}x")
    print(f"    Std dev   : {s['std_dev']:,}  |  CV: {s['cv_%']}%")
    print(f"    KL div from uniform: {s['kl_div_from_uniform']}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    set_seed(42)

    fer_df, fer_counts           = analyze_fer2013()
    ferplus_counts               = analyze_ferplus(fer_df)
    ravdess_df, ravdess_counts   = analyze_ravdess()
    cremad_df,  cremad_counts    = analyze_cremad()
    combined_audio               = analyze_combined_audio(ravdess_counts, cremad_counts)

    plot_all(fer_counts, ferplus_counts, ravdess_counts, cremad_counts, combined_audio)

    print_section("Summary table")
    rows = []
    for name, counts in [
        ("FER-2013", fer_counts),
        ("FER+", ferplus_counts),
        ("RAVDESS", ravdess_counts),
        ("CREMA-D", cremad_counts),
        ("Audio combined", combined_audio),
    ]:
        s = imbalance_stats(counts.reindex(EMOTIONS, fill_value=0), name)
        rows.append(s)
    summary = pd.DataFrame(rows).set_index("dataset")
    print(summary.to_string())


if __name__ == "__main__":
    main()
