"""Fusion benchmark: prove the fused model beats face-only and voice-only,
and measure inference latency against the <300ms real-time goal.

Because no large naturally-paired (same-recording) face+voice emotion dataset is
available here (RAVDESS is audio-only; CREMA-D video is .flv + has no 'surprise'
and a face-domain mismatch), we use EMOTION-MATCHED cross-dataset pairing: each
eval pair is a face test sample and an audio test sample sharing the same emotion
label. This is a standard way to evaluate late fusion and uses the exact test
sets both models were validated on. Limitation (different identities per pair)
is reported alongside the numbers.

Run:  python src/evaluate_fusion.py
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# NOTE: import numpy/pandas/sklearn BEFORE torch — on Windows, importing sklearn
# after torch triggers a native threadpool/OpenMP segfault.
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.labels import EMOTIONS, NUM_CLASSES
from utils.helpers import set_seed, get_device, CHECKPOINT_DIR, PROJECT_ROOT
from models import build_face_model, AudioModel
from datasets import get_combined_face_splits, get_combined_audio_splits
from fusion import weighted_average, confidence_gated

N_PAIRS_PER_CLASS = 300  # emotion-matched pairs sampled per emotion


@torch.no_grad()
def collect_probs_face(device):
    _, _, test_ds = get_combined_face_splits()
    loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2)
    model = build_face_model(num_classes=7, pretrained=False).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_DIR / "face_best.pt", map_location=device))
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        probs.append(F.softmax(model(x.to(device)), dim=1).cpu().numpy())
        labels.extend(y.tolist())
    return np.concatenate(probs), np.array(labels)


@torch.no_grad()
def collect_probs_audio(device):
    _, _, test_ds = get_combined_audio_splits()
    loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2)
    model = AudioModel(num_classes=7).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_DIR / "audio_best.pt", map_location=device))
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        probs.append(F.softmax(model(x.to(device)), dim=1).cpu().numpy())
        labels.extend(y.tolist())
    return np.concatenate(probs), np.array(labels)


def build_pairs(face_probs, face_labels, audio_probs, audio_labels, seed=42):
    """Emotion-matched pairs. Returns (face_p, audio_p, y) arrays."""
    rng = np.random.RandomState(seed)
    fp, ap, ys = [], [], []
    for c in range(NUM_CLASSES):
        f_idx = np.where(face_labels == c)[0]
        a_idx = np.where(audio_labels == c)[0]
        if len(f_idx) == 0 or len(a_idx) == 0:
            continue
        n = min(N_PAIRS_PER_CLASS, max(len(f_idx), len(a_idx)))
        fsel = rng.choice(f_idx, n, replace=len(f_idx) < n)
        asel = rng.choice(a_idx, n, replace=len(a_idx) < n)
        fp.append(face_probs[fsel]); ap.append(audio_probs[asel])
        ys.append(np.full(n, c))
    return np.concatenate(fp), np.concatenate(ap), np.concatenate(ys)


def score(name, preds, y):
    acc = accuracy_score(y, preds)
    mf1 = f1_score(y, preds, average="macro")
    print(f"  {name:<24} acc={acc:.4f}  macro-F1={mf1:.4f}")
    return {"model": name, "accuracy": round(acc, 4), "macro_f1": round(mf1, 4)}


def measure_latency(device, runs=50):
    """Single-sample inference latency (ms) on the current device."""
    face = build_face_model(num_classes=7, pretrained=False).to(device).eval()
    face.load_state_dict(torch.load(CHECKPOINT_DIR / "face_best.pt", map_location=device))
    audio = AudioModel(num_classes=7).to(device).eval()
    audio.load_state_dict(torch.load(CHECKPOINT_DIR / "audio_best.pt", map_location=device))

    fimg = torch.randn(1, 3, 96, 96, device=device)
    awav = torch.randn(1, 48000, device=device)

    def timeit(fn):
        # warmup
        for _ in range(5):
            fn()
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            fn()
        if device == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / runs * 1000

    with torch.no_grad():
        f_ms = timeit(lambda: F.softmax(face(fimg), 1))
        a_ms = timeit(lambda: F.softmax(audio(awav), 1))
    return f_ms, a_ms, f_ms + a_ms


def main():
    set_seed(42)
    device = get_device()
    print(f"Device: {device}\n")

    print("Collecting face probabilities...")
    face_probs, face_labels = collect_probs_face(device)
    print("Collecting audio probabilities...")
    audio_probs, audio_labels = collect_probs_audio(device)

    fp, ap, y = build_pairs(face_probs, face_labels, audio_probs, audio_labels)
    print(f"\nEmotion-matched eval pairs: {len(y)}\n")

    print("=== Fusion benchmark ===")
    results = []
    results.append(score("Face-only",            fp.argmax(1), y))
    results.append(score("Voice-only",           ap.argmax(1), y))
    wavg = np.array([weighted_average(fp[i], ap[i]) for i in range(len(y))])
    results.append(score("Fused (weighted avg)", wavg.argmax(1), y))
    gated = np.array([confidence_gated(fp[i], ap[i]) for i in range(len(y))])
    results.append(score("Fused (conf. gated)",  gated.argmax(1), y))

    print("\n=== Latency (single sample) ===")
    f_ms, a_ms, total_ms = measure_latency(device)
    print(f"  Face:   {f_ms:6.1f} ms")
    print(f"  Audio:  {a_ms:6.1f} ms")
    print(f"  Fused:  {total_ms:6.1f} ms   (target < 300 ms: "
          f"{'PASS' if total_ms < 300 else 'FAIL'})")

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    names = [r["model"] for r in results]
    accs = [r["accuracy"] for r in results]
    f1s = [r["macro_f1"] for r in results]
    x = np.arange(len(names)); w = 0.38
    colors = ["#bbb", "#bbb", "steelblue", "#2a6"]
    ax.bar(x - w/2, accs, w, label="accuracy", color=colors, edgecolor="black")
    ax.bar(x + w/2, f1s, w, label="macro-F1", color=colors, edgecolor="black", alpha=0.55)
    best_solo = max(accs[0], accs[1])
    ax.axhline(best_solo, color="darkorange", ls="--", lw=1, label="best solo acc")
    for i, (a, fv) in enumerate(zip(accs, f1s)):
        ax.text(i - w/2, a + 0.005, f"{a:.3f}", ha="center", fontsize=8)
        ax.text(i + w/2, fv + 0.005, f"{fv:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15)
    ax.set_ylim(0, 1.05); ax.set_ylabel("score")
    ax.set_title(f"Fusion Benchmark — fused vs solo  (latency {total_ms:.0f}ms)")
    ax.legend()
    plt.tight_layout()
    out = PROJECT_ROOT / "fusion_benchmark.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved chart -> {out}")

    pd.DataFrame(results).to_csv(PROJECT_ROOT / "fusion_benchmark.csv", index=False)
    print(f"Saved table -> {PROJECT_ROOT / 'fusion_benchmark.csv'}")
    print("\nNOTE: emotion-matched cross-dataset pairs (different identities per "
          "pair); tests late-fusion gain, not same-recording fusion.")


if __name__ == "__main__":
    main()
