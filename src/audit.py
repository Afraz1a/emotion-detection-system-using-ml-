"""Fairness audit: per-subgroup accuracy + demographic parity + equalized odds.

Audits both modalities on their test sets:
  - FACE: subgroups predicted by the FairFace annotator (gender, race)
  - AUDIO: real CREMA-D demographics + RAVDESS gender (gender, race, age)

Outputs:
  - fairness_report.csv   (per-group accuracy for every dimension)
  - fairness_audit.png    (per-group accuracy bar charts)
  - console summary with demographic parity / equalized odds differences

Run:  python src/audit.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame

from utils.labels import EMOTIONS, NUM_CLASSES
from utils.helpers import set_seed, get_device, CHECKPOINT_DIR, PROJECT_ROOT
from models import build_face_model, AudioModel
from datasets import get_combined_face_splits, get_combined_audio_splits
from subgroups import (load_subgroup_annotator, predict_face_subgroups,
                       subgroup_for_audio_path, collect_audio_paths)

MIN_GROUP = 20  # ignore subgroups with fewer than this many samples


# ---------------- prediction collection ----------------

@torch.no_grad()
def audit_face(device):
    print("Auditing FACE model...")
    _, _, test_ds = get_combined_face_splits()
    loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2)

    face = build_face_model(num_classes=7, pretrained=False).to(device)
    face.load_state_dict(torch.load(CHECKPOINT_DIR / "face_best.pt", map_location=device))
    face.eval()
    annot = load_subgroup_annotator(device)

    y_true, y_pred, genders, races = [], [], [], []
    for x, y in loader:
        x = x.to(device)
        y_pred.extend(face(x).argmax(1).cpu().tolist())
        y_true.extend(y.tolist())
        g, r = predict_face_subgroups(annot, x, device)
        genders.extend(g); races.extend(r)

    return (np.array(y_true), np.array(y_pred),
            {"gender": np.array(genders), "race": np.array(races)})


@torch.no_grad()
def audit_audio(device):
    print("Auditing AUDIO model...")
    _, _, test_ds = get_combined_audio_splits()
    loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2)

    audio = AudioModel(num_classes=7).to(device)
    audio.load_state_dict(torch.load(CHECKPOINT_DIR / "audio_best.pt", map_location=device))
    audio.eval()

    y_true, y_pred = [], []
    for x, y in loader:
        y_pred.extend(audio(x.to(device)).argmax(1).cpu().tolist())
        y_true.extend(y.tolist())

    paths = collect_audio_paths(test_ds)
    assert len(paths) == len(y_true), f"path/pred mismatch {len(paths)} vs {len(y_true)}"
    subs = [subgroup_for_audio_path(p) for p in paths]
    return (np.array(y_true), np.array(y_pred),
            {"gender": np.array([s["gender"] for s in subs]),
             "race":   np.array([s["race"]   for s in subs]),
             "age":    np.array([s["age"]    for s in subs])})


# ---------------- metrics ----------------

def _macro_dp_gap(y_pred, sf, groups, n_classes):
    """Demographic parity gap (multiclass): for each class, max-min selection
    rate across groups; averaged over classes. 0 = identical prediction mix."""
    gaps = []
    for c in range(n_classes):
        rates = [np.mean(y_pred[sf == g] == c) for g in groups]
        gaps.append(max(rates) - min(rates))
    return float(np.mean(gaps))


def _macro_eo_gap(y_true, y_pred, sf, groups, n_classes):
    """Equalized odds gap (multiclass): for each class, max-min recall (TPR)
    across groups that contain that class; averaged over classes."""
    gaps = []
    for c in range(n_classes):
        rates = []
        for g in groups:
            m = (sf == g) & (y_true == c)
            if m.sum() == 0:
                continue
            rates.append(np.mean(y_pred[m] == c))
        if len(rates) >= 2:
            gaps.append(max(rates) - min(rates))
    return float(np.mean(gaps)) if gaps else float("nan")


def audit_dimension(y_true, y_pred, sensitive, name):
    """Returns (per_group_df, dp_gap, eo_gap, acc_gap) for one sensitive
    dimension, dropping 'unknown' and tiny groups."""
    mask = sensitive != "unknown"
    yt, yp, sf = y_true[mask], y_pred[mask], sensitive[mask]
    # drop tiny groups
    counts = pd.Series(sf).value_counts()
    keep_groups = counts[counts >= MIN_GROUP].index
    keep = np.isin(sf, keep_groups)
    yt, yp, sf = yt[keep], yp[keep], sf[keep]
    if len(np.unique(sf)) < 2:
        return None, None, None, None

    mf = MetricFrame(metrics={"accuracy": accuracy_score, "count": lambda a, b: len(a)},
                     y_true=yt, y_pred=yp, sensitive_features=sf)
    per_group = mf.by_group.copy()
    per_group["dimension"] = name

    groups = list(keep_groups)
    dp_gap = _macro_dp_gap(yp, sf, groups, NUM_CLASSES)
    eo_gap = _macro_eo_gap(yt, yp, sf, groups, NUM_CLASSES)
    acc_gap = float(mf.by_group["accuracy"].max() - mf.by_group["accuracy"].min())
    return per_group, dp_gap, eo_gap, acc_gap


def run_modality(label, y_true, y_pred, sens_dict, rows, summary):
    overall = accuracy_score(y_true, y_pred)
    print(f"\n=== {label}  (overall accuracy {overall:.3f}) ===")
    for dim, sens in sens_dict.items():
        per_group, dp_gap, eo_gap, acc_gap = audit_dimension(y_true, y_pred, sens, dim)
        if per_group is None:
            print(f"  [{dim}] skipped (not enough labeled groups)")
            continue
        print(f"  [{dim}]  accuracy_gap={acc_gap:.3f}  "
              f"demographic_parity_gap={dp_gap:.3f}  equalized_odds_gap={eo_gap:.3f}")
        for grp, row in per_group.iterrows():
            print(f"      {grp:<18} acc={row['accuracy']:.3f}  n={int(row['count'])}")
            rows.append({"modality": label, "dimension": dim, "group": grp,
                         "accuracy": round(row["accuracy"], 4), "n": int(row["count"])})
        summary.append({"modality": label, "dimension": dim,
                        "accuracy_gap": round(acc_gap, 4),
                        "demographic_parity_gap": round(dp_gap, 4),
                        "equalized_odds_gap": round(eo_gap, 4)})


# ---------------- plot ----------------

def plot_report(rows_df, out):
    dims = rows_df[["modality", "dimension"]].drop_duplicates().values.tolist()
    n = len(dims)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    fig.suptitle("Fairness Audit — Per-Subgroup Accuracy", fontsize=14, fontweight="bold")
    for ax, (mod, dim) in zip(axes[0], dims):
        sub = rows_df[(rows_df.modality == mod) & (rows_df.dimension == dim)]
        bars = ax.bar(sub["group"], sub["accuracy"], color="steelblue",
                      edgecolor="black", alpha=0.85)
        for b, v, nn in zip(bars, sub["accuracy"], sub["n"]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}\nn={nn}",
                    ha="center", fontsize=8)
        ax.axhline(sub["accuracy"].mean(), color="darkorange", ls="--", lw=1,
                   label="mean")
        ax.set_title(f"{mod} — {dim}")
        ax.set_ylim(0, 1.1); ax.set_ylabel("accuracy")
        ax.tick_params(axis="x", rotation=30); ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved chart -> {out}")


# ---------------- main ----------------

def main():
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    rows, summary = [], []

    ft, fp, fsens = audit_face(device)
    run_modality("face", ft, fp, fsens, rows, summary)

    at, ap, asens = audit_audio(device)
    run_modality("audio", at, ap, asens, rows, summary)

    rows_df = pd.DataFrame(rows)
    out_csv = PROJECT_ROOT / "fairness_report.csv"
    rows_df.to_csv(out_csv, index=False)
    print(f"\nSaved per-group report -> {out_csv}")

    print("\n=== Fairness summary (lower diff = fairer) ===")
    print(pd.DataFrame(summary).to_string(index=False))

    plot_report(rows_df, PROJECT_ROOT / "fairness_audit.png")
    print("\nNOTE: face subgroups are PREDICTED by the FairFace annotator "
          "(noisy, esp. race on 48x48 FER images); audio uses real CREMA-D demographics.")


if __name__ == "__main__":
    main()
