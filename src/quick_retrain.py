"""Quick 5-epoch retrain to demonstrate imbalance-fix gains vs baseline checkpoints."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models
from sklearn.metrics import classification_report
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.labels import EMOTIONS, NUM_CLASSES
from utils.helpers import set_seed, get_device, CHECKPOINT_DIR
from datasets import FERPlusDataset, get_combined_audio_splits
from models import AudioModel

EPOCHS = 5
LABELS = list(EMOTIONS)


# ── shared ────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.1):
        super().__init__()
        self.gamma, self.weight, self.ls = gamma, weight, label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight,
                             label_smoothing=self.ls, reduction="none")
        return ((1 - torch.exp(-ce)) ** self.gamma * ce).mean()


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in tqdm(loader, leave=False, desc="train" if train else "eval"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, device):
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.extend(model(x.to(device)).argmax(1).cpu().tolist())
            truths.extend(y.tolist())
    return np.array(truths), np.array(preds)


def make_sampler(labels_list):
    labels = np.array(labels_list)
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float32)
    class_w = 1.0 / (counts + 1e-6)
    sw = torch.tensor([class_w[l] for l in labels], dtype=torch.float32)
    return WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)


def class_f1(truths, preds):
    from sklearn.metrics import f1_score
    return f1_score(truths, preds, average=None, labels=list(range(NUM_CLASSES)),
                    zero_division=0)


# ── BASELINE loaders (load old checkpoints, evaluate on same test sets) ───────

def baseline_face(device, test_loader):
    m = models.resnet34()
    m.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.fc.in_features, 7))
    m.load_state_dict(torch.load(CHECKPOINT_DIR / "face_best.pt", map_location=device))
    m = m.to(device)
    t, p = evaluate(m, test_loader, device)
    return t, p


def baseline_audio(device, test_loader):
    m = AudioModel(num_classes=7).to(device)
    m.load_state_dict(torch.load(CHECKPOINT_DIR / "audio_best.pt", map_location=device))
    t, p = evaluate(m, test_loader, device)
    return t, p


# ── FACE retrain ──────────────────────────────────────────────────────────────

def retrain_face(device):
    print("\n" + "="*55)
    print("FACE MODEL — 5-epoch fine-tune with imbalance fixes")
    print("="*55)
    train_ds = FERPlusDataset(split="Training",    augment=True)
    val_ds   = FERPlusDataset(split="PublicTest",  augment=False)
    test_ds  = FERPlusDataset(split="PrivateTest", augment=False)

    sampler = make_sampler(train_ds.df["unified_label"].tolist())
    train_loader = DataLoader(train_ds, batch_size=128, sampler=sampler,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False,
                              num_workers=2, pin_memory=True)

    # Start from existing checkpoint so 5 epochs is meaningful
    model = models.resnet34()
    model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(model.fc.in_features, 7))
    model.load_state_dict(torch.load(CHECKPOINT_DIR / "face_best.pt", map_location=device))
    model = model.to(device)

    labels_arr = np.array(train_ds.df["unified_label"].tolist())
    counts = np.bincount(labels_arr, minlength=NUM_CLASSES).astype(np.float32)
    class_w = torch.tensor(counts.sum() / (NUM_CLASSES * counts + 1e-6)).to(device)
    criterion = FocalLoss(gamma=2.0, weight=class_w, label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    for epoch in range(1, EPOCHS + 1):
        tl, ta = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        vl, va = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)
        scheduler.step()
        print(f"  Epoch {epoch}/{EPOCHS}  train_acc={ta:.3f}  val_acc={va:.3f}")

    torch.save(model.state_dict(), CHECKPOINT_DIR / "face_fixed.pt")
    t_base, p_base = baseline_face(device, test_loader)
    t_new,  p_new  = evaluate(model, test_loader, device)
    return t_base, p_base, t_new, p_new


# ── AUDIO retrain ─────────────────────────────────────────────────────────────

def retrain_audio(device):
    print("\n" + "="*55)
    print("AUDIO MODEL — 5-epoch fine-tune with imbalance fixes")
    print("="*55)
    train_ds, val_ds, test_ds = get_combined_audio_splits()

    sampler = make_sampler(train_ds.labels)
    train_loader = DataLoader(train_ds, batch_size=64, sampler=sampler, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False,   num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False,   num_workers=2)

    model = AudioModel(num_classes=7).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_DIR / "audio_best.pt", map_location=device))

    labels_arr = np.array(train_ds.labels)
    counts = np.bincount(labels_arr, minlength=NUM_CLASSES).astype(np.float32)
    class_w = torch.tensor(counts.sum() / (NUM_CLASSES * counts + 1e-6)).to(device)
    criterion = FocalLoss(gamma=2.0, weight=class_w, label_smoothing=0.05)
    optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    for epoch in range(1, EPOCHS + 1):
        tl, ta = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        vl, va = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)
        scheduler.step()
        print(f"  Epoch {epoch}/{EPOCHS}  train_acc={ta:.3f}  val_acc={va:.3f}")

    torch.save(model.state_dict(), CHECKPOINT_DIR / "audio_fixed.pt")
    t_base, p_base = baseline_audio(device, test_loader)
    t_new,  p_new  = evaluate(model, test_loader, device)
    return t_base, p_base, t_new, p_new


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_gains(face_base, face_new, audio_base, audio_new, out="results_gain.png"):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Per-Class F1 Gain: Baseline vs After Imbalance Fixes (5-epoch fine-tune)",
                 fontsize=13, fontweight="bold")

    x = np.arange(NUM_CLASSES)
    w = 0.35

    for ax, (t_b, p_b, p_n), title in zip(
        axes,
        [face_base + (face_new,), audio_base + (audio_new,)],
        ["Face Model (FER+)", "Audio Model (RAVDESS+CREMA-D)"],
    ):
        f1_base = class_f1(t_b, p_b)
        f1_new  = class_f1(t_b, p_n)
        delta   = f1_new - f1_base

        bars_b = ax.bar(x - w/2, f1_base, w, label="Baseline", color="tomato",
                        edgecolor="black", alpha=0.85)
        bars_n = ax.bar(x + w/2, f1_new,  w, label="Fixed",    color="steelblue",
                        edgecolor="black", alpha=0.85)

        for i, (b, n, d) in enumerate(zip(f1_base, f1_new, delta)):
            color = "green" if d >= 0 else "red"
            ax.text(i, max(b, n) + 0.02, f"{d:+.2f}", ha="center", fontsize=8,
                    color=color, fontweight="bold")

        acc_b = (np.array(p_b) == np.array(t_b)).mean()
        acc_n = (np.array(p_n) == np.array(t_b)).mean()
        ax.set_title(f"{title}\nOverall acc: {acc_b:.3f} -> {acc_n:.3f} "
                     f"({'↑' if acc_n > acc_b else '↓'}{abs(acc_n-acc_b):.3f})")
        ax.set_xticks(x); ax.set_xticklabels(LABELS, rotation=30)
        ax.set_ylim(0, 1.15); ax.set_ylabel("F1 score")
        ax.legend(); ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved -> {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    ft_b, fp_b, ft_n, fp_n = retrain_face(device)
    at_b, ap_b, at_n, ap_n = retrain_audio(device)

    print("\n--- FACE per-class report (baseline) ---")
    print(classification_report(ft_b, fp_b, target_names=LABELS, digits=3, zero_division=0))
    print("--- FACE per-class report (fixed) ---")
    print(classification_report(ft_b, fp_n, target_names=LABELS, digits=3, zero_division=0))

    print("\n--- AUDIO per-class report (baseline) ---")
    print(classification_report(at_b, ap_b, target_names=LABELS, digits=3, zero_division=0))
    print("--- AUDIO per-class report (fixed) ---")
    print(classification_report(at_b, ap_n, target_names=LABELS, digits=3, zero_division=0))

    plot_gains(
        face_base=(ft_b, fp_b),
        face_new=fp_n,
        audio_base=(at_b, ap_b),
        audio_new=ap_n,
        out="results_gain.png",
    )


if __name__ == "__main__":
    main()
