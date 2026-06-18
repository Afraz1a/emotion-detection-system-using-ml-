"""Stage 3: Train the audio model on RAVDESS."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from utils.labels import EMOTIONS, NUM_CLASSES
from utils.helpers import set_seed, get_device, CHECKPOINT_DIR
from datasets import get_combined_audio_splits
from models import AudioModel


class FocalLoss(nn.Module):
    """Focal loss: down-weights easy examples so the model focuses on hard ones."""
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.05):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, weight=self.weight,
                                         label_smoothing=self.label_smoothing,
                                         reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def compute_class_weights(labels):
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float32)
    weights = counts.sum() / (NUM_CLASSES * counts + 1e-6)
    return torch.tensor(weights, dtype=torch.float32)


def make_weighted_sampler(labels):
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float32)
    class_w = 1.0 / (counts + 1e-6)
    sample_weights = torch.tensor([class_w[l] for l in labels], dtype=torch.float32)
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights),
                                 replacement=True)


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="train" if train else "val ", leave=False)
    with torch.set_grad_enabled(train):
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
            pbar.set_postfix(loss=f"{total_loss/total:.4f}",
                             acc=f"{total_correct/total:.4f}")
    return total_loss / total, total_correct / total


def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds.extend(model(x).argmax(1).cpu().tolist())
            labels.extend(y.tolist())
    return np.array(labels), np.array(preds)


def main():
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    train_ds, val_ds, test_ds = get_combined_audio_splits()
    print(f"Train clips: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    train_labels = np.array(train_ds.labels)
    sampler = make_weighted_sampler(train_labels)
    # batch_size 16: fine-tuning wav2vec2 layers needs gradient memory (6GB RTX 4050).
    train_loader = DataLoader(train_ds, batch_size=16, sampler=sampler, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False,   num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=16, shuffle=False,   num_workers=2)

    model = AudioModel(num_classes=NUM_CLASSES, dropout=0.3, unfreeze_last_n=4).to(device)
    # NOTE: WeightedRandomSampler already balances batches, so we do NOT also apply
    # class weights here — stacking both over-corrects and caused mode collapse.
    criterion = FocalLoss(gamma=2.0, weight=None, label_smoothing=0.05)
    # Discriminative LR: fine-tune unfrozen wav2vec2 layers gently, train head faster.
    optimizer = AdamW([
        {"params": model.backbone_parameters(), "lr": 1e-5},
        {"params": model.head_parameters(),     "lr": 3e-4},
    ], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=50)

    best_val_acc, patience, bad = 0.0, 10, 0
    ckpt_path = CHECKPOINT_DIR / "audio_best.pt"

    for epoch in range(1, 51):
        tl, ta = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        vl, va = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()
        print(f"Epoch {epoch:02d} | train loss {tl:.4f} acc {ta:.4f} | "
              f"val loss {vl:.4f} acc {va:.4f}")

        if va > best_val_acc:
            best_val_acc = va
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> saved best  (val_acc={va:.4f})")
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stop at epoch {epoch}")
                break

    model.load_state_dict(torch.load(ckpt_path))
    y_true, y_pred = evaluate(model, test_loader, device)
    print(f"\nTest accuracy: {(y_true == y_pred).mean():.4f}")
    print("\nPer-class report:")
    print(classification_report(y_true, y_pred, target_names=list(EMOTIONS), digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()
