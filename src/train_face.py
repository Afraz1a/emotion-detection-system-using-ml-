"""Train the face model on FER+ (improved FER-2013 labels)."""

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
from torchvision import models

from utils.labels import EMOTIONS, NUM_CLASSES
from utils.helpers import set_seed, get_device, CHECKPOINT_DIR
from datasets import FERPlusDataset, get_combined_face_splits
from models import build_face_model


def build_model(num_classes=7, dropout=0.4):
    """ResNet-34 fine-tuned for emotion classification (shared builder)."""
    return build_face_model(num_classes=num_classes, dropout=dropout, pretrained=True)


class FocalLoss(nn.Module):
    """Focal loss: down-weights easy examples so the model focuses on hard ones."""
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.1):
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


def _labels_of(dataset):
    """Flat label array; supports both FERPlusDataset (.df) and combined (.labels)."""
    if hasattr(dataset, "labels"):
        return np.asarray(dataset.labels)
    return dataset.df["unified_label"].values


def compute_class_weights(dataset):
    labels = _labels_of(dataset)
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float32)
    weights = counts.sum() / (NUM_CLASSES * counts + 1e-6)
    return torch.tensor(weights, dtype=torch.float32)


def make_weighted_sampler(dataset):
    labels = _labels_of(dataset)
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

    train_ds, val_ds, test_ds = get_combined_face_splits(image_size=96)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    sampler = make_weighted_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=128, sampler=sampler,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False,
                             num_workers=2, pin_memory=True)

    model = build_model(num_classes=NUM_CLASSES, dropout=0.4).to(device)
    class_weights = compute_class_weights(train_ds).to(device)
    criterion = FocalLoss(gamma=2.0, weight=class_weights, label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=40)

    best_val_acc = 0.0
    patience, bad_epochs = 10, 0
    ckpt_path = CHECKPOINT_DIR / "face_best.pt"

    for epoch in range(1, 41):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer,
                                    device, train=True)
        vl_loss, vl_acc = run_epoch(model, val_loader, criterion, optimizer,
                                    device, train=False)
        scheduler.step()
        print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {vl_loss:.4f} acc {vl_acc:.4f}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> saved best to {ckpt_path}  (val_acc={vl_acc:.4f})")
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
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