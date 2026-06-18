"""Train a gender + race subgroup annotator on FairFace.

This model is NOT an emotion classifier. It is used by the fairness audit to
assign demographic subgroup labels (gender, race) to emotion test images that
lack such labels (FER+, AffectNet). CREMA-D has real demographics and does not
need this model.

Run:  python src/train_subgroup.py
Saves: checkpoints/subgroup_best.pt
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from utils.labels import FAIRFACE_RACES, FAIRFACE_GENDERS
from utils.helpers import set_seed, get_device, CHECKPOINT_DIR
from datasets import FairFaceDataset

N_GENDER = len(FAIRFACE_GENDERS)
N_RACE = len(FAIRFACE_RACES)
EPOCHS = 8


class SubgroupNet(nn.Module):
    """ResNet-18 backbone with two heads: gender and race."""

    def __init__(self, n_gender=N_GENDER, n_race=N_RACE, dropout=0.3, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        in_feat = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.drop = nn.Dropout(dropout)
        self.gender_head = nn.Linear(in_feat, n_gender)
        self.race_head = nn.Linear(in_feat, n_race)

    def forward(self, x):
        feat = self.drop(self.backbone(x))
        return self.gender_head(feat), self.race_head(feat)


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    tot_loss, n = 0.0, 0
    g_correct, r_correct = 0, 0
    with torch.set_grad_enabled(train):
        for x, g, r in tqdm(loader, leave=False, desc="train" if train else "val"):
            x, g, r = x.to(device), g.to(device), r.to(device)
            g_logits, r_logits = model(x)
            loss = criterion(g_logits, g) + criterion(r_logits, r)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            tot_loss += loss.item() * x.size(0)
            g_correct += (g_logits.argmax(1) == g).sum().item()
            r_correct += (r_logits.argmax(1) == r).sum().item()
            n += x.size(0)
    return tot_loss / n, g_correct / n, r_correct / n


def main():
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    train_ds = FairFaceDataset(split="train", augment=True)
    val_ds = FairFaceDataset(split="val", augment=False)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False,
                            num_workers=2, pin_memory=True)

    model = SubgroupNet().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_score, ckpt = 0.0, CHECKPOINT_DIR / "subgroup_best.pt"
    for epoch in range(1, EPOCHS + 1):
        tl, tg, tr = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        vl, vg, vr = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()
        print(f"Epoch {epoch:02d} | train g_acc={tg:.3f} r_acc={tr:.3f} | "
              f"val g_acc={vg:.3f} r_acc={vr:.3f}")
        score = (vg + vr) / 2
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), ckpt)
            print(f"  -> saved best (gender {vg:.3f}, race {vr:.3f})")


if __name__ == "__main__":
    main()
