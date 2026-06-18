"""Stage 4: Combine face and audio predictions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from utils.labels import EMOTIONS
from utils.helpers import get_device, CHECKPOINT_DIR
from models import build_face_model, AudioModel


def load_face_model(device, ckpt=None):
    ckpt = ckpt or (CHECKPOINT_DIR / "face_best.pt")
    m = build_face_model(num_classes=7, pretrained=False).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device))
    m.eval()
    return m


def load_audio_model(device, ckpt=None):
    ckpt = ckpt or (CHECKPOINT_DIR / "audio_best.pt")
    m = AudioModel(num_classes=7).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device))
    m.eval()
    return m


@torch.no_grad()
def predict_face(model, face_tensor, device):
    """face_tensor: (1, 3, H, W) already normalized."""
    return F.softmax(model(face_tensor.to(device)), dim=1).cpu().numpy()[0]


@torch.no_grad()
def predict_audio(model, audio_tensor, device):
    """audio_tensor: (1, 3, n_mfcc, T)."""
    return F.softmax(model(audio_tensor.to(device)), dim=1).cpu().numpy()[0]


def weighted_average(face_probs, audio_probs, face_w=0.6, audio_w=0.4):
    return face_w * face_probs + audio_w * audio_probs


def confidence_gated(face_probs, audio_probs, threshold=0.7, fallback_face_w=0.6):
    fc, ac = face_probs.max(), audio_probs.max()
    if fc >= threshold and fc >= ac:
        return face_probs
    if ac >= threshold and ac > fc:
        return audio_probs
    return weighted_average(face_probs, audio_probs, fallback_face_w, 1 - fallback_face_w)


def fuse(face_probs, audio_probs, strategy="weighted_average"):
    if strategy == "weighted_average":
        probs = weighted_average(face_probs, audio_probs)
    elif strategy == "confidence_gated":
        probs = confidence_gated(face_probs, audio_probs)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    pred = int(np.argmax(probs))
    return pred, EMOTIONS[pred], probs


def demo():
    """Sanity check: load both models and run a dummy forward pass."""
    device = get_device()
    print(f"Device: {device}")
    face = load_face_model(device)
    audio = load_audio_model(device)

    dummy_face = torch.randn(1, 3, 96, 96)
    dummy_audio = torch.randn(1, 48000)  # 3s waveform @ 16 kHz
    fp = predict_face(face, dummy_face, device)
    ap = predict_audio(audio, dummy_audio, device)
    pred, name, probs = fuse(fp, ap, strategy="weighted_average")
    print(f"Face probs:  {fp.round(3)}")
    print(f"Audio probs: {ap.round(3)}")
    print(f"Fused -> {name} ({probs[pred]:.3f})")


if __name__ == "__main__":
    demo()
