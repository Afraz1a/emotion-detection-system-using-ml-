"""Face CNN, audio model (wav2vec2), and fusion module."""

import torch
import torch.nn as nn
from torchvision import models


def build_face_model(num_classes=7, dropout=0.4, pretrained=True):
    """ResNet-34 fine-tuned for emotion. Single source of truth for the face
    architecture, used by both training and checkpoint loading so state_dict
    keys always match."""
    weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = models.resnet34(weights=weights)
    in_feat = backbone.fc.in_features
    backbone.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_feat, num_classes),
    )
    return backbone


class FaceModel(nn.Module):
    """[Legacy] Wrapper kept for backward compatibility. New code should use
    build_face_model(), whose state_dict keys match the saved checkpoints."""

    def __init__(self, num_classes=7, pretrained=True, dropout=0.4):
        super().__init__()
        self.backbone = build_face_model(num_classes, dropout, pretrained)

    def forward(self, x):
        return self.backbone(x)


class AudioCNN(nn.Module):
    """
    [Legacy] Deeper 2D CNN over log-mel + delta + delta-delta. Kept for reference.
    Superseded by the wav2vec2-based AudioModel because a from-scratch CNN
    overfits speaker identity on the ~115-actor combined dataset.
    """

    def __init__(self, num_classes=7, dropout=0.3, in_ch=3):
        super().__init__()

        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(in_ch, 32), block(32, 64), block(64, 128), block(128, 256),
        )
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))
        self.attn = nn.Sequential(nn.Linear(256, 64), nn.Tanh(), nn.Linear(64, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.freq_pool(x)
        x = x.squeeze(2).transpose(1, 2)
        w = torch.softmax(self.attn(x), dim=1)
        x = (x * w).sum(dim=1)
        return self.classifier(x)


class AudioModel(nn.Module):
    """
    wav2vec2-based emotion classifier.

    A frozen wav2vec2 backbone (pretrained on 960h LibriSpeech) extracts
    speaker-invariant speech features; only the attention-pooled head trains.
    Input: raw waveform (B, T) at 16 kHz. This generalizes across unseen
    speakers far better than a from-scratch CNN.
    """

    def __init__(self, num_classes=7, dropout=0.3, unfreeze_last_n=4):
        super().__init__()
        from torchaudio.pipelines import WAV2VEC2_BASE
        self.backbone = WAV2VEC2_BASE.get_model()
        self.feat_dim = 768
        self.unfreeze_last_n = unfreeze_last_n

        # Freeze everything, then selectively unfreeze the top N transformer layers.
        # The conv feature extractor stays frozen (low-level, speaker-agnostic).
        for p in self.backbone.parameters():
            p.requires_grad = False
        if unfreeze_last_n > 0:
            layers = self.backbone.encoder.transformer.layers
            for layer in layers[-unfreeze_last_n:]:
                for p in layer.parameters():
                    p.requires_grad = True

        self.attn = nn.Sequential(nn.Linear(self.feat_dim, 128), nn.Tanh(),
                                  nn.Linear(128, 1))
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Dropout(dropout),
            nn.Linear(self.feat_dim, 256), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def backbone_parameters(self):
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def head_parameters(self):
        return list(self.attn.parameters()) + list(self.classifier.parameters())

    def _features(self, wav):
        # wav: (B, T) at 16 kHz. Gradients flow only to unfrozen layers.
        feats, _ = self.backbone.extract_features(wav)
        return feats[-1]  # (B, T', 768) — last transformer layer

    def forward(self, wav):
        x = self._features(wav)                  # (B, T', 768)
        w = torch.softmax(self.attn(x), dim=1)   # (B, T', 1)
        x = (x * w).sum(dim=1)                    # (B, 768) attention pooled
        return self.classifier(x)


class FusionMLP(nn.Module):
    """Late fusion: concatenate two 7-dim probability vectors -> small MLP."""

    def __init__(self, num_classes=7, hidden_dim=32, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_classes * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, face_probs, audio_probs):
        return self.net(torch.cat([face_probs, audio_probs], dim=1))