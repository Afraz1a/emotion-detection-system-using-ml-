"""PyTorch Dataset classes for FER-2013, FER+, and RAVDESS."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from utils.labels import parse_ravdess_filename, AFFECTNET_NAME_TO_UNIFIED
from utils.helpers import DATA_DIR


# ---------------- FER-2013 (original, kept for comparison) ----------------

class FER2013Dataset(Dataset):
    def __init__(self, csv_path=None, split="Training", image_size=96, augment=False):
        csv_path = csv_path or (DATA_DIR / "fer2013.csv")
        df = pd.read_csv(csv_path)
        self.df = df[df["Usage"] == split].reset_index(drop=True)
        self.image_size = image_size
        self.tf = _build_transforms(image_size, augment)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pixels = np.array(row["pixels"].split(), dtype=np.uint8).reshape(48, 48)
        img = Image.fromarray(pixels).convert("RGB")
        return self.tf(img), int(row["emotion"])


# ---------------- FER+ (improved labels) ----------------

# FER+ has 8 emotions (adds "contempt"); we map to our 7-class scheme.
# Contempt -> dropped (no equivalent in our scheme).
# unknown, NF -> dropped (junk rows).
FERPLUS_TO_UNIFIED = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "happiness": 3,
    "sadness": 4,
    "surprise": 5,
    "neutral": 6,
}
FERPLUS_EMOTION_COLS = list(FERPLUS_TO_UNIFIED.keys())


class FERPlusDataset(Dataset):
    """
    Combines FER-2013 images with FER+ improved labels.

    Filtering rules:
      - Drop rows where 'NF' (not a face) is the majority vote
      - Drop rows where 'unknown' is the majority vote
      - Drop rows where 'contempt' is the majority vote (no equivalent in our 7-class scheme)
      - Drop rows where no emotion vote exists in our 7-class scheme

    Uses HARD labels (majority vote). Soft-label training comes later if needed.
    """

    def __init__(self, fer_csv=None, ferplus_csv=None, split="Training",
                 image_size=96, augment=False):
        fer_csv = fer_csv or (DATA_DIR / "fer2013.csv")
        ferplus_csv = ferplus_csv or (DATA_DIR / "fer2013new.csv")

        fer_df = pd.read_csv(fer_csv)
        plus_df = pd.read_csv(ferplus_csv)

        if len(fer_df) != len(plus_df):
            raise ValueError(
                f"FER ({len(fer_df)}) and FER+ ({len(plus_df)}) row counts don't match. "
                "Make sure both CSVs are intact."
            )

        # Combine side-by-side: FER pixels + FER+ vote columns
        plus_df = plus_df.reset_index(drop=True)
        fer_df = fer_df.reset_index(drop=True)
        plus_df["pixels"] = fer_df["pixels"]
        plus_df["fer_usage"] = fer_df["Usage"]

        # Use FER+ Usage column if it exists, otherwise fall back to FER's
        usage_col = "Usage" if "Usage" in plus_df.columns else "fer_usage"
        df = plus_df[plus_df[usage_col] == split].copy()

        # Determine majority-vote label
        # If NF > 0, throw it out
        df = df[df["NF"] < 1]
        # If unknown is majority, throw it out
        df = df[df["unknown"] < df[FERPLUS_EMOTION_COLS].max(axis=1)]
        # If contempt is majority, throw it out
        df = df[df["contempt"] < df[FERPLUS_EMOTION_COLS].max(axis=1)]

        # Take argmax over our 7 emotion columns
        votes = df[FERPLUS_EMOTION_COLS].values  # (N, 7)
        majority_col = votes.argmax(axis=1)
        df["unified_label"] = [FERPLUS_TO_UNIFIED[FERPLUS_EMOTION_COLS[i]]
                               for i in majority_col]

        # Drop rows where the majority emotion got 0 votes (edge case)
        max_votes = votes.max(axis=1)
        df = df[max_votes > 0].reset_index(drop=True)

        # Hard resample for training: bring EVERY class to the same target count
        # (the median). Augmentation makes repeated minority rows effectively distinct.
        if split == "Training":
            df = self._resample(df)

        self.df = df
        self.image_size = image_size
        self.tf = _build_transforms(image_size, augment)

        print(f"FERPlusDataset[{split}]: {len(self.df)} samples after filtering")

    @staticmethod
    def _resample(df, seed=42):
        """Resample every class to a single target count -> ratio 1.0x."""
        rng = np.random.RandomState(seed)
        counts = df["unified_label"].value_counts()
        target = int(np.median(counts.values))
        parts = []
        for label, group in df.groupby("unified_label"):
            n = len(group)
            if n >= target:
                parts.append(group.sample(target, random_state=rng.randint(0, 9999)))
            else:
                # repeat minority rows until we reach target, then trim
                repeats = int(np.ceil(target / n))
                repeated = pd.concat([group] * repeats, ignore_index=True)
                parts.append(repeated.sample(target, random_state=rng.randint(0, 9999)))
        return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pixels = np.array(row["pixels"].split(), dtype=np.uint8).reshape(48, 48)
        img = Image.fromarray(pixels).convert("RGB")
        return self.tf(img), int(row["unified_label"])


class AffectNetDataset(Dataset):
    """
    AffectNet (Kaggle), folder-per-emotion with a built-in Train/Test split:
        data/AffectNet/Train/<emotion>/*.jpg
        data/AffectNet/Test/<emotion>/*.jpg

    Labels come from the FOLDER name (original AffectNet manual annotation).
    The bundled labels.csv is a model re-annotation (only ~63% agreement) and is
    ignored. 'contempt' folders are dropped (no equivalent in our 7-class scheme).
    Folder casing is inconsistent in Test (e.g. 'Anger') so names are lowercased.

    NOTE: license caveat — this AffectNet redistribution is not formally
    open-access. Used here for prototyping only.
    """

    def __init__(self, root=None, image_size=96, augment=False, split="Train"):
        base = Path(root) if root else (DATA_DIR / "AffectNet")
        # Accept "Train"/"Test" (case-insensitive), defaulting to the real folder names.
        split_dir = "Train" if split.lower() == "train" else "Test"
        root = base / split_dir
        if not root.exists():
            raise FileNotFoundError(
                f"AffectNet split not found at {root}. Expected data/AffectNet/Train "
                "and data/AffectNet/Test (folder-per-emotion)."
            )

        samples = []  # (path, unified_label)
        for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            label = AFFECTNET_NAME_TO_UNIFIED.get(class_dir.name.lower().strip())
            if label is None:
                continue  # contempt or unknown folder
            for img in class_dir.rglob("*"):
                if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    samples.append((str(img), label))

        if not samples:
            raise RuntimeError(
                f"No images found under {root}. Check the folder structure "
                "(expected one subfolder per emotion)."
            )

        self.samples = samples
        self.labels = [lab for _, lab in samples]
        self.image_size = image_size
        self.tf = _build_transforms(image_size, augment)
        print(f"AffectNetDataset[{split_dir}]: {len(self.samples)} images "
              f"(contempt dropped)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.tf(img), int(label)


def _build_transforms(image_size, augment):
    if augment:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ---------------- FairFace (subgroup annotator training) ----------------

from utils.labels import FAIRFACE_RACES, FAIRFACE_GENDERS

_FF_GENDER_TO_ID = {g: i for i, g in enumerate(FAIRFACE_GENDERS)}
_FF_RACE_TO_ID = {r: i for i, r in enumerate(FAIRFACE_RACES)}


class FairFaceDataset(Dataset):
    """
    FairFace, used to train a gender+race subgroup annotator (NOT emotion).
    Returns (image_tensor, gender_id, race_id).
    Layout: data/FairFace/{train,val}/*.jpg  +  {train,val}_labels.csv
            CSV columns: file, age, gender, race, service_test
    The 'file' column already includes the train/ or val/ prefix.
    """

    def __init__(self, root=None, split="train", image_size=96, augment=False):
        root = Path(root) if root else (DATA_DIR / "FairFace")
        csv = root / f"{split}_labels.csv"
        if not csv.exists():
            raise FileNotFoundError(f"FairFace labels not found at {csv}.")
        df = pd.read_csv(csv)
        df = df[df["gender"].isin(_FF_GENDER_TO_ID) & df["race"].isin(_FF_RACE_TO_ID)]
        self.root = root
        self.files = df["file"].tolist()
        self.gender = [_FF_GENDER_TO_ID[g] for g in df["gender"]]
        self.race = [_FF_RACE_TO_ID[r] for r in df["race"]]
        self.tf = _build_transforms(image_size, augment)
        print(f"FairFaceDataset[{split}]: {len(self.files)} images")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.root / self.files[idx]).convert("RGB")
        return self.tf(img), int(self.gender[idx]), int(self.race[idx])


# ---------------- Combined face dataset (FER+ + AffectNet) ----------------

from torch.utils.data import ConcatDataset as _ConcatDataset


def get_combined_face_splits(image_size=96, seed=42):
    """
    Combine FER+ and AffectNet for face training.

    Returns (train_ds, val_ds, test_ds). The training set exposes a flat
    `.labels` list (for WeightedRandomSampler). FER+ contributes its own
    Training/PublicTest/PrivateTest splits; AffectNet contributes Train -> train
    and Test -> split evenly into val/test so both stay class-diverse.
    """
    fer_train = FERPlusDataset(split="Training",    image_size=image_size, augment=True)
    fer_val   = FERPlusDataset(split="PublicTest",  image_size=image_size, augment=False)
    fer_test  = FERPlusDataset(split="PrivateTest", image_size=image_size, augment=False)

    aff_train = AffectNetDataset(split="Train", image_size=image_size, augment=True)
    aff_eval  = AffectNetDataset(split="Test",  image_size=image_size, augment=False)

    # Split AffectNet Test 50/50 into val and test (deterministic)
    rng = np.random.RandomState(seed)
    idx = np.arange(len(aff_eval))
    rng.shuffle(idx)
    half = len(idx) // 2
    val_idx, test_idx = set(idx[:half].tolist()), set(idx[half:].tolist())
    aff_val  = _Subset(aff_eval, sorted(val_idx))
    aff_test = _Subset(aff_eval, sorted(test_idx))

    train_ds = _ConcatDataset([fer_train, aff_train])
    val_ds   = _ConcatDataset([fer_val,   aff_val])
    test_ds  = _ConcatDataset([fer_test,  aff_test])

    # Flat training labels for the sampler
    train_ds.labels = (
        fer_train.df["unified_label"].tolist() + list(aff_train.labels)
    )
    return train_ds, val_ds, test_ds


class _Subset(Dataset):
    """Lightweight index subset that preserves a `.labels` list."""
    def __init__(self, ds, indices):
        self.ds = ds
        self.indices = list(indices)
        self.labels = [ds.labels[i] for i in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.ds[self.indices[i]]


# ---------------- Audio feature extraction (shared) ----------------

def extract_logmel(y, sr, n_mels=64, n_fft=2048, hop_length=512):
    """Log-mel spectrogram + delta + delta-delta, per-channel standardized.

    Log-mel preserves spectral locality better than MFCC for CNNs, which is why
    we switched from 40-MFCC. Returns (3, n_mels, T) float32.
    """
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                         n_fft=n_fft, hop_length=hop_length)
    logmel = librosa.power_to_db(mel + 1e-9)
    d1 = librosa.feature.delta(logmel)
    d2 = librosa.feature.delta(logmel, order=2)
    feat = np.stack([logmel, d1, d2], axis=0).astype(np.float32)
    mean = feat.mean(axis=(1, 2), keepdims=True)
    std = feat.std(axis=(1, 2), keepdims=True) + 1e-6
    return (feat - mean) / std


# ---------------- RAVDESS ----------------

class RAVDESSDataset(Dataset):
    # 16 kHz: wav2vec2 expects this sample rate.
    def __init__(self, file_paths, sample_rate=16000, duration=3.0,
                 n_mfcc=40, augment=False):
        self.paths = list(file_paths)
        self.labels = [parse_ravdess_filename(p) for p in self.paths]
        kept = [(p, l) for p, l in zip(self.paths, self.labels) if l is not None]
        self.paths = [p for p, _ in kept]
        self.labels = [l for _, l in kept]
        self.sr = sample_rate
        self.duration = duration
        self.target_len = int(sample_rate * duration)
        self.n_mfcc = n_mfcc
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def _pad_or_crop(self, y):
        if len(y) >= self.target_len:
            start = (len(y) - self.target_len) // 2
            return y[start:start + self.target_len]
        return np.pad(y, (0, self.target_len - len(y)))

    def _augment(self, y):
        if np.random.random() < 0.3:
            y = y + np.random.normal(0, 0.005, y.shape)
        if np.random.random() < 0.3:
            shift = int(np.random.uniform(-0.1, 0.1) * len(y))
            y = np.roll(y, shift)
        # Pitch shift (+/-2 semitones) — especially helpful for surprise oversampling
        if np.random.random() < 0.4:
            n_steps = np.random.uniform(-2, 2)
            y = librosa.effects.pitch_shift(y, sr=self.sr, n_steps=n_steps)
        # Time stretch (0.85–1.15x)
        if np.random.random() < 0.4:
            rate = np.random.uniform(0.85, 1.15)
            y = librosa.effects.time_stretch(y, rate=rate)
        return y

    def __getitem__(self, idx):
        y, _ = librosa.load(self.paths[idx], sr=self.sr)
        y = self._pad_or_crop(y)
        if self.augment:
            y = self._augment(y)
        y = self._pad_or_crop(y)  # re-crop after time stretch may change length
        return torch.from_numpy(y.astype(np.float32)), int(self.labels[idx])


def get_ravdess_splits(ravdess_dir=None, seed=42):
    """Split RAVDESS by actor to avoid speaker leakage."""
    ravdess_dir = Path(ravdess_dir) if ravdess_dir else (DATA_DIR / "ravdess")
    all_files = sorted(ravdess_dir.rglob("*.wav"))

    rng = np.random.RandomState(seed)
    actors = list(range(1, 25))
    rng.shuffle(actors)
    train_actors = set(actors[:18])
    val_actors = set(actors[18:21])
    test_actors = set(actors[21:24])

    def actor_of(p):
        return int(p.stem.split("-")[-1])

    train = [str(p) for p in all_files if actor_of(p) in train_actors]
    val = [str(p) for p in all_files if actor_of(p) in val_actors]
    test = [str(p) for p in all_files if actor_of(p) in test_actors]
    return train, val, test


# ---------------- CREMA-D ----------------

# Filename format: ACTORID_SENTENCE_EMOTION_LEVEL.wav
# Emotions: ANG, DIS, FEA, HAP, NEU, SAD  (no SURPRISE)
CREMAD_TO_UNIFIED = {
    "ANG": 0, "DIS": 1, "FEA": 2, "HAP": 3, "SAD": 4, "NEU": 6,
}


class CREMADDataset(Dataset):
    # 16 kHz: wav2vec2 expects this sample rate.
    def __init__(self, file_paths, sample_rate=16000, duration=3.0,
                 n_mfcc=40, augment=False):
        rows = []
        for p in file_paths:
            parts = Path(p).stem.split("_")
            if len(parts) < 3:
                continue
            label = CREMAD_TO_UNIFIED.get(parts[2])
            if label is None:
                continue
            rows.append((str(p), label))
        self.paths = [r[0] for r in rows]
        self.labels = [r[1] for r in rows]
        self.sr = sample_rate
        self.target_len = int(sample_rate * duration)
        self.n_mfcc = n_mfcc
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def _pad_or_crop(self, y):
        if len(y) >= self.target_len:
            start = (len(y) - self.target_len) // 2
            return y[start:start + self.target_len]
        return np.pad(y, (0, self.target_len - len(y)))

    def _augment(self, y):
        if np.random.random() < 0.3:
            y = y + np.random.normal(0, 0.005, y.shape)
        if np.random.random() < 0.3:
            shift = int(np.random.uniform(-0.1, 0.1) * len(y))
            y = np.roll(y, shift)
        return y

    def __getitem__(self, idx):
        y, _ = librosa.load(self.paths[idx], sr=self.sr)
        y = self._pad_or_crop(y)
        if self.augment:
            y = self._augment(y)
        return torch.from_numpy(y.astype(np.float32)), int(self.labels[idx])


def get_cremad_splits(cremad_dir=None, seed=42):
    """Split CREMA-D by actor to avoid speaker leakage (80/10/10)."""
    cremad_dir = Path(cremad_dir) if cremad_dir else (DATA_DIR / "cremad" / "AudioWAV")
    all_files = sorted(cremad_dir.rglob("*.wav"))

    # Actor ID is the first part of the filename: e.g. 1001_DFA_ANG_XX.wav
    def actor_of(p):
        return p.stem.split("_")[0]

    actors = sorted({actor_of(p) for p in all_files})
    rng = np.random.RandomState(seed)
    rng.shuffle(actors)
    n = len(actors)
    train_actors = set(actors[:int(n * 0.8)])
    val_actors   = set(actors[int(n * 0.8):int(n * 0.9)])
    test_actors  = set(actors[int(n * 0.9):])

    train = [str(p) for p in all_files if actor_of(p) in train_actors]
    val   = [str(p) for p in all_files if actor_of(p) in val_actors]
    test  = [str(p) for p in all_files if actor_of(p) in test_actors]
    return train, val, test


# ---------------- Combined audio dataset (RAVDESS + CREMA-D) ----------------

from torch.utils.data import ConcatDataset


def get_combined_audio_splits(seed=42, surprise_target=None):
    """
    Returns (train_ds, val_ds, test_ds) combining RAVDESS and CREMA-D.
    Surprise clips (RAVDESS-only) are repeated with augmentation until
    surprise_target clips are in the training set (default: match median class size).
    """
    r_train, r_val, r_test = get_ravdess_splits(seed=seed)
    c_train, c_val, c_test = get_cremad_splits(seed=seed)

    # Identify surprise paths in the training split (RAVDESS label 5 = surprise)
    SURPRISE_ID = 5
    surprise_train = [p for p in r_train if parse_ravdess_filename(p) == SURPRISE_ID]
    non_surprise_ravdess_train = [p for p in r_train if parse_ravdess_filename(p) != SURPRISE_ID]

    # Determine target count: median of non-surprise classes in combined train
    base_ravdess = RAVDESSDataset(non_surprise_ravdess_train, augment=True)
    base_cremad  = CREMADDataset(c_train, augment=True)
    all_labels_except_surprise = base_ravdess.labels + base_cremad.labels
    counts_except = np.bincount(all_labels_except_surprise, minlength=7)
    if surprise_target is None:
        surprise_target = int(np.median(counts_except[counts_except > 0]))

    # Repeat surprise paths (with augmentation) until we reach surprise_target
    rng = np.random.RandomState(seed)
    repeated_surprise = []
    while len(repeated_surprise) < surprise_target:
        repeated_surprise += surprise_train
    rng.shuffle(repeated_surprise)
    repeated_surprise = repeated_surprise[:surprise_target]

    surprise_ds = RAVDESSDataset(repeated_surprise, augment=True)
    print(f"Surprise clips in training: {len(surprise_ds)} (target={surprise_target})")

    train_ds = ConcatDataset([base_ravdess, base_cremad, surprise_ds])
    val_ds = ConcatDataset([
        RAVDESSDataset(r_val,  augment=False),
        CREMADDataset(c_val,   augment=False),
    ])
    test_ds = ConcatDataset([
        RAVDESSDataset(r_test, augment=False),
        CREMADDataset(c_test,  augment=False),
    ])

    # Expose flat label lists for WeightedRandomSampler
    train_ds.labels = [
        label for ds in train_ds.datasets for label in ds.labels
    ]

    return train_ds, val_ds, test_ds