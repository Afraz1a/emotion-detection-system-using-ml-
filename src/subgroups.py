"""Assign demographic subgroup labels for the fairness audit.

Two sources:
  - FACE images (FER+, AffectNet): no demographic labels, so we PREDICT gender
    and race with the FairFace-trained annotator (checkpoints/subgroup_best.pt).
  - AUDIO clips: CREMA-D ships real demographics (VideoDemographics.csv); RAVDESS
    gender is derivable from the actor id (odd=male, even=female).

Predicted face subgroups are noisy (especially on grayscale 48x48 FER images and
for race). They are used only to slice fairness metrics, not as ground truth, and
this limitation is reported alongside the results.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from utils.labels import FAIRFACE_RACES, FAIRFACE_GENDERS
from utils.helpers import DATA_DIR, CHECKPOINT_DIR
from train_subgroup import SubgroupNet


# ---------------- Face: predicted subgroups ----------------

def load_subgroup_annotator(device, ckpt=None):
    ckpt = ckpt or (CHECKPOINT_DIR / "subgroup_best.pt")
    model = SubgroupNet(pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def predict_face_subgroups(annotator, image_tensor, device):
    """image_tensor: (B, 3, H, W) normalized like training.
    Returns (gender_names, race_names) lists of length B."""
    g_logits, r_logits = annotator(image_tensor.to(device))
    g = g_logits.argmax(1).cpu().numpy()
    r = r_logits.argmax(1).cpu().numpy()
    genders = [FAIRFACE_GENDERS[i] for i in g]
    races = [FAIRFACE_RACES[i] for i in r]
    return genders, races


# ---------------- Audio: real / derived subgroups ----------------

_CREMAD_DEMO = None


def _cremad_demographics():
    global _CREMAD_DEMO
    if _CREMAD_DEMO is None:
        df = pd.read_csv(DATA_DIR / "cremad" / "VideoDemographics.csv")
        df["ActorID"] = df["ActorID"].astype(str)
        _CREMAD_DEMO = df.set_index("ActorID")
    return _CREMAD_DEMO


def _age_band(age):
    try:
        age = int(age)
    except (ValueError, TypeError):
        return "unknown"
    if age < 30:
        return "20s"
    if age < 45:
        return "30-44"
    if age < 60:
        return "45-59"
    return "60+"


def subgroup_for_audio_path(path):
    """Return {'gender','race','age'} for a RAVDESS or CREMA-D clip path."""
    stem = Path(path).stem
    if "-" in stem and len(stem.split("-")) == 7:
        # RAVDESS: actor id is the last field; odd=male, even=female. Race unknown.
        actor = int(stem.split("-")[-1])
        return {"gender": "Male" if actor % 2 == 1 else "Female",
                "race": "unknown", "age": "unknown"}
    # CREMA-D: ACTORID_SENTENCE_EMOTION_LEVEL
    actor = stem.split("_")[0]
    demo = _cremad_demographics()
    if actor in demo.index:
        row = demo.loc[actor]
        return {"gender": str(row["Sex"]), "race": str(row["Race"]),
                "age": _age_band(row["Age"])}
    return {"gender": "unknown", "race": "unknown", "age": "unknown"}


def collect_audio_paths(concat_ds):
    """Flatten ordered file paths from a combined-audio ConcatDataset / Subset.

    Order matches a DataLoader(shuffle=False) over the same dataset.
    """
    paths = []
    datasets = getattr(concat_ds, "datasets", [concat_ds])
    for ds in datasets:
        base = getattr(ds, "ds", ds)          # unwrap _Subset
        idxs = getattr(ds, "indices", range(len(base)))
        for i in idxs:
            paths.append(base.paths[i])
    return paths
