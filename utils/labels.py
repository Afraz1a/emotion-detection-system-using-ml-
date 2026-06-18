"""7-class emotion labels and dataset mappings."""

EMOTIONS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")
NUM_CLASSES = len(EMOTIONS)
LABEL_TO_ID = {name: i for i, name in enumerate(EMOTIONS)}
ID_TO_LABEL = {i: name for i, name in enumerate(EMOTIONS)}

# FER-2013: 0=angry 1=disgust 2=fear 3=happy 4=sad 5=surprise 6=neutral
FER2013_TO_UNIFIED = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}

# RAVDESS: 01=neutral 02=calm 03=happy 04=sad 05=angry 06=fear 07=disgust 08=surprise
# Calm merged into neutral.
RAVDESS_TO_UNIFIED = {
    "01": 6, "02": 6, "03": 3, "04": 4,
    "05": 0, "06": 2, "07": 1, "08": 5,
}

# CREMA-D emotion codes -> unified (no surprise in CREMA-D).
CREMAD_TO_UNIFIED = {
    "ANG": 0, "DIS": 1, "FEA": 2, "HAP": 3, "SAD": 4, "NEU": 6,
}

# AffectNet(-HQ) classes: 0=neutral 1=happy 2=sad 3=surprise 4=fear 5=disgust
# 6=anger 7=contempt. Contempt (7) is dropped (no equivalent in our 7-class scheme).
# This maps the numeric class id; folder-name variants are mapped in AFFECTNET_NAME_TO_UNIFIED.
AFFECTNET_TO_UNIFIED = {
    6: 0,  # anger    -> angry
    5: 1,  # disgust  -> disgust
    4: 2,  # fear     -> fear
    1: 3,  # happy    -> happy
    2: 4,  # sad      -> sad
    3: 5,  # surprise -> surprise
    0: 6,  # neutral  -> neutral
    # 7 = contempt -> dropped
}

# AffectNet-HQ Kaggle versions are usually folder-per-class with these names.
AFFECTNET_NAME_TO_UNIFIED = {
    "anger": 0, "angry": 0,
    "disgust": 1, "contempt": None,
    "fear": 2,
    "happy": 3, "happiness": 3,
    "sad": 4, "sadness": 4,
    "surprise": 5,
    "neutral": 6,
}

# FairFace race categories (7) and gender (2) — used to train the subgroup annotator.
FAIRFACE_RACES = (
    "East Asian", "Indian", "Black", "White",
    "Middle Eastern", "Latino_Hispanic", "Southeast Asian",
)
FAIRFACE_GENDERS = ("Male", "Female")


def parse_ravdess_filename(filename):
    """RAVDESS filenames look like 03-01-05-01-02-01-12.wav"""
    stem = filename.split("/")[-1].split(".")[0]
    parts = stem.split("-")
    if len(parts) < 3:
        return None
    return RAVDESS_TO_UNIFIED.get(parts[2])
