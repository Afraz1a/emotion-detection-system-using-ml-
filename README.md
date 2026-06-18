# Emotion & Mood Detection from Face and Voice

A multimodal deep learning system that detects emotions from facial expressions
and voice, then combines both predictions for a more reliable result than either
input alone.

Final-semester ML project · Information Technology University

---

## Overview

Single-modality emotion systems are fragile. A camera fails in poor lighting; a
microphone gets confused by background noise; a forced smile won't match a
tense voice. This project combines both signals to make the system more robust.

The pipeline takes a face image and a voice clip, runs each through a dedicated
model, and fuses the two probability distributions into a final prediction
across **7 emotions**: angry, disgust, fear, happy, sad, surprise, neutral.

```
Face image  →  OpenCV face crop  →  ResNet-18 (fine-tuned)  ─┐
                                                              ├─→  Late fusion  →  Emotion + confidence
Voice clip  →  MFCC + Δ + Δ²      →  2D CNN                  ─┘
```

---

## Results

| Model            | Test Accuracy | F1 (macro) | Inference time |
|------------------|--------------:|-----------:|---------------:|
| Face only        | _TBD_         | _TBD_      | _TBD_          |
| Audio only       | _TBD_         | _TBD_      | _TBD_          |
| **Fused**        | _TBD_         | _TBD_      | _TBD_          |

Target: ≥75% accuracy across the 7 classes, real-time inference (<300ms),
running locally on a laptop with no paid services.

---

## Tech stack

- **PyTorch** — model training, pretrained ResNet-18 backbone
- **OpenCV** — face detection (Haar cascade)
- **Librosa** — audio loading and MFCC feature extraction
- **scikit-learn** — evaluation metrics
- **Streamlit** — interactive demo app
- Python 3.10+

---

## Datasets

| Dataset      | Modality    | Samples  | Source |
|--------------|-------------|---------:|--------|
| FER-2013     | Face images | ~35,000  | [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) |
| RAVDESS      | Audio       | ~1,400   | [Zenodo](https://zenodo.org/record/1188976) |
| CREMA-D      | Audio (opt.)| 7,442    | [GitHub](https://github.com/CheyneyComputerScience/CREMA-D) |

All datasets are publicly available. They are not included in this repo.

### Label mapping

All datasets are unified to a 7-class scheme. RAVDESS "calm" is merged into
"neutral".

| ID | Emotion  | FER-2013 | RAVDESS |
|----|----------|----------|---------|
| 0  | angry    | 0        | 05      |
| 1  | disgust  | 1        | 07      |
| 2  | fear     | 2        | 06      |
| 3  | happy    | 3        | 03      |
| 4  | sad      | 4        | 04      |
| 5  | surprise | 5        | 08      |
| 6  | neutral  | 6        | 01, 02  |

---

## Setup

**1. Clone and install**

```bash
git clone https://github.com/YOUR_USERNAME/emotion_detection.git
cd emotion_detection
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For GPU training, install PyTorch with CUDA support separately:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**2. Download the datasets**

Place files into `data/` like this:

```
data/
├── fer2013.csv
└── ravdess/
    ├── Actor_01/03-01-01-01-01-01-01.wav
    ├── Actor_02/...
    └── ...
```

- **FER-2013** — download `fer2013.csv` from Kaggle and place at `data/fer2013.csv`
- **RAVDESS** — download `Audio_Speech_Actors_01-24.zip` from Zenodo and unzip into `data/ravdess/`

---

## How to run

```bash
# 1. Explore the datasets (saves class-balance plots and sample grids)
python src/explore_fer2013.py
python src/explore_ravdess.py

# 2. Train the face model
python src/train_face.py

# 3. Train the audio model
python src/train_audio.py

# 4. Sanity-check fusion (loads both checkpoints, runs dummy forward pass)
python src/fusion.py

# 5. Launch the demo
streamlit run src/demo.py
```

Trained checkpoints are saved automatically to `checkpoints/`.

---

## Project structure

```
emotion_detection/
├── README.md
├── requirements.txt
├── data/                     # Datasets (gitignored)
├── utils/
│   ├── labels.py             # Unified 7-class scheme + dataset mappings
│   └── helpers.py            # Seeding, device selection, paths
└── src/
    ├── explore_fer2013.py    # EDA: class balance, sample inspection
    ├── explore_ravdess.py    # EDA: waveform + MFCC visualization
    ├── datasets.py           # PyTorch Dataset classes + transforms
    ├── models.py             # FaceModel, AudioModel, FusionMLP
    ├── train_face.py         # Face training loop
    ├── train_audio.py        # Audio training loop
    ├── fusion.py             # Late-fusion strategies
    └── demo.py               # Streamlit demo app
```

---

## Key design choices

- **Late fusion over early fusion.** Face and audio features have very different
  characteristics. Combining at the probability level (after each model is
  independently confident) is more robust and easier to debug than concatenating
  raw features.
- **Speaker-independent RAVDESS split.** Train/val/test split by actor, not by
  clip — otherwise the audio model can cheat by memorizing voices.
- **Class-weighted loss.** FER-2013 has ~10× imbalance between "happy" and
  "disgust". Weights balance gradient contribution.
- **MFCC + Δ + Δ² as 3 channels.** Lets the audio model use a standard 2D CNN,
  mirroring the face pipeline. Cleaner fusion downstream.
- **ResNet-18, not a heavier backbone.** FER-2013 images are 48×48 grayscale
  upscaled to 96×96 — deeper networks would just memorize noise. Also keeps
  inference under the 300ms target.

---

## Limitations

- The system outputs an **estimate, not a definitive label**. It will be wrong sometimes.
- It should not be used for consequential decisions about people (hiring,
  surveillance, mental health screening).
- Training data is mostly acted emotions in clean conditions; real-world subtle
  emotions are harder.
- FER-2013 has known label noise; expect ~25–30% irreducible error from
  mislabeled examples.

---

## References

1. Goodfellow et al., *Challenges in Representation Learning: A Report on Three
   ML Contests*, Neural Networks, 2015 — FER-2013.
2. Livingstone & Russo, *The Ryerson Audio-Visual Database of Emotional Speech
   and Song (RAVDESS)*, PLOS ONE, 2018.
3. Cao et al., *CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset*,
   IEEE Trans. Affective Computing, 2014.

---

## Author

**Afrazia Umer** — BSCS23029
Department of Computer Science 
Information Technology University
