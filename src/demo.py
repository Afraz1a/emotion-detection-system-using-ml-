"""Stage 5: Streamlit demo (multi-tab).

Run with:  python -m streamlit run src/demo.py

Tabs:
  - Live Detection : webcam snapshot, image/video/voice upload, fused prediction
                     + per-inference latency.
  - Fairness       : per-subgroup accuracy + parity/odds gaps from the audit.
  - About & Ethics : usage policy and limitations.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import cv2
import librosa
import torch
import streamlit as st
from PIL import Image
from torchvision import transforms

from utils.labels import EMOTIONS
from utils.helpers import get_device, PROJECT_ROOT
from fusion import load_face_model, load_audio_model, predict_face, predict_audio, fuse

try:
    import av
    from streamlit_webrtc import webrtc_streamer, WebRtcMode
    _WEBRTC_OK = True
except Exception:
    _WEBRTC_OK = False


DEVICE = get_device()

FACE_TF = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


@st.cache_resource
def load_models():
    return load_face_model(DEVICE), load_audio_model(DEVICE)


def detect_and_crop_face(img_bgr):
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    crop = img_bgr[y:y+h, x:x+w]
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (x, y, w, h)


def prepare_face_tensor(face_rgb):
    return FACE_TF(Image.fromarray(face_rgb)).unsqueeze(0)


def prepare_audio_tensor(y, sr, target_sr=16000, duration=3.0):
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    target_len = int(sr * duration)
    if len(y) >= target_len:
        start = (len(y) - target_len) // 2
        y = y[start:start + target_len]
    else:
        y = np.pad(y, (0, target_len - len(y)))
    return torch.from_numpy(y.astype(np.float32)).unsqueeze(0)


def load_audio_any(uploaded_file, target_sr=16000):
    """Load uploaded audio robustly. Tries soundfile/librosa first (wav/flac/ogg),
    then falls back to PyAV (ffmpeg) for m4a/mp3/aac that libsndfile can't open."""
    try:
        uploaded_file.seek(0)
        return librosa.load(uploaded_file, sr=None)
    except Exception:
        pass
    if not _WEBRTC_OK:  # av unavailable
        raise RuntimeError(
            "This audio format needs ffmpeg/PyAV. Please upload a .wav/.flac/.ogg "
            "file, or install with: pip install av"
        )
    import io
    uploaded_file.seek(0)
    container = av.open(io.BytesIO(uploaded_file.read()))
    stream = container.streams.audio[0]
    resampler = av.AudioResampler(format="flt", layout="mono", rate=target_sr)
    chunks = []
    for frame in container.decode(stream):
        for rf in resampler.resample(frame):
            chunks.append(rf.to_ndarray().flatten())
    container.close()
    if not chunks:
        raise RuntimeError("Could not decode the uploaded audio file.")
    return np.concatenate(chunks).astype(np.float32), target_sr


def robust_audio_probs(audio_model, y, sr, target_sr=16000,
                       window=3.0, hop=1.5, top_db=30):
    """Improved voice pipeline:
      - resample to 16 kHz
      - trim leading/trailing silence
      - peak-normalize
      - for clips longer than `window`, predict over overlapping windows and
        average the probabilities (more robust than a single center crop).
    Returns a 7-dim probability vector.
    """
    y = np.asarray(y, dtype=np.float32)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    # trim silence (fall back to original if trimming removes everything)
    trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    if len(trimmed) > sr * 0.3:
        y = trimmed
    # peak-normalize
    peak = float(np.max(np.abs(y))) + 1e-9
    y = y / peak * 0.95

    win = int(sr * window)
    hop_n = int(sr * hop)
    if len(y) <= win:
        t = prepare_audio_tensor(y, sr, target_sr, window)
        return predict_audio(audio_model, t, DEVICE)

    # sliding windows over the whole clip, average probs
    probs = []
    for start in range(0, len(y) - win + 1, hop_n):
        seg = y[start:start + win]
        t = torch.from_numpy(seg.astype(np.float32)).unsqueeze(0)
        probs.append(predict_audio(audio_model, t, DEVICE))
    return np.mean(probs, axis=0)


def face_probs_from_rgb(face_model, img_rgb):
    """Detect + crop + predict on an RGB image. Returns (probs, crop, ms) or (None, None, ms)."""
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    result = detect_and_crop_face(bgr)
    if result is None:
        return None, None, 0.0
    face_rgb, _ = result
    t0 = time.perf_counter()
    probs = predict_face(face_model, prepare_face_tensor(face_rgb), DEVICE)
    return probs, face_rgb, (time.perf_counter() - t0) * 1000


def sample_video_frames(path, n=8):
    """Evenly sample n frames (RGB) from a video file."""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        cap.release()
        return []
    idxs = np.linspace(0, total - 1, min(n, total)).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def make_live_callbacks(face_model, audio_model, every=5, audio_window=3.0):
    """Build (video_cb, audio_cb) for the live stream.

    Video: detect face every `every` frames, fuse with the latest audio
    prediction, overlay a box + label.
    Audio: buffer a rolling `audio_window` seconds of mic audio (resampled to
    16 kHz mono), run the audio model periodically, share probs with the video
    callback for real-time fusion.
    """
    import threading

    lock = threading.Lock()
    shared = {"audio_probs": None}
    vstate = {"count": 0, "label": "", "box": None}
    abuf = []                       # rolling float32 samples @ 16 kHz
    target_len = int(16000 * audio_window)
    actr = {"n": 0}
    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)

    def video_cb(frame):
        img = frame.to_ndarray(format="bgr24")
        vstate["count"] += 1
        if vstate["count"] % every == 0:
            result = detect_and_crop_face(img)
            face_probs = None
            if result is not None:
                face_rgb, box = result
                face_probs = predict_face(face_model, prepare_face_tensor(face_rgb), DEVICE)
                vstate["box"] = box
            else:
                vstate["box"] = None
            with lock:
                aprobs = shared["audio_probs"]
            if face_probs is not None and aprobs is not None:
                _, name, probs = fuse(face_probs, aprobs, strategy="confidence_gated")
                p = int(np.argmax(probs))
                vstate["label"] = f"{name} {probs[p]:.0%} (face+voice)"
            elif face_probs is not None:
                p = int(np.argmax(face_probs))
                vstate["label"] = f"{EMOTIONS[p]} {face_probs[p]:.0%} (face)"
        if vstate["box"] is not None:
            x, y, w, h = vstate["box"]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, vstate["label"], (x, max(20, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def audio_cb(frames):
        for frame in frames:
            for rf in resampler.resample(frame):
                arr = rf.to_ndarray().flatten().astype(np.float32) / 32768.0
                abuf.extend(arr.tolist())
        if len(abuf) > target_len:
            del abuf[:len(abuf) - target_len]
        actr["n"] += 1
        # run the audio model roughly once per second of incoming frames
        if len(abuf) >= target_len and actr["n"] % 10 == 0:
            wav = np.asarray(abuf[-target_len:], dtype=np.float32)
            probs = predict_audio(audio_model, torch.from_numpy(wav).unsqueeze(0), DEVICE)
            with lock:
                shared["audio_probs"] = probs
        return frames

    return video_cb, audio_cb


def face_probs_from_video(face_model, uploaded_file, n_frames=12):
    """Sample frames from an uploaded video, average face probs. Returns
    (probs|None, n_faces, ms)."""
    tmp = PROJECT_ROOT / "_tmp_demo_video"
    tmp.write_bytes(uploaded_file.getbuffer())
    frames = sample_video_frames(tmp, n=n_frames)
    tmp.unlink(missing_ok=True)
    collected = []
    t0 = time.perf_counter()
    for fr in frames:
        p, _, _ = face_probs_from_rgb(face_model, fr)
        if p is not None:
            collected.append(p)
    if not collected:
        return None, 0, 0.0
    return np.mean(collected, axis=0), len(collected), (time.perf_counter() - t0) * 1000


def show_result(face_probs, audio_probs, strategy, latency_ms):
    if face_probs is not None and audio_probs is not None:
        pred, name, probs = fuse(face_probs, audio_probs, strategy=strategy)
        st.success(f"Fused prediction: **{name}**  (confidence {probs[pred]:.1%})")
        st.bar_chart({EMOTIONS[i]: float(probs[i]) for i in range(7)})
    elif face_probs is not None:
        p = int(np.argmax(face_probs))
        st.info(f"Face-only: **{EMOTIONS[p]}**  ({face_probs[p]:.1%})")
        st.bar_chart({EMOTIONS[i]: float(face_probs[i]) for i in range(7)})
    elif audio_probs is not None:
        p = int(np.argmax(audio_probs))
        st.info(f"Voice-only: **{EMOTIONS[p]}**  ({audio_probs[p]:.1%})")
        st.bar_chart({EMOTIONS[i]: float(audio_probs[i]) for i in range(7)})
    if latency_ms:
        ok = latency_ms < 300
        st.caption(f"⏱ Inference latency: {latency_ms:.0f} ms "
                   f"({'✅ under' if ok else '⚠️ over'} 300 ms real-time target)")


# ---------------- Tabs ----------------

def tab_live(face_model, audio_model):
    st.subheader("Live Detection")
    strategy = st.selectbox("Fusion strategy", ["confidence_gated", "weighted_average"])

    options = ["Webcam snapshot"]
    if _WEBRTC_OK:
        options.insert(0, "Live video")
    src = st.radio("Mode", options, horizontal=True)
    face_probs, latency = None, 0.0

    if src == "Live video":
        use_mic = st.checkbox("Use microphone (live face + voice fusion)", value=True)
        st.caption("Continuous webcam emotion detection. Allow camera"
                   + (" and microphone" if use_mic else "")
                   + " access, then watch the live overlay. "
                   + ("Voice updates every ~3s; the label shows fused "
                      "face+voice when speech is detected. Use headphones to "
                      "avoid echo." if use_mic else ""))
        video_cb, audio_cb = make_live_callbacks(face_model, audio_model)
        webrtc_streamer(
            key="live-emotion",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=video_cb,
            queued_audio_frames_callback=audio_cb if use_mic else None,
            media_stream_constraints={"video": True, "audio": bool(use_mic)},
            async_processing=True,
        )
        return

    shot = st.camera_input("Take a photo")
    if shot:
        img = np.array(Image.open(shot).convert("RGB"))
        face_probs, crop, latency = face_probs_from_rgb(face_model, img)
        if crop is None:
            st.warning("No face detected — try centering your face.")
        else:
            st.image(crop, caption="Detected face", width=180)
            show_result(face_probs, None, strategy, latency)


def tab_upload(face_model, audio_model):
    st.subheader("Upload & Analyze")
    st.caption("Upload any combination of image, video, and audio. Each is "
               "analyzed separately; when a face input and audio are both "
               "present, the fused prediction is also shown.")
    strategy = st.selectbox("Fusion strategy", ["confidence_gated", "weighted_average"],
                            key="upload_strategy")

    face_probs, audio_probs, latency = None, None, 0.0

    img_col, vid_col = st.columns(2)
    with img_col:
        st.markdown("**🖼 Image**")
        img_f = st.file_uploader("Photo / image", type=["jpg", "jpeg", "png"],
                                 key="up_img")
    with vid_col:
        st.markdown("**🎞 Video**")
        vid_f = st.file_uploader("Video clip", type=["mp4", "mov", "avi", "mkv"],
                                 key="up_vid")

    st.markdown("**🎙 Audio**")
    aud_f = st.file_uploader("Voice clip", type=["wav", "mp3", "ogg", "flac", "m4a"],
                             key="up_aud")

    # ----- Image -----
    if img_f:
        st.divider()
        st.markdown("### Image result")
        img = np.array(Image.open(img_f).convert("RGB"))
        fp, crop, ms = face_probs_from_rgb(face_model, img)
        if crop is None:
            st.warning("No face detected in image.")
        else:
            st.image(crop, caption="Detected face", width=180)
            face_probs = fp; latency += ms
            show_result(fp, None, strategy, ms)

    # ----- Video (overrides image as the face source if both given) -----
    if vid_f:
        st.divider()
        st.markdown("### Video result")
        st.video(vid_f)
        fp, n, ms = face_probs_from_video(face_model, vid_f)
        if fp is None:
            st.warning("No faces detected in sampled video frames.")
        else:
            st.caption(f"Averaged {n} face frames.")
            face_probs = fp; latency += ms
            show_result(fp, None, strategy, ms)

    # ----- Audio (robust pipeline) -----
    if aud_f:
        st.divider()
        st.markdown("### Audio result")
        st.audio(aud_f)
        try:
            y, sr = load_audio_any(aud_f)
        except Exception as e:
            st.error(f"Could not read audio: {e}")
            y, sr = None, None
        if y is not None:
            t0 = time.perf_counter()
            audio_probs = robust_audio_probs(audio_model, y, sr)
            ms = (time.perf_counter() - t0) * 1000
            latency += ms
            show_result(None, audio_probs, strategy, ms)

    # ----- Fused (if a face source and audio are both present) -----
    if face_probs is not None and audio_probs is not None:
        st.divider()
        st.markdown("### 🔗 Fused (face + voice)")
        show_result(face_probs, audio_probs, strategy, latency)
    elif not (img_f or vid_f or aud_f):
        st.info("Upload an image, video, and/or audio file to begin.")


def tab_fairness():
    st.subheader("Fairness Dashboard")
    report = PROJECT_ROOT / "fairness_report.csv"
    if not report.exists():
        st.warning("Run `python src/audit.py` first to generate fairness_report.csv.")
        return
    df = pd.read_csv(report)
    st.caption("Per-subgroup accuracy. Audio uses real CREMA-D demographics; "
               "face subgroups are predicted (noisy) by a FairFace annotator.")
    for (mod, dim), sub in df.groupby(["modality", "dimension"]):
        st.markdown(f"**{mod} — {dim}**")
        chart = sub.set_index("group")["accuracy"]
        st.bar_chart(chart)
    img = PROJECT_ROOT / "fairness_audit.png"
    if img.exists():
        st.image(str(img), caption="Fairness audit overview")
    bench = PROJECT_ROOT / "fusion_benchmark.csv"
    if bench.exists():
        st.markdown("**Fusion vs solo models**")
        st.dataframe(pd.read_csv(bench), hide_index=True)


def tab_ethics():
    st.subheader("About & Ethics")
    st.markdown(
        """
        This is a **research demo** of multimodal (face + voice) emotion estimation
        with a fairness audit. It is **not** a diagnostic or decision-making tool.
        """
    )
    st.warning(
        "**Ethical use policy**\n\n"
        "- Output is an **estimate**, not a definitive label or diagnosis.\n"
        "- **Discouraged** for hiring, surveillance, or clinical/high-stakes use.\n"
        "- Trained only on **public, open-access** datasets (FER+, AffectNet, "
        "RAVDESS, CREMA-D, FairFace).\n"
        "- **No data is stored** beyond the current session."
    )
    st.markdown(
        """
        **Known limitations**
        - Face subgroup labels in the audit are *predicted*, so race slices are noisy.
        - 'Surprise' is weak in audio (absent from CREMA-D).
        - Emotion is contextual and cultural; a single label cannot capture it fully.
        """
    )


def main():
    st.set_page_config(page_title="Emotion & Mood Detection", layout="centered")
    st.title("Emotion & Mood Detection")
    st.caption(f"Face + Voice multimodal classifier with fairness auditing · device: {DEVICE}")

    face_model, audio_model = load_models()
    t1, t2, t3, t4 = st.tabs(["🎥 Live Detection", "📤 Upload & Analyze",
                              "⚖️ Fairness", "ℹ️ About & Ethics"])
    with t1:
        tab_live(face_model, audio_model)
    with t2:
        tab_upload(face_model, audio_model)
    with t3:
        tab_fairness()
    with t4:
        tab_ethics()


if __name__ == "__main__":
    main()
