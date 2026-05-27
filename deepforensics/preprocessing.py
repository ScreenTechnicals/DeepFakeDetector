"""
DeepForensics preprocessing — Modules 1 & 2 of the pipeline.

Module 1 (Input Media): ingests video/image, samples frames at 5 fps,
extracts audio via ffmpeg.

Module 2 (Preprocessing): detects faces with MediaPipe Face Mesh
(RetinaFace fallback), crops and normalises face regions, extracts
13-coefficient MFCCs from the audio track.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    import mediapipe as mp
except ImportError:
    mp = None  # type: ignore[assignment]

try:
    import librosa
except ImportError:
    librosa = None  # type: ignore[assignment]

from deepforensics.config import (
    AUDIO_SAMPLE_RATE_HZ,
    FRAME_SAMPLE_RATE_FPS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_RESOLUTION,
    MEDIAPIPE_FALLBACK_THRESHOLD,
    MFCC_HOP_MS,
    MFCC_NUM_COEFFICIENTS,
    MFCC_WINDOW_MS,
)

logger = logging.getLogger(__name__)

# File extensions recognised as video
_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# ---------------------------------------------------------------------------
# Module 1: Media Ingestion
# ---------------------------------------------------------------------------


class MediaIngester:
    """
    Ingests raw media (image or video), extracts frames at a fixed
    sampling rate, and optionally extracts the audio waveform.
    """

    def __init__(self, fps: int = FRAME_SAMPLE_RATE_FPS):
        self.fps = fps

    def ingest(
        self, media_path: str
    ) -> Tuple[List[np.ndarray], Optional[np.ndarray], float]:
        """
        Ingest a media file.

        Returns
        -------
        frames : list of np.ndarray (BGR, H×W×3)
            Sampled video frames, or a single image.
        audio : np.ndarray or None
            Mono 16 kHz waveform, or None if audio is unavailable.
        duration : float
            Duration of the video in seconds (0.0 for images).
        """
        path = Path(media_path)
        ext = path.suffix.lower()

        if ext in _IMAGE_EXTENSIONS:
            return self._ingest_image(str(path))
        elif ext in _VIDEO_EXTENSIONS:
            return self._ingest_video(str(path))
        else:
            # Try as video by default
            logger.warning("Unknown extension '%s', attempting video ingest.", ext)
            return self._ingest_video(str(path))

    # -- private ------------------------------------------------------------

    def _ingest_image(
        self, path: str
    ) -> Tuple[List[np.ndarray], None, float]:
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to read image: {path}")
        return [img], None, 0.0

    def _ingest_video(
        self, path: str
    ) -> Tuple[List[np.ndarray], Optional[np.ndarray], float]:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {path}")

        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / native_fps if native_fps > 0 else 0.0

        # Sample interval: pick every N-th frame to achieve the target fps
        sample_interval = max(1, int(round(native_fps / self.fps)))

        frames: List[np.ndarray] = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_interval == 0:
                frames.append(frame)
            frame_idx += 1
        cap.release()

        if not frames:
            raise ValueError(f"No frames extracted from video: {path}")

        # Extract audio
        audio = self._extract_audio(path)

        logger.info(
            "Ingested %d frames (%.1fs) from %s", len(frames), duration, path
        )
        return frames, audio, duration

    def _extract_audio(self, video_path: str) -> Optional[np.ndarray]:
        """Extract mono audio at 16 kHz via ffmpeg."""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            cmd = [
                "ffmpeg", "-y", "-i", video_path,
                "-vn",                         # no video
                "-acodec", "pcm_s16le",        # 16-bit PCM
                "-ar", str(AUDIO_SAMPLE_RATE_HZ),
                "-ac", "1",                    # mono
                tmp_path,
            ]
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=60,
                check=True,
            )

            if librosa is not None:
                waveform, _ = librosa.load(tmp_path, sr=AUDIO_SAMPLE_RATE_HZ, mono=True)
            else:
                # Fallback: scipy
                from scipy.io import wavfile
                sr, waveform = wavfile.read(tmp_path)
                waveform = waveform.astype(np.float32) / 32768.0
                if sr != AUDIO_SAMPLE_RATE_HZ:
                    logger.warning(
                        "Audio sample rate %d != expected %d", sr, AUDIO_SAMPLE_RATE_HZ
                    )

            Path(tmp_path).unlink(missing_ok=True)
            return waveform

        except (subprocess.CalledProcessError, FileNotFoundError, Exception) as exc:
            logger.warning("Audio extraction failed: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Module 2A: Face Detection
# ---------------------------------------------------------------------------


class FaceDetector:
    """
    Detects faces and extracts 468-point landmarks using MediaPipe Face
    Mesh, with RetinaFace as a fallback when MediaPipe fails on more than
    30 % of frames.
    """

    def __init__(self):
        self._mp_face_mesh = None
        self._retinaface = None

    def detect_faces(
        self, frames: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[Optional[np.ndarray]]]:
        """
        Detect faces across all frames.

        Returns
        -------
        crops : list of np.ndarray (BGR, 224×224×3)
            Face crops resized to the input resolution, one per frame.
            If detection fails on a frame, a zero-filled crop is used.
        landmarks : list of Optional[np.ndarray]
            MediaPipe 468-point landmarks as (468, 3) arrays, or None
            when detection fails for a frame.
        """
        crops, landmarks = self._detect_mediapipe(frames)

        # Check fallback condition
        fail_count = sum(1 for lm in landmarks if lm is None)
        fail_rate = fail_count / len(frames) if frames else 1.0

        if fail_rate > MEDIAPIPE_FALLBACK_THRESHOLD:
            logger.info(
                "MediaPipe failed on %.0f%% of frames — falling back to RetinaFace.",
                fail_rate * 100,
            )
            crops, landmarks = self._detect_retinaface(frames, crops, landmarks)

        return crops, landmarks

    def prepare_tensor(self, crops: List[np.ndarray]):
        """
        Convert BGR face crops to a normalised (T, 3, 224, 224) float
        tensor suitable for the EfficientNet-B4 backbone.
        """
        if torch is None:
            raise ImportError("PyTorch is required for tensor preparation.")

        tensors = []
        mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
        std = np.array(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)

        for crop in crops:
            # BGR → RGB, HWC → CHW, normalise to [0, 1]
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            chw = np.transpose(rgb, (2, 0, 1))  # (3, H, W)
            chw = (chw - mean) / std
            tensors.append(chw)

        return torch.tensor(np.stack(tensors, axis=0), dtype=torch.float32)

    # -- MediaPipe ----------------------------------------------------------

    def _detect_mediapipe(
        self, frames: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[Optional[np.ndarray]]]:
        if mp is None:
            logger.warning("MediaPipe not installed — returning empty detections.")
            blank = np.zeros((*INPUT_RESOLUTION, 3), dtype=np.uint8)
            return [blank.copy() for _ in frames], [None] * len(frames)

        if self._mp_face_mesh is None:
            self._mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

        crops: List[np.ndarray] = []
        landmarks: List[Optional[np.ndarray]] = []

        for frame in frames:
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self._mp_face_mesh.process(rgb)

            if result.multi_face_landmarks:
                face_lm = result.multi_face_landmarks[0]
                lm_array = np.array(
                    [(pt.x * w, pt.y * h, pt.z * w) for pt in face_lm.landmark],
                    dtype=np.float32,
                )  # (468, 3)

                # Bounding box from landmarks
                xs = lm_array[:, 0]
                ys = lm_array[:, 1]
                x1 = max(0, int(xs.min()) - 20)
                y1 = max(0, int(ys.min()) - 20)
                x2 = min(w, int(xs.max()) + 20)
                y2 = min(h, int(ys.max()) + 20)

                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    crops.append(
                        np.zeros((*INPUT_RESOLUTION, 3), dtype=np.uint8)
                    )
                    landmarks.append(None)
                else:
                    crops.append(
                        cv2.resize(face_crop, INPUT_RESOLUTION)
                    )
                    # Translate landmarks relative to crop
                    lm_array[:, 0] -= x1
                    lm_array[:, 1] -= y1
                    # Scale to 224×224
                    crop_w = x2 - x1
                    crop_h = y2 - y1
                    if crop_w > 0 and crop_h > 0:
                        lm_array[:, 0] *= INPUT_RESOLUTION[0] / crop_w
                        lm_array[:, 1] *= INPUT_RESOLUTION[1] / crop_h
                    landmarks.append(lm_array)
            else:
                crops.append(
                    np.zeros((*INPUT_RESOLUTION, 3), dtype=np.uint8)
                )
                landmarks.append(None)

        return crops, landmarks

    # -- RetinaFace fallback ------------------------------------------------

    def _detect_retinaface(
        self,
        frames: List[np.ndarray],
        existing_crops: List[np.ndarray],
        existing_landmarks: List[Optional[np.ndarray]],
    ) -> Tuple[List[np.ndarray], List[Optional[np.ndarray]]]:
        """
        Re-run detection on frames where MediaPipe failed, using
        RetinaFace.  Frames where MediaPipe already succeeded are kept.
        """
        try:
            from retinaface import RetinaFace as RF
        except ImportError:
            logger.warning(
                "RetinaFace not installed — cannot fallback. "
                "Install with: pip install retinaface"
            )
            return existing_crops, existing_landmarks

        crops = list(existing_crops)
        landmarks = list(existing_landmarks)

        for i, frame in enumerate(frames):
            if landmarks[i] is not None:
                continue  # MediaPipe succeeded

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = RF.detect_faces(rgb)
            if not detections:
                continue

            # Take the first (highest-confidence) face
            face_key = list(detections.keys())[0]
            face = detections[face_key]
            area = face.get("facial_area", None)
            if area is None:
                continue

            x1, y1, x2, y2 = area
            h, w = frame.shape[:2]
            x1 = max(0, x1 - 10)
            y1 = max(0, y1 - 10)
            x2 = min(w, x2 + 10)
            y2 = min(h, y2 + 10)

            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size > 0:
                crops[i] = cv2.resize(face_crop, INPUT_RESOLUTION)
                # RetinaFace returns 5 keypoints, not 468 — store None
                # for landmarks since semantic modules need MediaPipe format
                landmarks[i] = None

        return crops, landmarks


# ---------------------------------------------------------------------------
# Module 2B: MFCC Extraction
# ---------------------------------------------------------------------------


class MFCCExtractor:
    """
    Extracts 13-coefficient MFCCs from a mono audio waveform and
    derives a scalar energy envelope.
    """

    def __init__(
        self,
        n_mfcc: int = MFCC_NUM_COEFFICIENTS,
        win_ms: int = MFCC_WINDOW_MS,
        hop_ms: int = MFCC_HOP_MS,
        sr: int = AUDIO_SAMPLE_RATE_HZ,
    ):
        self.n_mfcc = n_mfcc
        self.n_fft = int(sr * win_ms / 1000)
        self.hop_length = int(sr * hop_ms / 1000)
        self.sr = sr

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCCs from a waveform.

        Parameters
        ----------
        audio : np.ndarray, shape (num_samples,)
            Mono waveform at ``self.sr`` Hz.

        Returns
        -------
        mfccs : np.ndarray, shape (n_mfcc, T_audio)
        """
        if librosa is None:
            raise ImportError("librosa is required for MFCC extraction.")

        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        return mfccs

    def energy_envelope(self, mfccs: np.ndarray) -> np.ndarray:
        """
        Compute a scalar energy envelope from MFCCs.

        Parameters
        ----------
        mfccs : np.ndarray, shape (n_mfcc, T_audio)

        Returns
        -------
        energy : np.ndarray, shape (T_audio,)
            L2 norm across MFCC coefficients per time step.
        """
        return np.linalg.norm(mfccs, axis=0)
