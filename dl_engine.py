"""
Deep learning face engine — OpenCV YuNet (detection) + SFace (recognition).
Both models are ONNX files from the OpenCV Zoo; no extra pip packages needed
beyond opencv-contrib-python >= 4.7.
Run  python download_models.py  to fetch the two ONNX files before using this.
"""
from __future__ import annotations
import os
from typing import Optional

import cv2
import numpy as np

MODELS_DIR      = "models"
DETECTOR_FILE   = "face_detection_yunet_2023mar.onnx"
RECOGNIZER_FILE = "face_recognition_sface_2021dec.onnx"

# Cosine similarity threshold recommended by the SFace paper / OpenCV docs.
# Pairs with score >= COSINE_THRESHOLD are the same person.
COSINE_THRESHOLD = 0.363

# cv2.FaceRecognizerSF.FR_COSINE == 0  (use int for version safety)
_FR_COSINE = 0


class DLEngine:
    """
    Wraps OpenCV's YuNet face detector and SFace face recogniser.

    Detection output shape: (N, 15) — each row is
        [x, y, w, h,  re_x, re_y,  le_x, le_y,  nt_x, nt_y,
         rcm_x, rcm_y,  lcm_x, lcm_y,  score]
    Embedding shape: (1, 128) float32.
    """

    def __init__(self):
        self._det:   Optional[cv2.FaceDetectorYN]   = None
        self._rec:   Optional[cv2.FaceRecognizerSF] = None
        self._ready: bool = False
        self._try_load()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def _try_load(self):
        det_path = os.path.join(MODELS_DIR, DETECTOR_FILE)
        rec_path = os.path.join(MODELS_DIR, RECOGNIZER_FILE)
        if not (os.path.exists(det_path) and os.path.exists(rec_path)):
            return
        try:
            self._det = cv2.FaceDetectorYN.create(
                det_path, "", (640, 480),
                score_threshold=0.55,
                nms_threshold=0.30,
                top_k=5000,
            )
            self._rec = cv2.FaceRecognizerSF.create(rec_path, "")
            self._ready = True
        except Exception as exc:
            print(f"[DLEngine] Could not load ONNX models: {exc}")

    @property
    def ready(self) -> bool:
        return self._ready

    # ── Detection ─────────────────────────────────────────────────────────────

    def detect(self, bgr: np.ndarray) -> np.ndarray:
        """
        Detect all faces in a BGR frame.
        Returns ndarray shape (N, 15) or empty (0, 15) array.
        """
        if not self._ready:
            return np.empty((0, 15), dtype=np.float32)
        h, w = bgr.shape[:2]
        self._det.setInputSize((w, h))
        _, faces = self._det.detect(bgr)
        return faces if faces is not None else np.empty((0, 15), dtype=np.float32)

    # ── Embedding ─────────────────────────────────────────────────────────────

    def embedding(self, bgr: np.ndarray, face_row: np.ndarray) -> np.ndarray:
        """
        Align-crop the face region and return a 128-d feature vector.
        face_row must be one row (shape (15,)) from detect().
        Returns ndarray shape (1, 128).
        """
        aligned = self._rec.alignCrop(bgr, face_row)
        return self._rec.feature(aligned)

    # ── Matching ──────────────────────────────────────────────────────────────

    def cosine_score(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Cosine similarity between two (1, 128) embeddings.
        Range ≈ –1 … 1.  Same person when score >= COSINE_THRESHOLD (0.363).
        """
        return float(self._rec.match(emb1, emb2, _FR_COSINE))

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def score_to_confidence(score: float) -> float:
        """
        Map cosine score to an intuitive 0–1 confidence value using a power
        curve so that a score of 0.5 ≈ 71 %, 0.7 ≈ 84 %, 0.9 ≈ 95 %.
        """
        return min(1.0, max(0.0, score ** 0.5))

    @staticmethod
    def tolerance_to_threshold(tolerance: float) -> float:
        """
        Convert the UI tolerance slider (0.1 strict → 0.9 loose) to a cosine
        similarity threshold.
        slider = 0.60 (default) → 0.370, which matches the OpenCV SFace default.
        """
        return max(0.10, 0.70 - tolerance * 0.55)

    @staticmethod
    def face_box(face_row: np.ndarray):
        """Return (x, y, w, h) integers from a YuNet face row."""
        return (int(face_row[0]), int(face_row[1]),
                int(face_row[2]), int(face_row[3]))
