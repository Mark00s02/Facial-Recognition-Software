"""
Core face detection and recognition engine.
Uses OpenCV Haar cascades (detection) + LBPH recognizer (recognition).
No dlib or face_recognition required — works on Python 3.14+.
"""

from __future__ import annotations
import io
from typing import List, Tuple, Optional

import cv2
import numpy as np

from database import Database

# (name, (top, right, bottom, left), confidence 0-1)
RecognitionResult = Tuple[str, Tuple[int, int, int, int], float]

FACE_SIZE = (100, 100)
# LBPH distances for real matches are typically 60–100.
# Setting this higher means 0.50 tolerance maps to threshold=90, which catches most matches.
LBPH_MAX_DIST = 180.0


class FaceEngine:
    def __init__(self, db: Database):
        self.db = db

        # Frontal face cascade
        frontal_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(frontal_path)

        # Profile (side-view) cascade — flipped to cover both left & right profiles
        profile_path = cv2.data.haarcascades + "haarcascade_profileface.xml"
        self._profile_detector = cv2.CascadeClassifier(profile_path)

        # LBPH recognizer from opencv-contrib-python
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1, neighbors=8, grid_x=8, grid_y=8
        )

        self._label_to_name: dict[int, str] = {}
        self._trained = False
        self.reload()

    # ── Encoding cache ────────────────────────────────────────────────────────

    def reload(self):
        """Retrain the LBPH recognizer from the database."""
        rows = self.db.get_all_faces()  # [(name, blob), ...]
        if not rows:
            self._trained = False
            self._label_to_name = {}
            return

        name_to_label: dict[str, int] = {}
        counter = 0
        images: List[np.ndarray] = []
        labels: List[int] = []

        for name, blob in rows:
            if name not in name_to_label:
                name_to_label[name] = counter
                counter += 1
            lbl = name_to_label[name]
            images.append(_blob_to_array(blob))
            labels.append(lbl)

        self._label_to_name = {v: k for k, v in name_to_label.items()}
        self.recognizer.train(images, np.array(labels, dtype=np.int32))
        self._trained = True

    # ── Face extraction helpers ───────────────────────────────────────────────

    def encode_from_file(self, image_path: str) -> Optional[np.ndarray]:
        """Return a grayscale face crop from an image file, or None."""
        img = cv2.imread(image_path)
        if img is None:
            return None
        return self._extract_face(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    def encode_from_array(self, rgb_array: np.ndarray) -> Optional[np.ndarray]:
        """Return a grayscale face crop from an RGB numpy array, or None."""
        gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
        return self._extract_face(gray)

    def _extract_face(self, gray: np.ndarray) -> Optional[np.ndarray]:
        faces = self._detect(gray)
        if len(faces) == 0:
            return None
        x, y, w, h = faces[0]
        return cv2.resize(gray[y : y + h, x : x + w], FACE_SIZE)

    def _detect(self, gray: np.ndarray):
        """Detect faces using frontal + both-direction profile cascades."""
        try:
            safe = np.ascontiguousarray(gray)

            # Frontal
            frontal = self.detector.detectMultiScale(
                safe, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )
            frontal = list(frontal) if len(frontal) > 0 else []

            # Profile — left-facing
            profile_l = self._profile_detector.detectMultiScale(
                safe, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )
            profile_l = list(profile_l) if len(profile_l) > 0 else []

            # Profile — right-facing (flip image horizontally, detect, then unflip coords)
            flipped = cv2.flip(safe, 1)
            profile_r_raw = self._profile_detector.detectMultiScale(
                flipped, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )
            w_img = safe.shape[1]
            profile_r = []
            if len(profile_r_raw) > 0:
                for (x, y, w, h) in profile_r_raw:
                    profile_r.append((w_img - x - w, y, w, h))

            all_faces = frontal + profile_l + profile_r
            if not all_faces:
                return []

            # Deduplicate overlapping boxes (keep larger box when IoU > 0.3)
            return _nms(all_faces)

        except (cv2.error, Exception):
            return []

    # ── Recognition ───────────────────────────────────────────────────────────

    def recognize(
        self, bgr_frame: np.ndarray, tolerance: float = 0.50
    ) -> List[RecognitionResult]:
        """
        Detect and identify faces in a BGR OpenCV frame.
        tolerance: 0.1 (strict) → 0.9 (loose) maps to LBPH distance threshold.
        Returns list of (name, (top, right, bottom, left), confidence).
        """
        # Copy before converting — bgr_frame may be updated by the camera thread
        gray = cv2.cvtColor(bgr_frame.copy(), cv2.COLOR_BGR2GRAY)
        faces = self._detect(gray)
        if not len(faces):
            return []

        # Map UI tolerance (0.1–0.9) to LBPH distance threshold
        dist_threshold = tolerance * LBPH_MAX_DIST

        results: List[RecognitionResult] = []
        for x, y, w, h in faces:
            top, right, bottom, left = y, x + w, y + h, x
            face_roi = cv2.resize(gray[y : y + h, x : x + w], FACE_SIZE)

            name = "Unknown"
            confidence = 0.0

            if self._trained:
                label, dist = self.recognizer.predict(face_roi)
                confidence = max(0.0, 1.0 - dist / LBPH_MAX_DIST)
                if dist <= dist_threshold:
                    name = self._label_to_name.get(label, "Unknown")

            results.append((name, (top, right, bottom, left), confidence))

        return results

    # ── Registration ─────────────────────────────────────────────────────────

    def register(self, name: str, face_array: np.ndarray):
        self.db.add_face(name, _array_to_blob(face_array))
        self.reload()

    def register_multiple(self, name: str, face_arrays: list[np.ndarray]):
        """Store multiple angle samples in one DB round-trip, then reload once."""
        for arr in face_arrays:
            self.db.add_face(name, _array_to_blob(arr))
        self.reload()

    def delete(self, face_id: int):
        self.db.delete_face(face_id)
        self.reload()

    # ── Frame annotation ─────────────────────────────────────────────────────

    @staticmethod
    def annotate_frame(
        frame: np.ndarray, results: List[RecognitionResult]
    ) -> np.ndarray:
        out = frame.copy()
        for name, (top, right, bottom, left), conf in results:
            color = (0, 220, 110) if name != "Unknown" else (30, 90, 255)
            cv2.rectangle(out, (left, top), (right, bottom), color, 2)
            label = f"{name}  {conf:.0%}" if name != "Unknown" else "Unknown"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(
                out,
                (left, bottom - th - 10),
                (left + tw + 6, bottom),
                color, -1,
            )
            cv2.putText(
                out, label, (left + 3, bottom - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA,
            )
        return out


# ── Serialisation helpers ─────────────────────────────────────────────────────

def _array_to_blob(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def _blob_to_array(blob: bytes) -> np.ndarray:
    return np.load(io.BytesIO(blob))


# ── Non-maximum suppression ───────────────────────────────────────────────────

def _nms(boxes, iou_threshold: float = 0.3):
    """Remove overlapping bounding boxes, keeping the largest."""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept = []
    for box in boxes:
        x1, y1, w1, h1 = box
        dominated = False
        for kx, ky, kw, kh in kept:
            ix = max(0, min(x1 + w1, kx + kw) - max(x1, kx))
            iy = max(0, min(y1 + h1, ky + kh) - max(y1, ky))
            intersection = ix * iy
            union = w1 * h1 + kw * kh - intersection
            if union > 0 and intersection / union > iou_threshold:
                dominated = True
                break
        if not dominated:
            kept.append(box)
    return kept
