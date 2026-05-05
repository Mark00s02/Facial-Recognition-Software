"""
Face detection and recognition engine.
Runs in DL mode (YuNet + SFace) when ONNX models are present;
falls back to OpenCV Haar + LBPH otherwise.
"""
from __future__ import annotations
import io
from typing import List, Tuple, Optional

import cv2
import numpy as np

from database import Database
from dl_engine import DLEngine

# (name, (top, right, bottom, left), confidence 0-1)
RecognitionResult = Tuple[str, Tuple[int, int, int, int], float]

FACE_SIZE        = (100, 100)
LBPH_MAX_DIST    = 180.0   # decision threshold: tolerance × this
LBPH_DISPLAY_MAX = 75.0    # display-only normaliser — makes typical distances read higher


class FaceEngine:
    def __init__(self, db: Database):
        self.db  = db
        self._dl = DLEngine()

        # Haar cascade detectors (used in LBPH mode and guided-capture preview)
        frontal_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        profile_path = cv2.data.haarcascades + "haarcascade_profileface.xml"
        self.detector          = cv2.CascadeClassifier(frontal_path)
        self._profile_detector = cv2.CascadeClassifier(profile_path)

        # LBPH recogniser (only trained in LBPH mode)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1, neighbors=8, grid_x=8, grid_y=8)

        # Runtime state
        self._label_to_name: dict[int, str] = {}
        self._dl_embeddings: List[Tuple[str, np.ndarray]] = []
        self._trained = False

        self.reload()

    # ── Mode ──────────────────────────────────────────────────────────────────

    @property
    def mode(self) -> str:
        """'DL' if ONNX models are loaded, 'LBPH' otherwise."""
        return "DL" if self._dl.ready else "LBPH"

    @property
    def dl(self) -> DLEngine:
        return self._dl

    # ── Reload / retrain ──────────────────────────────────────────────────────

    def reload(self):
        if self._dl.ready:
            self._reload_dl()
        else:
            self._reload_lbph()

    def _reload_dl(self):
        rows = self.db.get_all_dl_embeddings()
        self._dl_embeddings = [
            (name, _blob_to_emb(blob)) for name, blob in rows
        ]

    def _reload_lbph(self):
        rows = self.db.get_all_faces()
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
            images.append(_blob_to_array(blob))
            labels.append(name_to_label[name])
        self._label_to_name = {v: k for k, v in name_to_label.items()}
        self.recognizer.train(images, np.array(labels, dtype=np.int32))
        self._trained = True

    # ── Registration ──────────────────────────────────────────────────────────

    def register_dl(self, name: str,
                    captures: List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]) -> int:
        """
        Store DL embeddings for a new person.
        captures: list of (face_gray_100x100, bgr_frame, face_row_or_None)
        Returns number of embeddings successfully stored.
        """
        stored = 0
        for _, bgr, face_row in captures:
            row = face_row
            if row is None:
                # YuNet was not used during capture — re-detect now
                detected = self._dl.detect(bgr)
                if len(detected) == 0:
                    continue
                row = detected[0]
            try:
                emb = self._dl.embedding(bgr, row)
                self.db.add_dl_embedding(name, _emb_to_blob(emb))
                stored += 1
            except Exception:
                pass
        if stored > 0:
            self.reload()
        return stored

    def register_multiple(self, name: str, face_arrays: List[np.ndarray]):
        """LBPH registration (fallback / legacy)."""
        for arr in face_arrays:
            self.db.add_face(name, _array_to_blob(arr))
        self.reload()

    def register(self, name: str, face_array: np.ndarray):
        self.db.add_face(name, _array_to_blob(face_array))
        self.reload()

    def delete(self, face_id: int):
        self.db.delete_face(face_id)
        self.reload()

    # ── Recognition ───────────────────────────────────────────────────────────

    def recognize(self, bgr_frame: np.ndarray,
                  tolerance: float = 0.60) -> List[RecognitionResult]:
        if self._dl.ready:
            return self._recognize_dl(bgr_frame, tolerance)
        return self._recognize_lbph(bgr_frame, tolerance)

    def _recognize_dl(self, bgr: np.ndarray,
                      tolerance: float) -> List[RecognitionResult]:
        if not self._dl_embeddings:
            # Models loaded but no one registered yet — still show detections
            faces = self._dl.detect(bgr)
            results = []
            for face_row in faces:
                x, y, w, h = DLEngine.face_box(face_row)
                results.append(("Unknown", (y, x + w, y + h, x), 0.0))
            return results

        faces = self._dl.detect(bgr)
        if len(faces) == 0:
            return []

        threshold = DLEngine.tolerance_to_threshold(tolerance)
        results: List[RecognitionResult] = []

        for face_row in faces:
            x, y, w, h = DLEngine.face_box(face_row)
            top, right, bottom, left = y, x + w, y + h, x
            name       = "Unknown"
            confidence = 0.0
            try:
                query_emb  = self._dl.embedding(bgr, face_row)
                best_score = -1.0
                best_name  = "Unknown"
                for stored_name, stored_emb in self._dl_embeddings:
                    score = self._dl.cosine_score(query_emb, stored_emb)
                    if score > best_score:
                        best_score = score
                        best_name  = stored_name
                confidence = DLEngine.score_to_confidence(best_score)
                if best_score >= threshold:
                    name = best_name
            except Exception:
                pass
            results.append((name, (top, right, bottom, left), confidence))

        return results

    def _recognize_lbph(self, bgr: np.ndarray,
                         tolerance: float) -> List[RecognitionResult]:
        gray = cv2.cvtColor(bgr.copy(), cv2.COLOR_BGR2GRAY)
        faces = self._detect_haar(gray)
        if not len(faces):
            return []

        dist_threshold = tolerance * LBPH_MAX_DIST
        results: List[RecognitionResult] = []
        for x, y, w, h in faces:
            top, right, bottom, left = y, x + w, y + h, x
            face_roi   = cv2.resize(gray[y:y + h, x:x + w], FACE_SIZE)
            name       = "Unknown"
            confidence = 0.0
            if self._trained:
                label, dist = self.recognizer.predict(face_roi)
                confidence  = max(0.0, 1.0 - dist / LBPH_DISPLAY_MAX)
                if dist <= dist_threshold:
                    name = self._label_to_name.get(label, "Unknown")
            results.append((name, (top, right, bottom, left), confidence))
        return results

    # ── Frame annotation ──────────────────────────────────────────────────────

    @staticmethod
    def annotate_frame(frame: np.ndarray,
                       results: List[RecognitionResult]) -> np.ndarray:
        out = frame.copy()
        for name, (top, right, bottom, left), conf in results:
            color = (0, 220, 110) if name != "Unknown" else (30, 90, 255)
            cv2.rectangle(out, (left, top), (right, bottom), color, 2)
            label = f"{name}  {conf:.0%}" if name != "Unknown" else "Unknown"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(out,
                          (left, bottom - th - 10),
                          (left + tw + 6, bottom),
                          color, -1)
            cv2.putText(out, label, (left + 3, bottom - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 0, 0), 1, cv2.LINE_AA)
        return out

    # ── Haar helpers (LBPH mode / guided capture) ─────────────────────────────

    def encode_from_file(self, image_path: str) -> Optional[np.ndarray]:
        img = cv2.imread(image_path)
        if img is None:
            return None
        return self._extract_face(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    def _extract_face(self, gray: np.ndarray) -> Optional[np.ndarray]:
        faces = self._detect_haar(gray)
        if not faces:
            return None
        x, y, w, h = faces[0]
        return cv2.resize(gray[y:y + h, x:x + w], FACE_SIZE)

    def _detect_haar(self, gray: np.ndarray):
        try:
            safe     = np.ascontiguousarray(gray)
            frontal  = self.detector.detectMultiScale(
                safe, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            frontal  = list(frontal) if len(frontal) > 0 else []
            prof_l   = self._profile_detector.detectMultiScale(
                safe, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            prof_l   = list(prof_l) if len(prof_l) > 0 else []
            flipped  = cv2.flip(safe, 1)
            prof_r_r = self._profile_detector.detectMultiScale(
                flipped, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            w_img    = safe.shape[1]
            prof_r   = []
            if len(prof_r_r) > 0:
                for (x, y, w, h) in prof_r_r:
                    prof_r.append((w_img - x - w, y, w, h))
            all_faces = frontal + prof_l + prof_r
            return _nms(all_faces) if all_faces else []
        except Exception:
            return []


# ── Serialisation helpers ─────────────────────────────────────────────────────

def _array_to_blob(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()

def _blob_to_array(blob: bytes) -> np.ndarray:
    return np.load(io.BytesIO(blob))

def _emb_to_blob(emb: np.ndarray) -> bytes:
    return emb.astype(np.float32).tobytes()

def _blob_to_emb(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).reshape(1, 128)

def _nms(boxes, iou_threshold: float = 0.3):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept  = []
    for box in boxes:
        x1, y1, w1, h1 = box
        dominated = False
        for kx, ky, kw, kh in kept:
            ix = max(0, min(x1 + w1, kx + kw) - max(x1, kx))
            iy = max(0, min(y1 + h1, ky + kh) - max(y1, ky))
            inter = ix * iy
            union = w1 * h1 + kw * kh - inter
            if union > 0 and inter / union > iou_threshold:
                dominated = True
                break
        if not dominated:
            kept.append(box)
    return kept
