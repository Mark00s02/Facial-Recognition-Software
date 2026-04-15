"""
Overlay engine — all toggleable visual effects applied on top of the camera feed.

Features:
  - Age estimation     (requires downloaded Caffe models)
  - Gender detection   (requires downloaded Caffe models)
  - Emotion indicator  (fun heuristic, no model needed)
  - Blur faces         (Gaussian privacy blur)
  - Pixelate faces     (mosaic / pixel effect)
  - Sunglasses         (AR glasses drawn on faces)
  - Hat                (AR cap drawn above faces)
  - FPS counter
  - Face count badge
"""

from __future__ import annotations
import os
import time
from typing import List, Tuple

import cv2
import numpy as np

MODELS_DIR = "models"

AGE_BUCKETS = [
    "0-2 yrs", "4-6 yrs", "8-12 yrs", "15-20 yrs",
    "25-32 yrs", "38-43 yrs", "48-53 yrs", "60+ yrs",
]
GENDER_LIST = ["Male", "Female"]

# BGR colours
_YELLOW  = (0,   220, 255)
_CYAN    = (255, 220,   0)
_GREEN   = (80,  220, 100)
_WHITE   = (220, 220, 220)
_BLACK   = (10,   10,  10)
_ORANGE  = (0,   165, 255)

# Mean values used by the Caffe age/gender nets
_CAFFE_MEAN = (78.4263377603, 87.7689143744, 114.895847746)


class OverlayEngine:
    def __init__(self):
        # ── Feature toggles ───────────────────────────────────────────────────
        self.show_age        = False
        self.show_gender     = False
        self.show_emotion    = False
        self.blur_faces      = False
        self.pixelate_faces  = False
        self.show_sunglasses = False
        self.show_hat        = False
        self.show_fps        = False
        self.show_face_count = False

        self._age_net:    cv2.dnn_Net | None = None
        self._gender_net: cv2.dnn_Net | None = None
        self._models_ok   = False
        self._fps_times:  List[float] = []

        self.try_load_models()

    # ── Model loading ─────────────────────────────────────────────────────────

    def try_load_models(self) -> bool:
        """Attempt to load age/gender Caffe nets. Returns True if successful."""
        age_p   = os.path.join(MODELS_DIR, "deploy_age.prototxt")
        age_m   = os.path.join(MODELS_DIR, "age_net.caffemodel")
        gen_p   = os.path.join(MODELS_DIR, "deploy_gender.prototxt")
        gen_m   = os.path.join(MODELS_DIR, "gender_net.caffemodel")
        if all(os.path.exists(f) for f in [age_p, age_m, gen_p, gen_m]):
            try:
                self._age_net    = cv2.dnn.readNet(age_m, age_p)
                self._gender_net = cv2.dnn.readNet(gen_m, gen_p)
                self._models_ok  = True
                return True
            except Exception:
                pass
        self._models_ok = False
        return False

    @property
    def models_ready(self) -> bool:
        return self._models_ok

    # ── Main draw call ────────────────────────────────────────────────────────

    def draw(
        self,
        frame: np.ndarray,
        results: list,            # [(name, (top,right,bottom,left), conf), ...]
    ) -> np.ndarray:
        out = frame.copy()
        now = time.time()

        self._fps_times.append(now)
        self._fps_times = [t for t in self._fps_times if now - t < 1.0]

        for name, (top, right, bottom, left), conf in results:
            # Clamp to frame bounds
            fh, fw = out.shape[:2]
            t = max(0, top);    b = min(fh, bottom)
            l = max(0, left);   r = min(fw, right)
            face_bgr = frame[t:b, l:r]
            if face_bgr.size == 0:
                continue

            # ── Region effects (blur / pixelate) ──────────────────────────────
            if self.blur_faces:
                out = _blur_region(out, t, r, b, l)
            elif self.pixelate_faces:
                out = _pixelate_region(out, t, r, b, l)

            # ── AR props ──────────────────────────────────────────────────────
            if self.show_sunglasses:
                out = _draw_sunglasses(out, t, r, b, l)
            if self.show_hat:
                out = _draw_hat(out, t, r, b, l)

            # ── Text badge lines above face box ───────────────────────────────
            lines: List[Tuple[str, tuple]] = []

            if self.show_age or self.show_gender:
                if self._models_ok:
                    age_lbl, gen_lbl = self._predict_age_gender(face_bgr)
                    parts = []
                    if self.show_gender:
                        parts.append(gen_lbl)
                    if self.show_age:
                        parts.append(age_lbl)
                    lines.append(("  ".join(parts), _YELLOW))
                else:
                    lines.append(("⚠ Models not downloaded", _ORANGE))

            if self.show_emotion:
                emo = _estimate_emotion(face_bgr)
                lines.append((emo, _CYAN))

            # Stack badges above the face box
            for i, (text, colour) in enumerate(lines):
                y = max(top - 10 - i * 22, 16)
                (tw, th), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
                cv2.rectangle(out,
                              (l, y - th - 4), (l + tw + 8, y + 2),
                              _BLACK, -1)
                cv2.putText(out, text, (l + 4, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                            colour, 1, cv2.LINE_AA)

        # ── Frame-level overlays ──────────────────────────────────────────────
        y_off = 22
        if self.show_fps:
            fps = len(self._fps_times)
            _text_shadow(out, f"FPS  {fps}", (8, y_off), 0.60, _GREEN)
            y_off += 24

        if self.show_face_count:
            n = len(results)
            _text_shadow(out, f"Faces  {n}", (8, y_off), 0.60, _WHITE)

        return out

    # ── Age / gender DNN ──────────────────────────────────────────────────────

    def _predict_age_gender(self, face_bgr: np.ndarray) -> Tuple[str, str]:
        blob = cv2.dnn.blobFromImage(
            face_bgr, scalefactor=1.0, size=(227, 227),
            mean=_CAFFE_MEAN, swapRB=False, crop=False,
        )
        self._gender_net.setInput(blob)
        g_preds = self._gender_net.forward()
        gender  = GENDER_LIST[g_preds[0].argmax()]

        self._age_net.setInput(blob)
        a_preds = self._age_net.forward()
        age     = AGE_BUCKETS[a_preds[0].argmax()]

        return age, gender


# ── Pure-OpenCV helper functions (no class needed) ────────────────────────────

def _estimate_emotion(face_bgr: np.ndarray) -> str:
    """
    Heuristic emotion indicator — for fun, not clinical accuracy.
    Analyses brightness distribution in mouth / eye / brow regions.
    """
    if face_bgr.size == 0:
        return "😐 Neutral"
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if h < 20 or w < 20:
        return "😐 Neutral"

    # Region slices
    brow   = gray[int(h*0.15):int(h*0.32), int(w*0.1):int(w*0.9)]
    eyes   = gray[int(h*0.28):int(h*0.52), int(w*0.05):int(w*0.95)]
    mouth  = gray[int(h*0.62):int(h*0.88), int(w*0.2):int(w*0.8)]

    if mouth.size == 0 or eyes.size == 0 or brow.size == 0:
        return "😐 Neutral"

    mouth_var  = float(np.var(mouth))
    brow_var   = float(np.var(brow))
    eye_bright = float(np.mean(eyes))
    face_mean  = float(np.mean(gray))

    # Simple rules
    if mouth_var > 650:
        return "😄 Happy"
    if brow_var > 500 and eye_bright < face_mean * 0.80:
        return "😠 Angry"
    if eye_bright > face_mean * 1.12 and mouth_var < 120:
        return "😮 Surprised"
    if eye_bright < face_mean * 0.72:
        return "😔 Sad"
    return "😐 Neutral"


def _blur_region(frame, top, right, bottom, left):
    roi = frame[top:bottom, left:right]
    if roi.size == 0:
        return frame
    frame[top:bottom, left:right] = cv2.GaussianBlur(roi, (55, 55), 30)
    return frame


def _pixelate_region(frame, top, right, bottom, left, blocks: int = 12):
    roi = frame[top:bottom, left:right]
    if roi.size == 0:
        return frame
    h, w = roi.shape[:2]
    if h < blocks or w < blocks:
        return frame
    small = cv2.resize(roi, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
    frame[top:bottom, left:right] = cv2.resize(
        small, (w, h), interpolation=cv2.INTER_NEAREST)
    return frame


def _draw_sunglasses(frame, top, right, bottom, left):
    fh = bottom - top
    fw = right  - left
    ey  = top + int(fh * 0.38)          # eye vertical centre
    e_h = max(4, int(fh * 0.14))        # lens half-height
    e_w = max(4, int(fw * 0.22))        # lens half-width
    lx  = left + int(fw * 0.28)         # left lens centre x
    rx  = left + int(fw * 0.72)         # right lens centre x

    # Tinted lenses
    overlay = frame.copy()
    cv2.ellipse(overlay, (lx, ey), (e_w, e_h), 0, 0, 360, (20, 20, 20), -1)
    cv2.ellipse(overlay, (rx, ey), (e_w, e_h), 0, 0, 360, (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Frames
    col = (0, 200, 255)
    cv2.ellipse(frame, (lx, ey), (e_w, e_h), 0, 0, 360, col, 2)
    cv2.ellipse(frame, (rx, ey), (e_w, e_h), 0, 0, 360, col, 2)
    cv2.line(frame, (lx + e_w, ey), (rx - e_w, ey), col, 2)       # bridge
    cv2.line(frame, (left,     ey), (lx - e_w, ey), col, 2)        # left arm
    cv2.line(frame, (rx + e_w, ey), (right,    ey), col, 2)        # right arm
    return frame


def _draw_hat(frame, top, right, bottom, left):
    fw   = right - left
    fh   = bottom - top
    brim_y   = top - int(fh * 0.04)
    crown_y  = top - int(fh * 0.42)
    brim_x1  = left  - int(fw * 0.15)
    brim_x2  = right + int(fw * 0.15)
    crown_x1 = left  + int(fw * 0.12)
    crown_x2 = right - int(fw * 0.12)

    # Crown fill
    pts = np.array([
        [crown_x1, brim_y],
        [crown_x1, crown_y],
        [crown_x2, crown_y],
        [crown_x2, brim_y],
    ], np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], (30, 20, 10))
    # Brim fill
    cv2.rectangle(overlay, (brim_x1, brim_y - 8), (brim_x2, brim_y + 6),
                  (30, 20, 10), -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

    # Outline
    col = (60, 40, 20)
    cv2.polylines(frame, [pts], True, col, 2)
    cv2.rectangle(frame, (brim_x1, brim_y - 8), (brim_x2, brim_y + 6), col, 2)
    # Hat band
    band_y = brim_y - int(fh * 0.07)
    cv2.rectangle(frame, (crown_x1, band_y), (crown_x2, brim_y - 2),
                  (0, 80, 200), -1)
    return frame


def _text_shadow(frame, text, pos, scale, colour):
    x, y = pos
    cv2.putText(frame, text, (x + 1, y + 1),
                cv2.FONT_HERSHEY_SIMPLEX, scale, _BLACK, 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)
