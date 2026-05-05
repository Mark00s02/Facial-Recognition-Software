"""
Guided multi-angle face capture dialog.
Collects face samples by guiding the user through 6 head positions.
Uses YuNet (DL) for detection when the DL engine is ready, otherwise Haar cascades.
Capture tuple: (face_gray_100x100, bgr_frame, face_row_or_None)
"""

from __future__ import annotations
import time
import tkinter as tk
from tkinter import ttk
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

from camera import Camera

# ── Palette ───────────────────────────────────────────────────────────────────
C = {
    "base":     "#1e1e2e",
    "mantle":   "#181825",
    "surface0": "#313244",
    "surface1": "#45475a",
    "overlay0": "#6c7086",
    "text":     "#cdd6f4",
    "blue":     "#89b4fa",
    "green":    "#a6e3a1",
    "red":      "#f38ba8",
    "peach":    "#fab387",
    "yellow":   "#f9e2af",
}

FONT_BOLD  = ("Segoe UI", 10, "bold")
FONT_SMALL = ("Segoe UI", 9)

CANVAS_W, CANVAS_H = 520, 390
TARGET_SAMPLES   = 150
CAPTURE_INTERVAL = 0.35
FACE_SIZE        = (100, 100)

PHASES = [
    ("Phase 1 — Front",        "Look straight at the camera",             C["blue"]),
    ("Phase 2 — Rotate left",  "Slowly turn your head LEFT, then back",   C["peach"]),
    ("Phase 3 — Rotate right", "Slowly turn your head RIGHT, then back",  C["peach"]),
    ("Phase 4 — Look up/down", "Tilt your head UP then DOWN slowly",      C["yellow"]),
    ("Phase 5 — Tilt sides",   "Tilt head to each shoulder & back",       C["yellow"]),
    ("Phase 6 — Full circle",  "Do one slow full circular head movement", C["green"]),
]
SAMPLES_PER_PHASE = TARGET_SAMPLES // len(PHASES)

# (face_gray_100x100, bgr_frame, face_row_or_None)
Capture = Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]


class GuidedCaptureDialog(tk.Toplevel):
    """
    Modal guided capture dialog.
    Calls on_complete(list[Capture]) on success, on_complete(None) on cancel.
    """

    def __init__(
        self,
        parent: tk.Misc,
        camera: Camera,
        detector: cv2.CascadeClassifier,
        name: str,
        on_complete: Callable[[Optional[List[Capture]]], None],
        dl_engine=None,
    ):
        super().__init__(parent)
        self.title(f"Registering — {name}")
        self.resizable(False, False)
        self.configure(bg=C["base"])
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._cancel)

        self.camera      = camera
        self.detector    = detector
        self.dl_engine   = dl_engine        # DLEngine or None
        self.name        = name
        self.on_complete = on_complete

        self._profile_det = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_profileface.xml")

        self._captures:      List[Capture] = []
        self._phase_idx:     int   = 0
        self._phase_samples: int   = 0
        self._last_capture:  float = 0.0
        self._running:       bool  = True
        self._photo_ref             = None
        self._flash_until:   float = 0.0

        self._build_ui()
        self._tick()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        hdr = tk.Frame(self, bg=C["mantle"])
        hdr.pack(fill=tk.X)

        # Mode badge (DL / LBPH)
        mode_text  = "DL MODE" if (self.dl_engine and self.dl_engine.ready) else "LBPH MODE"
        mode_color = C["green"] if (self.dl_engine and self.dl_engine.ready) else C["peach"]
        tk.Label(hdr, text=f"FACE REGISTRATION  —  {self.name.upper()}",
                 font=("Segoe UI", 12, "bold"),
                 bg=C["mantle"], fg=C["blue"]).pack(side=tk.LEFT, padx=16, pady=10)
        tk.Label(hdr, text=mode_text,
                 font=("Segoe UI", 8, "bold"),
                 bg=C["mantle"], fg=mode_color).pack(side=tk.LEFT, pady=10)

        self._count_lbl = tk.Label(hdr, text=f"0 / {TARGET_SAMPLES}",
                                    font=("Segoe UI", 12, "bold"),
                                    bg=C["mantle"], fg=C["green"])
        self._count_lbl.pack(side=tk.RIGHT, padx=16)

        # Intro hint
        self._hint_frame = tk.Frame(self, bg=C["surface0"])
        self._hint_frame.pack(fill=tk.X, padx=12, pady=(8, 0))
        tk.Label(self._hint_frame,
                 text="  HOW TO REGISTER:  Slowly rotate your head in a complete circle — "
                      "front, left, right, up, down, diagonal. "
                      "The system captures automatically whenever it sees your face.",
                 bg=C["surface0"], fg=C["text"],
                 font=FONT_SMALL, wraplength=500, justify=tk.LEFT).pack(padx=8, pady=6)

        # Phase dots
        dot_row = tk.Frame(self, bg=C["base"])
        dot_row.pack(pady=(8, 2))
        self._dots: List[tk.Label] = []
        for i, (label, _, _col) in enumerate(PHASES):
            short = label.split("—")[1].strip() if "—" in label else label
            col = tk.Frame(dot_row, bg=C["base"])
            col.pack(side=tk.LEFT, padx=8)
            dot = tk.Label(col, text="○", font=("Segoe UI", 16),
                           bg=C["base"], fg=C["surface1"])
            dot.pack()
            tk.Label(col, text=short, font=("Segoe UI", 7),
                     bg=C["base"], fg=C["overlay0"]).pack()
            self._dots.append(dot)

        # Camera canvas
        self._canvas = tk.Canvas(self, width=CANVAS_W, height=CANVAS_H,
                                  bg="#0a0a15", highlightthickness=3,
                                  highlightbackground=C["surface1"])
        self._canvas.pack(padx=16, pady=6)

        # Phase / instruction labels
        self._phase_lbl = tk.Label(self, text="",
                                    font=("Segoe UI", 13, "bold"),
                                    bg=C["base"], fg=C["blue"])
        self._phase_lbl.pack()
        self._inst_lbl  = tk.Label(self, text="",
                                    font=("Segoe UI", 11),
                                    bg=C["base"], fg=C["text"])
        self._inst_lbl.pack(pady=(2, 4))

        # Overall progress bar
        bar_bg = tk.Frame(self, bg=C["surface0"], height=12)
        bar_bg.pack(fill=tk.X, padx=16, pady=(4, 0))
        bar_bg.pack_propagate(False)
        self._pbar = tk.Frame(bar_bg, bg=C["blue"], height=12)
        self._pbar.place(relwidth=0.0, relheight=1.0)

        # Phase progress bar
        pbar2_bg = tk.Frame(self, bg=C["surface0"], height=6)
        pbar2_bg.pack(fill=tk.X, padx=16, pady=(2, 8))
        pbar2_bg.pack_propagate(False)
        self._pbar2 = tk.Frame(pbar2_bg, bg=C["peach"], height=6)
        self._pbar2.place(relwidth=0.0, relheight=1.0)

        # Status row
        status_row = tk.Frame(self, bg=C["base"])
        status_row.pack(fill=tk.X, padx=16, pady=(0, 14))
        self._status_lbl = tk.Label(status_row, text="Waiting for face…",
                                     font=FONT_BOLD, bg=C["base"],
                                     fg=C["overlay0"])
        self._status_lbl.pack(side=tk.LEFT)
        ttk.Button(status_row, text="Cancel",
                   style="Danger.TButton",
                   command=self._cancel).pack(side=tk.RIGHT)

        self._update_phase_ui()

    def _update_phase_ui(self):
        if self._phase_idx >= len(PHASES):
            return
        label, inst, col = PHASES[self._phase_idx]
        self._phase_lbl.config(text=label, fg=col)
        self._inst_lbl.config(text=inst)
        for i, dot in enumerate(self._dots):
            if i < self._phase_idx:
                dot.config(text="●", fg=C["green"])
            elif i == self._phase_idx:
                dot.config(text="◉", fg=col)
            else:
                dot.config(text="○", fg=C["surface1"])

    # ── Tick ──────────────────────────────────────────────────────────────────

    def _tick(self):
        if not self._running:
            return
        frame = self.camera.get_frame()
        if frame is not None:
            self._process(frame.copy())
        self.after(33, self._tick)

    # ── Detection ─────────────────────────────────────────────────────────────

    def _detect(self, bgr: np.ndarray, gray: np.ndarray):
        """
        Returns (boxes, face_rows) where:
          boxes     = list of (x, y, w, h) for drawing / cropping
          face_rows = list of ndarray(15,) for DL embedding, or list of None for Haar
        """
        if self.dl_engine and self.dl_engine.ready:
            faces = self.dl_engine.detect(bgr)
            if len(faces) == 0:
                return [], []
            boxes     = [(int(r[0]), int(r[1]), int(r[2]), int(r[3])) for r in faces]
            face_rows = [faces[i] for i in range(len(faces))]
            return boxes, face_rows
        else:
            boxes = self._detect_haar(gray)
            return boxes, [None] * len(boxes)

    def _detect_haar(self, gray: np.ndarray):
        safe = np.ascontiguousarray(gray)
        try:
            frontal = self.detector.detectMultiScale(
                safe, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            frontal = list(frontal) if len(frontal) > 0 else []
            prof_l  = self._profile_det.detectMultiScale(
                safe, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            prof_l  = list(prof_l) if len(prof_l) > 0 else []
            flipped = cv2.flip(safe, 1)
            prof_rr = self._profile_det.detectMultiScale(
                flipped, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            w_img   = safe.shape[1]
            prof_r  = []
            if len(prof_rr) > 0:
                for (x, y, w, h) in prof_rr:
                    prof_r.append((w_img - x - w, y, w, h))
            all_faces = frontal + prof_l + prof_r
            if not all_faces:
                return []
            all_faces = sorted(all_faces, key=lambda b: b[2] * b[3], reverse=True)
            kept = []
            for box in all_faces:
                x1, y1, w1, h1 = box
                dup = False
                for kx, ky, kw, kh in kept:
                    ix    = max(0, min(x1+w1, kx+kw) - max(x1, kx))
                    iy    = max(0, min(y1+h1, ky+kh) - max(y1, ky))
                    inter = ix * iy
                    union = w1*h1 + kw*kh - inter
                    if union > 0 and inter / union > 0.3:
                        dup = True
                        break
                if not dup:
                    kept.append(box)
            return kept
        except Exception:
            return []

    # ── Frame processing ──────────────────────────────────────────────────────

    def _process(self, bgr: np.ndarray):
        gray            = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        boxes, face_rows = self._detect(bgr, gray)

        display    = bgr.copy()
        face_found = len(boxes) > 0
        now        = time.time()

        # Guide oval
        fh, fw = bgr.shape[:2]
        cx, cy = fw // 2, fh // 2
        cv2.ellipse(display, (cx, cy), (fw // 5, fh // 3), 0, 0, 360,
                    (80, 80, 130), 1)

        # Flash border on capture
        if now < self._flash_until:
            cv2.rectangle(display, (0, 0), (fw - 1, fh - 1), (0, 220, 80), 4)

        for (x, y, w, h) in boxes:
            color = (0, 220, 80) if now < self._flash_until else (180, 220, 255)
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
            pct = f"{len(self._captures)}/{TARGET_SAMPLES}"
            cv2.putText(display, pct, (x + 4, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # Auto-capture
        if face_found and (now - self._last_capture) >= CAPTURE_INTERVAL:
            x, y, w, h = boxes[0]
            face_crop   = cv2.resize(gray[y:y + h, x:x + w], FACE_SIZE)
            face_row    = face_rows[0]   # ndarray(15,) or None
            self._captures.append((face_crop, bgr, face_row))
            self._last_capture   = now
            self._flash_until    = now + 0.15
            self._phase_samples += 1

            total = len(self._captures)
            self._count_lbl.config(text=f"{total} / {TARGET_SAMPLES}")
            self._pbar.place(relwidth=min(total / TARGET_SAMPLES, 1.0))
            self._pbar2.place(relwidth=min(self._phase_samples / SAMPLES_PER_PHASE, 1.0))
            self._status_lbl.config(text="Capturing…  keep moving!", fg=C["green"])
            self._hint_frame.pack_forget()

            if self._phase_samples >= SAMPLES_PER_PHASE:
                self._phase_idx    += 1
                self._phase_samples = 0
                self._pbar2.place(relwidth=0.0)
                if self._phase_idx < len(PHASES):
                    self._update_phase_ui()

            if total >= TARGET_SAMPLES:
                self._finish()
                return

        elif not face_found:
            self._status_lbl.config(text="No face detected — move closer",
                                     fg=C["peach"])
        else:
            self._status_lbl.config(
                text=f"Face detected ✓  — keep moving  ({len(self._captures)}/{TARGET_SAMPLES})",
                fg=C["text"])

        # Render to canvas
        rgb   = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        img   = Image.fromarray(rgb).resize((CANVAS_W, CANVAS_H), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self._canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self._photo_ref = photo

    # ── Finish / cancel ───────────────────────────────────────────────────────

    def _finish(self):
        self._running = False
        for dot in self._dots:
            dot.config(text="●", fg=C["green"])
        self._pbar.place(relwidth=1.0)
        self.after(600, lambda: (self.on_complete(self._captures), self.destroy()))

    def _cancel(self):
        self._running = False
        self.on_complete(None)
        self.destroy()
