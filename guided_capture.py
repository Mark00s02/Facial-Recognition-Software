"""
Guided multi-angle face capture dialog.
Collects ~60 face samples by having the user slowly rotate their head in
a full circular motion. Auto-captures every 0.4 s whenever a face is detected.
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
TARGET_SAMPLES      = 60       # total face crops to collect
CAPTURE_INTERVAL    = 0.35     # seconds between auto-captures
FACE_SIZE           = (100, 100)

# Phases: (label, instruction, colour-hint)
PHASES = [
    ("Phase 1 — Front",        "Look straight at the camera",                C["blue"]),
    ("Phase 2 — Rotate left",  "Slowly turn your head LEFT, then back",      C["peach"]),
    ("Phase 3 — Rotate right", "Slowly turn your head RIGHT, then back",     C["peach"]),
    ("Phase 4 — Look up/down", "Tilt your head UP then DOWN slowly",         C["yellow"]),
    ("Phase 5 — Tilt sides",   "Tilt head to each shoulder & back",          C["yellow"]),
    ("Phase 6 — Full circle",  "Do one slow full circular head movement",    C["green"]),
]
SAMPLES_PER_PHASE = TARGET_SAMPLES // len(PHASES)   # 10 each

# Type alias: (face_gray_100x100, bgr_frame)
Capture = Tuple[np.ndarray, np.ndarray]


class GuidedCaptureDialog(tk.Toplevel):
    """
    Modal guided capture.
    Calls on_complete(list[Capture]) on success, on_complete(None) on cancel.
    """

    def __init__(
        self,
        parent: tk.Misc,
        camera: Camera,
        detector: cv2.CascadeClassifier,
        name: str,
        on_complete: Callable[[Optional[List[Capture]]], None],
    ):
        super().__init__(parent)
        self.title(f"Registering — {name}")
        self.resizable(False, False)
        self.configure(bg=C["base"])
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._cancel)

        self.camera           = camera
        self.detector         = detector   # frontal
        self._profile_det     = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_profileface.xml"
        )
        self.name        = name
        self.on_complete = on_complete

        self._captures:      List[Capture] = []
        self._phase_idx:     int           = 0
        self._phase_samples: int           = 0
        self._last_capture:  float         = 0.0
        self._running:       bool          = True
        self._photo_ref                    = None
        self._flash_until:   float         = 0.0

        self._build_ui()
        self._tick()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg=C["mantle"])
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text=f"FACE REGISTRATION  —  {self.name.upper()}",
                 font=("Segoe UI", 12, "bold"),
                 bg=C["mantle"], fg=C["blue"]).pack(side=tk.LEFT, padx=16, pady=10)

        self._count_lbl = tk.Label(hdr, text="0 / 60",
                                    font=("Segoe UI", 12, "bold"),
                                    bg=C["mantle"], fg=C["green"])
        self._count_lbl.pack(side=tk.RIGHT, padx=16)

        # Intro hint (hidden after first capture)
        self._hint_frame = tk.Frame(self, bg=C["surface0"])
        self._hint_frame.pack(fill=tk.X, padx=12, pady=(8, 0))
        tk.Label(self._hint_frame,
                 text="  HOW TO REGISTER:  Slowly rotate your head in a complete circle — "
                      "front, left, right, up, down, diagonal. The system captures "
                      "automatically whenever it sees your face.",
                 bg=C["surface0"], fg=C["text"],
                 font=FONT_SMALL, wraplength=500, justify=tk.LEFT).pack(
                     padx=8, pady=6)

        # Phase dot strip
        dot_row = tk.Frame(self, bg=C["base"])
        dot_row.pack(pady=(8, 2))
        self._dots: List[tk.Label] = []
        for i, (label, _, _) in enumerate(PHASES):
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

        # Phase label
        self._phase_lbl = tk.Label(self, text="",
                                    font=("Segoe UI", 13, "bold"),
                                    bg=C["base"], fg=C["blue"])
        self._phase_lbl.pack()

        self._inst_lbl = tk.Label(self, text="",
                                   font=("Segoe UI", 11),
                                   bg=C["base"], fg=C["text"])
        self._inst_lbl.pack(pady=(2, 4))

        # Overall progress bar
        bar_bg = tk.Frame(self, bg=C["surface0"], height=12)
        bar_bg.pack(fill=tk.X, padx=16, pady=(4, 10))
        bar_bg.pack_propagate(False)
        self._pbar = tk.Frame(bar_bg, bg=C["blue"], height=12)
        self._pbar.place(relwidth=0.0, relheight=1.0)

        # Phase progress bar (inner)
        pbar2_bg = tk.Frame(self, bg=C["surface0"], height=6)
        pbar2_bg.pack(fill=tk.X, padx=16, pady=(0, 8))
        pbar2_bg.pack_propagate(False)
        self._pbar2 = tk.Frame(pbar2_bg, bg=C["peach"], height=6)
        self._pbar2.place(relwidth=0.0, relheight=1.0)

        # Status + cancel
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

    # ── Main tick ─────────────────────────────────────────────────────────────

    def _tick(self):
        if not self._running:
            return

        frame = self.camera.get_frame()
        if frame is not None:
            self._process(frame.copy())

        self.after(33, self._tick)

    def _detect_all(self, gray: np.ndarray) -> list:
        """Frontal + left profile + right profile (flipped) cascades, deduplicated."""
        safe = np.ascontiguousarray(gray)
        try:
            frontal = self.detector.detectMultiScale(
                safe, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            frontal = list(frontal) if len(frontal) > 0 else []

            profile_l = self._profile_det.detectMultiScale(
                safe, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            profile_l = list(profile_l) if len(profile_l) > 0 else []

            flipped = cv2.flip(safe, 1)
            profile_r_raw = self._profile_det.detectMultiScale(
                flipped, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            w_img = safe.shape[1]
            profile_r = []
            if len(profile_r_raw) > 0:
                for (x, y, w, h) in profile_r_raw:
                    profile_r.append((w_img - x - w, y, w, h))

            all_faces = frontal + profile_l + profile_r
            if not all_faces:
                return []
            # Simple NMS — drop boxes with IoU > 0.3 vs a larger box
            all_faces = sorted(all_faces, key=lambda b: b[2]*b[3], reverse=True)
            kept = []
            for box in all_faces:
                x1,y1,w1,h1 = box
                dup = False
                for kx,ky,kw,kh in kept:
                    ix = max(0, min(x1+w1,kx+kw)-max(x1,kx))
                    iy = max(0, min(y1+h1,ky+kh)-max(y1,ky))
                    inter = ix*iy
                    union = w1*h1 + kw*kh - inter
                    if union > 0 and inter/union > 0.3:
                        dup = True; break
                if not dup:
                    kept.append(box)
            return kept
        except Exception:
            return []

    def _process(self, bgr: np.ndarray):
        gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = self._detect_all(gray)

        display    = bgr.copy()
        face_found = len(faces) > 0
        now        = time.time()

        # Draw guide oval (rough face placement guide)
        fh, fw = bgr.shape[:2]
        cx, cy = fw // 2, fh // 2
        cv2.ellipse(display, (cx, cy), (fw // 5, fh // 3), 0, 0, 360,
                    (80, 80, 130), 1)

        # Flash frame green on capture
        if now < self._flash_until:
            cv2.rectangle(display, (0, 0), (fw - 1, fh - 1), (0, 220, 80), 4)

        for (x, y, w, h) in faces:
            color = (0, 220, 80) if now < self._flash_until else (180, 220, 255)
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
            # Mini sample counter on face box
            pct = f"{len(self._captures)}/{TARGET_SAMPLES}"
            cv2.putText(display, pct, (x + 4, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # Auto-capture logic
        if face_found and (now - self._last_capture) >= CAPTURE_INTERVAL:
            x, y, w, h = faces[0]
            face_crop  = cv2.resize(gray[y:y+h, x:x+w], FACE_SIZE)
            self._captures.append((face_crop, bgr))
            self._last_capture  = now
            self._flash_until   = now + 0.15
            self._phase_samples += 1

            total = len(self._captures)
            self._count_lbl.config(text=f"{total} / {TARGET_SAMPLES}")
            self._pbar.place(relwidth=min(total / TARGET_SAMPLES, 1.0))
            self._pbar2.place(relwidth=min(self._phase_samples / SAMPLES_PER_PHASE, 1.0))
            self._status_lbl.config(text=f"Capturing…  keep moving!", fg=C["green"])

            # Hide intro hint after first capture
            self._hint_frame.pack_forget()

            # Advance phase
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
            self._status_lbl.config(text="No face detected — move closer", fg=C["peach"])
        else:
            remaining = max(0.0, CAPTURE_INTERVAL - (now - self._last_capture))
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
        # Mark all dots done
        for dot in self._dots:
            dot.config(text="●", fg=C["green"])
        self._pbar.place(relwidth=1.0)
        self.after(600, lambda: (self.on_complete(self._captures), self.destroy()))

    def _cancel(self):
        self._running = False
        self.on_complete(None)
        self.destroy()
