"""
Thread-safe camera wrapper around OpenCV VideoCapture.
"""

from __future__ import annotations
import threading
import time
from typing import Optional

import cv2
import numpy as np


class Camera:
    def __init__(self, index: int = 0):
        self.index = index
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            # Fallback without backend hint
            cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera {self.index}. "
                "Check that the camera is connected and not used by another app."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        self._cap = cap
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
            self._cap = None
        with self._lock:
            self._frame = None

    # ── Frame access ──────────────────────────────────────────────────────────

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def snapshot(self) -> Optional[np.ndarray]:
        return self.get_frame()

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Internal ──────────────────────────────────────────────────────────────

    def _loop(self):
        while self._running:
            if self._cap is None:
                break
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
            else:
                time.sleep(0.01)

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def available_cameras(max_probe: int = 6) -> list[int]:
        found = []
        for i in range(max_probe):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                found.append(i)
                cap.release()
        return found
