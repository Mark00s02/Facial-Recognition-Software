"""
Facial Recognition System — Modern UI v2.0
"""

from __future__ import annotations
import os
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

from camera import Camera
from database import Database
from face_engine import FaceEngine
from guided_capture import GuidedCaptureDialog
from overlays import OverlayEngine

try:
    import winsound
    _HAS_SOUND = True
except ImportError:
    _HAS_SOUND = False

# ── Palette ────────────────────────────────────────────────────────────────────
C = {
    "bg":       "#0d0d1a",
    "base":     "#11111b",
    "mantle":   "#181825",
    "surface0": "#1e1e2e",
    "surface1": "#313244",
    "surface2": "#45475a",
    "overlay0": "#6c7086",
    "subtext":  "#a6adc8",
    "text":     "#cdd6f4",
    "blue":     "#89b4fa",
    "lavender": "#b4befe",
    "green":    "#a6e3a1",
    "teal":     "#94e2d5",
    "red":      "#f38ba8",
    "peach":    "#fab387",
    "yellow":   "#f9e2af",
    "mauve":    "#cba6f7",
    "sky":      "#89dceb",
}

FONT_UI    = ("Segoe UI", 10)
FONT_BOLD  = ("Segoe UI", 10, "bold")
FONT_H1    = ("Segoe UI", 16, "bold")
FONT_SMALL = ("Segoe UI", 9)
FONT_MONO  = ("Consolas", 9)

CANVAS_W, CANVAS_H = 620, 460
SIDEBAR_W = 86

NAV_REGISTER = "register"
NAV_MANAGE   = "manage"
NAV_HUNT     = "hunt"
NAV_LOG      = "log"
NAV_SETTINGS = "settings"

NAV_ITEMS = [
    ("📹", "Register",  NAV_REGISTER),
    ("👥", "Manage",    NAV_MANAGE),
    ("🎯", "Hunt",      NAV_HUNT),
    ("📋", "Activity",  NAV_LOG),
    ("⚙",  "Settings", NAV_SETTINGS),
]


# ── Sidebar button ─────────────────────────────────────────────────────────────

class NavButton(tk.Frame):
    def __init__(self, parent, icon: str, label: str, command):
        super().__init__(parent, bg=C["mantle"], cursor="hand2")
        self._cmd    = command
        self._active = False

        # Left accent bar
        self._accent = tk.Frame(self, bg=C["mantle"], width=3)
        self._accent.pack(side=tk.LEFT, fill=tk.Y)

        body = tk.Frame(self, bg=C["mantle"])
        body.pack(fill=tk.BOTH, expand=True, pady=12)
        self._body = body

        self._icon_lbl = tk.Label(body, text=icon, font=("Segoe UI", 19),
                                   bg=C["mantle"], fg=C["text"])
        self._icon_lbl.pack()
        self._text_lbl = tk.Label(body, text=label, font=("Segoe UI", 7),
                                   bg=C["mantle"], fg=C["overlay0"])
        self._text_lbl.pack()

        for w in [self, body, self._icon_lbl, self._text_lbl, self._accent]:
            w.bind("<Button-1>", lambda _: self._cmd())
            w.bind("<Enter>",    self._on_enter)
            w.bind("<Leave>",    self._on_leave)

    def set_active(self, active: bool):
        self._active = active
        self._repaint()

    def _repaint(self):
        if self._active:
            bg = C["surface0"]
            self.config(bg=bg); self._body.config(bg=bg)
            self._icon_lbl.config(bg=bg, fg=C["blue"])
            self._text_lbl.config(bg=bg, fg=C["lavender"])
            self._accent.config(bg=C["blue"])
        else:
            bg = C["mantle"]
            self.config(bg=bg); self._body.config(bg=bg)
            self._icon_lbl.config(bg=bg, fg=C["text"])
            self._text_lbl.config(bg=bg, fg=C["overlay0"])
            self._accent.config(bg=C["mantle"])

    def _on_enter(self, _=None):
        if not self._active:
            bg = C["surface1"]
            self.config(bg=bg); self._body.config(bg=bg)
            self._icon_lbl.config(bg=bg); self._text_lbl.config(bg=bg)
            self._accent.config(bg=C["surface2"])

    def _on_leave(self, _=None):
        if not self._active:
            self._repaint()


# ── Main application ───────────────────────────────────────────────────────────

class App:
    def __init__(self, root: tk.Tk):
        self.root = root

        self.db      = Database()
        self.engine  = FaceEngine(self.db)
        self.camera  = Camera()
        self.overlay = OverlayEngine()

        # Camera state
        self._running    = False
        self._lock       = threading.Lock()
        self._raw_frame: Optional[np.ndarray] = None
        self._ann_frame: Optional[np.ndarray] = None
        self._photo_ref  = None
        self._last_logged: dict = {}

        # Settings vars
        self.tolerance  = tk.DoubleVar(value=0.60)
        self.cam_index  = tk.IntVar(value=0)
        self.show_conf  = tk.BooleanVar(value=True)

        def _toggle(attr):
            var = tk.BooleanVar(value=False)
            var.trace_add("write", lambda *_: setattr(self.overlay, attr, var.get()))
            return var

        self.ov_age        = _toggle("show_age")
        self.ov_gender     = _toggle("show_gender")
        self.ov_emotion    = _toggle("show_emotion")
        self.ov_blur       = _toggle("blur_faces")
        self.ov_pixelate   = _toggle("pixelate_faces")
        self.ov_sunglasses = _toggle("show_sunglasses")
        self.ov_hat        = _toggle("show_hat")
        self.ov_fps        = _toggle("show_fps")
        self.ov_facecount  = _toggle("show_face_count")

        # Hunt state
        self._hunt_active        = False
        self._hunt_target: Optional[str] = None
        self._hunt_scan_y        = 0
        self._hunt_flash_end     = 0.0
        self._hunt_total_scanned = 0
        self._hunt_alert_count   = 0
        self._hunt_cooldown_end  = 0.0   # prevents alert spam

        self._apply_styles()
        self._build_ui()
        self._refresh_manage()

    # ── Styles ─────────────────────────────────────────────────────────────────

    def _apply_styles(self):
        s = ttk.Style()
        s.theme_use("clam")

        s.configure("TFrame",       background=C["bg"])
        s.configure("TLabel",       background=C["bg"], foreground=C["text"], font=FONT_UI)
        s.configure("TEntry",       fieldbackground=C["surface1"], foreground=C["text"],
                    borderwidth=0, font=FONT_UI, insertcolor=C["text"], padding=4)
        s.configure("TScale",       background=C["bg"], troughcolor=C["surface1"],
                    sliderlength=14)
        s.configure("TCheckbutton", background=C["surface0"], foreground=C["text"], font=FONT_UI)
        s.configure("TRadiobutton", background=C["surface0"], foreground=C["text"], font=FONT_UI)
        s.configure("TScrollbar",   background=C["surface2"], troughcolor=C["surface0"],
                    borderwidth=0, arrowcolor=C["subtext"], arrowsize=11)
        s.configure("TCombobox",    fieldbackground=C["surface1"], foreground=C["text"],
                    background=C["surface1"], selectbackground=C["blue"],
                    selectforeground=C["base"], borderwidth=0, font=FONT_UI)
        s.map("TCombobox",
              fieldbackground=[("readonly", C["surface1"])],
              selectbackground=[("readonly", C["surface1"])],
              selectforeground=[("readonly", C["text"])])

        s.configure("Treeview",
                    background=C["surface0"], foreground=C["text"],
                    fieldbackground=C["surface0"], font=FONT_UI, rowheight=28,
                    borderwidth=0)
        s.configure("Treeview.Heading",
                    background=C["surface1"], foreground=C["subtext"],
                    font=("Consolas", 8), borderwidth=0, relief="flat")
        s.map("Treeview",
              background=[("selected", C["blue"])],
              foreground=[("selected", C["base"])])

        for name, bg, fg in [
            ("TButton",          C["surface1"], C["text"]),
            ("Success.TButton",  C["green"],    C["base"]),
            ("Danger.TButton",   C["red"],      C["base"]),
            ("Warn.TButton",     C["yellow"],   C["base"]),
            ("Hunt.TButton",     C["mauve"],    C["base"]),
        ]:
            s.configure(name, background=bg, foreground=fg,
                        font=FONT_BOLD, borderwidth=0, padding=[10, 6], relief="flat")
            s.map(name,
                  background=[("active", C["surface2"]), ("disabled", C["surface1"])],
                  foreground=[("disabled", C["overlay0"])])

    # ── Main build ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.root.title("Facial Recognition System")
        self.root.geometry("1260x760")
        self.root.minsize(1060, 680)
        self.root.configure(bg=C["bg"])

        self._build_header()

        body = tk.Frame(self.root, bg=C["bg"])
        body.pack(fill=tk.BOTH, expand=True)

        self._build_sidebar(body)
        self._build_camera_area(body)
        self._build_right_area(body)

        self._switch_panel(NAV_REGISTER)

    # ── Header ─────────────────────────────────────────────────────────────────

    def _build_header(self):
        hdr = tk.Frame(self.root, bg=C["mantle"], height=54)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)

        # Thin bottom border
        tk.Frame(self.root, bg=C["surface1"], height=1).pack(fill=tk.X)

        left = tk.Frame(hdr, bg=C["mantle"])
        left.pack(side=tk.LEFT, padx=18)

        tk.Label(left, text="◈", font=("Segoe UI", 22),
                 bg=C["mantle"], fg=C["blue"]).pack(side=tk.LEFT)
        info = tk.Frame(left, bg=C["mantle"])
        info.pack(side=tk.LEFT, padx=10)
        tk.Label(info, text="FACIAL RECOGNITION SYSTEM",
                 font=("Segoe UI", 13, "bold"),
                 bg=C["mantle"], fg=C["text"]).pack(anchor=tk.W)
        tk.Label(info, text="Real-time identity detection & crowd scanning",
                 font=("Segoe UI", 7),
                 bg=C["mantle"], fg=C["overlay0"]).pack(anchor=tk.W)

        right = tk.Frame(hdr, bg=C["mantle"])
        right.pack(side=tk.RIGHT, padx=18)

        self._clock_lbl = tk.Label(right, text="", font=("Consolas", 11),
                                    bg=C["mantle"], fg=C["overlay0"])
        self._clock_lbl.pack(side=tk.RIGHT, padx=(14, 0))

        # Status pill
        pill_outer = tk.Frame(right, bg=C["surface2"], padx=1, pady=1)
        pill_outer.pack(side=tk.RIGHT)
        pill_in = tk.Frame(pill_outer, bg=C["surface0"])
        pill_in.pack()
        self._status_dot = tk.Label(pill_in, text="●", font=("Segoe UI", 9),
                                     bg=C["surface0"], fg=C["red"])
        self._status_dot.pack(side=tk.LEFT, padx=(10, 3), pady=6)
        self._status_lbl = tk.Label(pill_in, text="OFFLINE",
                                     font=("Segoe UI", 9, "bold"),
                                     bg=C["surface0"], fg=C["red"])
        self._status_lbl.pack(side=tk.LEFT, padx=(0, 10), pady=6)

        # Engine mode badge (DL / LBPH)
        mode_is_dl   = self.engine.mode == "DL"
        mode_text    = "◈ DL MODE" if mode_is_dl else "◈ LBPH MODE"
        mode_color   = C["green"]  if mode_is_dl else C["peach"]
        mode_outer = tk.Frame(right, bg=C["surface2"], padx=1, pady=1)
        mode_outer.pack(side=tk.RIGHT, padx=(0, 8))
        mode_in = tk.Frame(mode_outer, bg=C["surface0"])
        mode_in.pack()
        tk.Label(mode_in, text=mode_text,
                 font=("Segoe UI", 8, "bold"),
                 bg=C["surface0"], fg=mode_color).pack(padx=10, pady=6)

        self._tick_clock()

    def _tick_clock(self):
        self._clock_lbl.config(text=time.strftime("%H:%M:%S"))
        self.root.after(1000, self._tick_clock)

    # ── Sidebar ────────────────────────────────────────────────────────────────

    def _build_sidebar(self, parent):
        sb = tk.Frame(parent, bg=C["mantle"], width=SIDEBAR_W)
        sb.pack(side=tk.LEFT, fill=tk.Y)
        sb.pack_propagate(False)
        tk.Frame(sb, bg=C["surface1"], width=1).pack(side=tk.RIGHT, fill=tk.Y)

        self._nav_btns: dict[str, NavButton] = {}
        for icon, label, sid in NAV_ITEMS:
            btn = NavButton(sb, icon, label, lambda s=sid: self._switch_panel(s))
            btn.pack(fill=tk.X)
            self._nav_btns[sid] = btn

    # ── Camera area ────────────────────────────────────────────────────────────

    def _build_camera_area(self, parent):
        cam = tk.Frame(parent, bg=C["bg"], width=652)
        cam.pack(side=tk.LEFT, fill=tk.Y)
        cam.pack_propagate(False)

        # Canvas with 1 px border card
        card = tk.Frame(cam, bg=C["surface2"], padx=1, pady=1)
        card.pack(padx=10, pady=(10, 0))
        self._canvas = tk.Canvas(card, bg="#050510",
                                  width=CANVAS_W, height=CANVAS_H,
                                  highlightthickness=0)
        self._canvas.pack()
        self._show_placeholder()

        # Detection banner
        banner = tk.Frame(cam, bg=C["surface0"])
        banner.pack(fill=tk.X, padx=10, pady=(1, 0))
        tk.Label(banner, text="DETECTED", font=("Consolas", 7),
                 bg=C["surface0"], fg=C["overlay0"]).pack(side=tk.LEFT, padx=(12, 6), pady=6)
        tk.Frame(banner, bg=C["surface2"], width=1).pack(side=tk.LEFT, fill=tk.Y, pady=3)
        self._detect_lbl = tk.Label(banner, text="—",
                                     bg=C["surface0"], fg=C["teal"],
                                     font=("Segoe UI", 10, "bold"))
        self._detect_lbl.pack(side=tk.LEFT, padx=10)

        # Buttons
        ctrl = tk.Frame(cam, bg=C["bg"])
        ctrl.pack(fill=tk.X, padx=10, pady=8)
        self._start_btn = ttk.Button(ctrl, text="▶  START CAMERA",
                                      style="Success.TButton",
                                      command=self._start_camera)
        self._start_btn.pack(side=tk.LEFT, padx=(0, 5))
        self._stop_btn  = ttk.Button(ctrl, text="■  STOP",
                                      style="Danger.TButton",
                                      state=tk.DISABLED,
                                      command=self._stop_camera)
        self._stop_btn.pack(side=tk.LEFT, padx=(0, 5))
        self._snap_btn  = ttk.Button(ctrl, text="⬤  SNAPSHOT",
                                      state=tk.DISABLED,
                                      command=self._take_snapshot)
        self._snap_btn.pack(side=tk.LEFT)

    # ── Right area ─────────────────────────────────────────────────────────────

    def _build_right_area(self, parent):
        self._rp = tk.Frame(parent, bg=C["bg"])
        self._rp.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8), pady=8)

        self._panels: dict[str, tk.Frame] = {
            NAV_REGISTER: self._make_register_panel(self._rp),
            NAV_MANAGE:   self._make_manage_panel(self._rp),
            NAV_HUNT:     self._make_hunt_panel(self._rp),
            NAV_LOG:      self._make_log_panel(self._rp),
            NAV_SETTINGS: self._make_settings_panel(self._rp),
        }
        self._current_panel: Optional[tk.Frame] = None

    def _switch_panel(self, sid: str):
        if self._current_panel:
            self._current_panel.pack_forget()
        self._panels[sid].pack(fill=tk.BOTH, expand=True)
        self._current_panel = self._panels[sid]
        for k, b in self._nav_btns.items():
            b.set_active(k == sid)
        if sid == NAV_MANAGE:
            self._refresh_manage()
        elif sid == NAV_LOG:
            self._refresh_log()
        elif sid == NAV_HUNT:
            self._refresh_hunt_targets()

    # ── UI helpers ─────────────────────────────────────────────────────────────

    def _section(self, parent, title: str, sub: str = ""):
        f = tk.Frame(parent, bg=C["bg"])
        f.pack(fill=tk.X, padx=18, pady=(14, 6))
        tk.Label(f, text=title, font=FONT_H1,
                 bg=C["bg"], fg=C["text"]).pack(anchor=tk.W)
        if sub:
            tk.Label(f, text=sub, font=FONT_SMALL,
                     bg=C["bg"], fg=C["overlay0"]).pack(anchor=tk.W, pady=(1, 0))
        tk.Frame(f, bg=C["surface2"], height=1).pack(fill=tk.X, pady=(6, 0))

    def _card(self, parent, px=18, py=4) -> tk.Frame:
        outer = tk.Frame(parent, bg=C["surface2"], padx=1, pady=1)
        outer.pack(fill=tk.X, padx=px, pady=py)
        inner = tk.Frame(outer, bg=C["surface0"])
        inner.pack(fill=tk.X)
        return inner

    def _clabel(self, parent, text: str):
        tk.Label(parent, text=text, font=("Consolas", 7),
                 bg=C["surface0"], fg=C["overlay0"]).pack(anchor=tk.W, padx=14, pady=(10, 2))

    def _stat_box(self, parent, label: str, color: str) -> tk.Label:
        box_outer = tk.Frame(parent, bg=C["surface2"], padx=1, pady=1)
        box_outer.pack(side=tk.LEFT, padx=(0, 10))
        box_in = tk.Frame(box_outer, bg=C["surface0"])
        box_in.pack()
        num = tk.Label(box_in, text="0", font=("Segoe UI", 22, "bold"),
                       bg=C["surface0"], fg=color, width=5)
        num.pack(padx=12, pady=(8, 2))
        tk.Label(box_in, text=label, font=("Consolas", 7),
                 bg=C["surface0"], fg=C["overlay0"]).pack(pady=(0, 8))
        return num

    # ── Register panel ─────────────────────────────────────────────────────────

    def _make_register_panel(self, parent) -> tk.Frame:
        f = tk.Frame(parent, bg=C["bg"])
        self._section(f, "Register Face", "Add a new person to the recognition database")

        nc = self._card(f)
        self._clabel(nc, "FULL NAME")
        self._name_entry = ttk.Entry(nc, font=("Segoe UI", 12))
        self._name_entry.pack(fill=tk.X, padx=12, pady=(0, 12), ipady=7)

        cc = self._card(f)
        ci = tk.Frame(cc, bg=C["surface0"])
        ci.pack(fill=tk.X, padx=14, pady=12)
        self._clabel(ci, "GUIDED MULTI-ANGLE CAPTURE")
        tk.Label(ci,
                 text="Start the camera, then click below.\n"
                      "You will be guided through 6 head angles — 60 samples captured automatically.",
                 font=FONT_SMALL, bg=C["surface0"], fg=C["subtext"],
                 justify=tk.LEFT).pack(anchor=tk.W, pady=(4, 10))
        ttk.Button(ci, text="◉  Start Guided Capture",
                   command=self._register_from_camera).pack(anchor=tk.W)

        self._reg_status = tk.Label(f, text="", font=FONT_SMALL,
                                     bg=C["bg"], fg=C["green"])
        self._reg_status.pack(pady=6, padx=20, anchor=tk.W)
        return f

    # ── Manage panel ───────────────────────────────────────────────────────────

    def _make_manage_panel(self, parent) -> tk.Frame:
        f = tk.Frame(parent, bg=C["bg"])
        self._section(f, "Manage Faces", "View and remove registered identities")

        tree_outer = tk.Frame(f, bg=C["surface2"], padx=1, pady=1)
        tree_outer.pack(fill=tk.BOTH, expand=True, padx=18, pady=4)
        tree_in = tk.Frame(tree_outer, bg=C["surface0"])
        tree_in.pack(fill=tk.BOTH, expand=True)

        self._faces_tree = ttk.Treeview(tree_in,
                                         columns=("name", "samples", "added"),
                                         show="headings", selectmode="browse")
        for col, hdr, w, anc in [
            ("name",    "NAME",          150, tk.W),
            ("samples", "SAMPLES",        80, tk.CENTER),
            ("added",   "REGISTERED ON", 180, tk.W),
        ]:
            self._faces_tree.heading(col, text=hdr)
            self._faces_tree.column(col, width=w, anchor=anc)

        vsb = ttk.Scrollbar(tree_in, orient=tk.VERTICAL,
                             command=self._faces_tree.yview)
        self._faces_tree.configure(yscroll=vsb.set)
        self._faces_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        ctrl = tk.Frame(f, bg=C["bg"])
        ctrl.pack(fill=tk.X, padx=18, pady=8)
        ttk.Button(ctrl, text="Refresh",
                   command=self._refresh_manage).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(ctrl, text="Delete Selected", style="Danger.TButton",
                   command=self._delete_face).pack(side=tk.LEFT)
        self._face_count_lbl = tk.Label(ctrl, text="",
                                         fg=C["overlay0"], font=FONT_SMALL, bg=C["bg"])
        self._face_count_lbl.pack(side=tk.RIGHT)
        return f

    # ── Hunt panel ─────────────────────────────────────────────────────────────

    def _make_hunt_panel(self, parent) -> tk.Frame:
        f = tk.Frame(parent, bg=C["bg"])
        self._section(f, "Hunt Mode",
                      "Scan a crowd in real-time for a specific target identity")

        # ── Target selector ──
        tc = self._card(f, py=4)
        self._clabel(tc, "TARGET IDENTITY")
        self._hunt_target_var = tk.StringVar()
        self._hunt_combo = ttk.Combobox(tc, textvariable=self._hunt_target_var,
                                         state="readonly", font=("Segoe UI", 11))
        self._hunt_combo.pack(fill=tk.X, padx=12, pady=(0, 12), ipady=6)

        # Target photo preview
        prev_row = tk.Frame(tc, bg=C["surface0"])
        prev_row.pack(fill=tk.X, padx=12, pady=(0, 10))
        self._hunt_preview_canvas = tk.Canvas(prev_row, bg=C["surface1"],
                                               width=72, height=72,
                                               highlightthickness=1,
                                               highlightbackground=C["surface2"])
        self._hunt_preview_canvas.pack(side=tk.LEFT)
        self._hunt_preview_canvas.create_text(36, 36, text="?",
                                               font=("Segoe UI", 22),
                                               fill=C["overlay0"],
                                               tags="placeholder")
        self._hunt_preview_photo = None

        preview_info = tk.Frame(prev_row, bg=C["surface0"])
        preview_info.pack(side=tk.LEFT, padx=12)
        tk.Label(preview_info, text="SELECT TARGET", font=("Consolas", 7),
                 bg=C["surface0"], fg=C["overlay0"]).pack(anchor=tk.W)
        self._hunt_target_name_lbl = tk.Label(preview_info, text="None selected",
                                               font=("Segoe UI", 10, "bold"),
                                               bg=C["surface0"], fg=C["text"])
        self._hunt_target_name_lbl.pack(anchor=tk.W, pady=(2, 0))
        self._hunt_sample_lbl = tk.Label(preview_info, text="",
                                          font=FONT_SMALL,
                                          bg=C["surface0"], fg=C["overlay0"])
        self._hunt_sample_lbl.pack(anchor=tk.W)

        self._hunt_combo.bind("<<ComboboxSelected>>", self._on_hunt_target_changed)

        # ── Controls ──
        ctrl_c = self._card(f, py=4)
        ctrl_r = tk.Frame(ctrl_c, bg=C["surface0"])
        ctrl_r.pack(fill=tk.X, padx=14, pady=10)
        self._hunt_start_btn = ttk.Button(ctrl_r, text="▶  ACTIVATE HUNT",
                                           style="Warn.TButton",
                                           command=self._start_hunt)
        self._hunt_start_btn.pack(side=tk.LEFT, padx=(0, 8))
        self._hunt_stop_btn  = ttk.Button(ctrl_r, text="■  DEACTIVATE",
                                           style="Danger.TButton",
                                           state=tk.DISABLED,
                                           command=self._stop_hunt)
        self._hunt_stop_btn.pack(side=tk.LEFT)

        # ── Status card ──
        sc = self._card(f, py=4)
        stat_r = tk.Frame(sc, bg=C["surface0"])
        stat_r.pack(fill=tk.X, padx=14, pady=(12, 8))
        self._hunt_dot = tk.Label(stat_r, text="●", font=("Segoe UI", 12),
                                   bg=C["surface0"], fg=C["overlay0"])
        self._hunt_dot.pack(side=tk.LEFT)
        self._hunt_stat_lbl = tk.Label(stat_r, text="INACTIVE",
                                        font=("Segoe UI", 11, "bold"),
                                        bg=C["surface0"], fg=C["overlay0"])
        self._hunt_stat_lbl.pack(side=tk.LEFT, padx=8)

        counters = tk.Frame(sc, bg=C["surface0"])
        counters.pack(fill=tk.X, padx=14, pady=(0, 12))
        self._hunt_scanned_lbl = self._stat_box(counters, "FACES SCANNED", C["sky"])
        self._hunt_alerts_lbl  = self._stat_box(counters, "ALERTS",        C["red"])

        # ── Alert log ──
        tk.Label(f, text="ALERT LOG", font=("Consolas", 7),
                 bg=C["bg"], fg=C["overlay0"]).pack(anchor=tk.W, padx=18, pady=(8, 2))

        log_outer = tk.Frame(f, bg=C["surface2"], padx=1, pady=1)
        log_outer.pack(fill=tk.BOTH, expand=True, padx=18, pady=(0, 4))
        log_in = tk.Frame(log_outer, bg=C["surface0"])
        log_in.pack(fill=tk.BOTH, expand=True)

        self._hunt_log = ttk.Treeview(log_in,
                                       columns=("target", "conf", "time"),
                                       show="headings", height=5)
        self._hunt_log.heading("target", text="TARGET")
        self._hunt_log.heading("conf",   text="CONFIDENCE")
        self._hunt_log.heading("time",   text="DETECTED AT")
        self._hunt_log.column("target", width=120)
        self._hunt_log.column("conf",   width=100, anchor=tk.CENTER)
        self._hunt_log.column("time",   width=100, anchor=tk.CENTER)
        vsb2 = ttk.Scrollbar(log_in, orient=tk.VERTICAL, command=self._hunt_log.yview)
        self._hunt_log.configure(yscroll=vsb2.set)
        self._hunt_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb2.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Button(f, text="Clear Alert Log", style="Danger.TButton",
                   command=self._clear_hunt_log).pack(anchor=tk.W, padx=18, pady=(4, 10))
        return f

    # ── Activity log panel ─────────────────────────────────────────────────────

    def _make_log_panel(self, parent) -> tk.Frame:
        f = tk.Frame(parent, bg=C["bg"])
        self._section(f, "Activity Log", "Recognition history from live camera sessions")

        log_outer = tk.Frame(f, bg=C["surface2"], padx=1, pady=1)
        log_outer.pack(fill=tk.BOTH, expand=True, padx=18, pady=4)
        log_in = tk.Frame(log_outer, bg=C["surface0"])
        log_in.pack(fill=tk.BOTH, expand=True)

        self._log_tree = ttk.Treeview(log_in, columns=("name", "conf", "time"),
                                       show="headings")
        self._log_tree.heading("name", text="NAME")
        self._log_tree.heading("conf", text="CONFIDENCE")
        self._log_tree.heading("time", text="TIMESTAMP")
        self._log_tree.column("name", width=150)
        self._log_tree.column("conf", width=100, anchor=tk.CENTER)
        self._log_tree.column("time", width=180)
        vsb = ttk.Scrollbar(log_in, orient=tk.VERTICAL, command=self._log_tree.yview)
        self._log_tree.configure(yscroll=vsb.set)
        self._log_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        ctrl = tk.Frame(f, bg=C["bg"])
        ctrl.pack(fill=tk.X, padx=18, pady=8)
        ttk.Button(ctrl, text="Refresh",
                   command=self._refresh_log).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(ctrl, text="Clear Log", style="Danger.TButton",
                   command=self._clear_log).pack(side=tk.LEFT)
        return f

    # ── Settings panel ─────────────────────────────────────────────────────────

    def _make_settings_panel(self, parent) -> tk.Frame:
        f = tk.Frame(parent, bg=C["bg"])

        sc = tk.Canvas(f, bg=C["bg"], highlightthickness=0)
        vsb = ttk.Scrollbar(f, orient=tk.VERTICAL, command=sc.yview)
        sc.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        sc.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        inner = tk.Frame(sc, bg=C["bg"])
        win = sc.create_window((0, 0), window=inner, anchor=tk.NW)
        inner.bind("<Configure>",
                   lambda _: sc.configure(scrollregion=sc.bbox("all")))
        sc.bind("<Configure>",
                lambda e: sc.itemconfig(win, width=e.width))
        sc.bind("<Enter>",
                lambda _: sc.bind_all("<MouseWheel>",
                    lambda e: sc.yview_scroll(-(1 if e.delta > 0 else -1), "units")))
        sc.bind("<Leave>", lambda _: sc.unbind_all("<MouseWheel>"))

        self._section(inner, "Settings", "Recognition, camera, and overlay configuration")

        # Recognition
        rc = self._card(inner)
        ri = tk.Frame(rc, bg=C["surface0"])
        ri.pack(fill=tk.X, padx=14, pady=12)
        self._clabel(ri, "RECOGNITION")

        tol_row = tk.Frame(ri, bg=C["surface0"])
        tol_row.pack(fill=tk.X, pady=(6, 2))
        tk.Label(tol_row, text="Match Tolerance", font=FONT_BOLD,
                 bg=C["surface0"], fg=C["text"]).pack(side=tk.LEFT)
        self._tol_lbl = tk.Label(tol_row, text="0.60",
                                  font=("Segoe UI", 12, "bold"),
                                  bg=C["surface0"], fg=C["blue"])
        self._tol_lbl.pack(side=tk.RIGHT)
        ttk.Scale(ri, from_=0.1, to=0.9, orient=tk.HORIZONTAL,
                  variable=self.tolerance,
                  command=lambda v: self._tol_lbl.config(
                      text=f"{float(v):.2f}")).pack(fill=tk.X, pady=(0, 2))
        tk.Label(ri, text="Lower = stricter  ·  Higher = more lenient",
                 font=FONT_SMALL, bg=C["surface0"], fg=C["overlay0"]).pack(anchor=tk.W)

        # Camera
        cc2 = self._card(inner)
        ci2 = tk.Frame(cc2, bg=C["surface0"])
        ci2.pack(fill=tk.X, padx=14, pady=12)
        self._clabel(ci2, "CAMERA")
        cam_row = tk.Frame(ci2, bg=C["surface0"])
        cam_row.pack(fill=tk.X, pady=(6, 0))
        tk.Label(cam_row, text="Camera Index", font=FONT_BOLD,
                 bg=C["surface0"], fg=C["text"]).pack(side=tk.LEFT)
        for i in range(5):
            ttk.Radiobutton(cam_row, text=str(i),
                            variable=self.cam_index, value=i).pack(side=tk.RIGHT, padx=4)

        # Overlays
        oc = self._card(inner)
        oi = tk.Frame(oc, bg=C["surface0"])
        oi.pack(fill=tk.X, padx=14, pady=12)
        self._clabel(oi, "LIVE OVERLAYS")

        models_ok = self.overlay.models_ready
        note = ("✓  Age/Gender models loaded" if models_ok
                else "⚠  Run  python download_models.py  to enable Age & Gender")
        tk.Label(oi, text=note, font=FONT_SMALL,
                 bg=C["surface0"],
                 fg=C["green"] if models_ok else C["peach"]).pack(anchor=tk.W, pady=(4, 8))

        ov_groups = [
            ("ANALYSIS", [
                ("Show Age",         self.ov_age,        True),
                ("Show Gender",      self.ov_gender,     True),
                ("Show Emotion",     self.ov_emotion,    False),
                ("Show Confidence",  self.show_conf,     False),
                ("FPS Counter",      self.ov_fps,        False),
                ("Face Count",       self.ov_facecount,  False),
            ]),
            ("PRIVACY", [
                ("Blur Faces",       self.ov_blur,       False),
                ("Pixelate Faces",   self.ov_pixelate,   False),
            ]),
            ("FUN / AR", [
                ("Sunglasses",       self.ov_sunglasses, False),
                ("Party Hat",        self.ov_hat,        False),
            ]),
        ]

        for grp_lbl, items in ov_groups:
            tk.Label(oi, text=grp_lbl, font=("Consolas", 7),
                     bg=C["surface0"], fg=C["overlay0"]).pack(anchor=tk.W, pady=(8, 2))
            for label, var, needs_model in items:
                row = tk.Frame(oi, bg=C["surface0"])
                row.pack(fill=tk.X, pady=1)
                cb = ttk.Checkbutton(row, text=label, variable=var)
                if needs_model and not models_ok:
                    cb.state(["disabled"])
                cb.pack(side=tk.LEFT)

        def _excl_blur(*_):
            if self.ov_blur.get(): self.ov_pixelate.set(False)
        def _excl_pix(*_):
            if self.ov_pixelate.get(): self.ov_blur.set(False)
        self.ov_blur.trace_add("write", _excl_blur)
        self.ov_pixelate.trace_add("write", _excl_pix)

        tk.Frame(inner, bg=C["bg"], height=20).pack()
        return f

    # ── Camera control ─────────────────────────────────────────────────────────

    def _start_camera(self):
        try:
            self.camera = Camera(self.cam_index.get())
            self.camera.start()
        except RuntimeError as exc:
            messagebox.showerror("Camera Error", str(exc))
            return
        self._running = True
        self._start_btn.config(state=tk.DISABLED)
        self._stop_btn.config(state=tk.NORMAL)
        self._snap_btn.config(state=tk.NORMAL)
        self._set_status("LIVE", C["green"])
        threading.Thread(target=self._recognition_loop, daemon=True).start()
        self._update_canvas()

    def _stop_camera(self):
        self._running = False
        self._hunt_active = False
        self.camera.stop()
        self._start_btn.config(state=tk.NORMAL)
        self._stop_btn.config(state=tk.DISABLED)
        self._snap_btn.config(state=tk.DISABLED)
        self._set_status("OFFLINE", C["red"])
        self._detect_lbl.config(text="—")
        self._show_placeholder()
        self._refresh_log()
        self._reset_hunt_ui()

    def _set_status(self, text: str, color: str):
        self._status_dot.config(fg=color)
        self._status_lbl.config(text=text, fg=color)

    def _take_snapshot(self):
        frame = self.camera.snapshot()
        if frame is None:
            return
        os.makedirs("data/captures", exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.abspath(f"data/captures/snapshot_{ts}.jpg")
        cv2.imwrite(path, frame)
        messagebox.showinfo("Snapshot Saved", f"Saved to:\n{path}")

    # ── Recognition loop ───────────────────────────────────────────────────────

    def _recognition_loop(self):
        tick = 0
        while self._running:
            frame = self.camera.get_frame()
            if frame is None:
                time.sleep(0.02)
                continue

            tick += 1
            if tick % 3 == 0:
                results   = self.engine.recognize(frame.copy(),
                                                   tolerance=self.tolerance.get())
                annotated = self.engine.annotate_frame(frame, results)
                annotated = self.overlay.draw(annotated, results)

                # Hunt mode check
                if self._hunt_active and self._hunt_target:
                    self._hunt_total_scanned += len(results)
                    now = time.time()
                    for name, _, conf in results:
                        if name == self._hunt_target and now > self._hunt_cooldown_end:
                            self._hunt_alert_count += 1
                            self._hunt_flash_end   = now + 1.2
                            self._hunt_cooldown_end = now + 3.0
                            ts = time.strftime("%H:%M:%S")
                            self.db.log_recognition(name, conf)
                            self.root.after(0, lambda t=ts, c=conf: self._on_hunt_alert(t, c))
                            if _HAS_SOUND:
                                threading.Thread(
                                    target=lambda: (winsound.Beep(1200, 180),
                                                    time.sleep(0.1),
                                                    winsound.Beep(1500, 180)),
                                    daemon=True).start()
                            break
                    self.root.after(0, self._update_hunt_stats)

                names = [r[0] for r in results]
                with self._lock:
                    self._raw_frame = frame
                    self._ann_frame = annotated

                text = ", ".join(names) if names else "No faces detected"
                self.root.after(0, lambda t=text: self._detect_lbl.config(text=t))

                now = time.time()
                for name, _, conf in results:
                    if name != "Unknown":
                        if now - self._last_logged.get(name, 0) >= 5.0:
                            self.db.log_recognition(name, conf)
                            self._last_logged[name] = now
                            self.root.after(0, self._refresh_log)
            else:
                with self._lock:
                    self._raw_frame = frame

            time.sleep(0.01)

    # ── Canvas update ──────────────────────────────────────────────────────────

    def _update_canvas(self):
        if not self._running:
            return

        with self._lock:
            frame = self._ann_frame if self._ann_frame is not None else self._raw_frame

        if frame is not None:
            display = frame.copy()
            fh, fw  = display.shape[:2]
            now     = time.time()

            if self._hunt_active:
                target_found = now < self._hunt_flash_end

                # Scanning line
                self._hunt_scan_y = (self._hunt_scan_y + 5) % fh
                y = self._hunt_scan_y
                scan_color = (0, 60, 255) if target_found else (0, 220, 80)

                glow = display.copy()
                for dy, _ in [(0, 1.0), (-2, 0.5), (2, 0.5), (-4, 0.2), (4, 0.2)]:
                    yy = max(0, min(fh - 1, y + dy))
                    cv2.line(glow, (0, yy), (fw, yy), scan_color, 1)
                cv2.addWeighted(glow, 0.55, display, 0.45, 0, display)

                if target_found:
                    # Red alert border (double rect)
                    cv2.rectangle(display, (0, 0), (fw - 1, fh - 1), (0, 30, 200), 10)
                    cv2.rectangle(display, (10, 10), (fw - 11, fh - 11), (0, 60, 240), 3)
                    # "TARGET LOCATED" banner
                    cv2.rectangle(display, (0, fh - 38), (fw, fh), (0, 20, 160), -1)
                    cv2.putText(display, "  !! TARGET LOCATED !!",
                                (10, fh - 12), cv2.FONT_HERSHEY_SIMPLEX,
                                0.65, (60, 120, 255), 2, cv2.LINE_AA)
                else:
                    # "SCANNING" badge top-right
                    cv2.rectangle(display, (fw - 140, 0), (fw, 32), (0, 0, 0), -1)
                    cv2.putText(display, "◈ SCANNING",
                                (fw - 132, 22), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (0, 200, 80), 1, cv2.LINE_AA)

                    # Target name top-left
                    if self._hunt_target:
                        label = f"TARGET: {self._hunt_target.upper()}"
                        cv2.rectangle(display, (0, 0), (len(label) * 9 + 10, 30),
                                      (0, 0, 0), -1)
                        cv2.putText(display, label,
                                    (6, 21), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.52, (80, 200, 255), 1, cv2.LINE_AA)

            rgb   = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            img   = Image.fromarray(rgb).resize((CANVAS_W, CANVAS_H), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self._canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self._photo_ref = photo

        self.root.after(30, self._update_canvas)

    def _show_placeholder(self):
        self._canvas.delete("all")
        cw, ch = CANVAS_W, CANVAS_H
        self._canvas.create_rectangle(0, 0, cw, ch, fill="#050510", outline="")
        # Subtle grid
        for x in range(0, cw, 44):
            self._canvas.create_line(x, 0, x, ch, fill="#0c0c22", width=1)
        for y in range(0, ch, 44):
            self._canvas.create_line(0, y, cw, y, fill="#0c0c22", width=1)
        cx, cy = cw // 2, ch // 2
        self._canvas.create_oval(cx - 52, cy - 52, cx + 52, cy + 52,
                                  outline=C["surface1"], width=1, fill=C["surface0"])
        self._canvas.create_text(cx, cy - 8, text="◈",
                                  font=("Segoe UI", 26), fill=C["surface2"])
        self._canvas.create_text(cx, cy + 26, text="No Camera Feed",
                                  font=("Segoe UI", 11), fill=C["surface2"])
        self._canvas.create_text(cx, cy + 46, text="Press  ▶ START CAMERA  to begin",
                                  font=("Segoe UI", 9), fill=C["overlay0"])

    # ── Hunt mode ──────────────────────────────────────────────────────────────

    def _refresh_hunt_targets(self):
        persons = [name for name, _, _ in self.db.get_face_names()]
        self._hunt_combo["values"] = persons
        if persons and not self._hunt_target_var.get():
            self._hunt_combo.current(0)
            self._on_hunt_target_changed(None)

    def _on_hunt_target_changed(self, _):
        name = self._hunt_target_var.get()
        if not name:
            return
        self._hunt_target_name_lbl.config(text=name)

        # Sample count
        persons = {n: cnt for n, cnt, _ in self.db.get_face_names()}
        cnt = persons.get(name, 0)
        self._hunt_sample_lbl.config(text=f"{cnt} training samples")

        # Preview image
        safe = "".join(c for c in name if c.isalnum() or c in " _-").strip()
        folder = os.path.join("data", "faces", safe)
        self._hunt_preview_canvas.delete("all")
        loaded = False
        if os.path.isdir(folder):
            imgs = sorted(f for f in os.listdir(folder) if f.endswith(".jpg"))
            if imgs:
                try:
                    img = Image.open(os.path.join(folder, imgs[0]))
                    img = img.resize((72, 72), Image.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    self._hunt_preview_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                    self._hunt_preview_photo = photo
                    loaded = True
                except Exception:
                    pass
        if not loaded:
            self._hunt_preview_canvas.create_rectangle(0, 0, 72, 72,
                                                        fill=C["surface1"], outline="")
            self._hunt_preview_canvas.create_text(36, 36, text="👤",
                                                   font=("Segoe UI", 22),
                                                   fill=C["overlay0"])

    def _start_hunt(self):
        target = self._hunt_target_var.get()
        if not target:
            messagebox.showwarning("No Target", "Select a target identity first.")
            return
        if not self._running:
            messagebox.showwarning("Camera Offline",
                                   "Start the camera before activating Hunt Mode.")
            return
        self._hunt_target        = target
        self._hunt_active        = True
        self._hunt_total_scanned = 0
        self._hunt_alert_count   = 0
        self._hunt_scan_y        = 0
        self._hunt_flash_end     = 0.0
        self._hunt_cooldown_end  = 0.0
        self._hunt_scanned_lbl.config(text="0")
        self._hunt_alerts_lbl.config(text="0")

        self._hunt_start_btn.config(state=tk.DISABLED)
        self._hunt_stop_btn.config(state=tk.NORMAL)
        self._hunt_dot.config(fg=C["yellow"])
        self._hunt_stat_lbl.config(
            text=f"SCANNING FOR  ·  {target.upper()}", fg=C["yellow"])
        self._set_status("HUNTING", C["yellow"])

    def _stop_hunt(self):
        self._hunt_active = False
        self._hunt_target = None
        self._hunt_start_btn.config(state=tk.NORMAL)
        self._hunt_stop_btn.config(state=tk.DISABLED)
        self._reset_hunt_ui()
        if self._running:
            self._set_status("LIVE", C["green"])

    def _reset_hunt_ui(self):
        self._hunt_dot.config(fg=C["overlay0"])
        self._hunt_stat_lbl.config(text="INACTIVE", fg=C["overlay0"])
        self._hunt_start_btn.config(state=tk.NORMAL)
        self._hunt_stop_btn.config(state=tk.DISABLED)

    def _update_hunt_stats(self):
        self._hunt_scanned_lbl.config(text=str(self._hunt_total_scanned))
        self._hunt_alerts_lbl.config(text=str(self._hunt_alert_count))

    def _on_hunt_alert(self, ts: str, conf: float):
        target = self._hunt_target or "Unknown"
        self._hunt_log.insert("", 0, values=(target, f"{conf:.1%}", ts))
        self._hunt_dot.config(fg=C["red"])
        self._hunt_stat_lbl.config(
            text=f"TARGET LOCATED  ·  {conf:.1%}", fg=C["red"])
        self._set_status("TARGET FOUND", C["red"])
        # Revert status after 2.5 s
        self.root.after(2500, self._revert_hunt_status)

    def _revert_hunt_status(self):
        if self._hunt_active and self._hunt_target:
            self._hunt_dot.config(fg=C["yellow"])
            self._hunt_stat_lbl.config(
                text=f"SCANNING FOR  ·  {self._hunt_target.upper()}", fg=C["yellow"])
            self._set_status("HUNTING", C["yellow"])

    def _clear_hunt_log(self):
        for item in self._hunt_log.get_children():
            self._hunt_log.delete(item)

    # ── Face management ────────────────────────────────────────────────────────

    def _refresh_manage(self):
        for row in self._faces_tree.get_children():
            self._faces_tree.delete(row)
        persons = self.db.get_face_names()
        for name, cnt, first_seen in persons:
            self._faces_tree.insert("", tk.END,
                                     values=(name, cnt, first_seen[:19]))
        n = len(persons)
        self._face_count_lbl.config(
            text=f"{n} person{'s' if n != 1 else ''} registered")

    def _delete_face(self):
        sel = self._faces_tree.selection()
        if not sel:
            messagebox.showwarning("No Selection", "Select a person to delete.")
            return
        vals  = self._faces_tree.item(sel[0])["values"]
        name, count = vals[0], vals[1]
        if messagebox.askyesno("Delete",
                                f"Remove '{name}' and all {count} samples?"):
            self.db.delete_person(name)
            self.engine.reload()
            self._refresh_manage()

    # ── Activity log ───────────────────────────────────────────────────────────

    def _refresh_log(self):
        for row in self._log_tree.get_children():
            self._log_tree.delete(row)
        for name, conf, ts in self.db.get_recognition_log():
            self._log_tree.insert("", tk.END,
                                   values=(name, f"{conf:.1%}", ts[:19]))

    def _clear_log(self):
        if messagebox.askyesno("Clear Log", "Delete all recognition history?"):
            self.db.clear_recognition_log()
            self._refresh_log()

    # ── Registration ───────────────────────────────────────────────────────────

    def _register_from_camera(self):
        name = self._name_entry.get().strip()
        if not name:
            self._set_reg_status("Please enter a name.", error=True)
            return
        if not self._running:
            self._set_reg_status("Start the camera first.", error=True)
            return

        safe = "".join(c for c in name if c.isalnum() or c in " _-").strip()

        def on_complete(captures):
            if not captures:
                self._set_reg_status("Capture cancelled.", warn=True)
                return
            folder = os.path.join("data", "faces", safe)
            os.makedirs(folder, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            for i, capture in enumerate(captures):
                cv2.imwrite(os.path.join(folder, f"sample_{i:03d}_{ts}.jpg"), capture[1])

            if self.engine.mode == "DL":
                n = self.engine.register_dl(name, captures)
                mode_note = f" (DL — {n} embeddings)"
            else:
                face_arrays = [c[0] for c in captures]
                self.engine.register_multiple(name, face_arrays)
                n = len(face_arrays)
                mode_note = f" (LBPH — {n} samples)"

            self._set_reg_status(f"✓  '{name}' registered{mode_note}.")
            self._name_entry.delete(0, tk.END)
            self._refresh_manage()

        GuidedCaptureDialog(self.root, self.camera,
                             self.engine.detector, name, on_complete,
                             dl_engine=self.engine.dl)

    def _set_reg_status(self, msg: str, error=False, warn=False):
        color = C["red"] if error else (C["peach"] if warn else C["green"])
        self._reg_status.config(text=msg, fg=color)

    # ── Shutdown ───────────────────────────────────────────────────────────────

    def on_close(self):
        self._running = False
        self._hunt_active = False
        self.camera.stop()
        self.db.close()
        self.root.destroy()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
