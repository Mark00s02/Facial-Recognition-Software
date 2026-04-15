"""
Facial Recognition System — Tkinter GUI
"""

from __future__ import annotations
import os
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

from camera import Camera
from database import Database
from face_engine import FaceEngine
from guided_capture import GuidedCaptureDialog
from overlays import OverlayEngine


# ── Colour palette (Catppuccin Mocha) ─────────────────────────────────────────
C = {
    "base":    "#1e1e2e",
    "mantle":  "#181825",
    "surface0": "#313244",
    "surface1": "#45475a",
    "overlay0": "#6c7086",
    "text":    "#cdd6f4",
    "blue":    "#89b4fa",
    "green":   "#a6e3a1",
    "red":     "#f38ba8",
    "peach":   "#fab387",
    "yellow":  "#f9e2af",
}

FONT_NORMAL = ("Segoe UI", 10)
FONT_BOLD   = ("Segoe UI", 10, "bold")
FONT_TITLE  = ("Segoe UI", 14, "bold")
FONT_SMALL  = ("Segoe UI", 9)
FONT_MONO   = ("Consolas", 10)

CANVAS_W, CANVAS_H = 620, 468


# ── App ────────────────────────────────────────────────────────────────────────

class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Facial Recognition System")
        self.root.geometry("1140x720")
        self.root.minsize(960, 640)
        self.root.configure(bg=C["base"])

        self.db      = Database()
        self.engine  = FaceEngine(self.db)
        self.camera  = Camera()
        self.overlay = OverlayEngine()

        self._running   = False
        self._lock      = threading.Lock()
        self._raw_frame: Optional[np.ndarray] = None
        self._ann_frame: Optional[np.ndarray] = None
        self._photo_ref = None
        self._last_logged: dict = {}

        self.tolerance     = tk.DoubleVar(value=0.60)
        self.cam_index     = tk.IntVar(value=0)
        self.show_conf     = tk.BooleanVar(value=True)
        self._sel_img_path: Optional[str] = None

        # ── Overlay toggles (BooleanVar → synced to OverlayEngine) ────────────
        def _make_toggle(attr):
            var = tk.BooleanVar(value=False)
            var.trace_add("write", lambda *_: setattr(self.overlay, attr, var.get()))
            return var

        self.ov_age        = _make_toggle("show_age")
        self.ov_gender     = _make_toggle("show_gender")
        self.ov_emotion    = _make_toggle("show_emotion")
        self.ov_blur       = _make_toggle("blur_faces")
        self.ov_pixelate   = _make_toggle("pixelate_faces")
        self.ov_sunglasses = _make_toggle("show_sunglasses")
        self.ov_hat        = _make_toggle("show_hat")
        self.ov_fps        = _make_toggle("show_fps")
        self.ov_facecount  = _make_toggle("show_face_count")

        self._apply_styles()
        self._build_ui()
        self._refresh_faces_list()

    # ── Styles ────────────────────────────────────────────────────────────────

    def _apply_styles(self):
        s = ttk.Style()
        s.theme_use("clam")

        s.configure("TFrame",         background=C["base"])
        s.configure("TLabel",         background=C["base"],    foreground=C["text"],    font=FONT_NORMAL)
        s.configure("TNotebook",      background=C["base"],    borderwidth=0)
        s.configure("TNotebook.Tab",  background=C["surface0"], foreground=C["text"],   padding=[14, 7], font=FONT_NORMAL)
        s.map("TNotebook.Tab",
              background=[("selected", C["blue"])],
              foreground=[("selected", C["base"])])

        for name, bg, fg in [
            ("TButton",         C["blue"],  C["base"]),
            ("Success.TButton", C["green"], C["base"]),
            ("Danger.TButton",  C["red"],   C["base"]),
            ("Warn.TButton",    C["peach"], C["base"]),
        ]:
            s.configure(name, background=bg, foreground=fg,
                        font=FONT_BOLD, borderwidth=0, padding=[10, 6], relief="flat")
            s.map(name,
                  background=[("active", C["overlay0"]), ("disabled", C["surface1"])],
                  foreground=[("disabled", C["overlay0"])])

        s.configure("TEntry",    fieldbackground=C["surface0"], foreground=C["text"],
                    borderwidth=1, font=FONT_NORMAL, insertcolor=C["text"])
        s.configure("TScale",    background=C["base"], troughcolor=C["surface0"])
        s.configure("TCheckbutton", background=C["base"], foreground=C["text"], font=FONT_NORMAL)

        s.configure("Treeview",  background=C["surface0"], foreground=C["text"],
                    fieldbackground=C["surface0"], font=FONT_NORMAL, rowheight=26)
        s.configure("Treeview.Heading", background=C["blue"], foreground=C["base"],
                    font=FONT_BOLD)
        s.map("Treeview",
              background=[("selected", C["blue"])],
              foreground=[("selected", C["base"])])

        s.configure("TScrollbar", background=C["surface1"], troughcolor=C["surface0"],
                    borderwidth=0, arrowcolor=C["text"])

    # ── UI Layout ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Header ──
        header = tk.Frame(self.root, bg=C["mantle"], height=56)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        tk.Label(header, text="FACIAL RECOGNITION SYSTEM",
                 font=("Segoe UI", 15, "bold"),
                 bg=C["mantle"], fg=C["blue"]).pack(side=tk.LEFT, padx=20, pady=14)

        self._status_lbl = tk.Label(header, text="⏺  Offline",
                                    font=FONT_BOLD, bg=C["mantle"], fg=C["red"])
        self._status_lbl.pack(side=tk.RIGHT, padx=20, pady=14)

        # ── Body ──
        body = tk.Frame(self.root, bg=C["base"])
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left — camera panel
        left = tk.Frame(body, bg=C["surface0"], width=648)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 6))
        left.pack_propagate(False)
        self._build_camera_panel(left)

        # Right — tabbed controls
        right = ttk.Frame(body)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        nb = ttk.Notebook(right)
        nb.pack(fill=tk.BOTH, expand=True)
        self._build_tab_register(nb)
        self._build_tab_manage(nb)
        self._build_tab_log(nb)
        self._build_tab_settings(nb)

    def _build_camera_panel(self, parent):
        # Canvas
        self._canvas = tk.Canvas(parent, bg="#0a0a15",
                                  width=CANVAS_W, height=CANVAS_H,
                                  highlightthickness=0)
        self._canvas.pack(padx=10, pady=(10, 6))
        self._show_placeholder()

        # Detection banner
        banner = tk.Frame(parent, bg=C["surface0"])
        banner.pack(fill=tk.X, padx=10)
        tk.Label(banner, text="DETECTED:", bg=C["surface0"],
                 fg=C["overlay0"], font=FONT_SMALL).pack(side=tk.LEFT, pady=4)
        self._detect_lbl = tk.Label(banner, text="—",
                                     bg=C["surface0"], fg=C["green"],
                                     font=FONT_BOLD)
        self._detect_lbl.pack(side=tk.LEFT, padx=8)

        # Buttons
        btn_row = tk.Frame(parent, bg=C["surface0"])
        btn_row.pack(fill=tk.X, padx=10, pady=8)

        self._start_btn = ttk.Button(btn_row, text="▶  Start Camera",
                                      style="Success.TButton",
                                      command=self._start_camera)
        self._start_btn.pack(side=tk.LEFT, padx=4)

        self._stop_btn = ttk.Button(btn_row, text="■  Stop",
                                     style="Danger.TButton",
                                     state=tk.DISABLED,
                                     command=self._stop_camera)
        self._stop_btn.pack(side=tk.LEFT, padx=4)

        self._snap_btn = ttk.Button(btn_row, text="📷  Snapshot",
                                     state=tk.DISABLED,
                                     command=self._take_snapshot)
        self._snap_btn.pack(side=tk.LEFT, padx=4)

    # ── Tab: Register ─────────────────────────────────────────────────────────

    def _build_tab_register(self, nb):
        tab = ttk.Frame(nb)
        nb.add(tab, text="  Register Face  ")

        tk.Label(tab, text="Register New Face", font=FONT_TITLE,
                 fg=C["blue"]).pack(pady=(20, 2))
        tk.Label(tab, text="Add a person to the recognition database.",
                 fg=C["overlay0"], font=FONT_SMALL).pack(pady=(0, 16))

        # Name
        f = tk.Frame(tab, bg=C["base"])
        f.pack(fill=tk.X, padx=30, pady=4)
        tk.Label(f, text="Name", font=FONT_BOLD).pack(anchor=tk.W)
        self._name_entry = ttk.Entry(f, font=("Segoe UI", 11))
        self._name_entry.pack(fill=tk.X, ipady=5, pady=4)

        self._sep(tab)

        # Camera capture — multi-angle guided
        tk.Label(tab, text="Guided Multi-Angle Capture",
                 font=FONT_BOLD, fg=C["text"]).pack(anchor=tk.W, padx=30)
        tk.Label(tab,
                 text="Start the camera first, then click below.\n"
                      "You will be guided to rotate your head in a full circle\n"
                      "while 60 face samples are captured automatically.",
                 fg=C["overlay0"], font=FONT_SMALL, justify=tk.LEFT).pack(anchor=tk.W, padx=32, pady=(4,0))

        ttk.Button(tab, text="📸  Start Guided Capture",
                   command=self._register_from_camera).pack(pady=14, padx=30, anchor=tk.W)

        # Status
        self._reg_status = tk.Label(tab, text="", font=FONT_NORMAL, fg=C["green"])
        self._reg_status.pack(pady=6)

    # ── Tab: Manage ───────────────────────────────────────────────────────────

    def _build_tab_manage(self, nb):
        tab = ttk.Frame(nb)
        nb.add(tab, text="  Manage Faces  ")

        tk.Label(tab, text="Registered Faces", font=FONT_TITLE,
                 fg=C["blue"]).pack(pady=(20, 10))

        tf = tk.Frame(tab, bg=C["base"])
        tf.pack(fill=tk.BOTH, expand=True, padx=15)

        self._faces_tree = ttk.Treeview(tf, columns=("name", "samples", "folder", "added"),
                                         show="headings", selectmode="browse")
        self._faces_tree.heading("name",    text="Name")
        self._faces_tree.heading("samples", text="Samples")
        self._faces_tree.heading("folder",  text="Image Folder")
        self._faces_tree.heading("added",   text="Registered On")
        self._faces_tree.column("name",    width=130)
        self._faces_tree.column("samples", width=70,  anchor=tk.CENTER)
        self._faces_tree.column("folder",  width=200)
        self._faces_tree.column("added",   width=145)

        vsb = ttk.Scrollbar(tf, orient=tk.VERTICAL, command=self._faces_tree.yview)
        self._faces_tree.configure(yscroll=vsb.set)
        self._faces_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        cf = tk.Frame(tab, bg=C["base"])
        cf.pack(fill=tk.X, padx=15, pady=8)
        ttk.Button(cf, text="Refresh",
                   command=self._refresh_faces_list).pack(side=tk.LEFT, padx=4)
        ttk.Button(cf, text="Delete Selected",
                   style="Danger.TButton",
                   command=self._delete_face).pack(side=tk.LEFT, padx=4)
        self._face_count_lbl = tk.Label(cf, text="",
                                         fg=C["overlay0"], font=FONT_SMALL)
        self._face_count_lbl.pack(side=tk.RIGHT, padx=4)

    # ── Tab: Log ──────────────────────────────────────────────────────────────

    def _build_tab_log(self, nb):
        tab = ttk.Frame(nb)
        nb.add(tab, text="  Recognition Log  ")

        tk.Label(tab, text="Recognition History", font=FONT_TITLE,
                 fg=C["blue"]).pack(pady=(20, 10))

        lf = tk.Frame(tab, bg=C["base"])
        lf.pack(fill=tk.BOTH, expand=True, padx=15)

        self._log_tree = ttk.Treeview(lf, columns=("name", "conf", "time"),
                                       show="headings")
        self._log_tree.heading("name", text="Name")
        self._log_tree.heading("conf", text="Confidence")
        self._log_tree.heading("time", text="Timestamp")
        self._log_tree.column("name", width=130)
        self._log_tree.column("conf", width=90,  anchor=tk.CENTER)
        self._log_tree.column("time", width=160)

        vsb = ttk.Scrollbar(lf, orient=tk.VERTICAL, command=self._log_tree.yview)
        self._log_tree.configure(yscroll=vsb.set)
        self._log_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        cf = tk.Frame(tab, bg=C["base"])
        cf.pack(fill=tk.X, padx=15, pady=8)
        ttk.Button(cf, text="Refresh Log",
                   command=self._refresh_log).pack(side=tk.LEFT, padx=4)
        ttk.Button(cf, text="Clear Log",
                   style="Danger.TButton",
                   command=self._clear_log).pack(side=tk.LEFT, padx=4)

    # ── Tab: Settings ─────────────────────────────────────────────────────────

    def _build_tab_settings(self, nb):
        tab = ttk.Frame(nb)
        nb.add(tab, text="  Settings  ")

        # Scrollable container
        canvas = tk.Canvas(tab, bg=C["base"], highlightthickness=0)
        vsb    = ttk.Scrollbar(tab, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        inner = tk.Frame(canvas, bg=C["base"])
        win   = canvas.create_window((0, 0), window=inner, anchor=tk.NW)
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                    lambda event: canvas.itemconfig(win, width=event.width))
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(-1*(1 if e.delta>0 else -1), "units"))

        # ── Section: Recognition ─────────────────────────────────────────────
        self._section(inner, "Recognition")

        sf = tk.Frame(inner, bg=C["base"])
        sf.pack(fill=tk.X, padx=30, pady=4)
        sf.columnconfigure(1, weight=1)
        row = 0

        tk.Label(sf, text="Match Tolerance", font=FONT_BOLD).grid(
            row=row, column=0, sticky=tk.W, pady=6)
        tol_row = tk.Frame(sf, bg=C["base"])
        tol_row.grid(row=row, column=1, sticky=tk.W, padx=12)
        self._tol_val_lbl = tk.Label(tol_row, text="0.60", width=5,
                                      fg=C["blue"], font=("Segoe UI", 13, "bold"))
        self._tol_val_lbl.pack(side=tk.LEFT)
        ttk.Scale(tol_row, from_=0.1, to=0.9, orient=tk.HORIZONTAL, length=170,
                  variable=self.tolerance,
                  command=lambda v: self._tol_val_lbl.config(
                      text=f"{float(v):.2f}")).pack(side=tk.LEFT, padx=6)
        row += 1
        tk.Label(sf, text="Lower = stricter  |  Higher = looser",
                 font=FONT_SMALL, fg=C["overlay0"]).grid(
            row=row, column=1, sticky=tk.W, padx=12)
        row += 1

        tk.Label(sf, text="Show Confidence %", font=FONT_BOLD).grid(
            row=row, column=0, sticky=tk.W, pady=6)
        ttk.Checkbutton(sf, variable=self.show_conf).grid(
            row=row, column=1, sticky=tk.W, padx=12)
        row += 1

        # ── Section: Camera ───────────────────────────────────────────────────
        self._section(inner, "Camera")

        cf2 = tk.Frame(inner, bg=C["base"])
        cf2.pack(fill=tk.X, padx=30, pady=4)
        cf2.columnconfigure(1, weight=1)
        row2 = 0

        tk.Label(cf2, text="Camera Index", font=FONT_BOLD).grid(
            row=row2, column=0, sticky=tk.W, pady=6)
        cam_row = tk.Frame(cf2, bg=C["base"])
        cam_row.grid(row=row2, column=1, sticky=tk.W, padx=12)
        for i in range(5):
            ttk.Radiobutton(cam_row, text=str(i),
                            variable=self.cam_index, value=i).pack(side=tk.LEFT, padx=5)
        row2 += 1

        tk.Label(cf2, text="Data Directory", font=FONT_BOLD).grid(
            row=row2, column=0, sticky=tk.W, pady=6)
        tk.Label(cf2, text=os.path.abspath("data/"),
                 fg=C["overlay0"], font=FONT_MONO).grid(
            row=row2, column=1, sticky=tk.W, padx=12)

        # ── Section: Overlays ─────────────────────────────────────────────────
        self._section(inner, "Live Overlays")

        models_ok = self.overlay.models_ready
        model_note = "✓ Models ready" if models_ok else \
                     "⚠ Run  python download_models.py  to enable Age & Gender"
        model_col  = C["green"] if models_ok else C["peach"]
        tk.Label(inner, text=model_note, font=FONT_SMALL,
                 fg=model_col).pack(anchor=tk.W, padx=32, pady=(0, 6))

        ov_items = [
            # (label, var, description, requires_model)
            ("Show Age",        self.ov_age,        "Guess age range (0-2 … 60+)",               True),
            ("Show Gender",     self.ov_gender,     "Predict Male / Female",                     True),
            ("Show Emotion",    self.ov_emotion,    "Happy / Sad / Angry / Surprised / Neutral", False),
            ("Show Confidence", self.show_conf,     "Display match % on the face box",           False),
            ("FPS Counter",     self.ov_fps,        "Frames per second in top-left corner",      False),
            ("Face Count",      self.ov_facecount,  "Number of faces in frame",                  False),
        ]

        pr_items = [
            ("Blur Faces",      self.ov_blur,       "Gaussian privacy blur over face regions",   False),
            ("Pixelate Faces",  self.ov_pixelate,   "Mosaic / pixel effect on faces",            False),
        ]

        ar_items = [
            ("Sunglasses",      self.ov_sunglasses, "Draw AR sunglasses on detected faces",      False),
            ("Party Hat",       self.ov_hat,        "Draw a hat above detected faces",           False),
        ]

        def _ov_row(parent, label, var, desc, needs_model):
            f = tk.Frame(parent, bg=C["base"])
            f.pack(fill=tk.X, padx=30, pady=2)
            cb = ttk.Checkbutton(f, variable=var)
            cb.pack(side=tk.LEFT)
            if needs_model and not models_ok:
                cb.state(["disabled"])
            tk.Label(f, text=label, font=FONT_BOLD, width=16,
                     anchor=tk.W).pack(side=tk.LEFT, padx=6)
            tk.Label(f, text=desc, font=FONT_SMALL,
                     fg=C["overlay0"]).pack(side=tk.LEFT)

        tk.Label(inner, text="Analysis", font=FONT_SMALL,
                 fg=C["blue"]).pack(anchor=tk.W, padx=32, pady=(4, 2))
        for args in ov_items:
            _ov_row(inner, *args)

        tk.Label(inner, text="Privacy", font=FONT_SMALL,
                 fg=C["red"]).pack(anchor=tk.W, padx=32, pady=(10, 2))
        for args in pr_items:
            _ov_row(inner, *args)

        tk.Label(inner, text="Fun / AR", font=FONT_SMALL,
                 fg=C["peach"]).pack(anchor=tk.W, padx=32, pady=(10, 2))
        for args in ar_items:
            _ov_row(inner, *args)

        # Mutual exclusion: blur and pixelate can't both be on
        def _excl_blur(*_):
            if self.ov_blur.get():
                self.ov_pixelate.set(False)
        def _excl_pix(*_):
            if self.ov_pixelate.get():
                self.ov_blur.set(False)
        self.ov_blur.trace_add("write", _excl_blur)
        self.ov_pixelate.trace_add("write", _excl_pix)

        tk.Frame(inner, bg=C["base"], height=20).pack()

    # ── Camera control ────────────────────────────────────────────────────────

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
        self._status_lbl.config(text="⏺  Live", fg=C["green"])

        threading.Thread(target=self._recognition_loop, daemon=True).start()
        self._update_canvas()

    def _stop_camera(self):
        self._running = False
        self.camera.stop()
        self._start_btn.config(state=tk.NORMAL)
        self._stop_btn.config(state=tk.DISABLED)
        self._snap_btn.config(state=tk.DISABLED)
        self._status_lbl.config(text="⏺  Offline", fg=C["red"])
        self._detect_lbl.config(text="—")
        self._show_placeholder()
        self._refresh_log()

    def _recognition_loop(self):
        tick = 0
        while self._running:
            frame = self.camera.get_frame()
            if frame is None:
                time.sleep(0.02)
                continue

            tick += 1
            # Run heavy recognition every 3rd frame to keep UI smooth
            if tick % 3 == 0:
                results   = self.engine.recognize(frame.copy(), tolerance=self.tolerance.get())
                annotated = self.engine.annotate_frame(frame, results)
                annotated = self.overlay.draw(annotated, results)

                names = [r[0] for r in results]
                with self._lock:
                    self._raw_frame = frame
                    self._ann_frame = annotated

                # Update detection banner from the main thread
                text = ", ".join(names) if names else "No faces"
                self.root.after(0, lambda t=text: self._detect_lbl.config(text=t))

                # Log known faces — throttled to once per 5 s per person
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

    def _update_canvas(self):
        if not self._running:
            return
        with self._lock:
            frame = self._ann_frame if self._ann_frame is not None else self._raw_frame

        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb).resize((CANVAS_W, CANVAS_H), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self._canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self._photo_ref = photo  # prevent GC

        self.root.after(30, self._update_canvas)

    def _show_placeholder(self):
        self._canvas.delete("all")
        self._canvas.create_rectangle(0, 0, CANVAS_W, CANVAS_H, fill="#0a0a15")
        self._canvas.create_text(
            CANVAS_W // 2, CANVAS_H // 2,
            text="No Camera Feed\n\nPress  ▶ Start Camera  to begin",
            font=("Segoe UI", 13), fill=C["surface1"], justify=tk.CENTER,
        )

    def _take_snapshot(self):
        frame = self.camera.snapshot()
        if frame is None:
            return
        os.makedirs("data/captures", exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.abspath(f"data/captures/snapshot_{ts}.jpg")
        cv2.imwrite(path, frame)
        messagebox.showinfo("Snapshot Saved", f"Saved to:\n{path}")

    # ── Registration ─────────────────────────────────────────────────────────

    def _browse_image(self):
        path = filedialog.askopenfilename(
            title="Select Photo",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self._sel_img_path = path
        self._file_lbl.config(text=os.path.basename(path), fg=C["text"])
        try:
            img = Image.open(path)
            img.thumbnail((200, 140))
            photo = ImageTk.PhotoImage(img)
            self._preview.config(image=photo, text="")
            self._preview._ref = photo
        except Exception:
            pass

    def _register_from_image(self):
        name = self._name_entry.get().strip()
        if not name:
            self._set_reg_status("Please enter a name.", error=True)
            return
        if not self._sel_img_path:
            self._set_reg_status("Please select an image file.", error=True)
            return
        self._set_reg_status("Processing…", warn=True)
        self.root.update_idletasks()

        enc = self.engine.encode_from_file(self._sel_img_path)
        if enc is None:
            self._set_reg_status("No face detected in the image!", error=True)
            return
        self.engine.register(name, enc)
        self._set_reg_status(f"✓  {name!r} registered successfully!")
        self._name_entry.delete(0, tk.END)
        self._sel_img_path = None
        self._file_lbl.config(text="No file selected", fg=C["overlay0"])
        self._preview.config(image="", text="Preview", fg=C["overlay0"])
        self._refresh_faces_list()

    def _register_from_camera(self):
        name = self._name_entry.get().strip()
        if not name:
            self._set_reg_status("Please enter a name.", error=True)
            return
        if not self._running:
            self._set_reg_status("Start the camera first.", error=True)
            return

        # Sanitise name for use as a folder name
        safe_name = "".join(c for c in name if c.isalnum() or c in " _-").strip()

        def on_complete(captures):
            if not captures:
                self._set_reg_status("Capture cancelled.", warn=True)
                return

            # Save angle images to data/faces/<name>/
            folder = os.path.join("data", "faces", safe_name)
            os.makedirs(folder, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")

            face_arrays = []
            for i, (face_gray, bgr_frame) in enumerate(captures):
                cv2.imwrite(os.path.join(folder, f"sample_{i:03d}_{ts}.jpg"), bgr_frame)
                face_arrays.append(face_gray)

            self.engine.register_multiple(name, face_arrays)
            n = len(captures)
            self._set_reg_status(
                f"✓  {name!r} registered with {n} samples! "
                f"Images saved to data/faces/{safe_name}/"
            )
            self._name_entry.delete(0, tk.END)
            self._refresh_faces_list()

        GuidedCaptureDialog(
            self.root,
            self.camera,
            self.engine.detector,
            name,
            on_complete,
        )

    def _set_reg_status(self, msg: str, error=False, warn=False):
        color = C["red"] if error else (C["peach"] if warn else C["green"])
        self._reg_status.config(text=msg, fg=color)

    # ── Management ────────────────────────────────────────────────────────────

    def _refresh_faces_list(self):
        for row in self._faces_tree.get_children():
            self._faces_tree.delete(row)
        persons = self.db.get_face_names()   # (name, sample_count, first_seen)
        for name, cnt, first_seen in persons:
            safe_name = "".join(c for c in name if c.isalnum() or c in " _-").strip()
            folder = os.path.join("data", "faces", safe_name)
            folder_display = folder if os.path.isdir(folder) else "—"
            self._faces_tree.insert("", tk.END,
                                     values=(name, cnt, folder_display, first_seen[:19]))
        n = len(persons)
        self._face_count_lbl.config(text=f"{n} person{'s' if n != 1 else ''} registered")

    def _delete_face(self):
        sel = self._faces_tree.selection()
        if not sel:
            messagebox.showwarning("No Selection", "Select a person to delete.")
            return
        vals  = self._faces_tree.item(sel[0])["values"]
        name  = vals[0]
        count = vals[1]
        if messagebox.askyesno("Delete Person",
                                f"Remove '{name}' and all {count} samples from the database?"):
            self.db.delete_person(name)
            self.engine.reload()
            self._refresh_faces_list()

    # ── Log ───────────────────────────────────────────────────────────────────

    def _refresh_log(self):
        for row in self._log_tree.get_children():
            self._log_tree.delete(row)
        for name, conf, ts in self.db.get_recognition_log():
            self._log_tree.insert("", tk.END, values=(name, f"{conf:.1%}", ts[:19]))

    def _clear_log(self):
        if messagebox.askyesno("Clear Log", "Delete all recognition history?"):
            self.db.clear_recognition_log()
            self._refresh_log()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _sep(self, parent):
        tk.Frame(parent, bg=C["surface0"], height=1).pack(fill=tk.X, padx=24, pady=10)

    def _section(self, parent, title: str):
        """Bold section heading with a separator line, used in the settings tab."""
        tk.Frame(parent, bg=C["surface0"], height=1).pack(fill=tk.X, padx=16, pady=(14, 4))
        tk.Label(parent, text=title.upper(), font=("Segoe UI", 9, "bold"),
                 bg=C["base"], fg=C["blue"]).pack(anchor=tk.W, padx=20, pady=(0, 4))

    def on_close(self):
        self._running = False
        self.camera.stop()
        self.db.close()
        self.root.destroy()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
