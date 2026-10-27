"""Main window of the UAP Kinematics Analyzer."""

from __future__ import annotations

import os
import sys
from collections import Counter
from typing import Callable, Optional

import numpy as np
from PySide6.QtCore import Qt, QThreadPool, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QMainWindow, QMessageBox,
                               QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSlider, QSpinBox, QSplitter, QTabWidget,
                               QTableWidget, QTableWidgetItem, QToolButton, QVBoxLayout, QWidget)

from .. import __version__
from ..detect import PRESETS, detect_blobs
from ..masks import MASK_KINDS, Mask, mask_image, suggest_static_overlays
from ..project import Project
from ..provenance import ERROR_SOURCES
from ..scenarios import rows_table
from ..tracker import Track
from . import worker
from .fovpanel import FovPanel
from .frameview import FrameView
from .timeline import Timeline

VIDEO_FILTER = "Video files (*.mp4 *.mov *.avi *.mkv *.m4v *.wmv *.mpg *.mpeg *.ts *.webm);;All files (*)"
SCEN_HINT = ("\n\nAccepted forms: a list '1km, 5km, 10nmi', a sweep '2km:40km:2km' or 'log:1km:100km:9'. "
             "Units: m, km, ft, kft, nmi, mi.")


def _canvas(fig):
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    c = FigureCanvasQTAgg(fig)
    w, h = fig.get_size_inches() * fig.dpi
    c.setMinimumSize(int(w * 0.45), int(h * 0.85))
    return c


class FigureHolder(QWidget):
    """Scrollable figure at a readable size, with a pop-out window (zoom/pan toolbar)."""

    def __init__(self, placeholder: str):
        super().__init__()
        self.fig = None
        self.placeholder = placeholder
        self.popup: Optional[QDialog] = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.btn = QPushButton("Open in window (zoom / pan / save)")
        self.btn.setEnabled(False)
        self.btn.clicked.connect(self._pop)
        lay.addWidget(self.btn)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        lay.addWidget(self.scroll, 1)
        self._placeholder(placeholder)

    def _placeholder(self, text: str):
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setWordWrap(True)
        self.scroll.setWidget(lbl)

    def set_figure(self, fig):
        self.fig = fig
        self.scroll.setWidget(_canvas(fig))
        self.btn.setEnabled(True)

    def clear(self, text: Optional[str] = None):
        """Back to the placeholder (results are stale or gone). An open pop-out window keeps its figure."""
        self.fig = None
        self._placeholder(text or self.placeholder)
        self.btn.setEnabled(False)

    def _pop(self):
        if self.fig is None:
            return
        if self.popup is not None:          # already open: bring it forward instead of re-parenting again
            self.popup.raise_()
            self.popup.activateWindow()
            return
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
        fig = self.fig
        dlg = QDialog(self)
        dlg.setAttribute(Qt.WA_DeleteOnClose)
        dlg.setWindowTitle(fig.axes[0].get_title() if fig.axes else "Figure")
        lay = QVBoxLayout(dlg)
        c = _canvas(fig)  # re-parents the figure to the pop-out canvas
        lay.addWidget(NavigationToolbar2QT(c, dlg))
        lay.addWidget(c, 1)
        w, h = fig.get_size_inches() * fig.dpi
        dlg.resize(int(w * 1.1), int(h * 1.1) + 40)
        self.popup = dlg
        self._placeholder("Figure is open in a separate window. Close that window to show it here again.")

        def restore(*_):
            self.popup = None
            if self.fig is fig:          # a newer figure may have replaced it meanwhile: keep that one
                self.set_figure(fig)

        dlg.finished.connect(restore)
        dlg.show()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"UAP Kinematics Analyzer {__version__} (offline)")
        self.resize(1500, 920)
        self.p = Project()
        self.k = 0
        self.job: Optional[worker.Job] = None
        self._job_label = ""
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._play_step)
        self._last_box = (0, 0, 15, 15)
        self._candidates = []
        self._cand_frame = -1
        self._build_ui()
        self._build_menu()
        self.setAcceptDrops(True)
        self._set_enabled(False)

    # ================================================================ UI construction
    def _build_ui(self):
        split = QSplitter(Qt.Horizontal)
        left = QWidget()
        ll = QVBoxLayout(left)
        self.view = FrameView()
        self.view.roiSelected.connect(self._on_roi)
        self.view.maskDrawn.connect(self._on_mask_drawn)
        self.view.pointClicked.connect(self._on_point)
        self.view.fovPointClicked.connect(self._on_fov_point)
        ll.addWidget(self.view, 1)
        self.timeline = Timeline()
        self.timeline.frameClicked.connect(self.goto)
        ll.addWidget(self.timeline)
        row = QHBoxLayout()
        self.btn_prev = QToolButton(text="◀")
        self.btn_prev.clicked.connect(lambda: self.goto(self.k - 1))
        self.btn_play = QToolButton(text="▶")
        self.btn_play.clicked.connect(self._toggle_play)
        self.btn_next = QToolButton(text="▶|")
        self.btn_next.clicked.connect(lambda: self.goto(self.k + 1))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.goto)
        self.spin_frame = QSpinBox()
        self.spin_frame.setToolTip("Frame index within the analysed segment")
        self.spin_frame.valueChanged.connect(self.goto)
        self.lbl_time = QLabel("–")
        self.lbl_time.setMinimumWidth(260)
        self.chk_enh = QCheckBox("Enhance")
        self.chk_enh.setToolTip("Contrast-stretch the display (does not affect analysis)")
        self.chk_enh.toggled.connect(lambda _: self.goto(self.k))
        for w in (self.btn_prev, self.btn_play, self.btn_next):
            row.addWidget(w)
        row.addWidget(self.slider, 1)
        row.addWidget(self.spin_frame)
        row.addWidget(self.lbl_time)
        row.addWidget(self.chk_enh)
        ll.addLayout(row)
        split.addWidget(left)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._tab_video(), "1 Video")
        self.tabs.addTab(self._tab_masks(), "2 Masks")
        self.tabs.addTab(self._tab_track(), "3 Track")
        self.tabs.addTab(self._tab_camera(), "4 Camera")
        self.tabs.addTab(self._tab_kin(), "5 Kinematics")
        self.tabs.addTab(self._tab_scen(), "6 Scenarios")
        self.tabs.addTab(self._tab_export(), "7 Export")
        self.tabs.setMinimumWidth(520)
        split.addWidget(self.tabs)
        split.setSizes([950, 550])
        self.setCentralWidget(split)

        sb = self.statusBar()
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(260)
        self.progress.setVisible(False)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setVisible(False)
        self.btn_cancel.clicked.connect(self.cancel_job)
        sb.addPermanentWidget(self.progress)
        sb.addPermanentWidget(self.btn_cancel)

    def _build_menu(self):
        m = self.menuBar().addMenu("&File")
        for text, fn, key in (("Open video…", self.open_video_dialog, QKeySequence.Open),
                              ("Open project…", self.open_project_dialog, None),
                              ("Save project…", self.save_project_dialog, QKeySequence.Save),
                              ("Export all…", self.export_dialog, None)):
            a = QAction(text, self)
            a.triggered.connect(fn)
            if key:
                a.setShortcut(key)
            m.addAction(a)
        m.addSeparator()
        a = QAction("Quit", self)
        a.triggered.connect(self.close)
        m.addAction(a)
        m = self.menuBar().addMenu("&Tools")
        a = QAction("Generate synthetic test clip (known ground truth)…", self)
        a.triggered.connect(self.synth_dialog)
        m.addAction(a)
        a = QAction("Run full analysis", self)
        a.setShortcut("Ctrl+R")
        a.triggered.connect(self.run_all)
        m.addAction(a)
        m = self.menuBar().addMenu("&Help")
        a = QAction("Method and labels…", self)
        a.triggered.connect(self._help)
        m.addAction(a)

    def _tab_video(self):
        w = QWidget()
        l = QVBoxLayout(w)
        row = QHBoxLayout()
        b = QPushButton("Open video…")
        b.clicked.connect(self.open_video_dialog)
        row.addWidget(b)
        row.addWidget(QLabel("Segment (s):"))
        self.sp_seg0 = QDoubleSpinBox(minimum=0, maximum=10**6, decimals=2, value=0.0)
        self.sp_seg1 = QDoubleSpinBox(minimum=0, maximum=10**6, decimals=2, value=0.0)
        self.sp_seg1.setToolTip("0 = to the end of the file")
        row.addWidget(self.sp_seg0)
        row.addWidget(QLabel("to"))
        row.addWidget(self.sp_seg1)
        self.btn_reload = QPushButton("Reload segment")
        self.btn_reload.setToolTip("Re-decode only this part of the file (long videos: analyse the part you need). "
                                   "Masks and keyframes are kept where they fall inside the new segment.")
        self.btn_reload.clicked.connect(self.reload_segment)
        row.addWidget(self.btn_reload)
        l.addLayout(row)
        self.txt_meta = QPlainTextEdit(readOnly=True)
        l.addWidget(self.txt_meta, 1)
        l.addWidget(QLabel("Workflow: 2 mask overlays → 3 mark the object → Run full analysis (Ctrl+R) → "
                           "5 enter the FOV → 6 scenario assumptions → 7 export."))
        self.btn_run = QPushButton("Run full analysis (track → camera → re-track → kinematics)")
        self.btn_run.clicked.connect(self.run_all)
        l.addWidget(self.btn_run)
        return w

    def _tab_masks(self):
        w = QWidget()
        l = QVBoxLayout(w)
        l.addWidget(QLabel("Masked regions are ignored by camera registration and blob detection, and neutralised "
                           "inside tracking templates. Mask HUD text, reticles, watermarks and redaction blocks."))
        row = QHBoxLayout()
        self.cmb_mask_kind = QComboBox()
        self.cmb_mask_kind.addItems(MASK_KINDS)
        self.btn_draw_mask = QPushButton("Draw mask")
        self.btn_draw_mask.setCheckable(True)
        self.btn_draw_mask.toggled.connect(lambda on: self._set_mode("mask" if on else "none"))
        row.addWidget(self.cmb_mask_kind)
        row.addWidget(self.btn_draw_mask)
        l.addLayout(row)
        b = QPushButton("Suggest static overlays (needs a moving camera)")
        b.clicked.connect(self._suggest_masks)
        l.addWidget(b)
        self.lst_masks = QListWidget()
        self.lst_masks.setSelectionMode(QListWidget.ExtendedSelection)
        l.addWidget(self.lst_masks, 1)
        row = QHBoxLayout()
        b = QPushButton("Delete selected")
        b.clicked.connect(self._delete_mask)
        row.addWidget(b)
        b = QPushButton("Clear all")
        b.clicked.connect(self._clear_masks)
        row.addWidget(b)
        l.addLayout(row)
        return w

    def _tab_track(self):
        w = QWidget()
        l = QVBoxLayout(w)
        g = QGroupBox("Objects")
        gl = QHBoxLayout(g)
        self.cmb_obj = QComboBox()
        self.cmb_obj.setToolTip("Active object: it drives the kinematics text, the scenario table and the overlays")
        self.cmb_obj.currentIndexChanged.connect(self._select_object)
        self.cmb_role = QComboBox()
        self.cmb_role.addItems(["target", "control"])
        self.cmb_role.setToolTip("Role of the active object. control = a feature fixed to the background (cloud, "
                                 "coastline, surface feature). Its camera-compensated motion should be zero, so it "
                                 "measures the compensation error.")
        self.cmb_role.currentTextChanged.connect(self._set_role)
        b_add = QPushButton("Add object")
        b_add.setToolTip("Add another tracked object (a second target or a background control)")
        b_add.clicked.connect(self._add_object)
        b_del = QPushButton("Delete object")
        b_del.clicked.connect(self._delete_object)
        for w_ in (self.cmb_obj, self.cmb_role, b_add, b_del):
            gl.addWidget(w_)
        l.addWidget(g)
        g = QGroupBox("Mark the object")
        gl = QVBoxLayout(g)
        self.btn_mark = QPushButton("Mark object: drag a box around it")
        self.btn_mark.setCheckable(True)
        self.btn_mark.toggled.connect(lambda on: self._set_mode("roi" if on else "none"))
        gl.addWidget(self.btn_mark)
        row = QHBoxLayout()
        self.cmb_pol = QComboBox()
        self.cmb_pol.addItems(["auto", "bright", "dark"])
        self.cmb_preset = QComboBox()
        self.cmb_preset.addItems(list(PRESETS))
        b = QPushButton("Auto-detect blobs")
        b.clicked.connect(self._detect)
        row.addWidget(QLabel("Polarity"))
        row.addWidget(self.cmb_pol)
        row.addWidget(QLabel("Background"))
        row.addWidget(self.cmb_preset)
        row.addWidget(b)
        gl.addLayout(row)
        self.lst_cand = QListWidget()
        self.lst_cand.setMaximumHeight(110)
        self.lst_cand.itemDoubleClicked.connect(self._pick_candidate)
        gl.addWidget(QLabel("Candidates (double-click to use as the object):"))
        gl.addWidget(self.lst_cand)
        l.addWidget(g)

        g = QGroupBox("Manual correction")
        gl = QVBoxLayout(g)
        self.btn_correct = QPushButton("Correct position: click the object in this frame")
        self.btn_correct.setCheckable(True)
        self.btn_correct.toggled.connect(lambda on: self._set_mode("point" if on else "none"))
        gl.addWidget(self.btn_correct)
        row = QHBoxLayout()
        b = QPushButton("Remove keyframe here")
        b.clicked.connect(self._remove_keyframe)
        row.addWidget(b)
        b = QPushButton("Start track here")
        b.setToolTip("Do not track this object before the current frame (it appears or splits off here). "
                     "Click again on the same frame to remove the limit.")
        b.clicked.connect(self._start_here)
        row.addWidget(b)
        b = QPushButton("End track here")
        b.setToolTip("Stop tracking this object after the current frame (it leaves, splits or becomes ambiguous). "
                     "Click again on the same frame to remove the limit.")
        b.clicked.connect(self._end_here)
        row.addWidget(b)
        b = QPushButton("Toggle exclude frame")
        b.setToolTip("Mark this frame unusable (occlusion, glare, interlace tear). Excluded frames are skipped.")
        b.clicked.connect(self._toggle_exclude)
        row.addWidget(b)
        gl.addLayout(row)
        self.lbl_kf = QLabel("Keyframes: none")
        self.lbl_kf.setWordWrap(True)
        gl.addWidget(self.lbl_kf)
        l.addWidget(g)

        g = QGroupBox("Tracker parameters")
        f = QFormLayout(g)
        tp = self.p.track_params
        self.sp_radius = QSpinBox(minimum=4, maximum=400, value=tp.search_radius_px)
        self.sp_lost = QDoubleSpinBox(minimum=0.05, maximum=0.99, singleStep=0.05, value=tp.lost_threshold)
        self.sp_tupd = QDoubleSpinBox(minimum=0.0, maximum=0.5, singleStep=0.01, value=tp.template_update)
        self.chk_centroid = QCheckBox(checked=tp.centroid_refine)
        self.chk_reacq = QCheckBox(checked=tp.reacquire)
        self.cmb_tmode = QComboBox()
        self.cmb_tmode.addItems(["auto", "point", "template"])
        self.cmb_tmode.setToolTip("point: matched filter for objects of a few pixels; template: extended objects")
        self.sp_snr = QDoubleSpinBox(minimum=2.0, maximum=50.0, singleStep=0.5, value=tp.snr_min)
        f.addRow("Matching mode", self.cmb_tmode)
        f.addRow("Point mode: min SNR", self.sp_snr)
        f.addRow("Search radius (px)", self.sp_radius)
        f.addRow("Lost threshold (NCC)", self.sp_lost)
        f.addRow("Template update rate", self.sp_tupd)
        f.addRow("Sub-pixel centroid refine", self.chk_centroid)
        f.addRow("Re-acquire with blob detection", self.chk_reacq)
        l.addWidget(g)
        b = QPushButton("Run tracking only")
        b.setToolTip("Track every object that has keyframes, without estimating camera motion")
        b.clicked.connect(self.run_tracking)
        l.addWidget(b)
        self.lbl_track = QLabel("")
        self.lbl_track.setWordWrap(True)
        l.addWidget(self.lbl_track)
        l.addStretch(1)
        return w

    def _tab_camera(self):
        w = QWidget()
        l = QVBoxLayout(w)
        l.addWidget(QLabel("Background registration estimates pan/tilt/roll/zoom between frames. Fixed-pattern "
                           "noise is removed first so registration cannot lock onto the sensor."))
        g = QGroupBox("Parameters")
        f = QFormLayout(g)
        cp = self.p.camera_params
        self.cmb_fpn = QComboBox()
        self.cmb_fpn.addItems(["auto", "off"])
        self.chk_destripe = QCheckBox(checked=cp.destripe)
        self.sp_smooth = QDoubleSpinBox(minimum=0, maximum=8, singleStep=0.5, value=cp.smooth_sigma_px)
        self.cmb_gop = QComboBox()
        self.cmb_gop.addItems(["auto", "off"])
        self.sp_maxw = QSpinBox(minimum=320, maximum=4096, singleStep=160, value=cp.max_width)
        self.chk_refine = QCheckBox(checked=cp.fpn_refine)
        self.chk_refine.setToolTip("When fixed-pattern noise dominates, re-estimate it from a scene mosaic built "
                                   "with the first motion estimate, then register again (2x time).")
        f.addRow("FPN removal", self.cmb_fpn)
        f.addRow("Column/row destripe", self.chk_destripe)
        f.addRow("Two-pass FPN refinement", self.chk_refine)
        f.addRow("Noise smoothing σ (px)", self.sp_smooth)
        f.addRow("Keyframe (I-frame) anchoring", self.cmb_gop)
        f.addRow("Max registration width (px)", self.sp_maxw)
        l.addWidget(g)
        b = QPushButton("Estimate camera motion")
        b.clicked.connect(self.run_camera)
        l.addWidget(b)
        self.txt_cam = QPlainTextEdit(readOnly=True)
        self.txt_cam.setMaximumHeight(200)
        l.addWidget(self.txt_cam)
        self.fig_cam = FigureHolder("Camera-motion confidence appears here")
        l.addWidget(self.fig_cam, 1)
        return w

    def _tab_kin(self):
        w = QWidget()
        l = QVBoxLayout(w)
        g = QGroupBox("Camera calibration (needed for deg/s)")
        f = QFormLayout(g)
        kp = self.p.kin_params
        self.sp_hfov = QDoubleSpinBox(minimum=0.0, maximum=179.0, decimals=4, singleStep=0.1, value=kp.hfov_deg)
        self.sp_hfov.setToolTip("Horizontal field of view of the FIRST analysed frame. 0 = unknown (pixel units only).")
        self.sp_par = QDoubleSpinBox(minimum=0.3, maximum=3.0, decimals=4, singleStep=0.01, value=kp.pixel_aspect)
        f.addRow("Horizontal FOV at frame 0 (deg)", self.sp_hfov)
        f.addRow("Pixel aspect ratio (w/h)", self.sp_par)
        self.fov_panel = FovPanel(self)
        f.addRow(self.fov_panel)
        self.sp_hfov.valueChanged.connect(self._hfov_edited)
        l.addWidget(g)
        g = QGroupBox("Derivatives")
        f = QFormLayout(g)
        self.sp_win = QSpinBox(minimum=5, maximum=301, singleStep=2, value=kp.window_frames)
        self.sp_order = QSpinBox(minimum=1, maximum=3, value=kp.poly_order)
        self.cmb_timing = QComboBox()
        self.cmb_timing.addItems(["nominal", "container"])
        self.sp_start = QSpinBox(minimum=0, maximum=10**7)
        self.sp_start.setToolTip("First video frame (file numbering) to use in the fits")
        self.sp_end = QSpinBox(minimum=-1, maximum=10**7, value=-1)
        self.sp_end.setToolTip("Last video frame (file numbering) to use in the fits; -1 = to the end")
        self.chk_usecam = QCheckBox(checked=True)
        f.addRow("Window (frames)", self.sp_win)
        f.addRow("Polynomial order", self.sp_order)
        f.addRow("Timing", self.cmb_timing)
        f.addRow("Start frame", self.sp_start)
        f.addRow("End frame (-1 = last)", self.sp_end)
        f.addRow("Compensate camera motion", self.chk_usecam)
        l.addWidget(g)
        b = QPushButton("Compute kinematics")
        b.clicked.connect(self.run_kinematics)
        l.addWidget(b)
        self.txt_kin = QPlainTextEdit(readOnly=True)
        self.txt_kin.setMaximumHeight(180)
        l.addWidget(self.txt_kin)
        sub = QTabWidget()
        self.fig_kin = FigureHolder("Angular velocity / acceleration plots appear here")
        self.fig_path = FigureHolder("Path plots appear here")
        sub.addTab(self.fig_kin, "Rates")
        sub.addTab(self.fig_path, "Path")
        l.addWidget(sub, 1)
        return w

    def _tab_scen(self):
        w = QWidget()
        l = QVBoxLayout(w)
        lab = QLabel("<b>Everything on this tab is ASSUMED.</b> Speed and g need a distance the video does not "
                     "contain. Enter several values (a sweep) to see how results scale. Units: m, km, ft, kft, nmi, mi. "
                     "Lists: <code>1km, 5km, 10nmi</code>; sweeps: <code>2km:40km:2km</code> or "
                     "<code>log:1km:100km:9</code>.")
        lab.setWordWrap(True)
        l.addWidget(lab)
        g = QGroupBox("Assumption")
        f = QFormLayout(g)
        sp = self.p.scenario
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["range", "size", "altitude"])
        self.cmb_mode.currentTextChanged.connect(self._scen_mode_changed)
        self.ed_values = QLineEdit(sp.values)
        self.ed_camalt = QLineEdit(sp.cam_altitude)
        self.sp_elev = QDoubleSpinBox(minimum=-90, maximum=90, decimals=3, value=sp.elevation_deg)
        self.sp_ovx = QDoubleSpinBox(minimum=-2000, maximum=2000, decimals=1, value=sp.observer_vx_ms)
        self.sp_ovy = QDoubleSpinBox(minimum=-2000, maximum=2000, decimals=1, value=sp.observer_vy_ms)
        self.lbl_values = QLabel("Slant ranges")
        f.addRow("Mode", self.cmb_mode)
        f.addRow(self.lbl_values, self.ed_values)
        f.addRow("Camera altitude (altitude mode)", self.ed_camalt)
        f.addRow("LOS elevation, deg (altitude mode)", self.sp_elev)
        f.addRow("Observer velocity across LOS, x (m/s)", self.sp_ovx)
        f.addRow("Observer velocity across LOS, y (m/s)", self.sp_ovy)
        l.addWidget(g)
        b = QPushButton("Compute scenario table")
        b.clicked.connect(self.run_scenarios)
        l.addWidget(b)
        self.tbl_scen = QTableWidget()
        l.addWidget(self.tbl_scen, 1)
        self.txt_scen = QPlainTextEdit(readOnly=True)
        self.txt_scen.setMaximumHeight(110)
        l.addWidget(self.txt_scen)
        self.fig_sweep = FigureHolder("Speed and g versus assumed range appear here")
        l.addWidget(self.fig_sweep, 1)
        return w

    def _tab_export(self):
        w = QWidget()
        l = QVBoxLayout(w)
        l.addWidget(QLabel("Writes: annotated video, per-frame CSV (every column tagged MEASURED / MEASURED+FOV / "
                           "ASSUMED), scenario CSVs, path and rate plots, report (Markdown + self-contained HTML), "
                           "and the project file for reproduction."))
        self.chk_vid = QCheckBox("Annotated video", checked=True)
        l.addWidget(self.chk_vid)
        b = QPushButton("Export all…")
        b.clicked.connect(self.export_dialog)
        l.addWidget(b)
        self.lst_files = QListWidget()
        l.addWidget(self.lst_files, 1)
        return w

    # ================================================================ state helpers
    def _msg(self, kind: str, title: str, text: str, detail: str = "") -> None:
        """Every user notification goes through here (tests replace it). kind: info | warning | error."""
        self.statusBar().showMessage(text.splitlines()[0], 10000)
        box = QMessageBox(self)
        box.setWindowTitle(title)
        box.setText(text[:3000])
        box.setIcon({"info": QMessageBox.Information, "warning": QMessageBox.Warning}.get(kind, QMessageBox.Critical))
        if detail:
            box.setDetailedText(detail)
        box.exec()

    def _status(self, text: str, ms: int = 8000) -> None:
        self.statusBar().showMessage(text, ms)

    def _need_video(self) -> bool:
        if self.p.cache is None:
            self._status("Open a video first (File > Open video…, or drop a file on the window).")
            return False
        return True

    def _set_enabled(self, has_video: bool):
        self.tabs.setEnabled(True)
        for wdg in (self.slider, self.spin_frame, self.btn_prev, self.btn_next, self.btn_play, self.btn_run,
                    self.btn_reload, self.btn_mark, self.btn_correct, self.btn_draw_mask):
            wdg.setEnabled(has_video)

    def _busy(self, on: bool):
        self.progress.setVisible(on)
        self.btn_cancel.setVisible(on)
        self.tabs.setEnabled(not on)
        self.menuBar().setEnabled(not on)
        if on:
            self.progress.setValue(0)

    def _run_job(self, label: str, fn, done, fail: Optional[Callable[[str], None]] = None) -> bool:
        """Run fn(progress, cancel) off the UI thread. Returns False (with feedback) if a job is still running."""
        if self.job is not None:
            self._status(f"Still busy: {self._job_label} Wait for it to finish or press Cancel.")
            QApplication.beep()
            return False
        self._sync_params()
        self._busy(True)
        self._job_label = label
        self.statusBar().showMessage(label)
        job = worker.Job(fn)
        job.signals.progress.connect(lambda f, m: (self.progress.setValue(int(f * 100)),
                                                   self.statusBar().showMessage(m)))

        def ok(res):
            self.job = None
            self._busy(False)
            done(res)

        def failed(msg):
            self.job = None
            self._busy(False)
            first = msg.splitlines()[0]
            if fail is not None:
                fail(first)
            if msg == "Cancelled.":
                self._status(f"Cancelled: {label}")
            else:
                self._msg("error", "Error", f"{label}\n\n{first}", detail=msg)

        job.signals.finished.connect(ok)
        job.signals.failed.connect(failed)
        self.job = worker.start(job)
        return True

    def cancel_job(self):
        if self.job is not None:
            self.job.cancel()
            self._status("Cancelling…")

    def _sync_params(self):
        tp = self.p.track_params
        tp.search_radius_px = self.sp_radius.value()
        tp.lost_threshold = self.sp_lost.value()
        tp.template_update = self.sp_tupd.value()
        tp.centroid_refine = self.chk_centroid.isChecked()
        tp.reacquire = self.chk_reacq.isChecked()
        tp.mode = self.cmb_tmode.currentText()
        tp.snr_min = self.sp_snr.value()
        tp.polarity = self.cmb_pol.currentText()
        tp.preset = self.cmb_preset.currentText()
        cp = self.p.camera_params
        cp.fpn_mode = self.cmb_fpn.currentText()
        cp.destripe = self.chk_destripe.isChecked()
        cp.fpn_refine = self.chk_refine.isChecked()
        cp.smooth_sigma_px = self.sp_smooth.value()
        cp.gop_mode = self.cmb_gop.currentText()
        cp.max_width = self.sp_maxw.value()
        kp = self.p.kin_params
        kp.hfov_deg = self.sp_hfov.value()
        kp.pixel_aspect = self.sp_par.value()
        kp.window_frames = self.sp_win.value() | 1
        kp.poly_order = self.sp_order.value()
        kp.timing = self.cmb_timing.currentText()
        kp.start_frame = self.sp_start.value()
        kp.end_frame = self.sp_end.value()
        kp.use_camera = self.chk_usecam.isChecked()
        sp = self.p.scenario
        sp.mode = self.cmb_mode.currentText()
        sp.values = self.ed_values.text()
        sp.cam_altitude = self.ed_camalt.text()
        sp.elevation_deg = self.sp_elev.value()
        sp.observer_vx_ms = self.sp_ovx.value()
        sp.observer_vy_ms = self.sp_ovy.value()

    def _load_params_to_ui(self):
        tp, cp, kp, sp = self.p.track_params, self.p.camera_params, self.p.kin_params, self.p.scenario
        self.sp_radius.setValue(tp.search_radius_px)
        self.sp_lost.setValue(tp.lost_threshold)
        self.sp_tupd.setValue(tp.template_update)
        self.chk_centroid.setChecked(tp.centroid_refine)
        self.chk_reacq.setChecked(tp.reacquire)
        self.cmb_tmode.setCurrentText(tp.mode)
        self.sp_snr.setValue(tp.snr_min)
        self.sp_seg0.setValue(self.p.segment_start_s)
        self.sp_seg1.setValue(self.p.segment_end_s)
        self.cmb_pol.setCurrentText(tp.polarity)
        self.cmb_preset.setCurrentText(tp.preset)
        self.cmb_fpn.setCurrentText(cp.fpn_mode)
        self.chk_destripe.setChecked(cp.destripe)
        self.chk_refine.setChecked(cp.fpn_refine)
        self.sp_smooth.setValue(cp.smooth_sigma_px)
        self.cmb_gop.setCurrentText(cp.gop_mode)
        self.sp_maxw.setValue(cp.max_width)
        self.sp_hfov.setValue(kp.hfov_deg)
        self.sp_par.setValue(kp.pixel_aspect)
        self.sp_win.setValue(kp.window_frames)
        self.sp_order.setValue(kp.poly_order)
        self.cmb_timing.setCurrentText(kp.timing)
        # start/end are file frame numbers: lift the range before setting, or the values get clamped
        self.sp_start.setMaximum(10**7)
        self.sp_end.setMaximum(10**7)
        self.sp_start.setValue(kp.start_frame)
        self.sp_end.setValue(kp.end_frame)
        self.chk_usecam.setChecked(kp.use_camera)
        # the mode change handler fills in example values, so the saved text must be set afterwards
        self.cmb_mode.setCurrentText(sp.mode)
        self.ed_values.setText(sp.values)
        self.ed_camalt.setText(sp.cam_altitude)
        self.sp_elev.setValue(sp.elevation_deg)
        self.sp_ovx.setValue(sp.observer_vx_ms)
        self.sp_ovy.setValue(sp.observer_vy_ms)
        self.fov_panel.refresh()

    def _set_mode(self, mode: str):
        for btn, m in ((self.btn_draw_mask, "mask"), (self.btn_mark, "roi"), (self.btn_correct, "point")):
            if m != mode and btn.isChecked():
                btn.blockSignals(True)
                btn.setChecked(False)
                btn.blockSignals(False)
        if mode != "fovpoint" and getattr(self, "fov_panel", None) is not None:
            self.fov_panel.cancel_pick()
        self.view.mode = mode
        self.view.update()
        if mode != "none":
            self.view.setFocus()

    # ---------------------------------------------------------------- result displays
    def _clear_camera_display(self):
        self.txt_cam.clear()
        self.fig_cam.clear()

    def _clear_kin_display(self, why: str = ""):
        self.txt_kin.setPlainText(why)
        self.fig_kin.clear()
        self.fig_path.clear()

    def _clear_scen_display(self, why: str = ""):
        self.tbl_scen.clear()
        self.tbl_scen.setRowCount(0)
        self.tbl_scen.setColumnCount(0)
        self.txt_scen.setPlainText(why)
        self.fig_sweep.clear()

    def _invalidate(self, camera: bool = False, why: str = ""):
        """Results no longer match the inputs (masks, keyframes, roles): drop them and their displays."""
        if camera:
            self.p.camera = None
            self._clear_camera_display()
        self.p.kins = []
        self.p.scen = None
        self._clear_kin_display(why)
        self._clear_scen_display(why)
        self.timeline.set_data(self.timeline.n, self.timeline.status, None, self.timeline.keyframes)

    def _drop_active_results(self, why: str = ""):
        """Kinematics and scenarios of the active object are stale; the other objects' results stay."""
        p = self.p
        if p.kins and p.active < len(p.kins):
            p.kins[p.active] = None
        p.scen = None
        self._clear_kin_display(why)
        self._clear_scen_display(why)

    def _recompute_results(self):
        """After a (re)track: refresh kinematics for every object and the scenario table if they existed."""
        p = self.p
        had_kin, had_scen = bool(p.kins), p.scen is not None
        if had_kin:
            p.run_kinematics()
        if had_scen:
            p.scen = None
            if p.kin is not None and p.kin.has_angles():
                try:
                    p.run_scenarios()
                except ValueError as e:
                    p.scen_error = str(e)
        self._show_kin()
        self._show_scen()

    # ================================================================ video / navigation
    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls() and any(u.isLocalFile() for u in ev.mimeData().urls()):
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        urls = [u for u in ev.mimeData().urls() if u.isLocalFile()]
        if urls:
            ev.acceptProposedAction()
            self.open_path(urls[0].toLocalFile())

    def open_path(self, path: str):
        if path.lower().endswith(".json"):
            self.open_project(path)
        else:
            self.open_video(path)

    def open_video_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open video", "", VIDEO_FILTER)
        if path:
            self.open_video(path)

    def reload_segment(self):
        if not self._need_video():
            return
        self.open_video(self.p.video_path, keep_project=True)

    def open_video(self, path: str, keep_project: bool = False, project: Optional[Project] = None) -> bool:
        """Decode `path` (the segment from the spin boxes) off the UI thread.

        keep_project: re-decode for the current project (segment change); keyframes, exclusions and start/end are
        remapped to the new segment and kept where they fall inside it. project: a loaded project file to attach
        the video to. The window's state is only replaced once decoding succeeded, so a failed open (bad file,
        cancelled) leaves the previous video usable.
        """
        if not os.path.isfile(path):
            self._msg("warning", "Open video", f"File not found:\n{path}")
            return False
        self._stop_play()
        if project is not None:
            p = project
        elif keep_project:
            p = self.p
        else:
            p = Project(camera_params=self.p.camera_params, track_params=self.p.track_params,
                        kin_params=self.p.kin_params, scenario=self.p.scenario)
        old_seg = (p.segment_start_s, p.segment_end_s)
        new_seg = (self.sp_seg0.value(), self.sp_seg1.value()) if project is None else old_seg
        if new_seg[1] and new_seg[1] <= new_seg[0]:
            self._msg("warning", "Segment", "The segment end must be after its start (0 = to the end of the file).")
            return False
        was_open = p.cache is not None

        def fn(prog, cancel):
            old_tracks, old_offset = list(p.tracks), (p.cache.offset if p.cache is not None else 0)
            p.segment_start_s, p.segment_end_s = new_seg
            try:
                warnings = p.open_video(path, prog, cancel)
            except BaseException:
                p.segment_start_s, p.segment_end_s = old_seg
                raise
            if was_open and old_seg != new_seg and old_tracks:
                kept = _remap_tracks(old_tracks, old_offset, p.cache.offset, len(p.cache))
                warnings = [w for w in warnings if not w.startswith("Saved tracks do not match")]
                if kept:
                    p.tracks, p.active = kept, 0
                    nk = sum(len(t.keyframes) for t in kept)
                    warnings.append(f"Segment changed: {nk} keyframe(s) of {len(kept)} object(s) remapped to the "
                                    "new segment; tracks must be re-run.")
                else:
                    warnings.append("Segment changed: no keyframes fall inside the new segment; objects reset.")
            return warnings

        def done(warnings):
            if p is not self.p and self.p.cache is not None:
                self.p.cache.close()
            self.p = p
            if project is not None:
                self._load_params_to_ui()
            self._video_opened(warnings)

        def fail(_msg):
            self.sp_seg0.setValue(old_seg[0] if was_open else self.sp_seg0.value())
            self.sp_seg1.setValue(old_seg[1] if was_open else self.sp_seg1.value())

        label = "Reloading segment…" if keep_project else f"Decoding {os.path.basename(path)}…"
        return self._run_job(label, fn, done, fail)

    def _video_opened(self, warnings):
        p = self.p
        n = len(p.cache)
        self.k = 0
        self.slider.setRange(0, n - 1)
        self.spin_frame.setRange(0, n - 1)
        # kinematics start/end are file frame numbers (the cache may be a segment)
        self.sp_start.setRange(0, p.file_frame(n - 1))
        self.sp_end.setRange(-1, p.file_frame(n - 1))
        self.sp_seg0.setValue(p.segment_start_s)
        self.sp_seg1.setValue(p.segment_end_s)
        lines = p.meta.summary_lines()
        if p.cache.offset or p.segment_end_s:
            lines.append(f"Segment: frames {p.file_frame(0)}–{p.file_frame(n - 1)} of the file ({n} frames)")
        lines.append(f"Duplicate frames: {int(p.cache.duplicate.sum())}")
        if warnings:
            lines += ["", "WARNINGS:"] + [f"- {w}" for w in warnings]
        self.txt_meta.setPlainText("\n".join(lines))
        self._candidates, self._cand_frame = [], -1
        self.lst_cand.clear()
        self.lst_files.clear()
        self._set_enabled(True)
        self._set_mode("none")
        self.p.scen = None
        self.p.kins = []
        self._clear_camera_display()
        self._clear_kin_display()
        self._clear_scen_display()
        self._refresh_masks()
        self._refresh_objects()
        self._refresh_track_info()
        self.fov_panel.refresh()
        self.goto(0)
        self.view.zoom_fit()
        msg = f"Loaded {os.path.basename(p.video_path)}: {n} frames"
        if warnings:
            msg += f" ({len(warnings)} warning(s), see the Video tab)"
        if any(t.keyframes for t in p.tracks):
            msg += ". Keyframes present: Run full analysis (Ctrl+R) to compute results."
        self._status(msg, 15000)

    def goto(self, k: int):
        if self.p.cache is None:
            return
        k = int(np.clip(k, 0, len(self.p.cache) - 1))
        self.k = k
        for w in (self.slider, self.spin_frame):
            w.blockSignals(True)
            w.setValue(k)
            w.blockSignals(False)
        self.view.set_frame(self.p.cache[k], self.chk_enh.isChecked())
        self.timeline.set_current(k)
        t = self.p.frame_times()[k]
        info = f"file frame {self.p.file_frame(k)}  t = {t:.3f} s"
        if self.p.cache.duplicate[k]:
            info += "  DUPLICATE"
        if self.p.cache.blank[k]:
            info += "  BLANK"
        tr = self.p.track
        st = tr.status[k] if tr is not None else "NONE"
        info += f"  track {st}"
        if self.p.camera is not None:
            info += f"  cam {self.p.camera.pairs[k].confidence:.2f}"
        self.lbl_time.setText(info)
        self._update_overlays()

    def _update_overlays(self):
        v, k, tr = self.view, self.k, self.p.track
        v.masks = self.p.masks
        v.box = None
        v.trail = None
        v.keyframe_here = tr is not None and k in tr.keyframes
        if tr is not None and tr.status[k] != "NONE" and np.isfinite(tr.x[k]):
            v.box = (tr.x[k], tr.y[k], tr.w[k], tr.h[k], tr.status[k])
        elif tr is not None and k in tr.keyframes:
            cx, cy, w, h = tr.keyframes[k]
            v.box = (cx, cy, w, h, "MANUAL")
        if tr is not None:
            lo = max(0, k - 45)
            idx = [j for j in range(lo, k + 1) if tr.status[j] in ("OK", "MANUAL", "REACQ")]
            if len(idx) > 1:
                pts = np.stack([tr.x[idx], tr.y[idx]], 1)
                cam = self.p.camera
                if cam is not None:  # re-project past positions into this frame (path relative to background)
                    Ak_inv = np.linalg.inv(cam.A[k])
                    ref = np.einsum("nij,nj->ni", cam.A[idx][:, :2, :2], pts) + cam.A[idx][:, :2, 2]
                    pts = ref @ Ak_inv[:2, :2].T + Ak_inv[:2, 2]
                v.trail = pts
        v.candidates = [(c.x, c.y, f"#{i + 1}") for i, c in enumerate(self._candidates)] if self._cand_frame == k else []
        v.others = [(t.x[k], t.y[k], f"{t.name}" + (" (control)" if t.role == "control" else ""))
                    for i, t in enumerate(self.p.tracks)
                    if i != self.p.active and t.status[k] not in ("NONE",) and np.isfinite(t.x[k])]
        v.fov_points = self.fov_panel.overlay_points(k)
        v.update()

    def _stop_play(self):
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.btn_play.setText("▶")

    def _toggle_play(self):
        if self.play_timer.isActive():
            self._stop_play()
        elif self.p.cache is not None:
            self.play_timer.start(int(1000 / max(self.p.meta.fps, 1)))
            self.btn_play.setText("❚❚")

    def _play_step(self):
        if self.p.cache is None or self.k >= len(self.p.cache) - 1:
            self._stop_play()
        else:
            self.goto(self.k + 1)

    def keyPressEvent(self, ev):
        if ev.key() in (Qt.Key_Right, Qt.Key_D, Qt.Key_Period):
            self.goto(self.k + 1)
        elif ev.key() in (Qt.Key_Left, Qt.Key_A, Qt.Key_Comma):
            self.goto(self.k - 1)
        elif ev.key() == Qt.Key_Home:
            self.goto(0)
        elif ev.key() == Qt.Key_End and self.p.cache is not None:
            self.goto(len(self.p.cache) - 1)
        elif ev.key() == Qt.Key_Space:
            self._toggle_play()
        elif ev.key() == Qt.Key_Escape:
            self._set_mode("none")
        else:
            super().keyPressEvent(ev)

    # ================================================================ masks
    def _on_mask_drawn(self, x, y, w, h):
        self.p.masks.append(Mask(x, y, w, h, self.cmb_mask_kind.currentText()))
        self._masks_changed()
        self._status(f"Mask added ({self.cmb_mask_kind.currentText()} {w}x{h} px). Draw another or untoggle "
                     "'Draw mask'.")

    def _masks_changed(self):
        if self.p.camera is not None or self.p.kins:
            self._invalidate(camera=True, why="Masks changed: run the analysis again.")
            self._status("Masks changed: camera motion and kinematics discarded. Run the analysis again.")
        self._refresh_masks()

    def _refresh_masks(self):
        self.lst_masks.clear()
        for m in self.p.masks:
            self.lst_masks.addItem(f"{m.kind:<10} x={m.x} y={m.y} w={m.w} h={m.h} {m.label}")
        self._update_overlays()

    def _delete_mask(self):
        rows = sorted((i.row() for i in self.lst_masks.selectedIndexes()), reverse=True)
        if not rows:
            self._status("Select a mask in the list first.")
            return
        for r in rows:
            del self.p.masks[r]
        self._masks_changed()

    def _clear_masks(self):
        if not self.p.masks:
            self._status("No masks to clear.")
            return
        self.p.masks.clear()
        self._masks_changed()

    def _suggest_masks(self):
        if not self._need_video():
            return
        excl_box = None
        if self.p.track and self.p.track.keyframes:
            cx, cy, w, h = next(iter(self.p.track.keyframes.values()))
            excl_box = (int(cx - w), int(cy - h), int(2 * w), int(2 * h))
        sugg, msg = suggest_static_overlays(self.p.cache.frames, exclude_box=excl_box)
        if sugg:
            self.p.masks += sugg
            self._masks_changed()
            self.tabs.setCurrentIndex(1)
        self._status(msg + (" Review them; delete any that cover sky/scene." if sugg else ""), 12000)

    # ================================================================ tracking
    def _on_roi(self, cx, cy, w, h):
        self.btn_mark.setChecked(False)
        tr = self.p.track
        if tr is None:
            return
        box = (cx, cy, max(w, 5), max(h, 5))
        tr.keyframes[self.k] = box
        tr.excluded.discard(self.k)          # marking the object here means the frame is usable
        self._last_box = box
        self._refresh_track_info()
        if (tr.status != "NONE").any():          # already tracked: re-track from the new keyframe
            self.run_tracking(quiet=True)
        else:
            self._drop_active_results()
            self._update_overlays()
            self._status(f"{tr.name} marked at frame {self.p.file_frame(self.k)}. Run full analysis (Ctrl+R) or "
                         "'Run tracking only'.")

    def _on_point(self, x, y):
        tr = self.p.track
        if tr is None:
            return
        if np.isfinite(tr.w[self.k]):
            w, h = tr.w[self.k], tr.h[self.k]
        else:
            w, h = self._last_box[2], self._last_box[3]
        tr.keyframes[self.k] = (x, y, w, h)
        tr.excluded.discard(self.k)
        self._refresh_track_info()
        if (tr.status != "NONE").any():
            self.run_tracking(quiet=True)
        else:
            self._update_overlays()
            self._status(f"Keyframe set at frame {self.p.file_frame(self.k)} ({x:.1f}, {y:.1f}). Run full analysis "
                         "(Ctrl+R) or 'Run tracking only'.")

    # ---------------------------------------------------------------- derived field of view (fovpanel.py)
    def _on_fov_point(self, x, y):
        self.fov_panel.on_point(x, y)

    def _hfov_edited(self, v: float):
        """The spin box is the HFOV: write it through so fov_status() sees a hand edit as 'manual' at once."""
        self.p.kin_params.hfov_deg = float(v)
        self.fov_panel.refresh()

    def apply_fov_estimate(self, est, index: Optional[int] = None) -> bool:
        """Use a derived FOV estimate as the HFOV (Project.apply_fov), update the spin box and the status, and
        recompute kinematics / scenarios the way a hand-entered HFOV change does."""
        try:
            warnings = self.p.apply_fov(est, index)
        except ValueError as e:
            self._msg("warning", "Apply FOV", str(e))
            return False
        self.sp_hfov.setValue(self.p.kin_params.hfov_deg)     # after apply_fov: the write-through keeps 'derived'
        self._sync_params()
        if self.p.kins:
            self._recompute_results()
        self.fov_panel.refresh()
        if warnings:
            self._msg("warning", "Apply FOV", "\n".join(warnings))
        kind, src = self.p.fov_status()
        self._status(f"HFOV set to {self.p.kin_params.hfov_deg:.4g} deg [{src.get('label') if src else kind}]"
                     + ("; kinematics recomputed." if self.p.kins else "."), 10000)
        return True

    def _remove_keyframe(self):
        tr = self.p.track
        if tr is None or self.k not in tr.keyframes:
            self._status("No keyframe on this frame.")
            return
        del tr.keyframes[self.k]
        self._refresh_track_info()
        if tr.keyframes:
            if (tr.status != "NONE").any():
                self.run_tracking(quiet=True)
            else:
                self._update_overlays()
        else:
            tr.reset_results()
            self._drop_active_results()
            self._refresh_track_info()
            self._update_overlays()
            self._status(f"{tr.name}: no keyframes left; its track was cleared.")

    def _kf_in_range(self, tr: Track, start: int, end: int) -> bool:
        n = tr.n
        last = n - 1 if end < 0 else min(end, n - 1)
        return any(max(0, start) <= k <= last for k in tr.keyframes)

    def _start_here(self):
        tr = self.p.track
        if tr is None or not self._need_video():
            return
        new = -1 if tr.start == self.k else self.k
        if tr.keyframes and not self._kf_in_range(tr, new, tr.end):
            self._msg("warning", "Start track here", f"No keyframe of {tr.name} would lie inside the track's "
                      "start/end range. Place a keyframe after this frame first, or move the start.")
            return
        tr.start = new
        self._status(f"{tr.name}: " + ("no start limit" if tr.start < 0 else
                                       f"starts at video frame {self.p.file_frame(tr.start)}"), 6000)
        if tr.keyframes:
            self.run_tracking(quiet=True)

    def _end_here(self):
        tr = self.p.track
        if tr is None or not self._need_video():
            return
        new = -1 if tr.end == self.k else self.k
        if tr.keyframes and not self._kf_in_range(tr, tr.start, new):
            self._msg("warning", "End track here", f"No keyframe of {tr.name} would lie inside the track's "
                      "start/end range. Place a keyframe before this frame first, or move the end.")
            return
        tr.end = new
        self._status(f"{tr.name}: " + ("no end limit" if tr.end < 0 else
                                       f"ends at video frame {self.p.file_frame(tr.end)}"), 6000)
        if tr.keyframes:
            self.run_tracking(quiet=True)

    def _toggle_exclude(self):
        tr = self.p.track
        if tr is None or not self._need_video():
            return
        if self.k in tr.keyframes and self.k not in tr.excluded:
            self._msg("warning", "Exclude frame", f"Frame {self.p.file_frame(self.k)} is a keyframe of {tr.name}. "
                      "Remove the keyframe first if the object is not usable here.")
            return
        tr.excluded ^= {self.k}
        on = self.k in tr.excluded
        if not on and tr.status[self.k] == "EXCLUDED":
            tr.status[self.k] = "NONE"
        self._refresh_track_info()
        self._status(f"Frame {self.p.file_frame(self.k)} " + ("excluded" if on else "included again") +
                     f" for {tr.name}.")
        if tr.keyframes:
            self.run_tracking(quiet=True)
        else:
            self._update_overlays()

    def _detect(self):
        if not self._need_video():
            return
        self._sync_params()
        pol = self.cmb_pol.currentText()
        excl = mask_image(self.p.masks, self.p.cache[0].shape)
        self._candidates = detect_blobs(self.p.cache[self.k], "both" if pol == "auto" else pol,
                                        self.cmb_preset.currentText(), excl, max_candidates=12)
        self._cand_frame = self.k
        self.lst_cand.clear()
        for i, c in enumerate(self._candidates):
            self.lst_cand.addItem(f"#{i + 1} {c.polarity} at ({c.x:.1f}, {c.y:.1f})  {c.w}x{c.h}px  SNR {c.snr:.1f}")
        self._update_overlays()
        if self._candidates:
            self._status(f"{len(self._candidates)} candidate(s) in this frame: double-click one to mark it.")
        else:
            self._status("No blob candidates in this frame: try the other polarity or background preset, "
                         "or mark the object by hand.")

    def _pick_candidate(self, item):
        i = self.lst_cand.row(item)
        if not 0 <= i < len(self._candidates) or self.p.cache is None:
            return
        if self._cand_frame != self.k:
            self.goto(self._cand_frame)        # candidates belong to the frame they were detected in
        c = self._candidates[i]
        size = max(9, max(c.w, c.h) + 6)
        if self.cmb_pol.currentText() == "auto":
            self.cmb_pol.setCurrentText(c.polarity)
        self._on_roi(c.x, c.y, size, size)

    def _refresh_track_info(self):
        tr = self.p.track
        if tr is None:
            self.lbl_kf.setText("Keyframes: none")
            self.lbl_track.setText("")
            self.timeline.set_data(0)
            return
        kfs = sorted(tr.keyframes)
        ff = self.p.file_frame
        lim = ""
        if tr.start >= 0 or tr.end >= 0:
            lim = (f"   Range: {ff(tr.start) if tr.start >= 0 else 'start'} → "
                   f"{ff(tr.end) if tr.end >= 0 else 'end'}")
        self.lbl_kf.setText(f"[{tr.name}] keyframes (video frames): {[ff(k) for k in kfs] if kfs else 'none'}   "
                            f"Excluded: {[ff(k) for k in sorted(tr.excluded)] if tr.excluded else 'none'}{lim}")
        sc = Counter(tr.status)
        if (tr.status != "NONE").any():
            self.lbl_track.setText(f"[{tr.name}] status: " + ", ".join(f"{k} {v}" for k, v in sorted(sc.items())) +
                                   ". Red/yellow on the timeline = check those frames.")
        else:
            self.lbl_track.setText(f"[{tr.name}] not tracked yet." if kfs else "")
        conf = self.p.camera.confidence() if self.p.camera is not None else None
        self.timeline.set_data(tr.n, tr.status, conf, kfs)

    # ---------------------------------------------------------------- objects
    def _refresh_objects(self):
        self.cmb_obj.blockSignals(True)
        self.cmb_obj.clear()
        for t in self.p.tracks:
            self.cmb_obj.addItem(f"{t.name} ({t.role})")
        self.cmb_obj.setCurrentIndex(self.p.active)
        self.cmb_obj.blockSignals(False)
        self.cmb_role.blockSignals(True)
        if self.p.track is not None:
            self.cmb_role.setCurrentText(self.p.track.role)
        self.cmb_role.blockSignals(False)

    def _select_object(self, i: int):
        if not (0 <= i < len(self.p.tracks)) or i == self.p.active:
            return
        self.p.active = i
        self._refresh_objects()
        self._refresh_track_info()
        # the scenario table belongs to the active object
        p = self.p
        if p.scen is not None:
            p.scen = None
            if p.kin is not None and p.kin.has_angles():
                try:
                    p.run_scenarios()
                except ValueError as e:
                    p.scen_error = str(e)
        self._show_kin()
        self._show_scen()
        self.goto(self.k)
        self._status(f"Active object: {p.track.name} ({p.track.role}).", 5000)

    def _add_object(self):
        if not self._need_video():
            return
        names = {t.name for t in self.p.tracks}
        i = len(self.p.tracks) + 1
        while f"obj{i}" in names:
            i += 1
        self.p.add_track(f"obj{i}", "target")
        self.p.active = len(self.p.tracks) - 1
        if self.p.kins:
            self.p.kins.append(None)
        self.p.scen = None
        self._refresh_objects()
        self._refresh_track_info()
        self._show_kin()
        self._show_scen()
        self.goto(self.k)
        self.btn_mark.setChecked(True)
        self._status(f"Added obj{i} (set its role in the box if it is a background control): drag a box around it.")

    def _delete_object(self):
        p = self.p
        if p.track is None:
            return
        if len(p.tracks) <= 1:
            self._status("The last object cannot be deleted; remove its keyframes instead.")
            return
        name = p.track.name
        del p.tracks[p.active]
        if p.active < len(p.kins):
            del p.kins[p.active]
        p.active = max(0, min(p.active, len(p.tracks) - 1))
        p.scen = None
        self._refresh_objects()
        self._refresh_track_info()
        if p.kins and p.kin is not None and p.kin.has_angles():
            try:
                p.run_scenarios()
            except ValueError as e:
                p.scen_error = str(e)
        self._show_kin()
        self._show_scen()
        self.goto(self.k)
        self._status(f"Deleted {name}. Active object: {p.track.name}.")

    def _set_role(self, role: str):
        tr = self.p.track
        if tr is None or tr.role == role:
            return
        tr.role = role
        if self.p.kin is not None:
            self.p.kin.role = role
        self._refresh_objects()
        self._show_kin()
        self.goto(self.k)
        self._status(f"{tr.name} is now a {role}. Run the full analysis again: targets are excluded from camera "
                     "registration, controls are not.")

    def run_tracking(self, quiet: bool = False):
        if not self._need_video():
            return
        p = self.p
        todo = [p.track] if quiet else [t for t in p.tracks if t.keyframes]
        if not any(t.keyframes for t in todo):
            if not quiet:
                self._msg("info", "Track", "Mark the object first (Track tab: drag a box, or auto-detect).")
            return
        names = ", ".join(t.name for t in todo)
        self._run_job(f"Tracking {names}…", lambda prog, cancel: p.run_tracking(prog, cancel, only_active=quiet),
                      self._after_track)

    def _after_track(self, _):
        self._refresh_track_info()
        self._recompute_results()
        self.goto(self.k)
        tr = self.p.track
        if tr is not None:
            n_ok = int(tr.usable().sum())
            self._status(f"{tr.name}: {n_ok}/{tr.n} frames usable ({tr.mode} mode)."
                         + ("" if n_ok else " The object was not found: check the keyframe and polarity."), 10000)

    # ================================================================ camera / kinematics / scenarios
    def run_camera(self):
        if not self._need_video():
            return
        p = self.p
        self._run_job("Estimating camera motion…", lambda prog, cancel: p.run_camera(prog, cancel),
                      lambda _: self._after_camera())

    def _after_camera(self):
        self._show_camera()
        if self.p.kins:
            self._recompute_results()
        self._status("Camera motion estimated.", 8000)

    def _show_camera(self):
        cam = self.p.camera
        if cam is None:
            self._clear_camera_display()
            return
        c = cam.confidence()[1:]
        fc = Counter(f for pr in cam.pairs for f in pr.flags)
        meth = Counter(pr.method for pr in cam.pairs[1:])
        z = 1 / cam.scale()
        lines = [cam.fpn_info, cam.gop_info,
                 f"Confidence: median {np.median(c):.2f}, min {c.min():.2f}; "
                 f"{int((c < self.p.kin_params.low_conf_threshold).sum())} of {len(c)} pairs below "
                 f"{self.p.kin_params.low_conf_threshold}",
                 "Methods: " + ", ".join(f"{k} {v}" for k, v in meth.items()),
                 f"Zoom range: {z.min():.3f}x – {z.max():.3f}x; roll range {cam.roll_deg().min():.2f} – "
                 f"{cam.roll_deg().max():.2f} deg",
                 "Flags: " + (", ".join(f"{k}×{v}" for k, v in fc.most_common()) or "none")]
        if fc.get("RAW_PHASECORR_LOCKED_TO_SENSOR"):
            lines.append("→ Naive phase correlation on raw frames would have reported ~zero camera motion in "
                         f"{fc['RAW_PHASECORR_LOCKED_TO_SENSOR']} frame pairs (fixed-pattern-noise lock).")
        self.txt_cam.setPlainText("\n".join(lines))
        from ..plots import _clean, _fig, C1, C2, TEXT2
        fig = _fig(6, 2.6)
        ax1, ax2 = fig.subplots(2, 1, sharex=True)
        t = self.p.frame_times()
        ax1.plot(t[1:], c, color=C1)
        ax1.axhline(self.p.kin_params.low_conf_threshold, color=TEXT2, ls="--", lw=1)
        ax1.set_ylabel("confidence")
        ax1.set_title("Camera-motion confidence [MEASURED]")
        steps = np.array([np.hypot(*pr.M[:, 2]) for pr in cam.pairs])
        ax2.plot(t, steps, color=C2)
        ax2.set_ylabel("px/frame")
        ax2.set_title("Background shift per frame [MEASURED]")
        ax2.set_xlabel("time (s)")
        for a in (ax1, ax2):
            _clean(a)
        self.fig_cam.set_figure(fig)
        self._refresh_track_info()
        self.goto(self.k)

    def run_kinematics(self):
        if not self._need_video():
            return
        tr = self.p.track
        if tr is None or not tr.usable().any():
            self._msg("info", "Kinematics", f"Track the object first ({tr.name if tr else 'no object'} has no "
                      "usable tracked frames). Run full analysis (Ctrl+R) or 'Run tracking only'.")
            return
        self._sync_params()
        self.p.run_kinematics()
        if self.p.scen is not None:            # the table was computed from the previous kinematics
            try:
                self.p.run_scenarios()
            except ValueError as e:
                self.p.scen, self.p.scen_error = None, str(e)
        self._show_kin()
        self._show_scen()
        if self.p.kin is not None and not self.p.kin.has_angles():
            self._status("Kinematics in pixel units only. Enter the horizontal FOV for deg/s.", 10000)

    def _show_kin(self):
        kin = self.p.kin
        if kin is None:
            tr = self.p.track
            self._clear_kin_display(f"No kinematics for {tr.name}: run the analysis (Ctrl+R)." if tr else "")
            return
        s = kin.summary()
        lines = [f"[{kin.name}, {kin.role}] Frames used: {s['n_frames_used']} over {s['duration_used_s']:.3f} s; "
                 f"{s['n_frames_trusted']} with trusted camera motion feed the statistics [MEASURED]"]
        if kin.has_angles() and "omega_median_deg_s" in s:
            v, sg, fr = s["omega_max_deg_s"]
            lines += [f"Angular velocity median {s['omega_median_deg_s']:.4g} deg/s [MEASURED+FOV]",
                      f"Angular velocity peak {v:.4g} ± {sg:.2g} deg/s at frame {fr} [MEASURED+FOV]"]
            if "alpha_max_deg_s2" in s:
                v, sg, fr = s["alpha_max_deg_s2"]
                lines.append(f"Angular acceleration peak {v:.4g} ± {sg:.2g} deg/s² at frame {fr}; noise floor "
                             f"{s['alpha_noise_floor_deg_s2']:.3g} deg/s² [MEASURED+FOV]")
            lines.append(f"Total angular travel {s['total_angular_travel_deg']:.4g} deg [MEASURED+FOV]")
        elif not kin.has_angles():
            lines.append("No FOV entered: pixel units only (enter the horizontal FOV above for deg/s).")
        if "screen_speed_median_px_s" in s:
            v, sg, fr = s["screen_speed_max_px_s"]
            lines.append(f"On-screen speed (NOT camera-compensated) median {s['screen_speed_median_px_s']:.4g} px/s, "
                         f"peak {v:.4g} ± {sg:.2g} px/s [MEASURED]")
        turns = kin.turns.get("screen", [])
        if turns:
            lines.append("On-screen turns: " + ", ".join(f"{t['angle_deg']:.0f}°@{t['t']:.1f}s" for t in turns))
            lines.append("Camera-compensated turns: " + (", ".join(f"{t['angle_deg']:.0f}°@{t['t']:.1f}s"
                                                                   for t in kin.turns.get("compensated", [])) or "none"))
        lines += [f"⚠ {w}" for w in kin.warnings]
        self.txt_kin.setPlainText("\n".join(lines))
        from ..plots import kinematics_figure, path_figure
        kins = self.p.kins
        controls = [k for k in kins if k is not None and k.role == "control" and k is not kin]
        self.fig_kin.set_figure(kinematics_figure(kin, controls))
        # path_figure drops objects without kinematics, so the primary index counts only the remaining ones
        present = [i for i, k in enumerate(kins) if k is not None]
        self.fig_path.set_figure(path_figure(kins, self.p.cache[0], present.index(self.p.active)))

    def _scen_mode_changed(self, mode):
        self.lbl_values.setText({"range": "Slant ranges", "size": "Object sizes", "altitude": "Object altitudes"}[mode])
        examples = {"range": "1km, 2km, 5km, 10km, 20km, 50km", "size": "0.5m, 1m, 2m, 5m, 15m",
                    "altitude": "5kft:25kft:5kft"}
        self.ed_values.setText(examples[mode])

    def run_scenarios(self):
        if not self._need_video():
            return
        self._sync_params()
        p = self.p
        if p.kin is None:
            if p.track is None or not p.track.usable().any():
                self._msg("info", "Scenarios", "Track the object and compute kinematics first.")
                return
            p.run_kinematics()
            self._show_kin()
        try:
            p.run_scenarios()
        except ValueError as e:
            p.scen, p.scen_error = None, str(e)
            self._clear_scen_display(f"Not computed: {e}")
            from ..units import parse_length_list
            try:
                parse_length_list(p.scenario.values)
                hint = ""
            except ValueError:               # the values text itself is the problem: show the accepted forms
                hint = SCEN_HINT
            self._msg("warning", "Scenarios", f"{e}{hint}")
            return
        self._show_scen()
        self._status(f"Scenario table for {p.kin.name}: {len(p.scen.rows)} assumption(s) [ASSUMED].", 8000)

    def _show_scen(self):
        res = self.p.scen
        if res is None:
            self._clear_scen_display(f"Not computed: {self.p.scen_error}" if self.p.scen_error else "")
            return
        header, rows = rows_table(res)
        self.tbl_scen.clear()
        self.tbl_scen.setColumnCount(len(header))
        self.tbl_scen.setRowCount(len(rows))
        self.tbl_scen.setHorizontalHeaderLabels([h.replace(" [ASSUMED]", "\n[ASSUMED]") for h in header])
        for i, r in enumerate(rows):
            for j, v in enumerate(r):
                self.tbl_scen.setItem(i, j, QTableWidgetItem(v))
        self.tbl_scen.resizeColumnsToContents()
        self.txt_scen.setPlainText("\n".join([f"[{self.p.kin.name}]"] + list(res.notes)) if self.p.kin else
                                   "\n".join(res.notes))
        from ..plots import sweep_figure
        self.fig_sweep.set_figure(sweep_figure(res))

    # ================================================================ full run / export / projects
    def run_all(self):
        if not self._need_video():
            return
        if not any(t.keyframes for t in self.p.tracks):
            self._msg("info", "Analysis", "Mark the object first (Track tab: drag a box, or auto-detect).")
            self.tabs.setCurrentIndex(2)
            return
        p = self.p

        def fn(prog, cancel):
            p.run_all(prog, cancel)
            return True

        self._run_job("Running full analysis…", fn, self._after_all)

    def _after_all(self, _):
        self._refresh_track_info()
        self._show_camera()
        self._show_kin()
        self._show_scen()
        self.goto(self.k)
        kin = self.p.kin
        if kin is None:
            self.tabs.setCurrentIndex(2)
            self._msg("warning", "Analysis", f"{self.p.track.name} could not be tracked in any frame. Check the "
                      "keyframe position, polarity and the matching mode on the Track tab.")
        elif not kin.has_angles():
            self.tabs.setCurrentIndex(4)
            self._status("Done. Enter the horizontal FOV on the Kinematics tab to get deg/s.", 15000)
        elif self.p.scen is None and self.p.scen_error:
            self.tabs.setCurrentIndex(5)
            self._status(f"Analysis complete. Scenarios not computed: {self.p.scen_error}", 20000)
        else:
            self._status("Analysis complete.", 8000)

    def export_dialog(self):
        if not self._export_ready():
            return
        d = QFileDialog.getExistingDirectory(self, "Export folder")
        if d:
            self.export_to(d)

    def _export_ready(self) -> bool:
        if not self._need_video():
            return False
        if self.p.track is None or not any(t.usable().any() for t in self.p.tracks):
            self._msg("info", "Export", "Run the analysis first.")
            return False
        return True

    def export_to(self, d: str, video: Optional[bool] = None) -> bool:
        """Export everything into folder d (the part of 'Export all…' after the folder was chosen)."""
        if not self._export_ready():
            return False
        self._sync_params()
        p = self.p
        video = self.chk_vid.isChecked() if video is None else video
        from ..export import export_all

        def fn(prog, cancel):
            p.run_kinematics()
            p.scen = None
            if p.kin is not None and p.kin.has_angles():
                try:
                    p.run_scenarios()
                except ValueError as e:
                    p.scen_error = str(e)
            return export_all(p, d, video=video, progress=prog, cancel=cancel)

        def done(files):
            self.lst_files.clear()
            self.lst_files.addItems(files)
            self.tabs.setCurrentIndex(6)
            self._show_kin()
            self._show_scen()
            self._status(f"Exported {len(files)} files to {d}", 10000)

        return self._run_job("Exporting…", fn, done)

    def save_project_dialog(self):
        if not self.p.video_path or self.p.cache is None:
            self._msg("info", "Save project", "Open a video first; the project file records which video it analyses.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save project", "project.json", "Project (*.json)")
        if path:
            self.save_project(path)

    def save_project(self, path: str) -> bool:
        if self.p.cache is None:
            self._status("Open a video first.")
            return False
        self._sync_params()
        try:
            self.p.save(path)
        except OSError as e:
            self._msg("error", "Save project", f"Could not write {path}:\n{e}")
            return False
        self._status(f"Saved {path}", 6000)
        return True

    def open_project_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open project", "", "Project (*.json)")
        if path:
            self.open_project(path)

    def _locate_video_dialog(self, missing: str) -> str:
        self._msg("info", "Locate video", f"Video not found:\n{missing}\nPlease locate it.")
        video, _ = QFileDialog.getOpenFileName(self, "Locate video", os.path.dirname(missing), VIDEO_FILTER)
        return video

    def open_project(self, path: str, locate: Optional[Callable[[str], str]] = None) -> bool:
        """Load a project file and decode its video. `locate(missing_path)` is asked for the video when the
        recorded path does not exist (default: a file dialog); returning '' aborts."""
        try:
            p = Project.load(path)
        except (OSError, ValueError, KeyError, TypeError) as e:
            self._msg("warning", "Open project", f"Could not read project:\n{path}\n{e}")
            return False
        video = p.video_path
        if not video or not os.path.isfile(video):
            # try next to the project file first (projects travel with their footage)
            cand = os.path.join(os.path.dirname(os.path.abspath(path)), os.path.basename(video)) if video else ""
            if cand and os.path.isfile(cand):
                video = cand
            else:
                video = (locate or self._locate_video_dialog)(video)
                if not video:
                    self._status("Project not opened: video not located.")
                    return False
        return self.open_video(video, project=p)

    def synth_dialog(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save synthetic clip", "synthetic_fpn.avi", "AVI (*.avi)")
        if path:
            self.generate_synthetic(path)

    def generate_synthetic(self, path: str) -> bool:
        from ..synth import SynthConfig, generate

        def fn(prog, cancel):
            prog(0.2, "Rendering synthetic clip…")
            return generate(path, SynthConfig(codec="FFV1"))

        def done(truth):
            self._msg("info", "Synthetic clip",
                      f"Wrote {path}\nand {path}.truth.json (ground truth).\n\nHFOV = {truth['config']['hfov_deg']} deg. "
                      f"Object starts near pixel ({truth['object_screen_px'][0][0]:.0f}, "
                      f"{truth['object_screen_px'][0][1]:.0f}). True angular rate "
                      f"{np.median(truth['angular_rate_deg_s']):.4f} deg/s.\nHUD/reticle/redaction masks are listed "
                      "in the truth file; try 'Suggest static overlays'.")
            self.sp_seg0.setValue(0)
            self.sp_seg1.setValue(0)
            self.open_video(path)

        return self._run_job("Generating synthetic clip…", fn, done)

    def _help(self):
        txt = ("<h3>Labels</h3><p><b>MEASURED</b>: from pixels and frame timing only. <b>MEASURED+FOV</b>: "
               "measured motion converted to angles with your FOV input (scales linearly with it). "
               "<b>ASSUMED</b>: depends on an assumed range/size/altitude; scales linearly with it.</p>"
               "<h3>Method</h3><p>Object: template + sub-pixel centroid tracking with camera-motion prediction. "
               "Camera: fixed-pattern-noise subtraction, band-pass, grid features + LK + RANSAC similarity, "
               "per-frame confidence; keyframe anchoring for compression lag. Rates: local weighted polynomial "
               "fits with propagated 1σ (tracking noise + chained registration error as a random walk).</p>"
               "<h3>Keys</h3><p>← → (A/D) step frames, Home/End, Space play/pause, F fit, 1 = 1:1, "
               "wheel zoom, right-drag pan, Esc leaves the drawing mode.</p>"
               "<h3>Error sources</h3><ul>" + "".join(f"<li><b>{s}</b>: {n}</li>" for _, s, n in ERROR_SOURCES)
               + "</ul>")
        QMessageBox.about(self, "Method and labels", txt)

    def closeEvent(self, ev):
        self._stop_play()
        if self.job is not None:
            job, self.job = self.job, None
            job.cancel()
            # the worker still reads the frame cache: let it notice the cancel before the cache is closed,
            # and detach its signals so nothing is delivered to a window that is going away
            for sig in (job.signals.progress, job.signals.finished, job.signals.failed):
                try:
                    sig.disconnect()
                except (RuntimeError, TypeError):
                    pass
            if not QThreadPool.globalInstance().waitForDone(5000):
                super().closeEvent(ev)       # still running: leave the cache to process exit rather than pull it away
                return
        if self.p.cache is not None:
            self.p.cache.close()
            self.p.cache = None
        super().closeEvent(ev)


def _remap_tracks(tracks: list, old_offset: int, new_offset: int, n: int) -> list:
    """Carry keyframes/exclusions/limits over to a re-decoded segment (file frame numbers are stable)."""
    d = old_offset - new_offset
    out = []
    for t in tracks:
        kfs = {k + d: v for k, v in t.keyframes.items() if 0 <= k + d < n}
        if not kfs:
            continue
        nt = Track(n, kfs, {k + d for k in t.excluded if 0 <= k + d < n}, t.name, t.role)
        nt.start = t.start + d if t.start >= 0 and 0 <= t.start + d < n else -1
        nt.end = t.end + d if t.end >= 0 and 0 <= t.end + d < n else -1
        out.append(nt)
    return out


def main(path: Optional[str] = None) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("UAP Kinematics Analyzer")
    w = MainWindow()
    w.show()
    if path:
        QTimer.singleShot(0, lambda: w.open_path(path))
    return app.exec()
