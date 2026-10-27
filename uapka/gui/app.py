"""Main window of the UAP Kinematics Analyzer."""

from __future__ import annotations

import os
import sys
from collections import Counter
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QTimer
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
from ..scenarios import ScenarioSpec, rows_table
from ..tracker import TrackParams
from . import worker
from .frameview import FrameView
from .timeline import Timeline

VIDEO_FILTER = "Video files (*.mp4 *.mov *.avi *.mkv *.m4v *.wmv *.mpg *.mpeg *.ts *.webm);;All files (*)"


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
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.btn = QPushButton("Open in window (zoom / pan / save)")
        self.btn.setEnabled(False)
        self.btn.clicked.connect(self._pop)
        lay.addWidget(self.btn)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        lbl = QLabel(placeholder)
        lbl.setAlignment(Qt.AlignCenter)
        self.scroll.setWidget(lbl)
        lay.addWidget(self.scroll, 1)

    def set_figure(self, fig):
        self.fig = fig
        self.scroll.setWidget(_canvas(fig))
        self.btn.setEnabled(True)

    def _pop(self):
        if self.fig is None:
            return
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
        dlg = QDialog(self)
        dlg.setWindowTitle(self.fig.axes[0].get_title() if self.fig.axes else "Figure")
        lay = QVBoxLayout(dlg)
        c = _canvas(self.fig)  # re-parents the figure to a new canvas
        lay.addWidget(NavigationToolbar2QT(c, dlg))
        lay.addWidget(c, 1)
        w, h = self.fig.get_size_inches() * self.fig.dpi
        dlg.resize(int(w * 1.1), int(h * 1.1) + 40)
        dlg.finished.connect(lambda _: self.set_figure(self.fig))
        dlg.show()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"UAP Kinematics Analyzer {__version__} (offline)")
        self.resize(1500, 920)
        self.p = Project()
        self.k = 0
        self.job: Optional[worker.Job] = None
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._play_step)
        self._last_box = (0, 0, 15, 15)
        self._candidates = []
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
        self.btn_cancel.clicked.connect(lambda: self.job and self.job.cancel())
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
        b = QPushButton("Open video…")
        b.clicked.connect(self.open_video_dialog)
        l.addWidget(b)
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
        f.addRow("Search radius (px)", self.sp_radius)
        f.addRow("Lost threshold (NCC)", self.sp_lost)
        f.addRow("Template update rate", self.sp_tupd)
        f.addRow("Sub-pixel centroid refine", self.chk_centroid)
        f.addRow("Re-acquire with blob detection", self.chk_reacq)
        l.addWidget(g)
        b = QPushButton("Run tracking only")
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
        l.addWidget(g)
        g = QGroupBox("Derivatives")
        f = QFormLayout(g)
        self.sp_win = QSpinBox(minimum=5, maximum=301, singleStep=2, value=kp.window_frames)
        self.sp_order = QSpinBox(minimum=1, maximum=3, value=kp.poly_order)
        self.cmb_timing = QComboBox()
        self.cmb_timing.addItems(["nominal", "container"])
        self.sp_start = QSpinBox(minimum=0, maximum=10**7)
        self.sp_end = QSpinBox(minimum=-1, maximum=10**7, value=-1)
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
    def _set_enabled(self, has_video: bool):
        self.tabs.setEnabled(True)
        for wdg in (self.slider, self.spin_frame, self.btn_prev, self.btn_next, self.btn_play, self.btn_run):
            wdg.setEnabled(has_video)

    def _busy(self, on: bool):
        self.progress.setVisible(on)
        self.btn_cancel.setVisible(on)
        self.tabs.setEnabled(not on)
        self.menuBar().setEnabled(not on)
        if on:
            self.progress.setValue(0)

    def _run_job(self, label: str, fn, done):
        if self.job is not None:
            return
        self._sync_params()
        self._busy(True)
        self.statusBar().showMessage(label)
        job = worker.Job(fn)
        job.signals.progress.connect(lambda f, m: (self.progress.setValue(int(f * 100)),
                                                   self.statusBar().showMessage(m)))

        def ok(res):
            self.job = None
            self._busy(False)
            done(res)

        def fail(msg):
            self.job = None
            self._busy(False)
            self.statusBar().showMessage(msg.splitlines()[0])
            if msg != "Cancelled.":
                QMessageBox.warning(self, "Error", msg[:3000])

        job.signals.finished.connect(ok)
        job.signals.failed.connect(fail)
        self.job = worker.start(job)

    def _sync_params(self):
        tp = self.p.track_params
        tp.search_radius_px = self.sp_radius.value()
        tp.lost_threshold = self.sp_lost.value()
        tp.template_update = self.sp_tupd.value()
        tp.centroid_refine = self.chk_centroid.isChecked()
        tp.reacquire = self.chk_reacq.isChecked()
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
        self.sp_start.setValue(kp.start_frame)
        self.sp_end.setValue(kp.end_frame)
        self.chk_usecam.setChecked(kp.use_camera)
        self.cmb_mode.setCurrentText(sp.mode)
        self.ed_values.setText(sp.values)
        self.ed_camalt.setText(sp.cam_altitude)
        self.sp_elev.setValue(sp.elevation_deg)
        self.sp_ovx.setValue(sp.observer_vx_ms)
        self.sp_ovy.setValue(sp.observer_vy_ms)

    def _set_mode(self, mode: str):
        for btn, m in ((self.btn_draw_mask, "mask"), (self.btn_mark, "roi"), (self.btn_correct, "point")):
            if m != mode and btn.isChecked():
                btn.blockSignals(True)
                btn.setChecked(False)
                btn.blockSignals(False)
        self.view.mode = mode
        self.view.update()

    # ================================================================ video / navigation
    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        urls = ev.mimeData().urls()
        if urls:
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

    def open_video(self, path: str, keep_project: bool = False):
        if not keep_project:
            self.p = Project(camera_params=self.p.camera_params, track_params=self.p.track_params,
                             kin_params=self.p.kin_params, scenario=self.p.scenario)
        p = self.p
        self._run_job("Decoding video…", lambda prog, cancel: p.open_video(path, prog, cancel), self._video_opened)

    def _video_opened(self, warnings):
        n = len(self.p.cache)
        self.slider.setRange(0, n - 1)
        self.spin_frame.setRange(0, n - 1)
        self.sp_start.setMaximum(n - 1)
        self.sp_end.setMaximum(n - 1)
        lines = self.p.meta.summary_lines()
        lines.append(f"Duplicate frames: {int(self.p.cache.duplicate.sum())}")
        if warnings:
            lines += ["", "WARNINGS:"] + [f"- {w}" for w in warnings]
        self.txt_meta.setPlainText("\n".join(lines))
        self._set_enabled(True)
        self._refresh_masks()
        self._refresh_track_info()
        self.goto(0)
        self.statusBar().showMessage(f"Loaded {os.path.basename(self.p.video_path)}: {n} frames", 8000)

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
        info = f"t = {t:.3f} s"
        if self.p.cache.duplicate[k]:
            info += "  DUPLICATE"
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
        v.update()

    _cand_frame = -1

    def _toggle_play(self):
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.btn_play.setText("▶")
        elif self.p.cache is not None:
            self.play_timer.start(int(1000 / max(self.p.meta.fps, 1)))
            self.btn_play.setText("❚❚")

    def _play_step(self):
        if self.k >= len(self.p.cache) - 1:
            self._toggle_play()
        else:
            self.goto(self.k + 1)

    def keyPressEvent(self, ev):
        if ev.key() in (Qt.Key_Right, Qt.Key_D, Qt.Key_Period):
            self.goto(self.k + 1)
        elif ev.key() in (Qt.Key_Left, Qt.Key_A, Qt.Key_Comma):
            self.goto(self.k - 1)
        elif ev.key() == Qt.Key_Space:
            self._toggle_play()
        else:
            super().keyPressEvent(ev)

    # ================================================================ masks
    def _on_mask_drawn(self, x, y, w, h):
        self.p.masks.append(Mask(x, y, w, h, self.cmb_mask_kind.currentText()))
        self._invalidate(camera=True)
        self._refresh_masks()

    def _refresh_masks(self):
        self.lst_masks.clear()
        for m in self.p.masks:
            self.lst_masks.addItem(f"{m.kind:<10} x={m.x} y={m.y} w={m.w} h={m.h} {m.label}")
        self._update_overlays()

    def _delete_mask(self):
        rows = sorted((i.row() for i in self.lst_masks.selectedIndexes()), reverse=True)
        for r in rows:
            del self.p.masks[r]
        self._invalidate(camera=True)
        self._refresh_masks()

    def _clear_masks(self):
        self.p.masks.clear()
        self._invalidate(camera=True)
        self._refresh_masks()

    def _suggest_masks(self):
        if self.p.cache is None:
            return
        excl_box = None
        if self.p.track and self.p.track.keyframes:
            cx, cy, w, h = next(iter(self.p.track.keyframes.values()))
            excl_box = (int(cx - w), int(cy - h), int(2 * w), int(2 * h))
        sugg, msg = suggest_static_overlays(self.p.cache.frames, exclude_box=excl_box)
        self.p.masks += sugg
        self._refresh_masks()
        self.statusBar().showMessage(msg + " Review them; delete any that cover sky/scene.", 10000)

    # ================================================================ tracking
    def _on_roi(self, cx, cy, w, h):
        self.btn_mark.setChecked(False)
        self.p.track.keyframes[self.k] = (cx, cy, max(w, 5), max(h, 5))
        self._last_box = (cx, cy, max(w, 5), max(h, 5))
        self._invalidate()
        self._refresh_track_info()
        self._update_overlays()
        self.statusBar().showMessage("Object marked. Run full analysis (Ctrl+R) or 'Run tracking only'.", 8000)

    def _on_point(self, x, y):
        tr = self.p.track
        if tr is None:
            return
        if np.isfinite(tr.w[self.k]):
            w, h = tr.w[self.k], tr.h[self.k]
        else:
            w, h = self._last_box[2], self._last_box[3]
        tr.keyframes[self.k] = (x, y, w, h)
        self._refresh_track_info()
        if any(tr.status != "NONE"):
            self.run_tracking(quiet=True)
        else:
            self._update_overlays()

    def _remove_keyframe(self):
        tr = self.p.track
        if tr and self.k in tr.keyframes:
            del tr.keyframes[self.k]
            self._refresh_track_info()
            if tr.keyframes:
                self.run_tracking(quiet=True)
            else:
                tr.reset_results()
                self._update_overlays()

    def _toggle_exclude(self):
        tr = self.p.track
        if tr is None:
            return
        tr.excluded ^= {self.k}
        self._refresh_track_info()
        if tr.keyframes:
            self.run_tracking(quiet=True)

    def _detect(self):
        if self.p.cache is None:
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

    def _pick_candidate(self, item):
        i = self.lst_cand.row(item)
        c = self._candidates[i]
        size = max(9, max(c.w, c.h) + 6)
        if self.cmb_pol.currentText() == "auto":
            self.cmb_pol.setCurrentText(c.polarity)
        self._on_roi(c.x, c.y, size, size)

    def _refresh_track_info(self):
        tr = self.p.track
        if tr is None:
            self.lbl_kf.setText("Keyframes: none")
            return
        kfs = sorted(tr.keyframes)
        self.lbl_kf.setText(f"Keyframes (manual positions): {kfs if kfs else 'none'}   "
                            f"Excluded frames: {sorted(tr.excluded) if tr.excluded else 'none'}")
        sc = Counter(tr.status)
        if any(tr.status != "NONE"):
            self.lbl_track.setText("Status: " + ", ".join(f"{k} {v}" for k, v in sorted(sc.items())) +
                                   ". Red/yellow on the timeline = check those frames.")
        conf = self.p.camera.confidence() if self.p.camera is not None else None
        self.timeline.set_data(tr.n, tr.status, conf, kfs)

    def _invalidate(self, camera: bool = False):
        if camera:
            self.p.camera = None
        self.p.kin = None
        self.p.scen = None

    def run_tracking(self, quiet: bool = False):
        if self.p.cache is None or not self.p.track.keyframes:
            if not quiet:
                QMessageBox.information(self, "Track", "Mark the object first (Track tab).")
            return
        p = self.p
        self._run_job("Tracking…", lambda prog, cancel: p.run_tracking(prog, cancel), self._after_track)

    def _after_track(self, _):
        self._refresh_track_info()
        if self.p.kin is not None:
            self.p.run_kinematics()
            self._show_kin()
        self.goto(self.k)

    # ================================================================ camera / kinematics / scenarios
    def run_camera(self):
        if self.p.cache is None:
            return
        p = self.p
        self._run_job("Estimating camera motion…", lambda prog, cancel: p.run_camera(prog, cancel),
                      lambda _: self._show_camera())

    def _show_camera(self):
        cam = self.p.camera
        if cam is None:
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
        if self.p.cache is None or not self.p.track.usable().any():
            QMessageBox.information(self, "Kinematics", "Track the object first.")
            return
        self._sync_params()
        self.p.run_kinematics()
        self._show_kin()

    def _show_kin(self):
        kin = self.p.kin
        if kin is None:
            return
        s = kin.summary()
        lines = [f"Frames used: {s['n_frames_used']} over {s['duration_used_s']:.3f} s; "
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
        lines += [f"⚠ {w}" for w in kin.warnings]
        self.txt_kin.setPlainText("\n".join(lines))
        from ..plots import kinematics_figure, path_figure
        self.fig_kin.set_figure(kinematics_figure(kin))
        self.fig_path.set_figure(path_figure(kin, self.p.cache[0]))

    def _scen_mode_changed(self, mode):
        self.lbl_values.setText({"range": "Slant ranges", "size": "Object sizes", "altitude": "Object altitudes"}[mode])
        examples = {"range": "1km, 2km, 5km, 10km, 20km, 50km", "size": "0.5m, 1m, 2m, 5m, 15m",
                    "altitude": "5kft:25kft:5kft"}
        self.ed_values.setText(examples[mode])

    def run_scenarios(self):
        self._sync_params()
        if self.p.kin is None:
            if self.p.track is None or not self.p.track.usable().any():
                QMessageBox.information(self, "Scenarios", "Track the object and compute kinematics first.")
                return
            self.p.run_kinematics()
            self._show_kin()
        try:
            self.p.run_scenarios()
        except ValueError as e:
            QMessageBox.warning(self, "Scenarios", str(e))
            return
        self._show_scen()

    def _show_scen(self):
        res = self.p.scen
        if res is None:
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
        self.txt_scen.setPlainText("\n".join(res.notes))
        from ..plots import sweep_figure
        self.fig_sweep.set_figure(sweep_figure(res))

    # ================================================================ full run / export / projects
    def run_all(self):
        if self.p.cache is None:
            return
        if not self.p.track.keyframes:
            QMessageBox.information(self, "Analysis", "Mark the object first (Track tab: drag a box, or auto-detect).")
            self.tabs.setCurrentIndex(2)
            return
        p = self.p

        def fn(prog, cancel):
            p.run_all(prog, cancel)
            return True

        self._run_job("Running full analysis…", fn, self._after_all)

    def _after_all(self, _):
        self._show_camera()
        self._show_kin()
        if self.p.scen is not None:
            self._show_scen()
        if not self.p.kin.has_angles():
            self.tabs.setCurrentIndex(4)
            self.statusBar().showMessage("Done. Enter the horizontal FOV on the Kinematics tab to get deg/s.", 15000)
        else:
            self.statusBar().showMessage("Analysis complete.", 8000)

    def export_dialog(self):
        if self.p.cache is None or self.p.track is None or not self.p.track.usable().any():
            QMessageBox.information(self, "Export", "Run the analysis first.")
            return
        d = QFileDialog.getExistingDirectory(self, "Export folder")
        if not d:
            return
        self._sync_params()
        p = self.p
        video = self.chk_vid.isChecked()
        from ..export import export_all

        def fn(prog, cancel):
            p.run_kinematics()
            if p.kin.has_angles():
                try:
                    p.run_scenarios()
                except ValueError:
                    p.scen = None
            return export_all(p, d, video=video, progress=prog, cancel=cancel)

        def done(files):
            self.lst_files.clear()
            self.lst_files.addItems(files)
            self.tabs.setCurrentIndex(6)
            self.statusBar().showMessage(f"Exported {len(files)} files to {d}", 10000)

        self._run_job("Exporting…", fn, done)

    def save_project_dialog(self):
        if not self.p.video_path:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save project", "project.json", "Project (*.json)")
        if path:
            self._sync_params()
            self.p.save(path)
            self.statusBar().showMessage(f"Saved {path}", 6000)

    def open_project_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open project", "", "Project (*.json)")
        if path:
            self.open_project(path)

    def open_project(self, path: str):
        try:
            p = Project.load(path)
        except (OSError, ValueError, KeyError, TypeError) as e:
            QMessageBox.warning(self, "Open project", f"Could not read project: {e}")
            return
        video = p.video_path
        if not os.path.exists(video):
            QMessageBox.information(self, "Locate video", f"Video not found:\n{video}\nPlease locate it.")
            video, _ = QFileDialog.getOpenFileName(self, "Locate video", "", VIDEO_FILTER)
            if not video:
                return
        self.p = p
        self._load_params_to_ui()
        self.open_video(video, keep_project=True)

    def synth_dialog(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save synthetic clip", "synthetic_fpn.avi", "AVI (*.avi)")
        if not path:
            return
        from ..synth import SynthConfig, generate

        def fn(prog, cancel):
            prog(0.2, "Rendering synthetic clip…")
            return generate(path, SynthConfig(codec="FFV1"))

        def done(truth):
            QMessageBox.information(
                self, "Synthetic clip",
                f"Wrote {path}\nand {path}.truth.json (ground truth).\n\nHFOV = {truth['config']['hfov_deg']} deg. "
                f"Object starts near pixel ({truth['object_screen_px'][0][0]:.0f}, "
                f"{truth['object_screen_px'][0][1]:.0f}). True angular rate "
                f"{np.median(truth['angular_rate_deg_s']):.4f} deg/s.\nHUD/reticle/redaction masks are listed in the "
                "truth file; try 'Suggest static overlays'.")
            self.open_video(path)

        self._run_job("Generating…", fn, done)

    def _help(self):
        txt = ("<h3>Labels</h3><p><b>MEASURED</b>: from pixels and frame timing only. <b>MEASURED+FOV</b>: "
               "measured motion converted to angles with your FOV input (scales linearly with it). "
               "<b>ASSUMED</b>: depends on an assumed range/size/altitude; scales linearly with it.</p>"
               "<h3>Method</h3><p>Object: template + sub-pixel centroid tracking with camera-motion prediction. "
               "Camera: fixed-pattern-noise subtraction, band-pass, grid features + LK + RANSAC similarity, "
               "per-frame confidence; keyframe anchoring for compression lag. Rates: local weighted polynomial "
               "fits with propagated 1σ (tracking noise + chained registration error as a random walk).</p>"
               "<h3>Error sources</h3><ul>" + "".join(f"<li><b>{s}</b>: {n}</li>" for _, s, n in ERROR_SOURCES)
               + "</ul>")
        QMessageBox.about(self, "Method and labels", txt)

    def closeEvent(self, ev):
        if self.job is not None:
            self.job.cancel()
        if self.p.cache is not None:
            self.p.cache.close()
        super().closeEvent(ev)


def main(path: Optional[str] = None) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("UAP Kinematics Analyzer")
    w = MainWindow()
    w.show()
    if path:
        QTimer.singleShot(0, lambda: w.open_path(path))
    return app.exec()
