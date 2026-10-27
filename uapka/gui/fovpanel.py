"""Field-of-view derivation in the GUI: the 'Derive FOV' group on the Kinematics tab, the (non-modal) derivation
dialog and the estimate history.

No maths lives here: every estimate comes from uapka.fov. The dialog collects clicked points (FrameView mode
"fovpoint"), the method inputs and the click uncertainty, calls the matching estimate_* function, shows
summary(), notes and the sigma breakdown, and applies the result through Project.apply_fov.

Two library additions are accommodated without depending on them (see available_methods / accepts):
  - an optional `region=(x0, y0, w, h)` keyword (inset image area) on every estimate_* plus `detect_inset(frames)`;
  - `estimate_ground(width, height, points=[((x, y), (lat, lon, alt_m)), ...], point_sigma_m=..., ...)`.
Everything is driven by methods that take plain values (no modal dialogs in the logic path), so the headless
smoke tests can call them directly; user notifications go through MainWindow._msg.
"""

from __future__ import annotations

import inspect
import math
import os
from html import escape
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
                               QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget, QPushButton, QScrollArea,
                               QSizePolicy, QSpinBox, QStackedWidget, QTableWidget, QTableWidgetItem, QTextEdit,
                               QVBoxLayout, QWidget)

from .. import fov
from ..fov import FovEstimate, describe_inputs, load_fov_json, save_fov_json
from ..tracker import refine_centroid
from ..units import parse_length

SNAP_SIGMA_PX = 0.5          # click sigma of a point snapped to a blob centroid
SNAP_BOX_PX = 8              # neighbourhood handed to refine_centroid
SNAP_MAX_SHIFT_PX = 6.0      # a centroid further away than this is another blob, not a refinement

# EXAMPLE only. Secondary sources give these HFOVs for the AN/ASQ-228 ATFLIR pod; the 0.7 deg NAR value is not
# from a primary specification and at least one source says 1.5 deg. It is offered as a pre-filled, editable
# example and stored with the estimate (inputs.table); it is never used unless the analyst keeps it.
HUD_EXAMPLE_TABLE = {"NAR": 0.7, "MFOV": 2.8, "WFOV": 6.0}
HUD_EXAMPLE_SIGMA_PCT = 30.0
HUD_EXAMPLE_NOTE = ("EXAMPLE table (ATFLIR pod, secondary sources): NAR 0.7°, MFOV 2.8°, WFOV 6.0°. The 0.7° NAR "
                    "value is from secondary sources and one source gives 1.5°, so a σ of ~30% is honest. Edit the "
                    "table to match your own source; whatever is here is stored with the estimate as your claim.")

METHOD_HELP = {
    "separation": "Two points with a KNOWN true angle between them (two stars, two surveyed towers, ...).",
    "size": "Two points spanning an object of ASSUMED physical size at an ASSUMED range. The FOV inherits both "
            "assumptions; IR blur/blooming inflate the span and bias it large.",
    "geo": "Camera position (lat, lon, alt) and ≥ 2 landmarks with known coordinates, each clicked in the frame. "
           "≥ 3 landmarks add a consistency check. For coastline / terrain footage.",
    "ground": "≥ 4 ground points with known coordinates, each clicked in the frame; the camera position is solved "
              "together with the FOV.",
    "hud": "Look the HUD zoom / FOV-mode label up in a table you supply. The table is your claim; the label is "
           "the only provenance.",
}
MIN_POINTS = {"separation": 2, "size": 2, "geo": 2, "ground": 4, "hud": 0}
FIXED_POINTS = {"separation", "size"}


# ---------------------------------------------------------------- library introspection
def available_methods() -> list[str]:
    """Methods the installed uapka.fov offers, in display order ('ground' only once estimate_ground exists)."""
    out = ["separation", "size", "geo"]
    if callable(getattr(fov, "estimate_ground", None)):
        out.append("ground")
    out.append("hud")
    return out


def estimator(method: str):
    return getattr(fov, f"estimate_{method}")


def accepts(fn, name: str) -> bool:
    """True if fn takes a keyword called `name` (checked on the live signature, so it works before and after a
    library change adds it)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


def snap_to_blob(img: np.ndarray, x: float, y: float, polarity: str = "auto") -> Optional[tuple[float, float]]:
    """Intensity-weighted centroid of the blob under (x, y), or None if there is no clear blob nearby."""
    pols = ("bright", "dark") if polarity not in ("bright", "dark") else (polarity,)
    best = None
    for pol in pols:
        r = refine_centroid(img, x, y, SNAP_BOX_PX, SNAP_BOX_PX, pol)
        if r is None:
            continue
        d = math.hypot(r[0] - x, r[1] - y)
        if d <= SNAP_MAX_SHIFT_PX and (best is None or d < best[0]):
            best = (d, float(r[0]), float(r[1]))
    return None if best is None else (best[1], best[2])


def parse_region(res) -> Optional[tuple[int, int, int, int]]:
    """Normalise whatever detect_inset returns to (x0, y0, w, h) or None."""
    if res is None:
        return None
    if isinstance(res, dict):
        for keys in (("x0", "y0", "w", "h"), ("x", "y", "w", "h"), ("x0", "y0", "width", "height")):
            if all(k in res for k in keys):
                return tuple(int(round(float(res[k]))) for k in keys)
        return None
    if all(hasattr(res, a) for a in ("x0", "y0", "w", "h")):
        return tuple(int(round(float(getattr(res, a)))) for a in ("x0", "y0", "w", "h"))
    try:
        vals = [float(v) for v in res]
    except (TypeError, ValueError):
        return None
    if len(vals) == 4:
        return tuple(int(round(v)) for v in vals)
    return None


def _geo(lat_s: str, lon_s: str, alt_s: str, what: str) -> tuple[float, float, float]:
    try:
        lat, lon = float(str(lat_s).strip()), float(str(lon_s).strip())
    except ValueError:
        raise ValueError(f"{what}: latitude and longitude must be decimal degrees (got {lat_s!r}, {lon_s!r})")
    if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 360.0):
        raise ValueError(f"{what}: latitude {lat} / longitude {lon} out of range")
    alt = parse_length(str(alt_s).strip() or "0")
    return lat, lon, alt


def _length(text: str, what: str, allow_zero: bool = False) -> float:
    text = str(text).strip()
    if not text:
        if allow_zero:
            return 0.0
        raise ValueError(f"{what} is empty")
    try:
        v = parse_length(text)
    except ValueError as e:
        raise ValueError(f"{what}: {e}")
    if v < 0 or (v == 0 and not allow_zero):
        raise ValueError(f"{what} must be positive (got {text!r})")
    return v


def _float(text: str, what: str) -> float:
    try:
        return float(str(text).strip())
    except ValueError:
        raise ValueError(f"{what}: not a number ({text!r})")


def status_text(project) -> str:
    """One line for the panel: manual / derived / none."""
    kind, src = project.fov_status()
    if kind == "none":
        return "HFOV not set (0 = unknown): type it in above or derive it."
    if kind == "manual":
        return f"manual {project.kin_params.hfov_deg:.4f} deg (user input, no uncertainty given)"
    est = FovEstimate.from_dict(src)
    frame = (src.get("inputs") or {}).get("frame")
    s = f"derived: {est.summary()}"
    if frame is not None:
        s += f" on frame {frame}"
    return s


def _sigma_rows(est: FovEstimate) -> list[tuple[str, float, float]]:
    """(input, sigma share in deg, share of the variance in %) sorted by size."""
    contrib = (est.diagnostics or {}).get("sigma_contributions_deg") or {}
    var = sum(v * v for v in contrib.values())
    rows = [(k, float(v), 100.0 * v * v / var if var > 0 else 0.0) for k, v in contrib.items()]
    return sorted(rows, key=lambda r: -r[1])


def result_html(est: FovEstimate, extra: list[str] = ()) -> str:
    """summary(), notes (alternative-solution / consistency-check notes highlighted) and the sigma breakdown."""
    h = [f"<p><b>{escape(est.summary())}</b></p>"]
    if extra:
        h.append("<ul>" + "".join(f"<li style='color:#b06000'>{escape(n)}</li>" for n in extra) + "</ul>")
    notes = []
    for n in est.notes:
        if "FAILED" in n:
            notes.append(f"<li style='color:#c02020'><b>{escape(n)}</b></li>")
        elif "second, very wide-angle solution" in n or "Consistency check" in n:
            notes.append(f"<li style='color:#b06000'><b>{escape(n)}</b></li>")
        elif "outside the" in n:
            notes.append(f"<li style='color:#c02020'>{escape(n)}</li>")
        else:
            notes.append(f"<li>{escape(n)}</li>")
    if notes:
        h.append("<p><b>Notes</b></p><ul>" + "".join(notes) + "</ul>")
    rows = _sigma_rows(est)
    if rows:
        h.append("<p><b>σ contributions</b> (1σ shift of the HFOV per input, combined in quadrature)</p>"
                 "<table cellpadding='2'><tr><th align='left'>input</th><th align='right'>deg</th>"
                 "<th align='right'>share</th></tr>")
        for k, v, pct in rows:
            h.append(f"<tr><td>{escape(k)}</td><td align='right'>{v:.3g}</td><td align='right'>{pct:.0f}%</td></tr>")
        h.append("</table>")
    elif est.method == "hud":
        h.append(f"<p>σ is the table uncertainty you entered ({(est.inputs or {}).get('sigma_pct', 0):g}%).</p>")
    d = est.diagnostics or {}
    diag = []
    if "focal_px" in d:
        diag.append(f"focal length {d['focal_px']:.1f} px")
    if "small_angle_hfov_deg" in d and np.isfinite(d["small_angle_hfov_deg"]):
        diag.append(f"small-angle shortcut {d['small_angle_hfov_deg']:.4g} deg"
                    + (f" ({d['small_angle_error_pct']:+.2f}%)" if "small_angle_error_pct" in d else ""))
    if "alternative_hfov_deg" in d:
        diag.append(f"alternative (wide) solution {d['alternative_hfov_deg']:.3g} deg")
    if "residual_rms_deg" in d:
        diag.append(f"residual RMS {d['residual_rms_deg']:.3g} deg (clicks alone "
                    f"~{d.get('expected_click_residual_deg', 0):.2g})")
    if diag:
        h.append("<p><b>Diagnostics:</b> " + "; ".join(escape(s) for s in diag) + "</p>")
    for lm in d.get("landmarks") or []:
        h.append(f"<p style='margin:0'>landmark ({lm['px'][0]:.1f}, {lm['px'][1]:.1f}): az {lm['azimuth_deg']:.2f}°, "
                 f"el {lm['elevation_deg']:.2f}°, range {lm['range_m'] / 1000:.2f} km</p>")
    return "".join(h)


# ---------------------------------------------------------------- the dialog
class FovDialog(QDialog):
    """Non-modal: the user can scrub, zoom and pan the frame view while picking points."""

    pickRequested = Signal(int)      # a row wants the next click on the frame view
    pickCancelled = Signal()
    pointsChanged = Signal()

    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.setWindowTitle("Derive the field of view")
        self.setModal(False)
        self.setWindowFlags(Qt.Tool | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
        self.resize(660, 860)
        self.estimate: Optional[FovEstimate] = None
        self.pick_row: Optional[int] = None
        self._snapped: dict[int, bool] = {}
        self._building = False
        self._build()
        self._method_changed(self.cmb_method.currentText())

    # ---------------------------------------------------------- construction
    def _build(self):
        # the body scrolls, so a small screen never squeezes the tables under the buttons
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        body = QWidget()
        scroll.setWidget(body)
        outer.addWidget(scroll)
        lay = QVBoxLayout(body)
        row = QHBoxLayout()
        row.addWidget(QLabel("Method"))
        self.cmb_method = QComboBox()
        self.cmb_method.addItems(available_methods())
        self.cmb_method.currentTextChanged.connect(self._method_changed)
        row.addWidget(self.cmb_method, 1)
        lay.addLayout(row)
        self.lbl_help = QLabel("")
        self.lbl_help.setWordWrap(True)
        lay.addWidget(self.lbl_help)

        self.grp_points = QGroupBox("Points (original video pixels; Pick = click the frame view, Esc cancels)")
        gl = QVBoxLayout(self.grp_points)
        self.tbl_points = QTableWidget(0, 8)
        self.tbl_points.setHorizontalHeaderLabels(["#", "x", "y", "frame", "lat", "lon", "alt", ""])
        self.tbl_points.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_points.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tbl_points.horizontalHeader().setSectionResizeMode(7, QHeaderView.ResizeToContents)
        self.tbl_points.verticalHeader().setVisible(False)
        self.tbl_points.setMinimumHeight(120)
        self.tbl_points.itemChanged.connect(lambda _: None if self._building else self.pointsChanged.emit())
        gl.addWidget(self.tbl_points)
        row = QHBoxLayout()
        self.btn_add = QPushButton("Add point")
        self.btn_add.clicked.connect(lambda: self.add_row())
        self.btn_remove = QPushButton("Remove last")
        self.btn_remove.clicked.connect(self.remove_row)
        self.chk_snap = QCheckBox(f"Snap to blob centroid (σ {SNAP_SIGMA_PX:g} px when it succeeds)")
        self.chk_snap.setChecked(True)
        row.addWidget(self.btn_add)
        row.addWidget(self.btn_remove)
        row.addWidget(self.chk_snap, 1)
        gl.addLayout(row)
        lay.addWidget(self.grp_points)

        self.stack = QStackedWidget()
        self.pages: dict[str, QWidget] = {}
        # separation
        w = QWidget()
        f = QFormLayout(w)
        self.sp_angle = QDoubleSpinBox(minimum=0.0, maximum=179.0, decimals=6, value=0.0)
        self.sp_angle_sigma = QDoubleSpinBox(minimum=0.0, maximum=179.0, decimals=6, value=0.0)
        f.addRow("True angle between the points (deg)", self.sp_angle)
        f.addRow("± σ (deg)", self.sp_angle_sigma)
        self._add_page("separation", w)
        # size
        w = QWidget()
        f = QFormLayout(w)
        self.ed_size = QLineEdit("15m")
        self.ed_size_sigma = QLineEdit("0")
        self.ed_range = QLineEdit("5km")
        self.ed_range_sigma = QLineEdit("0")
        f.addRow("ASSUMED size spanned by the points (m, ft, km, nmi…)", self.ed_size)
        f.addRow("± σ", self.ed_size_sigma)
        f.addRow("ASSUMED range", self.ed_range)
        f.addRow("± σ", self.ed_range_sigma)
        self._add_page("size", w)
        # geo
        w = QWidget()
        f = QFormLayout(w)
        self.ed_cam_lat = QLineEdit("")
        self.ed_cam_lon = QLineEdit("")
        self.ed_cam_alt = QLineEdit("0")
        self.ed_cam_sigma = QLineEdit("0")
        self.ed_lm_sigma = QLineEdit("0")
        f.addRow("Camera latitude (deg)", self.ed_cam_lat)
        f.addRow("Camera longitude (deg)", self.ed_cam_lon)
        f.addRow("Camera altitude (m, ft, …)", self.ed_cam_alt)
        f.addRow("Camera position ± σ per axis", self.ed_cam_sigma)
        f.addRow("Landmark position ± σ per axis", self.ed_lm_sigma)
        self._add_page("geo", w)
        # ground (only shown when the library has estimate_ground)
        w = QWidget()
        f = QFormLayout(w)
        self.ed_pt_sigma = QLineEdit("0")
        f.addRow("Ground point position ± σ per axis", self.ed_pt_sigma)
        self._add_page("ground", w)
        # hud
        w = QWidget()
        v = QVBoxLayout(w)
        f = QFormLayout()
        self.ed_hud_label = QLineEdit("")
        self.sp_hud_sigma = QDoubleSpinBox(minimum=0.0, maximum=100.0, decimals=1, value=0.0, suffix=" %")
        f.addRow("HUD label shown in the frame", self.ed_hud_label)
        f.addRow("Table value ± σ", self.sp_hud_sigma)
        v.addLayout(f)
        self.tbl_hud = QTableWidget(0, 2)
        self.tbl_hud.setHorizontalHeaderLabels(["label", "HFOV (deg)"])
        self.tbl_hud.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_hud.setMinimumHeight(130)
        v.addWidget(QLabel("Label → HFOV table (your claim: from a manual, a published spec or a calibration)"))
        v.addWidget(self.tbl_hud)
        row = QHBoxLayout()
        self.btn_hud_add = QPushButton("Add row")
        self.btn_hud_add.clicked.connect(lambda: self.set_hud_row(self.tbl_hud.rowCount(), "", ""))
        self.btn_hud_remove = QPushButton("Remove row")
        self.btn_hud_remove.clicked.connect(lambda: self.tbl_hud.removeRow(self.tbl_hud.rowCount() - 1))
        self.btn_hud_example = QPushButton("Fill EXAMPLE table (ATFLIR, secondary sources)")
        self.btn_hud_example.clicked.connect(self.load_hud_example)
        row.addWidget(self.btn_hud_add)
        row.addWidget(self.btn_hud_remove)
        row.addWidget(self.btn_hud_example, 1)
        v.addLayout(row)
        self.lbl_hud_note = QLabel("")
        self.lbl_hud_note.setWordWrap(True)
        v.addWidget(self.lbl_hud_note)
        self._add_page("hud", w)
        lay.addWidget(self.stack)

        g = QGroupBox("Common")
        f = QFormLayout(g)
        self.sp_click = QDoubleSpinBox(minimum=0.0, maximum=50.0, decimals=2, singleStep=0.25, value=1.0)
        self.sp_click.setToolTip("1σ uncertainty of each un-snapped click, per axis")
        f.addRow("Click σ (px)", self.sp_click)
        row = QHBoxLayout()
        self.sp_rx0 = QSpinBox(minimum=0, maximum=100000)
        self.sp_ry0 = QSpinBox(minimum=0, maximum=100000)
        self.sp_rw = QSpinBox(minimum=0, maximum=100000)
        self.sp_rh = QSpinBox(minimum=0, maximum=100000)
        for lbl, sp in (("x0", self.sp_rx0), ("y0", self.sp_ry0), ("w", self.sp_rw), ("h", self.sp_rh)):
            row.addWidget(QLabel(lbl))
            row.addWidget(sp)
        self.btn_detect = QPushButton("Detect")
        self.btn_detect.setToolTip("Find a picture-in-picture inset automatically (needs uapka.fov.detect_inset)")
        self.btn_detect.setEnabled(callable(getattr(fov, "detect_inset", None)))
        self.btn_detect.clicked.connect(self.detect_region)
        row.addWidget(self.btn_detect)
        rw = QWidget()
        rw.setLayout(row)
        rw.setToolTip("Leave w = h = 0 for the full frame. Set the inset's rectangle when the footage is a "
                      "picture-in-picture window inside a larger frame.")
        f.addRow("Region (inset)", rw)
        lay.addWidget(g)

        row = QHBoxLayout()
        self.btn_compute = QPushButton("Compute")
        self.btn_compute.clicked.connect(self.compute)
        self.btn_apply = QPushButton("Apply as HFOV")
        self.btn_apply.clicked.connect(self.apply)
        self.btn_save = QPushButton("Save JSON…")
        self.btn_save.clicked.connect(lambda: self.save_json())
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.close)
        for b in (self.btn_compute, self.btn_apply, self.btn_save, self.btn_close):
            row.addWidget(b)
        lay.addLayout(row)
        self.txt_result = QTextEdit()
        self.txt_result.setReadOnly(True)
        self.txt_result.setMinimumHeight(160)
        lay.addWidget(self.txt_result, 1)

    def _add_page(self, name: str, w: QWidget):
        self.pages[name] = w
        self.stack.addWidget(w)

    def _fit_stack(self, m: str):
        """Size the method page area to the current page, not to the tallest page (no blank gap on the short
        ones, no squeezed HUD table on the tall one)."""
        for name, page in self.pages.items():
            page.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred if name == m else QSizePolicy.Ignored)
        self.stack.setCurrentWidget(self.pages[m])
        self.stack.adjustSize()
        self.stack.updateGeometry()

    # ---------------------------------------------------------- method / rows
    def method(self) -> str:
        return self.cmb_method.currentText()

    def _method_changed(self, m: str):
        self.lbl_help.setText(METHOD_HELP.get(m, ""))
        self._fit_stack(m)
        geo = m in ("geo", "ground")
        for col in (4, 5, 6):
            self.tbl_points.setColumnHidden(col, not geo)
        fixed = m in FIXED_POINTS
        self.btn_add.setEnabled(not fixed and m != "hud")
        self.btn_remove.setEnabled(not fixed and m != "hud")
        self.grp_points.setEnabled(m != "hud")
        self.grp_points.setVisible(m != "hud")        # a lookup needs no points
        need = MIN_POINTS[m]
        if fixed:
            while self.tbl_points.rowCount() > need:
                self.tbl_points.removeRow(self.tbl_points.rowCount() - 1)
        while self.tbl_points.rowCount() < need:
            self.add_row()
        self.estimate = None
        self.cancel_pick()
        self.pointsChanged.emit()

    def add_row(self) -> int:
        self._building = True
        r = self.tbl_points.rowCount()
        self.tbl_points.insertRow(r)
        it = QTableWidgetItem(str(r + 1))
        it.setFlags(it.flags() & ~Qt.ItemIsEditable)
        self.tbl_points.setItem(r, 0, it)
        for c in range(1, 7):
            self.tbl_points.setItem(r, c, QTableWidgetItem(""))
        b = QPushButton("Pick")
        b.setCheckable(True)
        b.toggled.connect(lambda on, row=r: self.pick(row) if on else self._unarm(row))
        self.tbl_points.setCellWidget(r, 7, b)
        self._building = False
        return r

    def remove_row(self):
        r = self.tbl_points.rowCount() - 1
        if r < 0 or r < MIN_POINTS[self.method()]:
            self.win._status(f"The {self.method()} method needs at least {MIN_POINTS[self.method()]} points.")
            return
        if self.pick_row == r:
            self.cancel_pick()
        self.tbl_points.removeRow(r)
        self._snapped.pop(r, None)
        self.pointsChanged.emit()

    def _pick_button(self, row: int) -> Optional[QPushButton]:
        return self.tbl_points.cellWidget(row, 7)

    def _unarm(self, row: int):
        if self.pick_row == row:
            self.pick_row = None
            self.pickCancelled.emit()

    def pick(self, row: int):
        """Arm the frame view: its next click lands in `row`."""
        if not (0 <= row < self.tbl_points.rowCount()):
            return
        if self.pick_row is not None and self.pick_row != row:
            self._set_button(self.pick_row, False)
        self.pick_row = row
        self._set_button(row, True)
        self.pickRequested.emit(row)

    def _set_button(self, row: int, on: bool):
        b = self._pick_button(row)
        if b is not None and b.isChecked() != on:
            b.blockSignals(True)
            b.setChecked(on)
            b.blockSignals(False)

    def cancel_pick(self):
        if self.pick_row is not None:
            row, self.pick_row = self.pick_row, None
            self._set_button(row, False)
            self.pickCancelled.emit()

    def receive_point(self, x: float, y: float, frame: int, img: Optional[np.ndarray] = None,
                      polarity: str = "auto") -> Optional[int]:
        """A click on the frame view in 'fovpoint' mode (original video pixels, sub-pixel). Snaps to the blob
        centroid if enabled and found, fills the armed row and arms the next empty row. Returns the row filled."""
        row = self.pick_row
        if row is None:
            return None
        snapped = False
        if self.chk_snap.isChecked() and img is not None:
            s = snap_to_blob(img, x, y, polarity)
            if s is not None:
                x, y = s
                snapped = True
        self.set_point(row, x, y, frame, snapped)
        self._set_button(row, False)
        self.pick_row = None
        nxt = next((r for r in range(row + 1, self.tbl_points.rowCount()) if not self._cell(r, 1)), None)
        msg = f"Point {row + 1} at ({x:.2f}, {y:.2f}) on frame {frame}" + (" (snapped to blob)" if snapped else "")
        if nxt is not None:
            self.pick(nxt)                                     # arms first: its status line would hide ours
            self.win._status(msg + f". Now click point {nxt + 1} (Esc cancels).", 15000)
        else:
            self.win._status(msg + ".")
            self.pickCancelled.emit()
        return row

    def set_point(self, row: int, x: float, y: float, frame: int, snapped: bool = False):
        while self.tbl_points.rowCount() <= row:
            self.add_row()
        self._building = True
        self.tbl_points.item(row, 1).setText(f"{x:.3f}")
        self.tbl_points.item(row, 2).setText(f"{y:.3f}")
        self.tbl_points.item(row, 3).setText(str(int(frame)))
        self._building = False
        self._snapped[row] = snapped
        self.estimate = None
        self.pointsChanged.emit()

    def set_geo(self, row: int, lat, lon, alt):
        while self.tbl_points.rowCount() <= row:
            self.add_row()
        self._building = True
        self.tbl_points.item(row, 4).setText(str(lat))
        self.tbl_points.item(row, 5).setText(str(lon))
        self.tbl_points.item(row, 6).setText(str(alt))
        self._building = False
        self.estimate = None

    def _cell(self, row: int, col: int) -> str:
        it = self.tbl_points.item(row, col)
        return it.text().strip() if it is not None else ""

    def points(self) -> list[dict]:
        """Every row as {x, y, frame, lat, lon, alt, snapped}; rows without x/y raise."""
        out = []
        for r in range(self.tbl_points.rowCount()):
            xs, ys = self._cell(r, 1), self._cell(r, 2)
            if not xs or not ys:
                raise ValueError(f"point {r + 1} has not been picked yet (press Pick, then click the frame)")
            fr = self._cell(r, 3)
            out.append(dict(x=_float(xs, f"point {r + 1} x"), y=_float(ys, f"point {r + 1} y"),
                            frame=int(float(fr)) if fr else None, lat=self._cell(r, 4), lon=self._cell(r, 5),
                            alt=self._cell(r, 6), snapped=bool(self._snapped.get(r, False))))
        return out

    def overlay_points(self, frame: int) -> list[tuple[float, float, str]]:
        """Points picked on `frame`, for the FrameView overlay."""
        if self.method() == "hud":
            return []
        out = []
        for r in range(self.tbl_points.rowCount()):
            xs, ys, fr = self._cell(r, 1), self._cell(r, 2), self._cell(r, 3)
            if not xs or not ys:
                continue
            try:
                if fr and int(float(fr)) != frame:
                    continue
                out.append((float(xs), float(ys), f"F{r + 1}"))
            except ValueError:
                continue
        return out

    # ---------------------------------------------------------- HUD table
    def hud_table(self) -> dict:
        table = {}
        for r in range(self.tbl_hud.rowCount()):
            k = self.tbl_hud.item(r, 0)
            v = self.tbl_hud.item(r, 1)
            k = k.text().strip() if k else ""
            v = v.text().strip() if v else ""
            if not k and not v:
                continue
            if not k:
                raise ValueError(f"HUD table row {r + 1} has no label")
            table[k] = _float(v, f"HUD table entry {k!r}")
        if not table:
            raise ValueError("the HUD table is empty: add label → HFOV rows (or fill the example and edit it)")
        return table

    def set_hud_table(self, table: dict):
        self.tbl_hud.setRowCount(0)
        for k, v in table.items():
            self.set_hud_row(self.tbl_hud.rowCount(), str(k), f"{float(v):g}")

    def set_hud_row(self, row: int, label: str, value: str):
        if row >= self.tbl_hud.rowCount():
            self.tbl_hud.insertRow(row)
        self.tbl_hud.setItem(row, 0, QTableWidgetItem(label))
        self.tbl_hud.setItem(row, 1, QTableWidgetItem(value))

    def load_hud_example(self):
        self.set_hud_table(HUD_EXAMPLE_TABLE)
        self.sp_hud_sigma.setValue(HUD_EXAMPLE_SIGMA_PCT)
        self.lbl_hud_note.setText(HUD_EXAMPLE_NOTE)
        if not self.ed_hud_label.text().strip():
            self.ed_hud_label.setText(next(iter(HUD_EXAMPLE_TABLE)))

    def load_hud_table_from_project(self) -> bool:
        """Pre-fill the HUD table with the one from the most recent hud estimate stored in the project."""
        for e in reversed(self.win.p.fov_estimates):
            if e.get("method") == "hud" and (e.get("inputs") or {}).get("table"):
                self.set_hud_table(e["inputs"]["table"])
                self.sp_hud_sigma.setValue(float(e["inputs"].get("sigma_pct", 0.0)))
                self.ed_hud_label.setText(str(e["inputs"].get("label", "")))
                self.lbl_hud_note.setText("Table taken from the project's previous HUD estimate.")
                return True
        return False

    # ---------------------------------------------------------- region
    def region(self) -> Optional[tuple[int, int, int, int]]:
        w, h = self.sp_rw.value(), self.sp_rh.value()
        if w <= 0 or h <= 0:
            return None
        return self.sp_rx0.value(), self.sp_ry0.value(), w, h

    def set_region(self, region: Optional[tuple]):
        x0, y0, w, h = region if region else (0, 0, 0, 0)
        self.sp_rx0.setValue(int(x0))
        self.sp_ry0.setValue(int(y0))
        self.sp_rw.setValue(int(w))
        self.sp_rh.setValue(int(h))

    def detect_region(self) -> Optional[tuple]:
        fn = getattr(fov, "detect_inset", None)
        if not callable(fn):
            self.win._status("This version of uapka.fov has no detect_inset(); enter the inset rectangle by hand.")
            return None
        if self.win.p.cache is None:
            self.win._status("Open a video first.")
            return None
        try:
            res = fn(self.win.p.cache.frames)
        except Exception as e:   # the library's own validation errors are shown, never raised into Qt
            self.win._msg("warning", "Detect inset", f"Inset detection failed:\n{e}")
            return None
        reg = parse_region(res)
        if reg is None:
            self.win._msg("info", "Detect inset", "No picture-in-picture inset was detected; the full frame is used.")
            self.set_region(None)
            return None
        self.set_region(reg)
        self.win._status(f"Inset region x0={reg[0]} y0={reg[1]} w={reg[2]} h={reg[3]}.")
        return reg

    # ---------------------------------------------------------- compute / apply / save
    def _frame_and_click_sigma(self, pts: list[dict]) -> tuple[Optional[int], float, list[str]]:
        extra = []
        frames = sorted({p["frame"] for p in pts if p["frame"] is not None})
        frame = frames[0] if frames else (self.win.k if self.win.p.cache is not None else None)
        if len(frames) > 1:
            extra.append(f"Points were picked on different frames ({frames}); the FOV is attributed to frame "
                         f"{frame}. That is only valid if the zoom did not change between them.")
        sig = self.sp_click.value()
        per = [SNAP_SIGMA_PX if p["snapped"] else sig for p in pts]
        eff = max(per) if per else sig
        n_snap = sum(1 for p in pts if p["snapped"])
        if pts and n_snap and n_snap < len(pts):
            extra.append(f"{n_snap} of {len(pts)} points snapped to a blob centroid (σ {SNAP_SIGMA_PX:g} px); the "
                         f"others use the click σ of {sig:g} px, so {eff:g} px is applied to all (conservative).")
        elif pts and n_snap == len(pts):
            extra.append(f"All points snapped to blob centroids: click σ {eff:g} px.")
        return frame, eff, extra

    def compute(self) -> Optional[FovEstimate]:
        """Run the selected estimate_* on the dialog's inputs; errors become a warning message, not a traceback."""
        p = self.win.p
        if p.meta is None or p.cache is None:
            self.win._msg("info", "Derive FOV", "Open a video first: the frame size is part of the estimate.")
            return None
        W, H = int(p.meta.width), int(p.meta.height)
        par = float(self.win.sp_par.value())
        m = self.method()
        fn = estimator(m)
        try:
            if m == "hud":
                pts = []
                frame, eff, extra = self.win.k, 0.0, []
                label = self.ed_hud_label.text().strip()
                if not label:
                    raise ValueError("enter the HUD label shown in the frame (e.g. NAR, MFOV, WFOV, ZOOM 2)")
                kwargs = dict(label=label, table=self.hud_table(), sigma_pct=float(self.sp_hud_sigma.value()))
            else:
                pts = self.points()
                if len(pts) < MIN_POINTS[m]:
                    raise ValueError(f"the {m} method needs at least {MIN_POINTS[m]} points")
                frame, eff, extra = self._frame_and_click_sigma(pts)
                kwargs = dict(click_sigma_px=eff)
                if m == "separation":
                    if self.sp_angle.value() <= 0:
                        raise ValueError("enter the true angle between the two points (deg)")
                    kwargs.update(p1=(pts[0]["x"], pts[0]["y"]), p2=(pts[1]["x"], pts[1]["y"]),
                                  angle_deg=float(self.sp_angle.value()),
                                  angle_sigma_deg=float(self.sp_angle_sigma.value()))
                elif m == "size":
                    kwargs.update(p1=(pts[0]["x"], pts[0]["y"]), p2=(pts[1]["x"], pts[1]["y"]),
                                  size_m=_length(self.ed_size.text(), "size"),
                                  range_m=_length(self.ed_range.text(), "range"),
                                  size_sigma_m=_length(self.ed_size_sigma.text(), "size σ", allow_zero=True),
                                  range_sigma_m=_length(self.ed_range_sigma.text(), "range σ", allow_zero=True))
                elif m == "geo":
                    cam = _geo(self.ed_cam_lat.text(), self.ed_cam_lon.text(), self.ed_cam_alt.text(), "camera")
                    lms = [((q["x"], q["y"]), _geo(q["lat"], q["lon"], q["alt"], f"landmark {i + 1}"))
                           for i, q in enumerate(pts)]
                    kwargs.update(camera=cam, landmarks=lms,
                                  camera_sigma_m=_length(self.ed_cam_sigma.text(), "camera σ", allow_zero=True),
                                  landmark_sigma_m=_length(self.ed_lm_sigma.text(), "landmark σ", allow_zero=True))
                elif m == "ground":
                    gps = [((q["x"], q["y"]), _geo(q["lat"], q["lon"], q["alt"], f"point {i + 1}"))
                           for i, q in enumerate(pts)]
                    kwargs.update(points=gps,
                                  point_sigma_m=_length(self.ed_pt_sigma.text(), "point σ", allow_zero=True))
            reg = self.region()
            if reg is not None:
                if not accepts(fn, "region"):
                    raise ValueError("an inset region is set, but this version of uapka.fov does not take a "
                                     "region= argument. Clear the region (w = h = 0) or update the library.")
                kwargs["region"] = reg
            est = fn(W, H, pixel_aspect=par, frame=frame, **kwargs)
        except (ValueError, KeyError, TypeError) as e:
            msg = e.args[0] if isinstance(e, KeyError) and e.args else str(e)
            self.estimate = None
            self.txt_result.setHtml(f"<p style='color:#c02020'>{escape(str(msg))}</p>")
            self.win._msg("warning", "Derive FOV", f"Could not derive the FOV with the {m} method:\n{msg}")
            return None
        self.estimate = est
        self._extra = extra
        self.txt_result.setHtml(result_html(est, extra))
        self.win._status(est.summary() + "  (Apply to use it as the HFOV)", 15000)
        return est

    def apply(self) -> bool:
        if self.estimate is None and self.compute() is None:
            return False
        est = self.estimate
        frame = (est.inputs or {}).get("frame")
        if self.win.apply_fov_estimate(est):
            p, kp = self.win.p, self.win.p.kin_params
            if frame is not None and p.cache is not None and p.file_frame(int(frame)) != kp.start_frame:
                self.win._msg("warning", "Derive FOV",
                              f"The FOV was derived on frame {frame} (video frame {p.file_frame(int(frame))}) "
                              f"but the kinematics reference frame is video frame {kp.start_frame}. The HFOV belongs "
                              "to the reference frame unless the zoom is constant: check the camera's zoom trend "
                              "(Camera tab) or derive the FOV on the reference frame.")
            return True
        return False

    def save_json(self, path: Optional[str] = None) -> bool:
        if self.estimate is None and self.compute() is None:
            return False
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, "Save FOV estimate", "fov.json", "FOV estimate (*.json)")
            if not path:
                return False
        try:
            ests = save_fov_json(path, self.estimate)
        except (OSError, ValueError) as e:
            self.win._msg("error", "Save FOV estimate", f"Could not write {path}:\n{e}")
            return False
        self.win._status(f"Saved as estimate #{len(ests) - 1} in {path} (uapka analyze … --fov-json {path})", 10000)
        return True

    def closeEvent(self, ev):
        self.cancel_pick()
        super().closeEvent(ev)


# ---------------------------------------------------------------- history
class FovHistoryDialog(QDialog):
    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.setWindowTitle("FOV estimates in this project")
        self.setModal(False)
        self.setWindowFlags(Qt.Tool | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
        self.resize(640, 420)
        lay = QVBoxLayout(self)
        self.lst = QListWidget()
        self.lst.currentRowChanged.connect(self._show_details)
        lay.addWidget(self.lst, 1)
        self.txt = QTextEdit()
        self.txt.setReadOnly(True)
        self.txt.setMaximumHeight(160)
        lay.addWidget(self.txt)
        row = QHBoxLayout()
        self.btn_apply = QPushButton("Re-apply")
        self.btn_apply.clicked.connect(lambda: self.reapply())
        self.btn_delete = QPushButton("Delete")
        self.btn_delete.clicked.connect(lambda: self.delete())
        self.btn_export = QPushButton("Export selected…")
        self.btn_export.clicked.connect(lambda: self.export_selected())
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.close)
        for b in (self.btn_apply, self.btn_delete, self.btn_export, self.btn_close):
            row.addWidget(b)
        lay.addLayout(row)
        self.refresh()

    def refresh(self):
        p = self.win.p
        cur = self.lst.currentRow()
        self.lst.clear()
        src = p.fov_source or {}
        for i, e in enumerate(p.fov_estimates):
            est = FovEstimate.from_dict(e)
            frame = (e.get("inputs") or {}).get("frame")
            in_use = p.fov_status()[0] == "derived" and src.get("index") == i
            txt = f"#{i}  {est.summary()}" + (f"  frame {frame}" if frame is not None else "") + \
                  f"  {e.get('width')}x{e.get('height')}" + ("   ← in use" if in_use else "")
            self.lst.addItem(txt)
        if self.lst.count():
            self.lst.setCurrentRow(min(max(cur, 0), self.lst.count() - 1))
        else:
            self.txt.setPlainText("No FOV estimates in this project yet (Derive…, or Import JSON…).")

    def selected(self) -> int:
        return self.lst.currentRow()

    def _show_details(self, i: int):
        p = self.win.p
        if not (0 <= i < len(p.fov_estimates)):
            self.txt.clear()
            return
        e = p.fov_estimates[i]
        notes = e.get("notes") or []
        self.txt.setPlainText(describe_inputs(e) + ("\n\n" + "\n".join(f"• {n}" for n in notes) if notes else ""))

    def reapply(self, i: Optional[int] = None) -> bool:
        i = self.selected() if i is None else i
        p = self.win.p
        if not (0 <= i < len(p.fov_estimates)):
            self.win._status("Select an estimate first.")
            return False
        ok = self.win.apply_fov_estimate(p.fov_estimates[i], index=i)
        self.refresh()
        return ok

    def delete(self, i: Optional[int] = None) -> bool:
        i = self.selected() if i is None else i
        p = self.win.p
        if not (0 <= i < len(p.fov_estimates)):
            self.win._status("Select an estimate first.")
            return False
        kind, src = p.fov_status()
        if kind == "derived" and src is not None and src.get("index") == i:
            self.win._msg("info", "FOV history", "This estimate is the one in use. Apply another one (or type an "
                          "HFOV by hand) before deleting it, so the provenance of the HFOV stays on record.")
            return False
        del p.fov_estimates[i]
        if p.fov_source is not None and isinstance(p.fov_source.get("index"), int) and p.fov_source["index"] > i:
            p.fov_source["index"] -= 1
        self.refresh()
        self.win.fov_panel.refresh()
        self.win._status(f"Deleted FOV estimate #{i}.")
        return True

    def export_selected(self, path: Optional[str] = None, i: Optional[int] = None) -> bool:
        i = self.selected() if i is None else i
        p = self.win.p
        if not (0 <= i < len(p.fov_estimates)):
            self.win._status("Select an estimate first.")
            return False
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, "Export FOV estimate", "fov.json", "FOV estimate (*.json)")
            if not path:
                return False
        try:
            save_fov_json(path, p.fov_estimates[i])
        except (OSError, ValueError) as e:
            self.win._msg("error", "Export FOV estimate", f"Could not write {path}:\n{e}")
            return False
        self.win._status(f"Exported estimate #{i} to {path}", 8000)
        return True


# ---------------------------------------------------------------- the group on the Kinematics tab
class FovPanel(QGroupBox):
    def __init__(self, win):
        super().__init__("Derive FOV…")
        self.win = win
        self.dialog: Optional[FovDialog] = None
        self.history: Optional[FovHistoryDialog] = None
        lay = QVBoxLayout(self)
        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.lbl_status)
        row = QHBoxLayout()
        self.btn_derive = QPushButton("Derive…")
        self.btn_derive.setToolTip("Derive the HFOV from two points with a known angle, an assumed size and range, "
                                   "geo-located landmarks, or a HUD label table (± 1σ, inputs recorded)")
        self.btn_derive.clicked.connect(self.open_dialog)
        self.btn_history = QPushButton("History…")
        self.btn_history.clicked.connect(self.open_history)
        self.btn_import = QPushButton("Import JSON…")
        self.btn_import.clicked.connect(lambda: self.import_json())
        self.btn_export = QPushButton("Export JSON…")
        self.btn_export.clicked.connect(lambda: self.export_json())
        for b in (self.btn_derive, self.btn_history, self.btn_import, self.btn_export):
            row.addWidget(b)
        lay.addLayout(row)
        self.refresh()

    def refresh(self):
        self.lbl_status.setText(status_text(self.win.p))
        n = len(self.win.p.fov_estimates)
        self.btn_history.setText(f"History… ({n})" if n else "History…")
        if self.history is not None and self.history.isVisible():
            self.history.refresh()

    # ---------------------------------------------------------- dialogs
    def open_dialog(self) -> Optional[FovDialog]:
        if self.win.p.cache is None:
            self.win._status("Open a video first: the FOV is derived from points clicked in its frames.")
            return None
        if self.dialog is None:
            self.dialog = FovDialog(self.win)
            self.dialog.pickRequested.connect(self._arm)
            self.dialog.pickCancelled.connect(self._disarm)
            self.dialog.pointsChanged.connect(self.win._update_overlays)
            self.dialog.finished.connect(lambda _: self.win._update_overlays())
            self.dialog.load_hud_table_from_project()
        self.dialog.show()
        self.dialog.raise_()
        self.win._update_overlays()
        return self.dialog

    def open_history(self) -> FovHistoryDialog:
        if self.history is None:
            self.history = FovHistoryDialog(self.win)
        self.history.refresh()
        self.history.show()
        self.history.raise_()
        return self.history

    def _arm(self, row: int):
        self.win._set_mode("fovpoint")
        self.win._status(f"FOV point {row + 1}: click it in the frame view (scrub / zoom first if you like; Esc "
                         "cancels).", 15000)

    def _disarm(self):
        if self.win.view.mode == "fovpoint":
            self.win._set_mode("none")

    def cancel_pick(self):
        """The window left 'fovpoint' mode (Escape, another tool): forget the armed row."""
        if self.dialog is not None and self.dialog.pick_row is not None:
            self.dialog.cancel_pick()

    def on_point(self, x: float, y: float):
        """A click from the FrameView in 'fovpoint' mode."""
        if self.dialog is None or self.dialog.pick_row is None:
            self.win._set_mode("none")
            return
        p = self.win.p
        img = p.cache[self.win.k] if p.cache is not None else None
        pol = self.win.cmb_pol.currentText()
        self.dialog.receive_point(x, y, self.win.k, img, pol)

    def overlay_points(self, frame: int) -> list:
        if self.dialog is None or not self.dialog.isVisible():
            return []
        return self.dialog.overlay_points(frame)

    # ---------------------------------------------------------- JSON files
    def import_json(self, path: Optional[str] = None) -> bool:
        """Load estimates from a file written by 'uapka fov --save' / Export, keep them all, apply the last."""
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self.win, "Import FOV estimates", "", "FOV estimate (*.json)")
            if not path:
                return False
        try:
            ests = load_fov_json(path)
        except (OSError, ValueError, KeyError, TypeError) as e:
            self.win._msg("warning", "Import FOV estimates", f"Could not read {path}:\n{e}")
            return False
        if not ests:
            self.win._msg("warning", "Import FOV estimates", f"{path} holds no FOV estimates.")
            return False
        p = self.win.p
        added = 0
        for e in ests:
            e = dict(e)
            e.pop("index", None)
            if e not in p.fov_estimates:
                p.fov_estimates.append(e)
                added += 1
        last = dict(ests[-1])
        last.pop("index", None)
        ok = self.win.apply_fov_estimate(last, index=p.fov_estimates.index(last))
        self.refresh()
        if ok:
            self.win._status(f"Imported {len(ests)} estimate(s) from {os.path.basename(path)} ({added} new); the "
                             "last one is applied. History… lists them all.", 12000)
        return ok

    def export_json(self, path: Optional[str] = None) -> bool:
        """Write every estimate of the project to one file (readable by 'uapka analyze --fov-json')."""
        p = self.win.p
        if not p.fov_estimates:
            self.win._msg("info", "Export FOV estimates", "No FOV estimates in this project yet (Derive… first).")
            return False
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self.win, "Export FOV estimates", "fov.json", "FOV estimate (*.json)")
            if not path:
                return False
        try:
            for i, e in enumerate(p.fov_estimates):
                save_fov_json(path, e, append=i > 0)
        except (OSError, ValueError) as e:
            self.win._msg("error", "Export FOV estimates", f"Could not write {path}:\n{e}")
            return False
        self.win._status(f"Exported {len(p.fov_estimates)} estimate(s) to {path}", 8000)
        return True
