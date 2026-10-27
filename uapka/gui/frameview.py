"""Frame display with zoom/pan, overlays and mouse interaction modes."""

from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

STATUS_COLORS = {"OK": "#3fc85a", "MANUAL": "#2fd0e8", "REACQ": "#f0c020", "LOST": "#ff4040", "WEAK": "#ff4040",
                 "AMBIGUOUS": "#ff9a3c", "EXCLUDED": "#a0a0a0", "BLANK": "#606060"}
MASK_COLORS = {"HUD": "#2a78d6", "RETICLE": "#c040c0", "WATERMARK": "#d09020", "REDACTION": "#707070",
               "OTHER": "#909090"}


class FrameView(QWidget):
    roiSelected = Signal(float, float, float, float)     # cx, cy, w, h (image px)
    maskDrawn = Signal(int, int, int, int)               # x, y, w, h
    pointClicked = Signal(float, float)

    MODES = ("none", "roi", "mask", "point")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumSize(480, 360)
        self.setFocusPolicy(Qt.StrongFocus)
        self._img: Optional[QImage] = None
        self._buf = None
        self.mode = "none"
        self.zoom = 0.0          # 0 = fit
        self.center = QPointF(0, 0)
        self._drag_start = None
        self._drag_now = None
        self._pan_start = None
        self.hover = None
        # overlays (set by the window)
        self.masks = []
        self.box = None          # (cx, cy, w, h, status)
        self.trail = None        # (N,2) array in current-frame px
        self.keyframe_here = False
        self.candidates = []     # list of (x, y, label)
        self.others = []         # other tracked objects at this frame: (x, y, label)
        self.message = ""

    # ---------------------------------------------------------------- image
    def set_frame(self, gray: np.ndarray, enhance: bool = False):
        if enhance:
            lo, hi = np.percentile(gray[::2, ::2], [1, 99.5])
            gray = np.clip((gray.astype(np.float32) - lo) * (255.0 / max(hi - lo, 1)), 0, 255).astype(np.uint8)
        self._buf = np.ascontiguousarray(gray)
        h, w = self._buf.shape
        new = self._img is None or self._img.width() != w or self._img.height() != h
        self._img = QImage(self._buf.data, w, h, w, QImage.Format_Grayscale8)
        if new:
            self.zoom = 0.0
            self.center = QPointF(w / 2, h / 2)
        self.update()

    def _scale(self) -> float:
        if self._img is None:
            return 1.0
        fit = min(self.width() / self._img.width(), self.height() / self._img.height())
        return fit if self.zoom <= 0 else self.zoom

    def _origin(self) -> QPointF:
        s = self._scale()
        if self.zoom <= 0:
            return QPointF((self.width() - self._img.width() * s) / 2, (self.height() - self._img.height() * s) / 2)
        return QPointF(self.width() / 2 - self.center.x() * s, self.height() / 2 - self.center.y() * s)

    def to_img(self, p: QPointF) -> QPointF:
        s, o = self._scale(), self._origin()
        return QPointF((p.x() - o.x()) / s, (p.y() - o.y()) / s)

    def to_widget(self, x: float, y: float) -> QPointF:
        s, o = self._scale(), self._origin()
        return QPointF(o.x() + x * s, o.y() + y * s)

    def zoom_fit(self):
        self.zoom = 0.0
        self.update()

    def zoom_1to1(self):
        if self._img is not None:
            self.center = QPointF(self._img.width() / 2, self._img.height() / 2)
        self.zoom = 1.0
        self.update()

    # ---------------------------------------------------------------- paint
    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#1a1a19"))
        if self._img is None:
            p.setPen(QColor("#c3c2b7"))
            p.drawText(self.rect(), Qt.AlignCenter, "File > Open video…  (or drop a file here)")
            return
        s, o = self._scale(), self._origin()
        p.setRenderHint(QPainter.SmoothPixmapTransform, s < 2)
        p.drawImage(QRectF(o.x(), o.y(), self._img.width() * s, self._img.height() * s), self._img)
        p.setRenderHint(QPainter.Antialiasing, True)
        for m in self.masks:
            c = QColor(MASK_COLORS.get(m.kind, "#909090"))
            tl = self.to_widget(m.x, m.y)
            r = QRectF(tl.x(), tl.y(), m.w * s, m.h * s)
            fill = QColor(c)
            fill.setAlpha(55)
            p.fillRect(r, QBrush(fill, Qt.BDiagPattern))
            p.setPen(QPen(c, 1.5))
            p.drawRect(r)
            p.drawText(r.adjusted(3, 2, 0, 0), Qt.AlignLeft | Qt.AlignTop, m.kind)
        if self.trail is not None and len(self.trail) > 1:
            pen = QPen(QColor("#ff9a3c"), 1.5)
            p.setPen(pen)
            pts = [self.to_widget(x, y) for x, y in self.trail]
            for a, b in zip(pts[:-1], pts[1:]):
                p.drawLine(a, b)
        for x, y, label in self.others:
            c = self.to_widget(x, y)
            p.setPen(QPen(QColor("#e87ba4"), 1.2))
            p.drawEllipse(c, 7, 7)
            p.drawText(c + QPointF(9, -6), label)
        for x, y, label in self.candidates:
            c = self.to_widget(x, y)
            p.setPen(QPen(QColor("#f0c020"), 1.5, Qt.DashLine))
            p.drawEllipse(c, 10, 10)
            p.drawText(c + QPointF(12, -8), label)
        if self.box is not None:
            cx, cy, w, h, st = self.box
            col = QColor(STATUS_COLORS.get(st, "#ffffff"))
            p.setPen(QPen(col, 2 if self.keyframe_here else 1.5))
            tl = self.to_widget(cx - w / 2 - 2, cy - h / 2 - 2)
            p.drawRect(QRectF(tl.x(), tl.y(), (w + 4) * s, (h + 4) * s))
            c = self.to_widget(cx, cy)
            p.drawLine(c + QPointF(-6, 0), c + QPointF(-2, 0))
            p.drawLine(c + QPointF(2, 0), c + QPointF(6, 0))
            p.drawLine(c + QPointF(0, -6), c + QPointF(0, -2))
            p.drawLine(c + QPointF(0, 2), c + QPointF(0, 6))
            p.drawText(tl + QPointF(0, -4), st + (" (keyframe)" if self.keyframe_here else ""))
        if self._drag_start is not None and self._drag_now is not None:
            p.setPen(QPen(QColor("#2fd0e8") if self.mode == "roi" else QColor("#2a78d6"), 1.5, Qt.DashLine))
            p.drawRect(QRectF(self._drag_start, self._drag_now).normalized())
        # HUD text
        p.setPen(QColor("#f0efec"))
        hint = {"roi": "Drag a box around the object", "mask": "Drag to draw a mask region",
                "point": "Click the object's true position"}.get(self.mode, "")
        txt = "  ".join(t for t in (hint, self.message) if t)
        if self.hover is not None:
            x, y = self.hover
            if 0 <= x < self._img.width() and 0 <= y < self._img.height():
                txt += f"   ({x:.1f}, {y:.1f}) = {int(self._buf[int(y), int(x)])}"
        txt += f"   zoom {s:.2f}x (wheel / right-drag, F = fit, 1 = 1:1)"
        p.fillRect(QRectF(0, self.height() - 20, self.width(), 20), QColor(0, 0, 0, 160))
        p.drawText(QRectF(6, self.height() - 20, self.width() - 12, 20), Qt.AlignVCenter | Qt.AlignLeft, txt)

    # ---------------------------------------------------------------- input
    def wheelEvent(self, ev):
        if self._img is None:
            return
        before = self.to_img(ev.position())
        s = self._scale()
        factor = 1.25 if ev.angleDelta().y() > 0 else 0.8
        self.zoom = float(np.clip(s * factor, 0.1, 40))
        # keep the point under the cursor fixed
        self.center = QPointF(before.x() - (ev.position().x() - self.width() / 2) / self.zoom,
                              before.y() - (ev.position().y() - self.height() / 2) / self.zoom)
        self.update()

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_F:
            self.zoom_fit()
        elif ev.key() == Qt.Key_1:
            self.zoom_1to1()
        else:
            super().keyPressEvent(ev)

    def mousePressEvent(self, ev):
        if self._img is None:
            return
        if ev.button() in (Qt.RightButton, Qt.MiddleButton):
            if self.zoom <= 0:
                self.zoom = self._scale()
                self.center = self.to_img(QPointF(self.width() / 2, self.height() / 2))
            self._pan_start = (ev.position(), QPointF(self.center))
            return
        if ev.button() == Qt.LeftButton:
            if self.mode in ("roi", "mask"):
                self._drag_start = ev.position()
                self._drag_now = ev.position()
            elif self.mode == "point":
                q = self.to_img(ev.position())
                self.pointClicked.emit(q.x(), q.y())

    def mouseMoveEvent(self, ev):
        if self._img is None:
            return
        q = self.to_img(ev.position())
        self.hover = (q.x(), q.y())
        if self._pan_start is not None:
            start, c0 = self._pan_start
            d = ev.position() - start
            s = self._scale()
            self.center = QPointF(c0.x() - d.x() / s, c0.y() - d.y() / s)
        elif self._drag_start is not None:
            self._drag_now = ev.position()
        self.update()

    def mouseReleaseEvent(self, ev):
        if self._pan_start is not None and ev.button() in (Qt.RightButton, Qt.MiddleButton):
            self._pan_start = None
            return
        if self._drag_start is not None and ev.button() == Qt.LeftButton:
            a, b = self.to_img(self._drag_start), self.to_img(ev.position())
            self._drag_start = self._drag_now = None
            x0, x1 = sorted((a.x(), b.x()))
            y0, y1 = sorted((a.y(), b.y()))
            W, H = self._img.width(), self._img.height()
            x0, y0 = max(0.0, x0), max(0.0, y0)
            x1, y1 = min(float(W), x1), min(float(H), y1)
            if x1 - x0 >= 2 and y1 - y0 >= 2:
                if self.mode == "roi":
                    self.roiSelected.emit((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)
                else:
                    self.maskDrawn.emit(int(x0), int(y0), int(round(x1 - x0)), int(round(y1 - y0)))
            self.update()
