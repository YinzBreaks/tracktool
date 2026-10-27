"""Per-frame strip under the slider: track status (top) and camera-motion confidence (bottom).

Lets the user spot drift, lost frames and weak camera estimates at a glance and
jump there with a click.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from . import theme
from .frameview import STATUS_COLORS


class Timeline(QWidget):
    frameClicked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(34)
        self.setMouseTracking(True)
        self.n = 0
        self.status = None
        self.conf = None
        self.keyframes = []
        self.current = 0
        self._hover = None
        self.setToolTip("Track status (top: green OK, cyan manual, yellow re-acquired, red lost) and camera-motion "
                        "confidence (bottom: dark = low). Diamonds = keyframes. Click or drag to jump.")

    def set_data(self, n, status=None, conf=None, keyframes=()):
        self.n, self.status, self.conf, self.keyframes = n, status, conf, list(keyframes)
        self.update()

    def set_current(self, k):
        self.current = k
        self.update()

    def paintEvent(self, ev):
        t = theme.current()
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(t["input"]))
        p.setPen(QPen(QColor(t["border"]), 1))
        p.drawRect(QRectF(self.rect()).adjusted(0, 0, -1, -1))
        if self.n <= 0:
            return
        W, H = self.width(), self.height()
        bw = W / self.n
        top, mid, bot = 7.0, H * 0.52, H - 1.0      # status lane [top, mid), confidence lane [mid+2, bot)
        if self.status is not None:
            for k, st in enumerate(self.status):
                c = STATUS_COLORS.get(st)
                if c:
                    p.fillRect(QRectF(k * bw, top, max(bw, 1), mid - top), QColor(c))
        if self.conf is not None:
            lo, hi = QColor(t["viewport"]), QColor(t["accent"])
            for k, c in enumerate(self.conf):
                if np.isfinite(c):
                    f = float(np.clip(c, 0, 1))
                    col = QColor.fromRgbF(lo.redF() + (hi.redF() - lo.redF()) * f,
                                          lo.greenF() + (hi.greenF() - lo.greenF()) * f,
                                          lo.blueF() + (hi.blueF() - lo.blueF()) * f)
                    p.fillRect(QRectF(k * bw, mid + 2, max(bw, 1), bot - mid - 2), col)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(t["text"]))
        for k in self.keyframes:          # keyframes: small diamonds above the status lane
            x = (k + 0.5) * bw
            p.drawPolygon(QPolygonF([QPointF(x, 0.5), QPointF(x + 3, 3.5), QPointF(x, 6.5), QPointF(x - 3, 3.5)]))
        if self._hover is not None:
            x = (self._hover + 0.5) * bw
            p.setPen(QPen(QColor(t["text2"]), 1, Qt.DotLine))
            p.drawLine(QPointF(x, top), QPointF(x, H))
        x = (self.current + 0.5) * bw
        p.setPen(QPen(QColor(t["trail"]), 2))
        p.drawLine(QPointF(x, top), QPointF(x, H))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(t["trail"]))
        p.drawPolygon(QPolygonF([QPointF(x - 5, 0), QPointF(x + 5, 0), QPointF(x, 7)]))

    def leaveEvent(self, ev):
        self._hover = None
        self.update()

    def _frame_at(self, ev) -> int:
        return int(np.clip(ev.position().x() / self.width() * self.n, 0, self.n - 1))

    def mousePressEvent(self, ev):
        if self.n > 0:
            self.frameClicked.emit(self._frame_at(ev))

    def mouseMoveEvent(self, ev):
        if self.n <= 0:
            return
        self._hover = self._frame_at(ev)
        if ev.buttons() & Qt.LeftButton:
            self.frameClicked.emit(self._hover)
        self.update()
