"""Per-frame strip under the slider: track status (top) and camera-motion confidence (bottom).

Lets the user spot drift, lost frames and weak camera estimates at a glance and
jump there with a click.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRectF, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

from .frameview import STATUS_COLORS


class Timeline(QWidget):
    frameClicked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(30)
        self.n = 0
        self.status = None
        self.conf = None
        self.keyframes = []
        self.current = 0
        self.setToolTip("Track status (top: green OK, cyan manual, yellow re-acquired, red lost/weak, orange "
                        "ambiguous, grey excluded, dark grey duplicate/blank; white ticks = keyframes) and "
                        "camera-motion confidence (bottom: dark = low). Click to jump.")

    def set_data(self, n, status=None, conf=None, keyframes=()):
        self.n, self.status, self.conf, self.keyframes = int(n), status, conf, list(keyframes)
        if self.n <= 0:
            self.status = self.conf = None
            self.keyframes = []
        self.update()

    def set_current(self, k):
        self.current = k
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#2a2a28"))
        if self.n <= 0:
            return
        W, H = self.width(), self.height()
        bw = W / self.n
        if self.status is not None:
            for k, st in enumerate(self.status):
                c = STATUS_COLORS.get(st)
                if c:
                    p.fillRect(QRectF(k * bw, 0, max(bw, 1), H * 0.5), QColor(c))
        if self.conf is not None:
            for k, c in enumerate(self.conf):
                if np.isfinite(c):
                    g = int(40 + 200 * float(np.clip(c, 0, 1)))
                    p.fillRect(QRectF(k * bw, H * 0.55, max(bw, 1), H * 0.45), QColor(g // 3, g // 2, g))
        p.setPen(QPen(QColor("#ffffff"), 1))
        for k in self.keyframes:
            x = (k + 0.5) * bw
            p.drawLine(int(x), 0, int(x), 5)
        x = (self.current + 0.5) * bw
        p.setPen(QPen(QColor("#ff9a3c"), 2))
        p.drawLine(int(x), 0, int(x), H)

    def mousePressEvent(self, ev):
        if self.n > 0:
            self.frameClicked.emit(int(np.clip(ev.position().x() / self.width() * self.n, 0, self.n - 1)))

    mouseMoveEvent = mousePressEvent
