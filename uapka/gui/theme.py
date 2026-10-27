"""Instrument-style visual theme: palette, Qt stylesheet, provenance colours and matplotlib restyling.

Two themes: "dark" (default, for the desk) and "light" (for classroom projectors). Colour is used
for meaning first. The three provenance labels each have one colour, used in readouts, tables,
the status-bar legend and plots, so MEASURED, MEASURED+FOV and ASSUMED can be told apart at a
glance. ASSUMED is always drawn "softer" (italic, hatched) so it never reads like a measurement.

Exported figures and reports are not affected: they are built separately by ``uapka.plots`` in
its light print style. Only figures shown inside the GUI are restyled.
"""

from __future__ import annotations

import html
import os
import re
import tempfile

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QFontDatabase, QPalette

THEMES = {
    "dark": {
        "bg": "#0d1014", "surface": "#13181d", "raised": "#1a2027", "input": "#0f1317",
        "border": "#262e37", "border_strong": "#36414d",
        "text": "#d8dee4", "text2": "#8c97a3", "text3": "#5d6874",
        "accent": "#4cc3ff", "accent_text": "#04131c", "accent_dim": "#173547",
        "viewport": "#06080a", "hud_bg": (6, 8, 10, 200), "trail": "#ffa040",
        "danger": "#ff5a5a",
        "measured": "#5fd38a", "fov": "#5aa9ff", "assumed": "#f2b134",
        "plot_bg": "#13181d", "plot_grid": "#232b33", "plot_spine": "#36414d", "plot_lowconf": "#2a323b",
        "plot_series": ("#5aa9ff", "#f2814e", "#3fd09a"),
    },
    "light": {
        "bg": "#eef1f4", "surface": "#f8fafb", "raised": "#ffffff", "input": "#ffffff",
        "border": "#cdd4db", "border_strong": "#aab4be",
        "text": "#10151a", "text2": "#4f5b66", "text3": "#7f8a95",
        "accent": "#0b6bcb", "accent_text": "#ffffff", "accent_dim": "#d6e7f8",
        "viewport": "#1a1d20", "hud_bg": (6, 8, 10, 190), "trail": "#ff9a3c",
        "danger": "#c62828",
        "measured": "#1b8a4b", "fov": "#1f5fbf", "assumed": "#a8620a",
        "plot_bg": "#fcfcfb", "plot_grid": "#e4e3df", "plot_spine": "#b9b8b2", "plot_lowconf": "#d9d7d0",
        "plot_series": ("#2a78d6", "#eb6834", "#1baf7a"),
    },
}

PROVENANCE_KEYS = {"MEASURED": "measured", "MEASURED+FOV": "fov", "ASSUMED": "assumed"}
PROVENANCE_HELP = {
    "MEASURED": "From pixel positions and frame timing only.",
    "MEASURED+FOV": "Measured motion converted to angles with the horizontal FOV you entered. Scales linearly "
                    "with the FOV.",
    "ASSUMED": "Depends on an assumed range, size or altitude. Scales linearly with it. Not a measurement.",
}
# "[MEASURED]", "[MEASURED+FOV]", "[ASSUMED]", "[STATIC CAMERA ASSUMED]"
_LABEL_RE = re.compile(r"\[(MEASURED\+FOV|MEASURED|[A-Z ]*ASSUMED)\]")

_current = "dark"


def name() -> str:
    return _current


def current() -> dict:
    return THEMES[_current]


def saved_name() -> str:
    n = QSettings("uapka", "UAPKA").value("theme", "dark")
    return n if n in THEMES else "dark"


def save_name(n: str) -> None:
    QSettings("uapka", "UAPKA").setValue("theme", n)


def mono_font(point_delta: float = 0.0) -> QFont:
    f = QFontDatabase.systemFont(QFontDatabase.FixedFont)
    f.setStyleHint(QFont.Monospace)
    if point_delta:
        f.setPointSizeF(max(6.0, f.pointSizeF() + point_delta))
    return f


def provenance_color(label: str) -> QColor:
    """Colour for a provenance label; any '... ASSUMED' label uses the ASSUMED colour."""
    key = PROVENANCE_KEYS.get(label, "assumed" if label.endswith("ASSUMED") else "text2")
    return QColor(current()[key])


def assumed_brush() -> QBrush:
    """Hatched fill that marks ASSUMED cells as 'not a measurement'."""
    c = QColor(current()["assumed"])
    c.setAlpha(45)
    return QBrush(c, Qt.BDiagPattern)


# ------------------------------------------------------------------ Qt application styling
def _palette(t: dict) -> QPalette:
    p = QPalette()
    c = QColor
    for role, col in ((QPalette.Window, t["bg"]), (QPalette.WindowText, t["text"]), (QPalette.Base, t["input"]),
                      (QPalette.AlternateBase, t["surface"]), (QPalette.ToolTipBase, t["raised"]),
                      (QPalette.ToolTipText, t["text"]), (QPalette.PlaceholderText, t["text3"]),
                      (QPalette.Text, t["text"]), (QPalette.Button, t["raised"]), (QPalette.ButtonText, t["text"]),
                      (QPalette.BrightText, t["danger"]), (QPalette.Highlight, t["accent"]),
                      (QPalette.HighlightedText, t["accent_text"]), (QPalette.Link, t["accent"]),
                      (QPalette.Light, t["border_strong"]), (QPalette.Midlight, t["border"]),
                      (QPalette.Mid, t["border"]), (QPalette.Dark, t["bg"]), (QPalette.Shadow, "#000000")):
        p.setColor(role, c(col))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        p.setColor(QPalette.Disabled, role, c(t["text3"]))
    return p


_ICONS = {
    "check": '<path d="M3 8.5l3 3 7-7" fill="none" stroke="{c}" stroke-width="2.2" stroke-linecap="round" '
             'stroke-linejoin="round"/>',
    "up": '<path d="M3 10l5-5 5 5" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" '
          'stroke-linejoin="round"/>',
    "down": '<path d="M3 6l5 5 5-5" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" '
            'stroke-linejoin="round"/>',
}


def _icon_dir(theme_name: str, t: dict) -> str:
    """Write the few SVG glyphs the stylesheet needs (Qt stylesheets can only reference image files)."""
    d = os.path.join(tempfile.gettempdir(), f"uapka-theme-{theme_name}")
    os.makedirs(d, exist_ok=True)
    for key, body in _ICONS.items():
        for suffix, col in (("", t["text2"]), ("_on", t["accent_text"]), ("_off", t["text3"])):
            svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16">' \
                  f'{body.format(c=col)}</svg>'
            path = os.path.join(d, f"{key}{suffix}.svg")
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(svg)
            except OSError:
                pass
    return d.replace(os.sep, "/")


def stylesheet(t: dict, icons: str = "") -> str:
    mono = QFontDatabase.systemFont(QFontDatabase.FixedFont).family()
    return f"""
QMainWindow, QDialog {{ background: {t['bg']}; }}
QWidget {{ color: {t['text']}; }}
QToolTip {{ background: {t['raised']}; color: {t['text']}; border: 1px solid {t['border_strong']}; padding: 4px; }}

QMenuBar {{ background: {t['bg']}; border-bottom: 1px solid {t['border']}; padding: 2px 4px; }}
QMenuBar::item {{ padding: 4px 10px; background: transparent; border-radius: 3px; }}
QMenuBar::item:selected {{ background: {t['raised']}; }}
QMenu {{ background: {t['raised']}; border: 1px solid {t['border_strong']}; padding: 4px; }}
QMenu::item {{ padding: 5px 24px 5px 20px; border-radius: 3px; }}
QMenu::item:selected {{ background: {t['accent_dim']}; color: {t['text']}; }}
QMenu::separator {{ height: 1px; background: {t['border']}; margin: 4px 6px; }}

QStatusBar {{ background: {t['surface']}; border-top: 1px solid {t['border']}; color: {t['text2']}; }}
QStatusBar::item {{ border: none; }}
QSplitter::handle {{ background: {t['border']}; }}
QSplitter::handle:horizontal {{ width: 1px; }}

QTabWidget::pane {{ border: 1px solid {t['border']}; border-top: none; background: {t['surface']}; }}
QTabWidget#workflow > QTabBar::tab {{ min-width: 0; }}
QTabBar {{ qproperty-drawBase: 0; }}
QTabBar::tab {{ background: {t['bg']}; color: {t['text2']}; padding: 7px 11px; border: 1px solid {t['border']};
    border-bottom: 2px solid {t['border']}; margin-right: -1px; }}
QTabBar::tab:hover {{ color: {t['text']}; background: {t['surface']}; }}
QTabBar::tab:selected {{ color: {t['text']}; background: {t['surface']}; border-bottom: 2px solid {t['accent']}; }}
QTabBar::tab:disabled {{ color: {t['text3']}; }}

QGroupBox {{ background: {t['surface']}; border: 1px solid {t['border']}; border-radius: 4px;
    margin-top: 16px; padding: 10px 8px 6px 8px; }}
QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; left: 8px; top: 2px; padding: 0 2px;
    color: {t['text2']}; font-weight: 600; letter-spacing: 1px; }}

QPushButton {{ background: {t['raised']}; border: 1px solid {t['border_strong']}; border-radius: 4px;
    padding: 5px 8px; color: {t['text']}; }}
QPushButton:hover {{ border-color: {t['accent']}; }}
QPushButton:pressed {{ background: {t['accent_dim']}; }}
QPushButton:checked {{ background: {t['accent_dim']}; border-color: {t['accent']}; color: {t['text']}; }}
QPushButton:disabled {{ color: {t['text3']}; border-color: {t['border']}; background: {t['surface']}; }}
QPushButton#primary {{ background: {t['accent']}; color: {t['accent_text']}; border: 1px solid {t['accent']};
    font-weight: 600; padding: 7px 14px; }}
QPushButton#primary:hover {{ background: {t['accent']}; border-color: {t['text']}; }}
QPushButton#primary:disabled {{ background: {t['accent_dim']}; color: {t['text3']}; border-color: {t['accent_dim']}; }}

QToolButton#transport {{ background: {t['raised']}; border: 1px solid {t['border_strong']}; border-radius: 4px;
    min-width: 28px; min-height: 24px; color: {t['text']}; }}
QToolButton#transport:hover {{ border-color: {t['accent']}; color: {t['accent']}; }}
QToolButton#transport:disabled {{ color: {t['text3']}; border-color: {t['border']}; }}

QLineEdit, QAbstractSpinBox, QComboBox {{ background: {t['input']}; border: 1px solid {t['border_strong']};
    border-radius: 3px; padding: 3px 5px; selection-background-color: {t['accent']};
    selection-color: {t['accent_text']}; }}
QLineEdit:focus, QAbstractSpinBox:focus, QComboBox:focus {{ border-color: {t['accent']}; }}
QAbstractSpinBox {{ padding-right: 18px; }}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{ subcontrol-origin: border; width: 16px;
    border-left: 1px solid {t['border']}; background: {t['raised']}; }}
QAbstractSpinBox::up-button {{ subcontrol-position: top right; border-top-right-radius: 3px; }}
QAbstractSpinBox::down-button {{ subcontrol-position: bottom right; border-bottom-right-radius: 3px;
    border-top: 1px solid {t['border']}; }}
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {{ background: {t['accent_dim']}; }}
QAbstractSpinBox::up-arrow {{ image: url({icons}/up.svg); width: 9px; height: 9px; }}
QAbstractSpinBox::down-arrow {{ image: url({icons}/down.svg); width: 9px; height: 9px; }}
QAbstractSpinBox::up-arrow:disabled, QAbstractSpinBox::up-arrow:off {{ image: url({icons}/up_off.svg); }}
QAbstractSpinBox::down-arrow:disabled, QAbstractSpinBox::down-arrow:off {{ image: url({icons}/down_off.svg); }}
QComboBox {{ padding-right: 20px; }}
QComboBox::drop-down {{ subcontrol-origin: padding; subcontrol-position: center right; width: 18px; border: none; }}
QComboBox::down-arrow {{ image: url({icons}/down.svg); width: 10px; height: 10px; }}
QComboBox::down-arrow:disabled {{ image: url({icons}/down_off.svg); }}
QCheckBox {{ spacing: 7px; }}
QCheckBox::indicator {{ width: 14px; height: 14px; border: 1px solid {t['border_strong']}; border-radius: 3px;
    background: {t['input']}; }}
QCheckBox::indicator:hover {{ border-color: {t['accent']}; }}
QCheckBox::indicator:checked {{ background: {t['accent']}; border-color: {t['accent']};
    image: url({icons}/check_on.svg); }}
QCheckBox::indicator:disabled {{ border-color: {t['border']}; background: {t['surface']}; }}
QComboBox QAbstractItemView {{ background: {t['raised']}; border: 1px solid {t['border_strong']};
    selection-background-color: {t['accent_dim']}; selection-color: {t['text']}; }}

QPlainTextEdit, QTextEdit, QListWidget, QTableWidget {{ background: {t['input']}; border: 1px solid {t['border']};
    border-radius: 3px; selection-background-color: {t['accent_dim']}; selection-color: {t['text']}; }}
QPlainTextEdit, QTextEdit {{ font-family: "{mono}"; padding: 4px; }}
QListWidget {{ font-family: "{mono}"; }}
QListWidget::item {{ padding: 2px 4px; }}
QTableWidget {{ gridline-color: {t['border']}; font-family: "{mono}"; }}
QHeaderView::section {{ background: {t['surface']}; color: {t['text2']}; border: none;
    border-right: 1px solid {t['border']}; border-bottom: 1px solid {t['border']}; padding: 4px 6px; }}
QTableCornerButton::section {{ background: {t['surface']}; border: none; }}

QScrollArea {{ border: 1px solid {t['border']}; background: {t['plot_bg']}; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle {{ background: {t['border_strong']}; border-radius: 4px; min-height: 24px; min-width: 24px; }}
QScrollBar::handle:hover {{ background: {t['text3']}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}

QSlider::groove:horizontal {{ height: 4px; background: {t['border']}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {t['accent']}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {t['text']}; border: 2px solid {t['accent']}; width: 10px;
    margin: -6px 0; border-radius: 7px; }}
QSlider::handle:horizontal:disabled {{ background: {t['border_strong']}; border-color: {t['border']}; }}

QProgressBar {{ background: {t['input']}; border: 1px solid {t['border']}; border-radius: 3px; height: 12px;
    text-align: center; color: {t['text']}; }}
QProgressBar::chunk {{ background: {t['accent']}; border-radius: 2px; }}

QLabel#hint {{ color: {t['text2']}; }}
QLabel#readout {{ font-family: "{mono}"; color: {t['text']}; }}
QFrame#assumedBanner {{ background: {t['surface']}; border: 1px solid {t['assumed']};
    border-left: 4px solid {t['assumed']}; border-radius: 3px; }}
QFrame#assumedBanner QLabel {{ color: {t['text']}; }}
"""


def apply(app, theme_name: str) -> None:
    """Set the Fusion style, palette and stylesheet for the whole application."""
    global _current
    _current = theme_name if theme_name in THEMES else "dark"
    t = current()
    app.setStyle("Fusion")
    app.setPalette(_palette(t))
    app.setStyleSheet(stylesheet(t, _icon_dir(_current, t)))


# ------------------------------------------------------------------ provenance-aware rich text
def chip_html(label: str) -> str:
    col = provenance_color(label).name()
    style = f"color:{col}; font-weight:600;"
    if label.endswith("ASSUMED"):
        style += " font-style:italic;"
    return f'<span style="{style}">[{html.escape(label)}]</span>'


def provenance_html(text: str) -> str:
    """Plain multi-line text -> HTML in which every [LABEL] is coloured by provenance and ⚠ lines stand out."""
    t = current()
    out = []
    for line in text.splitlines():
        esc = html.escape(line)
        esc = _LABEL_RE.sub(lambda m: chip_html(m.group(1)), esc)
        if line.startswith("⚠") or line.startswith("WARNINGS"):
            esc = f'<span style="color:{t["assumed"]};">{esc}</span>'
        elif line.startswith("→"):
            esc = f'<span style="color:{t["text2"]};">{esc}</span>'
        out.append(esc or "&nbsp;")
    return (f'<div style="color:{t["text"]}; white-space:pre-wrap;">' + "<br>".join(out) + "</div>")


# ------------------------------------------------------------------ matplotlib figures shown in the GUI
def _hex(rgba) -> str:
    from matplotlib.colors import to_hex
    return to_hex(rgba, keep_alpha=False).lower()


def _color_map(t: dict) -> dict:
    from .. import plots as P
    m = {P.SURFACE: t["plot_bg"], P.TEXT: t["text"], P.TEXT2: t["text2"], P.GRID: t["plot_grid"],
         P.LOWCONF: t["plot_lowconf"], "#b9b8b2": t["plot_spine"], "#ffffff": t["plot_bg"], "#000000": t["text"],
         P.C1: t["plot_series"][0], P.C2: t["plot_series"][1], P.C3: t["plot_series"][2]}
    return {k.lower(): v for k, v in m.items()}


def restyle_figure(fig) -> None:
    """Recolour a uapka.plots figure for the current theme. Idempotent: the original colours are
    remembered on each artist, so switching themes back and forth never drifts."""
    import numpy as np
    from matplotlib.collections import Collection
    from matplotlib.colors import to_rgba
    from matplotlib.image import AxesImage
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.text import Text

    t = current()
    cmap = _color_map(t)

    def mapped(rgba):
        rgba = to_rgba(rgba)
        if rgba[3] == 0:
            return rgba
        new = cmap.get(_hex(rgba))
        return rgba if new is None else to_rgba(new, rgba[3])

    attrs = {Line2D: ("color", "markerfacecolor", "markeredgecolor"), Text: ("color",),
             Patch: ("facecolor", "edgecolor"), Collection: ("facecolor", "edgecolor")}
    for art in fig.findobj():
        if isinstance(art, AxesImage):
            continue
        for kind, names in attrs.items():
            if not isinstance(art, kind):
                continue
            orig = getattr(art, "_uapka_orig", None)
            if orig is None:
                orig = {}
                for n in names:
                    try:
                        v = getattr(art, f"get_{n}")()
                    except (AttributeError, TypeError, ValueError):
                        continue
                    if isinstance(v, str) and v in ("none", "auto"):
                        continue
                    orig[n] = v
                art._uapka_orig = orig
            for n, v in orig.items():
                try:
                    if kind is Collection:
                        arr = np.asarray(v)
                        if arr.size == 0 or art.get_array() is not None:
                            continue  # colour-mapped data (e.g. time-coloured path): leave alone
                        getattr(art, f"set_{n}")([mapped(c) for c in arr.reshape(-1, 4)])
                    else:
                        getattr(art, f"set_{n}")(mapped(v))
                except (AttributeError, TypeError, ValueError):
                    pass
    # tick and grid styles that matplotlib re-applies to ticks it creates later (zoom / pan)
    for ax in fig.axes:
        ax.tick_params(which="both", colors=t["text2"], labelcolor=t["text2"], grid_color=t["plot_grid"])
        leg = ax.get_legend()
        if leg is not None:
            for txt in leg.get_texts():
                txt.set_color(t["text2"])
    fig.set_facecolor(t["plot_bg"])
    if fig.canvas is not None:
        fig.canvas.draw_idle()
