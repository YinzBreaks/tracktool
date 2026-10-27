# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for UAPKA: one folder, windowed GUI plus a console CLI.

    pyinstaller --noconfirm --clean uapka.spec            (or build_windows.bat)

Produces dist/UAPKA/ with
    UAPKA(.exe)       windowed (no console window) - the GUI; double-click or pass a video/project path
    uapka-cli(.exe)   console - `uapka-cli info|synth|analyze ...`; a windowed exe has no stdout on Windows
Both executables share one _internal/ folder, so the build is not twice the size.
"""

import os

from PyInstaller.utils.hooks import collect_submodules

HERE = os.path.abspath(SPECPATH)

hiddenimports = collect_submodules("uapka")
hiddenimports += [
    # matplotlib: figures are built on Figure + explicit canvases; make sure both canvases are frozen
    "matplotlib.backends.backend_qtagg",
    "matplotlib.backends.backend_agg",
    # scipy modules used at runtime via lazy/local imports (pyinstaller's static analysis can miss them)
    "scipy.signal", "scipy.ndimage", "scipy.optimize", "scipy.special", "scipy.special._ufuncs",
    "scipy._lib.messagestream", "scipy.sparse.csgraph._validation",
    # Qt modules used by the GUI (the PySide6 hooks pull in their platform and imageformats plugins)
    "PySide6.QtCore", "PySide6.QtGui", "PySide6.QtWidgets",
]

excludes = [
    # GUI toolkits we do not use (tkinter is matplotlib's default backend; PyQt would clash with PySide6)
    "tkinter", "_tkinter", "PyQt5", "PyQt6", "PySide2",
    # Heavy Qt modules that nothing imports; listed so a stray transitive import cannot drag them in
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick", "PySide6.QtWebChannel",
    "PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtQuickWidgets", "PySide6.QtQuick3D", "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets", "PySide6.Qt3DCore", "PySide6.Qt3DRender", "PySide6.QtPdf", "PySide6.QtPdfWidgets",
    "PySide6.QtCharts", "PySide6.QtDataVisualization", "PySide6.QtLocation", "PySide6.QtPositioning",
    "PySide6.QtBluetooth", "PySide6.QtNfc", "PySide6.QtSensors", "PySide6.QtSerialPort", "PySide6.QtRemoteObjects",
    "PySide6.QtDesigner", "PySide6.QtHelp", "PySide6.QtTest", "PySide6.QtUiTools", "PySide6.QtSql",
    "PySide6.QtNetworkAuth", "PySide6.QtWebSockets", "PySide6.QtHttpServer", "PySide6.QtTextToSpeech",
    "PySide6.QtSpatialAudio", "PySide6.QtGraphs", "PySide6.QtScxml", "PySide6.QtStateMachine",
    # optional dependencies of numpy/scipy/matplotlib that nothing here uses (static analysis would drag them in)
    "pandas", "lxml", "pyarrow", "h5py", "sympy", "numba", "tornado", "zmq", "PIL.ImageTk", "PIL.ImageQt",
    # dev / test tooling
    "pytest", "_pytest", "IPython", "jupyter", "notebook", "setuptools", "pip", "wheel",
]

# Qt libraries and plugins the PySide6 hooks collect for QtGui/QtWidgets but that a Widgets-only app never loads
# (name fragments match both Linux .so and Windows .dll spellings). Qt skips a plugin whose library is absent.
QT_PRUNE = ("Qt6Qml", "Qt6Quick", "Qt6VirtualKeyboard", "qtvirtualkeyboardplugin", "Qt6Pdf", "libqpdf", "qpdf.dll",
            "Qt6Labs", "Qt6ShaderTools")


def _wanted(entry):
    dest = entry[0].replace("\\", "/")
    return "PySide6" not in dest or not any(frag in os.path.basename(dest) for frag in QT_PRUNE)

a = Analysis(
    [os.path.join(HERE, "uapka_launcher.py")],
    pathex=[HERE],
    binaries=[],
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

# opencv-python-headless is required: the full opencv-python wheel ships its own Qt plugins (cv2/qt/plugins),
# which would be collected next to PySide6's and crash the GUI at startup. Refuse to build in that case.
_cv2_qt = [b for b in a.binaries if os.sep + "qt" + os.sep in b[0].replace("/", os.sep) and b[0].startswith("cv2")]
if _cv2_qt:
    raise SystemExit("cv2 ships Qt plugins: uninstall opencv-python and install opencv-python-headless before building.")

a.binaries = [b for b in a.binaries if _wanted(b)]
a.datas = [d for d in a.datas if _wanted(d)]

pyz = PYZ(a.pure)

icon = os.path.join(HERE, "uapka.ico")
icon = icon if os.path.isfile(icon) else None

exe_gui = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="UAPKA",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,          # windowed: no console window behind the GUI
    disable_windowed_traceback=False,
    icon=icon,
)
exe_cli = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="uapka-cli",
    debug=False,
    strip=False,
    upx=False,
    console=True,           # console: info / synth / analyze print to the terminal
    icon=icon,
)
coll = COLLECT(
    exe_gui, exe_cli,
    a.binaries, a.datas,
    strip=False,
    upx=False,
    name="UAPKA",
)
