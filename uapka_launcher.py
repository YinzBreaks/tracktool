"""Entry script for PyInstaller builds (a package __main__ with relative imports cannot be frozen directly).

The spec (uapka.spec) builds two executables from this one script into the same folder:
  UAPKA.exe      windowed, no console: the GUI (double-click)
  uapka-cli.exe  console: `uapka-cli info|synth|analyze ...` with visible output
"""

import datetime
import faulthandler
import os
import sys
import tempfile


def _windowed_stdio() -> None:
    """A windowed (no-console) executable starts with sys.stdout/sys.stderr = None.

    Anything printed would then raise AttributeError, and an uncaught exception's traceback would
    vanish. Send both streams to a log file (%LOCALAPPDATA%\\uapka\\uapka.log on Windows) so that a
    GUI that fails silently leaves a trace. Nothing is written unless a stream really is missing.
    """
    if sys.stdout is not None and sys.stderr is not None:
        return
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_STATE_HOME") or tempfile.gettempdir()
    path = os.path.join(base, "uapka", "uapka.log")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.isfile(path) and os.path.getsize(path) > 2_000_000:
            os.replace(path, path + ".old")
        f = open(path, "a", encoding="utf-8", errors="replace", buffering=1)
    except OSError:
        return  # uapka.cli.configure_stdio() substitutes a null sink
    if sys.stdout is None:
        sys.stdout = f
    if sys.stderr is None:
        sys.stderr = f
    f.write(f"--- {datetime.datetime.now().isoformat(timespec='seconds')} uapka started, argv={sys.argv!r}\n")


_windowed_stdio()
try:
    faulthandler.enable(file=sys.stderr)   # a native crash (e.g. in Qt or OpenCV) still leaves a Python traceback
except (AttributeError, OSError, ValueError):   # stderr may be a null sink without a file descriptor
    pass

from uapka.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
