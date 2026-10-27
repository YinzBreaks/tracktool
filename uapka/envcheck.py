"""Environment self-check for uapka. Run it when something fails and paste the output into a bug report.

    python tools/check_env.py            (or: .venv\\Scripts\\python tools\\check_env.py on Windows)
    uapka check-env                      (also from the frozen build: uapka-cli.exe check-env)

Checks, without changing anything: Python and library versions, OpenCV's FFmpeg backend, whether the
video writers uapka needs (FFV1, HFYU, mp4v, MJPG) open, whether a Qt platform plugin loads, ffprobe
availability, console encoding, free disk in the frame-cache folder, available RAM and the RAM budget,
and whether a memory-mapped cache file can be deleted while still mapped (it cannot on Windows; uapka
handles that, this just reports it). Output is plain ASCII on purpose.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tempfile

FAILS = 0
WARNS = 0


def line(status: str, text: str) -> None:
    global FAILS, WARNS
    if status == "FAIL":
        FAILS += 1
    elif status == "WARN":
        WARNS += 1
    print(f"[{status:4}] {text}")


def section(title: str) -> None:
    print(f"\n== {title}")


def check_python() -> None:
    section("Python")
    line("INFO", f"Python {platform.python_version()} ({platform.architecture()[0]}) at {sys.executable}")
    line("INFO", f"Platform: {platform.platform()}; frozen build: {bool(getattr(sys, 'frozen', False))}")
    if sys.version_info < (3, 10):
        line("FAIL", "uapka needs Python 3.10 or newer")
    if sys.maxsize < 2 ** 32:
        line("WARN", "32-bit interpreter: large clips will need the disk cache; install 64-bit Python")
    for name in ("stdout", "stderr"):
        st = getattr(sys, name)
        enc = getattr(st, "encoding", None) if st is not None else None
        tag = "OK" if st is not None and (enc or "").lower().replace("-", "") == "utf8" else "INFO"
        line(tag, f"sys.{name}: encoding={enc} errors={getattr(st, 'errors', None) if st else None}"
                  + (" (None: windowed build; output goes to the log file)" if st is None else ""))
    if os.environ.get("PYTHONUTF8") != "1":
        line("INFO", "PYTHONUTF8 is not set (run_uapka.bat sets it; uapka reconfigures its own streams anyway)")


def check_libs() -> None:
    section("Libraries")
    for mod in ("numpy", "scipy", "matplotlib", "PySide6", "shiboken6", "cv2"):
        try:
            m = __import__(mod)
            line("OK", f"{mod} {getattr(m, '__version__', '?')}")
        except Exception as e:  # noqa: BLE001
            line("FAIL", f"{mod} cannot be imported: {type(e).__name__}: {e}")
    try:
        import cv2
        for bad in ("PyQt5", "PyQt6"):
            try:
                __import__(bad)
                line("WARN", f"{bad} is installed alongside PySide6; if the GUI crashes at start, uninstall it")
            except ImportError:
                pass
        qt_dir = os.path.join(os.path.dirname(cv2.__file__), "qt")
        if os.path.isdir(qt_dir):
            line("FAIL", "cv2 ships its own Qt plugins (opencv-python); uninstall it and install opencv-python-headless")
        else:
            line("OK", "cv2 has no bundled Qt plugins (opencv-python-headless)")
    except ImportError:
        pass
    try:
        from PySide6.QtCore import qVersion
        line("INFO", f"Qt runtime {qVersion()}")
    except Exception:  # noqa: BLE001
        pass


def check_opencv_video() -> None:
    section("OpenCV video I/O")
    try:
        import cv2
        import numpy as np
    except ImportError:
        line("FAIL", "cv2/numpy missing; skipping video checks")
        return
    info = cv2.getBuildInformation()
    ff = [ln.strip() for ln in info.splitlines() if ln.strip().startswith("FFMPEG:")]
    line("OK" if ff and "YES" in ff[0] else "FAIL", f"FFmpeg backend: {ff[0] if ff else 'not listed'}")
    try:
        backends = [cv2.videoio_registry.getBackendName(b) for b in cv2.videoio_registry.getWriterBackends()]
        line("INFO", "Writer backends: " + ", ".join(backends))
    except Exception:  # noqa: BLE001
        pass
    d = tempfile.mkdtemp(prefix="uapka_chk_")
    try:
        for codec, ext, what in (("FFV1", "avi", "lossless synth clips (tests)"), ("HFYU", "avi", "lossless fallback"),
                                 ("mp4v", "mp4", "annotated video export"), ("MJPG", "avi", "lossy fallback")):
            fn = os.path.join(d, f"w_{codec}.{ext}")
            w = cv2.VideoWriter(fn, cv2.VideoWriter_fourcc(*codec), 30.0, (64, 48))
            ok = w.isOpened()
            if ok:
                for _ in range(3):
                    w.write(np.zeros((48, 64, 3), np.uint8))
            w.release()
            n = int(cv2.VideoCapture(fn).get(cv2.CAP_PROP_FRAME_COUNT)) if ok else 0
            good = ok and n == 3
            line("OK" if good else ("WARN" if codec in ("HFYU", "MJPG") else "FAIL"),
                 f"writer {codec} -> .{ext}: {'opens, 3 frames read back' if good else 'cannot be opened'} ({what})")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def check_qt_platform() -> None:
    section("Qt platform plugin")
    env = dict(os.environ)
    if os.name != "nt" and sys.platform != "darwin" and not (env.get("DISPLAY") or env.get("WAYLAND_DISPLAY")):
        env.setdefault("QT_QPA_PLATFORM", "offscreen")
        line("INFO", "no DISPLAY: testing with QT_QPA_PLATFORM=offscreen")
    # Qt aborts the whole process when no platform plugin loads, so the probe runs in a child process:
    # the interpreter with -c, or (frozen build, where sys.executable is the exe) the exe itself with --qt-probe.
    if getattr(sys, "frozen", False):
        cmd = [sys.executable, "check-env", "--qt-probe"]
    else:
        cmd = [sys.executable, "-c", "from uapka.envcheck import qt_probe; qt_probe()"]
        env["PYTHONPATH"] = os.pathsep.join(p for p in [os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                        env.get("PYTHONPATH", "")] if p)
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=env, creationflags=flags,
                           encoding="utf-8", errors="replace")
    except (OSError, subprocess.TimeoutExpired) as e:
        line("FAIL", f"could not run the Qt probe: {e}")
        return
    out = (r.stdout or "").strip().splitlines()
    if r.returncode == 0 and out:
        for ln in out:
            line("OK", ln)
    else:
        err = (r.stderr or "").strip().splitlines()
        line("FAIL", f"QApplication could not start (exit {r.returncode}): " + (err[-1] if err else "no message"))
        for ln in err[-6:-1]:
            line("INFO", "  " + ln)
        line("INFO", "  Windows: install the Microsoft Visual C++ Redistributable (x64); set QT_DEBUG_PLUGINS=1 for detail")


def qt_probe() -> None:
    """Child-process body of check_qt_platform: create a QApplication and print what loaded."""
    from PySide6.QtGui import QImageReader
    from PySide6.QtWidgets import QApplication
    a = QApplication([])
    print("platform", a.platformName())
    print("imageformats", b",".join(bytes(f) for f in QImageReader.supportedImageFormats()).decode())


def check_ffprobe() -> None:
    section("ffprobe (optional)")
    exe = shutil.which("ffprobe")
    if not exe:
        line("INFO", "ffprobe not on PATH: metadata panel shows OpenCV fields only (decoding does not need it)")
        return
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    try:
        r = subprocess.run([exe, "-version"], capture_output=True, text=True, timeout=15, creationflags=flags,
                           encoding="utf-8", errors="replace")
        line("OK", f"{exe}: {(r.stdout or '').splitlines()[0] if r.stdout else 'runs'}")
    except (OSError, subprocess.TimeoutExpired) as e:
        line("WARN", f"ffprobe found at {exe} but failed to run: {e}")


def check_storage() -> None:
    section("Frame cache, RAM and disk")
    try:
        from uapka import video
    except Exception as e:  # noqa: BLE001
        line("FAIL", f"uapka cannot be imported: {e}")
        return
    d = video.default_cache_dir()
    line("INFO", f"cache folder: {d} (override with UAPKA_CACHE_DIR)")
    try:
        du = shutil.disk_usage(d)
        line("OK" if du.free > 2 * 1024 ** 3 else "WARN",
             f"free disk there: {du.free / 1e9:.1f} GB (a 1080p clip needs about 2 MB per frame on disk)")
    except OSError as e:
        line("FAIL", f"cannot stat the cache folder: {e}")
    avail = video.available_memory_bytes()
    line("INFO", f"available RAM: {'unknown' if avail is None else f'{avail / 1e9:.1f} GB'}; "
                 f"in-RAM budget {video.default_ram_budget_bytes() / 1e9:.2f} GB (override with UAPKA_RAM_BUDGET_MB)")
    # memmap delete semantics
    import numpy as np
    fd, fn = tempfile.mkstemp(prefix="uapka_chk_", suffix=".u8", dir=d)
    os.close(fd)
    try:
        mm = np.memmap(fn, dtype=np.uint8, mode="w+", shape=(4, 8, 8))
        view = mm[:2]
        try:
            os.remove(fn)
            line("INFO", "a mapped cache file can be deleted while mapped (POSIX semantics)")
        except OSError:
            line("INFO", "a mapped cache file cannot be deleted while mapped (Windows semantics; uapka retries at exit)")
        del view, mm
        import gc
        gc.collect()
        if os.path.exists(fn):
            os.remove(fn)
        line("OK", "cache file created, mapped and removed")
    except Exception as e:  # noqa: BLE001
        line("FAIL", f"memmap test failed in {d}: {type(e).__name__}: {e}")
        try:
            os.remove(fn)
        except OSError:
            pass


def main(argv=None) -> int:
    if argv and "--qt-probe" in argv:
        qt_probe()
        return 0
    global FAILS, WARNS
    FAILS = WARNS = 0
    try:
        from uapka.cli import configure_stdio
        configure_stdio()
    except Exception:  # noqa: BLE001
        pass
    check_python()
    check_libs()
    check_opencv_video()
    check_qt_platform()
    check_ffprobe()
    check_storage()
    print()
    print(f"Summary: {FAILS} failure(s), {WARNS} warning(s)")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
