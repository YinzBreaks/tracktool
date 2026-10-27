"""Video ingest: metadata, evidence hashing and an exact random-access frame cache.

OpenCV seeking (CAP_PROP_POS_FRAMES) is not frame-accurate for many codecs,
which silently corrupts kinematics when a user corrects the track at frame N
and the tool re-reads "frame N". So the video is decoded once, sequentially,
into a grayscale cache (in RAM, or a disk memmap for long clips) and every
later stage indexes that cache.
"""

from __future__ import annotations

import atexit
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

ProgressFn = Callable[[float, str], None]
CancelFn = Callable[[], bool]


@dataclass
class VideoMeta:
    path: str
    width: int
    height: int
    fps: float
    frame_count_reported: int
    frame_count_decoded: int = 0
    duration_s: float = 0.0
    fourcc: str = ""
    sha256: str = ""
    file_size: int = 0
    ffprobe: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary_lines(self) -> list[str]:
        lines = [
            f"File: {os.path.basename(self.path)}",
            f"SHA-256: {self.sha256}",
            f"Size: {self.file_size / 1e6:.2f} MB",
            f"Resolution: {self.width} x {self.height}",
            f"FPS (container): {self.fps:.3f}",
            f"Frames: {self.frame_count_decoded or self.frame_count_reported}"
            + ("" if not self.frame_count_decoded or self.frame_count_decoded == self.frame_count_reported
               else f" decoded ({self.frame_count_reported} reported)"),
            f"Duration: {self.duration_s:.3f} s",
            f"Codec (FourCC): {self.fourcc or 'unknown'}",
        ]
        fp = self.ffprobe
        if fp:
            for key in ("codec_name", "profile", "pix_fmt", "field_order", "r_frame_rate", "avg_frame_rate",
                        "bit_rate", "sample_aspect_ratio", "display_aspect_ratio"):
                if fp.get(key) not in (None, "", "N/A"):
                    lines.append(f"{key}: {fp[key]}")
        return lines


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _fourcc_str(v: float) -> str:
    i = int(v)
    s = "".join(chr((i >> (8 * k)) & 0xFF) for k in range(4))
    return s if s.isprintable() and s.strip() else ""


def run_ffprobe(path: str) -> dict:
    """Optional richer metadata from a local ffprobe binary, if on PATH. Never required.

    shutil.which appends PATHEXT on Windows, so "ffprobe" finds ffprobe.exe. The call never goes
    through a shell, and on Windows CREATE_NO_WINDOW stops a console window flashing up when the
    GUI (a windowed executable) runs it.
    """
    exe = shutil.which("ffprobe")
    if not exe:
        return {}
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    try:
        res = subprocess.run(
            [exe, "-v", "error", "-select_streams", "v:0", "-show_streams", "-of", "json", path],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30, check=False,
            creationflags=flags,
        )
        streams = json.loads(res.stdout or "{}").get("streams", [])
        return streams[0] if streams else {}
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return {}


def probe(path: str, compute_hash: bool = True) -> VideoMeta:
    if not os.path.isfile(path):
        raise IOError(f"No such file: {path}")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        hint = ""
        if os.name == "nt" and not path.isascii():
            hint = (" (the path contains non-ASCII characters; OpenCV's FFmpeg backend on Windows cannot open some "
                    "such paths - copy the file to a plain ASCII path and retry)")
        raise IOError(f"OpenCV could not open video: {path}{hint}")
    try:
        meta = VideoMeta(
            path=os.path.abspath(path),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(cap.get(cv2.CAP_PROP_FPS)) or 0.0,
            frame_count_reported=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            fourcc=_fourcc_str(cap.get(cv2.CAP_PROP_FOURCC)),
        )
    finally:
        cap.release()
    meta.file_size = os.path.getsize(path)
    if meta.fps <= 0 or meta.fps > 1000:
        meta.fps = 30.0  # flagged in warnings by the cache
    meta.duration_s = meta.frame_count_reported / meta.fps if meta.fps else 0.0
    meta.ffprobe = run_ffprobe(path)
    if compute_hash:
        meta.sha256 = sha256_file(path)
    return meta


class FrameCache:
    """Grayscale frames with exact indexing, plus per-frame diagnostics.

    A cache may hold a segment of the file: frame i of the cache is frame
    `offset + i` of the video, and times are reported in seconds from the start
    of the file so they match the original clip.
    """

    def __init__(self, meta: VideoMeta, frames: np.ndarray, pos_ms: np.ndarray, mad_prev: np.ndarray,
                 warnings: list[str], memmap_path: Optional[str] = None, offset: int = 0,
                 overlay_bits: Optional[list] = None, gray_mean: Optional[np.ndarray] = None,
                 gray_std: Optional[np.ndarray] = None):
        self.meta = meta
        self.frames = frames
        self.pos_ms = pos_ms
        self.mad_prev = mad_prev  # mean |frame_k - frame_{k-1}|, NaN for k=0
        self.warnings = warnings
        self._memmap_path = memmap_path
        self.offset = offset
        self._overlay_bits = overlay_bits or []
        n = len(frames)
        self.gray_mean = gray_mean if gray_mean is not None else np.full(n, np.nan)
        self.gray_std = gray_std if gray_std is not None else np.full(n, np.nan)
        self.duplicate = np.zeros(n, bool)
        self.blank = np.zeros(n, bool)

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, k: int) -> np.ndarray:
        return self.frames[k]

    def overlay(self, k: int) -> Optional[np.ndarray]:
        """uint8 mask (255 = coloured overlay pixel) for frame k, or None if no colour overlays were found."""
        if not self._overlay_bits or self._overlay_bits[k] is None:
            return None
        h, w = self.frames.shape[1:3]
        return (np.unpackbits(self._overlay_bits[k])[: h * w].reshape(h, w) * 255).astype(np.uint8)

    def has_overlays(self) -> bool:
        return any(b is not None for b in self._overlay_bits)

    def skip(self) -> np.ndarray:
        """Frames carrying no new image information: duplicates and blank/dropout frames."""
        return self.duplicate | self.blank

    def nominal_times(self) -> np.ndarray:
        return (self.offset + np.arange(len(self))) / self.meta.fps

    def container_times(self) -> Optional[np.ndarray]:
        t = self.pos_ms / 1000.0
        if len(t) < 2 or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
            return None
        return t - t[0] + self.offset / self.meta.fps

    def detect_duplicates(self, rel_thresh: float = 0.1, abs_thresh: float = 0.5) -> np.ndarray:
        """Flag frames that are (near-)copies of their predecessor.

        Frame-rate conversion and re-encoding insert duplicate frames. If they are
        kept, the object appears to stop and then jump, which creates spurious
        acceleration spikes. After lossy re-encoding duplicates are not bit-exact,
        and after e.g. 10->30 fps conversion MOST frames are duplicates, so
        "far below the typical difference" fails. The frame differences are
        therefore split into two clusters in log space; if they separate clearly
        (new-content frames >= 3x the near-copies), the low cluster is duplicates.
        """
        mad = self.mad_prev
        ok = np.isfinite(mad) & ~self.blank & ~np.roll(self.blank, 1)
        dup = np.zeros(len(mad), bool)
        if ok.sum() >= 6:
            lm = np.log10(mad[ok] + 1e-3)
            lo, hi = np.percentile(lm, 10), np.percentile(lm, 90)
            for _ in range(30):
                thr = 0.5 * (lo + hi)
                a, b = lm[lm <= thr], lm[lm > thr]
                if not len(a) or not len(b):
                    break
                lo, hi = a.mean(), b.mean()
            frac_low = float(np.mean(lm <= 0.5 * (lo + hi)))
            if hi - lo > np.log10(3) and 0.05 < frac_low < 0.95 and 10 ** lo < abs_thresh:
                cut = min(10 ** (0.5 * (lo + hi)), 1.0)
                dup[ok] = mad[ok] < cut
                if frac_low > 0.3:
                    # Frame-rate conversion: a regular cadence (e.g. 1 new + 2 copies for 10->30 fps). The
                    # clusters overlap after re-encoding, so within each cadence period keep only the frame
                    # with the largest difference (local maximum at least period-1 frames apart).
                    from scipy.signal import find_peaks
                    period = int(round(1.0 / max(1.0 - frac_low, 1e-3)))
                    filled = np.where(ok, mad, 0.0)
                    peaks, _ = find_peaks(np.concatenate([[0.0], filled[1:]]), distance=max(1, period - 1),
                                          height=0.6 * cut)
                    new = np.zeros(len(mad), bool)
                    new[peaks] = True
                    dup = ok & ~new
                    self._note(f"{frac_low:.0%} of frames are near-duplicates (cadence ~1 new frame per {period}): "
                               "the clip was likely frame-rate converted. Only new-content frames are used.")
            else:
                typical = float(np.median(mad[ok]))
                dup[ok] = (mad[ok] < abs_thresh) & (mad[ok] < rel_thresh * max(typical, 1e-6))
        self.duplicate = dup & ~self.blank
        return self.duplicate

    def detect_blank(self, std_ratio: float = 0.35) -> np.ndarray:
        """Flag dropout frames (uniform/blank, or a sudden brightness jump of the whole frame)."""
        from scipy.ndimage import median_filter
        sd, mu = self.gray_std, self.gray_mean
        blank = np.zeros(len(self), bool)
        if np.isfinite(sd).sum() >= 5:
            sdm = median_filter(np.nan_to_num(sd), size=31, mode="nearest")
            mum = median_filter(np.nan_to_num(mu), size=31, mode="nearest")
            spread = 1.4826 * np.median(np.abs(mu - mum)) + 1.0
            blank = (sd < std_ratio * sdm) | (np.abs(mu - mum) > 8 * spread + 15)
        self.blank = blank
        if blank.any():
            self._note(f"{int(blank.sum())} blank/dropout frames detected; they are skipped by tracking and "
                       "registration and excluded from kinematics.")
        return blank

    def _note(self, msg: str) -> None:
        if msg not in self.warnings:
            self.warnings.append(msg)

    def close(self) -> None:
        """Release the frame store and delete the disk cache, if any.

        On Windows a memory-mapped file cannot be deleted while any mapping of it is alive, and
        `self.frames` is a view (`store[:n]`) of the memmap, as is every frame handed out through
        `__getitem__` that a caller may still hold (the GUI's current frame, a tracker template...).
        So: drop our own references, collect cycles, try to delete, and if the file is still busy
        leave it for an atexit retry rather than failing the caller.
        """
        path, self._memmap_path = self._memmap_path, None
        self.frames = np.zeros((0, 1, 1), np.uint8)
        if path:
            gc.collect()
            _remove_cache_file(path)


_PENDING_REMOVAL: set[str] = set()


def _remove_cache_file(path: str) -> bool:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError:          # PermissionError on Windows while a view is still mapped
        if not _PENDING_REMOVAL:
            atexit.register(_remove_pending)
        _PENDING_REMOVAL.add(path)
        return False
    _PENDING_REMOVAL.discard(path)
    return True


def _remove_pending() -> None:
    gc.collect()
    for path in list(_PENDING_REMOVAL):
        try:
            os.remove(path)
            _PENDING_REMOVAL.discard(path)
        except OSError:
            pass


def default_cache_dir() -> str:
    """Folder for disk frame caches: $UAPKA_CACHE_DIR, else the system temp dir (%TEMP% on Windows)."""
    d = os.environ.get("UAPKA_CACHE_DIR")
    return os.path.abspath(d) if d else tempfile.gettempdir()


def available_memory_bytes() -> Optional[int]:
    """Physical memory not currently in use, without psutil. None if unknown."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                            ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                            ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
                            ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
                            ("ullAvailExtendedVirtual", ctypes.c_uint64)]
            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return int(st.ullAvailPhys)
            return None
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo", encoding="ascii", errors="replace") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
        if hasattr(os, "sysconf") and "SC_AVPHYS_PAGES" in os.sysconf_names:
            return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES"))
    except (OSError, ValueError, AttributeError):
        pass
    return None


def default_ram_budget_bytes() -> int:
    """Bytes of grayscale frames to keep in RAM before switching to a disk memmap.

    $UAPKA_RAM_BUDGET_MB overrides. Otherwise at most 2 GiB, and no more than half of the memory
    currently available (the decoded frames are not the only large array: FPN maps, templates and
    the GUI's colour frames come on top). Never below 256 MiB; a 32-bit interpreter is capped at 1 GiB.
    """
    env = os.environ.get("UAPKA_RAM_BUDGET_MB")
    if env:
        try:
            return max(64, int(float(env))) * 1024 ** 2
        except ValueError:
            pass
    budget = 2 * 1024 ** 3
    avail = available_memory_bytes()
    if avail:
        budget = min(budget, max(256 * 1024 ** 2, avail // 2))
    if sys.maxsize < 2 ** 32:
        budget = min(budget, 1024 ** 3)
    return int(budget)


def sweep_stale_caches(directory: Optional[str] = None, max_age_s: float = 24 * 3600) -> int:
    """Delete uapka_*.u8 cache files older than max_age_s left behind by crashed runs. Busy files are skipped."""
    directory = directory or default_cache_dir()
    n = 0
    try:
        names = os.listdir(directory)
    except OSError:
        return 0
    import time
    now = time.time()
    for name in names:
        if name.startswith("uapka_") and name.endswith(".u8"):
            fn = os.path.join(directory, name)
            try:
                if now - os.path.getmtime(fn) > max_age_s:
                    os.remove(fn)
                    n += 1
            except OSError:
                pass
    return n


def overlay_mask_bgr(frame: np.ndarray, chroma_thresh: int = 14) -> Optional[np.ndarray]:
    """Coloured pixels in otherwise grayscale (IR / monochrome) footage are burned-in symbology."""
    if frame.ndim != 3:
        return None
    ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    chroma = cv2.max(cv2.absdiff(ycc[..., 1], 128), cv2.absdiff(ycc[..., 2], 128))
    m = (chroma > chroma_thresh).astype(np.uint8)
    if m.sum() == 0:
        return None
    return cv2.dilate(m, np.ones((5, 5), np.uint8))


def load_frames(meta: VideoMeta, ram_budget_bytes: Optional[int] = None, cache_dir: Optional[str] = None,
                progress: Optional[ProgressFn] = None, cancel: Optional[CancelFn] = None,
                start_s: float = 0.0, end_s: Optional[float] = None, colour_overlays: bool = True) -> FrameCache:
    """Decode [start_s, end_s] of the video into the cache (sequentially; no seeking).

    Frames stay in RAM up to `ram_budget_bytes` (default: see default_ram_budget_bytes); longer clips go to
    a uint8 memmap file in `cache_dir` (default: see cache_dir()), which is checked for free space first.
    """
    if ram_budget_bytes is None:
        ram_budget_bytes = default_ram_budget_bytes()
    cap = cv2.VideoCapture(meta.path)
    if not cap.isOpened():
        raise IOError(f"OpenCV could not open video: {meta.path}")
    h, w = meta.height, meta.width
    first = max(0, int(round(start_s * meta.fps)))
    last = None if end_s is None or end_s <= 0 else int(round(end_s * meta.fps))
    total = max(meta.frame_count_reported, 1)
    expected = max(1, (min(last, total) if last is not None else total) - first + 1)
    use_memmap = expected * h * w > ram_budget_bytes
    warnings: list[str] = []
    memmap_path = None
    if use_memmap:
        cap_n = int(expected * 1.1) + 16
        directory = cache_dir or default_cache_dir()
        os.makedirs(directory, exist_ok=True)
        sweep_stale_caches(directory)
        need = cap_n * h * w
        try:
            free = shutil.disk_usage(directory).free
        except OSError:
            free = None
        if free is not None and free < need + 256 * 1024 ** 2:
            raise IOError(
                f"The clip needs a {need / 1e9:.1f} GB frame cache on disk but {directory} has only "
                f"{free / 1e9:.1f} GB free. Analyse a segment (start/end time), free space, or set "
                "UAPKA_CACHE_DIR to a drive with room.")
        fd, memmap_path = tempfile.mkstemp(prefix="uapka_", suffix=".u8", dir=directory)
        os.close(fd)
        try:
            store = np.memmap(memmap_path, dtype=np.uint8, mode="w+", shape=(cap_n, h, w))
        except (OSError, ValueError) as e:
            _remove_cache_file(memmap_path)
            raise IOError(f"Could not create the {need / 1e9:.1f} GB frame cache in {directory}: {e}") from e
    else:
        cap_n = None
        store = []
    pos_ms, mad, overlays, gmean, gstd = [], [], [], [], []
    prev = None
    n = 0
    idx = 0
    try:
        while idx < first:          # skip to the segment start without decoding
            if not cap.grab():
                break
            idx += 1
            if progress and idx % 100 == 0:
                progress(0.0, f"Seeking to frame {first} ({idx})")
        while last is None or idx <= last:
            if cancel and cancel():
                raise InterruptedError("Cancelled")
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ov = overlay_mask_bgr(frame) if colour_overlays else None
            if gray.shape != (h, w):
                gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
                ov = None if ov is None else cv2.resize(ov, (w, h), interpolation=cv2.INTER_NEAREST)
                if not any("resolution changes" in x for x in warnings):
                    warnings.append("Frame resolution changes mid-stream; frames were resized to the initial size.")
            if use_memmap:
                if n >= cap_n:
                    warnings.append(f"Video has more frames than the cache allows; truncated at {n}.")
                    break
                store[n] = gray
            else:
                store.append(gray)
            overlays.append(None if ov is None else np.packbits(ov.ravel()))
            sub = gray[::4, ::4]
            gmean.append(float(sub.mean()))
            gstd.append(float(sub.std()))
            pos_ms.append(cap.get(cv2.CAP_PROP_POS_MSEC))
            mad.append(np.nan if prev is None else float(cv2.absdiff(gray, prev).mean()))
            prev = gray
            n += 1
            if progress and n % 10 == 0:
                progress(min(n / expected, 0.99), f"Decoding frame {first + n}")
    except BaseException:
        cap.release()
        if use_memmap:
            del store
            gc.collect()
            _remove_cache_file(memmap_path)
        raise
    finally:
        cap.release()
    if n == 0:
        if use_memmap:
            del store
            gc.collect()
            _remove_cache_file(memmap_path)
        raise IOError("No frames could be decoded in the requested segment.")
    frames = store[:n] if use_memmap else np.stack(store)
    del store                  # keep a single reference chain: fc.frames -> memmap
    if first == 0 and last is None:
        meta.frame_count_decoded = n
        meta.duration_s = n / meta.fps
        if meta.frame_count_reported and abs(meta.frame_count_reported - n) > 1:
            warnings.append(f"Container reports {meta.frame_count_reported} frames but {n} were decoded.")
    else:
        warnings.append(f"Segment analysed: frames {first}-{first + n - 1} "
                        f"({first / meta.fps:.2f}-{(first + n - 1) / meta.fps:.2f} s of the file).")
    if not any(o is not None for o in overlays):
        overlays = []
    fc = FrameCache(meta, frames, np.asarray(pos_ms, float), np.asarray(mad, float), warnings, memmap_path,
                    offset=first, overlay_bits=overlays, gray_mean=np.asarray(gmean), gray_std=np.asarray(gstd))
    ct = fc.container_times()
    if ct is None:
        warnings.append("Container timestamps are missing or non-monotonic; nominal fps timing is used.")
    else:
        dev = np.abs(ct - fc.nominal_times())
        if dev.max() > 0.5 / meta.fps:
            warnings.append(
                f"Container timestamps deviate from nominal fps by up to {dev.max() * 1000:.1f} ms "
                "(variable frame rate?). Consider selecting container timing.")
    fo = (meta.ffprobe or {}).get("field_order")
    if fo and fo not in ("progressive", "unknown"):
        warnings.append(f"Stream is interlaced (field_order={fo}); each frame mixes two time instants.")
    if fc.has_overlays():
        warnings.append("Coloured symbology detected in grayscale video; coloured pixels are masked per frame.")
    if progress:
        progress(1.0, f"Decoded {n} frames")
    return fc


def read_color_frames(path: str, start: int = 0):
    """Sequential BGR frame generator for export (exact order, no seeking)."""
    cap = cv2.VideoCapture(path)
    try:
        for _ in range(start):
            if not cap.grab():
                return
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            yield frame
    finally:
        cap.release()
