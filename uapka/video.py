"""Video ingest: metadata, evidence hashing and an exact random-access frame cache.

OpenCV seeking (CAP_PROP_POS_FRAMES) is not frame-accurate for many codecs,
which silently corrupts kinematics when a user corrects the track at frame N
and the tool re-reads "frame N". So the video is decoded once, sequentially,
into a grayscale cache (in RAM, or a disk memmap for long clips) and every
later stage indexes that cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
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
    """Optional richer metadata from a local ffprobe binary, if on PATH. Never required."""
    exe = shutil.which("ffprobe")
    if not exe:
        return {}
    try:
        res = subprocess.run(
            [exe, "-v", "error", "-select_streams", "v:0", "-show_streams", "-of", "json", path],
            capture_output=True, text=True, timeout=30, check=False,
        )
        streams = json.loads(res.stdout or "{}").get("streams", [])
        return streams[0] if streams else {}
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return {}


def probe(path: str, compute_hash: bool = True) -> VideoMeta:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"OpenCV could not open video: {path}")
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
    """Grayscale frames with exact indexing, plus per-frame timing diagnostics."""

    def __init__(self, meta: VideoMeta, frames: np.ndarray, pos_ms: np.ndarray, mad_prev: np.ndarray,
                 warnings: list[str], memmap_path: Optional[str] = None):
        self.meta = meta
        self.frames = frames
        self.pos_ms = pos_ms
        self.mad_prev = mad_prev  # mean |frame_k - frame_{k-1}|, NaN for k=0
        self.warnings = warnings
        self._memmap_path = memmap_path
        self.duplicate = np.zeros(len(frames), bool)

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, k: int) -> np.ndarray:
        return self.frames[k]

    def nominal_times(self) -> np.ndarray:
        return np.arange(len(self)) / self.meta.fps

    def container_times(self) -> Optional[np.ndarray]:
        t = self.pos_ms / 1000.0
        if len(t) < 2 or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
            return None
        return t - t[0]

    def detect_duplicates(self, rel_thresh: float = 0.1, abs_thresh: float = 0.5) -> np.ndarray:
        """Flag frames that are (near-)copies of their predecessor.

        Frame-rate conversion and re-encoding insert duplicate frames. If they are
        kept, the object appears to stop and then jump, which creates spurious
        acceleration spikes. A frame is a duplicate if its mean absolute difference
        to the previous frame is both tiny in absolute terms and far below typical.
        """
        mad = self.mad_prev
        valid = np.isfinite(mad)
        dup = np.zeros(len(mad), bool)
        if valid.sum() >= 3:
            typical = float(np.median(mad[valid]))
            dup[valid] = (mad[valid] < abs_thresh) & (mad[valid] < rel_thresh * max(typical, 1e-6))
        self.duplicate = dup
        return dup

    def close(self) -> None:
        if self._memmap_path:
            self.frames = np.zeros((0, 1, 1), np.uint8)
            try:
                os.remove(self._memmap_path)
            except OSError:
                pass
            self._memmap_path = None


def load_frames(meta: VideoMeta, ram_budget_bytes: int = 2 * 1024**3, cache_dir: Optional[str] = None,
                progress: Optional[ProgressFn] = None, cancel: Optional[CancelFn] = None) -> FrameCache:
    cap = cv2.VideoCapture(meta.path)
    if not cap.isOpened():
        raise IOError(f"OpenCV could not open video: {meta.path}")
    h, w = meta.height, meta.width
    expected = max(meta.frame_count_reported, 1)
    use_memmap = expected * h * w > ram_budget_bytes
    warnings: list[str] = []
    memmap_path = None
    if use_memmap:
        cap_n = int(expected * 1.1) + 16
        fd, memmap_path = tempfile.mkstemp(prefix="uapka_", suffix=".u8", dir=cache_dir)
        os.close(fd)
        store = np.memmap(memmap_path, dtype=np.uint8, mode="w+", shape=(cap_n, h, w))
    else:
        cap_n = None
        store = []
    pos_ms, mad = [], []
    prev = None
    n = 0
    try:
        while True:
            if cancel and cancel():
                raise InterruptedError("Cancelled")
            ok, frame = cap.read()
            if not ok:
                break
            gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if gray.shape != (h, w):
                gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
                if not any("resolution changes" in x for x in warnings):
                    warnings.append("Frame resolution changes mid-stream; frames were resized to the initial size.")
            if use_memmap:
                if n >= cap_n:
                    warnings.append(f"Video has more frames than the cache allows; truncated at {n}.")
                    break
                store[n] = gray
            else:
                store.append(gray)
            pos_ms.append(cap.get(cv2.CAP_PROP_POS_MSEC))
            mad.append(np.nan if prev is None else float(cv2.absdiff(gray, prev).mean()))
            prev = gray
            n += 1
            if progress and n % 10 == 0:
                progress(min(n / expected, 0.99), f"Decoding frame {n}")
    finally:
        cap.release()
    if n == 0:
        raise IOError("No frames could be decoded.")
    frames = store[:n] if use_memmap else np.stack(store)
    meta.frame_count_decoded = n
    meta.duration_s = n / meta.fps
    if meta.frame_count_reported and abs(meta.frame_count_reported - n) > 1:
        warnings.append(f"Container reports {meta.frame_count_reported} frames but {n} were decoded.")
    fc = FrameCache(meta, frames, np.asarray(pos_ms, float), np.asarray(mad, float), warnings, memmap_path)
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
    if progress:
        progress(1.0, f"Decoded {n} frames")
    return fc


def read_color_frames(path: str):
    """Sequential BGR frame generator for export (exact order, no seeking)."""
    cap = cv2.VideoCapture(path)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            yield frame
    finally:
        cap.release()
