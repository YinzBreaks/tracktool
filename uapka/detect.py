"""Automatic detection of bright/dark blobs against sky, sea or terrain.

Each background preset is a different estimate of "what the background would
look like without the object" plus a different false-alarm threshold:
  sky     - smooth background; large median filter, low threshold
  sea     - glint clutter; opening removes single-pixel glints, higher threshold
  terrain - textured clutter; morphological top-hat at object scale, highest threshold
The detection statistic is local contrast in units of a robust (MAD) noise sigma.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

PRESETS = {
    "sky": dict(kernel=31, k_sigma=4.5, min_area=2, open_px=0),
    "sea": dict(kernel=21, k_sigma=6.0, min_area=4, open_px=3),
    "terrain": dict(kernel=15, k_sigma=7.0, min_area=4, open_px=3),
}


@dataclass
class Candidate:
    x: float
    y: float
    w: int
    h: int
    area: int
    contrast: float      # peak contrast, grey levels
    snr: float           # peak contrast / robust noise sigma
    polarity: str        # "bright" | "dark"

    @property
    def score(self) -> float:
        return self.snr * np.sqrt(self.area)


def _background(img: np.ndarray, preset: str) -> np.ndarray:
    p = PRESETS[preset]
    k = p["kernel"] | 1
    if preset == "terrain":
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        # opening/closing estimate the background under bright/dark objects respectively
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, se), cv2.morphologyEx(img, cv2.MORPH_CLOSE, se)
    bg = cv2.medianBlur(img, min(k, 255))
    return bg, bg


def detect_blobs(img: np.ndarray, polarity: str = "both", preset: str = "sky",
                 excluded: Optional[np.ndarray] = None, roi: Optional[tuple[int, int, int, int]] = None,
                 max_candidates: int = 20, max_area_frac: float = 0.05) -> list[Candidate]:
    """Find compact blobs. roi = (x, y, w, h) restricts the search (coords stay full-frame)."""
    x0 = y0 = 0
    if roi is not None:
        x, y, w, h = roi
        H, W = img.shape[:2]
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(W, int(x + w)), min(H, int(y + h))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return []
        img = img[y0:y1, x0:x1]
        if excluded is not None:
            excluded = excluded[y0:y1, x0:x1]
    p = PRESETS[preset]
    bg_bright, bg_dark = _background(img, preset)
    f = img.astype(np.float32)
    keep = np.ones(img.shape, bool) if excluded is None else excluded == 0
    out: list[Candidate] = []
    for pol, diff in (("bright", f - bg_bright.astype(np.float32)), ("dark", bg_dark.astype(np.float32) - f)):
        if polarity not in ("both", pol):
            continue
        vals = diff[keep]
        if vals.size < 50:
            continue
        med = float(np.median(vals))
        sigma = 1.4826 * float(np.median(np.abs(vals - med))) + 0.5
        det = ((diff - med) > p["k_sigma"] * sigma) & keep
        det = det.astype(np.uint8)
        if p["open_px"]:
            det = cv2.morphologyEx(det, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        num, lab, stats, cents = cv2.connectedComponentsWithStats(det, connectivity=8)
        max_area = max_area_frac * img.size
        for i in range(1, num):
            x, y, w, h, area = stats[i]
            if area < p["min_area"] or area > max_area:
                continue
            region = lab[y:y + h, x:x + w] == i
            dpatch = diff[y:y + h, x:x + w]
            peak = float(dpatch[region].max() - med)
            wts = np.clip(dpatch - med, 0, None) * region
            ys, xs = np.mgrid[y:y + h, x:x + w]
            sw = wts.sum()
            cx = float((wts * xs).sum() / sw) if sw > 0 else cents[i][0]
            cy = float((wts * ys).sum() / sw) if sw > 0 else cents[i][1]
            out.append(Candidate(cx + x0, cy + y0, int(w), int(h), int(area), peak, peak / sigma, pol))
    out.sort(key=lambda c: -c.score)
    return out[:max_candidates]
