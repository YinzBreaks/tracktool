"""Fixed-pattern-noise (FPN) estimation and removal.

Why this matters: IR sensors (and many CCD/CMOS sensors under low light) have a
static per-pixel offset pattern, column/row striping and hot/dead pixels. That
pattern does NOT move when the camera pans. On a low-texture background (clear
sky, IR sea) the FPN can carry more high-frequency energy than the scene, so a
global method like phase correlation finds its strongest peak at zero shift:
it registers the sensor to itself and reports "the camera did not move". Every
apparent object motion is then attributed to the object.

Estimate: high-pass every sampled frame and take the per-pixel temporal median.
Scene content moves (camera pans) and averages away; content fixed to the
sensor survives. Caveat: if the camera itself is static, scene detail survives
too, which the `static_fraction` diagnostic reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class FPNModel:
    pattern: np.ndarray            # float32 HxW, additive high-frequency pattern
    defects: np.ndarray            # uint8 HxW, 255 = hot/dead pixel
    static_fraction: float         # rms(pattern)/rms(high-passed frames); ~1 => static camera or very strong FPN
    n_samples: int
    hp_sigma: float

    def describe(self) -> str:
        s = (f"FPN map from {self.n_samples} frames, static fraction {self.static_fraction:.2f}, "
             f"{int((self.defects > 0).sum())} defect pixels.")
        if self.static_fraction > 0.8:
            s += (" WARNING: most high-frequency detail is static. Either the camera does not move "
                  "(then disable FPN removal) or fixed-pattern noise dominates the image.")
        return s


def blur(f: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur; large sigmas are done on a decimated image (same result, ~50x faster)."""
    if sigma <= 8:
        return cv2.GaussianBlur(f, (0, 0), sigma)
    k = max(1, int(sigma // 4))
    H, W = f.shape[:2]
    small = cv2.resize(f, (max(1, W // k), max(1, H // k)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / k)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)


def highpass(img: np.ndarray, sigma: float) -> np.ndarray:
    """Remove slow illumination gradients/vignetting; keeps scene texture down to ~sigma px."""
    f = img.astype(np.float32)
    return f - blur(f, sigma)


def estimate_fpn(frames, excluded: Optional[np.ndarray] = None, n_samples: int = 100, hp_sigma: float = 30.0,
                 mem_limit_bytes: int = 400 * 1024**2) -> FPNModel:
    n = len(frames)
    H, W = frames[0].shape[:2]
    n_use = int(max(3, min(n_samples, n, mem_limit_bytes // (H * W * 4))))
    idx = np.unique(np.linspace(0, n - 1, n_use).astype(int))
    stack = np.empty((len(idx), H, W), np.float32)
    for j, i in enumerate(idx):
        stack[j] = highpass(frames[i], hp_sigma)
    pattern = np.median(stack, axis=0).astype(np.float32)
    keep = np.ones((H, W), bool) if excluded is None else excluded == 0
    rms_frames = float(np.sqrt(np.mean(stack[:, keep] ** 2)))
    rms_pattern = float(np.sqrt(np.mean(pattern[keep] ** 2)))
    static_fraction = rms_pattern / max(rms_frames, 1e-6)
    med = np.median(pattern[keep])
    mad = 1.4826 * np.median(np.abs(pattern[keep] - med)) + 1e-6
    defects = _defects(pattern, med, mad, keep)
    del stack
    return FPNModel(pattern, defects, static_fraction, len(idx), hp_sigma)


def _defects(pattern: np.ndarray, med: float, mad: float, keep: np.ndarray, max_frac: float = 0.005) -> np.ndarray:
    """Hot/dead pixels: > 6 robust sigma, capped at the worst `max_frac` of pixels (heavy-tailed
    patterns such as vignetting residue would otherwise flag large areas and starve registration)."""
    z = np.abs(pattern - med) / mad
    z[~keep] = 0
    thr = max(6.0, float(np.quantile(z[keep], 1 - max_frac))) if keep.any() else 6.0
    return ((z > thr) & keep).astype(np.uint8) * 255


def clean_frame(img: np.ndarray, fpn: Optional[FPNModel], destripe: bool = True, hp_sigma: float = 30.0,
                smooth_sigma: float = 2.0) -> np.ndarray:
    """Band-passed, FPN-subtracted, destriped float32 frame (zero-mean texture image).

    The final smoothing matters as much as the FPN subtraction: whatever pixel-scale
    pattern survives subtraction (temporal noise, compression residue of the FPN,
    which a lossy encoder partially "freezes" in place) is white and sensor-locked,
    while low-texture IR scene content (cloud/sea structure) lives at lower spatial
    frequencies. Smoothing a few pixels suppresses the former and keeps the latter.
    """
    sigma = fpn.hp_sigma if fpn is not None else hp_sigma
    hp = highpass(img, sigma)
    if fpn is not None:
        hp -= fpn.pattern
        if fpn.defects.any():
            hp[fpn.defects > 0] = 0.0
    if destripe:
        # Column/row offsets that vary frame to frame (common in uncooled IR).
        hp -= np.median(hp[::4, :], axis=0, keepdims=True)
        hp -= np.median(hp[:, ::4], axis=1, keepdims=True)
    if smooth_sigma > 0:
        hp = cv2.GaussianBlur(hp, (0, 0), smooth_sigma)
    return hp


def to_uint8_pair(a: np.ndarray, b: np.ndarray, clahe: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Map two float texture images to uint8 with a SHARED mapping (keeps brightness constancy for LK)."""
    both = np.concatenate([a[::4, ::4].ravel(), b[::4, ::4].ravel()])
    lo, hi = np.percentile(both, [0.5, 99.5])
    scale = 255.0 / max(hi - lo, 1e-3)
    a8 = np.clip((a - lo) * scale, 0, 255).astype(np.uint8)
    b8 = np.clip((b - lo) * scale, 0, 255).astype(np.uint8)
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        a8, b8 = c.apply(a8), c.apply(b8)
    return a8, b8


def refine_fpn_with_motion(frames, A: np.ndarray, fpn: FPNModel, excluded: Optional[np.ndarray] = None,
                           n_samples: int = 80) -> FPNModel:
    """Scene-based FPN re-estimation using a first camera-motion estimate.

    A plain temporal median only separates scene from sensor pattern when the
    camera has panned several texture lengths. With motion known, the scene can
    be modelled explicitly: warp FPN-corrected frames into reference coordinates,
    average them into a scene mosaic, project the mosaic back into each frame and
    take the median of (frame - scene) as the sensor pattern.
    A: (N,3,3) frame -> reference transforms.
    """
    n = len(frames)
    H, W = frames[0].shape[:2]
    idx = np.unique(np.linspace(0, n - 1, min(n, n_samples)).astype(int))
    corners = np.array([[0, 0, 1], [W, 0, 1], [0, H, 1], [W, H, 1]], float).T
    pts = np.concatenate([(A[i] @ corners)[:2].T for i in idx])
    x0, y0 = np.floor(pts.min(0)).astype(int) - 2
    x1, y1 = np.ceil(pts.max(0)).astype(int) + 2
    MW, MH = int(x1 - x0), int(y1 - y0)
    if MW * MH > 64e6:      # absurd mosaic (bad motion estimate): keep the original
        return fpn
    shift = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], float)
    acc = np.zeros((MH, MW), np.float32)
    cnt = np.zeros((MH, MW), np.float32)
    ones = np.ones((H, W), np.float32)
    if excluded is not None:
        ones[excluded > 0] = 0
    for i in idx:
        f = highpass(frames[i], fpn.hp_sigma) - fpn.pattern
        M = (shift @ A[i])[:2]
        acc += cv2.warpAffine(f * ones, M, (MW, MH), flags=cv2.INTER_LINEAR)
        cnt += cv2.warpAffine(ones, M, (MW, MH), flags=cv2.INTER_LINEAR)
    scene = acc / np.maximum(cnt, 1e-3)
    stack = np.empty((len(idx), H, W), np.float32)
    for j, i in enumerate(idx):
        Minv = np.linalg.inv(shift @ A[i])[:2]
        back = cv2.warpAffine(scene, Minv, (W, H), flags=cv2.INTER_LINEAR)
        stack[j] = highpass(frames[i], fpn.hp_sigma) - back
    pattern = np.median(stack, axis=0).astype(np.float32)
    keep = np.ones((H, W), bool) if excluded is None else excluded == 0
    med = np.median(pattern[keep])
    mad = 1.4826 * np.median(np.abs(pattern[keep] - med)) + 1e-6
    defects = _defects(pattern, med, mad, keep)
    return FPNModel(pattern, defects, fpn.static_fraction, len(idx), fpn.hp_sigma)
