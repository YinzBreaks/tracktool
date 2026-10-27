"""Maskable regions: HUD symbology, reticles, watermarks, redaction blocks.

Masked pixels are excluded from background registration and blob detection,
and are neutralised (filled with the local median) inside tracking templates
so that a static reticle cannot "hold" the tracker in place.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

MASK_KINDS = ("HUD", "RETICLE", "WATERMARK", "REDACTION", "OTHER")


@dataclass
class Mask:
    x: int
    y: int
    w: int
    h: int
    kind: str = "HUD"
    label: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Mask":
        return Mask(int(d["x"]), int(d["y"]), int(d["w"]), int(d["h"]), d.get("kind", "OTHER"), d.get("label", ""))


def mask_image(masks: list[Mask], shape: tuple[int, int]) -> np.ndarray:
    """uint8 image, 255 where pixels are EXCLUDED."""
    m = np.zeros(shape[:2], np.uint8)
    H, W = shape[:2]
    for r in masks:
        x0, y0 = max(0, r.x), max(0, r.y)
        x1, y1 = min(W, r.x + r.w), min(H, r.y + r.h)
        if x1 > x0 and y1 > y0:
            m[y0:y1, x0:x1] = 255
    return m


def fill_masked(img: np.ndarray, excluded: np.ndarray) -> np.ndarray:
    """Replace excluded pixels with the median of the remaining ones."""
    if excluded is None or not excluded.any():
        return img
    out = img.copy()
    keep = excluded == 0
    fill = np.median(img[keep]) if keep.any() else 0
    out[~keep] = fill
    return out


def suggest_static_overlays(frames, n_samples: int = 60, min_area: int = 30,
                            exclude_box: tuple[int, int, int, int] | None = None) -> tuple[list[Mask], str]:
    """Suggest masks for burned-in overlays by finding pixels that never change
    while the rest of the image does (HUD text, reticles, logos, redaction bars).

    Returns (suggested masks, message). Only meaningful when the camera moves:
    for a locked-off camera everything is static and nothing is suggested.
    """
    n = len(frames)
    if n < 5:
        return [], "Need at least 5 frames."
    idx = np.unique(np.linspace(0, n - 1, min(n_samples, n)).astype(int))
    stack = np.stack([frames[i] for i in idx]).astype(np.float32)
    std = stack.std(axis=0)
    mean = stack.mean(axis=0)
    moving_level = float(np.median(std))
    if moving_level < 2.0:
        return [], "Background barely changes over the clip (static camera?); no overlay suggestions."
    static = (std < max(1.0, 0.15 * moving_level)).astype(np.uint8)
    # Only keep static pixels that are structured (text/lines) or part of flat blocks (redactions).
    gx = cv2.Sobel(mean, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(mean, cv2.CV_32F, 0, 1)
    grad = cv2.magnitude(gx, gy)
    structured = cv2.dilate((grad > 20).astype(np.uint8), np.ones((5, 5), np.uint8))
    flat_block = cv2.erode(static, np.ones((9, 9), np.uint8))
    cand = (static & structured) | flat_block
    # merge fragments (letters of a HUD word, arms of a reticle) into one region
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((17, 17), np.uint8))
    cand = cv2.dilate(cand, np.ones((5, 5), np.uint8))
    if exclude_box is not None:
        x, y, w, h = exclude_box
        cand[max(0, y):y + h, max(0, x):x + w] = 0
    if cand.mean() > 0.5:
        return [], "More than half the frame is static; camera probably not moving. No suggestions."
    num, _, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    H, W = cand.shape
    out = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        inner = (slice(y + h // 4, y + h - h // 4), slice(x + w // 4, x + w - w // 4))
        uniform = float(std[inner].mean()) < 1.0 and float(mean[inner].std()) < 3.0
        centred = abs(x + w / 2 - W / 2) < 0.05 * W and abs(y + h / 2 - H / 2) < 0.05 * H
        kind = "REDACTION" if (uniform and w * h > 0.01 * W * H) else ("RETICLE" if centred else "HUD")
        out.append(Mask(int(x), int(y), int(w), int(h), kind, "auto-suggested"))
    return out, f"{len(out)} static overlay region(s) suggested from {len(idx)} sampled frames."
