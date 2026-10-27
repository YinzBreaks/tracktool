"""Single-object tracker: user marks the object once, the tracker follows it.

Per frame: predict with the camera motion (if estimated) plus the object's own
recent motion, search with normalised cross-correlation against a slowly
updated template, refine to a sub-pixel intensity-weighted centroid, and fall
back to blob re-acquisition when the match is weak. Masked overlay pixels are
neutralised inside the template and search window so a reticle or HUD line
cannot hold the track.

Manual correction: every user click is a keyframe. Tracking runs forward from
each keyframe up to the next one (and backward from the first keyframe), so a
correction at frame N re-tracks only from N onwards.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import cv2
import numpy as np

from .camera import CameraTrack
from .detect import detect_blobs
from .masks import Mask, fill_masked, mask_image

OK, MANUAL, REACQ, LOST, NONE, EXCLUDED = "OK", "MANUAL", "REACQ", "LOST", "NONE", "EXCLUDED"
USABLE = (OK, MANUAL, REACQ)


@dataclass
class TrackParams:
    search_radius_px: int = 24
    lost_threshold: float = 0.45
    template_update: float = 0.05
    max_lost: int = 15
    centroid_refine: bool = True
    snap_manual: bool = True
    polarity: str = "auto"          # auto | bright | dark
    preset: str = "sky"             # background preset for re-acquisition
    reacquire: bool = True
    manual_sigma_px: float = 1.0


@dataclass
class Track:
    n: int
    keyframes: dict = field(default_factory=dict)   # frame -> (cx, cy, w, h), user-placed
    excluded: set = field(default_factory=set)      # frames the user marked unusable (occlusion etc.)

    def __post_init__(self):
        self.reset_results()

    def reset_results(self):
        n = self.n
        self.x = np.full(n, np.nan)
        self.y = np.full(n, np.nan)
        self.w = np.full(n, np.nan)
        self.h = np.full(n, np.nan)
        self.score = np.full(n, np.nan)
        self.sigma = np.full(n, np.nan)
        self.size_px = np.full(n, np.nan)
        self.status = np.array([NONE] * n, dtype=object)
        self.polarity = "bright"

    def usable(self) -> np.ndarray:
        return np.isin(self.status, USABLE) & np.isfinite(self.x)

    def box(self, k: int) -> Optional[tuple]:
        if 0 <= k < self.n and np.isfinite(self.x[k]) and self.status[k] != NONE:
            return (self.x[k], self.y[k], self.w[k], self.h[k])
        return None

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "keyframes": {str(k): list(map(float, v)) for k, v in self.keyframes.items()},
            "excluded": sorted(int(k) for k in self.excluded),
            "polarity": self.polarity,
            "results": {
                "x": _nan_list(self.x), "y": _nan_list(self.y), "w": _nan_list(self.w), "h": _nan_list(self.h),
                "score": _nan_list(self.score), "sigma": _nan_list(self.sigma), "size_px": _nan_list(self.size_px),
                "status": list(self.status),
            },
        }

    @staticmethod
    def from_dict(d: dict) -> "Track":
        t = Track(int(d["n"]), {int(k): tuple(v) for k, v in d.get("keyframes", {}).items()},
                  set(int(k) for k in d.get("excluded", [])))
        r = d.get("results")
        if r and len(r.get("x", [])) == t.n:
            for key in ("x", "y", "w", "h", "score", "sigma", "size_px"):
                setattr(t, key, np.array([np.nan if v is None else v for v in r[key]], float))
            t.status = np.array(r["status"], dtype=object)
            t.polarity = d.get("polarity", "bright")
        return t


def _nan_list(a: np.ndarray) -> list:
    return [None if not np.isfinite(v) else float(v) for v in a]


def _odd(v: float, lo: int = 9) -> int:
    v = max(lo, int(round(v)))
    return v if v % 2 else v + 1


def _patch(img: np.ndarray, cx: float, cy: float, w: int, h: int) -> np.ndarray:
    return cv2.getRectSubPix(img, (w, h), (float(cx), float(cy)))


def _polarity(img: np.ndarray, cx: float, cy: float, w: int, h: int) -> str:
    p = _patch(img, cx, cy, w * 2 + 1, h * 2 + 1).astype(np.float32)
    H, W = p.shape
    inner = p[H // 2 - h // 4:H // 2 + h // 4 + 1, W // 2 - w // 4:W // 2 + w // 4 + 1]
    ring = np.concatenate([p[:2].ravel(), p[-2:].ravel(), p[:, :2].ravel(), p[:, -2:].ravel()])
    return "bright" if inner.mean() >= np.median(ring) else "dark"


def refine_centroid(img: np.ndarray, x: float, y: float, w: float, h: float, polarity: str,
                    excluded: Optional[np.ndarray] = None) -> Optional[tuple[float, float, float]]:
    """Intensity-weighted centroid around (x, y). Returns (cx, cy, diameter_px) or None if no clear blob."""
    r = int(max(w, h) * 0.75) + 3
    H, W = img.shape
    x0, y0 = int(round(x)) - r, int(round(y)) - r
    x1, y1 = x0 + 2 * r + 1, y0 + 2 * r + 1
    if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
        return None
    f = img[y0:y1, x0:x1].astype(np.float32)
    border = np.concatenate([f[:2].ravel(), f[-2:].ravel(), f[2:-2, :2].ravel(), f[2:-2, -2:].ravel()])
    bg = float(np.median(border))
    noise = 1.4826 * float(np.median(np.abs(border - bg))) + 0.5
    sig = (f - bg) * (1.0 if polarity == "bright" else -1.0)
    if excluded is not None:
        sig[excluded[y0:y1, x0:x1] > 0] = 0
    pk = float(sig.max())
    if pk < 4 * noise:
        return None
    wts = np.clip(sig - 0.25 * pk, 0, None)
    s = wts.sum()
    ys, xs = np.mgrid[y0:y1, x0:x1]
    cx, cy = float((wts * xs).sum() / s), float((wts * ys).sum() / s)
    area = float((sig > 0.5 * pk).sum())
    return cx, cy, 2.0 * np.sqrt(area / np.pi)


def _subpixel_peak(res: np.ndarray, loc: tuple[int, int]) -> tuple[float, float]:
    x, y = loc
    dx = dy = 0.0
    if 0 < x < res.shape[1] - 1:
        l, c, r = res[y, x - 1], res[y, x], res[y, x + 1]
        den = l - 2 * c + r
        dx = 0.5 * (l - r) / den if den < 0 else 0.0
    if 0 < y < res.shape[0] - 1:
        u, c, d = res[y - 1, x], res[y, x], res[y + 1, x]
        den = u - 2 * c + d
        dy = 0.5 * (u - d) / den if den < 0 else 0.0
    return x + float(np.clip(dx, -0.5, 0.5)), y + float(np.clip(dy, -0.5, 0.5))


def _apply(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    return T[:2, :2] @ p + T[:2, 2]


def _track_segment(frames, track: Track, excl: np.ndarray, camera: Optional[CameraTrack], params: TrackParams,
                   k0: int, k_end: int, step: int, cancel=None) -> None:
    H, W = frames[0].shape
    cx, cy, bw, bh = track.keyframes[k0]
    tw, th = _odd(bw), _odd(bh)
    img0 = frames[k0]
    pol = params.polarity if params.polarity in ("bright", "dark") else _polarity(img0, cx, cy, tw, th)
    track.polarity = pol
    sigma0 = params.manual_sigma_px
    if params.snap_manual:
        rc = refine_centroid(img0, cx, cy, bw, bh, pol, excl)
        if rc is not None and np.hypot(rc[0] - cx, rc[1] - cy) < max(bw, bh) / 2:
            cx, cy = rc[0], rc[1]
            sigma0 = min(sigma0, 0.5)
            track.size_px[k0] = rc[2]
    track.x[k0], track.y[k0], track.w[k0], track.h[k0] = cx, cy, bw, bh
    track.status[k0], track.score[k0], track.sigma[k0] = MANUAL, 1.0, sigma0
    tmpl = fill_masked(_patch(img0, cx, cy, tw, th), _patch(excl, cx, cy, tw, th)).astype(np.float32)

    p = np.array([cx, cy])
    own = np.zeros(2)          # object motion relative to background, px/frame (in direction of travel)
    n_lost = 0
    k = k0 + step
    while (k <= k_end if step > 0 else k >= k_end):
        if cancel and cancel():
            raise InterruptedError("Cancelled")
        if camera is not None:
            T = camera.T(k) if step > 0 else np.linalg.inv(camera.T(k + 1))
        else:
            T = np.eye(3)
        p_cam = _apply(T, p)
        pred = p_cam + own
        track.w[k], track.h[k] = bw, bh
        if k in track.excluded:
            track.x[k], track.y[k], track.status[k] = pred[0], pred[1], EXCLUDED
            p = pred
            k += step
            continue
        img = frames[k]
        r = int(params.search_radius_px + 0.5 * np.hypot(*own))
        sw, sh = tw + 2 * r, th + 2 * r
        x0, y0 = int(round(pred[0])) - sw // 2, int(round(pred[1])) - sh // 2
        if x0 + sw < tw or y0 + sh < th or x0 > W - tw or y0 > H - th:
            track.x[k], track.y[k], track.status[k] = pred[0], pred[1], LOST
            n_lost += 1
            if n_lost > params.max_lost:
                break
            p = pred
            k += step
            continue
        win = _patch(img, x0 + (sw - 1) / 2, y0 + (sh - 1) / 2, sw, sh)
        wex = _patch(excl, x0 + (sw - 1) / 2, y0 + (sh - 1) / 2, sw, sh)
        win = fill_masked(win, wex).astype(np.float32)
        res = cv2.matchTemplate(win, tmpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        px, py = _subpixel_peak(res, loc)
        cand = np.array([x0 + px + (tw - 1) / 2, y0 + py + (th - 1) / 2])
        status = OK
        if score < params.lost_threshold and params.reacquire:
            roi = (pred[0] - 2 * r - tw, pred[1] - 2 * r - th, 4 * r + 2 * tw, 4 * r + 2 * th)
            cands = [c for c in detect_blobs(img, pol, params.preset, excl, roi, max_candidates=8)
                     if c.area <= 4 * tw * th]
            best = None
            for c in cands:
                pt = _patch(img, c.x, c.y, tw, th).astype(np.float32)
                s = float(cv2.matchTemplate(pt, tmpl, cv2.TM_CCOEFF_NORMED)[0, 0])
                d = np.hypot(c.x - pred[0], c.y - pred[1]) / (r + 1)
                val = s - 0.1 * d
                if s >= 0.7 * params.lost_threshold and (best is None or val > best[0]):
                    best = (val, s, c)
            if best is not None:
                cand, score, status = np.array([best[2].x, best[2].y]), best[1], REACQ
        if score < params.lost_threshold and status != REACQ:
            track.x[k], track.y[k], track.status[k], track.score[k] = pred[0], pred[1], LOST, score
            n_lost += 1
            if n_lost > params.max_lost:
                break
            own *= 0.9
            p = pred
            k += step
            continue
        if params.centroid_refine:
            rc = refine_centroid(img, cand[0], cand[1], bw, bh, pol, excl)
            if rc is not None and np.hypot(rc[0] - cand[0], rc[1] - cand[1]) < max(tw, th) / 2:
                cand = np.array(rc[:2])
                track.size_px[k] = rc[2]
        n_lost = 0
        track.x[k], track.y[k], track.status[k], track.score[k] = cand[0], cand[1], status, score
        sig = float(np.clip(0.15 + 0.85 * (1 - score), 0.15, 2.0))
        track.sigma[k] = sig * (1.5 if status == REACQ else 1.0)
        own = 0.7 * own + 0.3 * (cand - p_cam)
        p = cand
        if status == OK and score > 0.75 and params.template_update > 0:
            new = fill_masked(_patch(img, cand[0], cand[1], tw, th), _patch(excl, cand[0], cand[1], tw, th))
            tmpl = (1 - params.template_update) * tmpl + params.template_update * new.astype(np.float32)
        k += step


def run_tracking(frames, track: Track, masks: list[Mask], camera: Optional[CameraTrack], params: TrackParams,
                 progress=None, cancel=None) -> Track:
    """(Re)compute the whole track from the keyframes."""
    if not track.keyframes:
        raise ValueError("Mark the object in at least one frame first.")
    track.reset_results()
    excl = mask_image(masks, frames[0].shape)
    kfs = sorted(track.keyframes)
    n = len(frames)
    segs = [(kfs[0], 0, -1)] + [(k, (kfs[i + 1] - 1 if i + 1 < len(kfs) else n - 1), 1) for i, k in enumerate(kfs)]
    for j, (k0, k_end, step) in enumerate(segs):
        if progress:
            progress(j / len(segs), f"Tracking from keyframe {k0}")
        _track_segment(frames, track, excl, camera, params, k0, k_end, step, cancel)
    for k in track.excluded:
        if 0 <= k < n:
            track.status[k] = EXCLUDED
    if progress:
        progress(1.0, f"Tracked: {int(track.usable().sum())}/{n} frames usable")
    return track
