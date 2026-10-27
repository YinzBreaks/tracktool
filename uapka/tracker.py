"""Single-object tracker: user marks the object once, the tracker follows it.

Per frame: predict with the camera motion (if estimated) plus the object's own
recent motion, search with normalised cross-correlation against a slowly
updated template, refine to a sub-pixel intensity-weighted centroid, and fall
back to blob re-acquisition when the match is weak. Masked overlay pixels are
neutralised inside the template and search window so a reticle or HUD line
cannot hold the track.

Two matching modes, chosen automatically from the object's size when marked:
  template  extended objects: NCC template + centroid refinement
  point     objects of a few pixels (typical of real UAP footage): a template of
            a 2 px speck is mostly background and drifts, so instead a matched
            (difference-of-Gaussians) filter is evaluated in the search window,
            candidates above an SNR threshold are scored by SNR minus a penalty
            for distance from the motion prediction, and the frame is declared
            LOST (never guessed) when nothing clears the threshold.

Frames with no new image content (duplicates from frame-rate conversion, blank
dropouts) are stepped over; the object's own motion is predicted per elapsed
frame so gaps do not break the motion model.

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
DUP, BLANK, AMBIG, WEAK = "DUP", "BLANK", "AMBIGUOUS", "WEAK"
USABLE = (OK, MANUAL, REACQ)


def segment_ids(status: np.ndarray, skip: Optional[np.ndarray] = None, gap: int = 2) -> np.ndarray:
    """Continuous-track segment id per frame (-1 = not usable).

    A new segment starts at a re-acquisition, or after `gap` or more consecutive
    unusable frames (lost / weak / ambiguous / excluded). Derivatives must never be
    fitted across a break: the position jump would appear as a huge acceleration.
    """
    n = len(status)
    seg = np.full(n, -1)
    cur, bad_run, started = 0, 0, False
    for k in range(n):
        st = status[k]
        if (skip is not None and skip[k]) or st in (DUP, BLANK, NONE):
            continue
        if st in USABLE:
            if started and (st == REACQ or bad_run >= gap):
                cur += 1
            started = True
            bad_run = 0
            seg[k] = cur
        else:
            bad_run += 1
    return seg


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
    mode: str = "auto"              # auto | template | point
    point_max_diameter_px: float = 6.0   # auto mode: objects at most this size use point mode
    snr_min: float = 4.0            # point mode: detection threshold (matched-filter SNR)
    gate_weight: float = 2.0        # point mode: penalty (in SNR units) for being at the search-radius edge


ROLES = ("target", "control")


@dataclass
class Track:
    n: int
    keyframes: dict = field(default_factory=dict)   # frame -> (cx, cy, w, h), user-placed
    excluded: set = field(default_factory=set)      # frames the user marked unusable (occlusion etc.)
    name: str = "obj1"
    # "target": an object under investigation. "control": a feature known to be fixed to the background
    # (cloud, coastline, surface feature). A control's camera-compensated motion should be zero; whatever it
    # shows is a direct measurement of the camera-compensation error.
    role: str = "target"
    end: int = -1      # last frame (cache index) to track; -1 = no limit (e.g. when the object leaves or splits)
    start: int = -1    # first frame to track (backwards from the first keyframe); -1 = no limit

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
        self.mode = ""

    def usable(self) -> np.ndarray:
        return np.isin(self.status, USABLE) & np.isfinite(self.x)

    def box(self, k: int) -> Optional[tuple]:
        if 0 <= k < self.n and np.isfinite(self.x[k]) and self.status[k] != NONE:
            return (self.x[k], self.y[k], self.w[k], self.h[k])
        return None

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "name": self.name,
            "role": self.role,
            "end": self.end,
            "start": self.start,
            "keyframes": {str(k): list(map(float, v)) for k, v in self.keyframes.items()},
            "excluded": sorted(int(k) for k in self.excluded),
            "polarity": self.polarity,
            "mode": self.mode,
            "results": {
                "x": _nan_list(self.x), "y": _nan_list(self.y), "w": _nan_list(self.w), "h": _nan_list(self.h),
                "score": _nan_list(self.score), "sigma": _nan_list(self.sigma), "size_px": _nan_list(self.size_px),
                "status": list(self.status),
            },
        }

    @staticmethod
    def from_dict(d: dict) -> "Track":
        t = Track(int(d["n"]), {int(k): tuple(v) for k, v in d.get("keyframes", {}).items()},
                  set(int(k) for k in d.get("excluded", [])), d.get("name", "obj1"), d.get("role", "target"),
                  int(d.get("end", -1)), int(d.get("start", -1)))
        r = d.get("results")
        if r and len(r.get("x", [])) == t.n:
            for key in ("x", "y", "w", "h", "score", "sigma", "size_px"):
                setattr(t, key, np.array([np.nan if v is None else v for v in r[key]], float))
            t.status = np.array(r["status"], dtype=object)
            t.polarity = d.get("polarity", "bright")
            t.mode = d.get("mode", "")
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


def point_candidates(img: np.ndarray, excl: Optional[np.ndarray], pred: np.ndarray, r: int, polarity: str,
                     psf_sigma: float, snr_min: float) -> list[tuple[float, float, float]]:
    """Matched-filter (DoG) detections near `pred`: list of (x, y, snr), strongest first."""
    H, W = img.shape
    s1 = max(0.7, psf_sigma)
    s2 = 3.0 * s1 + 1.5
    pad = int(np.ceil(3 * s2))
    R = r + pad
    x0, y0 = int(round(pred[0])) - R, int(round(pred[1])) - R
    x1, y1 = x0 + 2 * R + 1, y0 + 2 * R + 1
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if cx1 - cx0 < 2 * pad + 3 or cy1 - cy0 < 2 * pad + 3:
        return []
    f = img[cy0:cy1, cx0:cx1].astype(np.float32)
    m = None if excl is None else excl[cy0:cy1, cx0:cx1] > 0
    if m is not None and m.any():
        f = f.copy()
        f[m] = float(np.median(f[~m])) if (~m).any() else 0.0
    resp = cv2.GaussianBlur(f, (0, 0), s1) - cv2.GaussianBlur(f, (0, 0), s2)
    if polarity == "dark":
        resp = -resp
    if m is not None and m.any():
        # the filter responds to mask EDGES (e.g. a black redaction block next to bright sky): guard band
        g = 2 * int(np.ceil(s2)) + 1
        resp[cv2.dilate(m.astype(np.uint8), np.ones((g, g), np.uint8)) > 0] = 0
    noise = 1.4826 * float(np.median(np.abs(resp - np.median(resp)))) + 1e-3
    peaks = (resp >= cv2.dilate(resp, np.ones((3, 3), np.uint8))) & (resp > snr_min * noise)
    ys, xs = np.nonzero(peaks)
    out = []
    for y, x in zip(ys, xs):
        gx, gy = x + cx0, y + cy0
        if abs(gx - pred[0]) > r or abs(gy - pred[1]) > r:
            continue
        # sub-pixel peak (parabola) on the filter response
        dx = dy = 0.0
        if 0 < x < resp.shape[1] - 1:
            a, b, c = resp[y, x - 1], resp[y, x], resp[y, x + 1]
            den = a - 2 * b + c
            dx = float(np.clip(0.5 * (a - c) / den, -0.5, 0.5)) if den < 0 else 0.0
        if 0 < y < resp.shape[0] - 1:
            a, b, c = resp[y - 1, x], resp[y, x], resp[y + 1, x]
            den = a - 2 * b + c
            dy = float(np.clip(0.5 * (a - c) / den, -0.5, 0.5)) if den < 0 else 0.0
        out.append((gx + dx, gy + dy, float(resp[y, x] / noise)))
    out.sort(key=lambda c: -c[2])
    return out


def _track_segment(frames, track: Track, excl_static: np.ndarray, camera: Optional[CameraTrack],
                   params: TrackParams, k0: int, k_end: int, step: int, cancel=None,
                   dup: Optional[np.ndarray] = None, blank: Optional[np.ndarray] = None, overlay=None,
                   avoid=None) -> None:
    H, W = frames[0].shape

    def free(cands, k):
        """Drop detections already claimed by another tracked object at this frame."""
        if avoid is None:
            return cands
        taken = avoid(k)
        return [c for c in cands if all(np.hypot(c[0] - x, c[1] - y) > 6 for x, y in taken)]

    def excl_at(k):
        ov = overlay(k) if overlay is not None else None
        return excl_static if ov is None else (excl_static | ov)

    cx, cy, bw, bh = track.keyframes[k0]
    tw, th = _odd(bw), _odd(bh)
    img0 = frames[k0]
    excl0 = excl_at(k0)
    if params.polarity in ("bright", "dark"):
        pol = params.polarity
    else:
        # strongest matched-filter response of either polarity inside the marked box; the centre-vs-ring test
        # alone fails when a click is a few pixels off a 2 px target
        best = {pp: point_candidates(img0, excl0, np.array([cx, cy]), int(max(bw, bh) / 2) + 2, pp, 1.0,
                                     params.snr_min) for pp in ("bright", "dark")}
        snr = {pp: (c[0][2] if c else 0.0) for pp, c in best.items()}
        pol = max(snr, key=snr.get) if max(snr.values()) > 0 else _polarity(img0, cx, cy, tw, th)
    track.polarity = pol
    sigma0 = params.manual_sigma_px
    diam = None
    if params.snap_manual:
        rc = refine_centroid(img0, cx, cy, bw, bh, pol, excl0)
        if rc is not None and np.hypot(rc[0] - cx, rc[1] - cy) < max(bw, bh) / 2:
            cx, cy = rc[0], rc[1]
            sigma0 = min(sigma0, 0.5)
            track.size_px[k0] = diam = rc[2]
    # size at the strongest matched-filter detection, measured with a small window (a generous user box on a
    # noisy background inflates the size estimate of a 2 px target)
    small = None
    pc0 = point_candidates(img0, excl0, np.array([cx, cy]), int(max(bw, bh)), pol, 1.0, params.snr_min)
    if pc0:
        rc2 = refine_centroid(img0, pc0[0][0], pc0[0][1], 7, 7, pol, excl0)
        small = rc2[2] if rc2 is not None else None
    sizes = [d for d in (diam, small) if d is not None]
    if params.mode == "auto" and track.mode in ("point", "template"):
        mode = track.mode            # decided once per object (at its first keyframe), not per segment
    elif params.mode == "point" or (params.mode == "auto" and sizes and min(sizes) <= params.point_max_diameter_px):
        mode = "point"
    elif params.mode == "auto" and not sizes and max(bw, bh) <= 12:
        mode = "point"
    else:
        mode = "template"
    recent_snr: list[float] = []
    if mode == "point" and params.snap_manual:
        # snap the click to the strongest matched-filter detection inside the marked box
        cands = point_candidates(img0, excl0, np.array([cx, cy]), int(max(bw, bh)), pol, 1.0, params.snr_min)
        if cands:
            cx, cy = cands[0][0], cands[0][1]
            recent_snr.append(cands[0][2])
            sigma0 = min(sigma0, 0.5)
            rc = refine_centroid(img0, cx, cy, 5, 5, pol, excl0)
            if rc is not None and np.hypot(rc[0] - cx, rc[1] - cy) < 1.5:
                cx, cy, diam = rc[0], rc[1], rc[2]
                track.size_px[k0] = diam
    track.mode = mode
    psf_sigma = max(0.7, (diam or 2.0) / 2.355)
    track.x[k0], track.y[k0], track.w[k0], track.h[k0] = cx, cy, bw, bh
    track.status[k0], track.score[k0], track.sigma[k0] = MANUAL, (1.0 if mode == "template" else np.nan), sigma0
    tmpl = fill_masked(_patch(img0, cx, cy, tw, th), _patch(excl0, cx, cy, tw, th)).astype(np.float32)

    p = np.array([cx, cy])
    k_last = k0
    own = np.zeros(2)          # object motion relative to background, px per frame
    n_lost = 0
    k = k0 + step
    while (k <= k_end if step > 0 else k >= k_end):
        if cancel and cancel():
            raise InterruptedError("Cancelled")
        track.w[k], track.h[k] = bw, bh
        if dup is not None and dup[k]:
            # no new image: hold the last measurement (not used for kinematics)
            track.x[k], track.y[k], track.status[k] = track.x[k_last], track.y[k_last], DUP
            k += step
            continue
        dk = abs(k - k_last)
        T = camera.between(k_last, k) if camera is not None else np.eye(3)
        p_cam = _apply(T, p)
        pred = p_cam + own * dk
        if (blank is not None and blank[k]) or k in track.excluded:
            track.x[k], track.y[k] = pred
            track.status[k] = BLANK if (blank is not None and blank[k]) else EXCLUDED
            k += step
            continue
        img = frames[k]
        excl = excl_at(k)
        r = int(params.search_radius_px + 0.5 * np.hypot(*own) * dk + (2 * n_lost if mode == "point" else 0))
        status, score, cand, sig = LOST, np.nan, None, np.nan
        if mode == "point":
            # Two motion hypotheses: continue (pred) or stop/reverse (p_cam: background motion only).
            # Constant-velocity prediction alone overshoots exactly at the turns an analyst cares about.
            centre = 0.5 * (pred + p_cam)
            r_search = int(r + 0.5 * np.hypot(*(pred - p_cam)))
            cands = free(point_candidates(img, excl, centre, r_search, pol, psf_sigma, params.snr_min), k)
            ref_snr = float(np.median(recent_snr[-5:])) if recent_snr else None
            best = None
            if cands:
                d = np.array([min(np.hypot(c[0] - pred[0], c[1] - pred[1]),
                                  1.3 * np.hypot(c[0] - p_cam[0], c[1] - p_cam[1])) for c in cands])
                snr = np.array([c[2] for c in cands])
                sc = snr - params.gate_weight * params.snr_min * (d / max(r, 1)) ** 2
                if ref_snr is not None:   # prefer object-like brightness over brighter or dimmer clutter
                    sc -= params.snr_min * np.abs(np.log(snr / ref_snr))
                order = np.argsort(-sc)
                best = cands[order[0]]
                cand, score, status = np.array(best[:2]), best[2], OK
                if len(order) > 1:
                    alt = cands[order[1]]
                    if sc[order[1]] > sc[order[0]] - 1.0 and np.hypot(alt[0] - best[0], alt[1] - best[1]) > 3:
                        status = AMBIG     # two plausible detections: do not trust this frame
            if ref_snr is not None and (best is None or best[2] < 0.35 * ref_snr):
                # Much dimmer than the object has been: typically a clutter grab after a sharp manoeuvre.
                # Look wider for something with object-like SNR before accepting it.
                # while lost, the search grows geometrically (the object kept moving while we could not see it)
                r_wide = int(min(3 * r * 1.5 ** min(n_lost, 8), 8 * r, max(W, H)))
                # object-like brightness both ways: much brighter detections are usually terrain/cloud texture
                wide = [c for c in free(point_candidates(img, excl, pred, r_wide, pol, psf_sigma, params.snr_min), k)
                        if 0.5 * ref_snr <= c[2] <= 2.5 * ref_snr]
                if wide:
                    wb = min(wide, key=lambda c: abs(np.log(c[2] / ref_snr))
                             + np.hypot(c[0] - pred[0], c[1] - pred[1]) / r_wide)
                    # only a genuine jump (outside the normal search area) counts as a re-acquisition
                    far = np.hypot(wb[0] - pred[0], wb[1] - pred[1]) > r_search
                    cand, score, status = np.array(wb[:2]), wb[2], (REACQ if far else OK)
                elif best is not None:
                    status = WEAK          # only clutter-level detections: do not follow them
            if status in (OK, AMBIG, REACQ):
                sig = float(np.clip(2.0 / score, 0.2, 2.0))
            if status == WEAK:
                track.x[k], track.y[k], track.status[k], track.score[k] = pred[0], pred[1], WEAK, score
                n_lost += 1
                if n_lost > params.max_lost:
                    break
                own *= 0.9
                p, k_last = pred, k
                k += step
                continue
        else:
            sw, sh = tw + 2 * r, th + 2 * r
            x0, y0 = int(round(pred[0])) - sw // 2, int(round(pred[1])) - sh // 2
            if not (x0 + sw < tw or y0 + sh < th or x0 > W - tw or y0 > H - th):
                win = _patch(img, x0 + (sw - 1) / 2, y0 + (sh - 1) / 2, sw, sh)
                wex = _patch(excl, x0 + (sw - 1) / 2, y0 + (sh - 1) / 2, sw, sh)
                win = fill_masked(win, wex).astype(np.float32)
                res = cv2.matchTemplate(win, tmpl, cv2.TM_CCOEFF_NORMED)
                _, score, _, loc = cv2.minMaxLoc(res)
                px, py = _subpixel_peak(res, loc)
                cand = np.array([x0 + px + (tw - 1) / 2, y0 + py + (th - 1) / 2])
                status = OK if score >= params.lost_threshold else LOST
                if status == LOST and params.reacquire:
                    roi = (pred[0] - 2 * r - tw, pred[1] - 2 * r - th, 4 * r + 2 * tw, 4 * r + 2 * th)
                    cands = [c for c in detect_blobs(img, pol, params.preset, excl, roi, max_candidates=8)
                             if c.area <= 4 * tw * th]
                    best = None
                    for c in cands:
                        pt = _patch(img, c.x, c.y, tw, th).astype(np.float32)
                        s_ = float(cv2.matchTemplate(pt, tmpl, cv2.TM_CCOEFF_NORMED)[0, 0])
                        val = s_ - 0.1 * np.hypot(c.x - pred[0], c.y - pred[1]) / (r + 1)
                        if s_ >= 0.7 * params.lost_threshold and (best is None or val > best[0]):
                            best = (val, s_, c)
                    if best is not None:
                        cand, score, status = np.array([best[2].x, best[2].y]), best[1], REACQ
                sig = float(np.clip(0.15 + 0.85 * (1 - score), 0.15, 2.0)) * (1.5 if status == REACQ else 1.0)
        if status == LOST or cand is None:
            track.x[k], track.y[k], track.status[k], track.score[k] = pred[0], pred[1], LOST, score
            n_lost += 1
            if n_lost > params.max_lost:
                break
            own *= 0.9
            p, k_last = pred, k
            k += step
            continue
        if params.centroid_refine:
            rc = refine_centroid(img, cand[0], cand[1], bw if mode == "template" else 5, bh if mode == "template" else 5,
                                 pol, excl)
            if rc is not None and np.hypot(rc[0] - cand[0], rc[1] - cand[1]) < (max(tw, th) / 2 if mode == "template"
                                                                                 else 1.5):
                cand = np.array(rc[:2])
                track.size_px[k] = rc[2]
        n_lost = 0
        if mode == "point" and status in (OK, REACQ):
            recent_snr.append(float(score))
        track.x[k], track.y[k], track.status[k], track.score[k] = cand[0], cand[1], status, score
        track.sigma[k] = sig * (2.0 if status == AMBIG else 1.0)
        own = 0.7 * own + 0.3 * (cand - p_cam) / dk
        p, k_last = cand, k
        if mode == "template" and status == OK and score > 0.75 and params.template_update > 0:
            new = fill_masked(_patch(img, cand[0], cand[1], tw, th), _patch(excl, cand[0], cand[1], tw, th))
            tmpl = (1 - params.template_update) * tmpl + params.template_update * new.astype(np.float32)
        k += step


def run_tracking(frames, track: Track, masks: list[Mask], camera: Optional[CameraTrack], params: TrackParams,
                 progress=None, cancel=None, dup: Optional[np.ndarray] = None, blank: Optional[np.ndarray] = None,
                 overlay=None, others: Optional[list] = None) -> Track:
    """(Re)compute the whole track from the keyframes. `others`: already-tracked objects whose detections this
    track must not claim (two nearby, similar point targets otherwise swap or merge)."""
    avoid = None
    if others:
        def avoid(k):
            return [(o.x[k], o.y[k]) for o in others if o.status[k] in USABLE and np.isfinite(o.x[k])]
    if not track.keyframes:
        raise ValueError("Mark the object in at least one frame first.")
    track.reset_results()          # also clears the matching mode, which is re-decided at the first keyframe
    excl = mask_image(masks, frames[0].shape)
    b = 10   # frame borders carry scaling/encoding artefacts that look like point targets
    excl[:b, :] = excl[-b:, :] = 255
    excl[:, :b] = excl[:, -b:] = 255
    n = len(frames)
    last = n - 1 if track.end < 0 else min(track.end, n - 1)
    first = max(0, track.start)
    kfs = [k for k in sorted(track.keyframes) if first <= k <= last]
    if not kfs:
        raise ValueError(f"{track.name}: no keyframes inside the track's start/end range.")
    segs = [(kfs[0], first, -1)] + [(k, (kfs[i + 1] - 1 if i + 1 < len(kfs) else last), 1) for i, k in enumerate(kfs)]
    for j, (k0, k_end, step) in enumerate(segs):
        if progress:
            progress(j / len(segs), f"Tracking from keyframe {k0}")
        _track_segment(frames, track, excl, camera, params, k0, k_end, step, cancel, dup, blank, overlay, avoid)
    for k in track.excluded:
        if 0 <= k < n:
            track.status[k] = EXCLUDED
    if progress:
        progress(1.0, f"Tracked: {int(track.usable().sum())}/{n} frames usable ({track.mode} mode)")
    return track
