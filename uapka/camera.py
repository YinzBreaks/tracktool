"""Camera-motion (pan/tilt/roll/zoom) estimation by background registration.

Primary method: sparse features (grid-distributed Shi-Tomasi corners) on
FPN-cleaned frames, tracked with pyramidal Lucas-Kanade, forward-backward
checked, and fitted with a RANSAC 2-D similarity (translation = pan/tilt,
rotation = roll / image derotation, scale = zoom). Masked overlays, the
tracked object and sensor defect pixels are excluded.

Phase correlation is still computed, on raw and on cleaned frames, but only as
(a) a diagnostic that exposes the "locked onto fixed-pattern noise" failure and
(b) a low-confidence fallback when there is not enough texture for features.

Every frame pair gets a confidence score in [0, 1] built from inlier count,
inlier ratio, residual, spatial coverage, agreement with cleaned phase
correlation, and a check for a competing near-zero-motion cluster (the
signature of sensor-locked content such as FPN or burned-in overlays).

Compressed footage adds a second sensor-lock mechanism: on faint backgrounds
the encoder's own motion search matches the static FPN, so P-frames carry a
background that lags or freezes and then "catches up" at the next I-frame.
The decoded pixels genuinely move too little between keyframes; no frame-to-
frame method can recover that. We detect it (step jumps coinciding with
frame-difference spikes) and, when present, register I-frame to I-frame
directly and interpolate the camera path in between.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

from scipy.interpolate import PchipInterpolator

import cv2
import numpy as np

from .fpn import FPNModel, clean_frame, estimate_fpn, refine_fpn_with_motion, to_uint8_pair
from .masks import Mask, mask_image

GRID = 4


@dataclass
class CameraParams:
    fpn_mode: str = "auto"          # "auto" | "off"
    destripe: bool = True
    hp_sigma_px: float = 30.0       # removes illumination gradients
    smooth_sigma_px: float = 2.0    # suppresses pixel-scale sensor-locked residue
    gop_mode: str = "auto"          # "auto" | "off": I-frame anchor correction for compression lag
    keyframe_mad_ratio: float = 2.0 # frame-difference spike threshold for keyframe (I-frame) detection
    max_width: int = 960            # register on a downscaled copy above this width (results in full-res px)
    motion_model: str = "auto"      # auto (zoom/roll kept only where a consistent trend is significant) | similarity | translation
    trend_window: int = 30          # new-content steps over which zoom/roll trends are tested
    closure_frames: int = 10        # long-baseline consistency check every N new-content frames (0 = off)
    fpn_refine: bool = True         # 2nd pass: scene-based FPN re-estimate using the 1st motion estimate
    max_features: int = 400
    ransac_thresh_px: float = 1.0
    fb_thresh_px: float = 0.7
    min_inliers: int = 8
    border_px: int = 8
    object_margin_px: int = 12
    pc_max_width: int = 400         # phase correlation is a diagnostic; run it downscaled


@dataclass
class PairResult:
    M: np.ndarray                  # 2x3 similarity mapping frame k-1 coords -> frame k coords (background)
    confidence: float
    method: str                    # "features" | "phasecorr" | "none" | "ref"
    n_tracked: int = 0
    n_inliers: int = 0
    rms_px: float = float("nan")
    coverage: float = 0.0
    sigma_px: float = 0.0          # 1-sigma uncertainty of this step's translation
    pc_raw: tuple = (0.0, 0.0, 0.0)    # dx, dy, response on raw frames
    pc_clean: tuple = (0.0, 0.0, 0.0)  # dx, dy, response on FPN-cleaned frames
    flags: list = field(default_factory=list)


@dataclass
class CameraTrack:
    pairs: list[PairResult]          # pairs[k] relates frame k-1 -> k; pairs[0] is the reference
    A: np.ndarray                    # (N,3,3): frame k pixel -> reference (frame 0) pixel
    fpn_info: str = ""
    gop_info: str = ""
    closure_info: str = ""
    model_info: str = ""
    anchors: list = field(default_factory=list)

    def T(self, k: int) -> np.ndarray:
        """3x3 mapping frame k-1 -> k (background)."""
        return self.between(k - 1, k)

    def between(self, j: int, k: int) -> np.ndarray:
        """3x3 mapping frame j -> frame k (background). Valid across skipped frames."""
        return np.linalg.inv(self.A[k]) @ self.A[j]

    def confidence(self) -> np.ndarray:
        return np.array([p.confidence for p in self.pairs])

    def sigma_px(self) -> np.ndarray:
        return np.array([p.sigma_px for p in self.pairs])

    def scale(self) -> np.ndarray:
        """Scale of A_k (frame k -> ref). <1 means zoomed in relative to frame 0."""
        return np.sqrt(np.abs(np.linalg.det(self.A[:, :2, :2])))

    def roll_deg(self) -> np.ndarray:
        return np.degrees(np.arctan2(self.A[:, 1, 0], self.A[:, 0, 0]))


def _similarity_params(M: np.ndarray) -> tuple[float, float, float, float]:
    s = float(np.sqrt(abs(np.linalg.det(M[:, :2]))))
    rot = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    return float(M[0, 2]), float(M[1, 2]), s, rot


def _is_identity_like(M: np.ndarray, tol_px: float = 0.3) -> bool:
    dx, dy, s, rot = _similarity_params(M)
    return np.hypot(dx, dy) < tol_px and abs(s - 1) < 1e-3 and abs(rot) < 0.05


def _allowed_bbox(allowed: np.ndarray) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) bounding box of usable pixels; the whole frame if nothing is allowed."""
    ys, xs = np.nonzero(allowed)
    if len(xs) == 0:
        return 0, 0, allowed.shape[1], allowed.shape[0]
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _grid_features(img8: np.ndarray, allowed: np.ndarray, max_features: int) -> Optional[np.ndarray]:
    # the grid spans the usable area (e.g. a sensor inset between redaction panels), so masked-off
    # regions do not eat the feature budget
    bx0, by0, bx1, by1 = _allowed_bbox(allowed)
    H, W = img8.shape
    if (bx1 - bx0) * (by1 - by0) > 0.6 * W * H:      # not a true inset: keep the plain frame grid
        bx0, by0, bx1, by1 = 0, 0, W, H
    per = max(4, max_features // (GRID * GRID))
    pts = []
    for gy in range(GRID):
        for gx in range(GRID):
            y0, y1 = by0 + gy * (by1 - by0) // GRID, by0 + (gy + 1) * (by1 - by0) // GRID
            x0, x1 = bx0 + gx * (bx1 - bx0) // GRID, bx0 + (gx + 1) * (bx1 - bx0) // GRID
            sub_mask = allowed[y0:y1, x0:x1]
            if sub_mask.sum() < 255 * 50:
                continue
            p = cv2.goodFeaturesToTrack(img8[y0:y1, x0:x1], maxCorners=per, qualityLevel=0.003,
                                        minDistance=7, mask=sub_mask, blockSize=7)
            if p is not None:
                p[:, 0, 0] += x0
                p[:, 0, 1] += y0
                pts.append(p)
    if not pts:
        return None
    return np.concatenate(pts).astype(np.float32)


def _coverage(pts: np.ndarray, allowed: np.ndarray) -> float:
    H, W = allowed.shape
    cells_allowed, cells_hit = 0, set()
    for gy in range(GRID):
        for gx in range(GRID):
            y0, y1 = gy * H // GRID, (gy + 1) * H // GRID
            x0, x1 = gx * W // GRID, (gx + 1) * W // GRID
            if allowed[y0:y1, x0:x1].mean() > 0.3 * 255:
                cells_allowed += 1
    for x, y in pts.reshape(-1, 2):
        cells_hit.add((min(GRID - 1, int(y * GRID / H)), min(GRID - 1, int(x * GRID / W))))
    return len(cells_hit) / max(cells_allowed, 1)


def _phase_corr(a: np.ndarray, b: np.ndarray, max_width: int, excluded: Optional[np.ndarray]) -> tuple:
    """Translation of b relative to a, with peak response. Downscaled for speed (diagnostic use)."""
    H, W = a.shape
    f = min(1.0, max_width / W)
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    if excluded is not None and excluded.any():
        a = a.copy()
        b = b.copy()
        a[excluded > 0] = float(np.mean(a[excluded == 0])) if (excluded == 0).any() else 0
        b[excluded > 0] = float(np.mean(b[excluded == 0])) if (excluded == 0).any() else 0
    if f < 1.0:
        size = (int(W * f), int(H * f))
        a = cv2.resize(a, size, interpolation=cv2.INTER_AREA)
        b = cv2.resize(b, size, interpolation=cv2.INTER_AREA)
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(a, b, win)
    return dx / f, dy / f, float(resp)


def _feature_register(prev_c: np.ndarray, cur_c: np.ndarray, excluded: np.ndarray, allowed: np.ndarray,
                      params: CameraParams) -> Optional[dict]:
    """Sparse LK + RANSAC similarity on one pair of cleaned images. None if it cannot be estimated."""
    H, W = prev_c.shape
    a8, b8 = to_uint8_pair(prev_c, cur_c)
    p0 = _grid_features(a8, allowed, params.max_features)
    if p0 is None or len(p0) < params.min_inliers:
        return None
    lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    p1, st1, _ = cv2.calcOpticalFlowPyrLK(a8, b8, p0, None, **lk)
    p0r, st2, _ = cv2.calcOpticalFlowPyrLK(b8, a8, p1, None, **lk)
    fb = np.linalg.norm((p0 - p0r).reshape(-1, 2), axis=1)
    ok = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb < params.fb_thresh_px)
    q = p1.reshape(-1, 2)
    ok &= (q[:, 0] >= 0) & (q[:, 0] < W) & (q[:, 1] >= 0) & (q[:, 1] < H)
    qi = np.clip(q.astype(int), 0, [W - 1, H - 1])
    ok &= excluded[qi[:, 1], qi[:, 0]] == 0
    src, dst = p0.reshape(-1, 2)[ok], q[ok]
    n_tracked = len(src)
    if n_tracked < params.min_inliers:
        return None
    M1, inl1 = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=params.ransac_thresh_px,
                                           maxIters=3000, confidence=0.995)
    if M1 is None:
        return None
    flags: list[str] = []
    inl1 = inl1.ravel().astype(bool)
    n1 = int(inl1.sum())
    M, inl = M1, inl1
    # Competing motion cluster? (e.g. FPN/overlay residue that stays put)
    rest = ~inl1
    ambiguity = 0.0
    if rest.sum() >= params.min_inliers:
        M2, inl2 = cv2.estimateAffinePartial2D(src[rest], dst[rest], method=cv2.RANSAC,
                                               ransacReprojThreshold=params.ransac_thresh_px,
                                               maxIters=2000, confidence=0.99)
        if M2 is not None:
            n2 = int(inl2.sum())
            if n2 >= params.min_inliers and n2 >= 0.5 * n1:
                ambiguity = n2 / (n1 + n2)
                id1, id2 = _is_identity_like(M1), _is_identity_like(M2)
                if id1 and not id2 and n2 >= 0.6 * n1:
                    # Prefer the moving cluster: a zero-motion cluster is the classic
                    # sensor-locked artefact (FPN, HUD, burned-in text).
                    M = M2
                    inl = np.zeros_like(inl1)
                    inl[np.flatnonzero(rest)[inl2.ravel().astype(bool)]] = True
                    flags.append("CHOSE_MOVING_OVER_STATIC_CLUSTER")
                flags.append("STATIC_CLUSTER_PRESENT" if (id1 or id2) else "TWO_MOTION_CLUSTERS")
    n_in = int(inl.sum())
    if params.motion_model == "translation" and n_in >= 3:
        t_only = np.median(dst[inl] - src[inl], axis=0)
        M = np.array([[1, 0, t_only[0]], [0, 1, t_only[1]]], float)
    pred = src[inl] @ M[:, :2].T + M[:, 2]
    rms = float(np.sqrt(np.mean(np.sum((pred - dst[inl]) ** 2, axis=1)))) if n_in else float("nan")
    coverage = _coverage(dst[inl], allowed)
    c_count = np.clip((n_in - params.min_inliers) / 40.0, 0, 1)
    c_ratio = np.clip((n_in / max(n_tracked, 1) - 0.2) / 0.6, 0, 1)
    # Inlier residuals of real footage sit at 0.5-0.9 px (interpolation, slight parallax, compression); only a
    # residual approaching the RANSAC threshold means the model does not fit. Synthetic frames reach ~0.1 px.
    c_resid = float(np.exp(-(rms / 1.5) ** 2))
    c_cov = np.clip(coverage / 0.5, 0, 1)
    quality = float((max(c_count, 1e-3) * max(c_ratio, 1e-3) * c_resid * max(c_cov, 1e-3)) ** 0.25)
    quality *= 1.0 - 0.6 * ambiguity
    return dict(M=M, n_tracked=n_tracked, n_in=n_in, rms=rms, coverage=coverage, quality=quality, flags=flags)


def register_pair(prev_raw: np.ndarray, cur_raw: np.ndarray, prev_c: np.ndarray, cur_c: np.ndarray,
                  excluded: np.ndarray, params: CameraParams) -> PairResult:
    """Estimate background motion prev -> cur. `excluded` is uint8 (255 = do not use)."""
    flags: list[str] = []
    allowed = _allowed(excluded, params.border_px)
    bx0, by0, bx1, by1 = _allowed_bbox(allowed)
    crop = (slice(by0, by1), slice(bx0, bx1))
    pc_raw = _phase_corr(prev_raw[crop], cur_raw[crop], params.pc_max_width, None)
    pc_clean = _phase_corr(prev_c[crop], cur_c[crop], params.pc_max_width, excluded[crop])
    est = _feature_register(prev_c, cur_c, excluded, allowed, params)
    if est is not None:
        M = est["M"]
        flags += est["flags"]
        d_pc = float(np.hypot(M[0, 2] - pc_clean[0], M[1, 2] - pc_clean[1]))
        # a weak phase-correlation peak (large motion, masked regions, little overlap) is no evidence against
        # the feature estimate: only a strong peak may lower the confidence
        strong = pc_clean[2] > 0.25
        agree = 0.5 + 0.5 * np.exp(-(d_pc / 1.5) ** 2) if strong else 1.0
        if strong and d_pc > 3:
            flags.append("PHASECORR_DISAGREES")
        conf = est["quality"] * agree
        method = "features"
        n_tracked, n_in, rms, coverage = est["n_tracked"], est["n_in"], est["rms"], est["coverage"]
        if n_in < 20:
            flags.append("LOW_FEATURES")
        sigma = _step_sigma(rms, n_in, conf)
    else:
        n_tracked = n_in = 0
        rms, coverage = float("nan"), 0.0
        # Fallback: translation from cleaned phase correlation if its peak is credible.
        if pc_clean[2] > 0.1:
            M = np.array([[1, 0, pc_clean[0]], [0, 1, pc_clean[1]]], float)
            method, conf, sigma = "phasecorr", float(np.clip(pc_clean[2], 0, 1)) * 0.4, 1.0
            flags.append("FALLBACK_PHASECORR")
        else:
            M = np.array([[1, 0, 0], [0, 1, 0]], float)
            method, conf, sigma = "none", 0.0, 3.0
            flags.append("NO_ESTIMATE_ASSUMED_STATIC")
    # Diagnostic: would naive phase correlation on raw frames have locked to the sensor?
    if np.hypot(M[0, 2], M[1, 2]) > 1.0 and np.hypot(pc_raw[0], pc_raw[1]) < 0.3 and method == "features":
        flags.append("RAW_PHASECORR_LOCKED_TO_SENSOR")
    return PairResult(M=M.astype(float), confidence=float(conf), method=method, n_tracked=n_tracked,
                      n_inliers=n_in, rms_px=rms, coverage=coverage, sigma_px=float(sigma),
                      pc_raw=pc_raw, pc_clean=pc_clean, flags=flags)


def _step_sigma(rms: float, n_in: int, conf: float) -> float:
    """1-sigma of a registration step's translation (px).

    Feature residuals on smooth, band-passed imagery are spatially correlated, so
    n inliers are NOT n independent measurements; the effective count is capped
    (calibrated on synthetic clips with known motion).
    """
    r = rms if np.isfinite(rms) else 1.0
    return max(r / np.sqrt(min(max(n_in, 1), 9)), 0.03) / max(conf, 0.1)


def _allowed(excluded: np.ndarray, border: int) -> np.ndarray:
    allowed = (excluded == 0).astype(np.uint8) * 255
    allowed[:border, :] = 0
    allowed[-border:, :] = 0
    allowed[:, :border] = 0
    allowed[:, -border:] = 0
    return allowed


def _sim_params(A: np.ndarray) -> np.ndarray:
    """3x3 similarity -> (tx, ty, log scale, angle) for interpolation."""
    s = np.sqrt(abs(np.linalg.det(A[:2, :2])))
    return np.array([A[0, 2], A[1, 2], np.log(s), np.arctan2(A[1, 0], A[0, 0])])


def _sim_matrix(p: np.ndarray) -> np.ndarray:
    s, a = np.exp(p[2]), p[3]
    return np.array([[s * np.cos(a), -s * np.sin(a), p[0]], [s * np.sin(a), s * np.cos(a), p[1]], [0, 0, 1]])


def detect_gop_lag(mad: np.ndarray, step_px: np.ndarray, ratio: float, min_gap: int = 4) -> tuple[list[int], float]:
    """Keyframes = frame-difference spikes. Returns (keyframe indices, catch-up ratio).

    Catch-up ratio = mean camera step at keyframes / mean step elsewhere. ~1 for
    healthy footage; well above 1 when the background lags between keyframes.
    """
    n = len(mad)
    m = np.nan_to_num(mad, nan=0.0)
    from scipy.ndimage import median_filter
    base = median_filter(m, size=31, mode="nearest")
    spikes = []
    for k in range(1, n):
        # keyframes are spaced; a spike right after frame 0 or another spike is the first P-frame, not a GOP edge
        if m[k] > ratio * max(base[k], 1e-3) and m[k] > 0.5 and k - (spikes[-1] if spikes else 0) >= min_gap:
            spikes.append(k)
    if len(spikes) < 2:
        return spikes, 1.0
    others = np.setdiff1d(np.arange(1, n), spikes)
    if len(others) == 0:
        return spikes, 1.0
    at = float(np.mean(step_px[spikes]))
    el = float(np.mean(step_px[others]))
    return spikes, at / max(el, 1e-3)


def object_exclusion(shape, boxes: list, margin: int) -> np.ndarray:
    """boxes: (cx, cy, w, h) tuples, None, or lists of those (several objects per frame)."""
    m = np.zeros(shape[:2], np.uint8)
    flat = []
    for bx in boxes:
        if isinstance(bx, list):
            flat += bx
        else:
            flat.append(bx)
    for bx in flat:
        if bx is None:
            continue
        cx, cy, w, h = bx
        x0, y0 = int(cx - w / 2 - margin), int(cy - h / 2 - margin)
        x1, y1 = int(cx + w / 2 + margin), int(cy + h / 2 + margin)
        m[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = 255
    return m


class _Scaled:
    """Lazy downscaled view of a frame sequence."""

    def __init__(self, frames, f: float):
        self.frames, self.f = frames, f
        H, W = frames[0].shape[:2]
        self.size = (max(1, int(round(W * f))), max(1, int(round(H * f))))

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, k):
        return cv2.resize(self.frames[k], self.size, interpolation=cv2.INTER_AREA)


def estimate_camera_motion(frames, masks: list[Mask], params: CameraParams,
                           object_boxes: Optional[Callable[[int], Optional[tuple]]] = None,
                           progress=None, cancel=None, skip: Optional[np.ndarray] = None,
                           overlay: Optional[Callable[[int], Optional[np.ndarray]]] = None) -> CameraTrack:
    """Register consecutive frames. object_boxes(k) -> (cx, cy, w, h) or None.

    skip[k] marks frames with no new image content (duplicates, blank dropouts):
    they are stepped over, and registration bridges from the last real frame.
    overlay(k) -> uint8 mask of per-frame burned-in symbology to exclude.
    All returned transforms are in full-resolution pixel coordinates.
    """
    W = frames[0].shape[1]
    if W <= params.max_width:
        return _estimate(frames, masks, params, object_boxes, progress, cancel, skip, overlay)
    f = params.max_width / W
    sm = [Mask(int(m.x * f), int(m.y * f), int(np.ceil(m.w * f)) + 1, int(np.ceil(m.h * f)) + 1, m.kind, m.label)
          for m in masks]
    ob = ov = None
    scaled = _Scaled(frames, f)
    if object_boxes is not None:
        def ob(k):
            b = object_boxes(k)
            if isinstance(b, list):
                return [tuple(v * f for v in x) for x in b if x is not None]
            return None if b is None else tuple(v * f for v in b)
    if overlay is not None:
        def ov(k):
            m = overlay(k)
            return None if m is None else cv2.resize(m, scaled.size, interpolation=cv2.INTER_NEAREST)
    ct = _estimate(scaled, sm, params, ob, progress, cancel, skip, ov)
    S, Si = np.diag([f, f, 1.0]), np.diag([1 / f, 1 / f, 1.0])
    ct.A = np.einsum("ij,njk,kl->nil", Si, ct.A, S)
    for p in ct.pairs:
        p.M = (Si @ np.vstack([p.M, [0, 0, 1]]) @ S)[:2]
        p.sigma_px /= f
        p.rms_px /= f
        p.pc_raw = (p.pc_raw[0] / f, p.pc_raw[1] / f, p.pc_raw[2])
        p.pc_clean = (p.pc_clean[0] / f, p.pc_clean[1] / f, p.pc_clean[2])
    ct.fpn_info += f" (registration at {f:.2f}x scale)"
    return ct


def _estimate(frames, masks: list[Mask], params: CameraParams, object_boxes, progress, cancel,
              skip=None, overlay=None) -> CameraTrack:
    n = len(frames)
    shape = frames[0].shape
    user_excl = mask_image(masks, shape)
    if user_excl.any():
        user_excl = cv2.dilate(user_excl, np.ones((7, 7), np.uint8))
    skip = np.zeros(n, bool) if skip is None else np.asarray(skip, bool)
    fpn: Optional[FPNModel] = None
    fpn_info = "FPN removal disabled."
    if params.fpn_mode != "off" and n >= 3:
        if progress:
            progress(0.0, "Estimating fixed-pattern noise")
        fpn = estimate_fpn(_Subset(frames, np.flatnonzero(~skip)), user_excl, hp_sigma=params.hp_sigma_px)
        fpn_info = fpn.describe()
    two_pass = fpn is not None and params.fpn_refine and fpn.static_fraction > 0.5
    span = 0.45 if two_pass else 0.9
    args = (user_excl, params, object_boxes, progress, cancel, skip, overlay)
    ct = _register_all(frames, fpn, *args, 0.0, span)
    if two_pass:
        if progress:
            progress(span, "Refining FPN with the scene model")
        act = np.flatnonzero(~skip)
        fpn = refine_fpn_with_motion(_Subset(frames, act), ct.A[act], fpn, user_excl)
        ct = _register_all(frames, fpn, *args, span, 0.9)
        fpn_info += " Refined in a second pass using the scene mosaic (scene-based nonuniformity correction)."
    ct.fpn_info = fpn_info
    if progress:
        progress(1.0, "Camera motion done")
    return ct


class _Subset:
    def __init__(self, frames, idx):
        self.frames, self.idx = frames, idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        return self.frames[int(self.idx[k])]


def _register_all(frames, fpn, user_excl, params, object_boxes, progress, cancel, skip, overlay,
                  p0, p1) -> CameraTrack:
    n = len(frames)
    shape = frames[0].shape
    defect_excl = cv2.dilate(fpn.defects, np.ones((5, 5), np.uint8)) if fpn is not None else None

    def prep(k):
        return clean_frame(frames[k], fpn, params.destripe, params.hp_sigma_px, params.smooth_sigma_px)

    def exclusion(*ks):
        excl = user_excl.copy()
        if defect_excl is not None:
            excl |= defect_excl
        if overlay is not None:
            for k in ks:
                m = overlay(k)
                if m is not None:
                    excl |= m
        if object_boxes is not None:
            excl |= object_exclusion(shape, [object_boxes(k) for k in ks], params.object_margin_px)
        return excl

    act = [k for k in range(n) if not skip[k]] or [0]
    ident = np.array([[1, 0, 0], [0, 1, 0]], float)
    pairs = [PairResult(M=ident.copy(), confidence=1.0, method="ref") for _ in range(n)]
    A = np.repeat(np.eye(3)[None], n, axis=0)
    mad_act = np.full(len(act), np.nan)
    step_act = np.zeros(len(act))
    prev_c = prep(act[0])
    for i in range(1, len(act)):
        a, b = act[i - 1], act[i]
        if cancel and cancel():
            raise InterruptedError("Cancelled")
        cur_c = prep(b)
        mad_act[i] = float(cv2.absdiff(frames[b], frames[a]).mean())
        pr = register_pair(frames[a], frames[b], prev_c, cur_c, exclusion(a, b), params)
        if b - a > 1:
            pr.flags = pr.flags + [f"BRIDGED_{b - a - 1}_SKIPPED"]
        pairs[b] = pr
        step_act[i] = float(np.hypot(pr.M[0, 2], pr.M[1, 2]))
        for k in range(a + 1, b):     # duplicates/blanks: no new image, camera unknown -> hold
            A[k] = A[a]
            pairs[k] = PairResult(M=ident.copy(), confidence=pr.confidence, method="skipped",
                                  sigma_px=0.0, flags=["SKIPPED_FRAME"])
        A[b] = A[a] @ np.linalg.inv(np.vstack([pr.M, [0, 0, 1]]))
        prev_c = cur_c
        if progress and i % 5 == 0:
            progress(p0 + (p1 - p0) * i / len(act), f"Registering frame {b}/{n - 1}")
    for k in range(act[-1] + 1, n):
        A[k] = A[act[-1]]
    ct = CameraTrack(pairs=pairs, A=A)
    if params.motion_model == "auto":
        ct.model_info = _simplify_model(ct, act, shape, params)

    # --- Compression lag check and I-frame anchor correction (on new-content frames only)
    keys_i, catchup = detect_gop_lag(mad_act, step_act, params.keyframe_mad_ratio)
    keys = [act[i] for i in keys_i]
    ct.gop_info = f"{len(keys)} keyframe-like frame-difference spikes; camera-step catch-up ratio {catchup:.2f}."
    if len(keys) >= 2 and catchup > 1.3:
        ct.gop_info += (" Background motion appears to lag between keyframes and catch up at them "
                        "(compression freezing a low-texture background).")
        if params.gop_mode == "auto":
            if _anchor_correct(ct, frames, [act[0]] + keys, prep, exclusion, params):
                ct.gop_info += f" Corrected by registering {len(ct.anchors)} keyframes directly and interpolating."
            else:
                ct.gop_info += (" Direct keyframe-to-keyframe registration either failed or agreed with the "
                                "frame-to-frame chain, so the lag hypothesis is not confirmed and no anchor "
                                "correction was applied.")
        else:
            ct.gop_info += " Anchor correction is OFF; treat in-between camera motion as unreliable."
    if params.closure_frames and not ct.anchors and len(act) > params.closure_frames:
        _closure_check(ct, frames, act, prep, exclusion, params)
    return ct


def _simplify_model(ct: CameraTrack, act: list[int], shape, params: CameraParams) -> str:
    """Keep zoom / roll only where they form a significant trend.

    A real zoom or roll is consistent across many steps; per-step estimates on weak backgrounds are noise, and
    chaining noise in scale/rotation produces fake zoom and roll (and, through the centre of rotation, fake pan).
    Over each window of `trend_window` steps the median log-scale (rotation) is compared with its random-walk
    expectation from the step-to-step scatter; where it is not > 5 sigma the term is removed, keeping the
    motion of the image centre unchanged.
    """
    H, W = shape[:2]
    c = np.array([W / 2.0, H / 2.0])
    steps = act[1:]
    if len(steps) < 6:
        return ""
    ls = np.array([np.log(np.sqrt(abs(np.linalg.det(ct.pairs[k].M[:, :2])))) for k in steps])
    rot = np.array([np.arctan2(ct.pairs[k].M[1, 0], ct.pairs[k].M[0, 0]) for k in steps])
    Wn = min(params.trend_window, len(steps))

    from scipy.ndimage import median_filter

    def significant(x):
        # windowed median: a real zoom/roll is a consistent per-step bias; isolated garbage steps (e.g. at
        # compression keyframes) must not count as a trend
        noise = 1.4826 * np.median(np.abs(np.diff(x))) / np.sqrt(2) + 1e-12
        movmed = median_filter(x, size=Wn, mode="nearest")
        return np.abs(movmed) > 5 * 1.25 * noise / np.sqrt(Wn)

    keep_s, keep_r = significant(ls), significant(rot)
    for i, k in enumerate(steps):
        M = np.vstack([ct.pairs[k].M, [0, 0, 1]])
        s_ = np.exp(ls[i]) if keep_s[i] else 1.0
        a_ = rot[i] if keep_r[i] else 0.0
        R = s_ * np.array([[np.cos(a_), -np.sin(a_)], [np.sin(a_), np.cos(a_)]])
        tc = (M[:2, :2] @ c + M[:2, 2]) - c          # motion of the image centre is preserved
        t = c + tc - R @ c
        ct.pairs[k].M = np.hstack([R, t[:, None]])
    A = ct.A
    for i in range(1, len(act)):
        a, b = act[i - 1], act[i]
        for k in range(a + 1, b):
            A[k] = A[a]
        A[b] = A[a] @ np.linalg.inv(np.vstack([ct.pairs[b].M, [0, 0, 1]]))
    for k in range(act[-1] + 1, len(A)):
        A[k] = A[act[-1]]
    return (f"Motion model: zoom kept in {int(keep_s.sum())}/{len(steps)} steps, roll kept in "
            f"{int(keep_r.sum())}/{len(steps)} steps (significant trends only); elsewhere translation only.")


def _closure_check(ct: CameraTrack, frames, act: list[int], prep, exclusion, params: CameraParams) -> None:
    """Long-baseline consistency check AND correction.

    Frame-to-frame registration can be confidently wrong: sensor-locked residue (translucent overlays, FPN and
    compression residue) biases every small step towards zero, and the bias is invisible at sub-pixel step
    sizes. Registering directly across a longer interval (10 new frames, or fewer on fast pans so the frames
    still overlap) separates real motion from the residue. Each interval is then handled as:
      consistent   direct registration agrees with the chain -> confidence kept
      corrected    direct registration succeeds but disagrees -> the chain inside the interval is re-scaled
                   so that its end-to-end motion equals the direct registration (the correction is spread
                   over the steps in proportion to the chain's own increments; evenly when the chain is
                   near-static), confidence = closure quality, flag CLOSURE_CORRECTED
      unverifiable scene cannot be registered across the interval -> chain kept, confidence capped at 0.25
    """
    L = params.closure_frames
    n_ok = n_fix = n_unv = 0
    H, W = frames[0].shape[:2]
    bx0, by0, bx1, by1 = _allowed_bbox(_allowed(exclusion(act[0]), params.border_px))
    span_px = 0.3 * min(bx1 - bx0, by1 - by0)
    centre = np.array([(bx0 + bx1) / 2.0, (by0 + by1) / 2.0, 1.0])
    A_chain = ct.A.copy()
    A_new = ct.A.copy()
    A_anc = A_chain[act[0]].copy()          # anchored pose of the current interval start

    def chain_disp(i, j):
        T = np.linalg.inv(A_chain[act[j]]) @ A_chain[act[i]]
        return float(np.hypot(*((T @ centre)[:2] - centre[:2])))

    i = 0
    while i < len(act) - 1:
        j = min(i + L, len(act) - 1)
        while j - i > 2 and chain_disp(i, j) > span_px:
            j -= 1
        a, b = act[i], act[j]
        guess = np.linalg.inv(A_chain[b]) @ A_chain[a]          # frame a -> frame b per the chain
        warped = cv2.warpAffine(prep(a), guess[:2], (W, H), flags=cv2.INTER_LINEAR)
        excl = exclusion(a, b) | cv2.warpAffine(exclusion(a), guess[:2], (W, H), flags=cv2.INTER_NEAREST,
                                                borderValue=255)
        est = _feature_register(warped, prep(b), excl, _allowed(excl, params.border_px), params)
        chain_sig = float(np.sqrt(sum(ct.pairs[act[m]].sigma_px ** 2 for m in range(i + 1, j + 1))))
        steps = list(range(i + 1, j + 1))
        if est is None or est["quality"] < 0.3:
            # keep the chain, but re-base it on the anchored pose so earlier corrections carry through
            shift = A_anc @ np.linalg.inv(A_chain[a])
            for m in steps:
                A_new[act[m]] = shift @ A_chain[act[m]]
                for k in range(act[m - 1] + 1, act[m]):        # duplicates hold the previous pose
                    A_new[k] = A_new[act[m - 1]]
                pr = ct.pairs[act[m]]
                pr.confidence = min(pr.confidence, 0.25)
                pr.flags = pr.flags + ["CLOSURE_UNVERIFIABLE"]
            A_anc = A_new[b].copy()
            n_unv += 1
            i = j
            continue
        M_ab = np.vstack([est["M"], [0, 0, 1]]) @ guess            # direct a -> b
        resid = float(np.hypot(*est["M"][:, 2]))
        tol = max(1.5, 3 * chain_sig, 1.0 * math.sqrt(j - i))
        A_end = A_anc @ np.linalg.inv(M_ab)                         # anchored pose at b
        if resid <= tol:
            flag, n_ok, conf_cap = None, n_ok + 1, 1.0
        else:
            flag, n_fix = "CLOSURE_CORRECTED", n_fix + 1
            conf_cap = float(est["quality"]) * 0.8
        # spread the end-to-end correction over the interval: weights = the chain's own step sizes
        # (near-static chain: even weights), applied to the similarity parameters of the correction
        mags = np.array([float(np.hypot(*ct.pairs[act[m]].M[:, 2])) for m in steps])
        wts = mags / mags.sum() if mags.sum() > 1.0 else np.full(len(steps), 1.0 / len(steps))
        cum = np.cumsum(wts)
        C = A_end @ np.linalg.inv(A_anc @ np.linalg.inv(A_chain[a]) @ A_chain[b])   # correction in ref space
        pC = _sim_params(C)
        for m, w in zip(steps, cum):
            Cw = _sim_matrix(pC * w)
            A_new[act[m]] = Cw @ A_anc @ np.linalg.inv(A_chain[a]) @ A_chain[act[m]]
            for k in range(act[m - 1] + 1, act[m]):            # duplicates hold the previous pose
                A_new[k] = A_new[act[m - 1]]
            if flag:
                pr = ct.pairs[act[m]]
                pr.flags = pr.flags + [flag]
                # the per-step registration was wrong here: the step inherits the closure registration's quality
                pr.confidence = conf_cap
                pr.sigma_px = float(max(pr.sigma_px, _step_sigma(est["rms"], est["n_in"], est["quality"]) /
                                        math.sqrt(len(steps))))
        A_anc = A_new[b].copy()
        i = j
    # per-step M consistent with the corrected poses
    for k in range(1, len(A_new)):
        if ct.pairs[k].method in ("skipped", "ref"):
            continue
        T = np.linalg.inv(A_new[k]) @ A_new[k - 1]
        ct.pairs[k].M = T[:2]
    ct.A = A_new
    ct.closure_info = (f"Long-baseline check (<= {L} new frames, shortened on fast pans): {n_ok} intervals consistent, "
                       f"{n_fix} corrected from the direct registration, {n_unv} unverifiable (scene could not be "
                       f"registered across the interval).")


def _anchor_correct(ct: CameraTrack, frames, anchors: list[int], prep, exclusion, params: CameraParams) -> bool:
    n = len(frames)
    anchors = sorted(set(a for a in anchors if 0 <= a < n))
    A_chain = ct.A.copy()
    A_anc = {anchors[0]: A_chain[anchors[0]]}
    anc_conf, anc_sigma = {anchors[0]: 1.0}, {anchors[0]: 0.0}
    H0, W0 = frames[0].shape[:2]
    centre = np.array([W0 / 2.0, H0 / 2.0, 1.0])
    disagree, travel, shrink = [], [], []
    prev_c = prep(anchors[0])
    for a, b in zip(anchors[:-1], anchors[1:]):
        cur_c = prep(b)
        guess = np.linalg.inv(A_chain[b]) @ A_chain[a]          # maps frame a -> frame b
        H, W = cur_c.shape
        warped = cv2.warpAffine(prev_c, guess[:2], (W, H), flags=cv2.INTER_LINEAR)
        excl = exclusion(a, b) | cv2.warpAffine(exclusion(a), guess[:2], (W, H), flags=cv2.INTER_NEAREST,
                                                borderValue=255)
        est = _feature_register(warped, cur_c, excl, _allowed(excl, params.border_px), params)
        if est is None:
            A_anc[b] = A_anc[a] @ np.linalg.inv(guess)
            anc_conf[b], anc_sigma[b] = 0.1, 3.0
        else:
            M_ab = np.vstack([est["M"], [0, 0, 1]]) @ guess
            A_anc[b] = A_anc[a] @ np.linalg.inv(M_ab)
            anc_conf[b] = est["quality"]
            if est["quality"] >= 0.3:
                # chain displacement vs direct keyframe registration over this interval
                d_anc = (M_ab @ centre)[:2] - centre[:2]
                d_chain = (guess @ centre)[:2] - centre[:2]
                disagree.append(float(np.hypot(*(d_anc - d_chain))))
                travel.append(float(np.hypot(*d_anc)))
                if travel[-1] > 3.0:
                    shrink.append(1.0 - float(np.dot(d_chain, d_anc) / np.dot(d_anc, d_anc)))
            anc_sigma[b] = _step_sigma(est["rms"], est["n_in"], est["quality"])
        prev_c = cur_c
    if np.median([anc_conf[a] for a in anchors[1:]]) < 0.3 or not disagree:
        return False
    # Lag is confirmed only if the chain genuinely lost motion relative to the keyframes: either grossly (large
    # disagreement) or systematically (the chain is consistently SHORT along the direction of travel; random
    # registration noise has no preferred sign). Otherwise anchoring would only replace good per-frame
    # measurements with an interpolation.
    gross = np.median(disagree) >= max(2.0, 0.05 * float(np.median(travel)))
    systematic = len(shrink) >= 2 and np.median(shrink) > 0.03 and np.mean(np.array(shrink) > 0) >= 0.75
    if not (gross or systematic):
        return False
    ks = np.array(anchors)
    P = np.array([_sim_params(A_anc[a]) for a in anchors])
    P[:, 3] = np.unwrap(P[:, 3])
    last = anchors[-1]
    interp = PchipInterpolator(ks, P, axis=0) if len(ks) >= 2 else None
    A_new = A_chain.copy()
    for k in range(n):
        if k <= last and interp is not None:
            A_new[k] = _sim_matrix(interp(k))
        else:  # after the final keyframe: chain increments relative to it
            A_new[k] = A_anc[last] @ np.linalg.inv(A_chain[last]) @ A_chain[k]
    ct.anchors = anchors
    first = anchors[0]
    for k in range(n):
        if k < first:
            A_new[k] = A_chain[k]
    ct.A = A_new
    for k in range(first + 1, n):
        if ct.pairs[k].method == "skipped":
            continue
        if k > last:
            pr = ct.pairs[k]
            pr.flags = pr.flags + ["AFTER_LAST_KEYFRAME"]
            pr.confidence = min(pr.confidence, 0.2)
            continue
        i = int(np.searchsorted(ks, k))   # first anchor >= k (ks[0] < k, so i >= 1)
        prv, nxt = int(ks[i - 1]), int(ks[i])
        pr = ct.pairs[k]
        T = np.linalg.inv(A_new[k]) @ A_new[k - 1]
        pr.M = T[:2]
        pr.method = "keyframe-anchored"
        pr.flags = pr.flags + ["GOP_ANCHOR_INTERPOLATED"]
        pr.confidence = float(min(anc_conf[prv], anc_conf[nxt])) * 0.8
        pr.sigma_px = float(max(anc_sigma[nxt], 0.05))
    return True


def naive_phase_correlation_track(frames) -> np.ndarray:
    """Cumulative translation from plain phase correlation on raw frames.

    Included for teaching: compare with estimate_camera_motion() on footage with
    fixed-pattern noise to see the failure this tool is designed to avoid.
    Returns (N, 2) cumulative camera translation in pixels.
    """
    out = np.zeros((len(frames), 2))
    for k in range(1, len(frames)):
        dx, dy, _ = _phase_corr(frames[k - 1], frames[k], 10**6, None)
        out[k] = out[k - 1] + (dx, dy)
    return out
