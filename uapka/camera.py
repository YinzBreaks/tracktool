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
    anchors: list = field(default_factory=list)

    def T(self, k: int) -> np.ndarray:
        """3x3 mapping frame k-1 -> k."""
        return np.vstack([self.pairs[k].M, [0, 0, 1]])

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


def _grid_features(img8: np.ndarray, allowed: np.ndarray, max_features: int) -> Optional[np.ndarray]:
    H, W = img8.shape
    per = max(4, max_features // (GRID * GRID))
    pts = []
    for gy in range(GRID):
        for gx in range(GRID):
            y0, y1 = gy * H // GRID, (gy + 1) * H // GRID
            x0, x1 = gx * W // GRID, (gx + 1) * W // GRID
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
    pred = src[inl] @ M[:, :2].T + M[:, 2]
    rms = float(np.sqrt(np.mean(np.sum((pred - dst[inl]) ** 2, axis=1)))) if n_in else float("nan")
    coverage = _coverage(dst[inl], allowed)
    c_count = np.clip((n_in - params.min_inliers) / 40.0, 0, 1)
    c_ratio = np.clip((n_in / max(n_tracked, 1) - 0.2) / 0.6, 0, 1)
    c_resid = float(np.exp(-(rms / 0.5) ** 2))
    c_cov = np.clip(coverage / 0.5, 0, 1)
    quality = float((max(c_count, 1e-3) * max(c_ratio, 1e-3) * c_resid * max(c_cov, 1e-3)) ** 0.25)
    quality *= 1.0 - 0.6 * ambiguity
    return dict(M=M, n_tracked=n_tracked, n_in=n_in, rms=rms, coverage=coverage, quality=quality, flags=flags)


def register_pair(prev_raw: np.ndarray, cur_raw: np.ndarray, prev_c: np.ndarray, cur_c: np.ndarray,
                  excluded: np.ndarray, params: CameraParams) -> PairResult:
    """Estimate background motion prev -> cur. `excluded` is uint8 (255 = do not use)."""
    flags: list[str] = []
    allowed = _allowed(excluded, params.border_px)
    pc_raw = _phase_corr(prev_raw, cur_raw, params.pc_max_width, None)
    pc_clean = _phase_corr(prev_c, cur_c, params.pc_max_width, excluded)
    est = _feature_register(prev_c, cur_c, excluded, allowed, params)
    if est is not None:
        M = est["M"]
        flags += est["flags"]
        d_pc = float(np.hypot(M[0, 2] - pc_clean[0], M[1, 2] - pc_clean[1]))
        agree = 0.5 + 0.5 * np.exp(-(d_pc / 1.5) ** 2) if pc_clean[2] > 0.08 else 1.0
        if pc_clean[2] > 0.08 and d_pc > 3:
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


def object_exclusion(shape, boxes: list[Optional[tuple[float, float, float, float]]], margin: int) -> np.ndarray:
    m = np.zeros(shape[:2], np.uint8)
    for bx in boxes:
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
                           progress=None, cancel=None) -> CameraTrack:
    """Register every consecutive frame pair. object_boxes(k) -> (cx, cy, w, h) or None.

    All returned transforms are in full-resolution pixel coordinates.
    """
    W = frames[0].shape[1]
    if W <= params.max_width:
        return _estimate(frames, masks, params, object_boxes, progress, cancel)
    f = params.max_width / W
    sm = [Mask(int(m.x * f), int(m.y * f), int(np.ceil(m.w * f)) + 1, int(np.ceil(m.h * f)) + 1, m.kind, m.label)
          for m in masks]
    ob = None
    if object_boxes is not None:
        def ob(k):
            b = object_boxes(k)
            return None if b is None else tuple(v * f for v in b)
    ct = _estimate(_Scaled(frames, f), sm, params, ob, progress, cancel)
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


def _estimate(frames, masks: list[Mask], params: CameraParams, object_boxes, progress, cancel) -> CameraTrack:
    n = len(frames)
    shape = frames[0].shape
    user_excl = mask_image(masks, shape)
    if user_excl.any():
        user_excl = cv2.dilate(user_excl, np.ones((7, 7), np.uint8))
    fpn: Optional[FPNModel] = None
    fpn_info = "FPN removal disabled."
    if params.fpn_mode != "off" and n >= 3:
        if progress:
            progress(0.0, "Estimating fixed-pattern noise")
        fpn = estimate_fpn(frames, user_excl, hp_sigma=params.hp_sigma_px)
        fpn_info = fpn.describe()
    two_pass = fpn is not None and params.fpn_refine and fpn.static_fraction > 0.5
    span = 0.45 if two_pass else 0.9
    ct = _register_all(frames, fpn, user_excl, params, object_boxes, progress, cancel, 0.0, span)
    if two_pass:
        if progress:
            progress(span, "Refining FPN with the scene model")
        fpn = refine_fpn_with_motion(frames, ct.A, fpn, user_excl)
        ct = _register_all(frames, fpn, user_excl, params, object_boxes, progress, cancel, span, 0.9)
        fpn_info += " Refined in a second pass using the scene mosaic (scene-based nonuniformity correction)."
    ct.fpn_info = fpn_info
    if progress:
        progress(1.0, "Camera motion done")
    return ct


def _register_all(frames, fpn, user_excl, params, object_boxes, progress, cancel, p0, p1) -> CameraTrack:
    n = len(frames)
    shape = frames[0].shape
    defect_excl = cv2.dilate(fpn.defects, np.ones((5, 5), np.uint8)) if fpn is not None else None

    def prep(k):
        return clean_frame(frames[k], fpn, params.destripe, params.hp_sigma_px, params.smooth_sigma_px)

    def exclusion(*ks):
        excl = user_excl.copy()
        if defect_excl is not None:
            excl |= defect_excl
        if object_boxes is not None:
            excl |= object_exclusion(shape, [object_boxes(k) for k in ks], params.object_margin_px)
        return excl

    pairs = [PairResult(M=np.array([[1, 0, 0], [0, 1, 0]], float), confidence=1.0, method="ref")]
    A = np.zeros((n, 3, 3))
    A[0] = np.eye(3)
    mad = np.full(n, np.nan)
    prev_c = prep(0)
    for k in range(1, n):
        if cancel and cancel():
            raise InterruptedError("Cancelled")
        cur_c = prep(k)
        mad[k] = float(cv2.absdiff(frames[k], frames[k - 1]).mean())
        pr = register_pair(frames[k - 1], frames[k], prev_c, cur_c, exclusion(k - 1, k), params)
        pairs.append(pr)
        A[k] = A[k - 1] @ np.linalg.inv(np.vstack([pr.M, [0, 0, 1]]))
        prev_c = cur_c
        if progress and k % 5 == 0:
            progress(p0 + (p1 - p0) * k / n, f"Registering frame {k}/{n - 1}")
    ct = CameraTrack(pairs=pairs, A=A)

    # --- Compression lag check and I-frame anchor correction
    step = np.array([0.0] + [float(np.hypot(p.M[0, 2], p.M[1, 2])) for p in pairs[1:]])
    keys, catchup = detect_gop_lag(mad, step, params.keyframe_mad_ratio)
    ct.gop_info = f"{len(keys)} keyframe-like frame-difference spikes; camera-step catch-up ratio {catchup:.2f}."
    if len(keys) >= 2 and catchup > 1.3:
        ct.gop_info += (" Background motion appears to lag between keyframes and catch up at them "
                        "(compression freezing a low-texture background).")
        if params.gop_mode == "auto":
            _anchor_correct(ct, frames, [0] + keys, prep, exclusion, params)
            ct.gop_info += f" Corrected by registering {len(ct.anchors)} keyframes directly and interpolating."
        else:
            ct.gop_info += " Anchor correction is OFF; treat in-between camera motion as unreliable."
    return ct


def _anchor_correct(ct: CameraTrack, frames, anchors: list[int], prep, exclusion, params: CameraParams) -> None:
    n = len(frames)
    anchors = sorted(set(a for a in anchors if 0 <= a < n))
    A_chain = ct.A.copy()
    A_anc = {anchors[0]: A_chain[anchors[0]]}
    anc_conf, anc_sigma = {anchors[0]: 1.0}, {anchors[0]: 0.0}
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
            anc_sigma[b] = _step_sigma(est["rms"], est["n_in"], est["quality"])
        prev_c = cur_c
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
    ct.A = A_new
    ct.anchors = anchors
    for k in range(1, n):
        if k > last:
            pr = ct.pairs[k]
            pr.flags = pr.flags + ["AFTER_LAST_KEYFRAME"]
            pr.confidence = min(pr.confidence, 0.2)
            continue
        i = int(np.searchsorted(ks, k))   # first anchor >= k (ks[0] == 0 < k, so i >= 1)
        prv, nxt = int(ks[i - 1]), int(ks[i])
        pr = ct.pairs[k]
        T = np.linalg.inv(A_new[k]) @ A_new[k - 1]
        pr.M = T[:2]
        pr.method = "keyframe-anchored"
        pr.flags = pr.flags + ["GOP_ANCHOR_INTERPOLATED"]
        pr.confidence = float(min(anc_conf[prv], anc_conf[nxt])) * 0.8
        pr.sigma_px = float(max(anc_sigma[nxt], 0.05))


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
