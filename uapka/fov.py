"""Field-of-view estimation from things an analyst can actually know.

Released footage almost never states its field of view, and without one every
angular result stays in pixels. This module DERIVES a horizontal FOV from
reference information, with a 1-sigma uncertainty and a full record of the
inputs, so that it can be fed into the kinematics pipeline and audited later.

Geometry (shared by every method)
---------------------------------
Pinhole camera, principal point at the image centre (W/2, H/2), square pixels
unless a pixel aspect ratio is given. For a focal length f (px), an image point
(x, y) corresponds to the ray

    r = (x - W/2, (y - H/2) / par, f)

and the angle between two image points is the angle between their rays. Given
the TRUE angle theta between two points (from any of the methods below), f is
the root of  angle(p1, p2; f) - theta = 0, which is monotone in f and solved
with Brent's method on f in [W/20, 200 W] (HFOV from ~168 deg down to ~0.3 deg).
Then HFOV = 2 atan(W / (2 f)).

The small-angle shortcut HFOV ~= theta * W / d (d = pixel separation) is only
computed as a sanity check: for points away from the centre of a wide-angle
frame it is wrong by several percent.

Methods (each returns a FovEstimate, labelled DERIVED-FOV(<method>))
  separation  two clicked points + a known angle between them (+ its sigma)
  size        two clicked points spanning an object of ASSUMED physical size
              L at ASSUMED range R:  theta = 2 atan(L / 2R)
  geo         camera position (lat, lon, alt) and >= 2 landmarks with known
              coordinates, each clicked in the frame. True angles come from the
              landmark directions in a local ENU frame (WGS-84 ellipsoid). With
              >= 3 landmarks all pairs are fitted together and the residual is
              reported as a consistency check. This is the method for
              coastline / terrain footage.
  hud         a user-supplied table of HUD zoom labels -> HFOV (lookup only).

Uncertainty
-----------
Click uncertainty (default +-1 px per point) and every input sigma are
propagated by central finite differences: each input is perturbed by +-sigma,
the resulting HFOV shifts are combined in quadrature. Simple and transparent;
the per-input contributions are kept in `diagnostics`. Angular rates and every
scenario value scale LINEARLY with the HFOV, so the relative sigma reported
here (sigma_pct) carries straight through to omega, alpha, speed and g.

No Qt in here: the GUI, the CLI and the tests all call the same functions.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from .provenance import derived_fov_label

Point = tuple[float, float]

F_MIN_FACTOR = 1.0 / 20.0     # f >= W/20   -> HFOV <= ~168 deg
F_MAX_FACTOR = 200.0          # f <= 200 W  -> HFOV >= ~0.29 deg
DEFAULT_CLICK_SIGMA_PX = 1.0

# WGS-84
_A = 6378137.0
_F = 1.0 / 298.257223563
_E2 = _F * (2.0 - _F)


# ---------------------------------------------------------------- pinhole geometry
def focal_px(width: int, hfov_deg: float) -> float:
    """Focal length in pixels for a horizontal FOV (same formula as kinematics.py)."""
    return (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


def hfov_from_focal(width: int, f_px: float) -> float:
    return math.degrees(2.0 * math.atan(width / (2.0 * f_px)))


def ray(p: Point, width: int, height: int, f_px: float, pixel_aspect: float = 1.0) -> np.ndarray:
    """Direction of the ray through image point p (any orientation convention works for angles)."""
    par = pixel_aspect if pixel_aspect > 0 else 1.0
    return np.array([p[0] - width / 2.0, (p[1] - height / 2.0) / par, f_px], float)


def angle_between_points(p1: Point, p2: Point, width: int, height: int, f_px: float,
                         pixel_aspect: float = 1.0) -> float:
    """Angle (rad) between the rays through two image points, exact pinhole."""
    a = ray(p1, width, height, f_px, pixel_aspect)
    b = ray(p2, width, height, f_px, pixel_aspect)
    return float(math.atan2(np.linalg.norm(np.cross(a, b)), float(a @ b)))


def project_direction(d: np.ndarray, width: int, height: int, f_px: float, forward: np.ndarray, right: np.ndarray,
                      down: np.ndarray, pixel_aspect: float = 1.0) -> Point:
    """Forward model used by tests and the GUI preview: pixel of a unit direction `d` for a camera whose
    optical axis is `forward`, image x along `right`, image y (downwards on screen) along `down`."""
    par = pixel_aspect if pixel_aspect > 0 else 1.0
    w = float(d @ forward)
    if w <= 0:
        raise ValueError("direction is behind the camera")
    return (width / 2.0 + f_px * float(d @ right) / w, height / 2.0 + f_px * float(d @ down) / w * par)


def small_angle_hfov(p1: Point, p2: Point, width: int, height: int, theta_deg: float,
                     pixel_aspect: float = 1.0) -> float:
    """The textbook shortcut HFOV = theta * W / d. Sanity check only (see module docstring)."""
    par = pixel_aspect if pixel_aspect > 0 else 1.0
    d = math.hypot(p2[0] - p1[0], (p2[1] - p1[1]) / par)
    if d <= 0:
        return float("nan")
    return theta_deg * width / d


def _log_grid(lo: float, hi: float, n: int = 256) -> np.ndarray:
    return np.exp(np.linspace(math.log(lo), math.log(hi), n))


def _refine(fun, grid: np.ndarray, k: int) -> float:
    """Minimise fun(f) between the grid neighbours of index k (bounded Brent on log f)."""
    a, b = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
    if b <= a:
        return float(grid[k])
    res = minimize_scalar(lambda u: fun(math.exp(u)), bounds=(math.log(a), math.log(b)), method="bounded",
                          options={"xatol": 1e-12, "maxiter": 500})
    return float(math.exp(res.x))


def solve_focal(width: int, height: int, pairs: Sequence[tuple[Point, Point, float]], pixel_aspect: float = 1.0,
                weights: Optional[Sequence[float]] = None) -> tuple[float, np.ndarray, Optional[float]]:
    """Focal length (px) that reproduces the true angles of one or more point pairs.

    pairs: (p1, p2, theta_rad). Returns (f_px, residuals_rad, alt_f_px): residuals are angle(f) - theta per pair.

    The angle between two rays is NOT always monotone in f: for points that straddle the image centre it falls
    monotonically from ~180 deg (f -> 0) to 0 (f -> inf), but for two points on the SAME side of the centre it rises
    from ~0, peaks near f = sqrt(r1 r2) and then falls. A known angle can therefore match two focal lengths. The
    solution on the falling (narrow-FOV) branch is returned; the other, very wide-angle one is returned as alt_f_px
    (None if there is none) so the caller can say so.
    One pair: exact root (Brent). Several: weighted least squares of the angle residuals over f, located on a
    log-spaced grid and refined with bounded Brent, so a secondary local minimum cannot capture the fit.
    """
    if not pairs:
        raise ValueError("at least one point pair is needed")
    lo, hi = width * F_MIN_FACTOR, width * F_MAX_FACTOR
    for p1, p2, th in pairs:
        if math.hypot(p2[0] - p1[0], p2[1] - p1[1]) < 1e-6:
            raise ValueError("the two points coincide; click two distinct points")
        if not (0 < th < math.pi):
            raise ValueError(f"true angle must be in (0, 180) deg, got {math.degrees(th):.4g}")
    grid = _log_grid(lo, hi)
    if len(pairs) == 1:
        p1, p2, th = pairs[0]

        def ang(f):
            return angle_between_points(p1, p2, width, height, f, pixel_aspect)
        vals = np.array([ang(f) for f in grid])
        k = int(np.argmax(vals))
        f_max = lo if k == 0 else _refine(lambda f: -ang(f), grid, k)
        a_max = ang(f_max)
        if a_max < th:
            raise ValueError(f"angle {math.degrees(th):.4g} deg is too large for these points: no pinhole camera "
                             f"gives more than {math.degrees(a_max):.4g} deg between them (at HFOV "
                             f"{hfov_from_focal(width, f_max):.0f} deg). Check the inputs")
        if ang(hi) > th:
            raise ValueError(f"angle {math.degrees(th):.4g} deg is too small for the pixel separation "
                             f"(would need HFOV < {hfov_from_focal(width, hi):.2g} deg): check the inputs")
        f = brentq(lambda f: ang(f) - th, f_max, hi, xtol=1e-9, rtol=1e-12, maxiter=200)
        alt = None
        if f_max > lo and ang(lo) < th:      # rising branch also crosses theta: a second, very wide-angle solution
            alt = brentq(lambda f: ang(f) - th, lo, f_max, xtol=1e-9, rtol=1e-12, maxiter=200)
        return float(f), np.array([ang(f) - th]), alt

    w = np.ones(len(pairs)) if weights is None else np.asarray(weights, float)

    def resid(f):
        return np.array([angle_between_points(p1, p2, width, height, f, pixel_aspect) - th for p1, p2, th in pairs])

    def cost(f):
        r = resid(f)
        return float((w * r * r).sum())
    costs = np.array([cost(f) for f in grid])
    k = int(np.argmin(costs))
    f = _refine(cost, grid, k)
    # a second local minimum elsewhere on the grid (same-side geometry) is reported if it fits nearly as well
    alt = None
    local = [i for i in range(1, len(grid) - 1) if costs[i] <= costs[i - 1] and costs[i] <= costs[i + 1]
             and abs(i - k) > 2]
    if local:
        j = min(local, key=lambda i: costs[i])
        fj = _refine(cost, grid, j)
        if cost(fj) <= 4.0 * max(cost(f), 1e-18):
            alt = fj
    return float(f), resid(f), alt


# ---------------------------------------------------------------- geodesy (WGS-84, no library needed)
def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    s, c = math.sin(lat), math.cos(lat)
    n = _A / math.sqrt(1.0 - _E2 * s * s)
    return np.array([(n + alt_m) * c * math.cos(lon), (n + alt_m) * c * math.sin(lon), (n * (1.0 - _E2) + alt_m) * s])


def ecef_to_enu(vec: np.ndarray, lat_deg: float, lon_deg: float) -> np.ndarray:
    """Rotate an ECEF difference vector into the local East/North/Up frame at (lat, lon)."""
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    sl, cl, sp, cp = math.sin(lon), math.cos(lon), math.sin(lat), math.cos(lat)
    R = np.array([[-sl, cl, 0.0], [-sp * cl, -sp * sl, cp], [cp * cl, cp * sl, sp]])
    return R @ vec


def enu_vector(cam: tuple[float, float, float], target: tuple[float, float, float]) -> np.ndarray:
    """ENU vector (m) from the camera (lat, lon, alt) to a target (lat, lon, alt)."""
    return ecef_to_enu(geodetic_to_ecef(*target) - geodetic_to_ecef(*cam), cam[0], cam[1])


def az_el_range(enu: np.ndarray) -> tuple[float, float, float]:
    """Azimuth (deg, clockwise from north), elevation (deg) and range (m) of an ENU vector."""
    e, n, u = (float(v) for v in enu)
    rng = math.sqrt(e * e + n * n + u * u)
    return math.degrees(math.atan2(e, n)) % 360.0, math.degrees(math.atan2(u, math.hypot(e, n))), rng


# ---------------------------------------------------------------- result
@dataclass
class FovEstimate:
    hfov_deg: float
    sigma_deg: float
    method: str                      # separation | size | geo | hud
    width: int
    height: int
    pixel_aspect: float = 1.0
    inputs: dict = field(default_factory=dict)        # everything that went in, JSON-serialisable
    notes: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)   # sigma contributions, small-angle check, geo residuals...

    @property
    def label(self) -> str:
        return derived_fov_label(self.method)

    @property
    def sigma_pct(self) -> float:
        return 100.0 * self.sigma_deg / self.hfov_deg if self.hfov_deg > 0 else float("nan")

    @property
    def focal_px(self) -> float:
        return focal_px(self.width, self.hfov_deg)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["label"] = self.label
        d["sigma_pct"] = self.sigma_pct
        return d

    @staticmethod
    def from_dict(d: dict) -> "FovEstimate":
        keys = ("hfov_deg", "sigma_deg", "method", "width", "height", "pixel_aspect", "inputs", "notes", "diagnostics")
        return FovEstimate(**{k: d[k] for k in keys if k in d})

    def summary(self) -> str:
        s = f"HFOV = {self.hfov_deg:.4g} ± {self.sigma_deg:.2g} deg ({self.sigma_pct:.1f}%) [{self.label}]"
        return s


SCALING_NOTE = ("Every angular rate (deg/s, deg/s²) and every scenario value (m/s, g) scales linearly with the HFOV, "
                "so this relative uncertainty applies to them unchanged (it is not included in their ±1σ).")


def describe_inputs(est: dict) -> str:
    """One-line, human-readable description of an estimate's inputs (for reports)."""
    m, inp = est.get("method", "?"), est.get("inputs", {})
    size = f"{est.get('width', '?')}x{est.get('height', '?')} px"
    if m == "separation":
        return (f"points {_pt(inp.get('p1'))} and {_pt(inp.get('p2'))} in a {size} frame, true angle "
                f"{inp.get('angle_deg', float('nan')):.4g} ± {inp.get('angle_sigma_deg', 0):.2g} deg, "
                f"click σ {inp.get('click_sigma_px', 0):g} px")
    if m == "size":
        return (f"points {_pt(inp.get('p1'))} and {_pt(inp.get('p2'))} in a {size} frame spanning an ASSUMED size of "
                f"{inp.get('size_m', float('nan')):.4g} ± {inp.get('size_sigma_m', 0):.2g} m at an ASSUMED range of "
                f"{inp.get('range_m', float('nan')):.5g} ± {inp.get('range_sigma_m', 0):.3g} m, "
                f"click σ {inp.get('click_sigma_px', 0):g} px")
    if m == "geo":
        cam = inp.get("camera", {})
        lms = inp.get("landmarks", [])
        lm_txt = "; ".join(f"{_pt(l.get('px'))} -> ({l.get('lat')}, {l.get('lon')}, {l.get('alt_m')} m)" for l in lms)
        return (f"camera at ({cam.get('lat')}, {cam.get('lon')}, {cam.get('alt_m')} m) ± {inp.get('camera_sigma_m', 0):g} m, "
                f"{len(lms)} landmarks in a {size} frame ({lm_txt}), landmark position σ "
                f"{inp.get('landmark_sigma_m', 0):g} m, click σ {inp.get('click_sigma_px', 0):g} px")
    if m == "hud":
        return (f"HUD label {inp.get('label')!r} looked up in the user-supplied table {inp.get('table')} "
                f"for a {size} frame")
    return f"inputs: {inp}"


def _pt(p) -> str:
    try:
        return f"({p[0]:.1f}, {p[1]:.1f})"
    except (TypeError, IndexError):
        return "(?)"


# ---------------------------------------------------------------- uncertainty propagation
def _propagate(fn, params: dict, sigmas: dict) -> tuple[float, float, dict]:
    """Central finite differences: perturb each input by ±sigma, combine the HFOV shifts in quadrature.

    Returns (value, sigma, contributions) where contributions[name] is that input's share (deg, 1 sigma).
    """
    base = fn(params)
    var, contrib = 0.0, {}
    for k, s in sigmas.items():
        if not s or s <= 0:
            continue
        up = fn({**params, k: params[k] + s})
        dn = fn({**params, k: params[k] - s})
        d = 0.5 * (up - dn)
        var += d * d
        contrib[k] = abs(float(d))
    return float(base), float(math.sqrt(var)), contrib


def _click_sigmas(names: Sequence[str], click_sigma_px: float) -> dict:
    return {f"{n}.{c}": click_sigma_px for n in names for c in ("x", "y")}


def _pts(params: dict, names: Sequence[str]) -> list[Point]:
    return [(params[f"{n}.x"], params[f"{n}.y"]) for n in names]


def _check_frame(width: int, height: int, pts: Sequence[Point]) -> list[str]:
    notes = []
    if width <= 0 or height <= 0:
        raise ValueError("frame width and height must be positive")
    for p in pts:
        if not (-0.5 <= p[0] <= width + 0.5 and -0.5 <= p[1] <= height + 0.5):
            notes.append(f"point {_pt(p)} lies outside the {width}x{height} frame; check the coordinates")
    return notes


def _finish(est: FovEstimate, p1: Point, p2: Point, theta_deg: float, alt_f_px: Optional[float] = None) -> FovEstimate:
    """Common notes: second (wide-angle) solution, small-angle sanity check and the linear-scaling reminder."""
    if alt_f_px is not None:
        alt = hfov_from_focal(est.width, alt_f_px)
        est.diagnostics["alternative_hfov_deg"] = alt
        est.notes.append(f"A second, very wide-angle solution (HFOV ≈ {alt:.3g} deg) also matches these points "
                         "because they lie on the same side of the image centre; the narrow-angle solution is used. "
                         "If the footage really is that wide, check with points on opposite sides of the centre.")
    sa = small_angle_hfov(p1, p2, est.width, est.height, theta_deg, est.pixel_aspect)
    est.diagnostics["small_angle_hfov_deg"] = sa
    if np.isfinite(sa) and est.hfov_deg > 0:
        dev = 100.0 * (sa - est.hfov_deg) / est.hfov_deg
        est.diagnostics["small_angle_error_pct"] = dev
        if abs(dev) > 1.0:
            est.notes.append(f"The small-angle shortcut (θ·W/d) would give {sa:.4g} deg, {dev:+.1f}% off: the exact "
                             "pinhole solution is used because the points are off-centre at this FOV.")
    est.notes.append(SCALING_NOTE)
    return est


# ---------------------------------------------------------------- methods
def estimate_separation(width: int, height: int, p1: Point, p2: Point, angle_deg: float, angle_sigma_deg: float = 0.0,
                        click_sigma_px: float = DEFAULT_CLICK_SIGMA_PX, pixel_aspect: float = 1.0,
                        frame: Optional[int] = None) -> FovEstimate:
    """Two clicked points with a KNOWN true angle between them (e.g. two stars, two surveyed towers)."""
    if angle_deg <= 0:
        raise ValueError("the true angle must be positive")
    if angle_sigma_deg >= angle_deg:
        raise ValueError("the angle sigma must be smaller than the angle itself (a ±σ perturbation would cross zero)")
    notes = _check_frame(width, height, [p1, p2])
    params = {"p1.x": p1[0], "p1.y": p1[1], "p2.x": p2[0], "p2.y": p2[1], "angle_deg": float(angle_deg)}
    sig = {**_click_sigmas(("p1", "p2"), click_sigma_px), "angle_deg": float(angle_sigma_deg)}

    def solve(q):
        a, b = _pts(q, ("p1", "p2"))
        return solve_focal(width, height, [(a, b, math.radians(q["angle_deg"]))], pixel_aspect)

    def fn(q):
        return hfov_from_focal(width, solve(q)[0])
    hfov, sigma, contrib = _propagate(fn, params, sig)
    alt = solve(params)[2]
    est = FovEstimate(hfov, sigma, "separation", width, height, pixel_aspect,
                      inputs=dict(p1=list(map(float, p1)), p2=list(map(float, p2)), angle_deg=float(angle_deg),
                                  angle_sigma_deg=float(angle_sigma_deg), click_sigma_px=float(click_sigma_px),
                                  frame=frame),
                      notes=notes, diagnostics={"sigma_contributions_deg": contrib, "focal_px": focal_px(width, hfov)})
    if angle_sigma_deg <= 0:
        est.notes.append("No uncertainty was given for the true angle; σ reflects click uncertainty only.")
    return _finish(est, p1, p2, angle_deg, alt)


def estimate_size(width: int, height: int, p1: Point, p2: Point, size_m: float, range_m: float,
                  size_sigma_m: float = 0.0, range_sigma_m: float = 0.0, click_sigma_px: float = DEFAULT_CLICK_SIGMA_PX,
                  pixel_aspect: float = 1.0, frame: Optional[int] = None) -> FovEstimate:
    """Two points spanning an object of ASSUMED size L at ASSUMED range R: theta = 2 atan(L / 2R).

    Both inputs are assumptions, so the result inherits them: a wrong size or range gives a wrong FOV in the same
    proportion (to first order). In IR, blur and blooming inflate apparent extents, which biases this FOV LARGE.
    """
    if size_m <= 0 or range_m <= 0:
        raise ValueError("size and range must be positive")
    notes = _check_frame(width, height, [p1, p2])
    params = {"p1.x": p1[0], "p1.y": p1[1], "p2.x": p2[0], "p2.y": p2[1], "size_m": float(size_m),
              "range_m": float(range_m)}
    sig = {**_click_sigmas(("p1", "p2"), click_sigma_px), "size_m": float(size_sigma_m), "range_m": float(range_sigma_m)}
    if range_sigma_m >= range_m or size_sigma_m >= size_m:
        raise ValueError("sigma must be smaller than the value itself (a ±σ perturbation would cross zero)")

    def solve(q):
        a, b = _pts(q, ("p1", "p2"))
        th = 2.0 * math.atan(q["size_m"] / (2.0 * q["range_m"]))
        return solve_focal(width, height, [(a, b, th)], pixel_aspect)

    def fn(q):
        return hfov_from_focal(width, solve(q)[0])
    theta_deg = math.degrees(2.0 * math.atan(size_m / (2.0 * range_m)))
    hfov, sigma, contrib = _propagate(fn, params, sig)
    alt = solve(params)[2]
    est = FovEstimate(hfov, sigma, "size", width, height, pixel_aspect,
                      inputs=dict(p1=list(map(float, p1)), p2=list(map(float, p2)), size_m=float(size_m),
                                  range_m=float(range_m), size_sigma_m=float(size_sigma_m),
                                  range_sigma_m=float(range_sigma_m), click_sigma_px=float(click_sigma_px),
                                  frame=frame),
                      notes=notes, diagnostics={"sigma_contributions_deg": contrib, "focal_px": focal_px(width, hfov),
                                                "angle_deg": theta_deg})
    est.notes.append("Size and range are ASSUMED: the FOV (and everything derived from it) scales with size/range. "
                     "If the span was measured on an IR blob, blur/blooming inflate it and bias this FOV large.")
    return _finish(est, p1, p2, theta_deg, alt)


def estimate_geo(width: int, height: int, camera: tuple[float, float, float],
                 landmarks: Sequence[tuple[Point, tuple[float, float, float]]], camera_sigma_m: float = 0.0,
                 landmark_sigma_m: float = 0.0, click_sigma_px: float = DEFAULT_CLICK_SIGMA_PX,
                 pixel_aspect: float = 1.0, frame: Optional[int] = None) -> FovEstimate:
    """Camera position + >= 2 landmarks with known (lat, lon, alt_m), each clicked in the frame.

    landmarks: [((x_px, y_px), (lat, lon, alt_m)), ...]. True angles between landmark directions are computed in the
    local ENU frame at the camera (WGS-84). All landmark pairs are fitted together; with >= 3 landmarks the residual
    RMS is a consistency check (misidentified landmark, wrong camera position, lens distortion).
    camera_sigma_m / landmark_sigma_m apply to each coordinate axis (E, N, U) of the camera / every landmark.
    """
    if len(landmarks) < 2:
        raise ValueError("at least two landmarks are needed")
    pts = [lm[0] for lm in landmarks]
    notes = _check_frame(width, height, pts)
    names = [f"lm{i}" for i in range(len(landmarks))]
    params: dict = {"cam.e": 0.0, "cam.n": 0.0, "cam.u": 0.0}
    sig: dict = {"cam.e": camera_sigma_m, "cam.n": camera_sigma_m, "cam.u": camera_sigma_m}
    for n, (p, geo) in zip(names, landmarks):
        params[f"{n}.x"], params[f"{n}.y"] = float(p[0]), float(p[1])
        params[f"{n}.e"] = params[f"{n}.n"] = params[f"{n}.u"] = 0.0
        sig[f"{n}.e"] = sig[f"{n}.n"] = sig[f"{n}.u"] = landmark_sigma_m
    sig.update(_click_sigmas(names, click_sigma_px))
    # Positions are perturbed in metres in the ENU frame (lat/lon perturbations in degrees would be opaque).
    enu0 = [enu_vector(camera, geo) for _, geo in landmarks]
    pairs_idx = [(i, j) for i in range(len(landmarks)) for j in range(i + 1, len(landmarks))]
    for (i, j) in pairs_idx:
        if math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]) < 1e-6:
            raise ValueError(f"landmarks {i} and {j} were clicked at the same pixel")

    def directions(q):
        cam = np.array([q["cam.e"], q["cam.n"], q["cam.u"]])
        return [enu0[i] + np.array([q[f"{n}.e"], q[f"{n}.n"], q[f"{n}.u"]]) - cam for i, n in enumerate(names)]

    def solve(q):
        d = directions(q)
        pairs = []
        for (i, j) in pairs_idx:
            th = math.atan2(np.linalg.norm(np.cross(d[i], d[j])), float(d[i] @ d[j]))
            pairs.append(((q[f"{names[i]}.x"], q[f"{names[i]}.y"]), (q[f"{names[j]}.x"], q[f"{names[j]}.y"]), th))
        return solve_focal(width, height, pairs, pixel_aspect)

    def fn(q):
        return hfov_from_focal(width, solve(q)[0])
    hfov, sigma, contrib = _propagate(fn, params, sig)
    f, resid, alt = solve(params)
    geo_rows = []
    for (p, geo), v in zip(landmarks, enu0):
        az, el, rng = az_el_range(v)
        geo_rows.append(dict(px=list(map(float, p)), lat=float(geo[0]), lon=float(geo[1]), alt_m=float(geo[2]),
                             azimuth_deg=az, elevation_deg=el, range_m=rng))
    diag: dict = {"sigma_contributions_deg": contrib, "focal_px": f, "landmarks": geo_rows,
                  "pair_true_angles_deg": [math.degrees(math.atan2(np.linalg.norm(np.cross(enu0[i], enu0[j])),
                                                                   float(enu0[i] @ enu0[j]))) for i, j in pairs_idx],
                  "pair_residuals_deg": [math.degrees(r) for r in resid]}
    est = FovEstimate(hfov, sigma, "geo", width, height, pixel_aspect,
                      inputs=dict(camera=dict(lat=float(camera[0]), lon=float(camera[1]), alt_m=float(camera[2])),
                                  landmarks=[dict(px=list(map(float, p)), lat=float(g[0]), lon=float(g[1]),
                                                  alt_m=float(g[2])) for p, g in landmarks],
                                  camera_sigma_m=float(camera_sigma_m), landmark_sigma_m=float(landmark_sigma_m),
                                  click_sigma_px=float(click_sigma_px), frame=frame),
                      notes=notes, diagnostics=diag)
    if len(landmarks) >= 3:
        rms = math.degrees(math.sqrt(float(np.mean(resid ** 2))))
        diag["residual_rms_deg"] = rms
        # expected residual from click noise alone: ~sqrt(2) * sigma_px / f per pair
        expect = math.degrees(math.sqrt(2.0) * click_sigma_px / f)
        diag["expected_click_residual_deg"] = expect
        if rms > 3.0 * max(expect, 1e-9):
            est.notes.append(f"Consistency check FAILED: {len(pairs_idx)} landmark pairs disagree with a single focal "
                             f"length (residual RMS {rms:.3g} deg vs ~{expect:.2g} deg expected from clicks). "
                             "Check landmark identification, the camera position, or lens distortion.")
        else:
            est.notes.append(f"Consistency check: {len(pairs_idx)} landmark pairs fitted with residual RMS "
                             f"{rms:.2g} deg (clicks alone would give ~{expect:.2g} deg).")
    else:
        est.notes.append("Only two landmarks: no consistency check possible (a third landmark would give one).")
    if camera_sigma_m <= 0 and landmark_sigma_m <= 0:
        est.notes.append("No position uncertainties given: σ reflects click uncertainty only.")
    i, j = pairs_idx[0]
    return _finish(est, pts[i], pts[j], diag["pair_true_angles_deg"][0], alt)


def estimate_hud(width: int, height: int, label: str, table: dict, sigma_pct: float = 0.0,
                 pixel_aspect: float = 1.0, frame: Optional[int] = None) -> FovEstimate:
    """Look a HUD zoom / FOV-mode label up in a user-supplied {label: hfov_deg} table.

    The table is the analyst's claim (from a manual, a published spec, or a calibration); the label is the provenance.
    Labels are matched case-insensitively.
    """
    norm = {str(k).strip().upper(): float(v) for k, v in table.items()}
    key = str(label).strip().upper()
    if key not in norm:
        raise KeyError(f"HUD label {label!r} is not in the table {sorted(table)}")
    hfov = norm[key]
    if hfov <= 0:
        raise ValueError(f"table entry for {label!r} must be a positive HFOV in degrees")
    est = FovEstimate(hfov, hfov * sigma_pct / 100.0, "hud", width, height, pixel_aspect,
                      inputs=dict(label=str(label), table={str(k): float(v) for k, v in table.items()},
                                  sigma_pct=float(sigma_pct), frame=frame))
    est.notes.append("The HUD label -> FOV table is user-supplied; the label is the only provenance. HUD zoom labels "
                     "are often digital zoom levels, not optical FOVs, and the same label can mean different FOVs on "
                     "different sensors.")
    if sigma_pct <= 0:
        est.notes.append("No uncertainty given for the table value.")
    est.notes.append(SCALING_NOTE)
    return est


METHODS = {"separation": estimate_separation, "size": estimate_size, "geo": estimate_geo, "hud": estimate_hud}


# ---------------------------------------------------------------- JSON files (CLI --save / analyze --fov-json)
def load_fov_json(path: str) -> list[dict]:
    """Read estimates from a file written by save_fov_json (or a bare list / single estimate dict)."""
    import json
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    if isinstance(d, dict) and "fov_estimates" in d:
        return list(d["fov_estimates"])
    if isinstance(d, dict) and "hfov_deg" in d:
        return [d]
    if isinstance(d, list):
        return list(d)
    raise ValueError(f"{path}: not a FOV estimate file")


def save_fov_json(path: str, estimate: FovEstimate | dict, append: bool = True) -> list[dict]:
    """Write (or append to) a {'fov_estimates': [...]} file; returns the list now in the file."""
    import json
    import os
    est = estimate.to_dict() if isinstance(estimate, FovEstimate) else dict(estimate)
    ests = load_fov_json(path) if (append and os.path.exists(path)) else []
    ests.append(est)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"tool": "uapka", "fov_estimates": ests}, fh, indent=1)
    return ests
