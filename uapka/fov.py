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
  ground      look-down footage with an UNKNOWN camera position but >= 4 ground
              features of known (lat, lon, alt) clicked in one frame: plane
              resection through a homography (see estimate_ground). Returns the
              HFOV and, as a by-product, the camera position, slant range,
              look-down angle and metres-per-pixel at the image centre.

Active image region (inset)
---------------------------
Released sensor video is often a sensor image INSET in a larger frame: a
1280x720 file whose live image occupies x = 360..922, the rest being redaction
or data panels. The pinhole principal point is then the centre of the inset,
not of the file frame, and the HFOV is that of the inset width. Every
estimate_* function takes an optional region=(x0, y0, w, h) in full-frame
pixels: clicked points are translated into inset coordinates and the inset
width/height are used for the solve, so hfov_deg is the HFOV OF THE INSET.
The region is stored in the estimate (`region`) and copied by
Project.apply_fov into kin_params.image_region, so that the kinematics pipeline
uses the same principal point and the same width when it converts pixels to
angles. Without that (or without cropping the video to the inset) an HFOV
derived for a 562-px-wide inset would be applied to a 1280-px frame and every
angle would be 2.3x too large. detect_inset() finds the live region
automatically from the sharpness and temporal change of the columns and rows.

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

import cmath
import math
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from .provenance import derived_fov_label

Point = tuple[float, float]
Region = tuple[float, float, float, float]      # (x0, y0, w, h) in full-frame pixels

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


def enu_to_ecef(enu: np.ndarray, lat_deg: float, lon_deg: float) -> np.ndarray:
    """Inverse of ecef_to_enu: an ENU vector at (lat, lon) back to an ECEF difference vector."""
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    sl, cl, sp, cp = math.sin(lon), math.cos(lon), math.sin(lat), math.cos(lat)
    R = np.array([[-sl, cl, 0.0], [-sp * cl, -sp * sl, cp], [cp * cl, cp * sl, sp]])
    return R.T @ np.asarray(enu, float)


def ecef_to_geodetic(xyz: np.ndarray) -> tuple[float, float, float]:
    """ECEF (m) -> (lat_deg, lon_deg, alt_m) on WGS-84, by the usual fixed-point iteration on the latitude
    (converges to < 1e-9 deg in a handful of steps for any point above the core)."""
    x, y, z = (float(v) for v in xyz)
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1.0 - _E2))
    for _ in range(20):
        s = math.sin(lat)
        n = _A / math.sqrt(1.0 - _E2 * s * s)
        new = math.atan2(z + _E2 * n * s, p)
        if abs(new - lat) < 1e-13:
            lat = new
            break
        lat = new
    s, c = math.sin(lat), math.cos(lat)
    n = _A / math.sqrt(1.0 - _E2 * s * s)
    alt = p / c - n if abs(c) > 1e-10 else abs(z) - n * (1.0 - _E2)
    return math.degrees(lat), math.degrees(lon), alt


def enu_to_geodetic(enu: np.ndarray, origin: tuple[float, float, float]) -> tuple[float, float, float]:
    """A point given as an ENU vector (m) from `origin` (lat, lon, alt) -> its (lat, lon, alt)."""
    return ecef_to_geodetic(geodetic_to_ecef(*origin) + enu_to_ecef(enu, origin[0], origin[1]))


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
    region: Optional[list] = None    # [x0, y0, w, h] active image region (inset) in full-frame px; hfov_deg is ITS HFOV

    @property
    def label(self) -> str:
        return derived_fov_label(self.method)

    @property
    def sigma_pct(self) -> float:
        return 100.0 * self.sigma_deg / self.hfov_deg if self.hfov_deg > 0 else float("nan")

    @property
    def solve_width(self) -> float:
        """Width the HFOV refers to: the inset width when a region is set, else the frame width."""
        return float(self.region[2]) if self.region else float(self.width)

    @property
    def solve_height(self) -> float:
        return float(self.region[3]) if self.region else float(self.height)

    @property
    def focal_px(self) -> float:
        return focal_px(self.solve_width, self.hfov_deg)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["label"] = self.label
        d["sigma_pct"] = self.sigma_pct
        return d

    @staticmethod
    def from_dict(d: dict) -> "FovEstimate":
        keys = ("hfov_deg", "sigma_deg", "method", "width", "height", "pixel_aspect", "inputs", "notes", "diagnostics",
                "region")
        est = FovEstimate(**{k: d[k] for k in keys if k in d})
        if est.region is not None:
            est.region = [float(v) for v in est.region]
        return est

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
                f"click σ {inp.get('click_sigma_px', 0):g} px{_region_txt(inp)}")
    if m == "size":
        return (f"points {_pt(inp.get('p1'))} and {_pt(inp.get('p2'))} in a {size} frame spanning an ASSUMED size of "
                f"{inp.get('size_m', float('nan')):.4g} ± {inp.get('size_sigma_m', 0):.2g} m at an ASSUMED range of "
                f"{inp.get('range_m', float('nan')):.5g} ± {inp.get('range_sigma_m', 0):.3g} m, "
                f"click σ {inp.get('click_sigma_px', 0):g} px{_region_txt(inp)}")
    if m == "geo":
        cam = inp.get("camera", {})
        lms = inp.get("landmarks", [])
        lm_txt = "; ".join(f"{_pt(l.get('px'))} -> ({l.get('lat')}, {l.get('lon')}, {l.get('alt_m')} m)" for l in lms)
        return (f"camera at ({cam.get('lat')}, {cam.get('lon')}, {cam.get('alt_m')} m) ± {inp.get('camera_sigma_m', 0):g} m, "
                f"{len(lms)} landmarks in a {size} frame ({lm_txt}), landmark position σ "
                f"{inp.get('landmark_sigma_m', 0):g} m, click σ {inp.get('click_sigma_px', 0):g} px{_region_txt(inp)}")
    if m == "hud":
        return (f"HUD label {inp.get('label')!r} looked up in the user-supplied table {inp.get('table')} "
                f"for a {size} frame{_region_txt(inp)}")
    if m == "ground":
        pts = inp.get("points", [])
        pt_txt = "; ".join(f"{_pt(q.get('px'))} -> ({q.get('lat')}, {q.get('lon')}, {q.get('alt_m')} m)" for q in pts)
        return (f"{len(pts)} ground points of known position clicked in a {size} frame ({pt_txt}), camera position "
                f"unknown (plane resection), point position σ {inp.get('point_sigma_m', 0):g} m, "
                f"click σ {inp.get('click_sigma_px', 0):g} px{_region_txt(inp)}")
    return f"inputs: {inp}"


def _region_txt(inp: dict) -> str:
    r = inp.get("region")
    if not r:
        return ""
    return (f", active image region x={r[0]:g}..{r[0] + r[2]:g}, y={r[1]:g}..{r[1] + r[3]:g} "
            f"({r[2]:g}x{r[3]:g} px inset)")


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


def normalise_region(width: int, height: int, region) -> Optional[list[float]]:
    """Validate an active-image-region spec (x0, y0, w, h) against the frame; None stays None."""
    if region is None:
        return None
    try:
        x0, y0, w, h = (float(v) for v in region)
    except (TypeError, ValueError):
        raise ValueError("the active image region must be (x0, y0, w, h) in full-frame pixels") from None
    if w <= 0 or h <= 0:
        raise ValueError("the active image region must have a positive width and height")
    if x0 < -0.5 or y0 < -0.5 or x0 + w > width + 0.5 or y0 + h > height + 0.5:
        raise ValueError(f"the active image region x={x0:g}..{x0 + w:g}, y={y0:g}..{y0 + h:g} does not fit in the "
                         f"{width}x{height} frame")
    return [x0, y0, w, h]


def _inset(width: int, height: int, region, pts: Sequence[Point]) -> tuple[float, float, list[Point], list[str],
                                                                            Optional[list[float]]]:
    """Translate full-frame points into the active image region (inset). Returns (w, h, points, notes, region):
    the width/height and points the pinhole solve must use, and the validated region (None = whole frame)."""
    notes = _check_frame(width, height, pts)
    reg = normalise_region(width, height, region)
    if reg is None:
        return float(width), float(height), [(float(p[0]), float(p[1])) for p in pts], notes, None
    x0, y0, w, h = reg
    out = [(float(p[0]) - x0, float(p[1]) - y0) for p in pts]
    for p, q in zip(pts, out):
        if not (-0.5 <= q[0] <= w + 0.5 and -0.5 <= q[1] <= h + 0.5):
            notes.append(f"point {_pt(p)} lies outside the active image region x={x0:g}..{x0 + w:g}, "
                         f"y={y0:g}..{y0 + h:g}; check the coordinates or the region")
    return w, h, out, notes, reg


def _finish(est: FovEstimate, p1: Point, p2: Point, theta_deg: float, alt_f_px: Optional[float] = None) -> FovEstimate:
    """Common notes: second (wide-angle) solution, small-angle sanity check and the linear-scaling reminder.
    p1, p2 are in inset coordinates (what the solve used)."""
    if alt_f_px is not None:
        alt = hfov_from_focal(est.solve_width, alt_f_px)
        est.diagnostics["alternative_hfov_deg"] = alt
        est.notes.append(f"A second, very wide-angle solution (HFOV ≈ {alt:.3g} deg) also matches these points "
                         "because they lie on the same side of the image centre; the narrow-angle solution is used. "
                         "If the footage really is that wide, check with points on opposite sides of the centre.")
    sa = small_angle_hfov(p1, p2, est.solve_width, est.solve_height, theta_deg, est.pixel_aspect)
    est.diagnostics["small_angle_hfov_deg"] = sa
    if np.isfinite(sa) and est.hfov_deg > 0:
        dev = 100.0 * (sa - est.hfov_deg) / est.hfov_deg
        est.diagnostics["small_angle_error_pct"] = dev
        if abs(dev) > 1.0:
            est.notes.append(f"The small-angle shortcut (θ·W/d) would give {sa:.4g} deg, {dev:+.1f}% off: the exact "
                             "pinhole solution is used because the points are off-centre at this FOV.")
    _region_note(est)
    est.notes.append(SCALING_NOTE)
    return est


def _region_note(est: FovEstimate) -> None:
    if est.region:
        x0, y0, w, h = est.region
        est.notes.append(f"Active image region x={x0:g}..{x0 + w:g}, y={y0:g}..{y0 + h:g}: the HFOV is that of the "
                         f"{w:g}-px-wide inset, with the principal point at its centre. Apply it with the same region "
                         "(kin_params.image_region) or crop the video to the inset; applied to the full "
                         f"{est.width}-px frame it would overstate every angle by {est.width / w:.3g}x.")


# ---------------------------------------------------------------- methods
def estimate_separation(width: int, height: int, p1: Point, p2: Point, angle_deg: float, angle_sigma_deg: float = 0.0,
                        click_sigma_px: float = DEFAULT_CLICK_SIGMA_PX, pixel_aspect: float = 1.0,
                        frame: Optional[int] = None, region: Optional[Region] = None) -> FovEstimate:
    """Two clicked points with a KNOWN true angle between them (e.g. two stars, two surveyed towers).

    region=(x0, y0, w, h): active image region (inset) in full-frame px; the result is the HFOV of the inset."""
    if angle_deg <= 0:
        raise ValueError("the true angle must be positive")
    if angle_sigma_deg >= angle_deg:
        raise ValueError("the angle sigma must be smaller than the angle itself (a ±σ perturbation would cross zero)")
    w, h, (q1, q2), notes, reg = _inset(width, height, region, [p1, p2])
    params = {"p1.x": q1[0], "p1.y": q1[1], "p2.x": q2[0], "p2.y": q2[1], "angle_deg": float(angle_deg)}
    sig = {**_click_sigmas(("p1", "p2"), click_sigma_px), "angle_deg": float(angle_sigma_deg)}

    def solve(q):
        a, b = _pts(q, ("p1", "p2"))
        return solve_focal(w, h, [(a, b, math.radians(q["angle_deg"]))], pixel_aspect)

    def fn(q):
        return hfov_from_focal(w, solve(q)[0])
    hfov, sigma, contrib = _propagate(fn, params, sig)
    alt = solve(params)[2]
    est = FovEstimate(hfov, sigma, "separation", width, height, pixel_aspect,
                      inputs=dict(p1=list(map(float, p1)), p2=list(map(float, p2)), angle_deg=float(angle_deg),
                                  angle_sigma_deg=float(angle_sigma_deg), click_sigma_px=float(click_sigma_px),
                                  frame=frame, region=reg),
                      notes=notes, diagnostics={"sigma_contributions_deg": contrib, "focal_px": focal_px(w, hfov)},
                      region=reg)
    if angle_sigma_deg <= 0:
        est.notes.append("No uncertainty was given for the true angle; σ reflects click uncertainty only.")
    return _finish(est, q1, q2, angle_deg, alt)


def estimate_size(width: int, height: int, p1: Point, p2: Point, size_m: float, range_m: float,
                  size_sigma_m: float = 0.0, range_sigma_m: float = 0.0, click_sigma_px: float = DEFAULT_CLICK_SIGMA_PX,
                  pixel_aspect: float = 1.0, frame: Optional[int] = None,
                  region: Optional[Region] = None) -> FovEstimate:
    """Two points spanning an object of ASSUMED size L at ASSUMED range R: theta = 2 atan(L / 2R).

    Both inputs are assumptions, so the result inherits them: a wrong size or range gives a wrong FOV in the same
    proportion (to first order). In IR, blur and blooming inflate apparent extents, which biases this FOV LARGE.
    region=(x0, y0, w, h): active image region (inset) in full-frame px; the result is the HFOV of the inset.
    """
    if size_m <= 0 or range_m <= 0:
        raise ValueError("size and range must be positive")
    w, h, (q1, q2), notes, reg = _inset(width, height, region, [p1, p2])
    params = {"p1.x": q1[0], "p1.y": q1[1], "p2.x": q2[0], "p2.y": q2[1], "size_m": float(size_m),
              "range_m": float(range_m)}
    sig = {**_click_sigmas(("p1", "p2"), click_sigma_px), "size_m": float(size_sigma_m), "range_m": float(range_sigma_m)}
    if range_sigma_m >= range_m or size_sigma_m >= size_m:
        raise ValueError("sigma must be smaller than the value itself (a ±σ perturbation would cross zero)")

    def solve(q):
        a, b = _pts(q, ("p1", "p2"))
        th = 2.0 * math.atan(q["size_m"] / (2.0 * q["range_m"]))
        return solve_focal(w, h, [(a, b, th)], pixel_aspect)

    def fn(q):
        return hfov_from_focal(w, solve(q)[0])
    theta_deg = math.degrees(2.0 * math.atan(size_m / (2.0 * range_m)))
    hfov, sigma, contrib = _propagate(fn, params, sig)
    alt = solve(params)[2]
    est = FovEstimate(hfov, sigma, "size", width, height, pixel_aspect,
                      inputs=dict(p1=list(map(float, p1)), p2=list(map(float, p2)), size_m=float(size_m),
                                  range_m=float(range_m), size_sigma_m=float(size_sigma_m),
                                  range_sigma_m=float(range_sigma_m), click_sigma_px=float(click_sigma_px),
                                  frame=frame, region=reg),
                      notes=notes, diagnostics={"sigma_contributions_deg": contrib, "focal_px": focal_px(w, hfov),
                                                "angle_deg": theta_deg}, region=reg)
    est.notes.append("Size and range are ASSUMED: the FOV (and everything derived from it) scales with size/range. "
                     "If the span was measured on an IR blob, blur/blooming inflate it and bias this FOV large.")
    return _finish(est, q1, q2, theta_deg, alt)


def estimate_geo(width: int, height: int, camera: tuple[float, float, float],
                 landmarks: Sequence[tuple[Point, tuple[float, float, float]]], camera_sigma_m: float = 0.0,
                 landmark_sigma_m: float = 0.0, click_sigma_px: float = DEFAULT_CLICK_SIGMA_PX,
                 pixel_aspect: float = 1.0, frame: Optional[int] = None,
                 region: Optional[Region] = None) -> FovEstimate:
    """Camera position + >= 2 landmarks with known (lat, lon, alt_m), each clicked in the frame.

    landmarks: [((x_px, y_px), (lat, lon, alt_m)), ...]. True angles between landmark directions are computed in the
    local ENU frame at the camera (WGS-84). All landmark pairs are fitted together; with >= 3 landmarks the residual
    RMS is a consistency check (misidentified landmark, wrong camera position, lens distortion).
    camera_sigma_m / landmark_sigma_m apply to each coordinate axis (E, N, U) of the camera / every landmark.
    region=(x0, y0, w, h): active image region (inset) in full-frame px; the result is the HFOV of the inset.
    """
    if len(landmarks) < 2:
        raise ValueError("at least two landmarks are needed")
    full_pts = [lm[0] for lm in landmarks]
    w, h, pts, notes, reg = _inset(width, height, region, full_pts)
    names = [f"lm{i}" for i in range(len(landmarks))]
    params: dict = {"cam.e": 0.0, "cam.n": 0.0, "cam.u": 0.0}
    sig: dict = {"cam.e": camera_sigma_m, "cam.n": camera_sigma_m, "cam.u": camera_sigma_m}
    for n, p, (_, geo) in zip(names, pts, landmarks):
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
        return solve_focal(w, h, pairs, pixel_aspect)

    def fn(q):
        return hfov_from_focal(w, solve(q)[0])
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
                                  click_sigma_px=float(click_sigma_px), frame=frame, region=reg),
                      notes=notes, diagnostics=diag, region=reg)
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
                 pixel_aspect: float = 1.0, frame: Optional[int] = None,
                 region: Optional[Region] = None) -> FovEstimate:
    """Look a HUD zoom / FOV-mode label up in a user-supplied {label: hfov_deg} table.

    The table is the analyst's claim (from a manual, a published spec, or a calibration); the label is the provenance.
    Labels are matched case-insensitively. region=(x0, y0, w, h): the table value is taken to be the HFOV of that
    inset (a sensor spec quotes the FOV of the sensor image, not of the frame it was pasted into).
    """
    reg = normalise_region(width, height, region)
    norm = {str(k).strip().upper(): float(v) for k, v in table.items()}
    key = str(label).strip().upper()
    if key not in norm:
        raise KeyError(f"HUD label {label!r} is not in the table {sorted(table)}")
    hfov = norm[key]
    if hfov <= 0:
        raise ValueError(f"table entry for {label!r} must be a positive HFOV in degrees")
    est = FovEstimate(hfov, hfov * sigma_pct / 100.0, "hud", width, height, pixel_aspect,
                      inputs=dict(label=str(label), table={str(k): float(v) for k, v in table.items()},
                                  sigma_pct=float(sigma_pct), frame=frame, region=reg), region=reg)
    est.notes.append("The HUD label -> FOV table is user-supplied; the label is the only provenance. HUD zoom labels "
                     "are often digital zoom levels, not optical FOVs, and the same label can mean different FOVs on "
                     "different sensors.")
    if sigma_pct <= 0:
        est.notes.append("No uncertainty given for the table value.")
    _region_note(est)
    est.notes.append(SCALING_NOTE)
    return est


# ---------------------------------------------------------------- ground-plane resection (camera position unknown)
def _hartley_T(pts: np.ndarray) -> np.ndarray:
    """Similarity that moves 2-D points to centroid 0 and mean distance sqrt(2) (Hartley normalisation)."""
    c = pts.mean(axis=0)
    d = float(np.mean(np.linalg.norm(pts - c, axis=1)))
    s = math.sqrt(2.0) / d if d > 0 else 1.0
    return np.array([[s, 0.0, -s * c[0]], [0.0, s, -s * c[1]], [0.0, 0.0, 1.0]])


def fit_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Direct linear transform with Hartley normalisation: H (3x3, Frobenius norm 1, H[2,2] >= 0) such that
    dst ~ H @ (src, 1) for >= 4 correspondences (exact for 4, algebraic least squares for more)."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    n = len(src)
    if n < 4 or len(dst) != n:
        raise ValueError("a homography needs at least four point correspondences")
    Ts, Td = _hartley_T(src), _hartley_T(dst)
    s = (Ts @ np.column_stack([src, np.ones(n)]).T).T
    d = (Td @ np.column_stack([dst, np.ones(n)]).T).T
    A = np.zeros((2 * n, 9))
    for i in range(n):
        u, v, _ = s[i]
        x, y, _ = d[i]
        A[2 * i] = [0, 0, 0, -u, -v, -1, y * u, y * v, y]
        A[2 * i + 1] = [u, v, 1, 0, 0, 0, -x * u, -x * v, -x]
    _, sv, Vt = np.linalg.svd(A)
    # rank must be exactly 8: the 8th singular value is non-zero (with 4 points there are only 8; with more, the
    # 9th is the fit residual). Three collinear points drop the rank to 7 and leave a 2-D null space.
    if len(sv) < 8 or sv[7] <= 1e-9 * max(sv[0], 1e-300):
        raise ValueError("the points do not determine a homography (degenerate configuration: three or more "
                         "collinear points, or coincident points)")
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    H /= np.linalg.norm(H)
    if H[2, 2] < 0:
        H = -H
    return H


def _apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    q = (H @ np.column_stack([pts, np.ones(len(pts))]).T).T
    return q[:, :2] / q[:, 2:3]


def _plane_frame(enu: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Least-squares plane through ENU points: (centroid, e1, e2, n, rms_m). n points up; e1 is East projected
    into the plane, e2 = n x e1, so (e1, e2, n) is right-handed."""
    c = enu.mean(axis=0)
    _, sv, Vt = np.linalg.svd(enu - c)
    n = Vt[-1]
    if n[2] < 0:
        n = -n
    rms = math.sqrt(float(np.mean(((enu - c) @ n) ** 2)))
    e1 = np.array([1.0, 0.0, 0.0]) - n[0] * n
    if np.linalg.norm(e1) < 1e-6:          # a vertical plane (pathological): use North instead
        e1 = np.array([0.0, 1.0, 0.0]) - n[1] * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return c, e1, e2, n, rms


def _iac_coeffs(H: np.ndarray) -> tuple[float, float, float, float]:
    a1, b1 = H[0, 0] * H[0, 1] + H[1, 0] * H[1, 1], H[2, 0] * H[2, 1]
    a2, b2 = H[0, 0] ** 2 + H[1, 0] ** 2 - H[0, 1] ** 2 - H[1, 1] ** 2, H[2, 0] ** 2 - H[2, 1] ** 2
    return float(a1), float(b1), float(a2), float(b2)


def _focal_from_iac(H: np.ndarray) -> tuple[float, float, float]:
    """Focal length (px, normalised image coordinates) from a plane->image homography H = K [r1 r2 t].

    With K = diag(f, f, 1) the image of the absolute conic is omega = K^-T K^-1 = diag(1/f^2, 1/f^2, 1) and the
    orthonormality of r1, r2 gives two equations linear in s = 1/f^2:
        h1' omega h2 = 0        ->  a1 s + b1 = 0,  a1 = h11 h12 + h21 h22,           b1 = h31 h32
        h1' omega h1 = h2' omega h2  ->  a2 s + b2 = 0,  a2 = h11^2 + h21^2 - h12^2 - h22^2,  b2 = h31^2 - h32^2
    They are the real and imaginary parts of ONE complex equation, (h1 + i h2)' omega (h1 + i h2) = 0, i.e.
    (a2 + 2i a1) s + (b2 + 2i b1) = 0. Rotating the plane axes by phi multiplies both complex coefficients by
    exp(-2i phi) and so mixes the two real equations (with East along the plane's u axis and a camera looking north,
    the first equation is empty and all the information sits in the second). The least-squares s of the complex
    equation is therefore used: it does not depend on how the plane axes were chosen. For the diagnostic, each real
    equation is solved alone in the plane frame rotated so that both carry the same weight (arg(a2 + 2i a1) = 45 deg).
    Returns (s_ls, f1, f2), f1/f2 NaN when that equation has no positive solution. When the view has no perspective
    (nadir: h31 = h32 = 0) both b's vanish and s is undetermined: that is the physics of a fronto-parallel plane,
    not a numerical failure.
    """
    a1, b1, a2, b2 = _iac_coeffs(H)
    A, B = complex(a2, 2.0 * a1), complex(b2, 2.0 * b1)
    if abs(A) == 0:
        return 0.0, float("nan"), float("nan")
    s = -(A.conjugate() * B).real / abs(A) ** 2
    phi = (cmath.phase(A) - math.pi / 4.0) / 2.0
    c, sn = math.cos(phi), math.sin(phi)
    a1r, b1r, a2r, b2r = _iac_coeffs(H @ np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]]))

    def one(a, b):
        if a == 0 or b == 0 or -a / b <= 0:
            return float("nan")
        return math.sqrt(-a / b)
    return float(s), one(a1r, b1r), one(a2r, b2r)


def _decompose_homography(H: np.ndarray, f: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """H = lambda K [r1 r2 t] -> (R, t, C): rotation plane->camera, translation, camera centre in plane coordinates.
    The sign of lambda is chosen so the plane origin is in front of the camera (t_z > 0)."""
    Hp = np.diag([1.0 / f, 1.0 / f, 1.0]) @ H
    lam = 2.0 / (np.linalg.norm(Hp[:, 0]) + np.linalg.norm(Hp[:, 1]))
    r1, r2, t = lam * Hp[:, 0], lam * Hp[:, 1], lam * Hp[:, 2]
    if t[2] < 0:
        r1, r2, t = -r1, -r2, -t
    R = np.column_stack([r1, r2, np.cross(r1, r2)])
    U, _, Vt = np.linalg.svd(R)                     # nearest proper rotation
    if np.linalg.det(U @ Vt) < 0:
        U[:, -1] = -U[:, -1]
    R = U @ Vt
    return R, t, -R.T @ t


def _ground_solve(w: float, h: float, par: float, img: np.ndarray, enu: np.ndarray) -> dict:
    """One full plane-resection solve (used for the value and for every finite-difference perturbation).

    img: clicked points in inset px (N x 2); enu: ground points in a local ENU frame (N x 3, metres).
    Returns a dict with the homography, focal length, HFOV, camera centre (ENU), slant range, LOS, m/px and the
    reprojection residuals. f is clamped to [w/20, 200 w] when the IAC equations give no usable value
    (`focal_determined` False); the quantities that do not need f (m/px, depth ratio, residuals) are then still valid.
    """
    c, e1, e2, n, plane_rms = _plane_frame(enu)
    d = enu - c
    uv = np.column_stack([d @ e1, d @ e2])
    xy = np.column_stack([img[:, 0] - w / 2.0, (img[:, 1] - h / 2.0) / par])
    H = fit_homography(uv, xy)
    s, f1, f2 = _focal_from_iac(H)
    lo, hi = w * F_MIN_FACTOR, w * F_MAX_FACTOR
    determined = s > 0 and lo <= 1.0 / math.sqrt(s) <= hi
    f = 1.0 / math.sqrt(s) if s > 0 else hi
    f = min(max(f, lo), hi)
    R, t, C = _decompose_homography(H, f)
    # depth of every ground point along the optical axis, from the third row of H alone (no f needed): the
    # max/min ratio is the amount of perspective in the view, i.e. how well f is constrained
    depth = (H[2] @ np.column_stack([uv, np.ones(len(uv))]).T)
    depth = np.abs(depth)
    depth_ratio = float(depth.max() / depth.min()) if depth.min() > 0 else float("inf")
    # principal ray in plane coordinates, intersected with the plane (w = 0)
    axis = R[2]
    hit = axis[2] < 0 and C[2] > 0
    rng = -C[2] / axis[2] if hit else float("nan")
    aim_p = C + rng * axis if hit else np.full(3, np.nan)
    los_enu = axis[0] * e1 + axis[1] * e2 + axis[2] * n
    az, el, _ = az_el_range(los_enu)
    cam_enu = c + C[0] * e1 + C[1] * e2 + C[2] * n
    aim_enu = c + aim_p[0] * e1 + aim_p[1] * e2 if hit else np.full(3, np.nan)
    # metres per screen pixel at the image centre: Jacobian of the image->plane map at the principal point
    G = np.linalg.inv(H)
    den = G[2, 2]
    J = np.array([[G[0, 0] * den - G[0, 2] * G[2, 0], G[0, 1] * den - G[0, 2] * G[2, 1]],
                  [G[1, 0] * den - G[1, 2] * G[2, 0], G[1, 1] * den - G[1, 2] * G[2, 1]]]) / (den * den)
    J = J @ np.diag([1.0, 1.0 / par])              # per SCREEN pixel in y (normalised y is y/par)
    gsd = math.sqrt(abs(float(np.linalg.det(J))))
    # reprojection: plane -> image (px, screen units) and image -> plane (m)
    rep = _apply_h(H, uv)
    res_px = np.column_stack([rep[:, 0] - xy[:, 0], (rep[:, 1] - xy[:, 1]) * par])
    back = _apply_h(G, xy)
    res_m = back - uv
    return dict(H=H, f=f, hfov=hfov_from_focal(w, f), determined=determined, s=s, f1=f1, f2=f2,
                R=R, t=t, C=C, cam_enu=cam_enu, height=float(C[2]), slant_range=float(rng), aim_enu=aim_enu,
                azimuth=az, elevation=el, gsd=gsd, gsd_x=float(np.linalg.norm(J[:, 0])),
                gsd_y=float(np.linalg.norm(J[:, 1])), depth_ratio=depth_ratio, plane_rms=plane_rms,
                res_px=res_px, res_m=res_m, uv=uv, origin_c=c)


def estimate_ground(width: int, height: int, points: Sequence[tuple[Point, tuple[float, float, float]]],
                    point_sigma_m: float = 0.0, click_sigma_px: float = DEFAULT_CLICK_SIGMA_PX,
                    pixel_aspect: float = 1.0, region: Optional[Region] = None,
                    frame: Optional[int] = None) -> FovEstimate:
    """Look-down footage, camera position UNKNOWN, >= 4 ground features of known position clicked in one frame.

    points: [((x_px, y_px), (lat, lon, alt_m)), ...], at least 4 (5-6 spread over the frame is much better: four
    points fit a homography exactly and leave no consistency check). Altitudes may vary a little: a least-squares
    plane is fitted through the points (its RMS is reported; a few metres over a kilometre is harmless, a hillside
    is not).

    Geometry. Points are converted to a local ENU frame at their centroid (WGS-84) and expressed in 2-D plane
    coordinates (u, v). A pinhole camera with principal point at the (inset) image centre, zero skew and square
    pixels (a pixel aspect ratio is handled by scaling y) maps the plane to the image by the homography
        H = lambda K [r1 r2 t],   K = diag(f, f, 1),
    which is fitted by the DLT (Hartley-normalised, algebraic least squares for > 4 points). Since r1, r2 are
    orthonormal, the image of the absolute conic omega = diag(1/f^2, 1/f^2, 1) satisfies h1' omega h2 = 0 and
    h1' omega h1 = h2' omega h2: two equations linear in 1/f^2, solved in closed-form least squares (in their
    rotation-invariant complex form, see _focal_from_iac); the focal lengths each equation gives alone are compared
    as a diagnostic. With f known, K^-1 H yields r1, r2, t (scale from |r1| = |r2| = 1, sign so the plane is in
    front of the camera), r3 = r1 x r2, and the camera centre C = -R' t in plane coordinates -> ENU -> lat/lon/alt.
    The principal ray R' (0, 0, 1) intersected with the plane gives the slant range to the image centre and the
    look-down angle; the Jacobian of H^-1 at the principal point gives the metres per screen pixel there.

    Degenerate geometry. The perspective in the view is what determines f: a near-nadir (fronto-parallel) view maps
    the plane to the image by a similarity, which fixes the ratio f / height but not f itself. The two IAC equations
    then carry no information (their constant terms vanish), f comes out ill-conditioned or unbounded and a note
    says so. This is physics, not a bug. The metres-per-pixel at the image centre is still well determined in that
    case (it only needs H), so it is reported with its own sigma: it is what a ground-relative speed needs.

    Uncertainty: click sigma (each pixel coordinate) and point_sigma_m (each ENU axis of every ground point) are
    propagated by finite differences through the whole solve, for the HFOV, the metres-per-pixel and the slant
    range; the per-input HFOV contributions are kept in diagnostics.

    Diagnostics (all SI, ENU azimuths clockwise from north, elevation negative = looking down):
      camera_lat, camera_lon, camera_alt_m          recovered camera position (WGS-84)
      camera_height_above_plane_m                   camera height above the fitted ground plane
      slant_range_m, slant_range_sigma_m            range from the camera to the ground point at the image centre
      aimpoint_lat, aimpoint_lon                    that ground point
      azimuth_deg, elevation_deg                    direction of the optical axis (elevation < 0 = looking down)
      ground_sample_m_per_px, ground_sample_sigma_m_per_px   metres per screen pixel at the image centre
                                                    (geometric mean of the along-x and along-y values
                                                    ground_sample_x_m_per_px / ground_sample_y_m_per_px)
      focal_px, hfov_determined, iac_focal_px_eq1/eq2, iac_disagreement_pct, perspective_depth_ratio
      plane_fit_rms_m, reprojection_rms_px, reprojection_rms_m, point_residuals_px
    region=(x0, y0, w, h): active image region (inset) in full-frame px; the result is the HFOV of the inset.
    """
    if len(points) < 4:
        raise ValueError("at least four ground points are needed (five or six, spread over the frame, are better)")
    full_pts = [p for p, _ in points]
    w, h, pts, notes, reg = _inset(width, height, region, full_pts)
    par = pixel_aspect if pixel_aspect > 0 else 1.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]) < 1e-6:
                raise ValueError(f"ground points {i} and {j} were clicked at the same pixel")
    geos = [(float(g[0]), float(g[1]), float(g[2])) for _, g in points]
    for la, lo, _ in geos:
        if not (-90.0 <= la <= 90.0 and -180.0 <= lo <= 360.0):
            raise ValueError(f"latitude/longitude ({la}, {lo}) out of range")
    origin = (float(np.mean([g[0] for g in geos])), float(np.mean([g[1] for g in geos])),
              float(np.mean([g[2] for g in geos])))
    enu0 = np.array([enu_vector(origin, g) for g in geos])
    for i in range(len(geos)):
        for j in range(i + 1, len(geos)):
            if np.linalg.norm(enu0[i] - enu0[j]) < 0.01:
                raise ValueError(f"ground points {i} and {j} have the same position")
    names = [f"p{i}" for i in range(len(points))]
    params: dict = {}
    sig: dict = {}
    for nm, p in zip(names, pts):
        params[f"{nm}.x"], params[f"{nm}.y"] = p
        params[f"{nm}.e"] = params[f"{nm}.n"] = params[f"{nm}.u"] = 0.0
        sig[f"{nm}.e"] = sig[f"{nm}.n"] = sig[f"{nm}.u"] = float(point_sigma_m)
    sig.update(_click_sigmas(names, click_sigma_px))

    def solve(q):
        img = np.array([[q[f"{nm}.x"], q[f"{nm}.y"]] for nm in names])
        enu = enu0 + np.array([[q[f"{nm}.e"], q[f"{nm}.n"], q[f"{nm}.u"]] for nm in names])
        return _ground_solve(w, h, par, img, enu)

    try:
        base = solve(params)
    except np.linalg.LinAlgError:
        raise ValueError("the points do not determine a homography (degenerate configuration)") from None
    hfov, sigma, contrib = _propagate(lambda q: solve(q)["hfov"], params, sig)
    gsd, gsd_sigma, gsd_contrib = _propagate(lambda q: solve(q)["gsd"], params, sig)
    rng, rng_sigma, _ = _propagate(lambda q: solve(q)["slant_range"], params, sig)
    cam_geo = enu_to_geodetic(base["cam_enu"], origin)
    aim_geo = enu_to_geodetic(base["aim_enu"], origin) if np.isfinite(base["aim_enu"]).all() else (float("nan"),) * 3
    res_px = np.linalg.norm(base["res_px"], axis=1)
    rep_px = math.sqrt(float(np.mean(res_px ** 2)))
    rep_m = math.sqrt(float(np.mean(np.sum(base["res_m"] ** 2, axis=1))))
    f1, f2 = base["f1"], base["f2"]
    disagree = 100.0 * abs(f1 - f2) / base["f"] if np.isfinite(f1) and np.isfinite(f2) else float("nan")
    diag: dict = {
        "sigma_contributions_deg": contrib, "focal_px": base["f"], "hfov_determined": bool(base["determined"]),
        "iac_focal_px_eq1": f1, "iac_focal_px_eq2": f2, "iac_disagreement_pct": disagree,
        "perspective_depth_ratio": base["depth_ratio"],
        "camera_lat": cam_geo[0], "camera_lon": cam_geo[1], "camera_alt_m": cam_geo[2],
        "camera_enu_m": [float(v) for v in base["cam_enu"]], "enu_origin": dict(lat=origin[0], lon=origin[1],
                                                                                 alt_m=origin[2]),
        "camera_height_above_plane_m": base["height"], "slant_range_m": base["slant_range"],
        "slant_range_sigma_m": rng_sigma, "aimpoint_lat": aim_geo[0], "aimpoint_lon": aim_geo[1],
        "azimuth_deg": base["azimuth"], "elevation_deg": base["elevation"],
        "ground_sample_m_per_px": gsd, "ground_sample_sigma_m_per_px": gsd_sigma,
        "ground_sample_x_m_per_px": base["gsd_x"], "ground_sample_y_m_per_px": base["gsd_y"],
        "ground_sample_sigma_contributions": gsd_contrib,
        "plane_fit_rms_m": base["plane_rms"], "reprojection_rms_px": rep_px, "reprojection_rms_m": rep_m,
        "point_residuals_px": [float(v) for v in res_px],
        "points": [dict(px=list(map(float, p)), lat=g[0], lon=g[1], alt_m=g[2], plane_u_m=float(uv[0]),
                        plane_v_m=float(uv[1])) for p, g, uv in zip(full_pts, geos, base["uv"])],
        "homography_plane_to_image": [[float(v) for v in row] for row in base["H"]],
    }
    est = FovEstimate(hfov, sigma, "ground", width, height, pixel_aspect,
                      inputs=dict(points=[dict(px=list(map(float, p)), lat=g[0], lon=g[1], alt_m=g[2])
                                          for p, g in zip(full_pts, geos)],
                                  point_sigma_m=float(point_sigma_m), click_sigma_px=float(click_sigma_px),
                                  frame=frame, region=reg),
                      notes=notes, diagnostics=diag, region=reg)
    # ---- geometry quality
    for label_, arr in (("ground points", base["uv"]), ("clicked points", np.array(pts))):
        sv = np.linalg.svd(arr - arr.mean(axis=0), compute_uv=False)
        if sv[0] > 0 and sv[1] / sv[0] < 0.1:
            est.notes.append(f"The {label_} are nearly collinear (spread ratio {sv[1] / sv[0]:.2g}): the homography "
                             "is ill-conditioned. Choose points that span the frame in both directions.")
    if base["plane_rms"] > 0:
        est.notes.append(f"Ground plane fitted through {len(points)} points with RMS {base['plane_rms']:.2g} m "
                         "(altitude scatter about the plane).")
    n_pts = len(points)
    if n_pts == 4:
        est.notes.append("Only four points: the homography fits them exactly, so there is no consistency check. "
                         "A fifth point gives one.")
    else:
        floor = click_sigma_px if click_sigma_px > 0 else 1.0
        if rep_px > 3.0 * floor:
            est.notes.append(f"Consistency check FAILED: reprojection residual RMS {rep_px:.2g} px ({rep_m:.3g} m on "
                             f"the ground; per point {', '.join(f'{v:.2g}' for v in res_px)} px) vs click σ "
                             f"{click_sigma_px:g} px. Check the point identification/coordinates, the altitudes "
                             "(a plane is assumed) or lens distortion.")
        else:
            est.notes.append(f"Consistency check: {n_pts} points fitted with reprojection residual RMS {rep_px:.2g} px "
                             f"({rep_m:.3g} m on the ground).")
    # ---- is f determined?
    both = bool(np.isfinite(disagree))                   # both absolute-conic equations gave a focal length
    degenerate = (not base["determined"] or base["depth_ratio"] < 1.02 or (both and disagree > 10.0)
                  or (hfov > 0 and sigma / hfov > 0.5))
    if degenerate:
        if not base["determined"]:
            est.sigma_deg = max(est.sigma_deg, hfov)
        why = (f"the view has almost no perspective (depth ratio across the points {base['depth_ratio']:.3f}; "
               "a near-nadir view maps the plane to the image by a similarity, which fixes f/height but not f)"
               if base["depth_ratio"] < 1.02 else
               f"the two absolute-conic equations give f = {f1:.4g} and {f2:.4g} px ({disagree:.0f}% apart)"
               if np.isfinite(disagree) and disagree > 10.0 else
               "the absolute-conic equations give no usable focal length" if not base["determined"] else
               f"the click/position uncertainties alone move it by {100 * sigma / hfov:.0f}%")
        est.notes.append(f"FOCAL LENGTH POORLY DETERMINED: {why}. This is expected physics for a look-down view "
                         "with little perspective, not a failure: do not use this HFOV or the camera position/range. "
                         f"The metres-per-pixel at the image centre IS well determined: {gsd:.4g} ± {gsd_sigma:.2g} "
                         "m/px (use it for ground-relative speeds). More oblique footage, or a wider spread of "
                         "points in depth, would determine the FOV.")
    else:
        if base["depth_ratio"] < 1.1:
            est.notes.append(f"Little perspective in this view (depth ratio across the points "
                             f"{base['depth_ratio']:.3f}): the HFOV is only weakly determined "
                             f"(σ {est.sigma_pct:.0f}%), the metres-per-pixel much better "
                             f"({gsd:.4g} ± {gsd_sigma:.2g} m/px).")
        if both:
            est.notes.append(f"The two absolute-conic equations agree on f to {disagree:.2g}% "
                             f"({f1:.5g} vs {f2:.5g} px).")
        else:
            est.notes.append("One of the two absolute-conic equations has no positive solution on its own; f comes "
                             "from their least-squares combination. Treat the HFOV with caution.")
    if not np.isfinite(base["slant_range"]):
        est.notes.append("The image centre looks above the ground plane (no aim point / slant range).")
    est.notes.append(f"Recovered camera position ({cam_geo[0]:.6f}, {cam_geo[1]:.6f}, {cam_geo[2]:.1f} m), "
                     f"{base['height']:.1f} m above the ground plane, looking az {base['azimuth']:.1f} deg, "
                     f"elevation {base['elevation']:.1f} deg; slant range to the image centre "
                     f"{base['slant_range']:.1f} ± {rng_sigma:.2g} m; {gsd:.4g} ± {gsd_sigma:.2g} m/px there.")
    if point_sigma_m <= 0:
        est.notes.append("No position uncertainty given for the ground points: σ reflects click uncertainty only.")
    _region_note(est)
    est.notes.append(SCALING_NOTE)
    return est


# ---------------------------------------------------------------- active image region detection
def detect_inset(frames, n_samples: int = 30, threshold: float = 0.5) -> Optional[tuple[int, int, int, int]]:
    """Find the live sensor image inside a frame that has blurred or static side/top panels.

    Over up to n_samples frames spread through the clip, the mean |Laplacian| (sharpness) of every pixel is
    computed and, when the clip has temporal change at all, the mean |frame difference| of every pixel. Each map is
    reduced to a per-column and a per-row profile by the MEDIAN across the other axis (a mean would let a HUD line
    or a reticle drag a whole row down), and each profile is normalised by its median over the central 20% of the
    frame. A column/row is "live" when the smaller of the two normalised profiles exceeds `threshold`; the largest
    contiguous run of live columns and rows is the inset. Returns (x0, y0, w, h), or None when that is the whole
    frame (nothing to crop).
    """
    n = len(frames)
    if n == 0:
        return None
    idx = np.unique(np.linspace(0, n - 1, min(n_samples, n)).astype(int))
    stack = []
    for i in idx:
        f = np.asarray(frames[i])
        if f.ndim == 3:
            f = f.mean(axis=2)
        stack.append(f.astype(np.float32))
    stack = np.stack(stack)
    import cv2
    sharp = np.mean([np.abs(cv2.Laplacian(f, cv2.CV_32F)) for f in stack], axis=0)
    profiles = [(np.median(sharp, axis=0), np.median(sharp, axis=1))]
    if len(stack) >= 2:
        change = np.mean(np.abs(np.diff(stack, axis=0)), axis=0)
        ch_col, ch_row = np.median(change, axis=0), np.median(change, axis=1)
        if _central_median(ch_col) > 0.5 and _central_median(ch_row) > 0.5:     # a static scene gives no gate
            profiles.append((ch_col, ch_row))

    def live(axis_profiles):
        score = None
        for p in axis_profiles:
            med = _central_median(p)
            s = p / med if med > 0 else np.zeros_like(p)
            score = s if score is None else np.minimum(score, s)
        return score >= threshold
    x0, w = _longest_run(live([p[0] for p in profiles]))
    y0, h = _longest_run(live([p[1] for p in profiles]))
    H, W = stack.shape[1:]
    if (x0, y0, w, h) == (0, 0, W, H):
        return None
    return int(x0), int(y0), int(w), int(h)


def _central_median(p: np.ndarray) -> float:
    n = len(p)
    lo, hi = int(0.4 * n), max(int(0.6 * n), int(0.4 * n) + 1)
    return float(np.median(p[lo:hi]))


def _longest_run(mask: np.ndarray) -> tuple[int, int]:
    """(start, length) of the longest run of True; the whole axis if nothing is True."""
    n = len(mask)
    if not mask.any():
        return 0, n
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
    starts, ends = edges[::2], edges[1::2]
    k = int(np.argmax(ends - starts))
    return int(starts[k]), int(ends[k] - starts[k])


METHODS = {"separation": estimate_separation, "size": estimate_size, "geo": estimate_geo, "hud": estimate_hud,
           "ground": estimate_ground}


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
