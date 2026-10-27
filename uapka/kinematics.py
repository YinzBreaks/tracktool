"""Angular kinematics: the quantities the video actually measures.

Pipeline per frame k:
  1. object pixel position p_k (tracker), camera transform A_k (frame k -> reference frame 0)
  2. line-of-sight (LOS) angle in reference-camera coordinates:
        theta = camera pointing (accumulated pan/tilt, linear in px/f)  +  in-frame offset (exact atan)
     x is right, y is up. Roll and zoom are taken from A_k.
  3. derivatives by local weighted polynomial regression on the real frame
     times: robust to gaps (lost/duplicate/excluded frames) and variable frame
     rate, and gives an analytic 1-sigma for every derivative from the position
     uncertainty (scaled up by the fit residual when the model does not fit).

Provenance: pixel quantities are MEASURED; angles need the field of view, so
they are MEASURED+FOV. Nothing here depends on range.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import factorial
from typing import Optional

import numpy as np

from .camera import CameraTrack
from .provenance import Prov, col
from .tracker import Track, segment_ids


@dataclass
class KinParams:
    hfov_deg: float = 0.0          # horizontal field of view at the reference frame (0 = unknown)
    pixel_aspect: float = 1.0      # pixel width / pixel height
    window_frames: int = 15        # local-fit window in new-content frames (odd); larger = less noise, less time resolution
    poly_order: int = 2            # 2: velocity + acceleration; 3: less bias on sharp turns, more noise
    timing: str = "nominal"        # "nominal" (frame/fps) | "container" (decoder timestamps)
    start_frame: int = 0
    end_frame: int = -1            # inclusive; -1 = last
    use_camera: bool = True
    low_conf_threshold: float = 0.3


def local_poly(t: np.ndarray, y: np.ndarray, sigma: np.ndarray, valid: np.ndarray, half_window: float,
               order: int, min_side: int = 1, segment: Optional[np.ndarray] = None) -> dict[str, np.ndarray]:
    """Local weighted polynomial fit centred on every valid sample.

    Returns value, d1, d2 and their 1-sigma, plus number of points used. Model:
    y(t_i + tau) = b0 + b1 tau + b2 tau^2/2 [+ b3 tau^3/6].
    Samples with fewer than `min_side` neighbours on either side (clip edges,
    gaps) get no derivative: one-sided fits extrapolate and produce spurious
    "manoeuvres" exactly where the data ends. If `segment` is given, fits never
    use samples from another track segment.
    """
    n = len(t)
    out = {k: np.full(n, np.nan) for k in ("val", "d1", "d2", "s_val", "s_d1", "s_d2", "chi2r")}
    out["n_used"] = np.zeros(n, int)
    idx_valid = np.flatnonzero(valid)
    tv, yv, sv = t[idx_valid], y[idx_valid], np.maximum(sigma[idx_valid], 1e-12)
    segv = None if segment is None else np.asarray(segment)[idx_valid]
    m = order + 1
    for j, i in enumerate(idx_valid):
        lo = np.searchsorted(tv, t[i] - half_window - 1e-9)
        hi = np.searchsorted(tv, t[i] + half_window + 1e-9)
        if segv is not None:
            same = np.flatnonzero(segv[lo:hi] == segv[j])
            lo, hi = lo + same[0], lo + same[-1] + 1
        if hi - lo < m + 1 or (j - lo) < min_side or (hi - 1 - j) < min_side:
            continue
        tau = tv[lo:hi] - t[i]
        X = np.stack([tau**p / factorial(p) for p in range(m)], axis=1)
        w = 1.0 / sv[lo:hi] ** 2
        XtW = X.T * w
        try:
            cov = np.linalg.inv(XtW @ X)
        except np.linalg.LinAlgError:
            continue
        beta = cov @ (XtW @ yv[lo:hi])
        r = yv[lo:hi] - X @ beta
        dof = (hi - lo) - m
        chi2r = float((w * r**2).sum() / dof) if dof > 0 else 1.0
        cov = cov * max(1.0, chi2r)
        out["val"][i], out["s_val"][i] = beta[0], np.sqrt(cov[0, 0])
        out["d1"][i], out["s_d1"][i] = beta[1], np.sqrt(cov[1, 1])
        if m >= 3:
            out["d2"][i], out["s_d2"][i] = beta[2], np.sqrt(cov[2, 2])
        out["chi2r"][i] = chi2r
        out["n_used"][i] = hi - lo
    return out


def random_walk_sigma(t: np.ndarray, step_sigma: np.ndarray, valid: np.ndarray,
                      half_window: float) -> tuple[np.ndarray, np.ndarray]:
    """1-sigma of the fitted 1st/2nd derivative caused by chained camera-step errors.

    Camera transforms are chained, so a per-step error e_j enters every later
    position: position error is a random walk, i.e. WHITE NOISE IN VELOCITY with
    sigma_v = sigma_step / dt. Fitting over N samples then gives
    sigma(d1) ~ sigma_v * sqrt(1.2 / N) and sigma(d2) ~ sigma_v * sqrt(12 / N^3) / dt.
    Treating these errors as white position noise would understate both by ~sqrt(N).
    """
    n = len(t)
    s1, s2 = np.full(n, np.nan), np.full(n, np.nan)
    if n < 2:
        return s1, s2
    dt = np.diff(t, prepend=t[0] - (t[1] - t[0]))
    var = np.nan_to_num(step_sigma, nan=3.0) ** 2
    csum = np.concatenate([[0.0], np.cumsum(var)])
    for i in np.flatnonzero(valid):
        lo = np.searchsorted(t, t[i] - half_window - 1e-9)
        hi = np.searchsorted(t, t[i] + half_window + 1e-9)
        N = hi - lo
        if N < 3:
            continue
        dtm = float(np.mean(dt[lo:hi]))
        sv = np.sqrt((csum[hi] - csum[lo]) / N) / dtm
        s1[i] = sv * np.sqrt(1.2 / N)
        s2[i] = sv * np.sqrt(12.0 / N**3) / dtm
    return s1, s2


def detect_turns(t: np.ndarray, x: np.ndarray, y: np.ndarray, vx: np.ndarray, vy: np.ndarray, svx: np.ndarray,
                 svy: np.ndarray, valid: np.ndarray, seg: np.ndarray, min_angle_deg: float = 45.0,
                 span_s: float = 0.5) -> list[dict]:
    """Heading changes of the path: compare the direction of motion span_s before and after each sample.

    Only counted where the speed on both sides is > 3 sigma (a direction is meaningless for a near-stationary
    object) and both sides lie in the same continuous track segment. Peaks above min_angle_deg, at least
    2*span_s apart, are returned with the turn angle (0-180 deg).
    """
    idx = np.flatnonzero(valid & np.isfinite(vx) & np.isfinite(vy))
    if len(idx) < 3:
        return []
    speed = np.hypot(vx, vy)
    sig = np.hypot(svx, svy)
    ang = np.full(len(t), np.nan)
    for i in idx:
        a = idx[(t[idx] <= t[i] - span_s) & (seg[idx] == seg[i])]
        b = idx[(t[idx] >= t[i] + span_s) & (seg[idx] == seg[i])]
        if not len(a) or not len(b):
            continue
        j, k = a[-1], b[0]
        if speed[j] < 3 * sig[j] or speed[k] < 3 * sig[k]:
            continue
        h1, h2 = np.arctan2(vy[j], vx[j]), np.arctan2(vy[k], vx[k])
        ang[i] = abs(np.degrees((h2 - h1 + np.pi) % (2 * np.pi) - np.pi))
    # A sharp corner gives a plateau of equal values (every window straddling it sees the full turn): report
    # the centre of the samples within 2 deg of each run's maximum.
    out = []
    above = np.isfinite(ang) & (ang >= min_angle_deg)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], above.astype(int), [0]])))
    for a, b in zip(edges[::2], edges[1::2]):
        run = np.arange(a, b)
        run = run[np.isfinite(ang[run])]
        top = run[ang[run] >= ang[run].max() - 2.0]
        i = int(top[len(top) // 2])
        out.append(dict(t=float(t[i]), index=i, x=float(x[i]), y=float(y[i]), angle_deg=float(ang[run].max())))
    out.sort(key=lambda o: -o["angle_deg"])
    kept = []
    for o in out:
        if all(abs(o["t"] - k["t"]) >= 2 * span_s for k in kept):
            kept.append(o)
    return sorted(kept, key=lambda o: o["t"])


def _mag(ax, ay, sx, sy):
    m = np.hypot(ax, ay)
    den = np.maximum(m, np.hypot(sx, sy))
    s = np.hypot(ax * sx, ay * sy) / np.where(den > 0, den, 1.0)
    s = np.maximum(s, np.hypot(sx, sy) / np.sqrt(2) * (m < np.hypot(sx, sy)))
    return m, s


@dataclass
class Kinematics:
    params: KinParams
    frame: np.ndarray
    t: np.ndarray
    valid: np.ndarray
    flags: list
    cols: dict = field(default_factory=dict)   # name -> (Prov, unit, array)
    warnings: list = field(default_factory=list)
    f0_px: float = float("nan")
    trusted: Optional[np.ndarray] = None   # valid AND no low-confidence camera step inside the derivative window
    turns: dict = field(default_factory=dict)   # "screen" / "compensated" -> list of turn dicts
    name: str = ""
    role: str = "target"

    def __post_init__(self):
        if self.trusted is None:
            self.trusted = self.valid.copy()

    def get(self, name: str) -> np.ndarray:
        return self.cols[name][2]

    def table(self) -> tuple[list[str], list[list]]:
        """Header (with provenance tags) and rows for CSV export."""
        names = list(self.cols)
        header = ["frame", col("t_s", Prov.MEASURED), "used_in_fit", "in_summaries", "flags"] + [
            col(f"{nm}_{self.cols[nm][1]}" if self.cols[nm][1] else nm, self.cols[nm][0]) for nm in names]
        rows = []
        for i in range(len(self.frame)):
            row = [int(self.frame[i]), f"{self.t[i]:.6f}", int(self.valid[i]), int(self.trusted[i]),
                   ";".join(self.flags[i])]
            for nm in names:
                v = self.cols[nm][2][i]
                row.append("" if (isinstance(v, float) or isinstance(v, np.floating)) and not np.isfinite(v)
                           else (f"{v:.6g}" if isinstance(v, (float, np.floating)) else v))
            rows.append(row)
        return header, rows

    def has_angles(self) -> bool:
        return np.isfinite(self.f0_px)

    def summary(self) -> dict:
        """Headline measured values (MEASURED or MEASURED+FOV only)."""
        s: dict = {}
        v = self.trusted
        if self.has_angles():
            w, sw = self.get("omega"), self.get("omega_sigma")
            if "alpha" in self.cols:        # absent for poly_order 1 (velocity only)
                a, sa = self.get("alpha"), self.get("alpha_sigma")
            else:
                a = sa = np.full_like(w, np.nan)
            ok = v & np.isfinite(w)
            if ok.any():
                i = np.nanargmax(np.where(ok, w, np.nan))
                s["omega_median_deg_s"] = float(np.nanmedian(w[ok]))
                s["omega_p95_deg_s"] = float(np.nanpercentile(w[ok], 95))
                s["omega_max_deg_s"] = (float(w[i]), float(sw[i]), int(self.frame[i]))
            oka = v & np.isfinite(a)
            if oka.any():
                i = np.nanargmax(np.where(oka, a, np.nan))
                s["alpha_median_deg_s2"] = float(np.nanmedian(a[oka]))
                s["alpha_max_deg_s2"] = (float(a[i]), float(sa[i]), int(self.frame[i]))
                s["alpha_noise_floor_deg_s2"] = float(np.nanmedian(sa[oka]))
            vv = self.valid
            s["total_angular_travel_deg"] = float(np.nansum(np.hypot(
                np.diff(self.get("los_x")[vv]), np.diff(self.get("los_y")[vv]))))
        vv = self.valid
        si = self.get("img_speed")
        if np.isfinite(si[vv]).any():
            s["screen_speed_median_px_s"] = float(np.nanmedian(si[vv]))
            i = int(np.nanargmax(np.where(vv, si, np.nan)))
            s["screen_speed_max_px_s"] = (float(si[i]), float(self.get("img_speed_sigma")[i]), int(self.frame[i]))
        if "screen_omega" in self.cols and np.isfinite(self.get("screen_omega")[vv]).any():
            w = self.get("screen_omega")
            i = int(np.nanargmax(np.where(vv, w, np.nan)))
            s["screen_omega_median_deg_s"] = float(np.nanmedian(w[vv]))
            s["screen_omega_max_deg_s"] = (float(w[i]), float(self.get("screen_omega_sigma")[i]), int(self.frame[i]))
        s["n_frames_used"] = int(self.valid.sum())
        s["n_frames_trusted"] = int((v & np.isfinite(self.get("omega"))).sum()) if self.has_angles() else int(v.sum())
        tv = self.valid
        s["duration_used_s"] = float(self.t[tv][-1] - self.t[tv][0]) if tv.sum() > 1 else 0.0
        return s


def compute_kinematics(track: Track, camera: Optional[CameraTrack], frame_times: np.ndarray, duplicate: np.ndarray,
                       shape: tuple[int, int], params: KinParams, frame_offset: int = 0,
                       blank: Optional[np.ndarray] = None) -> Kinematics:
    """Frame numbers (start/end, the 'frame' column) refer to the video file; the cache may hold a segment."""
    H, W = shape[:2]
    n = track.n
    frames = frame_offset + np.arange(n)
    end = frames[-1] if params.end_frame < 0 else params.end_frame
    in_range = (frames >= params.start_frame) & (frames <= end)
    blank = np.zeros(n, bool) if blank is None else np.asarray(blank, bool)
    duplicate = np.asarray(duplicate, bool) | blank
    t = np.asarray(frame_times, float)
    warnings: list[str] = []

    usable = track.usable() & in_range & ~duplicate
    flags = [[] for _ in range(n)]
    for k in range(n):
        if blank[k]:
            flags[k].append("BLANK_FRAME")
        elif duplicate[k]:
            flags[k].append("DUPLICATE_FRAME")
        if track.status[k] not in ("OK", "NONE", "DUP", "BLANK"):
            flags[k].append(f"TRACK_{track.status[k]}")

    use_cam = camera is not None and params.use_camera
    if use_cam:
        A = camera.A
        conf = camera.confidence()
        cam_sig = camera.sigma_px()
        for k in range(n):
            if k > 0 and conf[k] < params.low_conf_threshold:
                flags[k].append("LOW_CAMERA_CONFIDENCE")
            for fl in camera.pairs[k].flags:
                if fl in ("NO_ESTIMATE_ASSUMED_STATIC", "FALLBACK_PHASECORR", "STATIC_CLUSTER_PRESENT",
                          "CHOSE_MOVING_OVER_STATIC_CLUSTER", "GOP_ANCHOR_INTERPOLATED", "AFTER_LAST_KEYFRAME"):
                    flags[k].append(f"CAM_{fl}")
    else:
        A = np.repeat(np.eye(3)[None], n, axis=0)
        conf = np.full(n, np.nan)
        cam_sig = np.zeros(n)
        warnings.append("Camera motion NOT compensated: static camera ASSUMED. All on-screen motion is attributed "
                        "to the object; every angular value below inherits this assumption.")

    if usable.sum() < max(params.poly_order + 2, 4):
        warnings.append("Too few usable tracked frames for derivatives.")

    c = np.array([W / 2.0, H / 2.0])
    par = params.pixel_aspect if params.pixel_aspect > 0 else 1.0
    scale = np.sqrt(np.abs(np.linalg.det(A[:, :2, :2])))      # <1 when zoomed in vs reference
    # Pixel-domain positions
    px, py = track.x.copy(), track.y.copy()
    ref = np.einsum("nij,nj->ni", A[:, :2, :2], np.stack([px, py], 1)) + A[:, :2, 2]
    cam_pt = np.einsum("nij,j->ni", A[:, :2, :2], c) + A[:, :2, 2] - c    # boresight displacement, ref px
    sig_px = np.nan_to_num(track.sigma, nan=1.0) * scale       # white (per-frame) position noise

    cols: dict = {}
    cols["img_x"] = (Prov.MEASURED, "px", px)
    cols["img_y"] = (Prov.MEASURED, "px", py)
    cols["track_sigma"] = (Prov.MEASURED, "px", track.sigma)
    cols["track_score"] = (Prov.MEASURED, "", track.score)
    cols["apparent_size"] = (Prov.MEASURED, "px", track.size_px)
    if use_cam:
        cols["cam_confidence"] = (Prov.MEASURED, "", conf)
        cols["cam_step_sigma"] = (Prov.MEASURED, "px", cam_sig)
        cols["cam_zoom_factor"] = (Prov.MEASURED, "x", 1.0 / scale)
        cols["cam_roll"] = (Prov.MEASURED, "deg", camera.roll_deg())
        cols["cam_pan_x"] = (Prov.MEASURED, "refpx", cam_pt[:, 0])
        cols["cam_pan_y"] = (Prov.MEASURED, "refpx", cam_pt[:, 1])
    cols["stab_x"] = (Prov.MEASURED, "refpx", ref[:, 0])
    cols["stab_y"] = (Prov.MEASURED, "refpx", ref[:, 1])

    half = (max(params.window_frames, 3) - 1) / 2.0
    # The window counts NEW-CONTENT frames: with duplicates removed (e.g. 10 fps content in a 30 fps file) the
    # sample spacing is that of the usable frames, not the container frame rate.
    tu = t[usable]
    dt_med = float(np.median(np.diff(tu))) if len(tu) > 2 else (float(np.median(np.diff(t))) if n > 1 else 1.0)
    hw = half * dt_med
    order = int(np.clip(params.poly_order, 1, 3))
    ms = max(1, int(half) // 2)
    seg = segment_ids(track.status)
    if seg.max() > 0:
        warnings.append(f"Track has {seg.max() + 1} continuous segments (breaks at re-acquisitions or lost spans); "
                        "derivatives are never fitted across a break.")

    # Raw on-screen motion (no camera compensation) - purely measured
    sig_raw = np.nan_to_num(track.sigma, nan=1.0)
    fx = local_poly(t, px, sig_raw, usable, hw, order, ms, seg)
    fy = local_poly(t, py, sig_raw, usable, hw, order, ms, seg)
    v_img, sv_img = _mag(fx["d1"], fy["d1"], fx["s_d1"], fy["s_d1"])
    cols["img_speed"] = (Prov.MEASURED, "px_s", v_img)
    cols["img_speed_sigma"] = (Prov.MEASURED, "px_s", sv_img)
    # Camera-compensated motion in reference pixels - measured (depends on registration only)
    gx = local_poly(t, ref[:, 0], sig_px, usable, hw, order, ms, seg)
    gy = local_poly(t, ref[:, 1], sig_px, usable, hw, order, ms, seg)
    rw1, rw2 = random_walk_sigma(t, cam_sig * scale, usable, hw) if use_cam else (np.zeros(n), np.zeros(n))
    for g in (gx, gy):
        g["s_d1"] = np.hypot(g["s_d1"], rw1)
        g["s_d2"] = np.hypot(g["s_d2"], rw2)
    v_stab, sv_stab = _mag(gx["d1"], gy["d1"], gx["s_d1"], gy["s_d1"])
    cols["stab_speed"] = (Prov.MEASURED, "refpx_s", v_stab)
    cols["stab_speed_sigma"] = (Prov.MEASURED, "refpx_s", sv_stab)

    f0 = float("nan")
    if params.hfov_deg and params.hfov_deg > 0:
        f0 = (W / 2.0) / np.tan(np.radians(params.hfov_deg) / 2.0)
        # camera pointing: accumulated pan in angle (linear), x right / y up
        th_cx = cam_pt[:, 0] / f0
        th_cy = -(cam_pt[:, 1] / par) / f0
        # in-frame offset rotated/scaled into reference orientation, exact atan
        d = np.einsum("nij,nj->ni", A[:, :2, :2], np.stack([px - c[0], py - c[1]], 1))
        dxs, dys = d[:, 0], -d[:, 1] / par
        r = np.hypot(dxs, dys)
        k_off = np.where(r > 0, np.arctan(r / f0) / np.where(r > 0, r, 1), 1.0 / f0)
        th_x = th_cx + dxs * k_off
        th_y = th_cy + dys * k_off
        sig_th = sig_px / f0
        ax = local_poly(t, th_x, sig_th, usable, hw, order, ms, seg)
        ay = local_poly(t, th_y, sig_th, usable, hw, order, ms, seg)
        for g in (ax, ay):
            g["s_d1"] = np.hypot(g["s_d1"], rw1 / f0)
            g["s_d2"] = np.hypot(g["s_d2"], rw2 / f0)
        D = np.degrees
        cols["los_x"] = (Prov.MEASURED_FOV, "deg", D(th_x))
        cols["los_y"] = (Prov.MEASURED_FOV, "deg", D(th_y))
        cols["omega_x"] = (Prov.MEASURED_FOV, "deg_s", D(ax["d1"]))
        cols["omega_y"] = (Prov.MEASURED_FOV, "deg_s", D(ay["d1"]))
        w_m, w_s = _mag(ax["d1"], ay["d1"], ax["s_d1"], ay["s_d1"])
        cols["omega"] = (Prov.MEASURED_FOV, "deg_s", D(w_m))
        cols["omega_sigma"] = (Prov.MEASURED_FOV, "deg_s", D(w_s))
        if order >= 2:
            cols["alpha_x"] = (Prov.MEASURED_FOV, "deg_s2", D(ax["d2"]))
            cols["alpha_y"] = (Prov.MEASURED_FOV, "deg_s2", D(ay["d2"]))
            a_m, a_s = _mag(ax["d2"], ay["d2"], ax["s_d2"], ay["s_d2"])
            cols["alpha"] = (Prov.MEASURED_FOV, "deg_s2", D(a_m))
            cols["alpha_sigma"] = (Prov.MEASURED_FOV, "deg_s2", D(a_s))
            # path-normal (lateral) component of angular acceleration: turns the LOS track
            wm = np.where(w_m > 0, w_m, np.nan)
            a_n = np.abs(ax["d1"] * ay["d2"] - ay["d1"] * ax["d2"]) / wm
            s_n = np.hypot(ax["s_d2"], ay["s_d2"]) / np.sqrt(2) + a_m * w_s / np.maximum(wm, w_s)
            cols["alpha_lateral"] = (Prov.MEASURED_FOV, "deg_s2", D(a_n))
            cols["alpha_lateral_sigma"] = (Prov.MEASURED_FOV, "deg_s2", D(s_n))
        # On-screen angular rate: image motion only, NOT corrected for camera motion. Equals the object's
        # LOS rate only if the camera was not moving. Kept separate so the two are never confused.
        cols["screen_omega"] = (Prov.MEASURED_FOV, "deg_s", D(v_img / f0))
        cols["screen_omega_sigma"] = (Prov.MEASURED_FOV, "deg_s", D(sv_img / f0))
        if order >= 2:
            sa_m, sa_s = _mag(fx["d2"], fy["d2"], fx["s_d2"], fy["s_d2"])
            cols["screen_alpha"] = (Prov.MEASURED_FOV, "deg_s2", D(sa_m / f0))
            cols["screen_alpha_sigma"] = (Prov.MEASURED_FOV, "deg_s2", D(sa_s / f0))
        size_ref = track.size_px * scale
        cols["angular_size"] = (Prov.MEASURED_FOV, "deg", D(size_ref / f0))
        if use_cam:
            cx_ = local_poly(t, th_cx, cam_sig / f0 + 1e-9, in_range, hw, order, ms)
            cy_ = local_poly(t, th_cy, cam_sig / f0 + 1e-9, in_range, hw, order, ms)
            cols["cam_slew_rate"] = (Prov.MEASURED_FOV, "deg_s", D(np.hypot(cx_["d1"], cy_["d1"])))
        cols["fit_chi2r"] = (Prov.MEASURED, "", np.fmax(ax["chi2r"], ay["chi2r"]))
    else:
        warnings.append("No horizontal FOV supplied: angular rates (deg/s) cannot be computed; only pixel "
                        "quantities are reported. Enter the HFOV of the reference frame.")
        cols["fit_chi2r"] = (Prov.MEASURED, "", np.fmax(gx["chi2r"], gy["chi2r"]))

    trusted = usable & np.isfinite(px)
    if not use_cam:   # everything "camera-compensated" is really on-screen motion under a static-camera assumption
        for name in ("stab_x", "stab_y", "stab_speed", "stab_speed_sigma", "los_x", "los_y", "omega_x", "omega_y",
                     "omega", "omega_sigma", "alpha_x", "alpha_y", "alpha", "alpha_sigma", "alpha_lateral",
                     "alpha_lateral_sigma"):
            if name in cols:
                pv = Prov.STATIC_CAM_PX if name.startswith("stab_") else Prov.STATIC_CAM
                cols[name] = (pv, cols[name][1], cols[name][2])
    if use_cam:
        low = usable & (conf < params.low_conf_threshold)
        if low.any():
            warnings.append(f"{int(low.sum())} of {int(usable.sum())} usable frames have camera-motion confidence "
                            f"< {params.low_conf_threshold}; their angular values are unreliable.")
        # A derivative is only as good as the worst camera step inside its fit window.
        bad = np.convolve((conf < params.low_conf_threshold).astype(float), np.ones(2 * int(half) + 1), "same") > 0
        for k in np.flatnonzero(bad & usable):
            if "LOW_CAMERA_CONFIDENCE" not in flags[k]:
                flags[k].append("LOW_CAMERA_CONFIDENCE_IN_WINDOW")
        trusted &= ~bad
        if (usable & bad).any():
            warnings.append(f"{int((usable & bad).sum())} frames have a low-confidence camera step inside their "
                            "derivative window; they are kept in the CSV (flagged) but excluded from peak/median "
                            "summaries and scenario tables.")
    if duplicate[in_range].any():
        warnings.append(f"{int((duplicate & ~blank)[in_range].sum())} duplicate and {int(blank[in_range].sum())} "
                        "blank frames excluded from fits.")
    if abs(par - 1.0) > 1e-6:
        warnings.append(f"Non-square pixels (aspect {par:g}) applied to angles; camera roll estimation assumes "
                        "square pixels.")
    turns = {"screen": detect_turns(t, px, py, fx["d1"], fy["d1"], fx["s_d1"], fy["s_d1"], usable, seg),
             "compensated": detect_turns(t, ref[:, 0], ref[:, 1], gx["d1"], gy["d1"], gx["s_d1"], gy["s_d1"],
                                         trusted, seg) if use_cam else []}
    if not use_cam:   # under the static-camera assumption the on-screen turns ARE the object's turns
        turns["static_assumed"] = turns["screen"]
    return Kinematics(params=params, frame=frames, t=t, valid=usable & np.isfinite(px), flags=flags, cols=cols,
                      warnings=warnings, f0_px=f0, trusted=trusted, turns=turns, name=track.name, role=track.role)
