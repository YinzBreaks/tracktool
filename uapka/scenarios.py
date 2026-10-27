"""Scenario conversion: angular kinematics -> metres and g, ONLY under explicit assumptions.

A single camera measures angles. Turning angles into speed needs a distance,
and the distance is not in the video. So this module never produces "the"
speed: it produces a table over a user-chosen set of assumed ranges, where every
row states the assumption that produced it.

Ways to get a range (each is an assumption, not a measurement):
  range     slant range entered directly (a list or a sweep)
  size      assumed true object size L; R = L / angular_size (angular size is
            measured, but in IR it is dominated by glare/blur, so R is biased low)
  altitude  assumed object altitude with camera altitude and LOS elevation angle
            (e.g. from the HUD): R = (h_obj - h_cam) / sin(elevation); flat earth

For each range R (held constant over the clip):
  transverse velocity  v = R * omega (+ assumed observer velocity across the LOS)
  transverse accel.    a = R * alpha
  lateral accel.       a_lat = R * alpha_lateral  (component turning the path)
Radial motion is unobservable; speeds are lower bounds relative to the camera.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

from .kinematics import Kinematics
from .provenance import Prov
from .units import G0, MS_TO_KT, fmt_length, parse_length_list


@dataclass
class ScenarioSpec:
    mode: str = "range"                 # range | size | altitude
    values: str = "1km, 2km, 5km, 10km, 20km, 50km"   # ranges, sizes or object altitudes (unit suffixes allowed)
    cam_altitude: str = "0m"            # altitude mode
    elevation_deg: float = 0.0          # altitude mode: LOS elevation (negative = looking down)
    observer_vx_ms: float = 0.0         # assumed observer velocity across LOS, image +x (right)
    observer_vy_ms: float = 0.0         # assumed observer velocity across LOS, image +y (up)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScenarioRow:
    range_m: float
    assumption: str
    v_median: float
    v_p95: float
    v_max: float
    v_max_sigma: float
    a_max_g: float
    a_max_sigma_g: float
    a_lat_max_g: float
    a_lat_max_sigma_g: float
    a_lat_p95_g: float
    noise_floor_g: float
    significant: bool          # is the peak lateral g > 3 sigma at that frame?


@dataclass
class ScenarioResult:
    spec: ScenarioSpec
    rows: list[ScenarioRow]
    notes: list[str] = field(default_factory=list)
    timeseries: dict = field(default_factory=dict)   # range_m -> dict of arrays (v, v_sigma, a_lat_g, ...)


def ranges_from_spec(spec: ScenarioSpec, kin: Optional[Kinematics] = None) -> tuple[list[tuple[float, str]], list[str]]:
    vals = parse_length_list(spec.values)
    notes: list[str] = []
    out: list[tuple[float, str]] = []
    if spec.mode == "range":
        out = [(r, f"ASSUMED slant range {fmt_length(r)}") for r in vals]
    elif spec.mode == "size":
        if kin is None or not kin.has_angles():
            raise ValueError("Size mode needs angular kinematics (HFOV) and a measured apparent size.")
        ang = np.radians(kin.get("angular_size"))
        ok = kin.valid & np.isfinite(ang) & (ang > 0)
        if not ok.any():
            raise ValueError("No apparent-size measurements available (enable centroid refinement).")
        theta = float(np.median(ang[ok]))
        notes.append(f"Median apparent angular size {np.degrees(theta):.4g} deg [MEASURED+FOV] over {int(ok.sum())} "
                     "frames. Apparent size includes optics blur, glare and blooming, so it overstates the true "
                     "size and the derived ranges are biased SHORT (speeds biased LOW).")
        out = [(L / theta, f"ASSUMED object size {fmt_length(L)} -> range {fmt_length(L / theta)}") for L in vals]
    elif spec.mode == "altitude":
        from .units import parse_length
        h_cam = parse_length(spec.cam_altitude)
        s = np.sin(np.radians(spec.elevation_deg))
        if abs(s) < 1e-3:
            raise ValueError("Elevation angle too close to 0 deg; altitude difference cannot fix the range.")
        for h in vals:
            R = (h - h_cam) / s
            if R <= 0:
                notes.append(f"Object altitude {fmt_length(h)} is inconsistent with the LOS direction "
                             f"(elevation {spec.elevation_deg:+.2f} deg from {fmt_length(h_cam)}); skipped.")
                continue
            out.append((R, f"ASSUMED object altitude {fmt_length(h)}, camera {fmt_length(h_cam)}, "
                           f"elevation {spec.elevation_deg:+.2f} deg -> range {fmt_length(R)}"))
        notes.append("Altitude mode uses flat-earth geometry and a constant elevation angle.")
    else:
        raise ValueError(f"Unknown scenario mode {spec.mode!r}")
    if spec.observer_vx_ms or spec.observer_vy_ms:
        notes.append(f"ASSUMED observer velocity across the LOS ({spec.observer_vx_ms:g}, {spec.observer_vy_ms:g}) m/s "
                     "(image right, up) is added to remove platform-motion parallax.")
    else:
        notes.append("Observer (camera platform) motion NOT removed: velocities are relative to the camera.")
    return out, notes


def compute_scenarios(kin: Kinematics, spec: ScenarioSpec) -> ScenarioResult:
    if not kin.has_angles():
        raise ValueError("Scenarios need angular kinematics: enter the horizontal FOV first.")
    rngs, notes = ranges_from_spec(spec, kin)
    D2R = np.pi / 180.0
    wx, wy = kin.get("omega_x") * D2R, kin.get("omega_y") * D2R
    sw = kin.get("omega_sigma") * D2R
    has_acc = "alpha" in kin.cols
    a = kin.get("alpha") * D2R if has_acc else np.full_like(wx, np.nan)
    sa = kin.get("alpha_sigma") * D2R if has_acc else np.full_like(wx, np.nan)
    an = kin.get("alpha_lateral") * D2R if has_acc else np.full_like(wx, np.nan)
    san = kin.get("alpha_lateral_sigma") * D2R if has_acc else np.full_like(wx, np.nan)
    valid = kin.trusted & np.isfinite(wx)
    if not valid.any():
        raise ValueError("No frames have trusted camera-motion estimates in their derivative window, so the object's "
                         "angular motion is not separable from camera motion. Scenario values would be meaningless. "
                         "(To proceed anyway, switch camera compensation off: that makes 'static camera' an explicit "
                         "assumption carried by every value.)")
    if not kin.params.use_camera:
        notes.append("Static camera ASSUMED (camera-motion compensation off): all on-screen motion is attributed to "
                     "the object.")
    rows: list[ScenarioRow] = []
    ts: dict = {}
    for R, label in rngs:
        vx = R * wx + spec.observer_vx_ms
        vy = R * wy + spec.observer_vy_ms
        v = np.hypot(vx, vy)
        s_v = R * sw
        ag, s_ag = R * a / G0, R * sa / G0
        lg, s_lg = R * an / G0, R * san / G0
        ok = valid & np.isfinite(v)
        oka = valid & np.isfinite(lg)
        nan = float("nan")
        if ok.any():
            iv = int(np.nanargmax(np.where(ok, v, np.nan)))
            v_med, v_p95, v_max, v_max_s = (float(np.median(v[ok])), float(np.percentile(v[ok], 95)),
                                            float(v[iv]), float(s_v[iv]))
        else:
            v_med = v_p95 = v_max = v_max_s = nan
        if oka.any():
            ia = int(np.nanargmax(np.where(oka, ag, np.nan)))
            il = int(np.nanargmax(np.where(oka, lg, np.nan)))
            a_max, a_max_s = float(ag[ia]), float(s_ag[ia])
            l_max, l_max_s = float(lg[il]), float(s_lg[il])
            l_p95 = float(np.percentile(lg[oka], 95))
            floor = float(np.median(s_lg[oka]))
            signif = bool(l_max > 3 * l_max_s)
        else:
            a_max = a_max_s = l_max = l_max_s = l_p95 = floor = nan
            signif = False
        rows.append(ScenarioRow(R, label, v_med, v_p95, v_max, v_max_s, a_max, a_max_s, l_max, l_max_s, l_p95,
                                floor, signif))
        ts[R] = dict(v=v, v_sigma=s_v, a_g=ag, a_g_sigma=s_ag, a_lat_g=lg, a_lat_g_sigma=s_lg)
    notes.append(f"Statistics use the {int(valid.sum())} frames whose derivative window contains only trusted "
                 "camera-motion estimates.")
    notes.append("All speeds/accelerations scale linearly with the assumed range: doubling the range doubles them.")
    notes.append("Peak values of noisy derivatives are biased upwards; compare each peak with its sigma and with "
                 "the noise floor (median sigma).")
    return ScenarioResult(spec=spec, rows=rows, notes=notes, timeseries=ts)


def rows_table(res: ScenarioResult) -> tuple[list[str], list[list[str]]]:
    tag = f" [{Prov.ASSUMED.value}]"
    header = ["range_km" + tag, "assumption",
              "v_median_m_s" + tag, "v_p95_m_s" + tag, "v_max_m_s" + tag, "v_max_sigma_m_s" + tag,
              "v_max_kt" + tag, "a_total_max_g" + tag, "a_total_max_sigma_g" + tag,
              "a_lateral_max_g" + tag, "a_lateral_max_sigma_g" + tag, "a_lateral_p95_g" + tag,
              "noise_floor_g" + tag, "lateral_peak_above_3sigma"]
    out = []
    for r in res.rows:
        out.append([f"{r.range_m / 1000:.4g}", r.assumption, f"{r.v_median:.4g}", f"{r.v_p95:.4g}", f"{r.v_max:.4g}",
                    f"{r.v_max_sigma:.3g}", f"{r.v_max * MS_TO_KT:.4g}", f"{r.a_max_g:.4g}", f"{r.a_max_sigma_g:.3g}",
                    f"{r.a_lat_max_g:.4g}", f"{r.a_lat_max_sigma_g:.3g}", f"{r.a_lat_p95_g:.4g}",
                    f"{r.noise_floor_g:.3g}", "yes" if r.significant else "NO (within noise)"])
    return header, out
