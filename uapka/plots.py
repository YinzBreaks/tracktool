"""Figures. Built on matplotlib.figure.Figure (no pyplot) so they are safe to
create from worker threads and embed in the Qt GUI. Every title carries the
provenance label of what it shows."""

from __future__ import annotations

from typing import Optional

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from .kinematics import Kinematics
from .scenarios import ScenarioResult
from .units import fmt_length

# Validated categorical palette (light surface), slots 1-3 in fixed order.
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
TEXT, TEXT2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
LOWCONF = "#d9d7d0"

RC = {
    "font.size": 9,
    "axes.edgecolor": "#b9b8b2",
    "axes.labelcolor": TEXT2,
    "axes.titlesize": 10,
    "axes.titlecolor": TEXT,
    "xtick.color": TEXT2,
    "ytick.color": TEXT2,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "lines.linewidth": 1.6,
}


def _fig(w: float, h: float) -> Figure:
    import matplotlib
    matplotlib.rcParams.update(RC)
    fig = Figure(figsize=(w, h), dpi=110, layout="constrained")
    return fig


def _clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _band(ax, t, y, s, color, label):
    """Line with +-1 sigma band; NaN samples (e.g. duplicate frames) are dropped, not drawn as gaps."""
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    s = s[ok] if s is not None else None
    t, y, s = _break_gaps(t, y, s)
    ax.plot(t, y, color=color, label=label)
    if s is not None:
        ax.fill_between(t, y - s, y + s, color=color, alpha=0.18, linewidth=0, label=f"{label} ±1σ")


def _break_gaps(t, y, s=None):
    """Insert NaNs where samples are missing for a while (lost/excluded spans), so lines do not bridge them."""
    if len(t) < 3:
        return t, y, s
    dt = np.diff(t)
    gaps = np.flatnonzero(dt > 3 * np.median(dt))
    if not len(gaps):
        return t, y, s
    t = np.insert(t, gaps + 1, np.nan)
    y = np.insert(y, gaps + 1, np.nan)
    s = None if s is None else np.insert(s, gaps + 1, np.nan)
    return t, y, s


def _shade_low_conf(ax, t, low, valid=None):
    """Shade time spans of untrusted samples, merging across frames that carry no sample (duplicates)."""
    if low is None or not np.any(low):
        return
    idx = np.flatnonzero(valid) if valid is not None else np.arange(len(t))
    if not len(idx):
        return
    flags = low[idx]
    tt = t[idx]
    edges = np.flatnonzero(np.diff(np.concatenate([[0], flags.astype(int), [0]])))
    for a, b in zip(edges[::2], edges[1::2]):
        ax.axvspan(tt[a], tt[min(b, len(tt) - 1)], color=LOWCONF, alpha=0.5, linewidth=0)


OTHER_STYLES = [(C2, "x"), (C3, "+"), ("#6250d6", "^"), ("#e87ba4", "v"), ("#008300", "s")]


def path_figure(kins, background: Optional[np.ndarray] = None, primary: int = 0) -> Figure:
    """On-screen (left) and camera-compensated (right) paths of every tracked object.

    The primary object is coloured by time; others (second targets, background controls) use fixed colours and
    markers. Heading changes ("turns") of the primary object are annotated on both panels, so it is visible which
    turns survive camera compensation.
    """
    if isinstance(kins, Kinematics):
        kins = [kins]
    kins = [k for k in kins if k is not None]
    primary = min(primary, len(kins) - 1)
    k0 = kins[primary]
    fig = _fig(11, 4.8)
    ax1, ax2 = fig.subplots(1, 2)
    if background is not None:
        ax1.imshow(background, cmap="gray", alpha=0.6)
    use_angles = k0.has_angles()
    static = not k0.params.use_camera
    lc = None
    for i, kin in enumerate(kins):
        v = kin.valid
        label = f"{kin.name} ({kin.role})"
        sx, sy = kin.get("img_x"), kin.get("img_y")
        if use_angles:
            cx, cy = kin.get("los_x"), kin.get("los_y")
            vt = kin.trusted & v
        else:
            cx, cy = kin.get("stab_x"), kin.get("stab_y")
            vt = kin.trusted & v
        if i == primary:
            lc = _time_line(ax1, sx[v], sy[v], kin.t[v], label)
            _time_line(ax2, cx[vt], cy[vt], kin.t[vt], label)
        else:
            col, mk = OTHER_STYLES[(i - (i > primary)) % len(OTHER_STYLES)]
            ax1.plot(sx[v], sy[v], mk, color=col, ms=4, mew=1, label=label)
            ax2.plot(cx[vt], cy[vt], mk, color=col, ms=4, mew=1, label=label)
    # turn annotations for the primary object
    for ax, key, (X, Y) in ((ax1, "screen", ("img_x", "img_y")),
                            (ax2, "compensated", ("los_x", "los_y") if use_angles else ("stab_x", "stab_y"))):
        for tr in k0.turns.get(key, []):
            i = tr["index"]
            x, y = k0.get(X)[i], k0.get(Y)[i]
            if np.isfinite(x) and np.isfinite(y):
                ax.annotate(f"{tr['angle_deg']:.0f}° @{tr['t']:.1f}s", (x, y), xytext=(6, 6),
                            textcoords="offset points", fontsize=7, color=TEXT,
                            bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.75))
    ax1.set_title("On-screen path (raw image) [MEASURED]")
    ax1.set_xlabel("x (px)")
    ax1.set_ylabel("y (px)")
    if background is None:
        ax1.invert_yaxis()
    ax1.set_aspect("equal", adjustable="datalim")
    if use_angles:
        ax2.set_title("Camera-compensated LOS path " + ("[STATIC CAMERA ASSUMED]" if static else "[MEASURED+FOV]"))
        ax2.set_xlabel("LOS x (deg, right)")
        ax2.set_ylabel("LOS y (deg, up)")
    else:
        ax2.invert_yaxis()
        ax2.set_title("Camera-compensated path, ref. px " + ("[STATIC CAMERA ASSUMED]" if static else "[MEASURED]"))
        ax2.set_xlabel("x (reference-frame px)")
        ax2.set_ylabel("y (reference-frame px)")
    if not np.any(k0.trusted & k0.valid):
        ax2.text(0.5, 0.5, "No frames with trusted camera motion:\ncompensated path not available",
                 ha="center", va="center", transform=ax2.transAxes, color=TEXT2)
    ax2.set_aspect("equal", adjustable="datalim")
    for ax in (ax1, ax2):
        _clean(ax)
        ax.legend(loc="best", fontsize=7.5)
    if lc is not None:
        cb = fig.colorbar(lc, ax=[ax1, ax2], shrink=0.8)
        cb.set_label(f"time (s), {k0.name}")
    return fig


def _time_line(ax, x, y, t, label=None):
    if len(x) < 2:
        return None
    pts = np.stack([x, y], 1).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    # do not draw connecting lines across gaps in the track (lost spans, re-acquisitions)
    dt = np.diff(t)
    keep = dt <= 3 * np.median(dt)
    segs, tc = segs[keep], t[:-1][keep]
    from matplotlib import colormaps
    lc = LineCollection(segs, cmap=colormaps["plasma"], linewidths=1.6)
    lc.set_array(tc)
    lc.set_clim(t.min(), t.max())
    ax.add_collection(lc)
    sc = ax.scatter(x, y, c=t, cmap="plasma", s=7, vmin=t.min(), vmax=t.max(), label=label, zorder=3)
    ax.plot(x[0], y[0], "o", ms=6, color=C3, label="start")
    ax.plot(x[-1], y[-1], "s", ms=6, color=C2, label="end")
    ax.autoscale_view()
    return lc


def kinematics_figure(kin: Kinematics, controls: Optional[list] = None) -> Figure:
    rows = 4 if kin.has_angles() else 2
    fig = _fig(10, 2.0 * rows + 0.6)
    axs = fig.subplots(rows, 1, sharex=True)
    t = kin.t
    v = kin.valid
    conf = kin.cols.get("cam_confidence", (None, None, None))[2]
    # shade frames whose derivative window contains a low-confidence camera step
    low = (kin.valid & ~kin.trusted) if conf is not None else None

    def masked(a):
        return np.where(v, a, np.nan)      # _band drops the NaNs, so duplicate frames do not break lines

    i = 0
    if kin.has_angles():
        ax = axs[i]
        _shade_low_conf(ax, t, low, v)
        _band(ax, t, masked(kin.get("omega")), masked(kin.get("omega_sigma")), C1, "|ω|")
        if "screen_omega" in kin.cols and kin.params.use_camera:
            ok = v & np.isfinite(kin.get("screen_omega"))
            ax.plot(t[ok], kin.get("screen_omega")[ok], color=TEXT2, lw=1, ls=":", label="on-screen (uncompensated)")
        for c in controls or []:
            if c is not None and "omega" in c.cols:
                ok = c.valid & np.isfinite(c.get("omega"))
                ax.plot(c.t[ok], c.get("omega")[ok], color=C3, lw=1.2, ls="--", label=f"control '{c.name}' residual")
        ax.set_ylabel("deg/s")
        ax.set_title("Angular velocity of the line of sight " +
                     ("[MEASURED+FOV]" if kin.params.use_camera else "[STATIC CAMERA ASSUMED]"))
        ax.legend(loc="upper right", fontsize=8)
        i += 1
        if "alpha" in kin.cols:
            ax = axs[i]
            _shade_low_conf(ax, t, low, v)
            _band(ax, t, masked(kin.get("alpha")), masked(kin.get("alpha_sigma")), C1, "|α| total")
            _band(ax, t, masked(kin.get("alpha_lateral")), masked(kin.get("alpha_lateral_sigma")), C2, "α lateral")
            ax.set_ylabel("deg/s²")
            ax.set_title("Angular acceleration " +
                         ("[MEASURED+FOV]" if kin.params.use_camera else "[STATIC CAMERA ASSUMED]"))
            ax.legend(loc="upper right", fontsize=8, ncol=2)
        i += 1
    ax = axs[i]
    _shade_low_conf(ax, t, low, v)
    ok = v & np.isfinite(kin.get("img_speed"))
    ax.plot(*_break_gaps(t[ok], kin.get("img_speed")[ok])[:2], color=C1, label="raw on-screen (not camera-compensated)")
    if kin.params.use_camera:
        _band(ax, t, masked(kin.get("stab_speed")), masked(kin.get("stab_speed_sigma")), C2, "camera-compensated")
    ax.set_ylabel("px/s")
    ax.set_title("Image-plane speed [MEASURED]" if kin.params.use_camera else
                 "Image-plane speed [MEASURED] (static camera ASSUMED: all of it attributed to the object)")
    ax.legend(loc="upper right", fontsize=8)
    i += 1
    ax = axs[i]
    if conf is not None:
        okc = np.isfinite(conf) & v if v.any() else np.isfinite(conf)
        ax.plot(t[okc], conf[okc], color=C1, label="confidence")
        ax.axhline(kin.params.low_conf_threshold, color=TEXT2, lw=1, ls="--", label="low threshold")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title("Camera-motion confidence [MEASURED]  (grey bands = untrusted derivative windows)")
        ax.legend(loc="lower right", fontsize=8)
    else:
        ax.text(0.5, 0.5, "Camera motion not estimated (static camera assumed)", ha="center", va="center",
                transform=ax.transAxes, color=TEXT2)
        ax.set_title("Camera-motion estimate")
    ax.set_xlabel("time (s)")
    for a in np.atleast_1d(axs):
        _clean(a)
    return fig


def _yerr(v, s, log):
    s = np.nan_to_num(s)
    return np.vstack([np.minimum(s, 0.95 * np.abs(v)) if log else s, s])


def sweep_figure(res: ScenarioResult) -> Figure:
    fig = _fig(10, 4.2)
    ax1, ax2 = fig.subplots(1, 2)
    R = np.array([r.range_m for r in res.rows]) / 1000.0
    vmed = np.array([r.v_median for r in res.rows])
    vmax = np.array([r.v_max for r in res.rows])
    vms = np.array([r.v_max_sigma for r in res.rows])
    log = len(R) > 1 and R.max() / max(R.min(), 1e-9) > 20
    ax1.plot(R, vmed, "o-", color=C1, ms=4, label="median")
    ax1.errorbar(R, vmax, yerr=_yerr(vmax, vms, log), fmt="s-", color=C2, ms=4, capsize=2, label="peak ±1σ")
    ax1.set_xlabel("assumed range (km)")
    ax1.set_ylabel("transverse speed (m/s)")
    ax1.set_title("Speed vs ASSUMED range")
    lmax = np.array([r.a_lat_max_g for r in res.rows])
    lms = np.array([r.a_lat_max_sigma_g for r in res.rows])
    lp95 = np.array([r.a_lat_p95_g for r in res.rows])
    floor = np.array([r.noise_floor_g for r in res.rows])
    ax2.errorbar(R, lmax, yerr=_yerr(lmax, lms, log), fmt="s-", color=C2, ms=4, capsize=2,
                 label="peak ±1σ" + (" (lower bar clipped on log axis)" if log else ""))
    ax2.plot(R, lp95, "o-", color=C1, ms=4, label="95th percentile")
    ax2.plot(R, 3 * floor, "--", color=TEXT2, lw=1.2, label="3× noise floor")
    ax2.set_xlabel("assumed range (km)")
    ax2.set_ylabel("lateral acceleration (g)")
    ax2.set_title("Lateral g vs ASSUMED range")
    for ax in (ax1, ax2):
        if log:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.legend(fontsize=8)
        _clean(ax)
    mode = res.spec.mode
    fig.suptitle(f"[ASSUMED] scenario sweep ({mode} mode): values scale linearly with range; "
                 f"no row is a measurement", fontsize=9, color=TEXT2)
    return fig


def save(fig: Figure, path: str) -> None:
    fig.savefig(path, dpi=130, facecolor=SURFACE)
