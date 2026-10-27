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
    ax.plot(t, y, color=color, label=label)
    if s is not None:
        ax.fill_between(t, y - s, y + s, color=color, alpha=0.18, linewidth=0, label=f"{label} ±1σ")


def _shade_low_conf(ax, t, low):
    if low is None or not np.any(low):
        return
    edges = np.flatnonzero(np.diff(np.concatenate([[0], low.astype(int), [0]])))
    for a, b in zip(edges[::2], edges[1::2]):
        ax.axvspan(t[a], t[min(b, len(t) - 1)], color=LOWCONF, alpha=0.5, linewidth=0)


def path_figure(kin: Kinematics, background: Optional[np.ndarray] = None) -> Figure:
    fig = _fig(10, 4.4)
    ax1, ax2 = fig.subplots(1, 2)
    v = kin.valid
    t = kin.t
    x, y = kin.get("img_x"), kin.get("img_y")
    if background is not None:
        ax1.imshow(background, cmap="gray", alpha=0.6)
    _time_line(ax1, x[v], y[v], t[v])
    ax1.set_title("On-screen path (raw image) [MEASURED]")
    ax1.set_xlabel("x (px)")
    ax1.set_ylabel("y (px)")
    if background is None:
        ax1.invert_yaxis()
    ax1.set_aspect("equal", adjustable="datalim")
    if kin.has_angles():
        lx, ly = kin.get("los_x"), kin.get("los_y")
        lc = _time_line(ax2, lx[v], ly[v], t[v])
        ax2.set_title("Camera-compensated LOS path [MEASURED+FOV]")
        ax2.set_xlabel("LOS x (deg, right)")
        ax2.set_ylabel("LOS y (deg, up)")
    else:
        sx, sy = kin.get("stab_x"), kin.get("stab_y")
        lc = _time_line(ax2, sx[v], sy[v], t[v])
        ax2.invert_yaxis()
        ax2.set_title("Camera-compensated path (ref. px) [MEASURED]")
        ax2.set_xlabel("x (reference px)")
        ax2.set_ylabel("y (reference px)")
    ax2.set_aspect("equal", adjustable="datalim")
    for ax in (ax1, ax2):
        _clean(ax)
    if lc is not None:
        cb = fig.colorbar(lc, ax=[ax1, ax2], shrink=0.8)
        cb.set_label("time (s)")
    return fig


def _time_line(ax, x, y, t):
    if len(x) < 2:
        return None
    pts = np.stack([x, y], 1).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    from matplotlib import colormaps
    lc = LineCollection(segs, cmap=colormaps["Blues"], linewidths=2)
    lc.set_array(t[:-1])
    lc.set_clim(t.min(), t.max())
    ax.add_collection(lc)
    ax.plot(x[0], y[0], "o", ms=6, color=C3, label="start")
    ax.plot(x[-1], y[-1], "s", ms=6, color=C2, label="end")
    ax.autoscale_view()
    ax.legend(loc="best", fontsize=8)
    return lc


def kinematics_figure(kin: Kinematics) -> Figure:
    rows = 4 if kin.has_angles() else 2
    fig = _fig(10, 2.0 * rows + 0.6)
    axs = fig.subplots(rows, 1, sharex=True)
    t = kin.t
    v = kin.valid
    conf = kin.cols.get("cam_confidence", (None, None, None))[2]
    # shade frames whose derivative window contains a low-confidence camera step
    low = (kin.valid & ~kin.trusted) if conf is not None else None

    def masked(a):
        return np.where(v, a, np.nan)

    i = 0
    if kin.has_angles():
        ax = axs[i]
        _shade_low_conf(ax, t, low)
        _band(ax, t, masked(kin.get("omega")), masked(kin.get("omega_sigma")), C1, "|ω|")
        ax.set_ylabel("deg/s")
        ax.set_title("Angular velocity of the line of sight [MEASURED+FOV]")
        ax.legend(loc="upper right", fontsize=8)
        i += 1
        if "alpha" in kin.cols:
            ax = axs[i]
            _shade_low_conf(ax, t, low)
            _band(ax, t, masked(kin.get("alpha")), masked(kin.get("alpha_sigma")), C1, "|α| total")
            _band(ax, t, masked(kin.get("alpha_lateral")), masked(kin.get("alpha_lateral_sigma")), C2, "α lateral")
            ax.set_ylabel("deg/s²")
            ax.set_title("Angular acceleration [MEASURED+FOV]")
            ax.legend(loc="upper right", fontsize=8, ncol=2)
        i += 1
    ax = axs[i]
    _shade_low_conf(ax, t, low)
    ax.plot(t, masked(kin.get("img_speed")), color=C1, label="raw on-screen")
    _band(ax, t, masked(kin.get("stab_speed")), masked(kin.get("stab_speed_sigma")), C2, "camera-compensated")
    ax.set_ylabel("px/s")
    ax.set_title("Image-plane speed [MEASURED]")
    ax.legend(loc="upper right", fontsize=8)
    i += 1
    ax = axs[i]
    if conf is not None:
        ax.plot(t, conf, color=C1, label="confidence")
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
