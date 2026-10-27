"""Exports: per-frame CSV, scenario CSVs, figures, annotated video, report (Markdown + HTML)."""

from __future__ import annotations

import base64
import csv
import datetime as _dt
import html
import io
import os
from collections import Counter
from typing import Optional

import cv2
import numpy as np

from . import __version__
from .plots import kinematics_figure, path_figure, save, sweep_figure
from .project import Project
from .provenance import ERROR_SOURCES, PROV_DESCRIPTIONS, Prov
from .scenarios import rows_table
from .units import MS_TO_KT
from .video import read_color_frames

STATUS_BGR = {"OK": (80, 200, 80), "MANUAL": (230, 200, 40), "REACQ": (40, 200, 230), "LOST": (60, 60, 230),
              "WEAK": (60, 60, 230), "AMBIGUOUS": (0, 160, 255), "EXCLUDED": (160, 160, 160),
              "BLANK": (160, 160, 160)}
MASK_BGR = {"HUD": (200, 120, 40), "RETICLE": (200, 60, 200), "WATERMARK": (40, 160, 200),
            "REDACTION": (90, 90, 90), "BORDER": (90, 90, 90), "OTHER": (150, 150, 150)}


def write_csv(path: str, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name) or "obj"


def export_csvs(p: Project, outdir: str) -> list[str]:
    files = []
    for i, kin in enumerate(p.kins):
        if kin is None:
            continue
        header, rows = kin.table()
        # per_frame.csv is always the active target; other objects get their own file
        fn = os.path.join(outdir, "per_frame.csv" if i == p.active else f"per_frame_{_slug(kin.name)}.csv")
        write_csv(fn, header, rows)
        files.append(fn)
    fn = os.path.join(outdir, "provenance_legend.csv")
    write_csv(fn, ["label", "meaning"], [[k.value, v] for k, v in PROV_DESCRIPTIONS.items()])
    files.append(fn)
    turns = [(kin.name, kind, tr) for kin in p.kins if kin is not None for kind in ("screen", "compensated")
             for tr in kin.turns.get(kind, [])]
    if turns:
        fn = os.path.join(outdir, "turns.csv")
        write_csv(fn, ["object", "path", "t_s [MEASURED]", "frame", "x", "y", "heading_change_deg [MEASURED]"],
                  [[n, kind, f"{tr['t']:.3f}", p.file_frame(tr["index"]), f"{tr['x']:.1f}", f"{tr['y']:.1f}",
                    f"{tr['angle_deg']:.1f}"] for n, kind, tr in turns])
        files.append(fn)
    if p.scen is not None:
        h, r = rows_table(p.scen)
        fn = os.path.join(outdir, "scenario_sweep.csv")
        write_csv(fn, h, r)
        files.append(fn)
        if len(p.scen.rows) <= 12:
            ranges = list(p.scen.timeseries)
            h = ["frame", "t_s [MEASURED]"]
            for R in ranges:
                tag = f"@{R / 1000:.4g}km [ASSUMED]"
                h += [f"v_m_s{tag}", f"v_sigma_m_s{tag}", f"a_lateral_g{tag}", f"a_lateral_sigma_g{tag}"]
            rows = []
            for i in range(len(p.kin.t)):
                if not p.kin.valid[i]:
                    continue
                row = [p.file_frame(i), f"{p.kin.t[i]:.6f}"]
                for R in ranges:
                    ts = p.scen.timeseries[R]
                    row += [_f(ts["v"][i]), _f(ts["v_sigma"][i]), _f(ts["a_lat_g"][i]), _f(ts["a_lat_g_sigma"][i])]
                rows.append(row)
            fn = os.path.join(outdir, "scenario_timeseries.csv")
            write_csv(fn, h, rows)
            files.append(fn)
    return files


def _f(v) -> str:
    return "" if not np.isfinite(v) else f"{v:.5g}"


def export_figures(p: Project, outdir: str) -> dict[str, str]:
    figs = {}
    bg = p.cache[0] if p.cache is not None else None
    controls = [k for k in p.kins if k is not None and k.role == "control"]
    for name, fig in (("path", path_figure(p.kins, bg, p.active)), ("kinematics", kinematics_figure(p.kin, controls))):
        fn = os.path.join(outdir, f"{name}.png")
        save(fig, fn)
        figs[name] = fn
    if p.scen is not None:
        fn = os.path.join(outdir, "scenario_sweep.png")
        save(sweep_figure(p.scen), fn)
        figs["sweep"] = fn
    return figs


OBJ_BGR = [(255, 160, 40), (40, 120, 255), (120, 200, 30), (200, 80, 200), (30, 200, 230)]


def export_annotated_video(p: Project, path: str, trail_frames: int = 45, progress=None, cancel=None) -> str:
    meta, kin = p.meta, p.kin
    W, H = meta.width, meta.height
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), meta.fps, (W, H))
    if not writer.isOpened():
        path = os.path.splitext(path)[0] + ".avi"
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), meta.fps, (W, H))
    n = len(p.cache)
    A = p.camera.A if (p.camera is not None and p.kin_params.use_camera) else np.repeat(np.eye(3)[None], n, 0)
    omega = kin.cols.get("omega", (None, None, None))[2] if kin is not None else None
    omega_s = kin.cols.get("omega_sigma", (None, None, None))[2] if kin is not None else None
    conf = kin.cols.get("cam_confidence", (None, None, None))[2] if kin is not None else None
    fs = max(0.4, H / 1000)
    lw = max(1, int(round(H / 540)))
    for k, frame in enumerate(read_color_frames(meta.path, p.cache.offset)):
        if k >= n:
            break
        if cancel and cancel():
            raise InterruptedError("Cancelled")
        if frame.shape[:2] != (H, W):
            frame = cv2.resize(frame, (W, H))
        over = frame.copy()
        for m in p.masks:
            cv2.rectangle(over, (m.x, m.y), (m.x + m.w, m.y + m.h), MASK_BGR.get(m.kind, (150, 150, 150)), -1)
        frame = cv2.addWeighted(over, 0.18, frame, 0.82, 0)
        for m in p.masks:
            cv2.rectangle(frame, (m.x, m.y), (m.x + m.w, m.y + m.h), MASK_BGR.get(m.kind, (150, 150, 150)), 1)
        Ainv = np.linalg.inv(A[k])
        for i, (track, tk) in enumerate(zip(p.tracks, p.kins)):
            if tk is None:
                continue
            col = OBJ_BGR[i % len(OBJ_BGR)]
            js = [j for j in range(max(0, k - trail_frames), k + 1) if tk.valid[j]]
            # on-screen trail (what the image shows)
            raw = [np.array([tk.get("img_x")[j], tk.get("img_y")[j]]) for j in js]
            for a, b in zip(raw[:-1], raw[1:]):
                cv2.line(frame, tuple(np.int32(a)), tuple(np.int32(b)), col, lw, cv2.LINE_AA)
            # background-relative trail: past camera-compensated positions re-projected into this frame; drawn only
            # through frames whose camera motion is trusted (otherwise the re-projection is meaningless)
            if p.camera is not None and p.kin_params.use_camera and tk.trusted[k]:
                stab = np.stack([tk.get("stab_x"), tk.get("stab_y")], 1)
                rel = [Ainv[:2, :2] @ stab[j] + Ainv[:2, 2] for j in js if tk.trusted[j]]
                for a, b in zip(rel[:-1], rel[1:]):
                    cv2.line(frame, tuple(np.int32(a)), tuple(np.int32(b)), (0, 230, 255), lw, cv2.LINE_AA)
            st = track.status[k]
            if st not in ("NONE",) and np.isfinite(track.x[k]):
                c = STATUS_BGR.get(st, col) if i == p.active else col
                x, y, w, h = track.x[k], track.y[k], track.w[k], track.h[k]
                cv2.rectangle(frame, (int(x - w / 2 - 3), int(y - h / 2 - 3)), (int(x + w / 2 + 3), int(y + h / 2 + 3)),
                              c, lw)
                cv2.putText(frame, f"{track.name}{' (control)' if track.role == 'control' else ''} {st}",
                            (int(x + w / 2 + 6), int(y - h / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * fs * 1.2, c, 1,
                            cv2.LINE_AA)
        bar_h = int(46 * fs + 8)
        cv2.rectangle(frame, (0, 0), (W, bar_h), (0, 0, 0), -1)
        st = p.track.status[k]
        line1 = f"f{p.file_frame(k)} t={p.frame_times()[k]:.3f}s  {p.track.name}:{st}"
        if conf is not None and np.isfinite(conf[k]):
            line1 += f"  cam-conf:{conf[k]:.2f}"
        if omega is not None and kin.valid[k] and np.isfinite(omega[k]):
            basis = f"MEASURED+FOV, HFOV={p.kin_params.hfov_deg:g} deg" if p.kin_params.use_camera else \
                "STATIC CAMERA ASSUMED"
            line2 = f"|w| {omega[k]:.4g} +/- {omega_s[k]:.2g} deg/s [{basis}]"
            if not kin.trusted[k]:
                line2 += " UNTRUSTED (camera motion unverified)"
        elif kin is not None and kin.valid[k]:
            line2 = f"on-screen speed {kin.get('img_speed')[k]:.4g} px/s [MEASURED] (no FOV: no angles)"
        else:
            line2 = ""
        cv2.putText(frame, line1, (6, int(20 * fs + 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * fs * 1.2, (255, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(frame, line2, (6, int(42 * fs + 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * fs * 1.2, (200, 230, 255), 1,
                    cv2.LINE_AA)
        legend = "trail: object colour = on-screen; yellow = relative to background (trusted camera motion only)"
        cv2.putText(frame, legend, (6, H - int(26 * fs) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42 * fs * 1.2,
                    (220, 220, 220), 1, cv2.LINE_AA)
        if kin is not None and kin.flags[k]:
            cv2.putText(frame, ",".join(kin.flags[k])[:120], (6, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * fs * 1.2,
                        (80, 200, 255), 1, cv2.LINE_AA)
        writer.write(frame)
        if progress and k % 10 == 0:
            progress(k / n, f"Writing annotated frame {k}")
    writer.release()
    return path


# ---------------------------------------------------------------- report
def _fmt_pm(v, s, unit):
    return f"{v:.4g} ± {s:.2g} {unit}"


def _stat_rows(kin, lab_ang: str) -> list[str]:
    s = kin.summary()
    rows = []
    if "screen_speed_median_px_s" in s:
        v, sg, fr = s["screen_speed_max_px_s"]
        rows.append(f"| On-screen speed (NOT camera-compensated), median / peak (frame {fr}) | "
                    f"{s['screen_speed_median_px_s']:.4g} / {_fmt_pm(v, sg, 'px/s')} | MEASURED |")
    if "screen_omega_median_deg_s" in s:
        v, sg, fr = s["screen_omega_max_deg_s"]
        rows.append(f"| On-screen angular rate (NOT camera-compensated), median / peak | "
                    f"{s['screen_omega_median_deg_s']:.4g} / {_fmt_pm(v, sg, 'deg/s')} | MEASURED+FOV |")
    vimg = kin.get("stab_speed")[kin.trusted]
    if np.isfinite(vimg).any():
        rows.append(f"| Camera-compensated image speed, median | {np.nanmedian(vimg):.4g} ref-px/s | "
                    f"{'MEASURED' if kin.params.use_camera else Prov.STATIC_CAM.value} |")
    if kin.has_angles():
        if "omega_median_deg_s" in s:
            v, sg, fr = s["omega_max_deg_s"]
            rows.append(f"| Camera-compensated angular velocity, median | {s['omega_median_deg_s']:.4g} deg/s | {lab_ang} |")
            rows.append(f"| Camera-compensated angular velocity, peak (frame {fr}) | {_fmt_pm(v, sg, 'deg/s')} | {lab_ang} |")
        if "alpha_max_deg_s2" in s:
            v, sg, fr = s["alpha_max_deg_s2"]
            rows.append(f"| Angular acceleration, peak (frame {fr}) | {_fmt_pm(v, sg, 'deg/s²')} | {lab_ang} |")
            rows.append(f"| Angular acceleration noise floor (median 1σ) | {s['alpha_noise_floor_deg_s2']:.3g} deg/s² "
                        f"| {lab_ang} |")
    return rows


def build_report(p: Project, figs: dict[str, str]) -> tuple[str, str]:
    """Return (markdown, html)."""
    meta, kin, cam, track, kp = p.meta, p.kin, p.camera, p.track, p.kin_params
    now = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
    lab_ang = Prov.MEASURED_FOV.value if kp.use_camera else Prov.STATIC_CAM.value
    md: list[str] = []
    md.append("# UAP Kinematics Analyzer – analysis report\n")
    md.append(f"Generated {now} by uapka {__version__}. Offline analysis; no data left this machine.\n")
    md.append("> **How to read this report.** Every value carries a label. **MEASURED** values come from pixels and "
              "frame timing only. **MEASURED+FOV** values are measured motion converted to angles with the "
              "user-supplied field of view. **MEASURED+FOV+ASSUMED-STATIC-CAMERA** values attribute all on-screen "
              "motion to the object because the camera is assumed not to move. **ASSUMED** values additionally "
              "depend on a scenario assumption (range, size or altitude) and scale linearly with it – they are not "
              "measurements. This report does not identify the object.\n")

    # ---- verdict first: can on-screen motion be separated from camera motion at all?
    md.append("## Summary")
    if kin is None:
        md.append("- The active object has no usable track.\n")
    else:
        n_valid, n_trusted = int(kin.valid.sum()), int((kin.trusted & kin.valid).sum())
        if not kp.use_camera:
            md.append("- **Camera-motion compensation is OFF: a static camera is ASSUMED.** All on-screen motion is "
                      "attributed to the object; if the camera moved, every angular value is wrong by the camera's "
                      "own motion.")
        elif cam is not None:
            frac = n_trusted / max(n_valid, 1)
            md.append(f"- Camera motion verified (trusted) for {n_trusted} of {n_valid} tracked frames ({frac:.0%}).")
            if frac < 0.2:
                md.append("- **The camera's own motion could not be measured reliably in this footage.** On-screen "
                          "motion therefore cannot be separated into object motion and camera motion: on-screen "
                          "paths, speeds and turns describe the image, not the object. "
                          + (cam.closure_info or ""))
        turns_s, turns_c = kin.turns.get("screen", []), kin.turns.get("compensated", [])
        if turns_s:
            comp = (f"{len(turns_c)} (searched only within the {n_trusted / max(n_valid, 1):.0%} of the track with "
                    "trusted camera motion)") if (kp.use_camera and n_trusted) else "n/a"
            md.append(f"- {kin.name}: on-screen heading changes ≥45°: {len(turns_s)}; after camera compensation: "
                      f"{comp} (see section 5).")
        for c in p.kins:
            if c is None or c.role != "control" or not kp.use_camera:
                continue
            ok = c.trusted & c.valid
            if ok.any():
                on, comp_ = np.nanmedian(c.get("img_speed")[ok]), np.nanmedian(c.get("stab_speed")[ok])
                md.append(f"- Control '{c.name}' (background feature): on-screen speed {on:.3g} px/s, after camera "
                          f"compensation {comp_:.3g} px/s (should be 0) – the camera-motion estimate removes "
                          f"{100 * (1 - comp_ / max(on, 1e-9)):.0f}% of background motion; residual "
                          f"{comp_:.3g} px/s is the compensation error. [MEASURED]")
        if kin.role == "target" and kp.use_camera and n_trusted:
            ok = kin.trusted & kin.valid
            md.append(f"- {kin.name}: speed relative to the background (camera-compensated) median "
                      f"{np.nanmedian(kin.get('stab_speed')[ok]):.3g} px/s vs on-screen "
                      f"{np.nanmedian(kin.get('img_speed')[ok]):.3g} px/s over trusted frames. [MEASURED]")
    md.append("")

    md.append("## 1. Evidence")
    md += [f"- {line}" for line in meta.summary_lines()]
    md.append("")
    md.append("## 2. Inputs and assumptions")
    hfov = f"{kp.hfov_deg:g} deg (user input)" if kp.hfov_deg else "**not supplied** – angular values unavailable"
    md += [
        f"- Segment: video frames {p.file_frame(0)}–{p.file_frame(len(p.cache) - 1)} "
        f"({p.frame_times()[0]:.2f}–{p.frame_times()[-1]:.2f} s)",
        f"- Horizontal field of view at reference frame: {hfov}",
        f"- Pixel aspect ratio: {kp.pixel_aspect:g}",
        f"- Timing: {kp.timing} ({meta.fps:.3f} fps nominal)",
        f"- Derivative window: {kp.window_frames} frames, polynomial order {kp.poly_order}",
        f"- Camera-motion compensation: "
        + ("on" if (cam is not None and kp.use_camera) else "**off (static camera ASSUMED)**"),
        f"- FPN removal: {p.camera_params.fpn_mode}, destripe {p.camera_params.destripe}, "
        f"smoothing {p.camera_params.smooth_sigma_px:g} px; motion model {p.camera_params.motion_model}",
        f"- Masks: {len(p.masks)} ("
        + (", ".join(f"{m.kind} {m.w}x{m.h}@{m.x},{m.y}" for m in p.masks) or "none") + ")",
        "- Objects:",
    ]
    for i, t in enumerate(p.tracks):
        md.append(f"  - **{t.name}** ({t.role}{', active' if i == p.active else ''}), tracker mode "
                  f"{t.mode or 'n/a'}, polarity {t.polarity}; keyframes (video frames): "
                  f"{[p.file_frame(k) for k in sorted(t.keyframes)]}; excluded: "
                  f"{[p.file_frame(k) for k in sorted(t.excluded)] or 'none'}")
    md.append("")

    md.append("## 3. Data quality")
    for t in p.tracks:
        sc = Counter(t.status)
        md.append(f"- {t.name}: track status per frame: " + ", ".join(f"{k} {v}" for k, v in sorted(sc.items())))
    if cam is not None:
        c = cam.confidence()[1:]
        md.append(f"- Camera-motion confidence: median {np.median(c):.2f}, 5th pct {np.percentile(c, 5):.2f}; "
                  f"{int((c < kp.low_conf_threshold).sum())} of {len(c)} frame pairs below {kp.low_conf_threshold}")
        for info in (cam.fpn_info, cam.model_info, cam.gop_info, cam.closure_info):
            if info:
                md.append(f"- {info}")
        fc = Counter(f for pr in cam.pairs for f in pr.flags if not f.startswith("BRIDGED_"))
        if fc:
            md.append("- Registration flags: " + ", ".join(f"{k} ×{v}" for k, v in fc.most_common()))
            if fc.get("RAW_PHASECORR_LOCKED_TO_SENSOR"):
                md.append(f"  - In {fc['RAW_PHASECORR_LOCKED_TO_SENSOR']} frame pairs, naive phase correlation on the "
                          "raw frames reported ~zero camera motion while the FPN-cleaned feature registration measured "
                          "motion: the classic fixed-pattern-noise lock that this tool works around.")
    for w in (p.cache.warnings if p.cache else []) + (kin.warnings if kin else []):
        md.append(f"- ⚠ {w}")
    md.append("")

    md.append("## 4. Measured results")
    for i, k in enumerate(p.kins):
        if k is None:
            continue
        md.append(f"### {k.name} ({k.role}{', active' if i == p.active else ''})\n")
        s = k.summary()
        md.append("| Quantity | Value | Label |")
        md.append("|---|---|---|")
        md.append(f"| Frames used / duration | {s['n_frames_used']} / {s['duration_used_s']:.3f} s | MEASURED |")
        md.append(f"| Frames in compensated statistics (trusted camera motion in derivative window) | "
                  f"{s['n_frames_trusted']} | MEASURED |")
        md += _stat_rows(k, lab_ang)
        if k.role == "control":
            md.append("\nThis object was marked as a **control** (a feature fixed to the background). Its "
                      "camera-compensated motion should be zero: the values above are a direct measurement of the "
                      "camera-compensation error, and target motion of similar size is not distinguishable from it.")
        md.append("")
    md.append("Uncertainties are 1σ from tracking noise and chained camera-registration error; they do not include "
              "FOV calibration error or systematic effects listed in section 7.\n")

    md.append("## 5. Heading changes (turns)")
    any_turn = False
    for k in p.kins:
        if k is None:
            continue
        for kind, title in (("screen", "on-screen path (includes camera motion)"),
                            ("compensated", "camera-compensated path")):
            tl = k.turns.get(kind, [])
            if not tl:
                continue
            any_turn = True
            md.append(f"**{k.name}, {title}:** " + ", ".join(f"{tr['angle_deg']:.0f}° @ {tr['t']:.1f} s" for tr in tl))
    if any_turn:
        md.append("\nHeading change of the direction of motion, comparing 0.5 s before and after; only where the "
                  "speed on both sides exceeds 3σ. Turns that appear only on screen are produced by camera motion.\n")
    else:
        md.append("No heading changes ≥45° detected.\n")

    md.append("## 6. Scenario conversion (ASSUMED)")
    if p.scen is None:
        md.append("No scenario computed. " + (p.scen_error or "Requires the field of view and at least one assumption.")
                  + "\n")
    else:
        for n_ in p.scen.notes:
            md.append(f"- {n_}")
        md.append("")
        md.append("| Assumption | Range | Speed median | Speed peak ±1σ | Peak lateral accel ±1σ | Noise floor | Peak > 3σ? |")
        md.append("|---|---|---|---|---|---|---|")
        for r in p.scen.rows:
            md.append(f"| {r.assumption} | {r.range_m / 1000:.4g} km | {r.v_median:.4g} m/s | "
                      f"{r.v_max:.4g} ± {r.v_max_sigma:.2g} m/s ({r.v_max * MS_TO_KT:.4g} kt) | "
                      f"{r.a_lat_max_g:.3g} ± {r.a_lat_max_sigma_g:.2g} g | {r.noise_floor_g:.2g} g | "
                      f"{'yes' if r.significant else 'NO – within noise'} |")
        md.append("\nEvery row is labelled **ASSUMED**: it is a consequence of its assumption, not a measurement.\n")

    md.append("## 7. Error sources")
    md.append("| Affects | Source | Notes |")
    md.append("|---|---|---|")
    for a, src, note in ERROR_SOURCES:
        md.append(f"| {a} | {src} | {note} |")
    md.append("")

    md.append("## 8. Figures")
    for key, title in (("path", "On-screen and camera-compensated path"), ("kinematics", "Angular kinematics"),
                       ("sweep", "Scenario sweep (ASSUMED)")):
        if key in figs:
            md.append(f"### {title}\n\n![{title}]({os.path.basename(figs[key])})\n")
    md_text = "\n".join(md)
    return md_text, _md_to_html(md_text, figs)


def _md_to_html(md: str, figs: dict[str, str]) -> str:
    """Tiny Markdown subset -> self-contained HTML (images embedded as base64)."""
    imgs = {os.path.basename(v): v for v in figs.values()}
    out = []
    in_table = in_list = False

    def inline(s: str) -> str:
        s = html.escape(s)
        import re
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        return s

    for line in md.splitlines():
        if in_table and not line.startswith("|"):
            out.append("</table>")
            in_table = False
        if in_list and not line.lstrip().startswith("- "):
            out.append("</ul>")
            in_list = False
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(set(c) <= set("-") for c in cells):
                continue
            if not in_table:
                out.append("<table>")
                in_table = True
                out.append("<tr>" + "".join(f"<th>{inline(c)}</th>" for c in cells) + "</tr>")
            else:
                out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in cells) + "</tr>")
        elif line.startswith("### "):
            out.append(f"<h3>{inline(line[4:])}</h3>")
        elif line.startswith("## "):
            out.append(f"<h2>{inline(line[3:])}</h2>")
        elif line.startswith("# "):
            out.append(f"<h1>{inline(line[2:])}</h1>")
        elif line.startswith("> "):
            out.append(f"<blockquote>{inline(line[2:])}</blockquote>")
        elif line.lstrip().startswith("- "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{inline(line.lstrip()[2:])}</li>")
        elif line.startswith("!["):
            name = line[line.index("(") + 1:line.rindex(")")]
            if name in imgs:
                with open(imgs[name], "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                out.append(f'<img alt="{html.escape(name)}" src="data:image/png;base64,{b64}">')
        elif line.strip():
            out.append(f"<p>{inline(line)}</p>")
    if in_table:
        out.append("</table>")
    if in_list:
        out.append("</ul>")
    css = ("body{font-family:Segoe UI,Arial,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#0b0b0b;"
           "background:#fcfcfb;line-height:1.45}table{border-collapse:collapse;margin:8px 0;font-size:13px}"
           "td,th{border:1px solid #d9d7d0;padding:4px 8px;vertical-align:top}th{background:#efeee9}"
           "blockquote{border-left:4px solid #2a78d6;margin:12px 0;padding:6px 12px;background:#eef4fc}"
           "img{max-width:100%;border:1px solid #e4e3df}h2{border-bottom:1px solid #e4e3df;padding-bottom:4px}")
    return ("<!doctype html><html><head><meta charset='utf-8'><title>UAPKA report</title>"
            f"<style>{css}</style></head><body>" + "\n".join(out) + "</body></html>")


def export_all(p: Project, outdir: str, video: bool = True, progress=None, cancel=None) -> list[str]:
    os.makedirs(outdir, exist_ok=True)
    if p.kin is None:
        p.run_kinematics()
    files = export_csvs(p, outdir)
    figs = export_figures(p, outdir)
    files += list(figs.values())
    md, htm = build_report(p, figs)
    for name, text in (("report.md", md), ("report.html", htm)):
        fn = os.path.join(outdir, name)
        with open(fn, "w", encoding="utf-8") as f:
            f.write(text)
        files.append(fn)
    proj = os.path.join(outdir, "project.json")
    p.save(proj)
    files.append(proj)
    if video:
        files.append(export_annotated_video(p, os.path.join(outdir, "annotated.mp4"), progress=progress, cancel=cancel))
    return files
