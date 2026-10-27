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
              "EXCLUDED": (160, 160, 160)}
MASK_BGR = {"HUD": (200, 120, 40), "RETICLE": (200, 60, 200), "WATERMARK": (40, 160, 200),
            "REDACTION": (90, 90, 90), "OTHER": (150, 150, 150)}


def write_csv(path: str, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def export_csvs(p: Project, outdir: str) -> list[str]:
    files = []
    header, rows = p.kin.table()
    fn = os.path.join(outdir, "per_frame.csv")
    write_csv(fn, header, rows)
    files.append(fn)
    fn = os.path.join(outdir, "provenance_legend.csv")
    write_csv(fn, ["label", "meaning"], [[k.value, v] for k, v in PROV_DESCRIPTIONS.items()])
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
                row = [i, f"{p.kin.t[i]:.6f}"]
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
    for name, fig in (("path", path_figure(p.kin, bg)), ("kinematics", kinematics_figure(p.kin))):
        fn = os.path.join(outdir, f"{name}.png")
        save(fig, fn)
        figs[name] = fn
    if p.scen is not None:
        fn = os.path.join(outdir, "scenario_sweep.png")
        save(sweep_figure(p.scen), fn)
        figs["sweep"] = fn
    return figs


def export_annotated_video(p: Project, path: str, trail_frames: int = 45, progress=None, cancel=None) -> str:
    meta, kin, track = p.meta, p.kin, p.track
    W, H = meta.width, meta.height
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), meta.fps, (W, H))
    if not writer.isOpened():
        path = os.path.splitext(path)[0] + ".avi"
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), meta.fps, (W, H))
    A = p.camera.A if (p.camera is not None and p.kin_params.use_camera) else np.repeat(np.eye(3)[None], len(kin.t), 0)
    stab = np.stack([kin.get("stab_x"), kin.get("stab_y")], 1)
    omega = kin.cols.get("omega", (None, None, None))[2]
    omega_s = kin.cols.get("omega_sigma", (None, None, None))[2]
    conf = kin.cols.get("cam_confidence", (None, None, None))[2]
    fs = max(0.4, H / 1000)
    n = len(kin.t)
    for k, frame in enumerate(read_color_frames(meta.path)):
        if k >= n:
            break
        if cancel and cancel():
            raise InterruptedError("Cancelled")
        if frame.shape[:2] != (H, W):
            frame = cv2.resize(frame, (W, H))
        over = frame.copy()
        for m in p.masks:
            c = MASK_BGR.get(m.kind, (150, 150, 150))
            cv2.rectangle(over, (m.x, m.y), (m.x + m.w, m.y + m.h), c, -1)
        frame = cv2.addWeighted(over, 0.18, frame, 0.82, 0)
        for m in p.masks:
            cv2.rectangle(frame, (m.x, m.y), (m.x + m.w, m.y + m.h), MASK_BGR.get(m.kind, (150, 150, 150)), 1)
        # Trail: past camera-compensated positions re-projected into this frame (object path relative to background)
        Ainv = np.linalg.inv(A[k])
        lo = max(0, k - trail_frames)
        pts = []
        for j in range(lo, k + 1):
            if kin.valid[j]:
                q = Ainv[:2, :2] @ stab[j] + Ainv[:2, 2]
                pts.append(q)
        if len(pts) > 1:
            arr = np.array(pts, np.float32)
            for i in range(1, len(arr)):
                a = i / len(arr)
                cv2.line(frame, tuple(np.int32(arr[i - 1])), tuple(np.int32(arr[i])),
                         (int(40 + 200 * a), int(160 * a), 255), 1, cv2.LINE_AA)
        st = track.status[k]
        if st != "NONE" and np.isfinite(track.x[k]):
            c = STATUS_BGR.get(st, (255, 255, 255))
            x, y, w, h = track.x[k], track.y[k], track.w[k], track.h[k]
            cv2.rectangle(frame, (int(x - w / 2 - 3), int(y - h / 2 - 3)), (int(x + w / 2 + 3), int(y + h / 2 + 3)),
                          c, 1)
            cv2.drawMarker(frame, (int(round(x)), int(round(y))), c, cv2.MARKER_CROSS, 8, 1)
        bar_h = int(46 * fs + 8)
        cv2.rectangle(frame, (0, 0), (W, bar_h), (0, 0, 0), -1)
        line1 = f"f{k} t={kin.t[k]:.3f}s  track:{st}"
        if conf is not None and np.isfinite(conf[k]):
            line1 += f"  cam-conf:{conf[k]:.2f}"
        if omega is not None and kin.valid[k] and np.isfinite(omega[k]):
            line2 = f"|w| {omega[k]:.4g} +/- {omega_s[k]:.2g} deg/s [MEASURED+FOV, HFOV={p.kin_params.hfov_deg:g} deg]"
            if not kin.trusted[k]:
                line2 += " UNTRUSTED (low camera confidence)"
        else:
            line2 = "angular rate: n/a"
        cv2.putText(frame, line1, (6, int(20 * fs + 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * fs * 1.2, (255, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(frame, line2, (6, int(42 * fs + 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * fs * 1.2, (200, 230, 255), 1,
                    cv2.LINE_AA)
        if kin.flags[k]:
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


def build_report(p: Project, figs: dict[str, str]) -> tuple[str, str]:
    """Return (markdown, html)."""
    meta, kin, cam, track, kp = p.meta, p.kin, p.camera, p.track, p.kin_params
    now = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
    md: list[str] = []
    md.append("# UAP Kinematics Analyzer – analysis report\n")
    md.append(f"Generated {now} by uapka {__version__}. Offline analysis; no data left this machine.\n")
    md.append("> **How to read this report.** Every value carries a label. **MEASURED** values come from pixels and "
              "frame timing only. **MEASURED+FOV** values are measured motion converted to angles with the "
              "user-supplied field of view. **ASSUMED** values additionally depend on a scenario assumption "
              "(range, size or altitude) and scale linearly with it – they are not measurements. This report does "
              "not identify the object.\n")

    md.append("## 1. Evidence")
    md += [f"- {line}" for line in meta.summary_lines()]
    md.append("")
    md.append("## 2. Inputs and assumptions")
    hfov = f"{kp.hfov_deg:g} deg (user input)" if kp.hfov_deg else "**not supplied** – angular values unavailable"
    md += [
        f"- Horizontal field of view at reference frame: {hfov}",
        f"- Pixel aspect ratio: {kp.pixel_aspect:g}",
        f"- Timing: {kp.timing} ({meta.fps:.3f} fps nominal)",
        f"- Derivative window: {kp.window_frames} frames, polynomial order {kp.poly_order}",
        f"- Frame range: {kp.start_frame} – {kp.end_frame if kp.end_frame >= 0 else 'end'}",
        f"- Camera-motion compensation: {'on' if (cam is not None and kp.use_camera) else '**off** (static camera assumed)'}",
        f"- FPN removal: {p.camera_params.fpn_mode}, destripe {p.camera_params.destripe}, "
        f"smoothing {p.camera_params.smooth_sigma_px:g} px",
        f"- Masks: {len(p.masks)} ("
        + (", ".join(f"{m.kind} {m.w}x{m.h}@{m.x},{m.y}" for m in p.masks) or "none") + ")",
        f"- User keyframes (manual positions): {sorted(track.keyframes)}; excluded frames: {sorted(track.excluded) or 'none'}",
        "",
    ]

    md.append("## 3. Data quality")
    sc = Counter(track.status)
    md.append("- Track status per frame: " + ", ".join(f"{k} {v}" for k, v in sorted(sc.items())))
    if cam is not None:
        c = cam.confidence()[1:]
        md.append(f"- Camera-motion confidence: median {np.median(c):.2f}, 5th pct {np.percentile(c, 5):.2f}; "
                  f"{int((c < kp.low_conf_threshold).sum())} of {len(c)} frame pairs below {kp.low_conf_threshold}")
        md.append(f"- {cam.fpn_info}")
        md.append(f"- Compression check: {cam.gop_info}")
        fc = Counter(f for pr in cam.pairs for f in pr.flags)
        if fc:
            md.append("- Registration flags: " + ", ".join(f"{k} ×{v}" for k, v in fc.most_common()))
            if fc.get("RAW_PHASECORR_LOCKED_TO_SENSOR"):
                md.append(f"  - In {fc['RAW_PHASECORR_LOCKED_TO_SENSOR']} frame pairs, naive phase correlation on the "
                          "raw frames reported ~zero camera motion while the FPN-cleaned feature registration measured "
                          "motion: the classic fixed-pattern-noise lock that this tool works around.")
    for w in (p.cache.warnings if p.cache else []) + kin.warnings:
        md.append(f"- ⚠ {w}")
    md.append("")

    md.append("## 4. Measured results")
    s = kin.summary()
    md.append("| Quantity | Value | Label |")
    md.append("|---|---|---|")
    md.append(f"| Frames used / duration | {s['n_frames_used']} / {s['duration_used_s']:.3f} s | MEASURED |")
    md.append(f"| Frames in peak/median statistics (trusted camera motion in derivative window) | "
              f"{s['n_frames_trusted']} | MEASURED |")
    vimg = kin.get("stab_speed")[kin.trusted]
    if np.isfinite(vimg).any():
        md.append(f"| Camera-compensated image speed, median | {np.nanmedian(vimg):.4g} ref-px/s | MEASURED |")
    if kin.has_angles():
        if "omega_median_deg_s" in s:
            md.append(f"| Angular velocity, median | {s['omega_median_deg_s']:.4g} deg/s | MEASURED+FOV |")
            v, sg, fr = s["omega_max_deg_s"]
            md.append(f"| Angular velocity, peak (frame {fr}) | {_fmt_pm(v, sg, 'deg/s')} | MEASURED+FOV |")
        if "alpha_max_deg_s2" in s:
            v, sg, fr = s["alpha_max_deg_s2"]
            md.append(f"| Angular acceleration, peak (frame {fr}) | {_fmt_pm(v, sg, 'deg/s²')} | MEASURED+FOV |")
            md.append(f"| Angular acceleration noise floor (median 1σ) | {s['alpha_noise_floor_deg_s2']:.3g} deg/s² "
                      f"| MEASURED+FOV |")
        md.append(f"| Total angular travel | {s['total_angular_travel_deg']:.4g} deg | MEASURED+FOV |")
    md.append("\nUncertainties are 1σ from tracking noise and chained camera-registration error; they do not include "
              "FOV calibration error or systematic effects listed in section 6.\n")

    md.append("## 5. Scenario conversion (ASSUMED)")
    if p.scen is None:
        md.append("No scenario computed (requires the field of view and at least one assumption).\n")
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

    md.append("## 6. Error sources")
    md.append("| Affects | Source | Notes |")
    md.append("|---|---|---|")
    for a, src, note in ERROR_SOURCES:
        md.append(f"| {a} | {src} | {note} |")
    md.append("")

    md.append("## 7. Figures")
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
