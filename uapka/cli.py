"""Command-line interface.

  uapka                         launch the GUI
  uapka gui [VIDEO]             launch the GUI (optionally opening a video or project)
  uapka info VIDEO              metadata + SHA-256
  uapka synth OUT.avi           generate a synthetic test clip with ground truth
  uapka analyze VIDEO ...       headless analysis + export (reproducible, scriptable)
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _progress(f: float, msg: str) -> None:
    sys.stderr.write(f"\r[{f * 100:5.1f}%] {msg:<60}")
    sys.stderr.flush()
    if f >= 1.0:
        sys.stderr.write("\n")


def _parse_box(s: str) -> tuple[float, ...]:
    return tuple(float(v) for v in s.split(","))


def cmd_info(a) -> int:
    from .video import probe
    m = probe(a.video)
    print("\n".join(m.summary_lines()))
    return 0


def cmd_synth(a) -> int:
    from .synth import SynthConfig, generate
    cfg = SynthConfig(codec=a.codec, seconds=a.seconds, hfov_deg=a.hfov, fpn_sigma=a.fpn, scene_texture=a.texture,
                      zoom_end=a.zoom, turn_rate_deg_s=a.turn_rate, seed=a.seed)
    tr = generate(a.out, cfg)
    print(f"Wrote {a.out} and {a.out}.truth.json")
    print(f"Object starts at pixel {tr['object_screen_px'][0]}; HFOV {cfg.hfov_deg} deg; "
          f"suggested masks in truth JSON ('hud_masks').")
    return 0


def cmd_analyze(a) -> int:
    from .camera import naive_phase_correlation_track
    from .export import export_all
    from .masks import Mask, suggest_static_overlays
    from .project import Project
    from .scenarios import ScenarioSpec

    p = Project.load(a.project) if a.project else Project()
    if a.hfov is not None:
        p.kin_params.hfov_deg = a.hfov
    if a.par is not None:
        p.kin_params.pixel_aspect = a.par
    if a.window is not None:
        p.kin_params.window_frames = a.window
    if a.timing:
        p.kin_params.timing = a.timing
    if a.no_fpn:
        p.camera_params.fpn_mode = "off"
    for m in a.mask or []:
        box, _, kind = m.partition(":")
        x, y, w, h = (int(float(v)) for v in box.split(","))
        p.masks.append(Mask(x, y, w, h, (kind or "OTHER").upper()))
    if a.masks_json:
        with open(a.masks_json) as f:
            d = json.load(f)
        p.masks += [Mask.from_dict(m) for m in (d.get("hud_masks", d) if isinstance(d, dict) else d)]
    if a.ranges or a.sizes or a.altitudes:
        if a.ranges:
            p.scenario = ScenarioSpec(mode="range", values=a.ranges)
        elif a.sizes:
            p.scenario = ScenarioSpec(mode="size", values=a.sizes)
        else:
            p.scenario = ScenarioSpec(mode="altitude", values=a.altitudes, cam_altitude=a.cam_alt or "0m",
                                      elevation_deg=a.elevation or 0.0)
        p.scenario.observer_vx_ms, p.scenario.observer_vy_ms = a.observer_v or (0.0, 0.0)

    for w in p.open_video(a.video, progress=_progress):
        print("WARNING:", w, file=sys.stderr)
    if a.auto_masks:
        sugg, msg = suggest_static_overlays(p.cache.frames)
        print(msg, file=sys.stderr)
        p.masks += sugg
    if a.roi:
        cx, cy, w, h = _parse_box(a.roi)
        p.track.keyframes[a.frame] = (cx, cy, w, h)
    if not p.track.keyframes:
        print("ERROR: give the object position with --roi CX,CY,W,H (centre and box size) at --frame N", file=sys.stderr)
        return 2
    p.run_all(progress=_progress, with_camera=not a.static_camera)
    files = export_all(p, a.out, video=not a.no_video, progress=_progress)
    if a.compare_naive and p.camera is not None:
        import numpy as np
        nv = naive_phase_correlation_track(p.cache.frames)
        c = np.array([p.meta.width / 2, p.meta.height / 2, 1.0])
        est = p.camera.A[-1] @ c
        print(f"Cumulative camera pan, FPN-aware feature registration: "
              f"({est[0] - c[0]:+.1f}, {est[1] - c[1]:+.1f}) px", file=sys.stderr)
        print(f"Cumulative camera pan, naive raw phase correlation:    ({-nv[-1][0]:+.1f}, {-nv[-1][1]:+.1f}) px",
              file=sys.stderr)
    s = p.kin.summary()
    print(json.dumps({k: v for k, v in s.items()}, indent=1))
    for w in p.kin.warnings:
        print("WARNING:", w, file=sys.stderr)
    print("Wrote:\n  " + "\n  ".join(files))
    return 0


def cmd_gui(a) -> int:
    from .gui.app import main as gui_main
    return gui_main(getattr(a, "path", None))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="uapka", description="UAP Kinematics Analyzer (offline)")
    sub = ap.add_subparsers(dest="cmd")
    g = sub.add_parser("gui", help="launch the GUI")
    g.add_argument("path", nargs="?", help="video or project.json to open")
    g.set_defaults(fn=cmd_gui)

    i = sub.add_parser("info", help="print video metadata and SHA-256")
    i.add_argument("video")
    i.set_defaults(fn=cmd_info)

    s = sub.add_parser("synth", help="generate a synthetic test clip with known ground truth")
    s.add_argument("out")
    s.add_argument("--codec", default="FFV1", help="FFV1 (lossless, .avi) or mp4v (lossy, .mp4)")
    s.add_argument("--seconds", type=float, default=6.0)
    s.add_argument("--hfov", type=float, default=2.0)
    s.add_argument("--fpn", type=float, default=6.0, help="fixed-pattern noise sigma (grey levels)")
    s.add_argument("--texture", type=float, default=3.0, help="background texture amplitude (grey levels)")
    s.add_argument("--zoom", type=float, default=1.0, help="zoom factor reached at the end of the clip")
    s.add_argument("--turn-rate", type=float, default=0.0, help="object turn rate in angle space (deg/s of heading)")
    s.add_argument("--seed", type=int, default=1)
    s.set_defaults(fn=cmd_synth)

    z = sub.add_parser("analyze", help="headless analysis and export")
    z.add_argument("video")
    z.add_argument("--out", required=True, help="output folder")
    z.add_argument("--project", help="load settings/masks/keyframes from a project.json")
    z.add_argument("--roi", help="object centre and box: CX,CY,W,H")
    z.add_argument("--frame", type=int, default=0, help="frame index of --roi")
    z.add_argument("--hfov", type=float, help="horizontal field of view in degrees at the reference frame")
    z.add_argument("--par", type=float, help="pixel aspect ratio (width/height)")
    z.add_argument("--window", type=int, help="derivative window in frames")
    z.add_argument("--timing", choices=["nominal", "container"])
    z.add_argument("--mask", action="append", help="X,Y,W,H[:KIND] (repeatable); KIND in HUD/RETICLE/WATERMARK/REDACTION")
    z.add_argument("--masks-json", help="JSON list of masks (or a synth truth file)")
    z.add_argument("--auto-masks", action="store_true", help="add auto-suggested static overlay masks")
    z.add_argument("--no-fpn", action="store_true", help="disable fixed-pattern-noise removal")
    z.add_argument("--static-camera", action="store_true", help="skip camera-motion estimation")
    z.add_argument("--ranges", help="ASSUMED slant ranges, e.g. '1km,5km,10nmi' or '2km:40km:2km' or 'log:1km:100km:9'")
    z.add_argument("--sizes", help="ASSUMED object sizes, e.g. '1m,5m,15m'")
    z.add_argument("--altitudes", help="ASSUMED object altitudes, e.g. '5kft:25kft:5kft'")
    z.add_argument("--cam-alt", help="camera altitude for --altitudes, e.g. '25000ft'")
    z.add_argument("--elevation", type=float, help="LOS elevation angle in degrees for --altitudes (negative = down)")
    z.add_argument("--observer-v", type=float, nargs=2, metavar=("VX", "VY"),
                   help="ASSUMED observer velocity across the LOS in m/s (image right, up)")
    z.add_argument("--no-video", action="store_true", help="skip the annotated video")
    z.add_argument("--compare-naive", action="store_true", help="also print naive phase-correlation camera pan")
    z.set_defaults(fn=cmd_analyze)

    a = ap.parse_args(argv)
    if not getattr(a, "fn", None):
        return cmd_gui(a)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
