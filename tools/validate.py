"""Validate the pipeline against synthetic clips with known ground truth.

    python tools/validate.py [workdir]

Prints a Markdown table. Every clip has strong fixed-pattern noise, a faint
background, HUD overlays and a reticle; they differ in compression, zoom and
object motion. Use it in class to show what each failure mode does to the
naive estimate and to the FPN-aware one.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uapka.camera import naive_phase_correlation_track  # noqa: E402
from uapka.masks import Mask  # noqa: E402
from uapka.project import Project  # noqa: E402
from uapka.synth import SynthConfig, generate  # noqa: E402

CASES = [
    ("lossless (FFV1), FPN 6, texture 3", dict(codec="FFV1")),
    ("lossy mp4v, FPN 6, texture 3", dict(codec="mp4v")),
    ("lossy mp4v, FPN 6, texture 6", dict(codec="mp4v", scene_texture=6.0)),
    ("lossless, zoom 1.0→1.5", dict(codec="FFV1", zoom_end=1.5)),
    ("lossless, turning object 90°/s", dict(codec="FFV1", turn_rate_deg_s=90.0, pan_rate_px_s=(30.0, -8.0))),
]


def run(workdir: str) -> None:
    print("| Clip | Naive PC pan error | Our pan error | Median cam. conf. | ω true | ω measured (median) "
          "| ω z-spread* | Lateral α true | Lateral α measured |")
    print("|---|---|---|---|---|---|---|---|---|")
    for i, (name, kw) in enumerate(CASES):
        cfg = SynthConfig(**kw)
        path = os.path.join(workdir, f"case{i}.{'avi' if cfg.codec == 'FFV1' else 'mp4'}")
        truth = generate(path, cfg)
        p = Project()
        p.kin_params.hfov_deg = cfg.hfov_deg
        p.kin_params.window_frames = 21 if cfg.turn_rate_deg_s else 15
        p.masks = [Mask.from_dict(d) for d in truth["hud_masks"]]
        p.open_video(path)
        x, y = truth["object_screen_px"][0]
        p.track.keyframes[0] = (x + 1, y - 1, 11, 11)
        p.run_all()
        W, H = cfg.width, cfg.height
        c = np.array([W / 2, H / 2, 1.0])
        pan = np.array(truth["camera_pan_ref_px"])[-1]
        est = (p.camera.A[-1] @ c)[:2] - c[:2]
        naive = -naive_phase_correlation_track(p.cache.frames)[-1]
        pe = lambda e: f"{100 * np.linalg.norm(e - pan) / np.linalg.norm(pan):.1f}%"  # noqa: E731
        ang = np.array(truth["object_angle_deg"])
        t = np.array(truth["t"])
        rate = np.gradient(ang, t, axis=0)
        w_true = np.linalg.norm(rate, axis=1)
        kin = p.kin
        trusted = kin.valid & (kin.get("cam_confidence") >= kin.params.low_conf_threshold)
        w = kin.get("omega")
        z = (w - w_true) / kin.get("omega_sigma")
        if cfg.turn_rate_deg_s:
            lat_true = f"{np.hypot(0.15, 0.05) * np.radians(cfg.turn_rate_deg_s):.4f}"
            lat_meas = f"{np.nanmedian(kin.get('alpha_lateral')[kin.valid]):.4f}"
        else:
            lat_true, lat_meas = "0", f"{np.nanmedian(kin.get('alpha_lateral')[kin.valid]):.4f}"
        print(f"| {name} | {pe(naive)} | {pe(est)} | {np.median(p.camera.confidence()[1:]):.2f} | "
              f"{np.median(w_true):.4f} | {np.nanmedian(w[kin.valid]):.4f} | {np.nanstd(z[trusted]):.2f} | "
              f"{lat_true} | {lat_meas} |")
        p.cache.close()
    print("\n*standard deviation of (measured − true)/σ over frames with camera confidence ≥ threshold; "
          "1.0 = honest error bars, <1 = conservative. Angular units deg/s and deg/s².")


if __name__ == "__main__":
    wd = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="uapka_val_")
    os.makedirs(wd, exist_ok=True)
    run(wd)
