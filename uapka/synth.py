"""Synthetic test footage with known ground truth.

Generates an IR-like clip that reproduces the failure conditions this tool is
built for: a low-texture background, strong fixed-pattern noise (per-pixel
offsets, column stripes, hot pixels), a panning/zooming camera, a HUD with
text and a centre reticle, and a redaction block. The object follows a known
angular trajectory, so every measured value can be checked.

Use it in class: run the naive phase-correlation estimate and the FPN-aware
estimate on the same clip and compare both against truth.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SynthConfig:
    width: int = 640
    height: int = 480
    fps: float = 30.0
    seconds: float = 6.0
    hfov_deg: float = 2.0
    pan_rate_px_s: tuple = (60.0, -15.0)   # camera pan (background moves opposite on screen)
    zoom_end: float = 1.0                  # optical zoom factor at end of clip (1 = none)
    scene_texture: float = 3.0             # grey levels of background texture (low = hard)
    fpn_sigma: float = 6.0                 # fixed-pattern noise (grey levels)
    stripe_sigma: float = 4.0
    temporal_noise: float = 1.5
    object_amp: float = 70.0               # bright blob (+) or dark (-)
    object_sigma_px: float = 2.0
    turn_rate_deg_s: float = 0.0           # if >0, object path curves in the reference frame
    hud: bool = True
    codec: str = "mp4v"                   # "mp4v" (lossy) or "FFV1" (lossless, use .avi/.mkv)
    seed: int = 1


def _smooth_noise(rng, h, w, scale):
    n = rng.normal(size=(h // scale + 2, w // scale + 2)).astype(np.float32)
    n = cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
    return n / (n.std() + 1e-6)


def object_angle_truth(cfg: SynthConfig, t: np.ndarray) -> np.ndarray:
    """Object LOS angles (deg) in reference-frame camera coordinates, x right, y up."""
    v0 = np.array([0.15, 0.05])  # deg/s
    if cfg.turn_rate_deg_s <= 0:
        return np.stack([-0.3 + v0[0] * t, 0.1 + v0[1] * t], axis=1)
    # Constant-speed turn in angle space: angular speed |v0|, heading rate turn_rate (rad/s of heading)
    speed = np.linalg.norm(v0)
    w = np.radians(cfg.turn_rate_deg_s)
    h0 = np.arctan2(v0[1], v0[0])
    x = -0.3 + speed / w * (np.sin(h0 + w * t) - np.sin(h0))
    y = 0.1 - speed / w * (np.cos(h0 + w * t) - np.cos(h0))
    return np.stack([x, y], axis=1)


def generate(path: str, cfg: SynthConfig = SynthConfig()) -> dict:
    rng = np.random.default_rng(cfg.seed)
    W, H = cfg.width, cfg.height
    n = int(round(cfg.seconds * cfg.fps))
    t = np.arange(n) / cfg.fps
    f0 = (W / 2) / np.tan(np.radians(cfg.hfov_deg) / 2)

    pan = np.stack([cfg.pan_rate_px_s[0] * t, cfg.pan_rate_px_s[1] * t], axis=1)  # camera boresight, ref px
    zoom = 1.0 + (cfg.zoom_end - 1.0) * t / t[-1]
    margin = int(np.abs(pan).max() + max(W, H)) + 32
    WW, WH = W + 2 * margin, H + 2 * margin
    world = 128 + cfg.scene_texture * (0.6 * _smooth_noise(rng, WH, WW, 40) + 0.4 * _smooth_noise(rng, WH, WW, 6))
    world = world.astype(np.float32)

    fpn = rng.normal(0, cfg.fpn_sigma, (H, W)).astype(np.float32)
    fpn += rng.normal(0, cfg.stripe_sigma, (1, W)).astype(np.float32)
    hot = rng.integers(0, H * W, 40)
    fpn.ravel()[hot] += 80

    ang = object_angle_truth(cfg, t)
    obj_ref = np.stack([W / 2 + np.tan(np.radians(ang[:, 0])) * f0,
                        H / 2 - np.tan(np.radians(ang[:, 1])) * f0], axis=1)

    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*cfg.codec), cfg.fps, (W, H))
    if not writer.isOpened():
        raise IOError("Could not open video writer")
    screen = np.zeros((n, 2))
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    for k in range(n):
        s = zoom[k]
        # frame pixel p -> reference pixel: q = pan + c + (p - c)/s ; world pixel = q + margin
        c = np.array([W / 2, H / 2])
        A = np.array([[1 / s, 0, pan[k, 0] + c[0] - c[0] / s + margin],
                      [0, 1 / s, pan[k, 1] + c[1] - c[1] / s + margin]], np.float32)
        frame = cv2.warpAffine(world, A, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        p = (obj_ref[k] - pan[k] - c) * s + c
        screen[k] = p
        sig = cfg.object_sigma_px * s
        frame += cfg.object_amp * np.exp(-((xx - p[0]) ** 2 + (yy - p[1]) ** 2) / (2 * sig**2))
        frame += fpn + rng.normal(0, cfg.temporal_noise, (H, W)).astype(np.float32)
        img = np.clip(frame, 0, 255).astype(np.uint8)
        if cfg.hud:
            cv2.putText(img, "IR  NAR  1.0", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1, cv2.LINE_AA)
            cv2.putText(img, f"RNG ----  T+{t[k]:05.2f}", (12, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 255, 1,
                        cv2.LINE_AA)
            cx, cy = W // 2, H // 2
            for d in ((-30, -8), (8, 30)):
                cv2.line(img, (cx + d[0], cy), (cx + d[1], cy), 0, 2)
                cv2.line(img, (cx, cy + d[0]), (cx, cy + d[1]), 0, 2)
            cv2.rectangle(img, (W - 150, 10), (W - 10, 40), 0, -1)
        writer.write(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    writer.release()

    rate = np.gradient(ang, t, axis=0)
    truth = {
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.__dict__.items()},
        "f0_px": f0,
        "t": t.tolist(),
        "object_screen_px": screen.tolist(),
        "object_angle_deg": ang.tolist(),
        "camera_pan_ref_px": pan.tolist(),
        "zoom": zoom.tolist(),
        "angular_rate_deg_s": np.linalg.norm(rate, axis=1).tolist(),
        "hud_masks": [
            {"x": 0, "y": 0, "w": 200, "h": 34, "kind": "HUD"},
            {"x": 0, "y": H - 34, "w": 280, "h": 34, "kind": "HUD"},
            {"x": W // 2 - 32, "y": H // 2 - 32, "w": 64, "h": 64, "kind": "RETICLE"},
            {"x": W - 152, "y": 8, "w": 144, "h": 34, "kind": "REDACTION"},
        ] if cfg.hud else [],
    }
    with open(path + ".truth.json", "w") as f:
        json.dump(truth, f)
    return truth
