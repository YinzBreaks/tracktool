import json

import numpy as np
import pytest

from uapka.masks import Mask
from uapka.project import Project
from uapka.synth import SynthConfig, generate

SMALL = dict(width=320, height=240, seconds=3.0, pan_rate_px_s=(40.0, -10.0))


def make_clip(tmp_path_factory, name, **kw):
    cfg = SynthConfig(**{**SMALL, **kw})
    ext = "avi" if cfg.codec == "FFV1" else "mp4"
    path = str(tmp_path_factory.mktemp("clips") / f"{name}.{ext}")
    truth = generate(path, cfg)
    return path, truth


def analysed_project(path, truth, **kin):
    p = Project()
    p.kin_params.hfov_deg = truth["config"]["hfov_deg"]
    for k, v in kin.items():
        setattr(p.kin_params, k, v)
    p.masks = [Mask.from_dict(d) for d in truth["hud_masks"]]
    p.open_video(path)
    x, y = truth["object_screen_px"][0]
    p.track.keyframes[0] = (x + 1.0, y - 1.0, 11, 11)   # deliberately imprecise click
    p.run_all()
    return p


@pytest.fixture(scope="session")
def lossless_clip(tmp_path_factory):
    return make_clip(tmp_path_factory, "lossless", codec="FFV1")


@pytest.fixture(scope="session")
def lossless_project(lossless_clip):
    return analysed_project(*lossless_clip)


def true_rate(truth):
    ang = np.array(truth["object_angle_deg"])
    t = np.array(truth["t"])
    return np.gradient(ang, t, axis=0), t
