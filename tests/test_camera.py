import numpy as np

from uapka.camera import CameraParams, estimate_camera_motion, naive_phase_correlation_track
from uapka.masks import Mask
from uapka.video import load_frames, probe

from .conftest import make_clip


def _final_pan(ct, W, H):
    c = np.array([W / 2, H / 2, 1.0])
    return (ct.A[-1] @ c)[:2] - c[:2]


def test_fpn_defeats_naive_phase_correlation_but_not_us(lossless_clip):
    path, truth = lossless_clip
    fc = load_frames(probe(path))
    pan = np.array(truth["camera_pan_ref_px"][-1])
    naive = -naive_phase_correlation_track(fc.frames)[-1]
    # The known failure: raw phase correlation locks onto the fixed pattern and reports ~no motion.
    assert np.linalg.norm(naive) < 0.1 * np.linalg.norm(pan)
    masks = [Mask.from_dict(d) for d in truth["hud_masks"]]
    ct = estimate_camera_motion(fc.frames, masks, CameraParams())
    est = _final_pan(ct, fc.meta.width, fc.meta.height)
    assert np.linalg.norm(est - pan) < 0.02 * np.linalg.norm(pan)
    assert np.median(ct.confidence()[1:]) > 0.4
    assert any("RAW_PHASECORR_LOCKED_TO_SENSOR" in p.flags for p in ct.pairs)


def test_zoom_is_measured(tmp_path_factory):
    path, truth = make_clip(tmp_path_factory, "zoom", codec="FFV1", zoom_end=1.4)
    fc = load_frames(probe(path))
    ct = estimate_camera_motion(fc.frames, [Mask.from_dict(d) for d in truth["hud_masks"]], CameraParams())
    # Residual sensor-locked pattern pulls per-frame scale towards 1 as well; on this small, short clip
    # the chained zoom is ~1.5% low. Larger/longer clips measure it to <0.5% (see README, validation).
    assert abs(1.0 / ct.scale()[-1] - 1.4) < 0.03 * 1.4


def test_compression_lag_detected_and_anchored(tmp_path_factory):
    """Lossy encoding of a faint background under strong FPN freezes background motion between
    keyframes. Detect it and correct via keyframe-to-keyframe registration."""
    path, truth = make_clip(tmp_path_factory, "lossy", codec="mp4v")
    fc = load_frames(probe(path))
    masks = [Mask.from_dict(d) for d in truth["hud_masks"]]
    pan = np.array(truth["camera_pan_ref_px"])
    off = estimate_camera_motion(fc.frames, masks, CameraParams(gop_mode="off"))
    on = estimate_camera_motion(fc.frames, masks, CameraParams(gop_mode="auto"))
    assert "lag" in on.gop_info
    assert len(on.anchors) >= 2
    W, H = fc.meta.width, fc.meta.height
    c = np.array([W / 2, H / 2, 1.0])
    k = on.anchors[-1]
    err_off = np.linalg.norm((off.A[k] @ c)[:2] - c[:2] - pan[k])
    err_on = np.linalg.norm((on.A[k] @ c)[:2] - c[:2] - pan[k])
    assert err_on < 0.5 * err_off
