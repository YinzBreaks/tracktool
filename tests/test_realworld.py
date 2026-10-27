"""Features driven by real released footage: frame-rate-converted clips with duplicates and dropouts, segment
analysis, point-sized targets, multiple objects, turns, letterboxing."""

import cv2
import numpy as np
import pytest

from uapka.kinematics import detect_turns
from uapka.masks import suggest_static_overlays
from uapka.project import Project
from uapka.tracker import Track
from uapka.video import load_frames, probe

from .conftest import lossless_writer, make_clip


def _rewrite(src, dst, transform):
    cap = cv2.VideoCapture(src)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    out = transform(frames)
    w = lossless_writer(dst, 30.0, out[0].shape[1::-1])
    for f in out:
        w.write(f)
    w.release()
    return len(frames), len(out)


@pytest.fixture(scope="module")
def converted_clip(tmp_path_factory, lossless_clip):
    """10 fps content in a 30 fps file (each frame x3, with re-encoding-like noise) plus dropout frames."""
    path, truth = lossless_clip
    dst = str(tmp_path_factory.mktemp("conv") / "conv.avi")
    rng = np.random.default_rng(3)
    blank_at = {40, 41, 42, 70}

    def tf(frames):
        src10 = frames[::3]
        out = []
        for f in src10:
            for _ in range(3):
                # re-encoded copies are not bit-exact: sparse +-1 changes (difference ~0.1 grey levels, as measured
                # on real frame-rate-converted footage)
                noise = rng.integers(-1, 2, f.shape) * (rng.random(f.shape) < 0.1)
                out.append(np.clip(f.astype(np.int16) + noise, 0, 255).astype(np.uint8))
        for i in blank_at:
            out[i] = np.full_like(out[i], 30)
        return out
    _rewrite(path, dst, tf)
    return dst, truth, blank_at


def test_cadence_duplicates_and_dropouts(converted_clip):
    path, truth, blank_at = converted_clip
    fc = load_frames(probe(path))
    fc.detect_blank()
    fc.detect_duplicates()
    assert set(np.flatnonzero(fc.blank)) == blank_at
    new = np.flatnonzero(~fc.skip())
    # new content arrives on the first frame of each triplet; a dropout can shift one to the next frame
    assert np.mean(new % 3 == 0) > 0.9
    assert abs(len(new) - len(fc) / 3) <= 3
    assert 0.6 < fc.duplicate.mean() < 0.7
    assert any("frame-rate converted" in w for w in fc.warnings)


def test_converted_clip_kinematics_use_new_frames_only(converted_clip):
    path, truth, _ = converted_clip
    p = Project()
    p.kin_params.hfov_deg = truth["config"]["hfov_deg"]
    from uapka.masks import Mask
    p.masks = [Mask.from_dict(d) for d in truth["hud_masks"]]
    p.open_video(path)
    x, y = truth["object_screen_px"][0]
    p.track.keyframes[0] = (x, y, 11, 11)
    p.run_all()
    kin = p.kin
    assert np.isfinite(kin.get("omega")[kin.valid]).sum() > 0.5 * kin.valid.sum()
    ang = np.array(truth["object_angle_deg"])
    w_true = np.median(np.linalg.norm(np.gradient(ang, np.array(truth["t"]), axis=0), axis=1))
    assert np.nanmedian(kin.get("omega")[kin.valid]) == pytest.approx(w_true, rel=0.05)
    assert not p.track.usable()[p.cache.skip()].any()


def test_segment_offsets(lossless_clip):
    path, truth = lossless_clip
    p = Project(segment_start_s=1.0, segment_end_s=2.0)
    p.open_video(path)
    assert p.cache.offset == 30 and len(p.cache) == 31
    assert p.frame_times()[0] == pytest.approx(1.0)
    full = load_frames(probe(path))
    assert np.array_equal(p.cache[5], full[35])
    assert p.file_frame(5) == 35 and p.cache_index(35) == 5


def test_point_mode_tracks_two_pixel_target(tmp_path_factory):
    path, truth = make_clip(tmp_path_factory, "speck", codec="FFV1", object_sigma_px=0.8, object_amp=45.0)
    p = Project()
    p.open_video(path)
    from uapka.masks import Mask
    p.masks = [Mask.from_dict(d) for d in truth["hud_masks"]]
    x, y = truth["object_screen_px"][0]
    p.track.keyframes[0] = (x + 4, y - 3, 15, 15)      # sloppy click: snapping must fix it
    p.run_tracking()
    assert p.track.mode == "point"
    gt = np.array(truth["object_screen_px"])
    u = p.track.usable()
    err = np.hypot(p.track.x - gt[:, 0], p.track.y - gt[:, 1])[u]
    assert u.mean() > 0.9 and np.median(err) < 0.6


def test_detect_turns():
    t = np.arange(0, 4, 0.1)
    x = np.where(t < 2, t * 50, 100 + (t - 2) * 50 * np.cos(np.radians(120)))
    y = np.where(t < 2, 0.0, (t - 2) * 50 * np.sin(np.radians(120)))
    vx, vy = np.gradient(x, t), np.gradient(y, t)
    s = np.full_like(t, 0.5)
    tr = detect_turns(t, x, y, vx, vy, s, s, np.ones(len(t), bool), np.zeros(len(t), int))
    assert len(tr) == 1 and tr[0]["t"] == pytest.approx(2.0, abs=0.15) and tr[0]["angle_deg"] == pytest.approx(120, abs=3)


def test_track_start_end_and_multi_object_roundtrip(lossless_clip, tmp_path):
    path, truth = lossless_clip
    p = Project()
    p.open_video(path)
    gt = np.array(truth["object_screen_px"])
    p.track.keyframes[30] = (gt[30, 0], gt[30, 1], 11, 11)
    p.track.start, p.track.end = 20, 60
    p.add_track("bg", "control").keyframes[0] = (100, 100, 11, 11)
    p.run_tracking()
    st = p.track.status
    assert all(s_ == "NONE" for s_ in st[:20]) and all(s_ == "NONE" for s_ in st[61:])
    assert p.track.usable()[20:61].mean() > 0.9
    fn = str(tmp_path / "proj.json")
    p.save(fn)
    q = Project.load(fn)
    assert [t.name for t in q.tracks] == ["obj1", "bg"] and q.tracks[1].role == "control"
    assert (q.tracks[0].start, q.tracks[0].end) == (20, 60)


def test_letterbox_border_detected(tmp_path, lossless_clip):
    path, truth = lossless_clip
    dst = str(tmp_path / "lb.avi")

    def tf(frames):
        out = []
        for f in frames:
            g = f.copy()
            g[:12] = 0
            g[-12:] = 0
            out.append(g)
        return out
    _rewrite(path, dst, tf)
    fc = load_frames(probe(dst))
    masks, _ = suggest_static_overlays(fc.frames)
    borders = [m for m in masks if m.kind == "BORDER"]
    assert len(borders) == 2 and all(m.w == fc.meta.width for m in borders)
    assert not any(m.w == fc.meta.width and m.kind != "BORDER" for m in masks)
