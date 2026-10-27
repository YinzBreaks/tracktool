import csv
import json
import os
import re

import numpy as np
import pytest

from uapka.export import export_all
from uapka.masks import suggest_static_overlays
from uapka.project import Project
from uapka.scenarios import ScenarioSpec, compute_scenarios
from uapka.video import load_frames, probe

from .conftest import analysed_project, make_clip, true_rate


def test_track_accuracy(lossless_project, lossless_clip):
    _, truth = lossless_clip
    p = lossless_project
    gt = np.array(truth["object_screen_px"])
    u = p.track.usable()
    err = np.hypot(p.track.x - gt[:, 0], p.track.y - gt[:, 1])[u]
    assert u.mean() > 0.95
    assert np.median(err) < 0.5


def test_angular_rate_matches_truth(lossless_project, lossless_clip):
    _, truth = lossless_clip
    kin = lossless_project.kin
    rate, _ = true_rate(truth)
    w_true = np.median(np.linalg.norm(rate, axis=1))
    w = kin.get("omega")[kin.valid]
    assert np.nanmedian(w) == pytest.approx(w_true, rel=0.02)
    # reported sigma should not be overconfident
    z = (w - w_true) / kin.get("omega_sigma")[kin.valid]
    assert np.nanstd(z) < 1.5


def test_lateral_acceleration_of_turning_object(tmp_path_factory):
    path, truth = make_clip(tmp_path_factory, "turn", codec="FFV1", turn_rate_deg_s=90.0, seconds=4.0,
                            pan_rate_px_s=(20.0, -5.0))
    p = analysed_project(path, truth, window_frames=21)
    speed = np.hypot(0.15, 0.05)                         # deg/s, constant
    a_lat_true = speed * np.radians(90.0)                # deg/s^2 = v * heading rate
    a = p.kin.get("alpha_lateral")[p.kin.valid]
    assert np.nanmedian(a) == pytest.approx(a_lat_true, rel=0.25)


def test_every_csv_column_is_labelled_and_scenarios_scale(lossless_project, tmp_path):
    p = lossless_project
    p.scenario = ScenarioSpec(mode="range", values="1km, 2km, 4km")
    p.run_scenarios()
    files = export_all(p, str(tmp_path), video=True)
    names = {os.path.basename(f) for f in files}
    assert {"per_frame.csv", "scenario_sweep.csv", "report.md", "report.html", "path.png", "kinematics.png",
            "scenario_sweep.png", "annotated.mp4", "project.json"} <= names
    with open(tmp_path / "per_frame.csv") as f:
        header = next(csv.reader(f))
    data_cols = [h for h in header if h not in ("frame", "used_in_fit", "in_summaries", "flags")]
    assert all(re.search(r"\[(MEASURED|MEASURED\+FOV|ASSUMED|MEASURED\+FOV\+ASSUMED-STATIC-CAMERA)\]$", h)
               for h in data_cols), data_cols
    assert not any("ASSUMED" in h for h in header), "per-frame CSV must contain no scenario values"
    with open(tmp_path / "scenario_sweep.csv") as f:
        rows = list(csv.reader(f))
    assert all("[ASSUMED]" in h for h in rows[0] if h not in ("assumption", "lateral_peak_above_3sigma"))
    v = [float(r[2]) for r in rows[1:]]
    assert v[1] == pytest.approx(2 * v[0], rel=1e-3) and v[2] == pytest.approx(4 * v[0], rel=1e-3)
    assert all(r[1].startswith("ASSUMED") for r in rows[1:])
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert p.meta.sha256 in report and "Error sources" in report
    cap = __import__("cv2").VideoCapture(str(tmp_path / "annotated.mp4"))
    assert int(cap.get(7)) == len(p.cache)


def test_size_and_altitude_modes(lossless_project):
    kin = lossless_project.kin
    r = compute_scenarios(kin, ScenarioSpec(mode="size", values="1m, 2m"))
    assert r.rows[1].range_m == pytest.approx(2 * r.rows[0].range_m)
    assert any("biased SHORT" in n for n in r.notes)
    r = compute_scenarios(kin, ScenarioSpec(mode="altitude", values="1000m", cam_altitude="7000m",
                                            elevation_deg=-30))
    assert r.rows[0].range_m == pytest.approx(12000)
    r = compute_scenarios(kin, ScenarioSpec(mode="altitude", values="9000m", cam_altitude="7000m",
                                            elevation_deg=-30))
    assert not r.rows and any("inconsistent" in n for n in r.notes)


def test_no_fov_means_no_angles(lossless_project):
    p = lossless_project
    old = p.kin_params.hfov_deg
    p.kin_params.hfov_deg = 0
    try:
        kin = p.run_kinematics()
        assert not kin.has_angles() and "omega" not in kin.cols
        assert any("No horizontal FOV" in w for w in kin.warnings)
        with pytest.raises(ValueError):
            compute_scenarios(kin, ScenarioSpec())
    finally:
        p.kin_params.hfov_deg = old
        p.run_kinematics()


def test_duplicate_frames_are_detected(tmp_path, lossless_clip):
    import cv2
    path, truth = lossless_clip
    src = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = src.read()
        if not ok:
            break
        frames.append(f)
    out = str(tmp_path / "dups.avi")
    w = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"FFV1"), 30.0, frames[0].shape[1::-1])
    dup_idx = set(range(10, len(frames), 10))
    for i, f in enumerate(frames):
        w.write(frames[i - 1] if i in dup_idx else f)
    w.release()
    fc = load_frames(probe(out))
    dup = fc.detect_duplicates()
    assert set(np.flatnonzero(dup)) == dup_idx


def test_static_overlay_suggestions(lossless_clip):
    path, truth = lossless_clip
    fc = load_frames(probe(path))
    masks, msg = suggest_static_overlays(fc.frames)
    assert masks, msg
    # the redaction block (top-right) must be covered by a suggestion
    rx, ry = 320 - 80, 25
    assert any(m.x <= rx <= m.x + m.w and m.y <= ry <= m.y + m.h for m in masks)


def test_project_roundtrip(lossless_project, tmp_path):
    p = lossless_project
    fn = str(tmp_path / "proj.json")
    p.save(fn)
    q = Project.load(fn)
    assert q.video_sha256 == p.video_sha256
    assert q.track.keyframes == {k: tuple(v) for k, v in p.track.keyframes.items()}
    assert np.allclose(q.track.x, p.track.x, equal_nan=True)
    assert len(q.masks) == len(p.masks)
