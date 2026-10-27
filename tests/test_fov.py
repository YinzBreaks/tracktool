"""Field-of-view derivation (uapka.fov): pinhole round trips, uncertainty behaviour, geodesy, project and CLI plumbing."""

import dataclasses
import json
import math
import os

import cv2
import numpy as np
import pytest

from uapka.cli import main
from uapka.export import build_report
from uapka.fov import (FovEstimate, angle_between_points, az_el_range, describe_inputs, detect_inset, ecef_to_geodetic,
                       enu_to_geodetic, enu_vector, estimate_geo, estimate_ground, estimate_hud, estimate_separation,
                       estimate_size, fit_homography, focal_px, geodetic_to_ecef, hfov_from_focal, load_fov_json,
                       project_direction, small_angle_hfov)
from uapka.kinematics import KinParams, compute_kinematics
from uapka.masks import Mask
from uapka.project import Project
from uapka.provenance import PROV_DESCRIPTIONS, Prov, derived_fov_label

from .conftest import lossless_writer

W, H = 1920, 1080


def _true_angle_deg(p1, p2, hfov, par=1.0):
    return math.degrees(angle_between_points(p1, p2, W, H, focal_px(W, hfov), par))


# ---------------------------------------------------------------- geometry round trips
@pytest.mark.parametrize("hfov", [0.7, 2.0, 25.0, 60.0])
@pytest.mark.parametrize("p1,p2", [((100.0, 100.0), (1800.0, 900.0)), ((950.0, 530.0), (1010.0, 560.0)),
                                   ((1500.0, 200.0), (1900.0, 1000.0))])
def test_separation_round_trip(hfov, p1, p2):
    th = _true_angle_deg(p1, p2, hfov)
    est = estimate_separation(W, H, p1, p2, th, angle_sigma_deg=0.02 * th)
    assert est.hfov_deg == pytest.approx(hfov, rel=1e-6)
    assert est.method == "separation" and est.label == "DERIVED-FOV(separation)"
    assert est.sigma_deg > 0 and est.sigma_pct == pytest.approx(100 * est.sigma_deg / hfov)
    assert est.diagnostics["focal_px"] == pytest.approx(focal_px(W, hfov), rel=1e-6)
    json.dumps(est.to_dict())   # fully serialisable


def test_wide_fov_off_centre_small_angle_formula_is_wrong():
    """Two points on the same side of a 60 deg frame: the exact solution is right, theta*W/d is >5% off."""
    hfov = 60.0
    p1, p2 = (1500.0, 540.0), (1900.0, 540.0)
    th = _true_angle_deg(p1, p2, hfov)
    est = estimate_separation(W, H, p1, p2, th)
    assert est.hfov_deg == pytest.approx(hfov, rel=1e-3)
    sa = small_angle_hfov(p1, p2, W, H, th)
    assert abs(sa - hfov) / hfov > 0.05
    assert est.diagnostics["small_angle_hfov_deg"] == pytest.approx(sa)
    assert any("small-angle" in n for n in est.notes)
    # same-side points: the angle between the rays peaks near f = sqrt(r1 r2), so a second (very wide) solution
    # exists; it is reported, and the narrow one is returned
    alt = est.diagnostics.get("alternative_hfov_deg")
    assert alt is not None and alt > 100.0 and any("second, very wide-angle solution" in n for n in est.notes)
    assert _true_angle_deg(p1, p2, alt) == pytest.approx(th, rel=1e-6)
    # points straddling the centre: monotone, no second solution
    q1, q2 = (300.0, 540.0), (1600.0, 540.0)
    est2 = estimate_separation(W, H, q1, q2, _true_angle_deg(q1, q2, hfov))
    assert est2.hfov_deg == pytest.approx(hfov, rel=1e-6) and "alternative_hfov_deg" not in est2.diagnostics


def test_pixel_aspect_is_honoured():
    par = 1.1
    p1, p2 = (300.0, 200.0), (1500.0, 900.0)
    th = _true_angle_deg(p1, p2, 20.0, par)
    assert estimate_separation(W, H, p1, p2, th, pixel_aspect=par).hfov_deg == pytest.approx(20.0, rel=1e-6)
    assert estimate_separation(W, H, p1, p2, th, pixel_aspect=1.0).hfov_deg != pytest.approx(20.0, rel=1e-3)


def test_size_round_trip_and_assumption_propagation():
    hfov, R = 3.0, 5000.0
    p1, p2 = (900.0, 500.0), (1000.0, 560.0)
    th = math.radians(_true_angle_deg(p1, p2, hfov))
    L = 2 * R * math.tan(th / 2)
    est = estimate_size(W, H, p1, p2, L, R)
    assert est.hfov_deg == pytest.approx(hfov, rel=1e-6)
    assert est.label == "DERIVED-FOV(size)"
    # a 20% range sigma gives ~20% relative FOV sigma: theta ~ L/R, and the central difference of 1/R over
    # +-20% is (1/0.8 - 1/1.2)/2 = 20.8%, which is what finite differences honestly report
    est = estimate_size(W, H, p1, p2, L, R, range_sigma_m=0.2 * R, click_sigma_px=0.0)
    assert est.sigma_pct == pytest.approx(20.83, rel=0.01)
    est = estimate_size(W, H, p1, p2, L, R, size_sigma_m=0.1 * L, click_sigma_px=0.0)
    assert est.sigma_pct == pytest.approx(10.0, rel=0.01)
    assert any("ASSUMED" in n for n in est.notes)


def test_sigma_behaviour():
    p1, p2 = (100.0, 100.0), (1800.0, 900.0)
    th = _true_angle_deg(p1, p2, 2.0)
    # doubling the angle sigma doubles the FOV sigma (clicks switched off)
    s1 = estimate_separation(W, H, p1, p2, th, angle_sigma_deg=0.1, click_sigma_px=0.0).sigma_deg
    s2 = estimate_separation(W, H, p1, p2, th, angle_sigma_deg=0.2, click_sigma_px=0.0).sigma_deg
    assert s2 == pytest.approx(2 * s1, rel=1e-3)
    assert s1 / 2.0 == pytest.approx(0.1 / th, rel=1e-2)        # relative sigma carries through
    # sigma grows with click uncertainty, and the click share is ~linear in it
    c1 = estimate_separation(W, H, p1, p2, th, click_sigma_px=1.0).sigma_deg
    c2 = estimate_separation(W, H, p1, p2, th, click_sigma_px=2.0).sigma_deg
    assert c2 > c1 and c2 == pytest.approx(2 * c1, rel=1e-3)
    both = estimate_separation(W, H, p1, p2, th, angle_sigma_deg=0.1, click_sigma_px=1.0)
    assert both.sigma_deg == pytest.approx(math.hypot(s1, c1), rel=1e-3)     # quadrature
    contrib = both.diagnostics["sigma_contributions_deg"]
    assert set(contrib) == {"p1.x", "p1.y", "p2.x", "p2.y", "angle_deg"}
    # a short pixel baseline is more sensitive to clicks than a long one
    short = estimate_separation(W, H, (950.0, 540.0), (970.0, 540.0), _true_angle_deg((950.0, 540.0), (970.0, 540.0), 2.0))
    assert short.sigma_pct > c1 / 2.0 * 100 * 10


def test_input_validation():
    with pytest.raises(ValueError):
        estimate_separation(W, H, (100.0, 100.0), (100.0, 100.0), 1.0)
    with pytest.raises(ValueError):          # 1 deg between two neighbouring pixels: impossible FOV
        estimate_separation(W, H, (100.0, 100.0), (101.0, 100.0), 1.0)
    with pytest.raises(ValueError):          # 170 deg across a short baseline
        estimate_separation(W, H, (100.0, 100.0), (200.0, 100.0), 170.0)
    with pytest.raises(ValueError):
        estimate_size(W, H, (100.0, 100.0), (200.0, 100.0), 10.0, 1000.0, range_sigma_m=2000.0)
    with pytest.raises(KeyError):
        estimate_hud(W, H, "ZOOM9", {"NAR": 0.7})
    assert estimate_separation(W, H, (-30.0, 100.0), (200.0, 100.0), 1.0).notes[0].startswith("point (-30.0")


# ---------------------------------------------------------------- geodesy and the geo method
def test_enu_geodesy():
    e, n, u = enu_vector((40.0, 10.0, 0.0), (41.0, 10.0, 0.0))
    assert abs(e) < 1e-6 and n == pytest.approx(111_000, rel=0.002)
    assert u == pytest.approx(-n ** 2 / (2 * 6.37e6), rel=0.05)           # drops below the horizon by curvature
    az, el, rng = az_el_range(np.array([1000.0, 1000.0, -1000.0 * math.sqrt(2)]))
    assert az == pytest.approx(45.0) and el == pytest.approx(-45.0) and rng == pytest.approx(2000.0)
    # straight up from a point: pure altitude difference
    assert enu_vector((37.95, 23.72, 0.0), (37.95, 23.72, 8000.0))[2] == pytest.approx(8000.0, rel=1e-9)


def _synthetic_camera(cam, landmarks, fill=0.45):
    """Point a pinhole camera at the landmarks and choose the FOV so they fill `fill` of the half-width.
    Returns (hfov_true, pixel positions)."""
    u = [enu_vector(cam, g) for g in landmarks]
    u = [v / np.linalg.norm(v) for v in u]
    fwd = sum(u)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    xmax = max(abs(float(d @ right) / float(d @ fwd)) for d in u)
    f = fill * W / xmax
    pts = [project_direction(d, W, H, f, fwd, right, down) for d in u]
    assert all(0 <= x <= W and 0 <= y <= H for x, y in pts), pts
    return hfov_from_focal(W, f), pts


CAM = (37.95, 23.72, 8000.0)   # 8 km up over the Saronic Gulf, looking at the coast


def test_geo_two_landmarks_recovers_hfov():
    lms = [(37.85, 23.85, 0.0), (37.80, 23.80, 0.0)]            # 17.8 km and 19.8 km away
    hfov, pts = _synthetic_camera(CAM, lms)
    est = estimate_geo(W, H, CAM, list(zip(pts, lms)))
    assert est.hfov_deg == pytest.approx(hfov, rel=1e-6)
    assert est.label == "DERIVED-FOV(geo)"
    rows = est.diagnostics["landmarks"]
    assert all(10_000 < r["range_m"] < 20_000 and r["elevation_deg"] < -20 for r in rows)
    assert any("Only two landmarks" in n for n in est.notes)
    assert "residual_rms_deg" not in est.diagnostics


def test_geo_three_landmarks_narrow_fov_and_consistency():
    lms = [(37.85, 23.85, 0.0), (37.845, 23.858, 0.0), (37.856, 23.862, 20.0)]   # a few hundred metres apart
    hfov, pts = _synthetic_camera(CAM, lms)
    assert hfov < 8.0
    est = estimate_geo(W, H, CAM, list(zip(pts, lms)), camera_sigma_m=100.0, landmark_sigma_m=20.0)
    assert est.hfov_deg == pytest.approx(hfov, rel=1e-6)
    assert est.diagnostics["residual_rms_deg"] < 1e-5
    assert len(est.diagnostics["pair_residuals_deg"]) == 3
    assert any("Consistency check:" in n for n in est.notes)
    assert est.sigma_deg > estimate_geo(W, H, CAM, list(zip(pts, lms))).sigma_deg   # position sigmas add
    c = est.diagnostics["sigma_contributions_deg"]
    assert "cam.u" in c and "lm0.e" in c and "lm2.y" in c
    # a misidentified landmark (shifted ~700 m east) breaks the single-focal-length fit and is reported
    bad = list(zip(pts, lms))
    bad[2] = (bad[2][0], (37.856, 23.870, 20.0))
    est_bad = estimate_geo(W, H, CAM, bad)
    assert est_bad.diagnostics["residual_rms_deg"] > 10 * est_bad.diagnostics["expected_click_residual_deg"]
    assert any("FAILED" in n for n in est_bad.notes)


def test_geo_wide_fov_four_landmarks():
    lms = [(37.85, 23.85, 0.0), (37.80, 23.80, 0.0), (37.86, 23.82, 50.0), (37.83, 23.88, 0.0)]
    hfov, pts = _synthetic_camera(CAM, lms)
    assert hfov > 15.0
    est = estimate_geo(W, H, CAM, list(zip(pts, lms)))
    assert est.hfov_deg == pytest.approx(hfov, rel=1e-6)
    assert len(est.diagnostics["pair_residuals_deg"]) == 6 and est.diagnostics["residual_rms_deg"] < 1e-5


def test_hud_lookup():
    est = estimate_hud(W, H, "nar", {"NAR": 0.7, "WFOV": 2.5}, sigma_pct=10.0, frame=12)
    assert est.hfov_deg == 0.7 and est.sigma_deg == pytest.approx(0.07) and est.label == "DERIVED-FOV(hud)"
    assert est.inputs["label"] == "nar" and est.inputs["table"]["WFOV"] == 2.5 and est.inputs["frame"] == 12
    assert "HFOV = 0.7 ± 0.07 deg (10.0%) [DERIVED-FOV(hud)]" == est.summary()
    assert Prov.DERIVED_FOV in PROV_DESCRIPTIONS and derived_fov_label("hud") == "DERIVED-FOV(hud)"


def test_estimate_dict_round_trip():
    est = estimate_separation(W, H, (100.0, 100.0), (1800.0, 900.0), 3.2, 0.1, frame=7)
    d = json.loads(json.dumps(est.to_dict()))
    back = FovEstimate.from_dict(d)
    assert back == est and d["label"] == "DERIVED-FOV(separation)" and d["sigma_pct"] == pytest.approx(est.sigma_pct)


# ---------------------------------------------------------------- project and report integration
def test_project_apply_fov_and_round_trip(lossless_project, tmp_path):
    p = lossless_project
    old_hfov, old_src, old_ests = p.kin_params.hfov_deg, p.fov_source, list(p.fov_estimates)
    try:
        assert p.fov_status() == ("manual", None)
        est = estimate_separation(320, 240, (60.0, 80.0), (250.0, 170.0), 1.234, 0.05)
        w = p.apply_fov(est)
        assert w == [] and p.kin_params.hfov_deg == pytest.approx(est.hfov_deg)
        status, src = p.fov_status()
        assert status == "derived" and src["index"] == 0 and src["method"] == "separation"
        fn = str(tmp_path / "proj.json")
        p.save(fn)
        q = Project.load(fn)
        assert q.fov_estimates == [est.to_dict()] and q.fov_source == p.fov_source
        assert q.fov_status()[0] == "derived" and q.kin_params.hfov_deg == pytest.approx(est.hfov_deg)
        # report states the derivation
        md, html_ = build_report(p, {})
        assert "[DERIVED-FOV(separation)]" in md and "derived by the 'separation' method" in md
        assert f"{est.hfov_deg:.4g} ± {est.sigma_deg:.2g} deg" in md and "scales linearly" in md
        assert "DERIVED-FOV(separation)" in html_
        # editing the HFOV by hand afterwards demotes it to user input, the estimate stays on file
        p.kin_params.hfov_deg = est.hfov_deg * 1.5
        assert p.fov_status() == ("manual", None)
        md, _ = build_report(p, {})
        assert "user input, no uncertainty given" in md and "DERIVED-FOV" not in md
        # frame-size mismatch is warned about
        w = p.apply_fov(estimate_hud(1920, 1080, "NAR", {"NAR": 0.7}))
        assert any("1920x1080" in s for s in w) and len(p.fov_estimates) == 2 and p.fov_source["index"] == 1
    finally:
        p.kin_params.hfov_deg, p.fov_source, p.fov_estimates = old_hfov, old_src, old_ests


def test_old_project_files_still_load():
    for name in ("pr34_project.json", "pr35_project.json"):
        fn = os.path.join(os.path.dirname(__file__), "..", "examples", name)
        p = Project.load(fn)
        assert p.fov_estimates == [] and p.fov_source is None and p.fov_status() == ("none", None)
        assert "fov_estimates" in p.to_dict()


# ---------------------------------------------------------------- CLI
def test_cli_fov_subcommands_write_json(tmp_path, capsys):
    out = str(tmp_path / "fov.json")
    th = _true_angle_deg((100.0, 200.0), (900.0, 300.0), 3.0)
    assert main(["fov", "separation", "--width", "1920", "--height", "1080", "--p1", "100,200", "--p2", "900,300",
                 "--angle", f"{th:.10f}", "--angle-sigma", "0.1", "--save", out]) == 0
    assert "HFOV = 3 ±" in capsys.readouterr().out
    lms = [(37.85, 23.85, 0.0), (37.80, 23.80, 0.0)]
    hfov, pts = _synthetic_camera(CAM, lms)
    args = ["fov", "geo", "--width", "1920", "--height", "1080", "--camera", "37.95,23.72,8000m", "--save", out,
            "--landmark-sigma", "50m"]
    for (x, y), (la, lo, al) in zip(pts, lms):
        args += ["--landmark", f"{x:.3f},{y:.3f}={la},{lo},{al}"]
    assert main(args) == 0
    assert main(["fov", "size", "--width", "1920", "--height", "1080", "--p1", "900,500", "--p2", "1000,560",
                 "--size", "15m", "--range", "5km", "--range-sigma", "1km", "--save", out]) == 0
    assert main(["fov", "hud", "--width", "1920", "--height", "1080", "--label", "NAR", "--table", "NAR=0.7,WFOV=2.5",
                 "--save", out]) == 0
    ests = load_fov_json(out)
    assert [e["method"] for e in ests] == ["separation", "geo", "size", "hud"]
    assert ests[0]["hfov_deg"] == pytest.approx(3.0, rel=1e-6)
    assert ests[1]["hfov_deg"] == pytest.approx(hfov, rel=1e-4) and ests[1]["inputs"]["landmark_sigma_m"] == 50.0
    assert ests[2]["inputs"]["range_sigma_m"] == 1000.0 and ests[3]["hfov_deg"] == 0.7
    # bad input is a clean error, not a traceback
    assert main(["fov", "separation", "--width", "1920", "--height", "1080", "--p1", "100,200", "--p2", "101,200",
                 "--angle", "5"]) == 2


def test_cli_analyze_consumes_fov_json(lossless_clip, tmp_path):
    path, truth = lossless_clip
    x, y = truth["object_screen_px"][0]
    fov_json = str(tmp_path / "fov.json")
    th = math.degrees(angle_between_points((60.0, 80.0), (250.0, 170.0), 320, 240, focal_px(320, 2.0)))
    assert main(["fov", "separation", "--width", "320", "--height", "240", "--p1", "60,80", "--p2", "250,170",
                 "--angle", f"{th:.10f}", "--angle-sigma", "0.1", "--save", fov_json]) == 0
    base = ["analyze", path, "--roi", f"{x + 1:.1f},{y - 1:.1f},11,11", "--frame", "0", "--masks-json",
            path + ".truth.json", "--static-camera", "--no-video", "--fov-json", fov_json]
    out1 = str(tmp_path / "derived")
    assert main(base + ["--out", out1]) == 0
    with open(os.path.join(out1, "project.json")) as f:
        d = json.load(f)
    assert d["kin_params"]["hfov_deg"] == pytest.approx(2.0, rel=1e-6)
    assert d["fov_source"]["method"] == "separation" and d["fov_source"]["index"] == 0 and len(d["fov_estimates"]) == 1
    report = open(os.path.join(out1, "report.md"), encoding="utf-8").read()
    e = load_fov_json(fov_json)[0]
    assert "[DERIVED-FOV(separation)]" in report
    assert f"{e['hfov_deg']:.4g} ± {e['sigma_deg']:.2g} deg ({e['sigma_pct']:.1f}%)" in report
    assert "derived by the 'separation' method" in report and "scales linearly" in report
    # --hfov wins over --fov-json and is reported as user input; the estimate is kept on file
    out2 = str(tmp_path / "manual")
    assert main(base + ["--out", out2, "--hfov", "3.0"]) == 0
    with open(os.path.join(out2, "project.json")) as f:
        d = json.load(f)
    assert d["kin_params"]["hfov_deg"] == 3.0 and d["fov_source"] is None and len(d["fov_estimates"]) == 1
    report = open(os.path.join(out2, "report.md"), encoding="utf-8").read()
    assert "3 deg (user input, no uncertainty given)" in report and "DERIVED-FOV" not in report


# ---------------------------------------------------------------- active image region (inset)
def test_region_translates_points_and_reports_inset_hfov():
    """A 1280x720 file whose live image is the 562-px-wide inset at x=360: the HFOV is the inset's."""
    reg = (360.0, 40.0, 562.0, 640.0)
    w, h = int(reg[2]), int(reg[3])
    p1, p2 = (80.0, 100.0), (500.0, 560.0)                      # inset coordinates
    th = math.degrees(angle_between_points(p1, p2, w, h, focal_px(w, 20.0)))
    plain = estimate_separation(w, h, p1, p2, th, 0.1)
    full = estimate_separation(1280, 720, (p1[0] + reg[0], p1[1] + reg[1]), (p2[0] + reg[0], p2[1] + reg[1]), th, 0.1,
                               region=reg)
    assert full.hfov_deg == pytest.approx(plain.hfov_deg, rel=1e-9) and full.sigma_deg == pytest.approx(plain.sigma_deg)
    assert full.hfov_deg == pytest.approx(20.0, rel=1e-6)
    assert full.region == list(reg) and full.inputs["region"] == list(reg) and full.width == 1280
    assert full.solve_width == 562.0 and full.focal_px == pytest.approx(focal_px(562, 20.0))
    assert any("Active image region" in n and "2.28x" in n for n in full.notes)
    # without the region the same clicks give a wrong (full-frame) answer
    wrong = estimate_separation(1280, 720, (p1[0] + reg[0], p1[1] + reg[1]), (p2[0] + reg[0], p2[1] + reg[1]), th)
    assert wrong.hfov_deg != pytest.approx(20.0, rel=0.05)
    # region=(whole frame) is a no-op; serialisation keeps the region; the report describes it
    same = estimate_separation(w, h, p1, p2, th, 0.1, region=(0, 0, w, h))
    assert same.hfov_deg == plain.hfov_deg and same.region == [0.0, 0.0, float(w), float(h)]
    back = FovEstimate.from_dict(json.loads(json.dumps(full.to_dict())))
    assert back == full and "562x640 px inset" in describe_inputs(full.to_dict())
    # the other methods take it too
    lms = [(37.85, 23.85, 0.0), (37.80, 23.80, 0.0), (37.86, 23.82, 50.0)]
    hfov, pts = _synthetic_camera(CAM, lms)
    g = estimate_geo(W + 200, H + 100, CAM, [((x + 100.0, y + 50.0), lm) for (x, y), lm in zip(pts, lms)],
                     region=(100, 50, W, H))
    assert g.hfov_deg == pytest.approx(hfov, rel=1e-6) and g.region == [100.0, 50.0, float(W), float(H)]
    sz = estimate_size(1280, 720, (p1[0] + reg[0], p1[1] + reg[1]), (p2[0] + reg[0], p2[1] + reg[1]), 10.0, 1000.0,
                       region=reg)
    assert sz.region == list(reg) and sz.diagnostics["focal_px"] == pytest.approx(focal_px(562, sz.hfov_deg))
    hud = estimate_hud(1280, 720, "NAR", {"NAR": 0.7}, region=reg)
    assert hud.region == list(reg) and hud.focal_px == pytest.approx(focal_px(562, 0.7))
    # a point outside the inset is noted; a region that does not fit is an error
    assert any("outside the active image region" in n
               for n in estimate_separation(1280, 720, (100.0, 100.0), (700.0, 400.0), 5.0, region=reg).notes)
    with pytest.raises(ValueError):
        estimate_separation(1280, 720, p1, p2, th, region=(900, 0, 562, 720))
    with pytest.raises(ValueError):
        estimate_separation(1280, 720, p1, p2, th, region=(0, 0, 562))


def _texture_frames(n, h, w, rng):
    """Sharp, moving texture (a panning random field) as a list of uint8 frames."""
    big = rng.integers(0, 255, (h + 8, w + 4 * n + 8), dtype=np.uint8)
    return [big[4:4 + h, 4 + 4 * k:4 + 4 * k + w].copy() for k in range(n)]


def test_detect_inset_finds_live_region():
    rng = np.random.default_rng(5)
    live = _texture_frames(12, 200, 300, rng)
    assert detect_inset(live) is None                                     # the whole frame is live
    panel = cv2.GaussianBlur(rng.integers(0, 255, (260, 500), dtype=np.uint8), (0, 0), 5)   # blurred, static
    frames = []
    for f in live:
        g = panel.copy()
        g[30:230, 120:420] = f
        frames.append(g)
    assert detect_inset(frames) == (120, 30, 300, 200)
    # sharp but STATIC side panels (a logo, a data block) are not live either
    panel2 = rng.integers(0, 255, (260, 500), dtype=np.uint8)
    frames2 = []
    for f in live:
        g = panel2.copy()
        g[30:230, 120:420] = f
        frames2.append(g)
    assert detect_inset(frames2) == (120, 30, 300, 200)
    # black panels, as in redacted releases
    frames3 = [np.pad(f, ((0, 0), (64, 136))) for f in live]
    assert detect_inset(frames3) == (64, 0, 300, 200)
    assert detect_inset([]) is None


def _padded_clip(path, out, x0, y0, cw, ch):
    cap = cv2.VideoCapture(path)
    w = None
    n = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if w is None:
            w = lossless_writer(out, 30.0, (cw, ch))
        g = np.zeros((ch, cw, 3), np.uint8)
        g[y0:y0 + f.shape[0], x0:x0 + f.shape[1]] = f
        w.write(g)
        n += 1
    w.release()
    return n


def _padded_project(path, truth, x0, y0, with_camera, extra_masks=()):
    q = Project()
    q.kin_params.hfov_deg = truth["config"]["hfov_deg"]
    q.kin_params.image_region = [x0, y0, 320, 240]
    q.masks = [Mask(m["x"] + x0, m["y"] + y0, m["w"], m["h"], m["kind"]) for m in truth["hud_masks"]] + list(extra_masks)
    q.open_video(path)
    x, y = truth["object_screen_px"][0]
    q.track.keyframes[0] = (x + 1.0 + x0, y - 1.0 + y0, 11, 11)
    q.run_all(with_camera=with_camera)
    return q


def test_kinematics_image_region(lossless_project, lossless_clip, tmp_path):
    p = lossless_project
    kp = p.kin_params
    assert kp.image_region is None and kp.principal_point(320, 240) == (160.0, 120.0) and kp.fov_width(320) == 320.0
    # region = whole frame: bit-identical to no region
    whole = dataclasses.replace(kp, image_region=(0, 0, 320, 240))
    assert whole.image_region == [0.0, 0.0, 320.0, 240.0] and whole.focal_px(320) == pytest.approx(focal_px(320, kp.hfov_deg))
    k = compute_kinematics(p.track, p.camera, p.frame_times(), p.cache.duplicate, p.cache[0].shape, whole,
                           frame_offset=p.cache.offset, blank=p.cache.blank)
    for name in ("los_x", "los_y", "omega", "alpha", "angular_size"):
        assert np.array_equal(k.get(name), p.kin.get(name), equal_nan=True), name
    with pytest.raises(ValueError):
        KinParams(image_region=(0, 0, 320))
    # the same frames pasted into a wider black canvas at x = 80, analysed with the region set
    path, truth = lossless_clip
    x0, cw = 80, 480
    padded = str(tmp_path / "padded.avi")
    assert _padded_clip(path, padded, x0, 0, cw, 240) == len(p.cache)
    # static camera on both: tracking and the pixel->angle conversion must be identical, so the rates are too
    p0 = Project()
    p0.kin_params.hfov_deg = truth["config"]["hfov_deg"]
    p0.masks = [Mask.from_dict(d) for d in truth["hud_masks"]]
    p0.open_video(path)
    p0.track.keyframes[0] = p.track.keyframes[0]
    p0.run_all(with_camera=False)
    q0 = _padded_project(padded, truth, x0, 0, with_camera=False)
    assert q0.kin_params.image_region == [80.0, 0.0, 320.0, 240.0]
    inset = detect_inset(q0.cache.frames)
    assert inset is not None and inset[0] == x0 and inset[2] == 320          # HUD bands may trim rows, never columns
    for name in ("omega", "omega_x", "omega_y", "los_x", "los_y"):
        a, b = p0.kin.get(name), q0.kin.get(name)
        ok = p0.kin.valid & q0.kin.valid & np.isfinite(a) & np.isfinite(b)
        assert ok.sum() > 50 and np.allclose(a[ok], b[ok], rtol=1e-9, atol=1e-12), name
    assert abs(np.nanmedian(q0.kin.get("omega")[ok]) / np.nanmedian(p0.kin.get("omega")[ok]) - 1) < 0.005
    # full pipeline with camera compensation and the side panels masked (as an analyst would): the registration on
    # a different canvas is not bit-identical, so the median rate is only required to agree closely, not exactly
    panels = [Mask(0, 0, x0, 240, "REDACTION"), Mask(x0 + 320, 0, cw - x0 - 320, 240, "REDACTION")]
    q = _padded_project(padded, truth, x0, 0, with_camera=True, extra_masks=panels)
    a, b = p.kin.get("omega"), q.kin.get("omega")
    ok = p.kin.trusted & q.kin.trusted & np.isfinite(a) & np.isfinite(b)
    assert ok.sum() > 40
    assert abs(np.nanmedian(b[ok]) / np.nanmedian(a[ok]) - 1) < 0.03
    # ... and without the region the same HFOV would be spread over the 480-px canvas: every angle 320/480 too small
    bad = dataclasses.replace(q.kin_params, image_region=None)
    kb = compute_kinematics(q.track, q.camera, q.frame_times(), q.cache.duplicate, q.cache[0].shape, bad,
                            frame_offset=q.cache.offset, blank=q.cache.blank)
    assert np.nanmedian(kb.get("omega")[ok]) == pytest.approx(np.nanmedian(b[ok]) * 320 / 480, rel=0.02)
    # apply_fov copies the region into kin_params (and clears it for an estimate without one); it survives a save
    old = (q.kin_params.hfov_deg, q.kin_params.image_region, q.fov_source, list(q.fov_estimates))
    est = estimate_hud(cw, 240, "NAR", {"NAR": 2.0}, region=(x0, 0, 320, 240))
    assert q.apply_fov(est) == [] and q.kin_params.image_region == [80.0, 0.0, 320.0, 240.0]
    fn = str(tmp_path / "padded.json")
    q.save(fn)
    assert Project.load(fn).kin_params.image_region == [80.0, 0.0, 320.0, 240.0]
    q.apply_fov(estimate_hud(cw, 240, "NAR", {"NAR": 2.0}))
    assert q.kin_params.image_region is None


# ---------------------------------------------------------------- ground-plane resection
GROUND_ORIGIN = (37.95, 23.72, 0.0)


def _ground_scene(w, h, hfov, look_down_deg, height_m=1500.0, roll_deg=0.0, par=1.0, pts=None):
    """A pinhole camera `height_m` above a flat area, looking north and down by look_down_deg, rolled by roll_deg.
    Rays through chosen pixels are intersected with the ground (ENU z = 0 at GROUND_ORIGIN) and converted to
    geodetic coordinates. Returns (points, camera (lat, lon, alt), true m/px at the image centre, slant range)."""
    f = focal_px(w, hfov)
    ld, rl = math.radians(look_down_deg), math.radians(roll_deg)
    fwd = np.array([0.0, math.cos(ld), -math.sin(ld)])
    right0 = np.array([1.0, 0.0, 0.0])
    down0 = np.cross(fwd, right0)
    right = math.cos(rl) * right0 + math.sin(rl) * down0
    down = np.cross(fwd, right)
    C = np.array([0.0, -height_m / math.tan(ld) if look_down_deg < 90 else 0.0, height_m])

    def ground(x, y):
        d = (x - w / 2) * right + ((y - h / 2) / par) * down + f * fwd
        t = -C[2] / d[2]
        assert t > 0, "ray does not hit the ground"
        return C + t * d
    pts = pts or [(150.0, 120.0), (1100.0, 150.0), (560.0, 330.0), (200.0, 600.0), (1050.0, 620.0), (700.0, 250.0)]
    points = [((float(x), float(y)), enu_to_geodetic(ground(x, y), GROUND_ORIGIN)) for x, y in pts]
    jx = ground(w / 2 + 0.5, h / 2) - ground(w / 2 - 0.5, h / 2)
    jy = ground(w / 2, h / 2 + 0.5) - ground(w / 2, h / 2 - 0.5)
    gsd = math.sqrt(abs(jx[0] * jy[1] - jx[1] * jy[0]))
    return points, enu_to_geodetic(C, GROUND_ORIGIN), gsd, float(np.linalg.norm(ground(w / 2, h / 2) - C))


def _cam_error_m(est, cam):
    d = est.diagnostics
    return float(np.linalg.norm(enu_vector(cam, (d["camera_lat"], d["camera_lon"], d["camera_alt_m"]))))


def test_geodetic_inverse_and_homography():
    for geo in [(37.95, 23.72, 8000.0), (-33.9, 151.2, 12.0), (0.0, 0.0, 0.0), (89.0, -120.0, 300.0)]:
        la, lo, al = ecef_to_geodetic(geodetic_to_ecef(*geo))
        assert la == pytest.approx(geo[0], abs=1e-9) and lo == pytest.approx(geo[1], abs=1e-9)
        assert al == pytest.approx(geo[2], abs=1e-6)
    assert enu_to_geodetic(np.array([0.0, 0.0, 100.0]), (37.95, 23.72, 0.0)) == pytest.approx((37.95, 23.72, 100.0))
    rng = np.random.default_rng(1)
    Ht = np.array([[1.2, 0.1, 30.0], [-0.2, 0.9, -10.0], [1e-3, -2e-3, 1.0]])
    src = rng.uniform(-100, 100, (6, 2))
    q = (Ht @ np.column_stack([src, np.ones(6)]).T).T
    dst = q[:, :2] / q[:, 2:3]
    H = fit_homography(src, dst)
    assert np.allclose(H / H[2, 2], Ht / Ht[2, 2], atol=1e-9)
    with pytest.raises(ValueError):
        fit_homography(np.array([[0, 0], [1, 1], [2, 2], [3, 3.0]]), dst[:4])      # collinear


@pytest.mark.parametrize("look_down,roll", [(45.0, 0.0), (60.0, 20.0), (50.0, -35.0)])
def test_ground_recovers_fov_camera_and_scale(look_down, roll):
    w, h, hfov = 1280, 720, 30.0
    points, cam, gsd, rng = _ground_scene(w, h, hfov, look_down, roll_deg=roll)
    est = estimate_ground(w, h, points, click_sigma_px=1.0)
    d = est.diagnostics
    assert est.method == "ground" and est.label == "DERIVED-FOV(ground)"
    assert est.hfov_deg == pytest.approx(hfov, rel=1e-3) and d["hfov_determined"]
    assert _cam_error_m(est, cam) < 0.005 * rng
    assert d["slant_range_m"] == pytest.approx(rng, rel=0.005)
    assert d["ground_sample_m_per_px"] == pytest.approx(gsd, rel=0.005)
    assert d["elevation_deg"] == pytest.approx(-look_down, abs=0.05)
    assert min(d["azimuth_deg"], 360.0 - d["azimuth_deg"]) < 0.05
    assert d["camera_height_above_plane_m"] == pytest.approx(1500.0, rel=0.005)
    assert d["reprojection_rms_px"] < 1e-6 and d["plane_fit_rms_m"] < 1e-6
    assert d["aimpoint_lat"] == pytest.approx(GROUND_ORIGIN[0], abs=1e-6)
    assert any("Consistency check:" in n for n in est.notes) and not any("POORLY" in n for n in est.notes)
    # the two absolute-conic equations (compared in the equal-weight plane frame) agree on f exactly
    assert d["iac_disagreement_pct"] < 1e-3 and any("agree on f" in n for n in est.notes)
    assert d["iac_focal_px_eq1"] == pytest.approx(focal_px(w, hfov), rel=1e-4)
    assert est.sigma_deg > 0 and d["ground_sample_sigma_m_per_px"] > 0 and d["slant_range_sigma_m"] > 0
    assert set(d["sigma_contributions_deg"]) == {f"p{i}.{c}" for i in range(6) for c in "xy"}
    json.dumps(est.to_dict())
    assert "6 ground points" in describe_inputs(est.to_dict()) and "camera position unknown" in describe_inputs(est.to_dict())


def test_ground_noise_and_sigma_behaviour():
    w, h = 1280, 720
    points, cam, gsd, rng = _ground_scene(w, h, 30.0, 55.0, roll_deg=10.0)
    e1 = estimate_ground(w, h, points, click_sigma_px=1.0)
    e2 = estimate_ground(w, h, points, click_sigma_px=2.0)
    assert e2.sigma_deg == pytest.approx(2 * e1.sigma_deg, rel=0.05) and e2.sigma_deg > e1.sigma_deg
    assert e2.diagnostics["ground_sample_sigma_m_per_px"] > e1.diagnostics["ground_sample_sigma_m_per_px"]
    e3 = estimate_ground(w, h, points, point_sigma_m=3.0, click_sigma_px=0.0)
    assert e3.sigma_deg > 0 and "p0.e" in e3.diagnostics["sigma_contributions_deg"]
    assert any("click uncertainty only" in n for n in e1.notes) and not any("click uncertainty only" in n for n in e3.notes)
    # jittered clicks: the answer stays within its own error bar
    r = np.random.default_rng(7)
    noisy = [((x + r.normal(0, 0.7), y + r.normal(0, 0.7)), g) for (x, y), g in points]
    en = estimate_ground(w, h, noisy, click_sigma_px=0.7)
    assert abs(en.hfov_deg - 30.0) < 3 * en.sigma_deg and en.sigma_pct < 5.0
    assert _cam_error_m(en, cam) < 0.03 * rng
    assert en.diagnostics["ground_sample_m_per_px"] == pytest.approx(gsd, rel=0.01)


def test_ground_nadir_is_degenerate_but_scale_is_not():
    w, h = 1280, 720
    points, cam, gsd, rng = _ground_scene(w, h, 30.0, 90.0)
    est = estimate_ground(w, h, points)
    d = est.diagnostics
    assert any("FOCAL LENGTH POORLY DETERMINED" in n and "expected physics" in n for n in est.notes)
    assert not d["hfov_determined"] and d["perspective_depth_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert est.sigma_pct >= 100.0                     # unusable, and says so numerically too
    assert d["ground_sample_m_per_px"] == pytest.approx(gsd, rel=0.01) and d["ground_sample_sigma_m_per_px"] > 0
    assert d["ground_sample_sigma_m_per_px"] < 0.01 * gsd
    # 80 deg down: f is determined but only weakly; the note says so and the numbers are still right
    points, cam, gsd, rng = _ground_scene(w, h, 30.0, 80.0)
    est = estimate_ground(w, h, points)
    assert est.hfov_deg == pytest.approx(30.0, rel=1e-3) and est.sigma_pct > 5.0
    assert any("weakly determined" in n for n in est.notes) and not any("POORLY" in n for n in est.notes)
    assert est.diagnostics["ground_sample_m_per_px"] == pytest.approx(gsd, rel=0.005)


def test_ground_consistency_region_and_validation():
    w, h = 1280, 720
    points, cam, gsd, rng = _ground_scene(w, h, 30.0, 50.0, roll_deg=15.0)
    # one point 20 m off (misidentified): the fit no longer closes and the check fails
    bad = list(points)
    (x, y), g = bad[2]
    enu = enu_vector(GROUND_ORIGIN, g) + np.array([20.0, 0.0, 0.0])
    bad[2] = ((x, y), enu_to_geodetic(enu, GROUND_ORIGIN))
    eb = estimate_ground(w, h, bad)
    assert eb.diagnostics["reprojection_rms_px"] > 3.0 and any("FAILED" in n for n in eb.notes)
    assert max(eb.diagnostics["point_residuals_px"]) == eb.diagnostics["point_residuals_px"][2] or \
        eb.diagnostics["reprojection_rms_m"] > 5.0
    # four points: exact fit, no check
    e4 = estimate_ground(w, h, [points[0], points[1], points[3], points[4]])
    assert e4.hfov_deg == pytest.approx(30.0, rel=1e-3) and any("Only four points" in n for n in e4.notes)
    assert e4.diagnostics["reprojection_rms_px"] < 1e-6
    # three of four collinear in the image (hence on the ground): no homography
    col3 = _ground_scene(w, h, 30.0, 50.0, pts=[(1100.0, 150.0), (640.0, 380.0), (200.0, 600.0), (150.0, 120.0)])[0]
    with pytest.raises(ValueError):
        estimate_ground(w, h, col3)
    # the same clicks inside a 1920x1080 frame with the sensor image inset at (320, 180)
    inset = [((x + 320.0, y + 180.0), g) for (x, y), g in points]
    er = estimate_ground(1920, 1080, inset, region=(320, 180, w, h))
    e0 = estimate_ground(w, h, points)
    assert er.hfov_deg == pytest.approx(e0.hfov_deg, rel=1e-9) and er.region == [320.0, 180.0, 1280.0, 720.0]
    assert er.diagnostics["camera_alt_m"] == pytest.approx(e0.diagnostics["camera_alt_m"], abs=1e-6)
    assert er.diagnostics["ground_sample_m_per_px"] == pytest.approx(gsd, rel=0.005)
    assert er.inputs["points"][0]["px"] == [points[0][0][0] + 320.0, points[0][0][1] + 180.0]
    # pixel aspect ratio: y scaled
    par = 1.2
    pts_par, cam_par, gsd_par, _ = _ground_scene(w, h, 30.0, 50.0, roll_deg=15.0, par=par)
    ep = estimate_ground(w, h, pts_par, pixel_aspect=par)
    assert ep.hfov_deg == pytest.approx(30.0, rel=1e-3) and _cam_error_m(ep, cam_par) < 1.0
    assert ep.diagnostics["ground_sample_m_per_px"] == pytest.approx(gsd_par, rel=0.005)
    # validation
    with pytest.raises(ValueError):
        estimate_ground(w, h, points[:3])
    with pytest.raises(ValueError):
        estimate_ground(w, h, [points[0], points[0], points[1], points[2]])
    col = _ground_scene(w, h, 30.0, 50.0, pts=[(100.0, 300.0), (400.0, 330.0), (700.0, 335.0), (1000.0, 370.0),
                                               (1200.0, 380.0)])[0]
    assert any("nearly collinear" in n for n in estimate_ground(w, h, col).notes)


def test_cli_fov_ground_and_region(tmp_path, capsys):
    out = str(tmp_path / "fov.json")
    w, h = 1280, 720
    points, cam, gsd, rng = _ground_scene(w, h, 30.0, 50.0, roll_deg=10.0)
    args = ["fov", "ground", "--width", str(w), "--height", str(h), "--point-sigma", "2m", "--save", out]
    for (x, y), (la, lo, al) in points:
        args += ["--point", f"{x:.4f},{y:.4f}={la:.9f},{lo:.9f},{al:.4f}"]
    assert main(args) == 0
    txt = capsys.readouterr().out
    assert "HFOV = 30 ±" in txt and "slant range" in txt and "m/px at the image centre" in txt
    # inset variant on the CLI
    args = ["fov", "ground", "--width", "1920", "--height", "1080", "--region", "320,180,1280,720", "--save", out]
    for (x, y), (la, lo, al) in points:
        args += ["--point", f"{x + 320:.4f},{y + 180:.4f}={la:.9f},{lo:.9f},{al:.4f}"]
    assert main(args) == 0
    assert main(["fov", "separation", "--width", "1920", "--height", "1080", "--region", "320,180,1280,720", "--p1",
                 "420,280", "--p2", "1500,800", "--angle", "25", "--save", out]) == 0
    ests = load_fov_json(out)
    assert [e["method"] for e in ests] == ["ground", "ground", "separation"]
    assert ests[0]["hfov_deg"] == pytest.approx(30.0, rel=1e-3) and ests[0]["inputs"]["point_sigma_m"] == 2.0
    assert ests[0]["region"] is None and ests[1]["region"] == [320.0, 180.0, 1280.0, 720.0]
    assert ests[1]["hfov_deg"] == pytest.approx(ests[0]["hfov_deg"], rel=1e-6)
    d = ests[0]["diagnostics"]
    assert d["slant_range_m"] == pytest.approx(rng, rel=0.005) and d["ground_sample_m_per_px"] == pytest.approx(gsd, rel=0.005)
    assert d["camera_alt_m"] == pytest.approx(cam[2], abs=0.01 * rng)
    assert ests[2]["region"] == [320.0, 180.0, 1280.0, 720.0]
    # too few points / bad region are clean errors
    assert main(["fov", "ground", "--width", str(w), "--height", str(h), "--point", "1,1=37.9,23.7,0",
                 "--point", "2,2=37.91,23.7,0", "--point", "3,1=37.9,23.71,0"]) == 2
    assert main(["fov", "separation", "--width", "1920", "--height", "1080", "--region", "1900,0,1280,720", "--p1",
                 "420,280", "--p2", "1500,800", "--angle", "25"]) == 2
    # applying an inset estimate through analyze carries the region into the project
    p = Project()
    assert p.apply_fov(ests[1]) == [] and p.kin_params.image_region == [320.0, 180.0, 1280.0, 720.0]
