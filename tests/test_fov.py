"""Field-of-view derivation (uapka.fov): pinhole round trips, uncertainty behaviour, geodesy, project and CLI plumbing."""

import json
import math
import os

import numpy as np
import pytest

from uapka.cli import main
from uapka.export import build_report
from uapka.fov import (FovEstimate, angle_between_points, az_el_range, enu_vector, estimate_geo, estimate_hud,
                       estimate_separation, estimate_size, focal_px, hfov_from_focal, load_fov_json,
                       project_direction, small_angle_hfov)
from uapka.project import Project
from uapka.provenance import PROV_DESCRIPTIONS, Prov, derived_fov_label

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
