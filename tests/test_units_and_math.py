import numpy as np
import pytest

from uapka.kinematics import local_poly, random_walk_sigma
from uapka.units import parse_length, parse_length_list


def test_parse_length_units():
    assert parse_length("10km") == 10000
    assert parse_length("1 nmi") == 1852
    assert parse_length("30000ft") == pytest.approx(9144)
    assert parse_length("25kft") == pytest.approx(7620)
    assert parse_length("800") == 800
    with pytest.raises(ValueError):
        parse_length("10 parsecs")


def test_parse_lists_and_sweeps():
    assert parse_length_list("1km, 2km; 3km") == [1000, 2000, 3000]
    assert parse_length_list("2km:10km:2km") == [2000, 4000, 6000, 8000, 10000]
    lg = parse_length_list("log:1km:100km:3")
    assert lg == pytest.approx([1000, 10000, 100000])


def test_local_poly_exact_on_quadratic():
    t = np.arange(60) / 30.0
    y = 0.5 + 2.0 * t - 3.0 * t**2
    out = local_poly(t, y, np.full(60, 1e-3), np.ones(60, bool), 0.25, 2, min_side=3)
    ok = np.isfinite(out["d1"])
    assert ok.sum() > 40
    assert np.allclose(out["d1"][ok], 2.0 - 6.0 * t[ok], atol=1e-6)
    assert np.allclose(out["d2"][ok], -6.0, atol=1e-5)


def test_local_poly_edges_are_not_extrapolated():
    t = np.arange(30) / 30.0
    out = local_poly(t, t.copy(), np.full(30, 0.01), np.ones(30, bool), 7 / 30, 2, min_side=3)
    assert np.isnan(out["d1"][:3]).all() and np.isnan(out["d1"][-3:]).all()


def test_local_poly_sigma_is_calibrated():
    """Monte Carlo: predicted 1-sigma of the derivatives matches their actual scatter."""
    rng = np.random.default_rng(0)
    t = np.arange(200) / 30.0
    sig = 0.5
    d1, d2 = [], []
    for _ in range(200):
        y = 3.0 * t + rng.normal(0, sig, t.size)
        out = local_poly(t, y, np.full(t.size, sig), np.ones(t.size, bool), 7 / 30, 2, min_side=3)
        d1.append(out["d1"][100])
        d2.append(out["d2"][100])
        pred1, pred2 = out["s_d1"][100], out["s_d2"][100]
    assert np.std(d1) == pytest.approx(pred1, rel=0.25)
    assert np.std(d2) == pytest.approx(pred2, rel=0.25)


def test_random_walk_sigma_matches_simulation():
    """Chained per-step errors behave as white velocity noise; check the analytic sigma."""
    rng = np.random.default_rng(1)
    n, s_step = 300, 0.2
    t = np.arange(n) / 30.0
    s1_pred, _ = random_walk_sigma(t, np.full(n, s_step), np.ones(n, bool), 7 / 30)
    est = []
    for _ in range(300):
        y = np.cumsum(rng.normal(0, s_step, n))
        out = local_poly(t, y, np.full(n, 1.0), np.ones(n, bool), 7 / 30, 2, min_side=3)
        est.append(out["d1"][150])
    assert np.std(est) == pytest.approx(s1_pred[150], rel=0.3)
