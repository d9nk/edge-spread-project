from __future__ import annotations

import numpy as np
import pytest
from bidask import edge as bidask_edge

from edgespread import METHODS, SimConfig, ar, clip_zero, cs, edge, edge_components, estimate, roll, simulate, spread_from_squared, windows_with_lag

CS_K = 3.0 - 2.0 * np.sqrt(2.0)


def test_authors_vector_full_sample(authors_ohlc):
    df = authors_ohlc
    assert edge(df.Open, df.High, df.Low, df.Close) == pytest.approx(0.0101849034905478, abs=1e-15)


def test_authors_vector_signed_first_ten(authors_ohlc):
    df = authors_ohlc
    assert edge(df.Open[:10], df.High[:10], df.Low[:10], df.Close[:10], sign=True) == pytest.approx(-0.016889917516422, abs=1e-15)


def test_authors_vector_with_missing_values(authors_ohlc_miss):
    df = authors_ohlc_miss
    assert edge(df.Open, df.High, df.Low, df.Close) == pytest.approx(0.01013284969780197, abs=1e-15)


def test_no_price_movement_is_nan():
    assert np.isnan(edge([18.21, 17.61, 17.61], [18.21, 17.61, 17.61], [17.61, 17.61, 17.61], [17.61, 17.61, 17.61]))


@pytest.mark.parametrize("prob", [1.0, 0.1, 0.01])
@pytest.mark.parametrize("sign", [False, True])
def test_matches_authors_package_on_simulated_prices(prob, sign):
    bars = simulate(300, SimConfig(prob=prob, spread=0.02), np.random.default_rng(7))
    o, h, l, c = bars.T
    assert edge(o, h, l, c, sign=sign) == pytest.approx(bidask_edge(o, h, l, c, sign), abs=1e-14)


def test_matches_authors_package_with_missing_values():
    rng = np.random.default_rng(11)
    bars = simulate(400, SimConfig(prob=0.05, spread=0.01), rng)
    o, h, l, c = (x.copy() for x in bars.T)
    for x in (o, h, l, c):
        x[rng.random(400) < 0.05] = np.nan
    assert edge(o, h, l, c, sign=True) == pytest.approx(bidask_edge(o, h, l, c, True), abs=1e-14)


def test_batched_rows_equal_single_series():
    bars = simulate(5 * 60, SimConfig(prob=0.2), np.random.default_rng(3)).reshape(5, 60, 4)
    o, h, l, c = (bars[:, :, j] for j in range(4))
    batched = estimate(o, h, l, c, methods=METHODS, sign=True)
    for i in range(5):
        single = estimate(o[i], h[i], l[i], c[i], methods=METHODS, sign=True)
        for m in METHODS:
            assert batched[m][i] == pytest.approx(single[m], abs=1e-15, nan_ok=True)


def test_sign_handling_and_clipping():
    s2 = np.array([0.0004, -0.0004, 0.0])
    assert np.allclose(spread_from_squared(s2, sign=False), [0.02, 0.02, 0.0])
    assert np.allclose(spread_from_squared(s2, sign=True), [0.02, -0.02, 0.0])
    assert np.allclose(clip_zero(np.array([-0.1, 0.0, 0.3])), [0.0, 0.0, 0.3])


def test_fewer_than_three_observations_is_nan():
    assert np.isnan(edge([1.0, 1.1], [1.2, 1.2], [0.9, 1.0], [1.1, 1.0]))


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError):
        estimate([1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], methods=("EDGE", "FOO"))


def test_estimate_keeps_the_requested_order():
    o, h, l, c = simulate(100, SimConfig(), np.random.default_rng(0)).T
    assert list(estimate(o, h, l, c, methods=("roll", "EDGE", "cs"))) == ["ROLL", "EDGE", "CS"]


def test_shape_and_dimension_mismatches_are_rejected():
    with pytest.raises(ValueError):
        edge([1, 2, 3], [1, 2], [1, 2, 3], [1, 2, 3])
    with pytest.raises(ValueError):
        edge(np.ones((2, 3)), np.ones(3), np.ones(3), np.ones(3))


def test_edge_is_the_inverse_variance_combination_of_the_two_moments():
    o, h, l, c = simulate(500, SimConfig(prob=0.3), np.random.default_rng(5)).T
    comp = edge_components(o, h, l, c)
    w1, w2, e1, e2, v1, v2 = (float(comp[k][0]) for k in ("w1", "w2", "e1", "e2", "v1", "v2"))
    assert w1 + w2 == pytest.approx(1.0)
    assert w1 == pytest.approx(v2 / (v1 + v2))
    assert float(comp["EDGE"][0]) == pytest.approx(w1 * e1 + w2 * e2)
    assert float(comp["GMM"][0]) == pytest.approx(0.5 * (e1 + e2))
    s = edge(o, h, l, c, sign=True)
    assert float(comp["EDGE"][0]) == pytest.approx(np.sign(s) * s * s)


def test_roll_recovers_the_spread_from_a_pure_bounce():
    rng = np.random.default_rng(2)
    side = np.where(rng.random(200_000) < 0.5, 1.0, -1.0)
    close = 100.0 * (1.0 + 0.005 * side)
    assert roll(close) == pytest.approx(0.01, rel=0.02)


def test_ar_formula_on_a_small_example():
    o = np.array([10.0, 10.2, 10.1, 10.3, 10.0, 10.2])
    h, l, c = o + 0.3, o - 0.2, o + 0.1
    lh, ll, lc = (np.log(x) for x in (h, l, c))
    eta = (lh + ll) / 2
    s2 = np.mean(4.0 * (lc[:-1] - eta[:-1]) * (lc[:-1] - eta[1:]))
    assert ar(h, l, c, sign=True) == pytest.approx(np.sign(s2) * np.sqrt(abs(s2)))


def test_cs_formula_on_a_small_example():
    o = np.array([10.0, 10.2, 10.1, 10.3, 10.0, 10.2])
    h, l, c = o + 0.3, o - 0.2, o + 0.1
    lh, ll, lc = (np.log(x) for x in (h, l, c))
    h1, l1, c1, h2, l2 = lh[:-1], ll[:-1], lc[:-1], lh[1:], ll[1:]
    gap = np.maximum(0.0, c1 - h2) + np.minimum(0.0, c1 - l2)
    ah, al = h2 + gap, l2 + gap
    beta = (h2 - l2) ** 2 + (h1 - l1) ** 2
    gamma = (np.maximum(ah, h1) - np.minimum(al, l1)) ** 2
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / CS_K - np.sqrt(gamma / CS_K)
    s_t = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    assert cs(h, l, c, sign=True) == pytest.approx(np.mean(s_t))
    assert cs(h, l, c, two_period=True) == pytest.approx(np.mean(np.maximum(0.0, s_t)))


def test_roll_formula_on_a_small_example():
    c = np.array([10.0, 10.2, 10.1, 10.3, 10.0, 10.2, 10.1])
    r = np.diff(np.log(c))
    cov = np.cov(r[1:], r[:-1], ddof=1)[0, 1]
    s2 = -4.0 * cov
    assert roll(c, sign=True) == pytest.approx(np.sign(s2) * np.sqrt(abs(s2)))


def test_table2_panel_a_regression():
    bars = simulate(2000 * 21, SimConfig(prob=1.0, spread=0.01), np.random.default_rng(20261012))
    o, h, l, c = windows_with_lag(bars, 21)
    est = estimate(o, h, l, c, methods=("EDGE", "AR", "CS"), sign=True)
    means = {m: 100 * np.nanmean(clip_zero(v)) for m, v in est.items()}
    assert means["EDGE"] == pytest.approx(0.90, abs=0.04)
    assert means["AR"] == pytest.approx(0.95, abs=0.08)
    assert means["CS"] == pytest.approx(1.03, abs=0.06)


def test_table2_panel_b_ordering():
    bars = simulate(2000 * 21, SimConfig(prob=0.01, spread=0.01), np.random.default_rng(1))
    o, h, l, c = windows_with_lag(bars, 21)
    est = estimate(o, h, l, c, methods=("EDGE", "AR", "CS"), sign=True)
    means = {m: 100 * np.nanmean(clip_zero(v)) for m, v in est.items()}
    assert means["CS"] < 0.2
    assert 0.6 < means["AR"] < 0.95
    assert means["EDGE"] == pytest.approx(0.95, abs=0.1)
