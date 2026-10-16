from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from edgespread import SimConfig, simulate, windows_with_lag


def test_bars_are_consistent():
    bars = simulate(1000, SimConfig(prob=0.3), np.random.default_rng(0))
    o, h, l, c = bars.T
    assert bars.shape == (1000, 4)
    assert np.isfinite(bars).all()
    assert (l <= o).all() and (o <= h).all() and (l <= c).all() and (c <= h).all()


def test_same_seed_gives_the_same_path():
    a = simulate(300, SimConfig(prob=0.5), np.random.default_rng(42))
    b = simulate(300, SimConfig(prob=0.5), np.random.default_rng(42))
    assert np.array_equal(a, b)


def test_every_trade_observed_has_no_flat_periods():
    bars = simulate(500, SimConfig(prob=1.0), np.random.default_rng(1))
    assert (bars[:, 1] > bars[:, 2]).all()


def test_fill_rules_differ_only_in_empty_periods():
    cfg = SimConfig(trades=20, prob=0.1, spread=0.01)
    a = simulate(3000, cfg, np.random.default_rng(9))
    b = simulate(3000, replace(cfg, fill="prev_bar"), np.random.default_rng(9))
    differ = np.where(np.any(a != b, axis=1))[0]
    assert len(differ) > 0
    for i in differ:
        assert i > 0
        assert np.all(a[i] == a[i - 1, 3])
        assert np.array_equal(b[i], b[i - 1])


def test_reset_period_restarts_the_price():
    cfg = SimConfig(trades=50, volatility=0.5)
    rng_reset, rng_free = np.random.default_rng(4), np.random.default_rng(4)
    with_reset = simulate(100, replace(cfg, reset_period=10), rng_reset)
    without = simulate(100, cfg, rng_free)
    assert np.abs(np.log(with_reset[::10, 0])).max() < 0.5
    assert np.abs(np.log(without[::10, 0])).max() > 0.5


def test_tick_size_rounds_every_price():
    bars = simulate(200, SimConfig(tick_size=0.01, prob=0.5), np.random.default_rng(6))
    assert np.allclose(bars / 0.01, np.round(bars / 0.01), atol=1e-6)


def test_all_buys_without_volatility_sit_at_the_ask():
    bars = simulate(50, SimConfig(volatility=0.0, buy_prob=1.0, spread=0.01), np.random.default_rng(0))
    assert np.allclose(bars, 1.005)


def test_vol_pattern_validation():
    with pytest.raises(ValueError):
        SimConfig(trades=4, vol_pattern=(1.0, 1.0, 1.0))
    with pytest.raises(ValueError):
        SimConfig(trades=3, vol_pattern=(1.0, 2.0, 3.0))
    SimConfig(trades=3, vol_pattern=(0.5, 1.0, 1.5))


@pytest.mark.parametrize("kwargs", [{"prob": 0.0}, {"prob": 1.5}, {"rho": 1.0}, {"trades": 0}, {"fill": "x"}, {"buy_prob": 1.2}, {"spread": -0.01}, {"reset_period": 0}])
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        SimConfig(**kwargs)


def test_windows_with_lag_layout():
    bars = np.arange(40, dtype=float).reshape(10, 4)
    o, h, l, c = windows_with_lag(bars, 5)
    assert o.shape == (2, 6)
    assert np.isnan(o[0, 0]) and np.isnan(c[0, 0])
    assert c[1, 0] == bars[4, 3]
    assert np.array_equal(o[1, 1:], bars[5:, 0])
    with pytest.raises(ValueError):
        windows_with_lag(bars, 11)
    with pytest.raises(ValueError):
        windows_with_lag(bars, 0)


def test_invalid_simulation_arguments():
    with pytest.raises(ValueError):
        simulate(0, SimConfig())
    with pytest.raises(ValueError):
        simulate(10, SimConfig(), chunk=0)
