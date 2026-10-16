from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from edgespread.bars import bars_from_trades, resample_bars
from edgespread.benchmark import daily_benchmark, match_quotes, quoted_spread, trade_side_proxy

OHLC = ["open", "high", "low", "close"]
DAY = pd.Timestamp("2024-01-01", tz="UTC")


def _trades():
    times = pd.to_datetime(["2024-01-01 00:00:05", "2024-01-01 00:00:30", "2024-01-01 00:00:50.1", "2024-01-01 00:00:50.9", "2024-01-01 00:03:10"], utc=True, format="ISO8601")
    return times, [10.0, 11.0, 10.5, 10.8, 12.0]


def test_bars_from_trades_prev_close():
    times, prices = _trades()
    b = bars_from_trades(times, prices, "1min", DAY, DAY + pd.Timedelta(minutes=4), fill="prev_close")
    assert len(b) == 4
    assert b["n_trades"].tolist() == [4, 0, 0, 1]
    assert b.loc[0, OHLC].tolist() == [10.0, 11.0, 10.0, 10.8]
    assert b.loc[1, OHLC].tolist() == [10.8] * 4
    assert b.loc[2, OHLC].tolist() == [10.8] * 4
    assert b.loc[3, OHLC].tolist() == [12.0] * 4


def test_bars_from_trades_prev_bar_and_nan():
    times, prices = _trades()
    b = bars_from_trades(times, prices, "1min", DAY, DAY + pd.Timedelta(minutes=4), fill="prev_bar")
    assert b.loc[1, OHLC].tolist() == b.loc[0, OHLC].tolist()
    b = bars_from_trades(times, prices, "1min", DAY, DAY + pd.Timedelta(minutes=4), fill="nan")
    assert b.loc[1, OHLC].isna().all() and b.loc[2, OHLC].isna().all()
    assert b.loc[3, "close"] == 12.0


def test_bars_from_trades_edge_cases():
    assert bars_from_trades([], [], "1min").empty
    with pytest.raises(ValueError):
        bars_from_trades(*_trades(), "1min", fill="bogus")


def test_resample_bars_matches_direct_aggregation():
    rng = np.random.default_rng(0)
    times = DAY + pd.to_timedelta(np.sort(rng.integers(0, 600_000, 2000)), unit="ms")
    prices = 100 + np.cumsum(rng.normal(0, 0.01, 2000))
    end = DAY + pd.Timedelta(minutes=10)
    fine = bars_from_trades(times, prices, "1s", DAY, end, fill="nan")
    direct = bars_from_trades(times, prices, "1min", DAY, end, fill="nan")
    res = resample_bars(fine, "1min")
    pd.testing.assert_frame_equal(res.reset_index(drop=True), direct.reset_index(drop=True), check_dtype=False)


@pytest.mark.parametrize("fill", ["prev_close", "prev_bar"])
def test_resample_bars_refills_gaps_with_the_same_rule(fill):
    times, prices = _trades()
    end = DAY + pd.Timedelta(minutes=4)
    fine = bars_from_trades(times, prices, "1s", DAY, end, fill=fill)
    res = resample_bars(fine, "1min")
    direct = bars_from_trades(times, prices, "1min", DAY, end, fill=fill)
    assert res["n_trades"].tolist() == [4, 0, 0, 1]
    pd.testing.assert_frame_equal(res.reset_index(drop=True), direct.reset_index(drop=True), check_dtype=False)


def _quotes_and_trades():
    quotes = pd.DataFrame({"time": DAY + pd.to_timedelta([0, 10], unit="s"), "bid": [99.0, 100.0], "ask": [101.0, 102.0]})
    trades = pd.DataFrame({
        "time": DAY + pd.to_timedelta([5, 10, 15], unit="s"),
        "price": [101.0, 102.0, 100.0],
        "qty": [1.0, 2.0, 1.0],
        "is_buyer_maker": [False, False, True],
    })
    return quotes, trades


def test_match_quotes_uses_the_quote_strictly_before_the_trade():
    quotes, trades = _quotes_and_trades()
    m = match_quotes(trades, quotes)
    assert m["mid"].tolist() == [100.0, 100.0, 101.0]
    assert m["direction"].tolist() == [1.0, 1.0, -1.0]
    assert m["eff"].tolist() == pytest.approx([0.02, 0.04, 2.0 / 101.0])
    assert m["lee_ready"].tolist() == [1.0, 1.0, -1.0]


def test_daily_benchmark_summary():
    quotes, trades = _quotes_and_trades()
    out = daily_benchmark(match_quotes(trades, quotes))
    assert out["n_trades"] == 3 and out["n_matched"] == 3
    assert out["eff_mean"] == pytest.approx(np.mean([0.02, 0.04, 2.0 / 101.0]))
    assert out["eff_vw"] == pytest.approx(np.average([0.02, 0.04, 2.0 / 101.0], weights=[101.0, 204.0, 100.0]))
    assert out["share_at_quote"] == pytest.approx(2.0 / 3.0)
    assert out["lee_ready_error"] == 0.0
    assert out["buy_share"] == pytest.approx(2.0 / 3.0)
    assert daily_benchmark(match_quotes(trades.iloc[:0], quotes)) == {"n_trades": 0, "n_matched": 0}


def test_quoted_spread_is_time_weighted_and_skips_crossed_quotes():
    quotes = pd.DataFrame({"time": DAY + pd.to_timedelta([0, 30, 45], unit="s"), "bid": [99.0, 100.0, 101.0], "ask": [101.0, 100.5, 100.0]})
    out = quoted_spread(quotes, DAY, DAY + pd.Timedelta(seconds=60))
    rel = [2.0 * 2.0 / 200.0, 2.0 * 0.5 / 200.5]
    assert out["quoted_tw"] == pytest.approx((30 * rel[0] + 15 * rel[1]) / 45)
    assert out["quoted_eq"] == pytest.approx(np.mean(rel))
    assert out["n_quotes"] == 3
    assert np.isnan(quoted_spread(quotes.iloc[:0], DAY, DAY + pd.Timedelta(seconds=60))["quoted_tw"])


def test_trade_side_proxy_builds_the_midpoint_from_recent_trades():
    trades = pd.DataFrame({
        "time": DAY + pd.to_timedelta([0, 1, 2, 30], unit="s"),
        "price": [101.0, 99.0, 101.0, 101.0],
        "is_buyer_maker": [False, True, False, False],
    })
    out = trade_side_proxy(trades, max_gap=pd.Timedelta(seconds=10))
    assert np.isnan(out.loc[0, "eff_proxy"]) and np.isnan(out.loc[1, "eff_proxy"])
    assert out.loc[2, "mid_proxy"] == 100.0
    assert out.loc[2, "eff_proxy"] == pytest.approx(0.02)
    assert np.isnan(out.loc[3, "eff_proxy"])
