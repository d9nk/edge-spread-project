from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from edgespread.data_moex import is_share_mask, quoted_spread_by_day, read_quote_log, regular_grid


def test_is_share_mask_keeps_shares_and_drops_funds():
    df = pd.DataFrame({
        "SECID": ["SBER", "SBERP", "X5", "RU000A0JXP78", "XHOUSE", "TMOS", "FXRL", "GAZP"],
        "SHORTNAME": ["Сбербанк", "Сбербанк-п", "ИКС 5", "ЗПИФДОМ.РФ", "ЗПИФ ИНВНД", "TMOS ETF", "FinEx Russia", "ГАЗПРОМ ао"],
        "SECTYPE": ["1", "2", "1", "B", "B", "J", "J", "1"],
    })
    expected = [True, True, True, False, False, False, False, True]
    assert is_share_mask(df).tolist() == expected
    assert is_share_mask(df.drop(columns=["SECTYPE"])).tolist() == expected
    assert is_share_mask(df.drop(columns=["SECTYPE", "SHORTNAME"])).tolist() == [True, True, True, False, False, True, True, True]


def _bars():
    return pd.DataFrame({
        "begin": pd.to_datetime(["2026-10-12 10:00", "2026-10-12 10:01", "2026-10-12 10:04"]),
        "open": [10.0, 11.0, 12.0], "high": [10.5, 11.5, 12.5], "low": [9.5, 10.5, 11.5], "close": [10.2, 11.2, 12.2],
        "volume": [1.0, 2.0, 3.0], "value": [10.0, 20.0, 30.0],
    })


def test_regular_grid_fills_the_gaps():
    g = regular_grid(_bars(), "1min", fill="prev_close")
    assert len(g) == 5 and g["volume"].tolist() == [1.0, 2.0, 0.0, 0.0, 3.0]
    assert g.loc[2, ["open", "high", "low", "close"]].tolist() == [11.2] * 4
    g = regular_grid(_bars(), "1min", fill="prev_bar")
    assert g.loc[3, ["open", "high", "low", "close"]].tolist() == [11.0, 11.5, 10.5, 11.2]
    g = regular_grid(_bars(), "1min", fill="nan")
    assert g.loc[2, ["open", "high", "low", "close"]].isna().all()
    assert str(g.loc[0, "date"]) == "2026-10-12"
    with pytest.raises(ValueError):
        regular_grid(_bars(), "1min", fill="bogus")
    assert regular_grid(_bars().iloc[:0], "1min").empty


def _quote_log(tmp_path):
    lines = [
        "poll_utc,SECID,BID,OFFER,SPREAD,LAST,NUMTRADES,VALTODAY,UPDATETIME,TIME,SYSTIME",
        "2026-10-12T12:00:00+00:00,SBER,100,100.2,0.2,100.1,10,1000,14:59:50,14:59:50,2026-10-12 15:00:00",
        "2026-10-12T12:00:00+00:00,NOQT,,,,,0,0,14:59:50,14:59:50,2026-10-12 15:00:00",
        "2026-10-12T12:00:05+00:00,SBER,100,100.4,0.4,100.1,11,1100,15:00:04,15:00:04,2026-10-12 15:00:05",
        "2026-10-12T13:00:00+00:00,SBER,100,100.2,0.2,100.1,12,1200,16:00:00,16:00:00,2026-10-12 16:00:00",
    ]
    path = tmp_path / "moex_quotes_2026-10-12.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def test_read_quote_log(tmp_path):
    q = read_quote_log(_quote_log(tmp_path))
    assert list(q.columns) == ["secid", "exchange_time", "poll_utc", "bid", "offer", "last", "numtrades"]
    assert len(q) == 4
    assert q["exchange_time"].iloc[0] == pd.Timestamp("2026-10-12 14:59:50")
    assert q["bid"].dtype.kind == "f" and np.isnan(q.loc[q["secid"] == "NOQT", "bid"].iloc[0])


def test_quoted_spread_by_day_weights_by_time_within_the_observed_window(tmp_path):
    out = quoted_spread_by_day(read_quote_log(_quote_log(tmp_path))).set_index("secid")
    rel = [2 * 0.2 / 200.2, 2 * 0.4 / 200.4, 2 * 0.2 / 200.2]
    sber = out.loc["SBER"]
    assert sber["quoted_spread"] == pytest.approx((14 * rel[0] + 3596 * rel[1]) / 3610)
    assert sber["quoted_spread_eq"] == pytest.approx(np.mean(rel))
    assert sber["coverage"] == pytest.approx(1.0)
    assert sber["observed_share"] == pytest.approx(3610 / 60600)
    assert sber["n_changes"] == 3
    noqt = out.loc["NOQT"]
    assert np.isnan(noqt["quoted_spread"]) and noqt["coverage"] == 0.0
    assert str(sber["date"]) == "2026-10-12"


def test_quoted_spread_by_day_empty_input():
    out = quoted_spread_by_day(pd.DataFrame())
    assert out.empty and "observed_share" in out.columns
