from __future__ import annotations

import zipfile

import pandas as pd
import pytest

from edgespread.data_binance import _to_utc, read_book_ticker, read_futures_trades, read_klines

KLINE_HEADER = "open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore"
KLINE_ROWS = ["1700000000000,100,101,99,100.5,10,1700000059999,1005,7,4,402,0", "1700000060000,100.5,102,100,101,20,1700000119999,2020,9,8,808,0"]


def _zip(tmp_path, name, text):
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(name.replace(".zip", ".csv"), text)
    return str(path)


def test_to_utc_handles_milliseconds_and_microseconds():
    ms = pd.Series([1_700_000_000_000])
    us = pd.Series([1_700_000_000_000_000])
    assert _to_utc(ms).iloc[0] == pd.Timestamp("2023-11-14 22:13:20", tz="UTC")
    assert _to_utc(us).iloc[0] == pd.Timestamp("2023-11-14 22:13:20", tz="UTC")


@pytest.mark.parametrize("with_header", [True, False])
def test_read_klines(tmp_path, with_header):
    text = "\n".join(([KLINE_HEADER] if with_header else []) + KLINE_ROWS) + "\n"
    df = read_klines(_zip(tmp_path, f"klines_{with_header}.zip", text))
    assert len(df) == 2
    assert df["open_time"].iloc[0] == pd.Timestamp("2023-11-14 22:13:20", tz="UTC")
    assert df["n_trades"].tolist() == [7, 9]
    assert df["close"].tolist() == [100.5, 101.0]
    assert "close_time" not in df.columns


def test_read_book_ticker(tmp_path):
    text = "update_id,best_bid_price,best_bid_qty,best_ask_price,best_ask_qty,transaction_time,event_time\n2,100,1,100.1,2,1700000000100,1700000000101\n1,100,1,100.2,2,1700000000000,1700000000001\n"
    df = read_book_ticker(_zip(tmp_path, "bookTicker.zip", text))
    assert list(df.columns) == ["time", "update_id", "bid", "bid_qty", "ask", "ask_qty"]
    assert df["update_id"].tolist() == [1, 2]
    assert df["ask"].tolist() == [100.2, 100.1]


def test_read_futures_trades(tmp_path):
    text = "id,price,qty,quote_qty,time,is_buyer_maker\n2,100.5,1,100.5,1700000000500,true\n1,100,2,200,1700000000000,False\n"
    df = read_futures_trades(_zip(tmp_path, "trades.zip", text))
    assert df["id"].tolist() == [1, 2]
    assert df["is_buyer_maker"].tolist() == [False, True]
    assert df["is_buyer_maker"].dtype == bool
    assert df["time"].iloc[1] - df["time"].iloc[0] == pd.Timedelta(milliseconds=500)
