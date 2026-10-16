from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

Fill = Literal["prev_close", "prev_bar", "nan"]
OHLC = ["open", "high", "low", "close"]


def bars_from_trades(times: pd.Series | np.ndarray, prices: pd.Series | np.ndarray, freq: str, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None, fill: Fill = "prev_close") -> pd.DataFrame:
    if fill not in ("prev_close", "prev_bar", "nan"):
        raise ValueError("fill must be 'prev_close', 'prev_bar' or 'nan'")
    s = pd.Series(np.asarray(prices, dtype=float), index=pd.DatetimeIndex(times))
    s = s.sort_index(kind="mergesort")
    if len(s) == 0:
        return pd.DataFrame(columns=["open_time"] + OHLC + ["n_trades"])
    grp = s.resample(freq, label="left", closed="left")
    bars = grp.ohlc()
    bars["n_trades"] = grp.count()
    start = pd.Timestamp(start) if start is not None else bars.index.min()
    end = pd.Timestamp(end) if end is not None else bars.index.max() + pd.tseries.frequencies.to_offset(freq)
    grid = pd.date_range(start.floor(freq), end, freq=freq, inclusive="left")
    bars = bars.reindex(grid)
    bars["n_trades"] = bars["n_trades"].fillna(0).astype(int)
    if fill == "prev_close":
        prev = bars["close"].ffill()
        for c in OHLC:
            bars[c] = bars[c].fillna(prev)
    elif fill == "prev_bar":
        bars[OHLC] = bars[OHLC].ffill()
    return bars.rename_axis("open_time").reset_index()


def resample_bars(bars: pd.DataFrame, freq: str, time_col: str = "open_time") -> pd.DataFrame:
    if bars.empty:
        return bars.copy()
    b = bars.set_index(time_col).sort_index(kind="mergesort").copy()
    has_trades_col = "n_trades" in b.columns
    was_filled = False
    fill_prev_bar = False
    if has_trades_col:
        empty = b["n_trades"] == 0
        filled = empty & b["close"].notna()
        if filled.any():
            was_filled = True
            fill_prev_bar = bool((b.loc[filled, "open"] != b.loc[filled, "close"]).any())
            b.loc[empty, OHLC] = np.nan
    grp = b.resample(freq, label="left", closed="left")
    out = pd.DataFrame({
        "open": grp["open"].first(),
        "high": grp["high"].max(),
        "low": grp["low"].min(),
        "close": grp["close"].last(),
        "n_trades": grp["n_trades"].sum() if has_trades_col else grp["close"].count(),
    })
    if was_filled:
        if fill_prev_bar:
            out[OHLC] = out[OHLC].ffill()
        else:
            prev = out["close"].ffill()
            for c in OHLC:
                out[c] = out[c].fillna(prev)
    return out.rename_axis(time_col).reset_index()
