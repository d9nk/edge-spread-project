from __future__ import annotations

import numpy as np
import pandas as pd


def _ns(series: pd.Series) -> np.ndarray:
    s = pd.to_datetime(series)
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    return s.to_numpy(dtype="datetime64[ns]")


def _ns_scalar(ts) -> np.datetime64:
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return np.datetime64(t, "ns")


def match_quotes(trades: pd.DataFrame, quotes: pd.DataFrame) -> pd.DataFrame:
    t_cols = ["time", "id"] if "id" in trades.columns else ["time"]
    q_cols = ["time", "update_id"] if "update_id" in quotes.columns else ["time"]
    t = trades.sort_values(t_cols, kind="mergesort").reset_index(drop=True)
    q = quotes.sort_values(q_cols, kind="mergesort")[["time", "bid", "ask"]]
    q = q[(q["bid"] > 0) & (q["ask"] > q["bid"])]
    merged = pd.merge_asof(t, q, on="time", direction="backward", allow_exact_matches=False)
    merged["mid"] = 0.5 * (merged["bid"] + merged["ask"])
    merged["direction"] = np.where(merged["is_buyer_maker"], -1.0, 1.0)
    merged["eff"] = 2.0 * merged["direction"] * (merged["price"] - merged["mid"]) / merged["mid"]
    p = merged["price"].to_numpy()
    m = merged["mid"].to_numpy()
    d = np.where(p > m, 1.0, np.where(p < m, -1.0, np.nan))
    prev_p = np.concatenate([[np.nan], p[:-1]])
    tick = np.where(p > prev_p, 1.0, np.where(p < prev_p, -1.0, np.nan))
    tick = pd.Series(tick).ffill().to_numpy()
    merged["lee_ready"] = np.where(np.isnan(d), tick, d)
    return merged


def daily_benchmark(matched: pd.DataFrame) -> dict:
    ok = matched["mid"].notna() & np.isfinite(matched["eff"])
    m = matched[ok]
    if len(m) == 0:
        return {"n_trades": int(len(matched)), "n_matched": 0}
    w = (m["qty"] * m["price"]).to_numpy(dtype=float) if "qty" in m.columns else np.ones(len(m), dtype=float)
    eff = m["eff"].to_numpy()
    lr_known = m["lee_ready"].notna()
    w_sum = float(np.nansum(w))
    return {
        "n_trades": int(len(matched)),
        "n_matched": int(len(m)),
        "eff_mean": float(eff.mean()),
        "eff_vw": float(np.average(eff, weights=w)) if w_sum > 0 else float(eff.mean()),
        "eff_rms": float(np.sqrt(np.mean(eff * eff))),
        "eff_median": float(np.median(eff)),
        "share_at_quote": float(np.mean(np.where(m["direction"] > 0, m["price"] == m["ask"], m["price"] == m["bid"]))),
        "share_inside": float(np.mean((m["price"] > m["bid"]) & (m["price"] < m["ask"]))),
        "lee_ready_error": float(np.mean(m.loc[lr_known, "lee_ready"] != m.loc[lr_known, "direction"])) if lr_known.any() else np.nan,
        "eff_mean_lee_ready": float(np.mean(2.0 * m.loc[lr_known, "lee_ready"] * (m.loc[lr_known, "price"] - m.loc[lr_known, "mid"]) / m.loc[lr_known, "mid"])) if lr_known.any() else np.nan,
        "buy_share": float(np.mean(m["direction"] > 0)),
    }


def quoted_spread(quotes: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    q_cols = ["time", "update_id"] if "update_id" in quotes.columns else ["time"]
    q = quotes.sort_values(q_cols, kind="mergesort")[["time", "bid", "ask"]]
    t_all = _ns(q["time"])
    end_ns = _ns_scalar(end)
    start_ns = _ns_scalar(start)
    q = q[t_all < end_ns]
    t = t_all[t_all < end_ns]
    if q.empty:
        return {"quoted_tw": np.nan, "quoted_eq": np.nan, "n_quotes": 0}
    t_next = np.append(t[1:], end_ns)
    t_clip = np.maximum(t, start_ns)
    dur = np.maximum((t_next - t_clip) / np.timedelta64(1, "s"), 0.0)
    bid = q["bid"].to_numpy()
    ask = q["ask"].to_numpy()
    valid = (bid > 0) & (ask > bid)
    rel = np.where(valid, 2.0 * (ask - bid) / (ask + bid), 0.0)
    w = dur * valid
    in_win = t >= start_ns
    valid_eq = valid & in_win if (valid & in_win).any() else valid
    return {
        "quoted_tw": float(np.sum(w * rel) / w.sum()) if w.sum() > 0 else np.nan,
        "quoted_eq": float(np.mean(rel[valid_eq])) if valid_eq.any() else np.nan,
        "n_quotes": int(in_win.sum()) if in_win.any() else int(len(q)),
    }


def trade_side_proxy(trades: pd.DataFrame, max_gap: pd.Timedelta = pd.Timedelta(seconds=10)) -> pd.DataFrame:
    t_cols = ["time", "id"] if "id" in trades.columns else ["time"]
    t = trades.sort_values(t_cols, kind="mergesort").reset_index(drop=True)
    time = _ns(t["time"])
    price = t["price"].to_numpy(dtype=float)
    buyer_maker = t["is_buyer_maker"].to_numpy(dtype=bool)
    idx = np.arange(len(t))
    ask_idx = pd.Series(np.where(~buyer_maker, idx, -1)).cummax().shift(1, fill_value=-1).to_numpy()
    bid_idx = pd.Series(np.where(buyer_maker, idx, -1)).cummax().shift(1, fill_value=-1).to_numpy()
    ok = (ask_idx >= 0) & (bid_idx >= 0)
    ask_p = np.where(ok, price[np.maximum(ask_idx, 0)], np.nan)
    bid_p = np.where(ok, price[np.maximum(bid_idx, 0)], np.nan)
    ask_t = time[np.maximum(ask_idx, 0)]
    bid_t = time[np.maximum(bid_idx, 0)]
    gap = np.timedelta64(pd.Timedelta(max_gap).to_timedelta64())
    recent = ok & ((time - ask_t) <= gap) & ((time - bid_t) <= gap) & (ask_p >= bid_p)
    mid = np.where(recent, 0.5 * (ask_p + bid_p), np.nan)
    direction = np.where(buyer_maker, -1.0, 1.0)
    t["mid_proxy"] = mid
    t["eff_proxy"] = 2.0 * direction * (price - mid) / mid
    return t
