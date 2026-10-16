from __future__ import annotations

import datetime as dt
import logging
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

ISS = "https://iss.moex.com/iss"
MOSCOW = ZoneInfo("Europe/Moscow")
USER_AGENT = "edge-project-m2 (research seminar, HSE)"
HISTORY_COLUMNS = ["SECID", "TRADEDATE", "SHORTNAME", "NUMTRADES", "VALUE", "OPEN", "LOW", "HIGH", "CLOSE", "LEGALCLOSEPRICE", "WAPRICE", "VOLUME", "MARKETPRICE2", "TRADINGSESSION"]

SHARE_SECTYPES = ("1", "2", "D")
NON_SHARE_PATTERNS = ("ETF", "БПИФ", "ИПИФ", "ЗПИФ", "ОПИФ", "ПИФ", "FinEx", "FXRL", "FXRU", "FXGD", "FXUS", "FXIT", "FXDE", "FXCN", "FXMM", "FXTB")

SESSIONS = ((dt.time(6, 50), dt.time(9, 50)), (dt.time(9, 50), dt.time(18, 50)), (dt.time(19, 0), dt.time(23, 50)))

log = logging.getLogger(__name__)


def is_share_mask(df: pd.DataFrame, secid_col: str = "SECID") -> pd.Series:
    s = df[secid_col].astype(str)
    mask = ~s.str.match(r"^(RU000|X[A-Z]{5,6}$)")
    if "SECTYPE" in df.columns:
        mask &= df["SECTYPE"].astype(str).isin(SHARE_SECTYPES)
    if "SHORTNAME" in df.columns:
        mask &= ~df["SHORTNAME"].astype(str).str.contains("|".join(NON_SHARE_PATTERNS), case=False, regex=True)
    return mask


def session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


def _get_json(sess: requests.Session, url: str, params: dict, retries: int = 6, timeout: float = 60.0) -> dict:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = sess.get(url, params=params, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.RequestException(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as exc:
            last = exc
            wait = 3 * (attempt + 1)
            log.warning("ISS request failed (%s), retry in %ss: %s %s", exc, wait, url, params)
            time.sleep(wait)
    raise requests.RequestException(f"giving up on {url} {params}: {last}")


def _block(payload: dict, name: str) -> pd.DataFrame:
    block = payload[name]
    return pd.DataFrame(block["data"], columns=block["columns"])


def securities(sess: requests.Session | None = None, board: str = "TQBR") -> pd.DataFrame:
    sess = sess or session()
    payload = _get_json(sess, f"{ISS}/engines/stock/markets/shares/boards/{board}/securities.json", {"iss.meta": "off", "iss.only": "securities", "securities.columns": "SECID,SHORTNAME,SECNAME,SECTYPE,LISTLEVEL,LOTSIZE,MINSTEP,ISSUESIZE,ISIN,PREVPRICE"})
    return _block(payload, "securities")


def history_by_date(day: dt.date, sess: requests.Session | None = None, board: str = "TQBR") -> pd.DataFrame:
    sess = sess or session()
    url = f"{ISS}/history/engines/stock/markets/shares/boards/{board}/securities.json"
    frames = []
    start = 0
    while True:
        payload = _get_json(sess, url, {"date": day.isoformat(), "iss.meta": "off", "iss.only": "history,history.cursor", "history.columns": ",".join(HISTORY_COLUMNS), "start": start})
        df = _block(payload, "history")
        cursor = _block(payload, "history.cursor")
        frames.append(df)
        total = int(cursor["TOTAL"].iloc[0]) if len(cursor) else len(df)
        page = int(cursor["PAGESIZE"].iloc[0]) if len(cursor) else 100
        start += page
        if start >= total or df.empty:
            break
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=HISTORY_COLUMNS)
    return out


def history_security(secid: str, start: dt.date, end: dt.date, sess: requests.Session | None = None, board: str = "TQBR") -> pd.DataFrame:
    sess = sess or session()
    url = f"{ISS}/history/engines/stock/markets/shares/boards/{board}/securities/{secid}.json"
    frames = []
    cursor = 0
    while True:
        payload = _get_json(sess, url, {"from": start.isoformat(), "till": end.isoformat(), "iss.meta": "off", "iss.only": "history,history.cursor", "history.columns": ",".join(HISTORY_COLUMNS), "start": cursor})
        df = _block(payload, "history")
        if df.empty:
            break
        frames.append(df)
        total = int(_block(payload, "history.cursor")["TOTAL"].iloc[0])
        cursor += len(df)
        if cursor >= total:
            break
    if not frames:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])
    for c in ("NUMTRADES", "OPEN", "LOW", "HIGH", "CLOSE", "VOLUME", "VALUE"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("TRADEDATE").reset_index(drop=True)


def candles(secid: str, interval: int, start: dt.date, end: dt.date, sess: requests.Session | None = None, board: str = "TQBR") -> pd.DataFrame:
    sess = sess or session()
    url = f"{ISS}/engines/stock/markets/shares/boards/{board}/securities/{secid}/candles.json"
    frames = []
    cursor = 0
    while True:
        payload = _get_json(sess, url, {"interval": interval, "from": start.isoformat(), "till": end.isoformat(), "iss.meta": "off", "start": cursor})
        df = _block(payload, "candles")
        if df.empty:
            break
        frames.append(df)
        if len(df) < 500:
            break
        cursor += len(df)
    if not frames:
        return pd.DataFrame(columns=["begin", "open", "high", "low", "close", "volume", "value"])
    df = pd.concat(frames, ignore_index=True)
    df["begin"] = pd.to_datetime(df["begin"])
    df = df[["begin", "open", "high", "low", "close", "volume", "value"]].drop_duplicates("begin").sort_values("begin")
    for c in ("open", "high", "low", "close", "volume", "value"):
        df[c] = df[c].astype(float)
    return df.reset_index(drop=True)


def regular_grid(bars: pd.DataFrame, freq: str, fill: str = "prev_close") -> pd.DataFrame:
    if fill not in ("prev_close", "prev_bar", "nan"):
        raise ValueError("fill must be 'prev_close', 'prev_bar' or 'nan'")
    if bars.empty:
        return bars
    out = []
    for day, g in bars.groupby(bars["begin"].dt.date):
        g = g.set_index("begin").sort_index(kind="mergesort")
        grid = pd.date_range(g.index.min().floor(freq), g.index.max().floor(freq), freq=freq)
        g = g.reindex(grid)
        if fill == "prev_close":
            prev = g["close"].ffill()
            for c in ("open", "high", "low", "close"):
                g[c] = g[c].fillna(prev)
        elif fill == "prev_bar":
            g[["open", "high", "low", "close"]] = g[["open", "high", "low", "close"]].ffill()
        g[["volume", "value"]] = g[["volume", "value"]].fillna(0.0)
        g["date"] = day
        out.append(g.rename_axis("begin").reset_index())
    return pd.concat(out, ignore_index=True)


def read_quote_log(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"SECID": str})
    if df.empty:
        return pd.DataFrame(columns=["secid", "exchange_time", "poll_utc", "bid", "offer", "last", "numtrades"])
    day_str = pd.to_datetime(df["poll_utc"]).dt.tz_convert(MOSCOW).dt.strftime("%Y-%m-%d")
    df["exchange_time"] = pd.to_datetime(day_str + " " + df["TIME"].astype(str), errors="coerce")
    df = df.rename(columns={"SECID": "secid", "BID": "bid", "OFFER": "offer", "LAST": "last", "NUMTRADES": "numtrades"})
    for c in ("bid", "offer", "last"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["numtrades"] = pd.to_numeric(df["numtrades"], errors="coerce")
    return df[["secid", "exchange_time", "poll_utc", "bid", "offer", "last", "numtrades"]].dropna(subset=["exchange_time"])


def quoted_spread_by_day(quotes: pd.DataFrame) -> pd.DataFrame:
    if quotes.empty:
        return pd.DataFrame(columns=["secid", "date", "quoted_spread", "quoted_spread_eq", "coverage", "observed_share", "n_changes"])
    q_all = quotes.copy()
    q_all["_day"] = pd.to_datetime(q_all["exchange_time"]).dt.normalize()
    rows = []
    for day_ts, g_day in q_all.groupby("_day"):
        day = pd.Timestamp(day_ts)
        if "poll_utc" in g_day.columns and g_day["poll_utc"].notna().any():
            first_poll = g_day["poll_utc"].min()
            obs_start = np.datetime64(g_day.loc[g_day["poll_utc"] == first_poll, "exchange_time"].max())
            obs_end = np.datetime64(g_day["exchange_time"].max())
        else:
            obs_start = np.datetime64(day)
            obs_end = np.datetime64(day + pd.Timedelta(hours=23, minutes=50))
        session_seconds = sum((dt.datetime.combine(day.date(), e) - dt.datetime.combine(day.date(), s)).total_seconds() for s, e in SESSIONS)
        observed_seconds = 0.0
        for s_start, s_end in SESSIONS:
            lo = max(np.datetime64(pd.Timestamp.combine(day.date(), s_start)), obs_start)
            hi = min(np.datetime64(pd.Timestamp.combine(day.date(), s_end)), obs_end)
            observed_seconds += max((hi - lo) / np.timedelta64(1, "s"), 0.0)
        observed_share = observed_seconds / session_seconds if session_seconds > 0 else 0.0
        for secid, g in g_day.sort_values("exchange_time", kind="mergesort").groupby("secid"):
            g = g.drop_duplicates("exchange_time", keep="last")
            t = g["exchange_time"].to_numpy()
            t_next = np.append(t[1:], obs_end)
            bid = g["bid"].to_numpy(dtype=float)
            offer = g["offer"].to_numpy(dtype=float)
            valid = np.isfinite(bid) & np.isfinite(offer) & (bid > 0) & (offer > bid)
            weight = np.zeros(len(t))
            for s_start, s_end in SESSIONS:
                lo = np.maximum(np.datetime64(pd.Timestamp.combine(day.date(), s_start)), obs_start)
                hi = np.minimum(np.datetime64(pd.Timestamp.combine(day.date(), s_end)), obs_end)
                a = np.maximum(t, lo)
                b = np.minimum(t_next, hi)
                weight += np.maximum((b - a) / np.timedelta64(1, "s"), 0.0)
            w = weight * valid
            if w.sum() <= 0:
                rows.append({"secid": secid, "date": day.date(), "quoted_spread": np.nan, "quoted_spread_eq": np.nan, "coverage": 0.0, "observed_share": observed_share, "n_changes": int(len(t))})
                continue
            rel = 2.0 * (offer - bid) / (offer + bid)
            rows.append({
                "secid": secid,
                "date": day.date(),
                "quoted_spread": float(np.sum(w * np.where(valid, rel, 0.0)) / w.sum()),
                "quoted_spread_eq": float(np.mean(rel[valid])),
                "coverage": float(w.sum() / observed_seconds) if observed_seconds > 0 else 0.0,
                "observed_share": observed_share,
                "n_changes": int(len(t)),
            })
    return pd.DataFrame(rows)
