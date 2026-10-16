from __future__ import annotations

import datetime as dt
import io
import logging
import os
import time
import re
import zipfile

import numpy as np
import pandas as pd
import requests

VISION = "https://data.binance.vision/data"
VISION_S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
SPOT_API = "https://api.binance.com/api/v3"
USER_AGENT = "edge-project-m2 (research seminar, HSE)"

KLINE_COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "n_trades", "taker_buy_base", "taker_buy_quote", "ignore"]

log = logging.getLogger(__name__)


class VisionMissing(Exception):
    pass


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


def download(url: str, path: str, session: requests.Session | None = None, retries: int = 5, timeout: float = 300.0) -> str:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    session = session or _session()
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            with session.get(url, stream=True, timeout=timeout) as r:
                if r.status_code == 404:
                    raise VisionMissing(url)
                r.raise_for_status()
                tmp = path + ".part"
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                expected = int(r.headers.get("content-length", 0))
                got = os.path.getsize(tmp)
                if expected and got != expected:
                    raise IOError(f"incomplete download {got}/{expected} for {url}")
                os.replace(tmp, path)
                return path
        except VisionMissing:
            raise
        except (requests.RequestException, IOError) as exc:
            last_exc = exc
            wait = 5 * (attempt + 1)
            log.warning("download failed (%s), retry in %ss: %s", exc, wait, url)
            time.sleep(wait)
    raise requests.RequestException(f"giving up on {url}: {last_exc}")


def _list_bucket(prefix: str, tag: str, delimiter: bool, session: requests.Session | None) -> list[str]:
    session = session or _session()
    out: list[str] = []
    token = None
    pattern = re.compile(rf"<{tag}>([^<]+)</{tag}>")
    while True:
        params = {"prefix": prefix, "list-type": "2", "max-keys": "1000"}
        if delimiter:
            params["delimiter"] = "/"
        if token:
            params["continuation-token"] = token
        r = session.get(VISION_S3, params=params, timeout=60)
        r.raise_for_status()
        out.extend(pattern.findall(r.text))
        if "<IsTruncated>true</IsTruncated>" not in r.text:
            break
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", r.text)
        if not m:
            break
        token = m.group(1).replace("&amp;", "&")
    return out


def list_prefixes(prefix: str, session: requests.Session | None = None) -> list[str]:
    return _list_bucket(prefix, "Prefix", True, session)


def list_keys(prefix: str, session: requests.Session | None = None) -> list[str]:
    return [k for k in _list_bucket(prefix, "Key", False, session) if not k.endswith("CHECKSUM")]


def kline_month_url(symbol: str, interval: str, year: int, month: int) -> str:
    return f"{VISION}/spot/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{year:04d}-{month:02d}.zip"


def kline_month_path(root: str, symbol: str, interval: str, year: int, month: int) -> str:
    return os.path.join(root, "klines", symbol, interval, f"{symbol}-{interval}-{year:04d}-{month:02d}.zip")


def _to_utc(ts: pd.Series) -> pd.Series:
    ts = ts.astype("int64")
    micro = ts > 10**14
    out = pd.Series(np.where(micro, ts // 1000, ts), index=ts.index)
    return pd.to_datetime(out, unit="ms", utc=True)


def _read_csv_zip(path: str, names: list[str]) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        name = z.namelist()[0]
        raw = z.read(name)
    head = raw[:200].decode("utf-8", errors="ignore").split("\n", 1)[0]
    has_header = not head.split(",")[0].strip().lstrip("-").isdigit()
    return pd.read_csv(io.BytesIO(raw), header=0 if has_header else None, names=names)


def read_klines(path: str) -> pd.DataFrame:
    df = _read_csv_zip(path, KLINE_COLUMNS)
    df["open_time"] = _to_utc(df["open_time"])
    df = df.drop(columns=["close_time", "ignore"])
    for c in ("open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"):
        df[c] = df[c].astype(float)
    df["n_trades"] = df["n_trades"].astype("int64")
    return df.sort_values("open_time").reset_index(drop=True)


def api_klines(symbol: str, interval: str, start: dt.datetime, end: dt.datetime, session: requests.Session | None = None) -> pd.DataFrame:
    session = session or _session()
    frames = []
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    while start_ms < end_ms:
        for attempt in range(6):
            try:
                r = session.get(f"{SPOT_API}/klines", params={"symbol": symbol, "interval": interval, "startTime": start_ms, "endTime": end_ms, "limit": 1000}, timeout=120)
            except requests.RequestException as exc:
                log.warning("klines %s failed (%s), retrying", symbol, exc)
                time.sleep(10 * (attempt + 1))
                continue
            if r.status_code in (418, 429):
                wait = int(r.headers.get("Retry-After", 30))
                log.warning("rate limited, sleeping %ss", wait)
                time.sleep(wait)
                continue
            r.raise_for_status()
            break
        else:
            raise requests.RequestException(f"rate limit retries exhausted for {symbol}")
        rows = r.json()
        if not rows:
            break
        df = pd.DataFrame(rows, columns=KLINE_COLUMNS)
        frames.append(df)
        last_open = int(rows[-1][0])
        if len(rows) < 1000:
            break
        start_ms = last_open + 1
    if not frames:
        return pd.DataFrame(columns=[c for c in KLINE_COLUMNS if c not in ("close_time", "ignore")])
    df = pd.concat(frames, ignore_index=True)
    df["open_time"] = _to_utc(df["open_time"])
    df = df.drop(columns=["close_time", "ignore"])
    for c in ("open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"):
        df[c] = df[c].astype(float)
    df["n_trades"] = df["n_trades"].astype("int64")
    return df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)


def spot_symbols(session: requests.Session | None = None) -> pd.DataFrame:
    session = session or _session()
    last: Exception | None = None
    for attempt in range(6):
        try:
            r = session.get(f"{SPOT_API}/exchangeInfo", timeout=120)
            r.raise_for_status()
            break
        except requests.RequestException as exc:
            last = exc
            log.warning("exchangeInfo failed (%s), retrying", exc)
            time.sleep(10 * (attempt + 1))
    else:
        raise requests.RequestException(f"exchangeInfo failed repeatedly: {last}")
    rows = [{"symbol": s["symbol"], "base": s["baseAsset"], "quote": s["quoteAsset"], "status": s["status"]} for s in r.json()["symbols"]]
    return pd.DataFrame(rows)


def futures_daily_url(kind: str, symbol: str, day: dt.date) -> str:
    return f"{VISION}/futures/um/daily/{kind}/{symbol}/{symbol}-{kind}-{day.isoformat()}.zip"


def futures_daily_path(root: str, kind: str, symbol: str, day: dt.date) -> str:
    return os.path.join(root, "futures", kind, symbol, f"{symbol}-{kind}-{day.isoformat()}.zip")


def read_futures_trades(path: str) -> pd.DataFrame:
    df = _read_csv_zip(path, ["id", "price", "qty", "quote_qty", "time", "is_buyer_maker"])
    df["time"] = _to_utc(df["time"])
    df["price"] = df["price"].astype(float)
    df["qty"] = df["qty"].astype(float)
    df["quote_qty"] = df["quote_qty"].astype(float)
    df["is_buyer_maker"] = df["is_buyer_maker"].astype(str).str.lower().map({"true": True, "false": False}).astype(bool)
    return df.sort_values(["time", "id"]).reset_index(drop=True)


def read_spot_trades(path: str) -> pd.DataFrame:
    df = _read_csv_zip(path, ["id", "price", "qty", "quote_qty", "time", "is_buyer_maker", "is_best_match"])
    df["time"] = _to_utc(df["time"])
    df["price"] = df["price"].astype(float)
    df["qty"] = df["qty"].astype(float)
    df["quote_qty"] = df["quote_qty"].astype(float)
    df["is_buyer_maker"] = df["is_buyer_maker"].astype(str).str.lower().map({"true": True, "false": False}).astype(bool)
    return df.drop(columns=["is_best_match"]).sort_values(["time", "id"]).reset_index(drop=True)


def read_book_ticker(path: str) -> pd.DataFrame:
    df = _read_csv_zip(path, ["update_id", "bid", "bid_qty", "ask", "ask_qty", "transaction_time", "event_time"])
    df["time"] = _to_utc(df["transaction_time"])
    for c in ("bid", "bid_qty", "ask", "ask_qty"):
        df[c] = df[c].astype(float)
    df = df[["time", "update_id", "bid", "bid_qty", "ask", "ask_qty"]]
    return df.sort_values(["time", "update_id"]).reset_index(drop=True)


def spot_daily_trades_url(symbol: str, day: dt.date) -> str:
    return f"{VISION}/spot/daily/trades/{symbol}/{symbol}-trades-{day.isoformat()}.zip"


def spot_daily_trades_path(root: str, symbol: str, day: dt.date) -> str:
    return os.path.join(root, "spot_trades", symbol, f"{symbol}-trades-{day.isoformat()}.zip")
