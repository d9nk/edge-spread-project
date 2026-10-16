from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from edgespread import estimate
from edgespread.bars import bars_from_trades
from edgespread.benchmark import daily_benchmark, match_quotes, quoted_spread, trade_side_proxy
from edgespread.data_binance import read_book_ticker, read_futures_trades

METHODS = ("EDGE", "GMM", "OHL", "CHL", "OHLC", "CHLO", "AR", "CS", "ROLL")
log = logging.getLogger("compute_crypto_benchmark")


def pair_days(root: str) -> list[tuple[str, dt.date, str, str]]:
    out = []
    tdir = os.path.join(root, "trades")
    qdir = os.path.join(root, "bookTicker")
    if not os.path.isdir(tdir):
        return out
    for symbol in sorted(os.listdir(tdir)):
        for name in sorted(os.listdir(os.path.join(tdir, symbol))):
            if not name.endswith(".zip"):
                continue
            day = dt.date.fromisoformat(name[-14:-4])
            qpath = os.path.join(qdir, symbol, f"{symbol}-bookTicker-{day.isoformat()}.zip")
            tpath = os.path.join(tdir, symbol, name)
            if os.path.exists(qpath) and os.path.getsize(qpath) > 0 and os.path.getsize(tpath) > 0:
                out.append((symbol, day, tpath, qpath))
    return out


def process(symbol: str, day: dt.date, tpath: str, qpath: str) -> dict:
    t0 = time.time()
    trades = read_futures_trades(tpath)
    quotes = read_book_ticker(qpath)
    start = pd.Timestamp(day, tz="UTC")
    end = start + pd.Timedelta(days=1)
    trades = trades[(trades["time"] >= start) & (trades["time"] < end)]
    row = {"symbol": symbol, "date": day.isoformat()}
    matched = match_quotes(trades, quotes)
    row.update(daily_benchmark(matched))
    row.update(quoted_spread(quotes, start, end))
    prox = trade_side_proxy(trades, max_gap=pd.Timedelta(seconds=10))
    e = prox["eff_proxy"].dropna()
    row["eff_proxy_mean"] = float(e.mean()) if len(e) else np.nan
    row["eff_proxy_coverage"] = float(len(e) / max(len(prox), 1))
    row["price"] = float(trades["price"].iloc[-1]) if len(trades) else np.nan
    row["dollar_volume"] = float(trades["quote_qty"].sum())
    for freq, tag in (("1s", "1s"), ("10s", "10s"), ("1min", "1m"), ("5min", "5m"), ("15min", "15m")):
        b = bars_from_trades(trades["time"], trades["price"], freq, start, end, fill="prev_close")
        o, h, l, c = (b[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
        est = estimate(o, h, l, c, methods=METHODS, sign=True)
        for m, v in est.items():
            row[f"{m}_{tag}"] = float(v)
        row[f"n_bars_{tag}"] = int(len(b))
        row[f"empty_bars_{tag}"] = float((b["n_trades"] == 0).mean())
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.diff(np.log(c))
        row[f"sd_ret_{tag}"] = float(np.nanstd(r))
    for fill in ("nan", "prev_bar"):
        b = bars_from_trades(trades["time"], trades["price"], "1min", start, end, fill=fill)
        o, h, l, c = (b[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
        est = estimate(o, h, l, c, methods=("EDGE", "AR", "CS", "ROLL"), sign=True)
        for m, v in est.items():
            row[f"{m}_1m_fill_{fill}"] = float(v)
    row["seconds"] = time.time() - t0
    return row


def _task(args):
    symbol, day, tpath, qpath = args
    try:
        return process(symbol, day, tpath, qpath)
    except Exception as exc:
        return {"symbol": symbol, "date": day.isoformat(), "error": repr(exc)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/binance/futures")
    ap.add_argument("--out", default="results/crypto")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    os.makedirs(args.out, exist_ok=True)
    jsonl = os.path.join(args.out, "benchmark_pair_day.jsonl")
    done = set()
    if os.path.exists(jsonl):
        with open(jsonl) as f:
            for line in f:
                r = json.loads(line)
                if "error" not in r:
                    done.add((r["symbol"], r["date"]))
    jobs = [j for j in pair_days(args.root) if (j[0], j[1].isoformat()) not in done]
    log.info("%d pair-days to process (%d already done)", len(jobs), len(done))
    with ProcessPoolExecutor(max_workers=args.workers) as pool, open(jsonl, "a") as f:
        futures = [pool.submit(_task, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            f.write(json.dumps(row) + "\n")
            f.flush()
            if "error" in row:
                log.error("%s %s: %s", row["symbol"], row["date"], row["error"])
            elif i % 10 == 0 or i == len(futures):
                log.info("%d/%d done, last %s %s: eff %.4f%% EDGE_1m %.4f%% (%.0fs)", i, len(futures), row["symbol"], row["date"], 100 * row.get("eff_mean", np.nan), 100 * row.get("EDGE_1m", np.nan), row["seconds"])
    rows = []
    with open(jsonl) as f:
        for line in f:
            r = json.loads(line)
            if "error" not in r:
                rows.append(r)
    df = pd.DataFrame(rows).sort_values(["symbol", "date"])
    df.to_csv(os.path.join(args.out, "benchmark_pair_day.csv"), index=False)
    log.info("wrote %d pair-days to %s", len(df), os.path.join(args.out, "benchmark_pair_day.csv"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
