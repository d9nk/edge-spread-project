from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from edgespread.data_binance import api_klines, spot_symbols

log = logging.getLogger("download_binance_daily_api")


def fetch(symbol: str, out_dir: str, start: dt.datetime, end: dt.datetime) -> dict | None:
    path = os.path.join(out_dir, f"{symbol}.parquet")
    if os.path.exists(path):
        df = pd.read_parquet(path)
    else:
        df = api_klines(symbol, "1d", start, end)
        if df.empty:
            return None
        df.to_parquet(path, index=False)
    return {
        "symbol": symbol,
        "first_day": df["open_time"].min().date().isoformat(),
        "last_day": df["open_time"].max().date().isoformat(),
        "n_days": int(len(df)),
        "n_trades": int(df["n_trades"].sum()),
        "quote_volume": float(df["quote_volume"].sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quotes", nargs="+", default=["USDT", "BTC"], help="quote assets to include")
    ap.add_argument("--out", default="data/binance/daily")
    ap.add_argument("--start", default="2017-07-01")
    ap.add_argument("--end", default=dt.date.today().isoformat())
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    os.makedirs(args.out, exist_ok=True)

    symbols = spot_symbols()
    symbols = symbols[(symbols["status"] == "TRADING") & symbols["quote"].isin(args.quotes)]
    log.info("%d listed symbols with quote in %s", len(symbols), args.quotes)
    start = dt.datetime.fromisoformat(args.start).replace(tzinfo=dt.timezone.utc)
    end = dt.datetime.fromisoformat(args.end).replace(tzinfo=dt.timezone.utc)

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, s, args.out, start, end): s for s in symbols["symbol"]}
        for i, fut in enumerate(as_completed(futures), 1):
            sym = futures[fut]
            try:
                info = fut.result()
            except Exception as exc:
                log.error("%s failed: %r", sym, exc)
                continue
            if info is not None:
                rows.append(info)
            if i % 50 == 0:
                log.info("%d/%d symbols done", i, len(futures))
    if not rows:
        log.warning("no symbols downloaded")
        return 1
    table = pd.DataFrame(rows).merge(symbols, on="symbol", how="left").sort_values("n_trades", ascending=False)
    table.to_csv(os.path.join(args.out, "symbols.csv"), index=False)
    log.info("wrote %s with %d symbols", os.path.join(args.out, "symbols.csv"), len(table))
    return 0


if __name__ == "__main__":
    sys.exit(main())
