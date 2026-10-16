from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from edgespread.data_binance import VisionMissing, download, futures_daily_path, futures_daily_url

log = logging.getLogger("download_binance_futures_taq")

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "LINKUSDT", "LTCUSDT", "AVAXUSDT", "DOTUSDT",
           "MATICUSDT", "ATOMUSDT", "UNIUSDT", "FILUSDT", "APTUSDT", "ARBUSDT", "OPUSDT", "INJUSDT", "SANDUSDT", "GALAUSDT",
           "CHZUSDT", "ONEUSDT", "IOTAUSDT", "ZILUSDT", "ANKRUSDT", "CELRUSDT", "DENTUSDT", "HOTUSDT", "KEYUSDT", "STMXUSDT"]
QUICK = ["XRPUSDT", "ADAUSDT", "LINKUSDT", "SANDUSDT", "ZILUSDT", "DENTUSDT"]
QUICK_DAYS = ["2023-06-01", "2023-10-01", "2024-03-01"]


def default_days() -> list[dt.date]:
    days = []
    for year, month in [(2023, m) for m in range(6, 13)] + [(2024, m) for m in range(1, 4)]:
        days.append(dt.date(year, month, 1))
        days.append(dt.date(year, month, 15))
    return days


def task(root: str, kind: str, symbol: str, day: dt.date, session: requests.Session) -> str:
    path = futures_daily_path(root, kind, symbol, day)
    marker = path + ".missing"
    if os.path.exists(marker):
        return "missing"
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return "cached"
    try:
        download(futures_daily_url(kind, symbol, day), path, session=session)
        return "downloaded"
    except VisionMissing:
        os.makedirs(os.path.dirname(marker), exist_ok=True)
        open(marker, "w").close()
        return "missing"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=None, help="futures symbols; default: thirty symbols across the liquidity range")
    ap.add_argument("--days", nargs="*", default=None, help="ISO dates; default: 1st and 15th of each month 2023-06 .. 2024-03")
    ap.add_argument("--quick", action="store_true", help="six symbols x three days, about 1 GB")
    ap.add_argument("--root", default="data/binance")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)

    symbols = args.symbols or (QUICK if args.quick else SYMBOLS)
    day_strings = args.days or (QUICK_DAYS if args.quick else None)
    days = [dt.date.fromisoformat(d) for d in day_strings] if day_strings else default_days()
    jobs = [(kind, sym, day) for sym in symbols for day in days for kind in ("trades", "bookTicker")]
    log.info("%d files for %d symbols x %d days", len(jobs), len(symbols), len(days))
    counts = {"downloaded": 0, "cached": 0, "missing": 0, "failed": 0}
    sessions = [requests.Session() for _ in range(args.workers)]
    for s in sessions:
        s.headers["User-Agent"] = "edge-project-m2 (research seminar, HSE)"
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(task, args.root, kind, sym, day, sessions[i % args.workers]): (kind, sym, day) for i, (kind, sym, day) in enumerate(jobs)}
        for i, fut in enumerate(as_completed(futures), 1):
            key = futures[fut]
            try:
                counts[fut.result()] += 1
            except Exception as exc:
                counts["failed"] += 1
                log.error("%s failed: %r", key, exc)
            if i % 20 == 0 or i == len(futures):
                log.info("%d/%d files: %s", i, len(futures), counts)
    log.info("done: %s", counts)
    return 0 if counts["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
