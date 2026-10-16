from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from edgespread.data_binance import VisionMissing, download, kline_month_path, kline_month_url

log = logging.getLogger("download_binance_klines")


def months(start: str, end: str) -> list[tuple[int, int]]:
    y0, m0 = (int(x) for x in start.split("-"))
    y1, m1 = (int(x) for x in end.split("-"))
    out = []
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        out.append((y, m))
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def task(root: str, symbol: str, interval: str, year: int, month: int, session: requests.Session) -> str:
    path = kline_month_path(root, symbol, interval, year, month)
    marker = path + ".missing"
    if os.path.exists(marker):
        return "missing"
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return "cached"
    try:
        download(kline_month_url(symbol, interval, year, month), path, session=session)
        return "downloaded"
    except VisionMissing:
        os.makedirs(os.path.dirname(marker), exist_ok=True)
        open(marker, "w").close()
        return "missing"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="data/binance/symbols_selected.json", help="JSON with lists per interval, or a plain list")
    ap.add_argument("--intervals", nargs="+", default=["1h", "1m"])
    ap.add_argument("--start", default="2017-08")
    ap.add_argument("--end", default="2026-09")
    ap.add_argument("--root", default="data/binance/raw")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)

    with open(args.symbols) as f:
        spec = json.load(f)
    jobs = []
    for interval in args.intervals:
        symbols = spec[interval] if isinstance(spec, dict) else spec
        for symbol in symbols:
            for y, m in months(args.start, args.end):
                jobs.append((symbol, interval, y, m))
    log.info("%d files to check", len(jobs))
    counts = {"downloaded": 0, "cached": 0, "missing": 0, "failed": 0}
    sessions = [requests.Session() for _ in range(args.workers)]
    for s in sessions:
        s.headers["User-Agent"] = "edge-project-m2 (research seminar, HSE)"
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(task, args.root, sym, iv, y, m, sessions[i % args.workers]): (sym, iv, y, m) for i, (sym, iv, y, m) in enumerate(jobs)}
        for i, fut in enumerate(as_completed(futures), 1):
            key = futures[fut]
            try:
                counts[fut.result()] += 1
            except Exception as exc:
                counts["failed"] += 1
                log.error("%s failed: %r", key, exc)
            if i % 200 == 0 or i == len(futures):
                log.info("%d/%d files: %s", i, len(futures), counts)
    log.info("done: %s", counts)
    return 0 if counts["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
