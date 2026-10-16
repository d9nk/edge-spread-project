from __future__ import annotations

import argparse
import datetime as dt
import gzip
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from edgespread.data_moex import ISS, MOSCOW, _get_json, securities, session

TRADE_COLUMNS = "TRADENO,TRADETIME,SECID,PRICE,QUANTITY,VALUE,BUYSELL,TRADINGSESSION,TRADEDATE"
log = logging.getLogger("fetch_moex_trades")


def fetch_security(secid: str, sess, page: int = 5000) -> pd.DataFrame:
    url = f"{ISS}/engines/stock/markets/shares/boards/TQBR/securities/{secid}/trades.json"
    frames = []
    start = 0
    while True:
        payload = _get_json(sess, url, {"iss.meta": "off", "iss.only": "trades", "trades.columns": TRADE_COLUMNS, "limit": page, "start": start})
        block = payload["trades"]
        df = pd.DataFrame(block["data"], columns=block["columns"])
        if df.empty:
            break
        frames.append(df)
        if len(df) < page:
            break
        start += len(df)
    if not frames:
        return pd.DataFrame(columns=TRADE_COLUMNS.split(","))
    return pd.concat(frames, ignore_index=True).drop_duplicates("TRADENO").sort_values("TRADENO")


def task(secid: str, out_dir: str, sess) -> int:
    path = os.path.join(out_dir, f"{secid}.csv.gz")
    if os.path.exists(path):
        return -1
    df = fetch_security(secid, sess)
    tmp = path + ".part"
    with gzip.open(tmp, "wt") as f:
        df.to_csv(f, index=False)
    os.replace(tmp, path)
    return len(df)


def run_once(root: str, workers: int) -> None:
    now = dt.datetime.now(MOSCOW)
    day = now.date()
    out_dir = os.path.join(root, day.isoformat())
    os.makedirs(out_dir, exist_ok=True)
    secs = securities()["SECID"].tolist()
    sessions = [session() for _ in range(workers)]
    total = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(task, s, out_dir, sessions[i % workers]): s for i, s in enumerate(secs)}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                n = fut.result()
                total += max(n, 0)
            except Exception as exc:
                log.error("%s failed: %r", futures[fut], exc)
            if i % 100 == 0 or i == len(futures):
                log.info("%s: %d/%d securities, %d trades", day, i, len(futures), total)


def sleep_until(hhmm: str) -> None:
    hour, minute = (int(x) for x in hhmm.split(":"))
    now = dt.datetime.now(MOSCOW)
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += dt.timedelta(days=1)
    log.info("sleeping until %s Moscow time", target.isoformat(timespec="minutes"))
    time.sleep((target - now).total_seconds())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/moex/trades")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--wait-until", default=None, help="HH:MM Moscow time to start (after the evening session, e.g. 23:56)")
    ap.add_argument("--daily", action="store_true", help="repeat every day at --wait-until")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    while True:
        if args.wait_until:
            sleep_until(args.wait_until)
        if dt.datetime.now(MOSCOW).weekday() < 5:
            run_once(args.root, args.workers)
        else:
            log.info("weekend, nothing to fetch")
        if not args.daily:
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
