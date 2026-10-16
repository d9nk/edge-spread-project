from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from edgespread.data_moex import HISTORY_COLUMNS, history_by_date, session

log = logging.getLogger("download_moex_history")


def day_path(root: str, day: dt.date) -> str:
    return os.path.join(root, f"{day.year}", f"history_{day.isoformat()}.csv")


def task(root: str, day: dt.date, sess) -> str:
    path = day_path(root, day)
    if os.path.exists(path):
        return "cached"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = history_by_date(day, sess)
    tmp = path + ".part"
    if df.empty:
        open(tmp, "w").close()
    else:
        df.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return "downloaded" if not df.empty else "empty"


def combine(root: str, out: str) -> pd.DataFrame:
    frames = []
    for year in sorted(os.listdir(root)):
        ydir = os.path.join(root, year)
        if not os.path.isdir(ydir):
            continue
        for name in sorted(os.listdir(ydir)):
            path = os.path.join(ydir, name)
            if name.endswith(".csv") and os.path.getsize(path) > 0:
                frames.append(pd.read_csv(path, dtype={"SECID": str, "SHORTNAME": str}))
    if not frames:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])
    for c in ("NUMTRADES", "VALUE", "OPEN", "LOW", "HIGH", "CLOSE", "LEGALCLOSEPRICE", "WAPRICE", "VOLUME", "MARKETPRICE2"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values(["SECID", "TRADEDATE"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_parquet(out, index=False)
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2013-03-25", help="first day of the TQBR board history")
    ap.add_argument("--end", default=(dt.date.today() - dt.timedelta(days=1)).isoformat())
    ap.add_argument("--root", default="data/moex/raw/history")
    ap.add_argument("--out", default="data/moex/daily_tqbr.parquet")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    days = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
    log.info("%d calendar days to check", len(days))
    sessions = [session() for _ in range(args.workers)]
    counts = {"downloaded": 0, "cached": 0, "empty": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(task, args.root, d, sessions[i % args.workers]): d for i, d in enumerate(days)}
        for i, fut in enumerate(as_completed(futures), 1):
            d = futures[fut]
            try:
                counts[fut.result()] += 1
            except Exception as exc:
                counts["failed"] += 1
                log.error("%s failed: %r", d, exc)
            if i % 100 == 0 or i == len(futures):
                log.info("%d/%d days: %s", i, len(futures), counts)
    df = combine(args.root, args.out)
    if not df.empty:
        log.info("combined %d rows, %d securities, %s .. %s -> %s", len(df), df["SECID"].nunique(), df["TRADEDATE"].min().date(), df["TRADEDATE"].max().date(), args.out)
    else:
        log.warning("combined 0 rows -> %s", args.out)
    return 0 if counts["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
