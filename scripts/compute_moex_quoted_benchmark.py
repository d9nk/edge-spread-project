from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from edgespread import estimate
from edgespread.benchmark import trade_side_proxy
from edgespread.data_moex import MOSCOW, candles, is_share_mask, quoted_spread_by_day, read_quote_log, regular_grid, securities, session

METHODS = ("EDGE", "GMM", "AR", "CS", "ROLL")
log = logging.getLogger("compute_moex_quoted_benchmark")


def cached_candles(secid: str, day: dt.date, interval: int, root: str, sess) -> pd.DataFrame:
    path = os.path.join(root, f"{interval}m", day.isoformat(), f"{secid}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["begin"])
        return df
    df = candles(secid, interval, day, day, sess)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return df


def estimates_from_candles(df: pd.DataFrame, freq: str) -> dict:
    out = {}
    if len(df) < 3:
        return out
    grid = regular_grid(df, freq, fill="prev_close")
    o, h, l, c = (grid[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    tag = freq.replace("min", "m")
    for m, v in estimate(o, h, l, c, methods=METHODS, sign=True).items():
        out[f"{m}_{tag}"] = float(v)
    out[f"n_bars_{tag}"] = int(len(grid))
    out[f"empty_bars_{tag}"] = float((grid["volume"] == 0).mean())
    if freq == "1min":
        grid_nan = regular_grid(df, freq, fill="nan")
        o, h, l, c = (grid_nan[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
        out["EDGE_1m_fill_nan"] = float(estimate(o, h, l, c, methods=("EDGE",), sign=True)["EDGE"])
    return out


def effective_from_trades(secid: str, day: dt.date, trades_root: str) -> dict:
    path = os.path.join(trades_root, day.isoformat(), f"{secid}.csv.gz")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    if len(df) < 2 or "BUYSELL" not in df.columns:
        return {"n_trades": int(len(df))}
    df = df[df["BUYSELL"].isin(["B", "S"])]
    t = pd.DataFrame({
        "time": pd.to_datetime(day.isoformat() + " " + df["TRADETIME"].astype(str)),
        "price": df["PRICE"].astype(float),
        "qty": df["QUANTITY"].astype(float),
        "is_buyer_maker": df["BUYSELL"].eq("S").to_numpy(),
    })
    prox = trade_side_proxy(t, max_gap=pd.Timedelta(minutes=5))
    e = prox["eff_proxy"].dropna()
    return {
        "n_trades": int(len(df)),
        "eff_proxy_mean": float(e.mean()) if len(e) else np.nan,
        "eff_proxy_rms": float(np.sqrt(np.mean(e * e))) if len(e) else np.nan,
        "eff_proxy_coverage": float(len(e) / max(len(prox), 1)),
    }


def process_security(secid: str, day: dt.date, root: str, sess, trades_root: str = "data/moex/trades") -> dict:
    row = {"secid": secid, "date": day.isoformat()}
    row.update(effective_from_trades(secid, day, trades_root))
    for interval, freq in ((1, "1min"), (10, "10min")):
        df = cached_candles(secid, day, interval, root, sess)
        if interval == 1:
            row["n_candles_1m"] = int(len(df))
            row["volume"] = float(df["volume"].sum()) if len(df) else 0.0
            row["value"] = float(df["value"].sum()) if len(df) else 0.0
        row.update(estimates_from_candles(df, freq))
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quotes", default="data/moex/quotes")
    ap.add_argument("--candles", default="data/moex/candles")
    ap.add_argument("--out", default="results/moex")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--include-today", action="store_true")
    ap.add_argument("--min-coverage", type=float, default=0.5, help="minimum share of the observed session time with a valid two-sided quote")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    os.makedirs(args.out, exist_ok=True)

    today = dt.datetime.now(MOSCOW).date()
    logs = sorted(f for f in os.listdir(args.quotes) if f.startswith("moex_quotes_") and f.endswith(".csv"))
    frames = []
    sessions = [session() for _ in range(args.workers)]
    secs = securities(sessions[0])
    share_ids = set(secs.loc[is_share_mask(secs, secid_col="SECID"), "SECID"])
    for name in logs:
        day = dt.date.fromisoformat(name[len("moex_quotes_") : -4])
        if day >= today and not args.include_today:
            log.info("skipping %s (not finished)", day)
            continue
        quotes = read_quote_log(os.path.join(args.quotes, name))
        if quotes.empty:
            continue
        qs = quoted_spread_by_day(quotes)
        qs = qs[qs["secid"].isin(share_ids) & (qs["coverage"] >= args.min_coverage) & (qs["n_changes"] >= 2)]
        log.info("%s: %d shares with quote coverage >= %.0f%%", day, len(qs), 100 * args.min_coverage)
        rows = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_security, secid, day, args.candles, sessions[i % args.workers]): secid for i, secid in enumerate(qs["secid"])}
            for i, fut in enumerate(as_completed(futures), 1):
                try:
                    rows.append(fut.result())
                except Exception as exc:
                    log.error("%s %s failed: %r", futures[fut], day, exc)
                if i % 100 == 0:
                    log.info("%s: %d/%d securities", day, i, len(futures))
        if not rows:
            log.warning("%s: no securities passed the filters, skipping the day", day)
            continue
        est = pd.DataFrame(rows)
        qs = qs.assign(date=qs["date"].astype(str))
        frames.append(qs.merge(est, on=["secid", "date"], how="left"))
    if not frames:
        log.info("nothing to do")
        return 0
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(os.path.join(args.out, "quoted_benchmark_stock_day.csv"), index=False)
    ok = df.dropna(subset=["quoted_spread", "EDGE_1m"])
    if len(ok) > 2:
        log.info("%d stock-days; corr(quoted, EDGE_1m) = %.3f, corr(quoted, AR_1m) = %.3f, corr(quoted, CS_1m) = %.3f", len(ok), ok["quoted_spread"].corr(np.maximum(0, ok["EDGE_1m"])), ok["quoted_spread"].corr(np.maximum(0, ok["AR_1m"])), ok["quoted_spread"].corr(np.maximum(0, ok["CS_1m"])))
    if "eff_proxy_mean" in df.columns:
        ok = df.dropna(subset=["eff_proxy_mean", "EDGE_1m"])
        if len(ok) > 2:
            log.info("%d stock-days with trades; corr(effective proxy, EDGE_1m) = %.3f, AR_1m %.3f, CS_1m %.3f", len(ok), ok["eff_proxy_mean"].corr(np.maximum(0, ok["EDGE_1m"])), ok["eff_proxy_mean"].corr(np.maximum(0, ok["AR_1m"])), ok["eff_proxy_mean"].corr(np.maximum(0, ok["CS_1m"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
