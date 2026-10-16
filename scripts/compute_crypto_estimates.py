from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from edgespread import estimate
from edgespread.data_binance import read_klines

METHODS = ("EDGE", "GMM", "OHL", "CHL", "OHLC", "CHLO", "AR", "CS", "ROLL")
log = logging.getLogger("compute_crypto_estimates")


def month_row(symbol: str, yyyymm: int, freq: str, bars: pd.DataFrame) -> dict | None:
    if len(bars) < 3:
        return None
    o, h, l, c = (bars[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    est = estimate(o, h, l, c, methods=METHODS, sign=True)
    close = bars["close"].to_numpy(dtype=float)
    row = {
        "SYMBOL": symbol,
        "YYYYMM": yyyymm,
        "FREQ": freq,
        "NUMTRD": int(bars["n_trades"].sum()),
        "PRC": float(close[-1]),
        "VOL": float(bars["volume"].sum()),
        "PRCVOL": float((bars["close"] * bars["volume"]).sum()),
        "SDRET": float(np.std(close[1:] / close[:-1], ddof=1)) if len(close) > 2 else np.nan,
        "n_bars": int(len(bars)),
        "empty_share": float((bars["n_trades"] == 0).mean()),
    }
    row.update({m: float(v) for m, v in est.items()})
    return row


def process_zip(symbol: str, freq: str, path: str) -> dict | None:
    bars = read_klines(path)
    name = os.path.basename(path)
    yyyymm = int(name[-11:-7] + name[-6:-4])
    return month_row(symbol, yyyymm, freq, bars)


def process_daily(symbol: str, path: str) -> list[dict]:
    df = pd.read_parquet(path)
    rows = []
    for period, g in df.groupby(df["open_time"].dt.to_period("M")):
        r = month_row(symbol, int(period.strftime("%Y%m")), "1d", g.sort_values("open_time"))
        if r is not None:
            rows.append(r)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="data/binance/symbols_selected.json")
    ap.add_argument("--klines", default="data/binance/raw/klines")
    ap.add_argument("--daily", default="data/binance/daily")
    ap.add_argument("--authors", default="data/authors/binance_spread_estimates_dataverse_9AVA2B.csv", help="the authors' released estimates (downloaded from Harvard Dataverse if missing)")
    ap.add_argument("--out", default="results/crypto")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    os.makedirs(args.out, exist_ok=True)

    with open(args.symbols) as f:
        spec = json.load(f)
    symbols = sorted(set(spec["1h"]) | set(spec["1m"]))
    jobs = []
    for symbol in symbols:
        for freq in ("1h", "1m"):
            d = os.path.join(args.klines, symbol, freq)
            if os.path.isdir(d):
                for name in sorted(os.listdir(d)):
                    if name.endswith(".zip"):
                        jobs.append((symbol, freq, os.path.join(d, name)))
    log.info("%d monthly kline files, %d symbols", len(jobs), len(symbols))
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_zip, *j) for j in jobs]
        futures += [pool.submit(process_daily, s, os.path.join(args.daily, f"{s}.parquet")) for s in symbols if os.path.exists(os.path.join(args.daily, f"{s}.parquet"))]
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            if res is None:
                continue
            if isinstance(res, list):
                rows.extend(res)
            else:
                rows.append(res)
            if i % 500 == 0:
                log.info("%d/%d files", i, len(futures))
    df = pd.DataFrame(rows).sort_values(["SYMBOL", "YYYYMM", "FREQ"]).reset_index(drop=True)
    df.to_csv(os.path.join(args.out, "pair_month_estimates.csv"), index=False)
    log.info("%d pair-month-frequency rows", len(df))

    if not os.path.exists(args.authors):
        os.makedirs(os.path.dirname(args.authors), exist_ok=True)
        r = requests.get("https://dataverse.harvard.edu/api/access/datafile/10372530?format=original", timeout=120)
        r.raise_for_status()
        with open(args.authors, "wb") as f:
            f.write(r.content)
        log.info("downloaded the authors' estimates to %s", args.authors)
    authors = pd.read_csv(args.authors)
    ours = df[df["FREQ"] == "1m"][["SYMBOL", "YYYYMM", "EDGE"]].rename(columns={"EDGE": "ours"})
    val = authors.merge(ours, on=["SYMBOL", "YYYYMM"], how="inner").rename(columns={"SPREAD": "authors"})
    val["abs_diff"] = (val["authors"] - val["ours"]).abs()
    val.to_csv(os.path.join(args.out, "validation_vs_authors.csv"), index=False)
    if len(val):
        log.info("validation: %d pair-months, max |diff| %.2e, mean |diff| %.2e, identical to 1e-8: %d", len(val), val["abs_diff"].max(), val["abs_diff"].mean(), int((val["abs_diff"] < 1e-8).sum()))

    long = df.melt(id_vars=["SYMBOL", "YYYYMM", "FREQ"], value_vars=["EDGE", "AR", "CS", "ROLL", "GMM"], var_name="estimator", value_name="estimate").dropna()
    counts = long.groupby(["SYMBOL", "YYYYMM", "estimator"])["FREQ"].transform("nunique")
    long = long[counts == 3]
    long["estimate"] = np.maximum(0.0, long["estimate"])
    means = long.groupby(["YYYYMM", "estimator", "FREQ"]).agg(mean_estimate=("estimate", "mean"), n_pairs=("SYMBOL", "nunique")).reset_index()
    means["month"] = pd.to_datetime(means["YYYYMM"].astype(str), format="%Y%m")
    means = means.rename(columns={"FREQ": "freq"})
    means.to_csv(os.path.join(args.out, "fig8_monthly_means.csv"), index=False)

    a = authors.copy()
    a["estimate"] = np.maximum(0.0, a["SPREAD"])
    am = a.groupby("YYYYMM").agg(mean_estimate=("estimate", "mean"), n_pairs=("SYMBOL", "nunique")).reset_index()
    am["month"] = pd.to_datetime(am["YYYYMM"].astype(str), format="%Y%m")
    am.to_csv(os.path.join(args.out, "fig8_authors_1m_means.csv"), index=False)
    log.info("wrote Fig. 8 data to %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
