from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from edgespread import clip_zero, estimate
from edgespread.data_moex import is_share_mask

METHODS = ("EDGE", "GMM", "OHL", "CHL", "OHLC", "CHLO", "AR", "CS", "ROLL")
log = logging.getLogger("compute_moex_daily")


def clean(df: pd.DataFrame, drop_no_trade_days: bool) -> pd.DataFrame:
    df = df.copy()
    df = df[is_share_mask(df, secid_col="SECID")]
    df["NUMTRADES"] = df["NUMTRADES"].fillna(0).astype(int)
    traded = df["NUMTRADES"] > 0
    ok = traded & df[["OPEN", "HIGH", "LOW", "CLOSE"]].notna().all(axis=1) & (df["LOW"] <= df["HIGH"]) & (df["OPEN"].between(df["LOW"], df["HIGH"])) & (df["CLOSE"].between(df["LOW"], df["HIGH"])) & (df["LOW"] > 0)
    bad_prices = traded & ~ok
    log.info("%d rows, %d traded days with inconsistent prices dropped", len(df), int(bad_prices.sum()))
    df = df[~bad_prices]
    if drop_no_trade_days:
        df = df[df["NUMTRADES"] > 0]
    else:
        df = df.sort_values(["SECID", "TRADEDATE"])
        prev_close = df.groupby("SECID")["CLOSE"].ffill()
        no_trade = df["NUMTRADES"] == 0
        for c in ("OPEN", "HIGH", "LOW", "CLOSE"):
            df.loc[no_trade, c] = prev_close[no_trade]
        df = df[df["CLOSE"].notna()]
    return df.sort_values(["SECID", "TRADEDATE"]).reset_index(drop=True)


def stock_month_estimates(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for secid, g in df.groupby("SECID", sort=False):
        g = g.sort_values("TRADEDATE")
        months = g["TRADEDATE"].dt.to_period("M").to_numpy()
        uniq, starts = np.unique(months, return_index=True)
        ends = np.append(starts[1:], len(g))
        n_months = len(uniq)
        width = int(np.max(ends - starts)) + 1
        arrs = {c: np.full((n_months, width), np.nan) for c in ("OPEN", "HIGH", "LOW", "CLOSE")}
        vals = {c: g[c].to_numpy(dtype=float) for c in arrs}
        for i, (s, e) in enumerate(zip(starts, ends)):
            lo = s - 1
            n = e - lo
            if lo < 0:
                for c in arrs:
                    arrs[c][i, 1 : 1 + e - s] = vals[c][s:e]
            else:
                for c in arrs:
                    arrs[c][i, :n] = vals[c][lo:e]
        est = estimate(arrs["OPEN"], arrs["HIGH"], arrs["LOW"], arrs["CLOSE"], methods=METHODS, sign=True)
        numtr = g["NUMTRADES"].to_numpy()
        traded = numtr > 0
        o, h, l, c = (vals[k] for k in ("OPEN", "HIGH", "LOW", "CLOSE"))
        for i, (s, e) in enumerate(zip(starts, ends)):
            sl = slice(s, e)
            tr = traded[sl]
            row = {"secid": secid, "month": str(uniq[i]), "n_days": int(e - s), "n_traded_days": int(tr.sum()), "avg_trades_per_day": float(numtr[sl].mean()), "price": float(c[e - 1])}
            if tr.any():
                row["frac_oh"] = float(np.mean(o[sl][tr] == h[sl][tr]))
                row["frac_ol"] = float(np.mean(o[sl][tr] == l[sl][tr]))
                row["frac_ch"] = float(np.mean(c[sl][tr] == h[sl][tr]))
                row["frac_cl"] = float(np.mean(c[sl][tr] == l[sl][tr]))
            for m in METHODS:
                row[m] = float(est[m][i])
            rows.append(row)
    out = pd.DataFrame(rows)
    out = out[(out["n_days"] >= 3) & (out["n_traded_days"] >= 1)]
    return out


def tiers_by_month(df: pd.DataFrame) -> pd.Series:
    r = df.groupby("month")["avg_trades_per_day"].rank(pct=True)
    return pd.Series(np.where(r <= 0.5, "Small", np.where(r <= 0.8, "Mid", "Large")), index=df.index)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/moex/daily_tqbr.parquet")
    ap.add_argument("--out", default="results/moex")
    ap.add_argument("--drop-no-trade-days", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_parquet(args.input)
    df = clean(df, args.drop_no_trade_days)
    log.info("%d securities, %d rows after cleaning", df["SECID"].nunique(), len(df))
    sm = stock_month_estimates(df)
    sm.to_csv(os.path.join(args.out, "stock_month_estimates.csv"), index=False)
    log.info("%d stock-months", len(sm))

    sm["tier"] = tiers_by_month(sm)
    sm["frac"] = sm[["frac_oh", "frac_ol", "frac_ch", "frac_cl"]].mean(axis=1)
    fig1 = sm.dropna(subset=["frac"]).groupby(["month", "tier"])["frac"].mean().rename("prob").reset_index()
    fig1["month"] = pd.PeriodIndex(fig1["month"], freq="M").to_timestamp()
    fig1.to_csv(os.path.join(args.out, "fig1_prob_oc_hl.csv"), index=False)

    long = sm.melt(id_vars=["month", "tier"], value_vars=["EDGE", "AR", "CS", "ROLL"], var_name="estimator", value_name="estimate").dropna()
    long["estimate"] = clip_zero(long["estimate"])
    by_tier = long.groupby(["month", "tier", "estimator"])["estimate"].agg(mean_estimate="mean", n="count").reset_index()
    by_tier["month"] = pd.PeriodIndex(by_tier["month"], freq="M").to_timestamp()
    by_tier.to_csv(os.path.join(args.out, "spread_by_tier_month.csv"), index=False)

    per_stock = sm.groupby("secid")["avg_trades_per_day"].mean()
    quint = pd.qcut(per_stock.rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    sm["numtrd_quintile"] = sm["secid"].map(quint)
    rows = []
    for q, g in sm.groupby("numtrd_quintile", observed=True):
        g_valid = g.dropna(subset=["EDGE"])
        med_trd = float(g_valid["avg_trades_per_day"].median())
        med_str = f"{med_trd:.1f}" if med_trd < 10 else f"{med_trd:.0f}"
        row = {"Numtrd quintile": int(q), "stock-months": len(g_valid), "median trades/day": med_str, "P[o/c=h/l]": f"{100 * g_valid['frac'].mean():.1f}"}
        for m in ("EDGE", "GMM", "OHLC", "CHLO", "OHL", "CHL", "AR", "CS", "ROLL"):
            v = g_valid[m].dropna()
            row[f"{m} mean"] = f"{100 * clip_zero(v).mean():.2f}"
            row[f"{m} zero%"] = f"{100 * (v <= 0).mean():.0f}"
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "table4E_by_numtrades.csv"), index=False)
    log.info("wrote tables to %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
