from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="data/binance/daily/symbols.csv")
    ap.add_argument("--out", default="data/binance/symbols_selected.json")
    ap.add_argument("--n-hourly-usdt", type=int, default=120)
    ap.add_argument("--n-hourly-btc", type=int, default=30)
    ap.add_argument("--n-minute", type=int, default=22)
    ap.add_argument("--n-hourly-new", type=int, default=30, help="most traded USDT pairs listed after --listed-before")
    ap.add_argument("--listed-before", default="2021-01-01", help="pairs listed before this date are preferred (they cover the paper's sample)")
    args = ap.parse_args()

    df = pd.read_csv(args.symbols)
    df = df[df["n_days"] >= 90]
    df["first_day"] = pd.to_datetime(df["first_day"])
    old_cut = pd.Timestamp(args.listed_before)
    usdt = df[df["quote"] == "USDT"].sort_values("n_trades", ascending=False).reset_index(drop=True)
    btc = df[df["quote"] == "BTC"].sort_values("n_trades", ascending=False).reset_index(drop=True)
    usdt_old = usdt[usdt["first_day"] < old_cut].reset_index(drop=True)
    usdt_new = usdt[usdt["first_day"] >= old_cut].reset_index(drop=True)
    hourly = usdt_old["symbol"].head(args.n_hourly_usdt).tolist() + usdt_new["symbol"].head(args.n_hourly_new).tolist() + btc["symbol"].head(args.n_hourly_btc).tolist()

    minute = ["BTCUSDT", "ETHUSDT", "ETHBTC"]
    new_picks = [s for s in usdt_new["symbol"].head(3).tolist() if s not in minute]
    target_old = max(0, args.n_minute - len(minute) - len(new_picks))
    cand_old = usdt_old[~usdt_old["symbol"].isin(minute)].reset_index(drop=True)
    if target_old > 0 and len(cand_old) > 0:
        grid = np.exp(np.linspace(np.log(1), np.log(len(cand_old)), target_old * 4))
        for r in np.round(grid).astype(int):
            sym = cand_old.loc[r - 1, "symbol"]
            if sym not in minute:
                minute.append(sym)
                if len(minute) >= args.n_minute - len(new_picks):
                    break
        if len(minute) < args.n_minute - len(new_picks):
            for sym in cand_old["symbol"]:
                if sym not in minute:
                    minute.append(sym)
                    if len(minute) >= args.n_minute - len(new_picks):
                        break
    minute = (minute + new_picks)[: args.n_minute]
    hourly = sorted(set(hourly) | set(minute))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"1h": hourly, "1m": minute}, f, indent=2)
    info = df.set_index("symbol")
    print(f"hourly: {len(hourly)} pairs; minute: {len(minute)} pairs")
    for s in minute:
        print(f"  {s:12s} listed {info.loc[s, 'first_day'].date()}  trades {info.loc[s, 'n_trades'] / 1e6:8.1f}M  rank among USDT pairs {int(usdt.index[usdt['symbol'] == s][0]) + 1 if s in set(usdt['symbol']) else '-'}")
    return 0


if __name__ == "__main__":
    main()
