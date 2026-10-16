from __future__ import annotations

import argparse
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from edgespread import SimConfig, clip_zero, estimate, simulate, windows_with_lag

ALL_METHODS = ("EDGE", "GMM", "OHL", "CHL", "OHLC", "CHLO", "AR", "AR2", "CS", "CS2", "ROLL")
PAPER_METHODS = ("EDGE", "OHLC", "CHLO", "OHL", "CHL", "AR", "CS", "ROLL")
N_SIM = 10_000
N_OBS = 21
SEED = 20261012


def _seed(*parts) -> int:
    words = [SEED, *[zlib.crc32(str(p).encode("utf-8")) for p in parts]]
    return int(np.random.SeedSequence(words).generate_state(1)[0])


def run_single_path(cfg: SimConfig, n_periods: int, seed: int) -> dict:
    bars = simulate(n_periods, cfg, np.random.default_rng(seed))
    o, h, l, c = bars.T
    est = estimate(o, h, l, c, methods=ALL_METHODS, sign=True)
    return {**{k: v for k, v in asdict(cfg).items() if k != "vol_pattern"}, "n_periods": n_periods, **est}


def run_panel(cfg: SimConfig, n_sim: int, n_obs: int, seed: int, independent: bool = False) -> pd.DataFrame:
    if independent:
        cfg = replace(cfg, reset_period=n_obs + 1)
        bars = simulate(n_sim * (n_obs + 1), cfg, np.random.default_rng(seed))
        w = bars.reshape(n_sim, n_obs + 1, 4)
        o, h, l, c = (w[:, :, j] for j in range(4))
    else:
        bars = simulate(n_sim * n_obs, cfg, np.random.default_rng(seed))
        o, h, l, c = windows_with_lag(bars, n_obs)
    est = estimate(o, h, l, c, methods=ALL_METHODS, sign=True)
    return pd.DataFrame(est)


def summarize_panel(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in df.columns:
        s = clip_zero(df[m].to_numpy())
        rows.append({"estimator": m, "mean": np.nanmean(s), "sd": np.nanstd(s, ddof=1), "zero_frac": np.nanmean(s == 0), "nan_frac": np.mean(np.isnan(s))})
    return pd.DataFrame(rows)


def _panel_task(args):
    name, cfg, n_sim, n_obs, extra = args
    t0 = time.time()
    df = run_panel(cfg, n_sim, n_obs, _seed(name, cfg.label, cfg.prob, cfg.spread, n_obs), independent=bool(extra.get("independent", False)))
    summary = summarize_panel(df)
    for k, v in extra.items():
        summary[k] = v
    summary["seconds"] = time.time() - t0
    return name, cfg, extra, df, summary


def _path_task(args):
    name, cfg, n_periods, extra = args
    t0 = time.time()
    res = run_single_path(cfg, n_periods, _seed(name, cfg.label, cfg.prob, cfg.spread))
    res.update(extra)
    res["seconds"] = time.time() - t0
    return name, res


def u_shape(trades: int, depth: float) -> tuple[float, ...]:
    x = np.linspace(-1.0, 1.0, trades)
    pat = 1.0 + depth * (x * x - np.mean(x * x))
    pat = np.maximum(pat, 0.05)
    return tuple(pat / pat.mean())


def build_tasks(experiments: list[str]) -> tuple[list, list]:
    panel_tasks, path_tasks = [], []
    if "fig2" in experiments:
        probs = np.exp(np.linspace(np.log(1 / 390), np.log(1.0), 15))
        for p in probs:
            cfg = SimConfig(prob=float(p), spread=0.01, label="fig2")
            path_tasks.append(("fig2", cfg, N_SIM * N_OBS, {"trades_per_day": float(p) * 390}))
    if "fig3" in experiments or "table2" in experiments:
        spreads = sorted({0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08})
        for s in spreads:
            cfg = SimConfig(prob=1.0, spread=s, label="panelA")
            panel_tasks.append(("table2A", cfg, N_SIM, N_OBS, {"panel": "A", "spread": s}))
        if "table2" in experiments:
            for s in (0.005, 0.01, 0.03, 0.05, 0.08):
                cfg = SimConfig(prob=0.01, spread=s, label="panelB")
                panel_tasks.append(("table2B", cfg, N_SIM, N_OBS, {"panel": "B", "spread": s}))
    if "robust" in experiments:
        base = SimConfig(spread=0.01)
        variants: list[tuple[str, SimConfig]] = [("paper", base)]
        for tick in (0.0005, 0.002, 0.005):
            variants.append((f"tick={tick}", replace(base, tick_size=tick)))
        variants.append(("independent", base))
        for rho in (-0.2, 0.2, 0.5):
            variants.append((f"rho={rho}", replace(base, rho=rho)))
        for depth in (1.0, 3.0):
            variants.append((f"ushape={depth}", replace(base, vol_pattern=u_shape(base.trades, depth))))
        for bp in (0.6, 0.75):
            variants.append((f"buy_prob={bp}", replace(base, buy_prob=bp)))
        for on in (0.01, 0.03):
            variants.append((f"overnight={on}", replace(base, overnight=on)))
        variants.append(("fill=prev_bar", replace(base, fill="prev_bar")))
        for label, cfg in variants:
            for prob in (1.0, 20 / 390, 0.01):
                if label == "fill=prev_bar" and prob == 1.0:
                    continue
                cfg_p = replace(cfg, prob=prob, label=label)
                independent = (label == "independent") or label.startswith("tick=")
                panel_tasks.append(("robust", cfg_p, N_SIM, N_OBS, {"variant": label, "prob": prob, "trades_per_day": prob * 390, "independent": independent}))
    if "negatives" in experiments:
        for prob in (1.0, 0.01):
            for s in (0.005, 0.01, 0.03):
                for n_obs in (21, 63, 252):
                    n_sim = N_SIM if n_obs == 21 else max(2000, N_SIM * 21 // n_obs)
                    cfg = SimConfig(prob=prob, spread=s, label="negatives")
                    panel_tasks.append(("negatives", cfg, n_sim, n_obs, {"prob": prob, "spread": s, "n_obs": n_obs, "n_sim": n_sim}))
    return panel_tasks, path_tasks


def negatives_summary(df: pd.DataFrame, extra: dict) -> pd.DataFrame:
    rows = []
    for m in df.columns:
        s = df[m].to_numpy()
        s = s[~np.isnan(s)]
        rows.append({
            **extra,
            "estimator": m,
            "mean_clip_zero": np.mean(clip_zero(s)),
            "mean_abs": np.mean(np.abs(s)),
            "mean_signed": np.mean(s),
            "rms_signed": np.sign(np.mean(np.sign(s) * s * s)) * np.sqrt(abs(np.mean(np.sign(s) * s * s))),
            "neg_frac": np.mean(s < 0),
            "sd_clip_zero": np.std(clip_zero(s), ddof=1),
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", nargs="+", default=["fig2", "fig3", "table2", "robust", "negatives"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default="results/sim")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "raw"), exist_ok=True)

    panel_tasks, path_tasks = build_tasks(args.experiments)
    print(f"{len(panel_tasks)} panel tasks, {len(path_tasks)} path tasks, {args.workers} workers", flush=True)
    t0 = time.time()
    panel_summaries: dict[str, list[pd.DataFrame]] = {}
    negatives: list[pd.DataFrame] = []
    path_results: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_panel_task, t) for t in panel_tasks] + [pool.submit(_path_task, t) for t in path_tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            if len(res) == 5:
                name, cfg, extra, df, summary = res
                tag = "_".join(f"{k}={v}" for k, v in extra.items() if k in ("panel", "spread", "variant", "prob", "n_obs"))
                df.to_parquet(os.path.join(args.out, "raw", f"{name}_{tag}.parquet".replace("/", "")), index=False)
                panel_summaries.setdefault(name, []).append(summary)
                if name == "negatives":
                    negatives.append(negatives_summary(df, extra))
            else:
                name, r = res
                path_results.append(r)
            print(f"[{i}/{len(futures)}] {res[0]} done, {time.time() - t0:.0f}s elapsed", flush=True)

    if path_results:
        df = pd.DataFrame(path_results).sort_values("trades_per_day")
        df.to_csv(os.path.join(args.out, "fig2_bias.csv"), index=False)
    for name, parts in panel_summaries.items():
        df = pd.concat(parts, ignore_index=True)
        sort_cols = [c for c in ("panel", "variant", "prob", "spread", "n_obs", "estimator") if c in df.columns]
        df = df.sort_values(sort_cols)
        df.to_csv(os.path.join(args.out, f"{name}.csv"), index=False)
    if negatives:
        pd.concat(negatives, ignore_index=True).to_csv(os.path.join(args.out, "negatives.csv"), index=False)
    print(f"all done in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
