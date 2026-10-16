from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter

RESULTS = "results"
REPORTS = "reports"

COLORS = {"EDGE": "#1f77b4", "AR": "#ff7f0e", "CS": "#d62728", "ROLL": "#8c564b", "OHL": "#e377c2", "CHL": "#e377c2", "OHLC": "#9467bd", "CHLO": "#9467bd", "GMM": "#17becf", "HJ": "#111111", "QS": "#666666", "EFF": "#111111"}
MARKERS = {"EDGE": "o", "AR": "^", "CS": "s", "ROLL": "D", "OHL": "v", "CHL": "x", "OHLC": "d", "CHLO": "+", "GMM": "*"}
LINESTYLES = {"OHL": "--", "CHL": "--", "OHLC": "--", "CHLO": "--", "GMM": "--"}
FREQ_COLORS = {"1d": "#1f77b4", "1h": "#ff7f0e", "1m": "#d62728"}
FREQ_LABELS = {"1d": "Daily prices", "1h": "Hourly prices", "1m": "Minute prices"}

PAPER_TABLE2 = {
    "A": {
        "mean": {"EDGE": [0.44, 0.90, 2.88, 4.87, 7.84], "OHLC": [0.46, 0.88, 2.87, 4.86, 7.78], "CHLO": [0.46, 0.88, 2.88, 4.87, 7.79], "OHL": [0.79, 1.03, 2.92, 4.92, 7.88], "CHL": [0.79, 1.03, 2.93, 4.93, 7.89], "AR": [0.70, 0.95, 2.92, 4.97, 7.99], "CS": [0.60, 1.03, 2.95, 4.90, 7.86], "ROLL": [1.44, 1.59, 2.95, 4.90, 7.93]},
        "sd": {"EDGE": [0.33, 0.42, 0.41, 0.42, 0.45], "OHLC": [0.40, 0.55, 0.69, 0.81, 1.11], "CHLO": [0.39, 0.55, 0.69, 0.81, 1.10], "OHL": [0.79, 0.86, 0.73, 0.62, 0.64], "CHL": [0.79, 0.86, 0.72, 0.62, 0.64], "AR": [0.77, 0.85, 0.70, 0.58, 0.54], "CS": [0.49, 0.58, 0.61, 0.61, 0.62], "ROLL": [1.43, 1.49, 1.83, 2.14, 2.63]},
    },
    "B": {
        "mean": {"EDGE": [0.71, 0.95, 2.89, 5.02, 8.19], "OHLC": [0.77, 0.99, 2.76, 4.89, 8.10], "CHLO": [0.79, 0.99, 2.76, 4.92, 8.06], "OHL": [0.89, 1.11, 2.86, 5.01, 8.23], "CHL": [0.91, 1.10, 2.86, 5.04, 8.20], "AR": [0.65, 0.81, 2.26, 4.04, 6.59], "CS": [0.02, 0.04, 0.35, 1.17, 2.66], "ROLL": [1.44, 1.56, 2.89, 4.83, 7.71]},
        "sd": {"EDGE": [0.75, 0.83, 0.83, 0.81, 0.96], "OHLC": [0.87, 0.97, 1.23, 1.32, 1.59], "CHLO": [0.88, 0.96, 1.23, 1.33, 1.62], "OHL": [0.96, 1.03, 1.20, 1.13, 1.24], "CHL": [0.97, 1.04, 1.19, 1.13, 1.26], "AR": [0.73, 0.80, 0.92, 0.85, 0.94], "CS": [0.07, 0.10, 0.36, 0.62, 0.96], "ROLL": [1.42, 1.47, 1.82, 2.12, 2.65]},
    },
}
TABLE2_SPREADS = [0.005, 0.01, 0.03, 0.05, 0.08]
PAPER_METHODS = ["EDGE", "OHLC", "CHLO", "OHL", "CHL", "AR", "CS", "ROLL"]
LIQUID_MINUTES = 400


def _md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in df.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def _style():
    plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 200, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})


def _save(fig, name: str):
    os.makedirs(REPORTS, exist_ok=True)
    path = os.path.join(REPORTS, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def _exists(*parts) -> bool:
    return os.path.exists(os.path.join(*parts))


def fig2():
    path = os.path.join(RESULTS, "sim", "fig2_bias.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.axhline(0.01, color="black", linestyle=(0, (5, 5)), linewidth=1)
    for m in ("EDGE", "AR", "CS"):
        ax.plot(df["trades_per_day"], np.maximum(0, df[m]), marker=MARKERS[m], color=COLORS[m], label=m, markersize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Average Number of Trades per Day")
    ax.set_ylabel("Bid-Ask Spread")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=2))
    ax.set_ylim(-0.0003, 0.0112)
    ax.set_xticks([1, 10, 100])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.legend(title="Estimator", ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.suptitle("Our reproduction of Fig. 2: spread 1%, one path of 210,000 days per point", fontsize=9, x=0.12, ha="left", y=1.04)
    _save(fig, "fig2_bias_reproduction.png")


def fig3():
    path = os.path.join(RESULTS, "sim", "table2A.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for m in ("EDGE", "GMM", "AR", "CS", "OHL", "CHL", "OHLC", "CHLO"):
        d = df[df["estimator"] == m].sort_values("spread")
        ax.plot(d["spread"], d["sd"], marker=MARKERS[m], color=COLORS[m], linestyle=LINESTYLES.get(m, "-"), label=m, markersize=7, markerfacecolor="none" if m in LINESTYLES else None)
    ax.set_xlabel("Bid-Ask Spread")
    ax.set_ylabel("Standard Deviation")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=2))
    ax.set_ylim(0.002, 0.0125)
    ax.legend(title="Estimator", ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    fig.suptitle("Our reproduction of Fig. 3: sd of monthly estimates, 10,000 months, 390 trades/day", fontsize=9, x=0.12, ha="left", y=1.08)
    _save(fig, "fig3_variance_reproduction.png")


def table2() -> str | None:
    lines = []
    for panel in ("A", "B"):
        path = os.path.join(RESULTS, "sim", f"table2{panel}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df = df[df["spread"].isin(TABLE2_SPREADS)]
        title = "Panel A: every trade observed (390 trades per day)" if panel == "A" else "Panel B: 1 % of trades observed (3.9 trades per day)"
        lines.append(f"\n**Table 2, {title}.** Monthly estimates across 10,000 simulated months of 21 days, in %. Each cell: ours / paper (mean) and ours / paper (sd).\n")
        lines.append("| S | " + " | ".join(PAPER_METHODS) + " |")
        lines.append("|---|" + "|".join(["---"] * len(PAPER_METHODS)) + "|")
        for i, s in enumerate(TABLE2_SPREADS):
            row_mean, row_sd = [], []
            for m in PAPER_METHODS:
                r = df[(df["spread"] == s) & (df["estimator"] == m)].iloc[0]
                row_mean.append(f"{100 * r['mean']:.2f} / {PAPER_TABLE2[panel]['mean'][m][i]:.2f}")
                row_sd.append(f"({100 * r['sd']:.2f} / {PAPER_TABLE2[panel]['sd'][m][i]:.2f})")
            lines.append(f"| {100 * s:.1f}% mean | " + " | ".join(row_mean) + " |")
            lines.append(f"| {100 * s:.1f}% sd | " + " | ".join(row_sd) + " |")
    if not lines:
        return None
    devs = []
    for panel in ("A", "B"):
        path = os.path.join(RESULTS, "sim", f"table2{panel}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        for i, s in enumerate(TABLE2_SPREADS):
            for m in PAPER_METHODS:
                r = df[(df["spread"] == s) & (df["estimator"] == m)].iloc[0]
                devs.append(abs(100 * r["mean"] - PAPER_TABLE2[panel]["mean"][m][i]))
    lines.append(f"\nLargest absolute deviation of a mean from the paper: {max(devs):.2f} pp (Monte Carlo standard error of a mean with sd 0.4-2.6 pp over 10,000 months: 0.004-0.03 pp, plus the paper's rounding to 0.01 pp).\n")
    return "\n".join(lines)


def robustness():
    path = os.path.join(RESULTS, "sim", "robust.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["variant"] = df["variant"].replace({"tick=0 reset": "independent"})
    methods = ["EDGE", "GMM", "AR", "CS", "ROLL"]
    order = ["paper", "independent", "tick=0.0005", "tick=0.002", "tick=0.005", "rho=-0.2", "rho=0.2", "rho=0.5", "ushape=1.0", "ushape=3.0", "buy_prob=0.6", "buy_prob=0.75", "overnight=0.01", "overnight=0.03", "fill=prev_bar"]
    labels = {"paper": "paper setup", "independent": "independent months", "tick=0.0005": "tick = 5% of spread", "tick=0.002": "tick = 20% of spread", "tick=0.005": "tick = 50% of spread", "rho=-0.2": "AR(1) returns, rho = -0.2", "rho=0.2": "AR(1) returns, rho = 0.2", "rho=0.5": "AR(1) returns, rho = 0.5", "ushape=1.0": "intraday U-shape, mild", "ushape=3.0": "intraday U-shape, strong", "buy_prob=0.6": "60% buys", "buy_prob=0.75": "75% buys", "overnight=0.01": "overnight vol 1%", "overnight=0.03": "overnight vol 3%", "fill=prev_bar": "gap fill: previous bar (SAS)"}
    probs = sorted(df["prob"].unique(), reverse=True)
    fig, axes = plt.subplots(1, len(probs), figsize=(4.6 * len(probs), 6.5), sharey=True)
    for ax, p in zip(np.atleast_1d(axes), probs):
        d = df[df["prob"] == p]
        variants = [v for v in order if v in set(d["variant"])]
        y = np.arange(len(variants))
        width = 0.8 / len(methods)
        for j, m in enumerate(methods):
            vals = [100 * d[(d["variant"] == v) & (d["estimator"] == m)]["mean"].iloc[0] for v in variants]
            ax.barh(y + (j - (len(methods) - 1) / 2) * width, vals, height=width, color=COLORS[m], label=m)
        ax.axvline(1.0, color="black", linestyle="--", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels([labels.get(v, v) for v in variants])
        ax.invert_yaxis()
        ax.set_xlabel("mean monthly estimate, % (true spread 1%)")
        ax.set_title(f"{p * 390:.1f} trades per day", fontsize=10)
        ax.set_xlim(0, 2.25)
    axes[0].legend(ncol=5, loc="upper center", bbox_to_anchor=(1.6, 1.12))
    _save(fig, "robustness_assumptions.png")
    rows = []
    for p in probs:
        for v in order:
            d = df[(df["prob"] == p) & (df["variant"] == v)]
            if d.empty:
                continue
            row = {"trades/day": f"{p * 390:.1f}", "variant": labels.get(v, v)}
            for m in methods:
                r = d[d["estimator"] == m].iloc[0]
                row[m] = f"{100 * r['mean']:.2f} ({100 * r['sd']:.2f})"
            rows.append(row)
    t = pd.DataFrame(rows)
    return "\n**Robustness of the simulation to the assumptions (true spread 1 %; mean and sd of 10,000 monthly estimates, negative reset to zero, in %).**\n\n" + _md(t) + "\n"


def negatives():
    path = os.path.join(RESULTS, "sim", "negatives.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    methods = ["EDGE", "AR", "CS", "ROLL"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for ax, prob in zip(axes, (1.0, 0.01)):
        d = df[(df["prob"] == prob) & (df["spread"] == 0.005)]
        x = np.arange(3)
        width = 0.2
        for j, m in enumerate(methods):
            dm = d[d["estimator"] == m].sort_values("n_obs")
            ax.bar(x + (j - 1.5) * width, 100 * dm["mean_clip_zero"], width, color=COLORS[m], label=f"{m}: reset to zero")
            ax.plot(x + (j - 1.5) * width, 100 * dm["rms_signed"], "k_", markersize=12, markeredgewidth=2)
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(["21 days", "63 days", "252 days"])
        ax.set_ylabel("mean estimate, %")
        ax.set_title(f"true spread 0.5%, {prob * 390:.1f} trades/day", fontsize=10)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Bars: average after resetting negatives to zero (paper). Black ticks: signed root-mean-square of the squared spreads.", fontsize=9)
    _save(fig, "negatives_window_length.png")
    rows = []
    for _, r in df[df["estimator"].isin(methods)].sort_values(["prob", "spread", "n_obs", "estimator"]).iterrows():
        rows.append({"trades/day": f"{r['prob'] * 390:.1f}", "spread": f"{100 * r['spread']:.1f}%", "window": int(r["n_obs"]), "estimator": r["estimator"], "negative share": f"{100 * r['neg_frac']:.0f}%", "mean, reset to 0": f"{100 * r['mean_clip_zero']:.2f}", "mean, abs": f"{100 * r['mean_abs']:.2f}", "signed RMS": f"{100 * r['rms_signed']:.2f}"})
    return "\n**Treatment of negative estimates and estimation window (in %).**\n\n" + _md(pd.DataFrame(rows)) + "\n"


def fig8() -> str | None:
    path = os.path.join(RESULTS, "crypto", "fig8_monthly_means.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["month"])
    authors_path = os.path.join(RESULTS, "crypto", "fig8_authors_1m_means.csv")
    authors = pd.read_csv(authors_path, parse_dates=["month"]) if os.path.exists(authors_path) else None
    summary_rows = []
    for tag, label, end in (("paper_period", "July 2017 - December 2021 (paper period)", pd.Timestamp("2021-12-31")), ("full", "July 2017 - September 2026 (full sample)", None)):
        d = df if end is None else df[df["month"] <= end]
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
        for ax, m in zip(axes, ("EDGE", "AR", "CS")):
            for f in ("1d", "1h", "1m"):
                dd = d[(d["estimator"] == m) & (d["freq"] == f)].sort_values("month")
                ax.plot(dd["month"], dd["mean_estimate"], color=FREQ_COLORS[f], label=FREQ_LABELS[f], linewidth=1.2)
            if m == "EDGE" and authors is not None:
                aa = authors if end is None else authors[authors["month"] <= end]
                ax.plot(aa["month"], aa["mean_estimate"], color="black", linestyle=":", linewidth=1.2, label="authors, minute, all pairs")
            ax.set_title(m)
            ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
            ax.xaxis.set_major_locator(mdates.YearLocator(1 if end is not None else 2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ymax = 1.05 * float(d[d["estimator"].isin(("EDGE", "AR", "CS"))]["mean_estimate"].max())
        axes[0].set_ylim(0, ymax)
        axes[0].set_ylabel("Bid-Ask Spread")
        axes[0].legend(title="Frequency", ncol=3, loc="upper center", bbox_to_anchor=(1.7, 1.28))
        n = d["n_pairs"].max() if "n_pairs" in d.columns else None
        fig.suptitle(f"Average spread across Binance pairs by month ({'July 2017 - December 2021, as Fig. 8' if end is not None else 'extended to 2026'}; {n} pairs)", y=1.12, fontsize=9)
        _save(fig, f"fig8_crypto_{tag}.png")
        for m in ("EDGE", "AR", "CS"):
            row = {"period": label, "estimator": m}
            for f in ("1d", "1h", "1m"):
                dd = d[(d["estimator"] == m) & (d["freq"] == f)]
                row[FREQ_LABELS[f]] = f"{100 * dd['mean_estimate'].mean():.2f}%"
            summary_rows.append(row)
    return "\n**Crypto (Fig. 8): average monthly spread estimates across 21 Binance pairs by frequency.**\n\n" + _md(pd.DataFrame(summary_rows)) + "\n"


def crypto_validation() -> str | None:
    path = os.path.join(RESULTS, "crypto", "validation_vs_authors.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(df["authors"], df["ours"], s=6, alpha=0.5)
    lim = [0, max(df["authors"].max(), df["ours"].max())]
    ax.plot(lim, lim, "k--", linewidth=1)
    ax.set_xlabel("authors' released estimate (EDGE, minute prices)")
    ax.set_ylabel("our estimate from the same Binance files")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title(f"{len(df)} pair-months; max |difference| = {df['abs_diff'].max():.1e}", fontsize=9)
    _save(fig, "crypto_validation_vs_authors.png")
    return f"\n**Validation against the authors' released estimates.** {len(df)} pair-months of EDGE from minute prices recomputed from the same Binance files: mean absolute difference {df['abs_diff'].mean():.2e}, maximum {df['abs_diff'].max():.2e}, {int((df['abs_diff'] < 1e-8).sum())} of {len(df)} identical to 1e-8.\n"


def crypto_benchmark() -> str | None:
    path = os.path.join(RESULTS, "crypto", "benchmark_pair_day.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    out = []
    bench = "eff_mean"
    freqs = [c for c in ("1s", "10s", "1m", "5m", "15m") if f"EDGE_{c}" in df.columns]
    rows = []
    for f in freqs:
        for m in ("EDGE", "GMM", "AR", "CS", "ROLL"):
            col = f"{m}_{f}"
            if col not in df.columns:
                continue
            d = df[[bench, col]].dropna()
            est = np.maximum(0, d[col])
            pos = est > 0
            rows.append({
                "bars": f,
                "estimator": m,
                "n pair-days": len(d),
                "Pearson corr": f"{100 * np.corrcoef(d[bench], est)[0, 1]:.1f}",
                "Spearman corr": f"{100 * d[bench].corr(est, method='spearman'):.1f}",
                "MAPE (log)": f"{100 * np.mean(np.abs(np.log(est[pos]) - np.log(d[bench][pos]))):.1f}",
                "zero share": f"{100 * np.mean(~pos):.1f}%",
                "mean estimate / mean benchmark": f"{est.mean() / d[bench].mean():.2f}",
            })
    table = pd.DataFrame(rows)
    out.append("\n**Binance futures: estimators from intraday bars vs the effective spread from trades and quotes (pair-days).**\n\n" + _md(table) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    x = np.arange(len(freqs))
    for m in ("EDGE", "GMM", "AR", "CS", "ROLL"):
        corr, zero = [], []
        for f in freqs:
            d = df[[bench, f"{m}_{f}"]].dropna()
            est = np.maximum(0, d[f"{m}_{f}"])
            corr.append(100 * np.corrcoef(d[bench], est)[0, 1])
            zero.append(100 * np.mean(est <= 0))
        axes[0].plot(x, corr, marker=MARKERS.get(m, "o"), color=COLORS[m], label=m)
        axes[1].plot(x, zero, marker=MARKERS.get(m, "o"), color=COLORS[m], label=m)
    for ax, lab in zip(axes, ("Pearson correlation with the effective spread, %", "share of non-positive estimates, %")):
        ax.set_xticks(x)
        ax.set_xticklabels(freqs)
        ax.set_xlabel("bar length")
        ax.set_ylabel(lab)
    axes[0].legend(ncol=5, loc="upper center", bbox_to_anchor=(1.1, 1.18))
    _save(fig, "crypto_benchmark_by_frequency.png")
    if "quoted_tw" in df.columns and "eff_proxy_mean" in df.columns:
        d = df[["eff_mean", "quoted_tw", "eff_proxy_mean"]].dropna()
        out.append(f"\n**Benchmarks on the same pair-days ({len(d)}):** mean effective spread {100 * d['eff_mean'].mean():.4f}%, mean time-weighted quoted spread {100 * d['quoted_tw'].mean():.4f}% (effective / quoted = {d['eff_mean'].mean() / d['quoted_tw'].mean():.2f}); the trade-side proxy without quotes has mean {100 * d['eff_proxy_mean'].mean():.4f}% and correlation {100 * np.corrcoef(d['eff_mean'], d['eff_proxy_mean'])[0, 1]:.1f}% with the true effective spread.\n")
    if "lee_ready_error" in df.columns:
        out.append(f"Lee-Ready classification error against the true aggressor flag: mean {100 * df['lee_ready_error'].mean():.1f}% of trades; share of trades executed exactly at the quote: {100 * df['share_at_quote'].mean():.1f}%.\n")
    if "quoted_tw" in df.columns and len(df) >= 30:
        d_all = df.dropna(subset=["quoted_tw", bench]).copy()
        d_all["regime"] = pd.qcut(d_all["quoted_tw"], 3, labels=["smallest third of quoted spreads", "middle third", "largest third"])
        rows = []
        for regime, g in d_all.groupby("regime", observed=True):
            row = {"quoted-spread tercile": regime, "n pair-days": len(g), "median quoted spread": f"{1e4 * g['quoted_tw'].median():.2f} bp", "median effective spread": f"{1e4 * g[bench].median():.2f} bp"}
            for f in ("1s", "10s", "1m"):
                for m in ("EDGE", "AR", "CS"):
                    col = f"{m}_{f}"
                    if col in g.columns:
                        sub = g[[bench, col]].dropna()
                        est = np.maximum(0, sub[col])
                        row[f"{m} {f}: Spearman / zero %"] = f"{100 * sub[bench].corr(est, method='spearman'):.0f} / {100 * np.mean(est <= 0):.0f}"
            rows.append(row)
        out.append("\n**The same comparison by spread regime.**\n\n" + _md(pd.DataFrame(rows)) + "\n")
    if "EDGE_1m" in df.columns:
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharex=True, sharey=True)
        for ax, m in zip(axes, ("EDGE", "AR", "CS", "ROLL")):
            d = df[[bench, f"{m}_1m"]].dropna()
            est = np.maximum(1e-6, d[f"{m}_1m"])
            ax.scatter(d[bench], est, s=10, alpha=0.6, color=COLORS[m])
            ax.plot([1e-6, 2e-2], [1e-6, 2e-2], "k--", linewidth=1)
            ax.axhline(1e-6, color="gray", linestyle=":", linewidth=0.8)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim([1e-5, 2e-2])
            ax.set_ylim([8e-7, 2e-2])
            ax.set_title(f"{m} from 1-minute bars", fontsize=10)
            ax.set_xlabel("effective spread of the day")
        axes[0].set_ylabel("estimate from bars of the same day")
        fig.suptitle("Binance futures pair-days: estimates vs the effective spread from trades matched to quotes (log axes; zero estimates shown at 1e-6)", fontsize=9, y=1.02)
        _save(fig, "crypto_benchmark_scatter_1m.png")
    return "\n".join(out)


def moex_figures() -> str | None:
    out = []
    p = os.path.join(RESULTS, "moex", "fig1_prob_oc_hl.csv")
    if os.path.exists(p):
        df = pd.read_csv(p, parse_dates=["month"])
        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        for tier, color in (("Small", "#d62728"), ("Mid", "#ff7f0e"), ("Large", "#1f77b4")):
            d = df[df["tier"] == tier].sort_values("month")
            ax.plot(d["month"], d["prob"], color=color, label=tier, linewidth=1.2)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_ylabel("Probability o/c = h/l")
        ax.legend(title="Liquidity tier (trades per day)", ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.13))
        fig.suptitle("MOEX TQBR stocks: analogue of Fig. 1 (share of days with open or close at the high or low)", fontsize=9, x=0.12, ha="left", y=1.05)
        _save(fig, "moex_fig1_prob_oc_hl.png")
    p = os.path.join(RESULTS, "moex", "spread_by_tier_month.csv")
    if os.path.exists(p):
        df = pd.read_csv(p, parse_dates=["month"])
        tiers = ["Small", "Mid", "Large"]
        fig, axes = plt.subplots(1, len(tiers), figsize=(12.5, 3.9))
        for ax, tier in zip(axes, tiers):
            for m in ("EDGE", "AR", "CS", "ROLL"):
                d = df[(df["tier"] == tier) & (df["estimator"] == m)].sort_values("month")
                ax.plot(d["month"], d["mean_estimate"], color=COLORS[m], label=m, linewidth=1.1)
            ax.axvline(pd.Timestamp("2022-02-25"), color="black", linestyle=":", linewidth=1)
            ax.set_title(tier, fontsize=11)
            ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
            ax.xaxis.set_major_locator(mdates.YearLocator(4))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axes[0].set_ylabel("average spread estimate")
        axes[1].legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.22))
        fig.suptitle("MOEX TQBR: monthly average of stock-level estimates from daily prices by liquidity tier; dotted line: trading halt of February-March 2022", fontsize=9, y=1.14)
        _save(fig, "moex_spread_by_tier.png")
    p = os.path.join(RESULTS, "moex", "table4E_by_numtrades.csv")
    if os.path.exists(p):
        df = pd.read_csv(p, dtype=str)
        out.append("\n**MOEX TQBR: analogue of Table 4 Panel E. Stock-months sorted into quintiles of average trades per day; mean estimate (negative reset to zero) and share of zero estimates, in %.**\n\n" + _md(df) + "\n")
    p = os.path.join(RESULTS, "moex", "quoted_benchmark_stock_day.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        benches = [("quoted_spread", "time-weighted quoted spread (collector)")]
        if "eff_proxy_mean" in df.columns and df["eff_proxy_mean"].notna().sum() >= 10:
            benches.append(("eff_proxy_mean", "effective spread from trades with the aggressor flag (trade-side proxy)"))
        def benchmark_rows(source: pd.DataFrame, bench: str, freqs: tuple[str, ...]) -> list[dict]:
            rows = []
            for f in freqs:
                cols = [f"{m}_{f}" for m in ("EDGE", "GMM", "AR", "CS", "ROLL") if f"{m}_{f}" in source.columns]
                if not cols:
                    continue
                d = source[[bench] + cols].dropna()
                if len(d) < 10:
                    continue
                for col in cols:
                    est = np.maximum(0, d[col])
                    pos = est > 0
                    rows.append({
                        "bars": f,
                        "estimator": col[: -len(f) - 1],
                        "n stock-days": len(d),
                        "Pearson corr": f"{100 * np.corrcoef(d[bench], est)[0, 1]:.1f}",
                        "Spearman corr": f"{100 * d[bench].corr(est, method='spearman'):.1f}",
                        "MAPE (log)": f"{100 * np.mean(np.abs(np.log(est[pos]) - np.log(d[bench][pos]))):.1f}" if pos.any() else "nan",
                        "zero share": f"{100 * np.mean(~pos):.1f}%",
                        "mean estimate / mean benchmark": f"{est.mean() / d[bench].mean():.2f}",
                    })
            return rows

        days = df["date"].nunique() if "date" in df.columns else 1
        for bench, label in benches:
            rows = benchmark_rows(df, bench, ("1m", "10m"))
            if rows:
                out.append(f"\n**MOEX TQBR: estimators from intraday candles vs the {label}; {days} trading day(s), stock-days.**\n\n" + _md(pd.DataFrame(rows)) + "\n")
            if "n_candles_1m" in df.columns:
                liquid = df[df["n_candles_1m"] >= LIQUID_MINUTES]
                rows = benchmark_rows(liquid, bench, ("1m",))
                if rows:
                    out.append(f"\n**The same comparison for the {rows[0]['n stock-days']} liquid stock-days with at least {LIQUID_MINUTES} traded minutes, vs the {label}.**\n\n" + _md(pd.DataFrame(rows)) + "\n")
        if "EDGE_1m" in df.columns:
            fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharex=True, sharey=True)
            cols_1m = [f"{m}_1m" for m in ("EDGE", "AR", "CS", "ROLL") if f"{m}_1m" in df.columns]
            d_common = df[["quoted_spread"] + cols_1m].dropna()
            for ax, m in zip(axes, ("EDGE", "AR", "CS", "ROLL")):
                est = np.maximum(1e-5, d_common[f"{m}_1m"])
                ax.scatter(d_common["quoted_spread"], est, s=10, alpha=0.6, color=COLORS[m])
                ax.plot([1e-5, 1e-1], [1e-5, 1e-1], "k--", linewidth=1)
                ax.axhline(1e-5, color="gray", linestyle=":", linewidth=0.8)
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlim([1e-4, 1e-1])
                ax.set_ylim([8e-6, 1e-1])
                ax.set_title(f"{m} from 1-minute candles", fontsize=10)
                ax.set_xlabel("quoted spread of the day")
            axes[0].set_ylabel("estimate from candles of the same day")
            fig.suptitle("MOEX TQBR stock-days: estimates vs the time-weighted quoted spread (log axes; zero estimates shown at 1e-5)", fontsize=9, y=1.02)
            _save(fig, "moex_quoted_benchmark_scatter_1m.png")
    return "\n".join(out) if out else None


def equations():
    items = [
        ("eq_correction.png", [
            r"$\tau_t = \mathbf{1}\{h_t \neq l_t \;\mathrm{or}\; l_t \neq c_{t-1}\},\qquad \eta_t = \frac{1}{2}(h_t + l_t)$",
            r"$S^2 = \frac{-8\;\mathbb{E}\left[(\bar{\eta}_t - \bar{o}_t)(o_t - \eta_{t-1})\right]}{\mathbb{P}[o_t \neq h_t,\ \tau_t = 1] + \mathbb{P}[o_t \neq l_t,\ \tau_t = 1]}$",
            r"$\bar{r}_t = r_t - \tau_t\,\mathbb{E}[r_t]/\mathbb{E}[\tau_t]$",
            "denominator = 2 with continuous prices (Abdi-Ranaldo),\nsmaller when open or close sit on the high or low",
        ]),
        ("eq_edge.png", [
            r"$x_{1,t} = \frac{1}{2}\,(\mathrm{OHL}_t + \mathrm{CHL}_t),\qquad x_{2,t} = \frac{1}{2}\,(\mathrm{OHLC}_t + \mathrm{CHLO}_t)$",
            r"$\mathbb{E}[x_{1,t}] = \mathbb{E}[x_{2,t}] = S^2$",
            r"$\hat{S}^2_{\mathrm{EDGE}} = w_1\,\bar{x}_1 + w_2\,\bar{x}_2,\qquad w_1 = \frac{\widehat{\mathrm{Var}}[x_2]}{\widehat{\mathrm{Var}}[x_1] + \widehat{\mathrm{Var}}[x_2]}$",
            r"$\hat{S} = \sqrt{\max(0,\ \hat{S}^2)}$",
        ]),
    ]
    for name, lines in items:
        fig = plt.figure(figsize=(7.6, 4.4))
        fig.patch.set_facecolor("white")
        y = 0.9
        for line in lines:
            size = 19 if line.startswith("$") else 12
            fig.text(0.04, y, line, fontsize=size, va="top", color="#0F2D6B" if line.startswith("$") else "#444444")
            y -= 0.27 if line.startswith("$") else 0.2
        plt.axis("off")
        _save(fig, name)


def main() -> int:
    _style()
    equations()
    os.makedirs(REPORTS, exist_ok=True)
    sections = []
    fig2()
    fig3()
    t2 = table2()
    if t2:
        sections.append("## Simulation: Table 2\n" + t2)
    rb = robustness()
    if rb:
        sections.append("## Simulation: robustness to the assumptions\n" + rb)
    ng = negatives()
    if ng:
        sections.append("## Simulation: negative estimates\n" + ng)
    f8 = fig8()
    if f8:
        sections.append("## Crypto: Fig. 8 summary\n" + f8)
    cv = crypto_validation()
    if cv:
        sections.append("## Crypto: validation against the authors' numbers\n" + cv)
    cb = crypto_benchmark()
    if cb:
        sections.append("## Crypto: benchmark from trades and quotes\n" + cb)
    mx = moex_figures()
    if mx:
        sections.append("## MOEX\n" + mx)
    with open(os.path.join(REPORTS, "tables.md"), "w") as f:
        f.write("# Tables (auto-generated by scripts/make_figures.py)\n" + "\n".join(sections))
    print("wrote", os.path.join(REPORTS, "tables.md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
