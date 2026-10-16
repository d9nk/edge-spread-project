# EDGE: bid-ask spreads from OHLC prices — reproduction and extensions

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/d9nk/edge-spread-project/blob/main/EDGE.ipynb)

Research seminar, HSE University. We re-implement and test Ardia, Guidotti, Kroencke, [*Efficient estimation of bid-ask spreads from open, high, low, and close prices*](https://doi.org/10.1016/j.jfineco.2024.103916), JFE 161 (2024): the EDGE estimator of the effective spread from daily or intraday OHLC prices.

Kurbatov Daniil, Kuznetsov Georgiy, Gusev Dmitriy, Sorokin Grigoriy

## Results

- **Own implementation** of EDGE, its building blocks (OHL, CHL, OHLC, CHLO, GMM) and ROLL, CS, AR. Matches the authors' package to 1e-14 and reproduces 902 of their released Binance estimates to 1e-16.
- **Simulations reproduced:** all 80 means of Table 2 within 0.07 pp (EDGE within 0.02 pp), Fig. 2 and Fig. 3 with the same axes.
- **Crypto (Fig. 8)** on 21 Binance pairs: over the paper period (2017–2021) EDGE averages 0.18 / 0.14 / 0.13 % from daily / hourly / minute prices (AR 0.91 / 0.21 / 0.08 %, CS 0.87 / 0.18 / 0.02 %); extended through 2026, EDGE averages 0.13 / 0.09 / 0.09 % (AR 0.81 / 0.15 / 0.06 %, CS 0.76 / 0.14 / 0.01 %).
- **New benchmark the paper lacks:** Binance futures trades and best bid/ask (600 pair-days). EDGE from 1-second bars has a 95.6 % Pearson / 82.5 % Spearman correlation with the effective spread (AR 61.3 / 73.2 %, CS 21.5 / 53.4 %).
- **New market:** Moscow Exchange, 411 common and preferred shares and depositary receipts on the main board since 2013, delisted ones included (the ETFs, funds and mortgage certificates that share the board are excluded), 38,469 stock-months (37,427 with an EDGE estimate). The paper's Table 4 pattern holds: in the least traded quintile EDGE averages 3.1 %, AR 1.7 % and CS 0.4 %; small-stock spreads fell four-fold since 2014; the February 2022 crash and trading halt show in every estimator.
- **Our own MOEX quote feed:** quotes polled every 5 s; on one trading day (the collector ran from 15:28 to the close) across 218 TQBR shares, 10-minute EDGE has a 66 % Pearson / 55 % Spearman correlation with the time-weighted quoted spread and a 0.90 mean ratio (GMM 68 / 59 %, ratio 0.96), whereas CS is negatively correlated (-32 / -44 %); for the 97 shares with at least 400 traded minutes, 1-minute EDGE has a 93 % Spearman correlation (AR 68 %, CS 4 %).
- **Robustness:** tick size, AR(1) returns, a mild intraday U-shape and 1 % overnight jumps change EDGE by at most 0.07 pp; a strong U-shape or 3 % overnight jumps raise it by 0.1-0.2 pp while CS falls to 0.36 %; a 75 % buy share biases it to 0.75 % for a true 1 %. Reset-to-zero inflates small spreads in short windows. The authors' SAS code fills empty bars with the previous bar rather than the previous close (+0.07 pp on EDGE at 3.9 trades/day).

All tables are in `results/`, figures and `tables.md` in `reports/`.

## Run

```bash
git clone https://github.com/d9nk/edge-spread-project.git && cd edge-spread-project
pip install numpy pandas scipy matplotlib pyarrow requests tqdm bidask pytest
pytest                                     # 67 tests, about a second (fetches the authors' two test files once)
python scripts/run_simulations.py          # Table 2, Fig. 2, Fig. 3, robustness, negatives; 13 s on 14 cores, seeds fixed
python scripts/make_figures.py             # figures and tables.md from results/
```

Data steps (public sources, no accounts, resumable): `download_binance_daily_api.py`, `select_binance_symbols.py`, `download_binance_klines.py`, `compute_crypto_estimates.py`; `download_binance_futures_taq.py` (`--quick` for a 1 GB sample) and `compute_crypto_benchmark.py`; `download_moex_history.py`, `compute_moex_daily.py`; `collect_moex_quotes.py`, `fetch_moex_trades.py`, `compute_moex_quoted_benchmark.py`.

Or open `EDGE.ipynb` in Colab: it clones the repository, checks the estimator against the authors' package and test vectors, runs the tests, reruns the paper's simulations at a smaller scale, computes the estimates on the extracts in `data/sample/` and recomputes both benchmarks from the per-day tables in `results/`.

## Data

`data/sample/` holds two small extracts for the notebook (ETHUSDT daily klines, SBER and LSNG daily history); everything else is downloaded by the scripts. Sources: Binance Market Data (data.binance.vision) and REST API (the REST API refuses US addresses, so Colab uses the extract); Moscow Exchange ISS (weekend sessions since March 2025 are booked by ISS into the next business day; working Saturdays are separate days and included); the authors' released crypto estimates (Harvard Dataverse, doi:10.7910/DVN/9AVA2B, CC BY 4.0) and the test data of their `bidask` package.
