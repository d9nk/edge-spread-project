from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.signal import lfilter

FillRule = Literal["prev_close", "prev_bar"]


@dataclass(frozen=True)
class SimConfig:
    trades: int = 390
    prob: float = 1.0
    spread: float = 0.01
    volatility: float = 0.03
    overnight: float = 0.0
    drift: float = 0.0
    buy_prob: float = 0.5
    tick_size: float = 0.0
    rho: float = 0.0
    vol_pattern: tuple[float, ...] | None = None
    reset_period: int | None = None
    fill: FillRule = "prev_close"
    label: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        if self.trades < 1:
            raise ValueError("trades must be at least 1")
        if not 0 < self.prob <= 1:
            raise ValueError("prob must be in (0, 1]")
        if not 0 <= self.buy_prob <= 1:
            raise ValueError("buy_prob must be in [0, 1]")
        if self.spread < 0 or self.volatility < 0 or self.overnight < 0 or self.tick_size < 0:
            raise ValueError("spread, volatility, overnight and tick_size must be non-negative")
        if not -1 < self.rho < 1:
            raise ValueError("rho must be in (-1, 1)")
        if self.vol_pattern is not None:
            pat = np.asarray(self.vol_pattern, dtype=float)
            if pat.shape != (self.trades,) or np.any(pat < 0):
                raise ValueError("vol_pattern must have one non-negative entry per trade")
            if not np.isclose(pat.mean(), 1.0):
                raise ValueError("vol_pattern must have mean one")
        if self.reset_period is not None and self.reset_period < 1:
            raise ValueError("reset_period must be positive")
        if self.fill not in ("prev_close", "prev_bar"):
            raise ValueError("fill must be 'prev_close' or 'prev_bar'")


@dataclass
class _State:
    log_price: float = 0.0
    prev_close: float = float("nan")
    prev_bar: np.ndarray | None = None
    ar_state: np.ndarray | None = None
    periods_done: int = 0


def _ffill_index(valid: np.ndarray) -> np.ndarray:
    idx = np.where(valid, np.arange(valid.size), -1)
    return np.maximum.accumulate(idx)


def _ohlc_from_trades(p: np.ndarray, keep: np.ndarray, state: _State, fill: FillRule) -> np.ndarray:
    n, k = p.shape
    any_kept = keep.any(axis=1)
    first = keep.argmax(axis=1)
    last = k - 1 - keep[:, ::-1].argmax(axis=1)
    rows = np.arange(n)
    pk = np.where(keep, p, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        high = np.nanmax(pk, axis=1)
        low = np.nanmin(pk, axis=1)
    bars = np.column_stack([p[rows, first], high, low, p[rows, last]])
    if not any_kept.all():
        src = _ffill_index(any_kept)
        empty = ~any_kept
        has_src = src >= 0
        if fill == "prev_close":
            filled = np.where(has_src, bars[np.maximum(src, 0), 3], state.prev_close)
            bars[empty, :] = filled[empty][:, None]
        else:
            prev = state.prev_bar
            if prev is None:
                prev = np.full(4, state.prev_close)
            bars[empty & has_src, :] = bars[np.maximum(src, 0)[empty & has_src], :]
            bars[empty & ~has_src, :] = prev
    state.prev_close = float(bars[-1, 3])
    state.prev_bar = bars[-1].copy()
    return bars


def simulate(n_periods: int, cfg: SimConfig | None = None, rng: np.random.Generator | None = None, chunk: int = 4096) -> np.ndarray:
    cfg = cfg or SimConfig()
    rng = rng or np.random.default_rng()
    if n_periods < 1:
        raise ValueError("n_periods must be positive")
    if chunk < 1:
        raise ValueError("chunk must be positive")
    out = np.empty((n_periods, 4))
    state = _State()
    sd_trade = cfg.volatility / np.sqrt(cfg.trades)
    pattern = None if cfg.vol_pattern is None else np.asarray(cfg.vol_pattern, dtype=float)
    first_price_set = False
    start = 0
    while start < n_periods:
        n = min(chunk, n_periods - start)
        if cfg.reset_period is not None:
            next_reset = cfg.reset_period - (state.periods_done % cfg.reset_period)
            n = min(n, next_reset)
            if state.periods_done % cfg.reset_period == 0:
                state.log_price = 0.0
                state.prev_bar = None
                state.ar_state = None
                first_price_set = False
        m = n * cfg.trades
        eps = rng.normal(0.0, sd_trade, size=m)
        if pattern is not None:
            eps *= np.tile(pattern, n)
        if cfg.rho != 0.0:
            eps *= 1.0 - cfg.rho
            zi = state.ar_state if state.ar_state is not None else np.zeros(1)
            eps, state.ar_state = lfilter([1.0], [1.0, -cfg.rho], eps, zi=zi)
        r = eps + cfg.drift / cfg.trades
        if cfg.overnight > 0:
            r[0::cfg.trades] += rng.normal(0.0, cfg.overnight, size=n)
        log_fund = state.log_price + np.cumsum(r)
        state.log_price = float(log_fund[-1])
        side = np.where(rng.random(m) < cfg.buy_prob, 1.0, -1.0)
        p = np.exp(log_fund) * (1.0 + 0.5 * cfg.spread * side)
        if cfg.tick_size > 0:
            p = np.maximum(np.round(p / cfg.tick_size), 1.0) * cfg.tick_size
        keep = np.ones(m, dtype=bool) if cfg.prob >= 1 else rng.random(m) < cfg.prob
        if not first_price_set:
            state.prev_close = float(p[0])
            first_price_set = True
        bars = _ohlc_from_trades(p.reshape(n, cfg.trades), keep.reshape(n, cfg.trades), state, cfg.fill)
        out[start : start + n] = bars
        start += n
        state.periods_done += n
    return out


def windows_with_lag(bars: np.ndarray, n_obs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if n_obs < 1:
        raise ValueError("n_obs must be positive")
    n_periods = bars.shape[0]
    n_windows = n_periods // n_obs
    if n_windows < 1:
        raise ValueError("fewer periods than n_obs")
    idx = np.arange(n_windows)[:, None] * n_obs + np.arange(-1, n_obs)[None, :]
    first = idx < 0
    idx = np.where(first, 0, idx)
    cols = []
    for j in range(4):
        w = bars[idx, j]
        w = np.where(first, np.nan, w)
        cols.append(w)
    return cols[0], cols[1], cols[2], cols[3]
