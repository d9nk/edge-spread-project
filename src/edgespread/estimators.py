from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np

METHODS = ("EDGE", "GMM", "OHL", "CHL", "OHLC", "CHLO", "AR", "AR2", "CS", "CS2", "ROLL")

_SQRT2 = np.sqrt(2.0)
_CS_K = 3.0 - 2.0 * _SQRT2


def _nanmean(x: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(x, axis=1)


def _log_prices(*prices: object) -> tuple[list[np.ndarray], bool]:
    arrays: list[np.ndarray] = []
    was_1d = None
    for p in prices:
        a = np.asarray(p, dtype=float)
        if a.ndim == 1:
            a = a[None, :]
            one_d = True
        elif a.ndim == 2:
            one_d = False
        else:
            raise ValueError("prices must be 1-D or 2-D arrays")
        if was_1d is None:
            was_1d = one_d
        elif was_1d != one_d:
            raise ValueError("open, high, low and close must all be 1-D or all be 2-D")
        arrays.append(a)
    shapes = {a.shape for a in arrays}
    if len(shapes) != 1:
        raise ValueError("open, high, low and close must have the same shape")
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = [np.where(a > 0, np.log(np.where(a > 0, a, 1.0)), np.nan) for a in arrays]
    return logs, bool(was_1d)


def _finish(values: np.ndarray, was_1d: bool) -> np.ndarray | float:
    return float(values[0]) if was_1d else values


def spread_from_squared(s2: np.ndarray | float, sign: bool) -> np.ndarray | float:
    s = np.sqrt(np.abs(s2))
    if sign:
        s = s * np.sign(s2)
    return s


def _indicator(cond: np.ndarray, *inputs: np.ndarray) -> np.ndarray:
    missing = np.zeros(cond.shape, dtype=bool)
    for x in inputs:
        missing |= np.isnan(x)
    return np.where(missing, np.nan, cond.astype(float))


def edge_components(open, high, low, close) -> dict[str, np.ndarray]:
    (o, h, l, c), _ = _log_prices(open, high, low, close)
    eta = (h + l) / 2.0

    h1, l1, c1, eta1 = h[:, :-1], l[:, :-1], c[:, :-1], eta[:, :-1]
    o, h, l, eta = o[:, 1:], h[:, 1:], l[:, 1:], eta[:, 1:]

    tau = _indicator((h != l) | (l != c1), h, l, c1)
    phi_oh = _indicator(o != h, o, h) * tau
    phi_ol = _indicator(o != l, o, l) * tau
    phi_ch = _indicator(c1 != h1, c1, h1) * tau
    phi_cl = _indicator(c1 != l1, c1, l1) * tau

    p_tau = _nanmean(tau)
    n_tau = np.nansum(tau, axis=1)
    p_o = _nanmean(phi_oh) + _nanmean(phi_ol)
    p_c = _nanmean(phi_ch) + _nanmean(phi_cl)

    with np.errstate(divide="ignore", invalid="ignore"):
        pi_o = -8.0 / p_o
        pi_c = -8.0 / p_c

    r_eo = eta - o
    r_oe = o - eta1
    r_ec = eta - c1
    r_ce = c1 - eta1
    r_oc = o - c1

    def demean(r: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return r - tau * (_nanmean(r) / p_tau)[:, None]

    d_eo, d_ec, d_oc = demean(r_eo), demean(r_ec), demean(r_oc)

    with np.errstate(invalid="ignore"):
        s2_ohl = pi_o[:, None] * d_eo * r_oe
        s2_ohlc = pi_o[:, None] * d_eo * r_oc
        s2_chl = pi_c[:, None] * d_ec * r_ce
        s2_chlo = pi_c[:, None] * d_oc * r_ce

    x1 = 0.5 * (s2_ohl + s2_chl)
    x2 = 0.5 * (s2_ohlc + s2_chlo)
    with np.errstate(divide="ignore", invalid="ignore"):
        e1, e2 = _nanmean(x1), _nanmean(x2)
        v1 = _nanmean(x1 * x1) - e1 * e1
        v2 = _nanmean(x2 * x2) - e2 * e2
        vt = v1 + v2
        w1 = np.where(vt > 0, v2 / vt, 0.5)
    w2 = 1.0 - w1

    bad_o = (n_tau < 2) | ~(p_o > 0)
    bad_c = (n_tau < 2) | ~(p_c > 0)
    bad = bad_o | bad_c

    def masked(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return np.where(mask, np.nan, values)

    return {
        "OHL": masked(_nanmean(s2_ohl), bad_o),
        "OHLC": masked(_nanmean(s2_ohlc), bad_o),
        "CHL": masked(_nanmean(s2_chl), bad_c),
        "CHLO": masked(_nanmean(s2_chlo), bad_c),
        "GMM": masked(0.5 * (e1 + e2), bad),
        "EDGE": masked(w1 * e1 + w2 * e2, bad),
        "e1": masked(e1, bad),
        "e2": masked(e2, bad),
        "v1": masked(v1, bad),
        "v2": masked(v2, bad),
        "w1": masked(w1, bad),
        "w2": masked(w2, bad),
        "p_tau": p_tau,
        "p_o": p_o,
        "p_c": p_c,
        "n_tau": n_tau,
    }


def edge(open, high, low, close, sign: bool = False) -> np.ndarray | float:
    _, was_1d = _log_prices(open, high, low, close)
    s2 = edge_components(open, high, low, close)["EDGE"]
    return _finish(spread_from_squared(s2, sign), was_1d)


def ar(high, low, close, sign: bool = False, two_period: bool = False) -> np.ndarray | float:
    (h, l, c), was_1d = _log_prices(high, low, close)
    eta = (h + l) / 2.0
    c1, eta1, eta = c[:, :-1], eta[:, :-1], eta[:, 1:]
    s2_t = 4.0 * (c1 - eta1) * (c1 - eta)
    if two_period:
        s = _nanmean(np.sqrt(np.maximum(0.0, s2_t)))
    else:
        s = spread_from_squared(_nanmean(s2_t), sign)
    return _finish(s, was_1d)


def cs(high, low, close, sign: bool = False, two_period: bool = False) -> np.ndarray | float:
    (h, l, c), was_1d = _log_prices(high, low, close)
    h1, l1, c1 = h[:, :-1], l[:, :-1], c[:, :-1]
    h, l = h[:, 1:], l[:, 1:]
    gap = np.maximum(0.0, c1 - h) + np.minimum(0.0, c1 - l)
    ah, al = h + gap, l + gap
    beta = (h - l) ** 2 + (h1 - l1) ** 2
    gamma = (np.maximum(ah, h1) - np.minimum(al, l1)) ** 2
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / _CS_K - np.sqrt(gamma / _CS_K)
    s_t = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    if two_period:
        s = _nanmean(np.maximum(0.0, s_t))
    else:
        s = _nanmean(s_t)
        if not sign:
            s = np.abs(s)
    return _finish(s, was_1d)


def roll(close, sign: bool = False) -> np.ndarray | float:
    (c,), was_1d = _log_prices(close)
    r1 = c[:, 2:] - c[:, 1:-1]
    r2 = c[:, 1:-1] - c[:, :-2]
    valid = ~np.isnan(r1) & ~np.isnan(r2)
    n = valid.sum(axis=1)
    r1v = np.where(valid, r1, np.nan)
    r2v = np.where(valid, r2, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        cov = _nanmean(r1v * r2v) - _nanmean(r1v) * _nanmean(r2v)
        s2 = np.where(n >= 2, -4.0 * n / (n - 1.0) * cov, np.nan)
    return _finish(spread_from_squared(s2, sign), was_1d)


def estimate(open, high, low, close, methods: Iterable[str] = ("EDGE", "AR", "CS", "ROLL"), sign: bool = True):
    methods = tuple(m.upper() for m in methods)
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise ValueError(f"unknown methods {sorted(unknown)}; valid: {METHODS}")
    _, was_1d = _log_prices(open, high, low, close)
    out: dict[str, np.ndarray | float] = {}
    table1 = {"EDGE", "GMM", "OHL", "CHL", "OHLC", "CHLO"} & set(methods)
    if table1:
        comp = edge_components(open, high, low, close)
        for m in table1:
            out[m] = _finish(spread_from_squared(comp[m], sign), was_1d)
    if "AR" in methods:
        out["AR"] = ar(high, low, close, sign=sign)
    if "AR2" in methods:
        out["AR2"] = ar(high, low, close, sign=sign, two_period=True)
    if "CS" in methods:
        out["CS"] = cs(high, low, close, sign=sign)
    if "CS2" in methods:
        out["CS2"] = cs(high, low, close, sign=sign, two_period=True)
    if "ROLL" in methods:
        out["ROLL"] = roll(close, sign=sign)
    return {m: out[m] for m in methods}


def clip_zero(s):
    return np.maximum(0.0, s)
