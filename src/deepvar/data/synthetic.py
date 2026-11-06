"""Synthetic OHLCV generator (GARCH(1,1) + fat tails + rare jumps).

ONLY for unit tests, CI and offline demos. Never report results from this
data as if they were real market results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_ohlcv(
    n_days: int = 4000,
    start: str = "2015-01-01",
    seed: int = 0,
    p0: float = 100.0,
    uncond_vol: float = 0.025,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    alpha, beta = 0.10, 0.88
    omega = uncond_vol**2 * (1 - alpha - beta)
    nu = 6.0

    var_t = uncond_vol**2
    rets = np.empty(n_days)
    sig = np.empty(n_days)
    for t in range(n_days):
        sig[t] = np.sqrt(var_t)
        eps = rng.standard_t(nu) / np.sqrt(nu / (nu - 2))
        if rng.random() < 0.003:  # rare market-wide shock
            eps *= 4
        rets[t] = sig[t] * eps
        var_t = omega + alpha * rets[t] ** 2 + beta * var_t

    close = p0 * np.exp(np.cumsum(rets))
    prev_close = np.concatenate([[p0], close[:-1]])
    open_ = prev_close * np.exp(rng.normal(0, 0.15 * sig))
    hi_extra = np.abs(rng.normal(0, 0.6 * sig))
    lo_extra = np.abs(rng.normal(0, 0.6 * sig))
    high = np.maximum(open_, close) * np.exp(hi_extra)
    low = np.minimum(open_, close) * np.exp(-lo_extra)
    volume = np.exp(rng.normal(15, 0.4, n_days) + 8 * (sig - uncond_vol))

    idx = pd.date_range(start=start, periods=n_days, freq="D", name="Date")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )
