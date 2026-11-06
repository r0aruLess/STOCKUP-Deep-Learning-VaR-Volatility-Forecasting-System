"""Classic benchmarks the LSTM must beat (or at least match) to be worth using."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_std_sigma(log_ret: pd.Series, window: int = 21) -> pd.Series:
    return log_ret.rolling(window).std().rename("sigma_rolling")


def ewma_sigma(log_ret: pd.Series, lam: float = 0.94) -> pd.Series:
    """RiskMetrics EWMA volatility: var_t = lam*var_{t-1} + (1-lam)*r_t^2."""
    var = log_ret.pow(2).ewm(alpha=1 - lam, adjust=False).mean()
    return np.sqrt(var).rename("sigma_ewma")


def historical_threshold(log_ret: pd.Series, window: int = 250, confidence: float = 0.99) -> pd.Series:
    """Historical-simulation VaR: empirical (1-confidence) quantile of the last `window` returns."""
    return log_ret.rolling(window).quantile(1 - confidence).rename("threshold_hist")
