"""Feature engineering: log returns, rolling volatility, range, forward target."""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLS = ["log_ret", "roll_vol", "hl_range"]


def build_features(df: pd.DataFrame, vol_window: int = 21) -> pd.DataFrame:
    """R_t = ln(P_t / P_{t-1}); roll_vol = 21d std of R; hl_range = ln(High/Low)."""
    out = pd.DataFrame(index=df.index)
    out["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))
    out["roll_vol"] = out["log_ret"].rolling(vol_window).std()
    out["hl_range"] = np.log(df["High"] / df["Low"])
    return out.dropna()


def make_target(log_ret: pd.Series, horizon: int = 1) -> pd.Series:
    """1-day ahead realised volatility: |R_{t+1}|.

    Predicting tomorrow's absolute log-return as a proxy for next-day volatility.
    This avoids overlapping multi-day windows that cause heavy autocorrelation in
    training targets — keeping the target clean, non-overlapping, and mathematically
    honest (each row's label is independent of its neighbours).

    The last row is NaN because tomorrow has not happened yet.
    The `horizon` parameter is kept for API compatibility but is always treated as 1.
    """
    return log_ret.abs().shift(-1).rename("target_vol")
