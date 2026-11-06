"""Value-at-Risk maths."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, t

SIGMA_FLOOR = 1e-4


def z_score(confidence: float, distribution: str = "normal", t_df: float = 5.0) -> float:
    """One-sided quantile multiplier. 99% normal -> 2.326 (the overview rounds to 2.33)."""
    if distribution == "normal":
        return float(norm.ppf(confidence))
    if distribution == "t":
        # Student-t rescaled to unit variance so that sigma keeps its meaning
        return float(t.ppf(confidence, t_df) * np.sqrt((t_df - 2) / t_df))
    raise ValueError(f"unknown distribution {distribution!r}")


def var_threshold(sigma, confidence=0.99, mu=0.0, distribution="normal", t_df=5.0):
    """Return threshold (log-return): a next-day return BELOW this is a VaR breach."""
    return mu - z_score(confidence, distribution, t_df) * np.asarray(sigma)


def var_amount(sigma, portfolio_value, confidence=0.99, mu=0.0, distribution="normal", t_df=5.0):
    """Dollar VaR as a POSITIVE loss number (e.g. 52,000 means 'could lose 52k')."""
    thr = var_threshold(sigma, confidence, mu, distribution, t_df)
    return portfolio_value * (1.0 - np.exp(thr))


def var_floor_price(close: pd.Series, sigma: pd.Series, **kw) -> pd.Series:
    """Price level below which tomorrow's close would be a breach (for the dashboard)."""
    thr = pd.Series(var_threshold(sigma.values, **kw), index=sigma.index)
    return close.reindex(sigma.index) * np.exp(thr)


# ---------------------------------------------------------------------------
# McNeil & Frey (2000) — Volatility-Filtered Historical Simulation
# ---------------------------------------------------------------------------
def calibrate_quantile(
    sigma_val: pd.Series,
    log_ret_val: pd.Series,
    confidence: float = 0.99,
) -> float:
    """
    Compute the empirical (1 - confidence) left-tail quantile of
    LSTM-standardised residuals on the validation set.

    McNeil & Frey (2000) show that if the conditional volatility model is
    correct, the standardised residuals  e_t = R_t / sigma_t  are i.i.d.
    Taking the empirical alpha-quantile of e_t and using it as the VaR
    multiplier produces well-calibrated tail coverage without imposing
    a parametric distributional assumption (normal or t).

    Returns
    -------
    cal_q : float
        A *negative* number, e.g. -2.47.
        Use:  var_threshold = sigma_test * cal_q   (already on loss side)
    """
    aligned = pd.DataFrame({"s": sigma_val, "r": log_ret_val}).dropna()
    aligned = aligned[aligned["s"] > SIGMA_FLOOR]
    if len(aligned) < 30:
        raise ValueError(
            f"Only {len(aligned)} clean validation observations — "
            "too few to calibrate an empirical quantile reliably."
        )
    std_resid = aligned["r"] / aligned["s"]
    alpha = 1.0 - confidence          # e.g. 0.01 for 99% VaR
    return float(np.percentile(std_resid.values, alpha * 100))


def var_threshold_calibrated(sigma, cal_q: float) -> np.ndarray:
    """
    Calibrated VaR threshold using the empirical quantile *cal_q*.
    A return BELOW this threshold is a VaR breach.
    cal_q is negative (left tail), so threshold = sigma * cal_q < 0.
    """
    return cal_q * np.asarray(sigma)
