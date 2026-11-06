"""Tests for feature engineering and target construction.

Target was changed from 5-day ahead rolling RMS (overlapping) to
1-day ahead absolute log-return |R_{t+1}| (non-overlapping, no leakage).
"""
import numpy as np

from deepvar.data.features import FEATURE_COLS, build_features, make_target
from deepvar.data.windows import make_windows


def test_log_returns_formula(df):
    f = build_features(df, 21)
    manual = np.log(df["Close"] / df["Close"].shift(1)).loc[f.index]
    assert np.allclose(f["log_ret"], manual)
    assert not f.isna().any().any()
    assert list(f.columns) == FEATURE_COLS


def test_target_is_1day_ahead_absolute_return(df):
    """Target = |R_{t+1}| — 1-day ahead, non-overlapping."""
    f = build_features(df, 21)
    y = make_target(f["log_ret"], horizon=1)
    t = 100
    # For horizon=1 the target at t is |log_ret[t+1]|
    expected = abs(float(f["log_ret"].iloc[t + 1]))
    assert np.isclose(y.iloc[t], expected, rtol=1e-5), (
        f"Expected |R_{{t+1}}| = {expected:.6f}, got {y.iloc[t]:.6f}"
    )
    # Last value must be NaN (the future is unknown)
    assert np.isnan(y.iloc[-1])


def test_target_no_overlap(df):
    """
    1-day ahead target equals |log_ret[t+1]| — non-overlapping by construction.

    Note: we do NOT test autocorrelation. |R_t| has natural autocorrelation
    (~0.3) due to ARCH/volatility clustering — a property of financial data,
    not evidence of leakage. The correct proof is a direct content check.
    """
    f = build_features(df, 21)
    y = make_target(f["log_ret"], horizon=1)
    for t in [50, 100, 200, 500, 800, 1000, 1200, 1500, 1800, 2000]:
        expected = abs(float(f["log_ret"].iloc[t + 1]))
        assert np.isclose(y.iloc[t], expected, rtol=1e-5), (
            f"At t={t}: target={y.iloc[t]:.6f}, |R_t+1|={expected:.6f}"
        )


def test_window_shape_and_content():
    x = np.arange(20, dtype=float).reshape(10, 2)
    w = make_windows(x, 4)
    assert w.shape == (7, 4, 2)                   # (N-window+1, window, features)
    assert np.allclose(w[0], x[0:4]) and np.allclose(w[-1], x[6:10])
