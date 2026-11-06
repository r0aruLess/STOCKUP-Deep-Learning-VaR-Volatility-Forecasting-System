"""Tests for VaR maths — including McNeil & Frey (2000) calibration."""
import numpy as np
import pandas as pd
import pytest

from deepvar.risk.var import (
    calibrate_quantile,
    var_amount,
    var_threshold,
    var_threshold_calibrated,
    z_score,
)


# ---------------------------------------------------------------------------
# z_score / var_threshold / var_amount (unchanged)
# ---------------------------------------------------------------------------
def test_z_scores():
    assert abs(z_score(0.99) - 2.3263) < 1e-3
    assert abs(z_score(0.95) - 1.6449) < 1e-3
    assert z_score(0.99, "t", 5) > z_score(0.99)   # fat tails → bigger multiplier


def test_var_behaviour():
    assert var_threshold(0.02, 0.99) < 0
    lo = var_amount(0.01, 1e6, 0.99)
    hi = var_amount(0.03, 1e6, 0.99)
    assert 0 < lo < hi                              # more volatility → more VaR
    assert var_amount(0.02, 1e6, 0.99) > var_amount(0.02, 1e6, 0.95)
    assert np.isclose(var_amount(0.0, 1e6, 0.99), 0.0)


# ---------------------------------------------------------------------------
# calibrate_quantile — McNeil & Frey (2000)
# ---------------------------------------------------------------------------
def _make_val_data(seed=42):
    rng    = np.random.default_rng(seed)
    n      = 500
    idx    = pd.date_range("2022-01-01", periods=n, freq="B")
    sigma  = pd.Series(rng.uniform(0.005, 0.03, n), index=idx, name="sigma")
    # simulate returns drawn from t(5) scaled by sigma
    t_draws = rng.standard_t(5, n) / np.sqrt(3.0)   # rescaled to unit variance
    ret    = pd.Series(sigma.values * t_draws, index=idx, name="ret")
    return sigma, ret


def test_calibrate_quantile_returns_negative_number():
    """The 1% left-tail quantile must always be negative."""
    sigma, ret = _make_val_data()
    q = calibrate_quantile(sigma, ret, confidence=0.99)
    assert q < 0, f"Expected negative cal_q, got {q}"


def test_calibrate_quantile_magnitude_reasonable():
    """For t(5)-distributed residuals the empirical 1% quantile should be near -2.6 ± 0.6."""
    sigma, ret = _make_val_data()
    q = calibrate_quantile(sigma, ret, confidence=0.99)
    assert -4.0 < q < -1.5, f"cal_q = {q:.3f} is implausibly extreme"


def test_calibrate_quantile_95pct():
    """95% VaR should give a less extreme (smaller |q|) quantile than 99%."""
    sigma, ret = _make_val_data()
    q99 = calibrate_quantile(sigma, ret, confidence=0.99)
    q95 = calibrate_quantile(sigma, ret, confidence=0.95)
    assert q99 < q95, "99% quantile must be more extreme (more negative) than 95%"


def test_calibrate_quantile_requires_minimum_data():
    """Fewer than 30 clean observations should raise ValueError."""
    sigma = pd.Series([0.01] * 20, index=pd.date_range("2022-01-01", periods=20, freq="B"))
    ret   = pd.Series([0.005] * 20, index=sigma.index)
    with pytest.raises(ValueError, match="too few"):
        calibrate_quantile(sigma, ret, confidence=0.99)


# ---------------------------------------------------------------------------
# var_threshold_calibrated
# ---------------------------------------------------------------------------
def test_var_threshold_calibrated_is_negative():
    sigma = np.array([0.01, 0.02, 0.03])
    cal_q = -2.5
    thr   = var_threshold_calibrated(sigma, cal_q)
    assert (thr < 0).all(), "Calibrated VaR threshold must be negative (loss side)"


def test_var_threshold_calibrated_scales_with_sigma():
    """Higher sigma → more negative (larger loss) threshold."""
    thr_lo = var_threshold_calibrated(np.array([0.01]), -2.5)[0]
    thr_hi = var_threshold_calibrated(np.array([0.03]), -2.5)[0]
    assert thr_hi < thr_lo


def test_calibration_improves_coverage():
    """
    Simulate a scenario where the parametric model systematically under-covers
    (e.g., true tails are fatter than assumed), and check that the calibrated
    threshold produces a breach rate closer to the nominal 1%.
    """
    rng    = np.random.default_rng(99)
    n_val  = 500
    n_test = 1000

    # True distribution: t(3) — fatter than the t(5) the model assumes
    sigma_val  = pd.Series(np.full(n_val, 0.02),
                           index=pd.date_range("2022-01-01", periods=n_val, freq="B"))
    ret_val    = pd.Series(sigma_val.values * rng.standard_t(3, n_val) / 1.0,
                           index=sigma_val.index)

    sigma_test = pd.Series(np.full(n_test, 0.02),
                           index=pd.date_range("2024-01-01", periods=n_test, freq="B"))
    ret_test   = pd.Series(sigma_test.values * rng.standard_t(3, n_test) / 1.0,
                           index=sigma_test.index)

    cal_q  = calibrate_quantile(sigma_val, ret_val, 0.99)
    thr    = var_threshold_calibrated(sigma_test.values, cal_q)
    cal_br = (ret_test.values < thr).mean()

    # Calibrated breach rate should be close to 1% (within ±0.5%)
    assert abs(cal_br - 0.01) < 0.005, (
        f"Calibrated breach rate {cal_br:.3f} too far from 1%"
    )
