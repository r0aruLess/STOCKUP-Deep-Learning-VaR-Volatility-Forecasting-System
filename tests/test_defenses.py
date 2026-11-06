"""Tests for the 2-rule input validation gate and fail-safe floor.

Gate rules (after M.Tech streamlining):
  - hard_limit : |log-return| > max_abs_return  (catches flash-crash / fat-finger)
  - collapse   : rolling vol < threshold         (catches stealth vol-suppression)
"""
import numpy as np
import pandas as pd

from deepvar.security.attacks import inject_fat_finger, inject_flash_crash, suppress_volatility
from deepvar.security.defenses import InputValidationGate, apply_failsafe


def _gate(df, **kw):
    return InputValidationGate(**kw).fit(df.iloc[:1000])


def test_gate_has_exactly_two_rules(df):
    """Gate must expose exactly hard_limit and collapse columns — no more, no less."""
    res = _gate(df).check(df)
    assert set(res.flags.columns) == {"hard_limit", "collapse"}


def test_gate_catches_spikes_and_repairs(df):
    # max_abs_return=0.20 → any |log-ret| > 20% triggers hard_limit.
    # inject_flash_crash(depth=0.3) → log(0.7) ≈ -0.357 > 0.20 ✓
    # inject_fat_finger(size=0.5)   → log(1.5) ≈ +0.405 > 0.20 ✓
    gate = _gate(df, max_abs_return=0.20)
    for attack in (inject_flash_crash, inject_fat_finger):
        bad, _ = attack(df, 1400, 1)
        res = gate.check(bad)
        assert res.flagged.iloc[1400], "gate must flag the injected spike day"
        # repaired close ≈ last trusted close (carry-forward), not the spoofed value
        assert np.isclose(res.repaired["Close"].iloc[1400], df["Close"].iloc[1399])


def test_gate_catches_stealth_suppression(df):
    # alpha=0.05 → returns shrunk to 5% of original.
    # Suppressed 5d-RMS ≈ 0.05 × daily_vol, which is well below
    # the collapse threshold of 0.35 × long_run_median_vol.
    gate = _gate(df)
    lr = np.log(df["Close"] / df["Close"].shift(1))
    storm = (lr**2).rolling(20).sum().loc["2018-01-01":"2020-12-31"].idxmax()
    s = df.index.get_loc(storm) - 19
    bad, _ = suppress_volatility(df, s, 15, alpha=0.05)
    assert gate.check(bad).flags["collapse"].iloc[s:s + 25].any()


def test_gate_mostly_quiet_on_clean_data(df):
    res = _gate(df).check(df)
    # On realistic synthetic data the hard_limit rule fires very rarely
    assert res.flags["hard_limit"].mean() < 0.01


def test_failsafe_never_lowers_sigma_during_alert():
    idx    = pd.date_range("2024-01-01", periods=120, freq="D")
    sigma  = pd.Series(0.03, index=idx)
    sigma.iloc[60:] = 0.01                        # poisoned: model suddenly "calm"
    flagged = pd.Series(False, index=idx)
    flagged.iloc[65:70] = True
    out = apply_failsafe(sigma, flagged, hold_days=21, ref_window=30, lag=10)
    assert (out >= sigma).all()
    assert out.iloc[70] >= 0.03 - 1e-12           # floor restored to last-known-good


def test_gate_repair_shape_unchanged(df):
    """Repaired DataFrame must have the same index and columns as input."""
    res = _gate(df).check(df)
    assert res.repaired.shape == df.shape
    assert list(res.repaired.columns) == list(df.columns)
