"""Tests for feed-spoofing attacks.

pgd_underestimate was removed (white-box attack, unrealistic for feed attacker).
Tests cover the 3 remaining data-feed attacks.
"""
import numpy as np

from deepvar.security.attacks import inject_fat_finger, inject_flash_crash, suppress_volatility


def _lr(d):
    return np.log(d["Close"] / d["Close"].shift(1))


def test_flash_crash_only_touches_window(df):
    out, m = inject_flash_crash(df, 500, 1, depth=0.3)
    assert m.sum() == 1
    assert np.isclose(out["Close"].iloc[500], df["Close"].iloc[500] * 0.7)
    assert np.allclose(out["Close"].iloc[:500], df["Close"].iloc[:500])
    assert np.allclose(out["Close"].iloc[501:], df["Close"].iloc[501:])


def test_fat_finger_spikes_up(df):
    out, _ = inject_fat_finger(df, 700, 1, size=0.5)
    assert out["Close"].iloc[700] > df["Close"].iloc[700] * 1.4


def test_suppression_scales_returns_and_is_stealthy(df):
    alpha, s, n = 0.2, 600, 10
    out, m = suppress_volatility(df, s, n, alpha)
    a, b = _lr(df), _lr(out)
    assert np.allclose(b.iloc[s:s + n], alpha * a.iloc[s:s + n])
    assert np.allclose(b.iloc[1:s], a.iloc[1:s])
    assert np.allclose(b.iloc[s + n:], a.iloc[s + n:], atol=1e-9)
    assert (out["High"] >= out[["Open", "Close"]].max(axis=1) - 1e-9).all()
    assert (out["Low"] <= out[["Open", "Close"]].min(axis=1) + 1e-9).all()
    assert m.sum() == n


def test_attacks_dict_has_expected_keys():
    """Confirm the ATTACKS registry only exposes feed-level attacks."""
    from deepvar.security.attacks import ATTACKS
    assert set(ATTACKS.keys()) == {"flash_crash", "fat_finger", "vol_suppression"}
