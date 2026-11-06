"""Attack simulations against the market-data ingestion pipeline.

Threat model: the attacker can tamper with the upstream price feed (API
man-in-the-middle, compromised vendor, spoofed quotes) but NOT with the real
market. We therefore always evaluate risk against the TRUE returns.

Every feed-level attack has the signature:  attack(df, start, n_days, **kw)
and returns (tampered_df, tampered_mask).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_COLS = ["Open", "High", "Low", "Close"]


def _mask(df: pd.DataFrame, start: int, n_days: int) -> pd.Series:
    m = pd.Series(False, index=df.index)
    m.iloc[start:start + n_days] = True
    return m


def inject_flash_crash(df, start: int, n_days: int = 1, depth: float = 0.30):
    """Spoofed price dip: prices read `depth` (30%) lower for n_days, then snap back."""
    out = df.copy()
    sl = slice(start, start + n_days)
    out.iloc[sl, [out.columns.get_loc(c) for c in PRICE_COLS]] *= (1 - depth)
    return out, _mask(df, start, n_days)


def inject_fat_finger(df, start: int, n_days: int = 1, size: float = 0.50):
    """Erroneous/spoofed print: close and high are `size` (+50%) too high."""
    out = df.copy()
    sl = slice(start, start + n_days)
    for c in ("High", "Close"):
        out.iloc[sl, out.columns.get_loc(c)] *= (1 + size)
    return out, _mask(df, start, n_days)


def suppress_volatility(df, start: int, n_days: int = 10, alpha: float = 0.2):
    """STEALTH DATA POISONING (the main threat in this project).

    During a real market storm the attacker replays "calm" data: every daily
    return inside the window is multiplied by `alpha` (0.2 = 80% calmer) and the
    intraday range is squeezed the same way. Prices after the window are re-based
    so the series stays continuous (no suspicious jump when the attack stops).
    The values look *ordinary* - only smaller - so spike detectors do not fire.
    """
    out = df.copy()
    lr = np.log(df["Close"]).diff().to_numpy()
    new_lr = lr.copy()
    end = start + n_days
    new_lr[start:end] = alpha * lr[start:end]
    new_close = df["Close"].iloc[0] * np.exp(np.cumsum(np.nan_to_num(new_lr)))
    ratio = new_close / df["Close"].to_numpy()

    for c in PRICE_COLS:
        out[c] = df[c].to_numpy() * ratio
    rel_close = df["Close"].to_numpy()
    sl = slice(start, end)
    for c in ("Open", "High", "Low"):
        rel = df[c].to_numpy()[sl] / rel_close[sl]
        out.iloc[sl, out.columns.get_loc(c)] = new_close[sl] * rel**alpha
    return out, _mask(df, start, n_days)


ATTACKS = {
    "flash_crash": inject_flash_crash,
    "fat_finger": inject_fat_finger,
    "vol_suppression": suppress_volatility,
}
