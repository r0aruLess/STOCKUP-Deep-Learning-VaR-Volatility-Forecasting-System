"""Input-validation gate ("the digital bouncer") + fail-safe volatility floor.

Two transparent, threshold-based rules — cheapest first:

Rule 1  hard_limit   |daily log return| exceeds a business-defined threshold.
                     Catches fat-finger errors, flash-crash data spoofs, and any
                     price print that is physically implausible for the asset.
                     Response: carry the last trusted close forward (repair).

Rule 2  collapse     The 5-day rolling RMS volatility drops to less than
                     `collapse_ratio` (default 35%) of the 1-year median vol.
                     Catches the main threat: a calm-injection/suppression attack
                     where an adversary replays quiet data during a real storm.
                     Response: cannot be repaired — flag for the fail-safe floor.

Why two rules are enough for this threat model
----------------------------------------------
* suppress_volatility (primary attack) → caught by Rule 2.
  The collapse check compares current rolling vol against the asset's own
  long-run median, so it flags "suspiciously calm" windows even when the
  returns look numerically ordinary.
* inject_flash_crash / inject_fat_finger (spike attacks) → caught by Rule 1.
  A direct hard-limit is simpler, faster, and more explainable than a
  trained ML model for detecting large one-day price dislocations.

Both rules are fully transparent and require zero training data or ML models,
making them straightforward to explain to an evaluation panel.

The fail-safe (apply_failsafe) holds the volatility forecast at the
last-known-good level while any alert is active, ensuring we NEVER under-report
risk during a suspected data-integrity event.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PRICE_COLS = ["Open", "High", "Low", "Close"]


@dataclass
class GateResult:
    flags: pd.DataFrame      # one boolean column per rule
    flagged: pd.Series       # any rule fired (day-level)
    repaired: pd.DataFrame   # input with spike-type bad rows replaced by last trusted close


class InputValidationGate:
    """Two transparent, threshold-based rules — cheapest first.

    Rule 1  hard_limit  — |daily log return| above a business limit.
    Rule 2  collapse    — rolling RMS vol falls suspiciously far below
                          the long-run median (stealth calm-injection attack).

    Rule 1 fires  → row is repaired (last trusted close carried forward).
    Rule 2 fires  → flagged only; fail-safe floor kicks in downstream.

    A no-op fit() is provided for API compatibility with pipeline.py,
    which calls InputValidationGate(...).fit(trusted_df).
    """

    def __init__(
        self,
        max_abs_return: float = 0.40,
        collapse_window: int = 5,
        collapse_ratio: float = 0.35,
        vol_window: int = 21,
        **_ignored,                    # absorb any extra config keys safely
    ):
        self.max_abs_return = max_abs_return
        self.collapse_window = collapse_window
        self.collapse_ratio = collapse_ratio
        self.vol_window = vol_window

    # ------------------------------------------------------------------
    def fit(self, trusted_df: pd.DataFrame) -> "InputValidationGate":  # noqa: ARG002
        """No-op — kept for API compatibility with pipeline.py."""
        return self

    # ------------------------------------------------------------------
    def check(self, df: pd.DataFrame) -> GateResult:
        """Apply both rules and return a GateResult.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV price data with a DatetimeIndex.

        Returns
        -------
        GateResult
            flags    — DataFrame with bool columns: ``hard_limit``, ``collapse``
            flagged  — Series, True on any day either rule fires
            repaired — df with hard_limit rows replaced by last good close
        """
        close   = df["Close"]
        log_ret = np.log(close / close.shift(1))
        flags   = pd.DataFrame(index=df.index)

        # -- Rule 1: hard-limit -----------------------------------------------
        # Flag any day where the absolute log-return exceeds the business limit.
        flags["hard_limit"] = log_ret.abs() > self.max_abs_return

        # -- Rule 2: collapse (stealth calm-injection) ------------------------
        # Flag any day where the 5-day rolling RMS drops below a fraction of
        # the 1-year median vol (computed from history only — no lookahead).
        k            = self.collapse_window
        rms_k        = np.sqrt((log_ret ** 2).rolling(k).mean())
        long_run_vol = (
            log_ret.rolling(self.vol_window).std()
                   .rolling(252, min_periods=60)
                   .median()
                   .shift(k)          # shift ensures we only use past data
        )
        flags["collapse"] = (rms_k < self.collapse_ratio * long_run_vol).fillna(False)

        flags   = flags.fillna(False).astype(bool)
        flagged = flags.any(axis=1)

        # -- Repair: carry last trusted close forward for hard-limit spikes ---
        spike      = flags["hard_limit"]
        repaired   = df.copy()
        good_close = close.where(~spike).ffill()
        if spike.any():
            vals = good_close[spike].values                          # shape (n,)
            repaired.loc[spike, PRICE_COLS] = np.column_stack(      # shape (n, 4)
                [vals] * len(PRICE_COLS)
            )
        repaired   = repaired.dropna(subset=PRICE_COLS)


        return GateResult(flags=flags, flagged=flagged, repaired=repaired)


# ---------------------------------------------------------------------------
# Fail-safe volatility floor
# ---------------------------------------------------------------------------
def apply_failsafe(
    sigma: pd.Series,
    flagged: pd.Series,
    hold_days: int = 21,
    ref_window: int = 60,
    lag: int = 10,
) -> pd.Series:
    """While an alert is active (and ``hold_days`` after), never let sigma fall
    below the last-known-good level (rolling max of earlier, pre-detection forecasts).

    Fail-safe design: when in doubt, hold MORE capital, never less.

    Parameters
    ----------
    sigma     : LSTM forecast series
    flagged   : boolean series from GateResult.flagged
    hold_days : how many extra days the floor stays active after the last flag
    ref_window: how many past forecasts to take the max over
    lag       : how many days back to anchor the reference level
    """
    flagged = flagged.reindex(sigma.index).fillna(False).astype(bool)
    alert   = (
        flagged.astype(int)
               .rolling(hold_days, min_periods=1)
               .max()
               .astype(bool)
               .to_numpy()
    )
    ref    = sigma.rolling(ref_window, min_periods=1).max().shift(lag).bfill().to_numpy()
    s      = sigma.to_numpy().copy()
    frozen = None
    for i in range(len(s)):
        if alert[i]:
            if frozen is None:
                frozen = ref[i]
            s[i] = max(s[i], frozen)
        else:
            frozen = None
    return pd.Series(s, index=sigma.index, name=sigma.name)
