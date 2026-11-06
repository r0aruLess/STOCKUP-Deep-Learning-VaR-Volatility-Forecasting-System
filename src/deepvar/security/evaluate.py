"""Attack experiments: how badly does each attack hurt, and does the defence help?"""
from __future__ import annotations

import numpy as np
import pandas as pd

from deepvar.risk.backtest import breach_frame
from deepvar.security.attacks import ATTACKS


def most_volatile_start(df: pd.DataFrame, after: str, length: int = 20, margin: int = 40) -> int:
    """Integer position of the most turbulent `length`-day stretch after `after`.
    The attacker strikes when it hurts most: during a real market storm."""
    lr = np.log(df["Close"] / df["Close"].shift(1))
    fwd = (lr**2).rolling(length).sum().shift(-length + 1)
    pos0 = df.index.searchsorted(pd.Timestamp(after))
    cand = fwd.iloc[pos0: len(df) - length - margin]
    return int(df.index.get_loc(cand.idxmax()))


def run_attack(vanilla, secure, df: pd.DataFrame, attack: str, start: int, n_days: int,
               conf: float = 0.99, mu: float = 0.0, post_days: int = 10,
               history: int = 450, **attack_kw) -> dict:
    """Run one attack. Returns metrics + the three sigma series for plotting."""
    attacked, _ = ATTACKS[attack](df, start, n_days, **attack_kw)
    lo = max(0, start - history)
    hi = min(len(df), start + n_days + post_days + 2)
    clean_s, att_s = df.iloc[lo:hi], attacked.iloc[lo:hi]

    sig_clean = vanilla.sigma(clean_s)
    sig_vanilla = vanilla.sigma(att_s)
    sig_secure, gate_res = secure.sigma(att_s)

    t0, t1 = df.index[start], df.index[min(start + n_days + post_days, len(df) - 1)]
    true_lr = np.log(df["Close"] / df["Close"].shift(1))

    def stats(sig):
        w = sig.loc[t0:t1]
        bf = breach_frame(true_lr, w, conf, mu).loc[t0:t1]
        return float(w.mean()), int(bf["breach"].sum()), len(bf)

    m_clean, b_clean, n = stats(sig_clean)
    m_van, b_van, _ = stats(sig_vanilla)
    m_sec, b_sec, _ = stats(sig_secure)
    detected = bool(gate_res.flagged.loc[t0:t1].any())
    return {
        "attack": attack, "n_days": n_days, "days_evaluated": n,
        "mean_sigma_clean": m_clean, "mean_sigma_vanilla": m_van, "mean_sigma_secure": m_sec,
        "sigma_ratio_vanilla": m_van / m_clean, "sigma_ratio_secure": m_sec / m_clean,
        "breaches_clean": b_clean, "breaches_vanilla": b_van, "breaches_secure": b_sec,
        "expected_breaches": n * (1 - conf), "gate_detected": detected,
        "series": {"clean": sig_clean, "vanilla": sig_vanilla, "secure": sig_secure},
    }


def poison_sweep(vanilla, secure, df, start: int, alpha: float = 0.2, max_days: int = 30,
                 step: int = 5, blind_threshold: float = 0.30, **kw) -> pd.DataFrame:
    """How many consecutive poisoned days does it take to 'blind' the model?
    'Blind' = average forecast at least `blind_threshold` (30%) lower than the truth."""
    rows = []
    for n in range(step, max_days + 1, step):
        r = run_attack(vanilla, secure, df, "vol_suppression", start, n, alpha=alpha, **kw)
        rows.append({
            "poisoned_days": n, "alpha": alpha,
            "sigma_ratio_vanilla": r["sigma_ratio_vanilla"],
            "sigma_ratio_secure": r["sigma_ratio_secure"],
            "breaches_vanilla": r["breaches_vanilla"], "breaches_secure": r["breaches_secure"],
            "gate_detected": r["gate_detected"],
            "vanilla_blinded": r["sigma_ratio_vanilla"] < 1 - blind_threshold,
            "secure_blinded": r["sigma_ratio_secure"] < 1 - blind_threshold,
        })
    return pd.DataFrame(rows)
