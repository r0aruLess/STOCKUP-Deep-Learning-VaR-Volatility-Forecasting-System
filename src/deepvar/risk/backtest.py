"""VaR back-testing: breach series, Kupiec POF, Christoffersen independence."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2

from deepvar.risk.var import var_threshold


def breach_frame(log_ret: pd.Series, sigma: pd.Series, confidence=0.99, mu=0.0,
                 distribution="normal", t_df=5.0) -> pd.DataFrame:
    """Row t holds sigma forecast at close of day t and the return realised on day t+1."""
    df = pd.DataFrame({"sigma": sigma})
    df["threshold"] = var_threshold(df["sigma"].values, confidence, mu, distribution, t_df)
    df["realized"] = log_ret.shift(-1).reindex(df.index)
    df = df.dropna()
    df["breach"] = df["realized"] < df["threshold"]
    return df


def _xlogy(x, y):
    return 0.0 if x == 0 else x * np.log(y)


def kupiec_pof(breaches, confidence: float = 0.99, alpha: float = 0.05) -> dict:
    """Kupiec (1995) proportion-of-failures test.

    H0: the breach probability equals 1 - confidence.  Reject H0 => model is mis-calibrated.
    """
    b = np.asarray(breaches, dtype=bool)
    n, x = len(b), int(b.sum())
    p = 1.0 - confidence
    if n == 0:
        raise ValueError("no observations")
    phat = x / n
    ll0 = _xlogy(n - x, 1 - p) + _xlogy(x, p)
    ll1 = _xlogy(n - x, 1 - phat) + _xlogy(x, phat)
    lr = float(max(-2.0 * (ll0 - ll1), 0.0))
    pval = float(1.0 - chi2.cdf(lr, df=1))
    return {
        "n": n, "breaches": x, "expected": n * p, "breach_rate": phat,
        "LR_pof": lr, "p_value": pval, "reject_H0": pval < alpha,
        "verdict": "FAIL (mis-calibrated)" if pval < alpha else "PASS (statistically acceptable)",
    }


def christoffersen_independence(breaches, alpha: float = 0.05) -> dict:
    """Tests whether breaches cluster (a good model's breaches are independent)."""
    b = np.asarray(breaches, dtype=int)
    if len(b) < 2:
        return {"LR_ind": float("nan"), "p_value": float("nan"), "reject_H0": False}
    prev, cur = b[:-1], b[1:]
    n00 = int(((prev == 0) & (cur == 0)).sum()); n01 = int(((prev == 0) & (cur == 1)).sum())
    n10 = int(((prev == 1) & (cur == 0)).sum()); n11 = int(((prev == 1) & (cur == 1)).sum())
    pi01 = n01 / max(n00 + n01, 1); pi11 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    ll0 = _xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
    ll1 = (_xlogy(n00, 1 - pi01) + _xlogy(n01, pi01) + _xlogy(n10, 1 - pi11) + _xlogy(n11, pi11))
    lr = float(max(-2.0 * (ll0 - ll1), 0.0))
    pval = float(1.0 - chi2.cdf(lr, df=1))
    return {"LR_ind": lr, "p_value": pval, "reject_H0": pval < alpha}
