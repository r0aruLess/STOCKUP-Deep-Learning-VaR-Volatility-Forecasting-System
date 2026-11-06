"""FRED VIX vs LSTM predicted sigma cross-check on the S&P 500 test period.

Usage:
    python scripts/fred_crosscheck.py

What it does:
    Loads the cached VIXCLS.csv (downloaded by download_all.py).
    Loads the trained S&P 500 LSTM model.
    Aligns VIXCLS with the LSTM daily sigma on the TEST period.
    Computes Pearson and Spearman correlation.
    Saves reports/fred_crosscheck.csv

Interpretation:
    VIX is the market's forward-looking 30-day implied vol (annualised).
    LSTM sigma is daily realised vol (not annualised).
    We compare them in rank-correlation terms (Spearman) so the different
    units don't matter — we care that the LSTM tracks fear/calm correctly.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from deepvar.config import RAW_DIR, REPORT_DIR, load_config
from deepvar.pipeline import Forecaster, artifact_path, load_data

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR.mkdir(parents=True, exist_ok=True)

cfg    = load_config(ROOT / "configs" / "sp500.yaml")
ticker = cfg["data"]["ticker"]
test_start = str((pd.Timestamp(cfg["data"]["val_end"]) + pd.Timedelta(days=1)).date())

print("=" * 60)
print("  FRED VIX vs LSTM σ — S&P 500 cross-check")
print(f"  Test period: {test_start} onward")
print("=" * 60)

# Load S&P 500 price data and LSTM model
df  = load_data(cfg)
fc  = Forecaster.load(artifact_path(cfg))
sig = fc.sigma(df).loc[test_start:]          # daily LSTM sigma on test period

# Load cached VIX
vix_path = RAW_DIR / "VIXCLS.csv"
if not vix_path.exists():
    raise FileNotFoundError(
        "data/raw/VIXCLS.csv not found — run scripts/download_all.py first"
    )
vix = pd.read_csv(vix_path, index_col=0, parse_dates=True).squeeze()
vix = pd.to_numeric(vix, errors="coerce").dropna()

# Align on test period
vix_test = vix.loc[test_start:].reindex(sig.index).ffill()
both     = pd.DataFrame({"lstm_sigma": sig, "vix": vix_test}).dropna()

if len(both) < 10:
    print("  Not enough overlapping data for correlation — check VIX date range.")
else:
    # VIX is annualised % → convert to daily fraction for comparison
    both["vix_daily"] = both["vix"] / 100 / np.sqrt(252)

    pr, pp = pearsonr(both["lstm_sigma"],  both["vix_daily"])
    sr, sp = spearmanr(both["lstm_sigma"], both["vix_daily"])

    print(f"\n  Overlapping days: {len(both)}")
    print(f"  Pearson  r = {pr:+.3f}  (p = {pp:.4f})")
    print(f"  Spearman r = {sr:+.3f}  (p = {sp:.4f})")
    print()
    if sr > 0.5:
        print("  Interpretation: STRONG positive rank-correlation.")
        print("  The LSTM correctly tracks market fear/calm in the test period.")
    elif sr > 0.2:
        print("  Interpretation: MODERATE positive rank-correlation.")
        print("  The LSTM broadly tracks VIX direction but misses some spikes.")
    else:
        print("  Interpretation: WEAK correlation — the LSTM may need retraining")
        print("  or the test period is atypically calm.")

    both.to_csv(REPORT_DIR / "fred_crosscheck.csv")
    print(f"\n  Saved reports/fred_crosscheck.csv")
