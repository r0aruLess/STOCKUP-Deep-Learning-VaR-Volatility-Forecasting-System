"""Backtest LSTM-VaR vs classical baselines on every asset in configs/.

Auto-discovers every *.yaml in configs/ — no hardcoded list needed.

Runs FOUR models per asset:
  1. LSTM (DeepVaR)         — raw parametric VaR (t-distribution z-score)
  2. LSTM (Calibrated)      — McNeil & Frey (2000) volatility-filtered historical
                              simulation; val-set empirical tail quantile
  3. EWMA (RiskMetrics)     — exponentially weighted moving average baseline
  4. Rolling 21d std        — simple rolling standard deviation baseline
  5. Historical simulation  — rolling empirical quantile baseline

Usage:
    python scripts/backtest_all.py

Output:
    reports/backtest_summary_all.csv
    reports/var_backtest_<ticker>.csv  (per-asset breach frame)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from deepvar.config import REPORT_DIR, load_config
from deepvar.pipeline import Forecaster, artifact_path, backtest_report, load_data
from deepvar.risk.backtest import kupiec_pof
from deepvar.risk.baselines import ewma_sigma, historical_threshold, rolling_std_sigma
from deepvar.risk.var import calibrate_quantile, var_threshold_calibrated

ROOT      = Path(__file__).resolve().parents[1]
REPORT_DIR.mkdir(parents=True, exist_ok=True)
cfg_paths = sorted((ROOT / "configs").glob("*.yaml"))
all_rows  = []

print(f"Found {len(cfg_paths)} config(s). Running backtests...\n")

for cfg_path in cfg_paths:
    cfg    = load_config(cfg_path)
    ticker = cfg["data"]["ticker"]
    safe   = re.sub(r"[^A-Za-z0-9]+", "_", ticker).strip("_")
    r      = cfg["risk"]

    val_start  = str((pd.Timestamp(cfg["data"]["train_end"])
                      + pd.Timedelta(days=1)).date())
    val_end    = cfg["data"]["val_end"]
    test_start = str((pd.Timestamp(val_end) + pd.Timedelta(days=1)).date())

    print("=" * 65)
    print(f"  {ticker}  (test from {test_start})")
    print("=" * 65)

    art = artifact_path(cfg)
    if not (art / "model.pt").exists():
        print(f"  No model found at {art} — skipping (run train_all.py first)\n")
        continue

    try:
        df      = load_data(cfg)
        fc      = Forecaster.load(art)
        log_ret = np.log(df["Close"] / df["Close"].shift(1))

        # ------------------------------------------------------------------ #
        # 1. Raw LSTM (parametric t-distribution VaR)                        #
        # ------------------------------------------------------------------ #
        sig_lstm = fc.sigma(df)
        lstm_rep = backtest_report(cfg, df, sig_lstm, start=test_start)
        k        = lstm_rep["kupiec"]
        ind_p    = lstm_rep.get("independence_p",
                                k.get("independence_p", float("nan")))
        all_rows.append({
            "asset": ticker, "model": "LSTM (DeepVaR)",
            "test_days": k["n"], "breaches": k["breaches"],
            "expected": round(k["expected"], 1),
            "breach_rate": round(k["breach_rate"], 4),
            "kupiec_p": round(k["p_value"], 4),
            "independence_p": round(ind_p, 4) if not np.isnan(ind_p) else float("nan"),
            "cal_q": float("nan"),
            "verdict": k["verdict"],
        })
        lstm_rep["frame"].to_csv(REPORT_DIR / f"var_backtest_{safe}.csv")

        # ------------------------------------------------------------------ #
        # 2. LSTM (Calibrated) — McNeil & Frey 2000                          #
        # ------------------------------------------------------------------ #
        try:
            sig_val  = sig_lstm.loc[val_start:val_end]
            ret_val  = log_ret.loc[val_start:val_end]
            cal_q    = calibrate_quantile(sig_val, ret_val, r["confidence"])

            sig_test = sig_lstm.loc[test_start:]
            thr_cal  = pd.Series(
                var_threshold_calibrated(sig_test.values, cal_q),
                index=sig_test.index,
            )
            ret_test = log_ret.reindex(thr_cal.index)
            aligned  = pd.DataFrame({"thr": thr_cal, "ret": ret_test}).dropna()
            breaches_cal = (aligned["ret"] < aligned["thr"]).values
            k_cal    = kupiec_pof(breaches_cal, r["confidence"])

            all_rows.append({
                "asset": ticker, "model": "LSTM (Calibrated)",
                "test_days": k_cal["n"], "breaches": k_cal["breaches"],
                "expected": round(k_cal["expected"], 1),
                "breach_rate": round(k_cal["breach_rate"], 4),
                "kupiec_p": round(k_cal["p_value"], 4),
                "independence_p": float("nan"),
                "cal_q": round(cal_q, 4),
                "verdict": k_cal["verdict"],
            })
        except Exception as exc:
            print(f"  Calibration skipped: {exc}")
            all_rows.append({
                "asset": ticker, "model": "LSTM (Calibrated)",
                "test_days": 0, "breaches": 0, "expected": 0,
                "breach_rate": float("nan"), "kupiec_p": float("nan"),
                "independence_p": float("nan"),
                "cal_q": float("nan"),
                "verdict": f"SKIPPED: {exc}",
            })

        # ------------------------------------------------------------------ #
        # 3. EWMA (RiskMetrics)                                               #
        # ------------------------------------------------------------------ #
        sig_ewma = ewma_sigma(log_ret, r["ewma_lambda"]).dropna()
        ewma_rep = backtest_report(cfg, df, sig_ewma, start=test_start)
        k_e      = ewma_rep["kupiec"]
        all_rows.append({
            "asset": ticker, "model": "EWMA (RiskMetrics)",
            "test_days": k_e["n"], "breaches": k_e["breaches"],
            "expected": round(k_e["expected"], 1),
            "breach_rate": round(k_e["breach_rate"], 4),
            "kupiec_p": round(k_e["p_value"], 4),
            "independence_p": float("nan"),
            "cal_q": float("nan"),
            "verdict": k_e["verdict"],
        })

        # ------------------------------------------------------------------ #
        # 4. Rolling 21d std                                                  #
        # ------------------------------------------------------------------ #
        sig_roll = rolling_std_sigma(log_ret, 21).dropna()
        roll_rep = backtest_report(cfg, df, sig_roll, start=test_start)
        k_r      = roll_rep["kupiec"]
        all_rows.append({
            "asset": ticker, "model": "Rolling 21d std",
            "test_days": k_r["n"], "breaches": k_r["breaches"],
            "expected": round(k_r["expected"], 1),
            "breach_rate": round(k_r["breach_rate"], 4),
            "kupiec_p": round(k_r["p_value"], 4),
            "independence_p": float("nan"),
            "cal_q": float("nan"),
            "verdict": k_r["verdict"],
        })

        # ------------------------------------------------------------------ #
        # 5. Historical simulation                                             #
        # ------------------------------------------------------------------ #
        thr_hs  = historical_threshold(log_ret, r["hist_window"],
                                       r["confidence"]).dropna()
        hs_df   = pd.DataFrame(
            {"threshold": thr_hs, "realized": log_ret.shift(-1)}
        ).dropna().loc[test_start:]
        k_hs    = kupiec_pof(
            (hs_df["realized"] < hs_df["threshold"]).values, r["confidence"]
        )
        all_rows.append({
            "asset": ticker, "model": "Historical simulation",
            "test_days": k_hs["n"], "breaches": k_hs["breaches"],
            "expected": round(k_hs["expected"], 1),
            "breach_rate": round(k_hs["breach_rate"], 4),
            "kupiec_p": round(k_hs["p_value"], 4),
            "independence_p": float("nan"),
            "cal_q": float("nan"),
            "verdict": k_hs["verdict"],
        })

        # Print per-asset summary
        asset_rows = [rw for rw in all_rows if rw["asset"] == ticker]
        sub = pd.DataFrame(asset_rows)
        print(sub[["model", "test_days", "breaches", "expected",
                   "breach_rate", "kupiec_p", "verdict"]].to_string(index=False))
        print()

    except Exception as exc:
        print(f"  FAILED: {exc}\n")
        all_rows.append({
            "asset": ticker, "model": "ALL",
            "verdict": f"FAILED: {exc}",
        })

# ---------------------------------------------------------------------------
# Combined report
# ---------------------------------------------------------------------------
combined = pd.DataFrame(all_rows)
combined.to_csv(REPORT_DIR / "backtest_summary_all.csv", index=False)

print()
print("=" * 75)
print("  KUPIEC PASS/FAIL SUMMARY BY MODEL")
print("=" * 75)
pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 200)

for model_name in [
    "LSTM (DeepVaR)", "LSTM (Calibrated)",
    "EWMA (RiskMetrics)", "Rolling 21d std", "Historical simulation",
]:
    sub = combined[combined["model"] == model_name]
    if sub.empty:
        continue
    n_pass = (sub["verdict"].str.startswith("PASS")).sum()
    n_tot  = len(sub)
    print(f"  {model_name:<28}  {n_pass:>2}/{n_tot} PASS")

print()
print("=" * 75)
print("  FULL COMBINED BACKTEST SUMMARY")
print("=" * 75)
print(combined[["asset", "model", "test_days", "breaches", "expected",
               "breach_rate", "kupiec_p", "verdict"]].to_string(index=False))
print(f"\nSaved reports/backtest_summary_all.csv")

# Calibrated-quantile table
cal_rows = combined[combined["model"] == "LSTM (Calibrated)"][
    ["asset", "cal_q", "breaches", "expected", "kupiec_p", "verdict"]
].copy()
print()
print("=" * 75)
print("  CALIBRATED QUANTILE (val-set empirical 1% tail) PER ASSET")
print("=" * 75)
print(cal_rows.to_string(index=False))
