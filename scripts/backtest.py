"""Backtest LSTM-VaR vs classic baselines with Kupiec POF.   python scripts/backtest.py"""
import numpy as np
import pandas as pd
from _common import parse_args

from deepvar.config import REPORT_DIR
from deepvar.pipeline import Forecaster, artifact_path, backtest_report, load_data
from deepvar.risk.backtest import kupiec_pof
from deepvar.risk.baselines import ewma_sigma, historical_threshold, rolling_std_sigma

cfg, _ = parse_args("Backtest VaR models")
df = load_data(cfg)
fc = Forecaster.load(artifact_path(cfg))
r = cfg["risk"]
start = pd.Timestamp(cfg["data"]["val_end"]) + pd.Timedelta(days=1)
log_ret = np.log(df["Close"] / df["Close"].shift(1))

models = {
    "LSTM (this project)": fc.sigma(df),
    "EWMA (RiskMetrics)": ewma_sigma(log_ret, r["ewma_lambda"]).dropna(),
    "Rolling 21d std": rolling_std_sigma(log_ret, 21).dropna(),
}
rows = []
for name, sig in models.items():
    rep = backtest_report(cfg, df, sig, start=str(start.date()))
    k = rep["kupiec"]
    rows.append({"model": name, "days": k["n"], "breaches": k["breaches"],
                 "expected": round(k["expected"], 1), "breach_rate": round(k["breach_rate"], 4),
                 "kupiec_p": round(k["p_value"], 4), "independence_p": round(k["independence_p"], 4),
                 "verdict": k["verdict"]})
    if name.startswith("LSTM"):
        rep["frame"].to_csv(REPORT_DIR / "var_backtest_lstm.csv")

# historical simulation (threshold directly)
thr = historical_threshold(log_ret, r["hist_window"], r["confidence"]).dropna()
hs = pd.DataFrame({"threshold": thr, "realized": log_ret.shift(-1)}).dropna().loc[start:]
k = kupiec_pof((hs["realized"] < hs["threshold"]).values, r["confidence"])
rows.append({"model": "Historical simulation", "days": k["n"], "breaches": k["breaches"],
             "expected": round(k["expected"], 1), "breach_rate": round(k["breach_rate"], 4),
             "kupiec_p": round(k["p_value"], 4), "independence_p": np.nan, "verdict": k["verdict"]})

table = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(f"\nVaR {r['confidence']:.0%} backtest on test period from {start.date()} "
      f"({r['distribution']} quantile)\n")
print(table.to_string(index=False))
table.to_csv(REPORT_DIR / "backtest_summary.csv", index=False)
print("\nsaved reports/backtest_summary.csv")
