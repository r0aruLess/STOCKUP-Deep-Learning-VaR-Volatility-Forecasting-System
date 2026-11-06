"""Security gate evaluation on NAB (Numenta Anomaly Benchmark) datasets.

Usage:
    python scripts/nab_gate_eval.py

What it does:
    Loads 3 NAB CSV files (downloaded by download_all.py).
    Loads the NAB anomaly label windows (combined_windows.json).
    Runs the 2-rule InputValidationGate on each file (treating the value
    column as a 'price' so the gate can compute log-returns and rolling vol).
    Computes Precision, Recall, and F1 for each file.
    Saves reports/nab_gate_eval.csv

Why NAB?
    NAB provides real-world time-series with human-labelled anomaly windows.
    Running our gate on these (non-financial) series validates that the
    hard-limit + collapse rules generalise beyond financial data — supporting
    the thesis claim of a broadly applicable data-integrity detection approach.

Note on methodology:
    NAB anomaly windows are RANGES of time (not single points).
    We label every timestamp inside any anomaly window as True anomaly.
    We then compare gate.flagged against these labels.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from deepvar.config import REPORT_DIR
from deepvar.security.defenses import InputValidationGate

ROOT    = Path(__file__).resolve().parents[1]
NAB_DIR = ROOT / "data" / "nab"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

NAB_FILES = [
    "speed_6005.csv",
    "ambient_temperature_system_failure.csv",
    "ec2_cpu_utilization_5f5533.csv",
]

# Map filename → NAB key used in combined_windows.json
NAB_KEY_MAP = {
    "speed_6005.csv":
        "realTraffic/speed_6005.csv",
    "ambient_temperature_system_failure.csv":
        "realKnownCause/ambient_temperature_system_failure.csv",
    "ec2_cpu_utilization_5f5533.csv":
        "realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv",
}

label_path = NAB_DIR / "combined_windows.json"
if not label_path.exists():
    raise FileNotFoundError("data/nab/combined_windows.json not found — run download_all.py first")

with open(label_path) as f:
    all_windows = json.load(f)

print("=" * 65)
print("  NAB Security Gate Evaluation")
print("=" * 65)

rows = []

for fname in NAB_FILES:
    fpath = NAB_DIR / fname
    if not fpath.exists():
        print(f"\n  {fname}: FILE NOT FOUND — skipping")
        continue

    # -- Load NAB CSV -------------------------------------------------
    df_raw = pd.read_csv(fpath, parse_dates=["timestamp"])
    df_raw = df_raw.set_index("timestamp").sort_index()
    series = df_raw["value"].ffill().dropna()

    # -- Build anomaly label vector ----------------------------------
    nab_key = NAB_KEY_MAP[fname]
    windows = all_windows.get(nab_key, [])
    labels  = pd.Series(False, index=series.index)
    for w_start, w_end in windows:
        t0 = pd.Timestamp(w_start)
        t1 = pd.Timestamp(w_end)
        labels.loc[(labels.index >= t0) & (labels.index <= t1)] = True

    # -- Adapt series for the gate -----------------------------------
    # The gate expects a DataFrame with Open/High/Low/Close/Volume columns.
    # We treat 'value' as 'Close' and synthesise minimal OHLC from it.
    # This lets the gate compute log-returns and rolling vol on the series.
    close  = series.clip(lower=1e-6)          # avoid log(0)
    vol_df = pd.DataFrame({
        "Open":   close * 0.999,
        "High":   close * 1.001,
        "Low":    close * 0.999,
        "Close":  close,
        "Volume": 1.0,
    })

    # -- Run gate ----------------------------------------------------
    gate = InputValidationGate(
        max_abs_return=0.30,   # NAB series have different scale — slightly relaxed
        collapse_window=5,
        collapse_ratio=0.30,   # slightly relaxed for non-financial series
        vol_window=21,
    )
    gate.fit(vol_df)
    result  = gate.check(vol_df)
    flagged = result.flagged.reindex(labels.index).fillna(False)

    # -- Metrics -----------------------------------------------------
    tp = int((flagged & labels).sum())
    fp = int((flagged & ~labels).sum())
    fn = int((~flagged & labels).sum())
    tn = int((~flagged & ~labels).sum())

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)

    print(f"\n  File : {fname}")
    print(f"  Total timestamps : {len(labels)}")
    print(f"  True anomalies   : {int(labels.sum())}  ({labels.mean():.1%})")
    print(f"  Gate flagged     : {int(flagged.sum())}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Precision = {precision:.3f} | Recall = {recall:.3f} | F1 = {f1:.3f}")

    rows.append({
        "file": fname, "total": len(labels),
        "true_anomalies": int(labels.sum()),
        "gate_flagged": int(flagged.sum()),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": round(precision, 3),
        "recall":    round(recall, 3),
        "f1":        round(f1, 3),
    })

out = pd.DataFrame(rows)
out.to_csv(REPORT_DIR / "nab_gate_eval.csv", index=False)
print()
print("=" * 65)
print("  SUMMARY")
print("=" * 65)
pd.set_option("display.width", 200)
print(out[["file","precision","recall","f1","TP","FP","FN"]].to_string(index=False))
print(f"\nSaved reports/nab_gate_eval.csv")
