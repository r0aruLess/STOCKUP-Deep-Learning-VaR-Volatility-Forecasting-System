"""Turn a minute-level Kaggle CSV (e.g. 'Bitcoin Historical Data') into daily OHLCV.

    python scripts/resample_minute_csv.py path/to/btcusd_1-min_data.csv --name BTC_kaggle
Then set in your config:  data.source: csv   data.csv_path: data/raw/BTC_kaggle.csv
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def resample_to_daily(raw: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in raw.columns}
    tcol = next((cols[k] for k in ("timestamp", "date", "time", "datetime") if k in cols), None)
    if tcol is None:
        raise ValueError("no Timestamp/Date column found")
    ts = raw[tcol]
    idx = (pd.to_datetime(ts, unit="s") if pd.api.types.is_numeric_dtype(ts)
           else pd.to_datetime(ts))
    raw = raw.set_index(pd.DatetimeIndex(idx, name="Date"))
    vol = next((c for c in raw.columns if c.lower().startswith("volume")), None)
    agg = {cols["open"]: "first", cols["high"]: "max", cols["low"]: "min", cols["close"]: "last"}
    if vol:
        agg[vol] = "sum"
    daily = raw.resample("1D").agg(agg).dropna(subset=[cols["close"]])
    daily = daily.rename(columns={cols["open"]: "Open", cols["high"]: "High", cols["low"]: "Low",
                                  cols["close"]: "Close", **({vol: "Volume"} if vol else {})})
    if "Volume" not in daily:
        daily["Volume"] = 0.0
    return daily[["Open", "High", "Low", "Close", "Volume"]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--name", default="resampled")
    a = ap.parse_args()
    out = resample_to_daily(pd.read_csv(a.csv))
    dest = ROOT / "data" / "raw" / f"{a.name}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest)
    print(out.tail(), f"\nsaved {len(out)} daily rows -> {dest}")
