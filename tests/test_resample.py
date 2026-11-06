import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from resample_minute_csv import resample_to_daily  # noqa: E402


def test_minute_to_daily():
    ts = pd.date_range("2020-01-01", periods=3 * 1440, freq="min")
    px = 100 + np.arange(len(ts)) * 0.01
    raw = pd.DataFrame({"Timestamp": ((ts - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)), "Open": px, "High": px + 1,
                        "Low": px - 1, "Close": px, "Volume_(BTC)": 1.0})
    d = resample_to_daily(raw)
    assert len(d) == 3 and d["Volume"].iloc[0] == 1440
    assert d["High"].iloc[0] == px[:1440].max() + 1 and d["Open"].iloc[1] == px[1440]
