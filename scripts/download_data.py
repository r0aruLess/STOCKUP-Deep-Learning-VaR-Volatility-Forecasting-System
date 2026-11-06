"""Download + cache price data.   python scripts/download_data.py --ticker BTC-USD"""
from _common import parse_args

from deepvar.data.download import load_prices

cfg, _ = parse_args("Download market data")
d = cfg["data"]
df = load_prices(d["ticker"], d["start"], d.get("end"), d["source"], d.get("csv_path"), refresh=True)
print(df.tail())
print(f"\n{len(df)} rows  {df.index.min().date()} -> {df.index.max().date()}  (cached in data/raw/)")
