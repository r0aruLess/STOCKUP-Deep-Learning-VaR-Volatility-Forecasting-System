"""Download all assets defined in configs/ + FRED VIXCLS + NAB files.

Auto-discovers every *.yaml in configs/ — no hardcoded list needed.
New tickers are picked up automatically when you add a new config file.

Usage:
    python scripts/download_all.py
"""
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml
from deepvar.config import RAW_DIR, ROOT
from deepvar.data.download import download_yfinance, load_fred_series

RAW_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Auto-discover tickers from every config yaml
# ---------------------------------------------------------------------------
configs_dir = ROOT / "configs"
seen = {}  # ticker -> start date (deduplicate)

for cfg_path in sorted(configs_dir.glob("*.yaml")):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    d = cfg.get("data", {})
    ticker = d.get("ticker")
    start  = d.get("start", "2015-01-01")
    if ticker and ticker not in seen:
        seen[ticker] = start

print("=" * 65)
print(f"Step 1 — Downloading {len(seen)} tickers (auto-discovered from configs/)")
print("=" * 65)

for ticker, start in sorted(seen.items()):
    safe  = ticker.replace("^", "").replace("-", "_").replace(".", "_").replace("=", "_")
    cache = RAW_DIR / f"{safe}.csv"
    if cache.exists():
        print(f"  {ticker:<22} CACHED (delete to re-download)")
        continue
    print(f"  {ticker:<22}", end=" ", flush=True)
    try:
        df = download_yfinance(ticker, start=start)
        df.to_csv(cache)
        print(f"OK  {len(df)} rows  {df.index.min().date()} -> {df.index.max().date()}")
    except Exception as exc:
        print(f"FAILED: {exc}")

# ---------------------------------------------------------------------------
# 2. FRED VIXCLS
# ---------------------------------------------------------------------------
print()
print("=" * 65)
print("Step 2 — FRED VIXCLS")
print("=" * 65)
vix_cache = RAW_DIR / "VIXCLS.csv"
if vix_cache.exists():
    print("  VIXCLS  CACHED")
else:
    try:
        vix = load_fred_series("VIXCLS")
        vix.to_csv(vix_cache, header=True)
        print(f"  VIXCLS  OK  {len(vix)} rows")
    except Exception as exc:
        print(f"  VIXCLS  FAILED: {exc}")

# ---------------------------------------------------------------------------
# 3. NAB files
# ---------------------------------------------------------------------------
NAB_DIR = ROOT / "data" / "nab"
NAB_DIR.mkdir(parents=True, exist_ok=True)

NAB_BASE  = "https://raw.githubusercontent.com/numenta/NAB/master/data"
NAB_LABEL = "https://raw.githubusercontent.com/numenta/NAB/master/labels/combined_windows.json"
NAB_FILES = [
    "realTraffic/speed_6005.csv",
    "realKnownCause/ambient_temperature_system_failure.csv",
    "realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv",
]

print()
print("=" * 65)
print("Step 3 — NAB anomaly benchmark files")
print("=" * 65)
for rel in NAB_FILES:
    dest = NAB_DIR / Path(rel).name
    if dest.exists():
        print(f"  {Path(rel).name:<52} CACHED")
        continue
    print(f"  {Path(rel).name:<52}", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(f"{NAB_BASE}/{rel}", dest)
        print("OK")
    except Exception as exc:
        print(f"FAILED: {exc}")

label_dest = NAB_DIR / "combined_windows.json"
if label_dest.exists():
    print(f"  {'combined_windows.json':<52} CACHED")
else:
    print(f"  {'combined_windows.json':<52}", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(NAB_LABEL, label_dest)
        print("OK")
    except Exception as exc:
        print(f"FAILED: {exc}")

print()
print("All downloads complete.")
