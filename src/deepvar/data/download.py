"""Data acquisition: Yahoo Finance (yfinance), local CSV (Kaggle), FRED."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from deepvar.config import RAW_DIR
from deepvar.data.synthetic import make_synthetic_ohlcv

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean OHLCV frame: DatetimeIndex, sorted, no dupes, positive prices."""
    if isinstance(df.columns, pd.MultiIndex):  # newer yfinance returns (field, ticker)
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={c: str(c).strip().title() for c in df.columns})
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    missing = [c for c in OHLCV if c not in df.columns]
    if missing:
        raise ValueError(f"Data is missing required columns: {missing}")

    df = df[OHLCV].copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "Date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df[(df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    return df


def download_yfinance(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    import yfinance as yf  # imported lazily so tests run without network

    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError(
            f"yfinance returned no data for {ticker}. Check the ticker / your internet, "
            "or use source: csv / synthetic in the config."
        )
    return standardize_ohlcv(raw)


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a Kaggle-style CSV with Date/Open/High/Low/Close/Volume columns."""
    raw = pd.read_csv(path)
    date_col = next((c for c in raw.columns if c.lower() in {"date", "timestamp", "time"}), None)
    if date_col is None:
        raise ValueError("CSV needs a Date/Timestamp column")
    raw = raw.set_index(date_col)
    raw.index = pd.to_datetime(raw.index)
    return standardize_ohlcv(raw)


def load_fred_series(series_id: str = "VIXCLS") -> pd.Series:
    """Download one FRED series (e.g. VIXCLS = CBOE VIX, SP500 = S&P 500 level)."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    s = pd.read_csv(url, index_col=0, parse_dates=True).iloc[:, 0]
    return pd.to_numeric(s, errors="coerce").dropna().rename(series_id)


def load_prices(
    ticker: str,
    start: str,
    end: str | None = None,
    source: str = "yfinance",
    csv_path: str | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Single entry point used by the rest of the project (caches to data/raw)."""
    if source == "synthetic":
        return make_synthetic_ohlcv(start=start)
    if source == "csv":
        if not csv_path:
            raise ValueError("source=csv requires data.csv_path")
        return load_csv(csv_path)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9]+", "_", ticker).strip("_")
    cache = RAW_DIR / f"{safe}.csv"
    if cache.exists() and not refresh:
        return load_csv(cache)
    df = download_yfinance(ticker, start, end)
    df.to_csv(cache)
    return df
