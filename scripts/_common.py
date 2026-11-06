"""Shared CLI plumbing for the scripts."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # run without pip install

from deepvar.config import load_config  # noqa: E402


def parse_args(description: str, extra=None):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=None, help="path to YAML config")
    p.add_argument("--ticker", default=None, help="override data.ticker, e.g. ^GSPC")
    p.add_argument("--source", default=None, choices=["yfinance", "csv", "synthetic"])
    p.add_argument("--csv-path", default=None)
    p.add_argument("--epochs", type=int, default=None)
    if extra:
        extra(p)
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.ticker:
        cfg["data"]["ticker"] = args.ticker
    if args.source:
        cfg["data"]["source"] = args.source
    if args.csv_path:
        cfg["data"]["csv_path"] = args.csv_path
    if args.epochs:
        cfg["model"]["epochs"] = args.epochs
    return cfg, args
