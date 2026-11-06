"""Train the LSTM on every asset defined in configs/.

Auto-discovers every *.yaml in configs/ — no hardcoded list needed.

Usage:
    python scripts/train_all.py [--skip-existing]

Flags:
    --skip-existing   Skip assets whose model artifact already exists.
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from deepvar.config import ARTIFACT_DIR, REPORT_DIR, load_config
from deepvar.pipeline import artifact_path, load_data, train_forecaster

parser = argparse.ArgumentParser()
parser.add_argument("--skip-existing", action="store_true",
                    help="Skip assets whose model artifact already exists")
args = parser.parse_args()

ROOT       = Path(__file__).resolve().parents[1]
REPORT_DIR.mkdir(parents=True, exist_ok=True)

cfg_paths  = sorted((ROOT / "configs").glob("*.yaml"))
results    = []

print(f"Found {len(cfg_paths)} config(s) in configs/")
print()

for cfg_path in cfg_paths:
    cfg    = load_config(cfg_path)
    ticker = cfg["data"]["ticker"]
    safe   = re.sub(r"[^A-Za-z0-9]+", "_", ticker).strip("_")
    art    = artifact_path(cfg)

    if args.skip_existing and (art / "model.pt").exists():
        print(f"  {ticker:<22} SKIPPED (model exists)")
        continue

    print("=" * 65)
    print(f"  Training: {ticker}  ({cfg_path.name})")
    print("=" * 65)

    try:
        df = load_data(cfg)
        print(f"  Data: {len(df)} rows  "
              f"{df.index.min().date()} -> {df.index.max().date()}")

        fc, history, ds = train_forecaster(cfg, df, verbose=True)
        fc.save(art, cfg, history)

        loss_csv = REPORT_DIR / f"loss_curve_{safe}.csv"
        pd.DataFrame(history).to_csv(loss_csv, index_label="epoch")

        best_val   = min(history["val_loss"])
        final_tr   = history["train_loss"][-1]
        gap        = best_val / max(final_tr, 1e-9)
        epochs_run = len(history["train_loss"])

        print(f"\n  Saved -> {art}")
        print(f"  Epochs: {epochs_run} | "
              f"Train MSE: {final_tr:.4f} | "
              f"Best Val MSE: {best_val:.4f} | "
              f"Val/Train: {gap:.2f}x")
        print()

        results.append({
            "config": cfg_path.name, "ticker": ticker,
            "train_rows": ds.info["n_train"],
            "val_rows":   ds.info["n_val"],
            "test_rows":  ds.info["n_test"],
            "epochs_run": epochs_run,
            "train_mse":  round(final_tr, 5),
            "best_val_mse": round(best_val, 5),
            "val_train_ratio": round(gap, 2),
            "status": "OK",
        })

    except Exception as exc:
        print(f"  FAILED: {exc}\n")
        results.append({"config": cfg_path.name, "ticker": ticker,
                        "status": f"FAILED: {exc}"})

# Summary
summary = pd.DataFrame(results)
summary.to_csv(REPORT_DIR / "training_summary.csv", index=False)
print()
print("=" * 65)
print("  TRAINING SUMMARY")
print("=" * 65)
pd.set_option("display.width", 200)
cols = [c for c in ["ticker","epochs_run","train_mse","best_val_mse",
                     "val_train_ratio","train_rows","val_rows","test_rows","status"]
        if c in summary.columns]
print(summary[cols].to_string(index=False))
print(f"\nSaved reports/training_summary.csv")
