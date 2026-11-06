"""Train the LSTM.   python scripts/train.py [--ticker ^GSPC] [--source synthetic]"""
import pandas as pd
from _common import parse_args

from deepvar.config import REPORT_DIR
from deepvar.pipeline import artifact_path, load_data, train_forecaster

cfg, _ = parse_args("Train the LSTM volatility model")
df = load_data(cfg)
print(f"loaded {len(df)} rows for {cfg['data']['ticker']}: {df.index.min().date()} -> {df.index.max().date()}")
fc, history, ds = train_forecaster(cfg, df)

out = artifact_path(cfg)
fc.save(out, cfg, history)
REPORT_DIR.mkdir(exist_ok=True)
pd.DataFrame(history).to_csv(REPORT_DIR / "loss_curve.csv", index_label="epoch")
gap = history["val_loss"][-1] / max(history["train_loss"][-1], 1e-9)
print(f"\nsaved model -> {out}")
print(f"final train MSE {history['train_loss'][-1]:.4f} | val MSE {history['val_loss'][-1]:.4f} "
      f"(val/train = {gap:.1f}x; a LARGE ratio means overfitting)")
