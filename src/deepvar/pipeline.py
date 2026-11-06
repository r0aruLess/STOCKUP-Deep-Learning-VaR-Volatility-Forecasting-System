"""End-to-end orchestration: data -> features -> LSTM -> sigma -> VaR (+ secure variant)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from deepvar.data.download import load_prices
from deepvar.data.features import FEATURE_COLS, build_features, make_target
from deepvar.data.windows import Dataset, Scaler, build_dataset, make_windows
from deepvar.models.lstm import VolLSTM
from deepvar.models.train import train_model
from deepvar.risk.backtest import breach_frame, christoffersen_independence, kupiec_pof
from deepvar.security.defenses import InputValidationGate, apply_failsafe

SIGMA_FLOOR = 1e-4


# ---------------------------------------------------------------- data prep
def load_data(cfg: dict) -> pd.DataFrame:
    d = cfg["data"]
    return load_prices(d["ticker"], d["start"], d.get("end"), d.get("source", "yfinance"),
                       d.get("csv_path"))


def prepare_dataset(cfg: dict, df: pd.DataFrame) -> Dataset:
    d = cfg["data"]
    feats = build_features(df, d["vol_window"])
    target = make_target(feats["log_ret"], d["target_horizon"])
    return build_dataset(feats, target, cfg["model"]["window"], d["train_end"], d["val_end"],
                         d["target_horizon"])


# ---------------------------------------------------------------- forecaster
@dataclass
class Forecaster:
    """Wraps model + scalers: raw OHLCV in -> daily sigma forecast out."""
    model: VolLSTM
    feat_scaler: Scaler
    target_scaler: Scaler
    window: int
    vol_window: int

    def sigma(self, df: pd.DataFrame) -> pd.Series:
        feats = build_features(df, self.vol_window)
        if len(feats) < self.window:
            return pd.Series(dtype=float, name="sigma")
        X = make_windows(self.feat_scaler.transform(feats[FEATURE_COLS].values), self.window)
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.from_numpy(X)).numpy()
        sig = self.target_scaler.inverse(out.reshape(-1, 1)).ravel()
        return pd.Series(np.maximum(sig, SIGMA_FLOOR), index=feats.index[self.window - 1:],
                         name="sigma")

    # -- persistence
    def save(self, out_dir: Path, cfg: dict, history: dict | None = None) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), out_dir / "model.pt")
        meta = {"cfg": cfg, "feat_scaler": self.feat_scaler.to_dict(),
                "target_scaler": self.target_scaler.to_dict(), "history": history}
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, out_dir: Path) -> "Forecaster":
        out_dir = Path(out_dir)
        meta = json.loads((out_dir / "meta.json").read_text())
        m = meta["cfg"]["model"]
        model = VolLSTM(len(FEATURE_COLS), m["hidden1"], m["hidden2"], m["dropout"])
        model.load_state_dict(torch.load(out_dir / "model.pt", map_location="cpu"))
        model.eval()
        return cls(model, Scaler.from_dict(meta["feat_scaler"]),
                   Scaler.from_dict(meta["target_scaler"]), m["window"],
                   meta["cfg"]["data"]["vol_window"])


def train_forecaster(cfg: dict, df: pd.DataFrame, verbose: bool = True):
    ds = prepare_dataset(cfg, df)
    if verbose:
        print("dataset:", ds.info, "| X_train", ds.X_train.shape)
    model, history = train_model(ds, cfg["model"], verbose=verbose)
    fc = Forecaster(model, ds.feat_scaler, ds.target_scaler, cfg["model"]["window"],
                    cfg["data"]["vol_window"])
    return fc, history, ds


# ---------------------------------------------------------------- secure variant
class SecureForecaster:
    """Gate -> repair -> LSTM -> fail-safe floor."""

    def __init__(self, base: Forecaster, gate: InputValidationGate, sec_cfg: dict):
        self.base, self.gate, self.cfg = base, gate, sec_cfg

    @classmethod
    def build(cls, base: Forecaster, cfg: dict, trusted_df: pd.DataFrame) -> "SecureForecaster":
        gate = InputValidationGate(vol_window=cfg["data"]["vol_window"], seed=cfg["model"]["seed"],
                                   **cfg["security"]).fit(trusted_df)
        return cls(base, gate, cfg["security"])

    def sigma(self, df: pd.DataFrame):
        res = self.gate.check(df)
        s = self.base.sigma(res.repaired)
        if s.empty:
            return s, res
        flagged = res.flagged.reindex(s.index).fillna(False)
        s = apply_failsafe(s, flagged, self.cfg["hold_days"], self.cfg["ref_window"],
                           self.cfg["collapse_window"])
        return s, res


# ---------------------------------------------------------------- reporting
def backtest_report(cfg: dict, df: pd.DataFrame, sigma: pd.Series, start: str | None = None) -> dict:
    r = cfg["risk"]
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    bf = breach_frame(log_ret, sigma, r["confidence"], r["mu"], r["distribution"], r["t_df"])
    if start:
        bf = bf.loc[start:]
    out = kupiec_pof(bf["breach"].values, r["confidence"])
    out["independence_p"] = christoffersen_independence(bf["breach"].values)["p_value"]
    return {"frame": bf, "kupiec": out}


# ---------------------------------------------------------------- helpers for scripts / app
def artifact_path(cfg: dict) -> Path:
    import re
    from deepvar.config import ARTIFACT_DIR
    safe = re.sub(r"[^A-Za-z0-9]+", "_", cfg["data"]["ticker"]).strip("_")
    tag = "synthetic" if cfg["data"].get("source") == "synthetic" else safe
    return ARTIFACT_DIR / tag


def trusted_history(cfg: dict, df: pd.DataFrame) -> pd.DataFrame:
    """The slice of data we trust enough to fit the gate on (the training period)."""
    return df.loc[: cfg["data"]["train_end"]]
