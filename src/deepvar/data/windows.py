"""Scaling, sliding windows (3D tensors) and leak-free chronological splits."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from deepvar.data.features import FEATURE_COLS


class Scaler:
    """Tiny standardiser (mean/std) that we can save as JSON."""

    def __init__(self, mean=None, std=None):
        self.mean = None if mean is None else np.asarray(mean, dtype=float)
        self.std = None if std is None else np.asarray(std, dtype=float)

    def fit(self, arr):
        arr = np.asarray(arr, dtype=float)
        self.mean = arr.mean(axis=0)
        self.std = np.maximum(arr.std(axis=0), 1e-12)
        return self

    def transform(self, arr):
        return (np.asarray(arr, dtype=float) - self.mean) / self.std

    def inverse(self, arr):
        return np.asarray(arr, dtype=float) * self.std + self.mean

    def to_dict(self):
        return {"mean": np.atleast_1d(self.mean).tolist(), "std": np.atleast_1d(self.std).tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(d["mean"], d["std"])


def make_windows(x2d: np.ndarray, window: int) -> np.ndarray:
    """(N, F) -> (N-window+1, window, F): the 3D tensor an LSTM expects."""
    x2d = np.asarray(x2d, dtype=np.float32)
    if len(x2d) < window:
        return np.empty((0, window, x2d.shape[1]), dtype=np.float32)
    w = sliding_window_view(x2d, (window, x2d.shape[1]))[:, 0]
    return np.ascontiguousarray(w)


@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    dates_test: pd.DatetimeIndex
    feat_scaler: Scaler
    target_scaler: Scaler
    info: dict = field(default_factory=dict)


def build_dataset(
    feats: pd.DataFrame,
    target: pd.Series,
    window: int,
    train_end: str,
    val_end: str,
    horizon: int,
) -> Dataset:
    """Chronological train/val/test split with scaler fitted on TRAIN ONLY.

    Leak protection:
      1. Split strictly by date (no shuffling across time).
      2. Scalers are fitted on training rows only.
      3. "Purge": the last `horizon` samples of train and val are dropped because
         their targets use returns from the *next* period.
    """
    cols = FEATURE_COLS
    feat_scaler = Scaler().fit(feats.loc[:train_end, cols].values)
    Xs = feat_scaler.transform(feats[cols].values)
    W = make_windows(Xs, window)
    dates = feats.index[window - 1:]
    y = target.reindex(feats.index).values[window - 1:]

    ok = ~np.isnan(y)
    W, y, dates = W[ok], y[ok], dates[ok]

    tr = dates <= pd.Timestamp(train_end)
    va = (dates > pd.Timestamp(train_end)) & (dates <= pd.Timestamp(val_end))
    te = dates > pd.Timestamp(val_end)

    def purge(mask):
        idx = np.where(mask)[0]
        return idx[:-horizon] if len(idx) > horizon else idx[:0]

    i_tr, i_va, i_te = purge(tr), purge(va), np.where(te)[0]
    target_scaler = Scaler().fit(y[i_tr].reshape(-1, 1))

    def ys(i):
        return target_scaler.transform(y[i].reshape(-1, 1)).ravel().astype(np.float32)

    return Dataset(
        X_train=W[i_tr], y_train=ys(i_tr),
        X_val=W[i_va], y_val=ys(i_va),
        X_test=W[i_te], y_test=ys(i_te),
        dates_test=dates[i_te],
        feat_scaler=feat_scaler, target_scaler=target_scaler,
        info={"n_train": len(i_tr), "n_val": len(i_va), "n_test": len(i_te)},
    )
