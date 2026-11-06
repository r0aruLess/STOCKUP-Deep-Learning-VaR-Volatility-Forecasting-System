"""Training loop with early stopping (the part Keras hides inside model.fit)."""
from __future__ import annotations

import copy
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from deepvar.models.lstm import VolLSTM


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _loss_on(model, X, y, loss_fn, batch=1024):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch])
            yb = torch.from_numpy(y[i:i + batch])
            total += loss_fn(model(xb), yb).item() * len(xb)
            n += len(xb)
    return total / max(n, 1)


def train_model(ds, cfg_model: dict, verbose: bool = True):
    """Train on ds.X_train, early-stop on ds.X_val. Returns (best_model, history)."""
    if len(ds.X_train) == 0 or len(ds.X_val) == 0:
        raise ValueError("Empty train/val set - check train_end / val_end dates and data length.")
    set_seed(cfg_model["seed"])
    n_features = ds.X_train.shape[-1]
    model = VolLSTM(n_features, cfg_model["hidden1"], cfg_model["hidden2"], cfg_model["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg_model["lr"])
    loss_fn = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(torch.from_numpy(ds.X_train), torch.from_numpy(ds.y_train)),
        batch_size=cfg_model["batch_size"], shuffle=True,
    )
    history = {"train_loss": [], "val_loss": []}
    best_val, best_state, bad = float("inf"), None, 0

    for epoch in range(1, cfg_model["epochs"] + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        tr = _loss_on(model, ds.X_train, ds.y_train, loss_fn)
        va = _loss_on(model, ds.X_val, ds.y_val, loss_fn)
        history["train_loss"].append(tr)
        history["val_loss"].append(va)
        if verbose:
            print(f"epoch {epoch:3d} | train MSE {tr:.4f} | val MSE {va:.4f}")
        if va < best_val - 1e-5:
            best_val, best_state, bad = va, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg_model["patience"]:
                if verbose:
                    print(f"early stop at epoch {epoch} (best val MSE {best_val:.4f})")
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, history
