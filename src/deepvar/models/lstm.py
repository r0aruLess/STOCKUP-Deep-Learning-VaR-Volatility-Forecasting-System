"""The 2-layer LSTM volatility forecaster described in the project overview."""
from __future__ import annotations

import torch
from torch import nn


class VolLSTM(nn.Module):
    """Input  : (batch, window=60, n_features)
    LSTM 1   : 64 units, returns the whole sequence (keeps temporal memory)
    Dropout  : 0.2 (randomly switches off 20% of units while training)
    LSTM 2   : 32 units, we keep only the LAST time step
    Dense    : 1 output = (scaled) predicted volatility sigma_{t+1}
    """

    def __init__(self, n_features: int, hidden1: int = 64, hidden2: int = 32, dropout: float = 0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, hidden1, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.head = nn.Linear(hidden2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm1(x)          # (B, T, hidden1)
        out = self.drop(out)
        out, _ = self.lstm2(out)        # (B, T, hidden2)
        return self.head(out[:, -1, :]).squeeze(-1)  # (B,)
