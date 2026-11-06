import numpy as np
import torch

from deepvar.models.lstm import VolLSTM
from deepvar.pipeline import Forecaster, prepare_dataset, train_forecaster


def test_lstm_output_shape():
    m = VolLSTM(3, 8, 4)
    assert m(torch.randn(5, 60, 3)).shape == (5,)


def test_training_runs_and_save_load_roundtrip(df, cfg, tmp_path):
    fc, history, ds = train_forecaster(cfg, df, verbose=False)
    assert history["train_loss"][-1] <= history["train_loss"][0]
    s = fc.sigma(df)
    assert s.index[0] == df.index[0:].to_series().loc[s.index[0]] or len(s) > 0
    assert (s > 0).all() and not s.isna().any()

    fc.save(tmp_path, cfg, history)
    fc2 = Forecaster.load(tmp_path)
    assert np.allclose(fc2.sigma(df).values, s.values, atol=1e-6)


def test_dataset_shapes(df, cfg):
    ds = prepare_dataset(cfg, df)
    assert ds.X_train.ndim == 3 and ds.X_train.shape[1] == cfg["model"]["window"]
    assert len(ds.X_train) == len(ds.y_train) > 0
