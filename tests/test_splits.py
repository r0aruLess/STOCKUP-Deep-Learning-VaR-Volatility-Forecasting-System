import numpy as np
import pandas as pd

from deepvar.data.features import build_features, make_target
from deepvar.data.windows import build_dataset


def test_chronological_split_no_leak(df):
    H, W = 5, 60
    f = build_features(df, 21)
    y = make_target(f["log_ret"], H)
    ds = build_dataset(f, y, W, "2018-12-31", "2019-12-31", H)

    assert ds.X_train.shape[1:] == (W, 3)
    assert ds.dates_test.min() > pd.Timestamp("2019-12-31")
    # scaler fitted on TRAIN rows only -> train features have mean ~0
    assert np.allclose(ds.X_train[:, -1, :].mean(axis=0), 0, atol=0.25)
    # purge: last H train samples dropped
    train_dates = (f.index[W - 1:] <= pd.Timestamp("2018-12-31")).sum()
    assert ds.info["n_train"] <= train_dates - H
