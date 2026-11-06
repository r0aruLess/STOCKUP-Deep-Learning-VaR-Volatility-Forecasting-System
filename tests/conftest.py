import pytest

from deepvar.config import load_config
from deepvar.data.synthetic import make_synthetic_ohlcv


@pytest.fixture(scope="session")
def df():
    return make_synthetic_ohlcv(n_days=2400, seed=1)


@pytest.fixture(scope="session")
def cfg():
    c = load_config()
    c["data"].update({"source": "synthetic", "train_end": "2018-12-31", "val_end": "2019-12-31"})
    c["model"].update({"epochs": 3, "patience": 3, "hidden1": 8, "hidden2": 4})
    return c
