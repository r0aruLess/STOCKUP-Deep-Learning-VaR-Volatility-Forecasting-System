import numpy as np

from deepvar.risk.backtest import christoffersen_independence, kupiec_pof


def test_kupiec_perfect_calibration():
    b = np.zeros(100, dtype=bool); b[0] = True       # exactly 1% breaches
    r = kupiec_pof(b, 0.99)
    assert r["LR_pof"] < 1e-9 and not r["reject_H0"]


def test_kupiec_rejects_too_many_breaches():
    b = np.zeros(100, dtype=bool); b[:10] = True     # 10% breaches at 99% VaR
    assert kupiec_pof(b, 0.99)["reject_H0"]


def test_kupiec_zero_breaches_edge_case():
    r = kupiec_pof(np.zeros(250, dtype=bool), 0.99)
    assert np.isfinite(r["LR_pof"]) and r["reject_H0"]  # 0 breaches in 250 days is *too safe*


def test_christoffersen_detects_clustering():
    b = np.zeros(300, dtype=int); b[100:108] = 1     # one tight cluster
    assert christoffersen_independence(b)["reject_H0"]
