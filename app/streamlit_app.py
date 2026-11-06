"""DeepVaR Sentinel — interactive dashboard.
Run:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import streamlit as st

from deepvar.config import load_config
from deepvar.pipeline import (Forecaster, SecureForecaster, artifact_path,
                               backtest_report, load_data, trusted_history)
from deepvar.risk.var import (calibrate_quantile, var_amount,
                               var_floor_price, var_threshold_calibrated)
from deepvar.risk.backtest import kupiec_pof
from deepvar.security.attacks import ATTACKS
from deepvar.security.evaluate import most_volatile_start

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DeepVaR Sentinel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .stMetric label { font-size: 0.78rem !important; color: #9ca3af !important; }
    .stMetric [data-testid="metric-container"] { background:#111827; border-radius:10px;
        padding:14px 18px; border: 1px solid #1f2937; }
    div[data-testid="stSidebar"] { background: #0f172a; }
    .title-row { display:flex; align-items:center; gap:10px; margin-bottom:4px; }
    .badge { display:inline-block; padding:2px 10px; border-radius:9999px;
             font-size:0.75rem; font-weight:600; }
    .badge-pass { background:#14532d; color:#4ade80; }
    .badge-fail { background:#7f1d1d; color:#f87171; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Asset catalogue — all 20 trained assets
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

ASSET_CATALOGUE = {
    # label                           : (ticker,        config_file,          category)
    "🟡 Bitcoin (BTC-USD)"           : ("BTC-USD",      "default.yaml",       "Crypto"),
    "🟡 Ethereum (ETH-USD)"          : ("ETH-USD",      "eth.yaml",           "Crypto"),
    "🟡 Ripple (XRP-USD)"            : ("XRP-USD",      "xrp.yaml",           "Crypto"),
    "🔵 S&P 500 (^GSPC)"             : ("^GSPC",        "sp500.yaml",         "US Index"),
    "🔵 Dow Jones (^DJI)"            : ("^DJI",         "dow.yaml",           "US Index"),
    "🔵 NASDAQ (^IXIC)"              : ("^IXIC",        "nasdaq.yaml",        "US Index"),
    "🟢 Apple (AAPL)"                : ("AAPL",         "aapl.yaml",          "US Equity"),
    "🟢 Microsoft (MSFT)"            : ("MSFT",         "msft.yaml",          "US Equity"),
    "🟢 Tesla (TSLA)"                : ("TSLA",         "tsla.yaml",          "US Equity"),
    "🟠 Nifty 50 (^NSEI)"            : ("^NSEI",        "nifty50.yaml",       "India Index"),
    "🟠 HDFC Bank (HDFCBANK.NS)"     : ("HDFCBANK.NS",  "india_bank_hdfc.yaml","India Equity"),
    "🟠 ICICI Bank (ICICIBANK.NS)"   : ("ICICIBANK.NS", "india_bank_icici.yaml","India Equity"),
    "🟠 Infosys (INFY.NS)"           : ("INFY.NS",      "infy.yaml",          "India Equity"),
    "🟠 Reliance (RELIANCE.NS)"      : ("RELIANCE.NS",  "reliance.yaml",      "India Equity"),
    "🟠 TCS (TCS.NS)"                : ("TCS.NS",       "tcs.yaml",           "India Equity"),
    "🌏 Hang Seng (^HSI)"            : ("^HSI",         "hangseng.yaml",      "Global Index"),
    "🌏 Nikkei 225 (^N225)"          : ("^N225",        "nikkei.yaml",        "Global Index"),
    "🌏 FTSE 100 (^FTSE)"            : ("^FTSE",        "ftse.yaml",          "Global Index"),
    "⚫ Gold Futures (GC=F)"          : ("GC=F",         "gold.yaml",          "Commodity"),
    "⚫ Crude Oil (CL=F)"             : ("CL=F",         "crude_oil.yaml",     "Commodity"),
}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛡️ DeepVaR Sentinel")
    st.caption("LSTM volatility forecasting → VaR → adversarial robustness")
    st.divider()

    st.subheader("📊 Asset")
    label = st.selectbox("Select asset", list(ASSET_CATALOGUE), label_visibility="collapsed")
    ticker, cfg_file, category = ASSET_CATALOGUE[label]

    st.subheader("⚙️ Risk parameters")
    conf  = st.radio("VaR confidence", [0.95, 0.99], index=1,
                     format_func=lambda c: f"{c:.0%}", horizontal=True)
    value = st.number_input("Portfolio value ($)", min_value=1000,
                            value=1_000_000, step=50_000)
    dist  = st.selectbox("Return distribution", ["t", "normal"],
                         help="'t' has fatter tails — usually better for crypto/equities")

    st.divider()
    st.subheader("🔐 Security test")
    attack_on = st.toggle("Simulate adversarial data injection", value=False)
    attack    = st.selectbox("Attack type", list(ATTACKS), index=2,
                             disabled=not attack_on)
    n_days    = st.slider("Attack length (days)", 1, 30, 10,
                          disabled=not attack_on)
    strength  = st.slider("Suppression factor α (↓ = stealthier)", 0.1, 0.9, 0.2,
                          disabled=not (attack_on and attack == "vol_suppression"))

# ---------------------------------------------------------------------------
# Load data + model (cached per ticker)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="⏳ Loading data and model…")
def get_system(ticker: str, cfg_file: str):
    cfg_path = ROOT / "configs" / cfg_file
    cfg = load_config(cfg_path)
    df  = load_data(cfg)
    art = artifact_path(cfg)
    if (art / "model.pt").exists():
        fc = Forecaster.load(art)
    else:
        st.warning("Model not trained yet — training now (may take ~30s)…")
        cfg["model"]["epochs"] = 30
        fc, history, _ = train_forecaster(cfg, df, verbose=False)
        fc.save(art, cfg, history)
    sec = SecureForecaster.build(fc, cfg, trusted_history(cfg, df))
    return cfg, df, fc, sec


st.title(f"🛡️ DeepVaR Sentinel  —  {label}")
st.caption(f"Category: **{category}** · Ticker: `{ticker}` · Config: `{cfg_file}`")

try:
    cfg, df, fc, sec = get_system(ticker, cfg_file)
except Exception as exc:
    st.error(f"Could not load data/model: {exc}")
    st.stop()

cfg["risk"].update({"confidence": conf, "distribution": dist})
rk         = dict(confidence=conf, mu=cfg["risk"]["mu"],
                  distribution=dist, t_df=cfg["risk"]["t_df"])
val_start  = str((pd.Timestamp(cfg["data"]["train_end"])
                  + pd.Timedelta(days=1)).date())
val_end    = cfg["data"]["val_end"]
test_start = str((pd.Timestamp(val_end) + pd.Timedelta(days=1)).date())
log_ret    = np.log(df["Close"] / df["Close"].shift(1))

# ---------------------------------------------------------------------------
# Forecasts
# ---------------------------------------------------------------------------
sig_clean = fc.sigma(df)

df_view, sig_vanilla, sig_secure, res = df, None, None, sec.gate.check(df)
if attack_on:
    start = most_volatile_start(df, cfg["data"]["val_end"], 20)
    kw    = {"alpha": strength} if attack == "vol_suppression" else {}
    df_view, _  = ATTACKS[attack](df, start, n_days, **kw)
    sig_vanilla, _   = fc.sigma(df_view), None
    sig_secure, res  = sec.sigma(df_view)
    st.warning(
        f"⚠️ Simulating **{attack}** for **{n_days} day(s)** starting "
        f"**{df.index[start].date()}** — most turbulent stretch of the test period."
    )

# ---------------------------------------------------------------------------
# Calibrated quantile (McNeil & Frey 2000)
# ---------------------------------------------------------------------------
try:
    sig_val = sig_clean.loc[val_start:val_end]
    ret_val = log_ret.loc[val_start:val_end]
    cal_q   = calibrate_quantile(sig_val, ret_val, conf)

    sig_test = sig_clean.loc[test_start:]
    thr_cal  = pd.Series(var_threshold_calibrated(sig_test.values, cal_q),
                         index=sig_test.index)
    ret_test = log_ret.reindex(thr_cal.index)
    aligned  = pd.DataFrame({"thr": thr_cal, "ret": ret_test}).dropna()
    breaches_cal = (aligned["ret"] < aligned["thr"]).values
    k_cal    = kupiec_pof(breaches_cal, conf)
    has_cal  = True
except Exception:
    has_cal  = False
    k_cal    = {}

# ---------------------------------------------------------------------------
# Backtest (raw)
# ---------------------------------------------------------------------------
rep = backtest_report(cfg, df, sig_clean, start=test_start)
k   = rep["kupiec"]

live_sigma   = (sig_secure[0] if isinstance(sig_secure, tuple)
                else sig_clean).iloc[-1] if attack_on else sig_clean.iloc[-1]

recent_flags = res.flagged.iloc[-30:].any() if not attack_on else \
    res.flagged.iloc[start: start + n_days + 10].any()

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
st.divider()
c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    f"1-Day {conf:.0%} VaR (latest)",
    f"${float(var_amount(live_sigma, value, **rk)):,.0f}",
    help="Max expected daily loss at this confidence"
)
c2.metric("Daily volatility (σ̂)", f"{live_sigma:.2%}")
c3.metric(
    "Kupiec — Raw LSTM",
    "PASS ✅" if not k["reject_H0"] else "FAIL ❌",
    f"{k['breaches']} / {k['expected']:.1f} expected  (p={k['p_value']:.3f})",
    delta_color="off",
)
if has_cal:
    c4.metric(
        "Kupiec — Calibrated LSTM",
        "PASS ✅" if not k_cal["reject_H0"] else "FAIL ❌",
        f"{k_cal['breaches']} / {k_cal['expected']:.1f} expected  "
        f"(p={k_cal['p_value']:.3f})  cal_q={cal_q:.3f}",
        delta_color="off",
        help="McNeil & Frey (2000): val-set empirical tail replaces t-distribution z-score",
    )
else:
    c4.metric("Kupiec — Calibrated LSTM", "N/A", "Insufficient val data")

c5.metric(
    "Data integrity gate",
    "🔴 ANOMALY DETECTED" if recent_flags else "🟢 CLEAN",
    help="2-rule input validation gate status"
)

# ---------------------------------------------------------------------------
# Main chart
# ---------------------------------------------------------------------------
st.divider()
horizon = st.slider("Days shown in chart", 60, min(900, len(sig_clean)), 365)
idx     = sig_clean.index[-horizon:]

chart = pd.DataFrame({"Close price": df["Close"].reindex(idx)})
if attack_on:
    chart["Close (tampered feed)"]          = df_view["Close"].reindex(idx)
    chart["VaR floor — vanilla LSTM"]       = var_floor_price(
        df_view["Close"], fc.sigma(df_view), **rk).reindex(idx)
    chart["VaR floor — secure pipeline"]    = var_floor_price(
        df_view["Close"],
        sig_secure[0] if isinstance(sig_secure, tuple) else sig_secure,
        **rk).reindex(idx)
    colors = ["#60a5fa", "#93c5fd", "#ef4444", "#22c55e"]
else:
    chart["VaR floor — Raw LSTM"]           = var_floor_price(
        df["Close"], sig_clean, **rk).reindex(idx)
    colors = ["#60a5fa", "#ef4444"]

st.subheader("📈 Price vs Dynamic VaR Floor")
st.line_chart(chart, color=colors)
st.caption(
    "Red line = VaR floor. If tomorrow's close drops below it, that day is a breach. "
    "A floor that sinks far below the price during turbulence = model under-estimating risk."
)

# ---------------------------------------------------------------------------
# Attack metrics
# ---------------------------------------------------------------------------
if attack_on:
    t0 = df.index[start]
    t1 = df.index[min(start + n_days + 10, len(df) - 1)]
    get_m = lambda s: float(s.loc[t0:t1].mean())   # noqa: E731
    sig_sec_series = (sig_secure[0] if isinstance(sig_secure, tuple) else sig_secure)

    st.divider()
    st.subheader("🔐 Attack impact")
    a, b, c = st.columns(3)
    a.metric("Avg σ — clean data",        f"{get_m(sig_clean):.2%}")
    b.metric("Avg σ — vanilla LSTM",      f"{get_m(fc.sigma(df_view)):.2%}",
             f"{get_m(fc.sigma(df_view))/get_m(sig_clean)-1:+.0%}", delta_color="inverse")
    c.metric("Avg σ — secure pipeline",   f"{get_m(sig_sec_series):.2%}",
             f"{get_m(sig_sec_series)/get_m(sig_clean)-1:+.0%}", delta_color="inverse")
    st.dataframe(res.flags.loc[t0:t1].astype(int).T, use_container_width=True)
    st.caption("Rows = validation rules · Columns = days · **1** = rule fired")

# ---------------------------------------------------------------------------
# Calibration detail expander
# ---------------------------------------------------------------------------
if has_cal:
    with st.expander("📐 Calibration detail — McNeil & Frey (2000)"):
        t5_z = float(__import__("scipy").stats.t.ppf(1 - conf,
               cfg["risk"]["t_df"]) * __import__("numpy").sqrt(
               (cfg["risk"]["t_df"] - 2) / cfg["risk"]["t_df"]))
        norm_z = float(__import__("scipy").stats.norm.ppf(1 - conf))
        st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| Confidence level | {conf:.0%} |
| Theoretical z (t₅) | `{t5_z:.4f}` |
| Theoretical z (normal) | `{norm_z:.4f}` |
| **Empirical cal_q (val set)** | **`{cal_q:.4f}`** |
| Calibrated breaches (test) | {k_cal['breaches']} vs {k_cal['expected']:.1f} expected |
| Kupiec p-value | {k_cal['p_value']:.4f} |
| Verdict | {"✅ PASS" if not k_cal["reject_H0"] else "❌ FAIL"} |

> **Interpretation:** The empirical tail multiplier `cal_q = {cal_q:.3f}` replaces the
> theoretical z-score. If `|cal_q| > |z_theoretical|`, real tails are fatter than assumed.
        """)

# ---------------------------------------------------------------------------
# About
# ---------------------------------------------------------------------------
with st.expander("ℹ️ About this model"):
    st.markdown("""
- **Architecture:** 2-layer LSTM → sigma forecast → parametric VaR (t/normal distribution)
- **Calibration:** McNeil & Frey (2000) volatility-filtered historical simulation on val set
- **Security gate:** 2-rule input validation (hard_limit + collapse) + fail-safe VaR floor
- **Baselines:** EWMA (RiskMetrics λ=0.94), Rolling 21d std, Historical simulation (250d)
- **Training split:** Train 2015–2021 | Val 2022–2023 | Test 2024 onward
- ⚠️ Educational project — not financial advice, not a production risk system.
    """)
