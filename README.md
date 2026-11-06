# 🛡️ DeepVaR Sentinel

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)
![Tests](https://img.shields.io/badge/Tests-32%20passed-brightgreen?logo=pytest)
![Coverage](https://img.shields.io/badge/Coverage-Multi--Asset-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **DeepVaR Sentinel** is a modular deep-learning framework for financial volatility forecasting and Value-at-Risk (VaR) estimation, engineered with integrated data-integrity validation and adversarial defenses.

---

## 🤔 What is this?

Every bank and financial trading firm needs to answer one question every day:

> *"How much money could we lose tomorrow in a worst-case scenario?"*

That number is called **Value-at-Risk (VaR)**. For example, if a portfolio says **"Our 1-day 99% VaR is \$50,000"**, it means: *"We are 99% confident our portfolio will not lose more than \$50,000 tomorrow."*

**DeepVaR Sentinel provides an end-to-end pipeline that:**
1. Ingests historical market prices (equities, indices, commodities, and crypto).
2. Runs incoming feeds through an **input validation gate** to protect against corrupted or spoofed prices.
3. Uses an **LSTM neural network** to forecast tomorrow's conditional volatility ($\hat{\sigma}_{t+1}$).
4. Converts forecasted volatility into dynamic **Value-at-Risk thresholds** using both parametric formulas and volatility-filtered tail calibration.
5. Employs a **fail-safe volatility floor** so risk thresholds never artificially drop during market data-integrity incidents.
6. Evaluates forecasting quality against classical industry baselines using standard statistical tests.

---

## ❓ What problem does this solve?

### 1. Traditional models react too slowly to regime shifts
Standard classical methods (such as rolling historical volatility) often struggle during sudden transitions between quiet and turbulent market states. An LSTM neural network is designed to capture sequential temporal dependencies across historical lookback windows.

### 2. The MSE-to-Tail Gap
Standard machine learning models are trained with Mean Squared Error (MSE) to predict average volatility. However, risk management focuses strictly on the extreme 1% worst-case tails. To bridge this gap, the pipeline integrates volatility-filtered historical simulation (McNeil & Frey, 2000), calibrating empirical tail quantiles directly on an out-of-sample validation set.

### 3. Data Integrity & Feed Attacks
Financial automated systems often assume price feeds are trustworthy. If an upstream data feed is manipulated—such as through sudden price spikes or artificial volatility suppression—a naive model might report false calm right before a market downturn. DeepVaR Sentinel includes input validation rules and an automatic fail-safe hold mechanism to mitigate this risk.

---

## ⚙️ System Architecture

```
Market Data Feed
(OHLCV Time Series)
        │
        ▼
┌─────────────────────────────────┐
│   Input Validation Gate         │  • Checks for abnormal price moves (Hard Limit)
│   (Data Integrity Defense)      │  • Detects artificial calmness (Collapse Check)
└───────────────┬─────────────────┘
                │ clean / repaired
                ▼
┌─────────────────────────────────┐
│   Feature Engineering           │  • Continuously compounded log returns
│   (Rolling Window Builder)      │  • 21-day historical volatility
│                                 │  • Normalized intraday High-Low range
└───────────────┬─────────────────┘
                ▼
┌─────────────────────────────────┐
│   LSTM Forecaster               │  • 2-layer LSTM (64 → 32 units)
│   (PyTorch Neural Network)      │  • Predicts 1-day ahead volatility proxy |R_{t+1}|
└───────────────┬─────────────────┘
                ▼
┌─────────────────────────────────┐
│   Fail-Safe Volatility Floor    │  • Anchors forecast to trusted historical level
│   (Safety Mechanism)            │    during active data alerts
└───────────────┬─────────────────┘
                ▼
┌─────────────────────────────────┐
│   VaR Estimation & Calibration  │  • Parametric formula (Normal / Student's t)
│   (Risk Engine)                 │  • Volatility-filtered quantile calibration
└───────────────┬─────────────────┘
                ▼
┌─────────────────────────────────┐
│   Regulatory Backtesting        │  • Kupiec POF Likelihood Ratio Test
│   & Interactive Dashboard       │  • Christoffersen Independence Test
└─────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11 or newer
- [`uv`](https://github.com/astral-sh/uv) (fast Python package installer) or `pip`

### Setup (Windows)

```powershell
# 1. Clone repository
git clone <your-repo-url> deepvar-sentinel
cd deepvar-sentinel

# 2. Create and activate a clean virtual environment
uv venv .venv
.venv\Scripts\Activate.ps1

# 3. Install dependencies
uv pip install -r requirements.txt
uv pip install -e .

# 4. Verify installation via the test suite (uses synthetic data, no internet needed)
pytest -q
# Expected: 32 passed
```

### Setup (Linux / macOS)

```bash
git clone <your-repo-url> deepvar-sentinel && cd deepvar-sentinel
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt && uv pip install -e .
pytest -q
```

### Running the Pipeline

```powershell
# 1. Download price datasets from public endpoints
python scripts\download_all.py

# 2. Train the LSTM models
python scripts\train_all.py

# 3. Execute backtesting across all benchmark models
python scripts\backtest_all.py

# 4. Run external cross-checks (CBOE VIX)
python scripts\fred_crosscheck.py

# 5. Launch the interactive Streamlit dashboard
streamlit run app\streamlit_app.py
```

---

## 🤖 Models & Benchmarks Compared

The platform evaluates 5 distinct risk forecasting methodologies:

| Method | Category | Description |
|---|---|---|
| **LSTM (Calibrated)** | Deep Learning + EVT | 2-layer LSTM volatility forecaster with McNeil & Frey (2000) validation quantile calibration. |
| **LSTM (Raw)** | Deep Learning | Direct parametric VaR derived from LSTM volatility output assuming theoretical Student's $t$ tails. |
| **EWMA (RiskMetrics)** | Econometric Baseline | Exponentially Weighted Moving Average ($\lambda = 0.94$) weighting recent squared returns. |
| **Rolling 21-Day Std** | Empirical Baseline | Unweighted sample standard deviation over a trailing 21-trading-day window. |
| **Historical Simulation** | Non-Parametric | Direct empirical quantile derived from trailing 250 trading days of observed returns. |

---

## 📐 Mathematical Formulation

### 1. Returns & Feature Construction

Given daily closing prices $P_t$, high prices $H_t$, and low prices $L_t$:

* **Continuously Compounded Log Return:**
  $$R_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$

* **Rolling 21-Day Historical Volatility:**
  $$\sigma_{21, t} = \sqrt{\frac{1}{20} \sum_{i=0}^{20} \left(R_{t-i} - \bar{R}_t\right)^2}, \quad \text{where } \bar{R}_t = \frac{1}{21}\sum_{i=0}^{20} R_{t-i}$$

* **Intraday Normalized High-Low Range:**
  $$\text{Range}_t = \frac{H_t - L_t}{P_t}$$

* **Target Formulation ($H=1$ Day Ahead):**
  $$y_t = \left|R_{t+1}\right|$$
  Under conditionally zero-mean returns, the conditional expectation $\mathbb{E}[|R_{t+1}| \mid \mathcal{F}_t]$ serves as a proxy for conditional standard deviation $\sigma_{t+1}$.

---

### 2. Parametric Value-at-Risk (VaR) & Price Floor

For portfolio value $V$, confidence level $\alpha \in (0, 1)$ (e.g., $\alpha = 0.99$), zero drift assumption $\mu = 0$, and forecasted volatility $\hat{\sigma}_{t+1}$:

* **Parametric VaR Threshold (Return Space):**
  $$\text{VaR}_{\alpha, t+1}^{\text{ret}} = \mu - z_\alpha \hat{\sigma}_{t+1}$$

* **Parametric VaR Monetary Loss:**
  $$\text{VaR}_{\alpha, t+1}^{\$} = V \cdot \left(1 - \exp\left(\mu - z_\alpha \hat{\sigma}_{t+1}\right)\right)$$

* **Dynamic VaR Price Floor:**
  $$\text{Floor}_{t+1} = P_t \cdot \exp\left(\mu - z_\alpha \hat{\sigma}_{t+1}\right)$$

Where quantile multiplier $z_\alpha$ is defined as:
* **Normal Distribution ($\mathcal{N}(0, 1)$):** $z_\alpha = \Phi^{-1}(\alpha)$
* **Student's $t$-Distribution with $\nu$ Degrees of Freedom ($t_\nu$ rescaled to unit variance):**
  $$z_\alpha = t_\nu^{-1}(\alpha) \cdot \sqrt{\frac{\nu - 2}{\nu}}$$

---

### 3. Volatility-Filtered Tail Calibration (McNeil & Frey, 2000)

To adjust for empirical tail fatness:

1. **Standardized Residuals on Out-of-Sample Validation Set ($\mathcal{D}_{\text{val}}$):**
   $$e_t = \frac{R_t}{\hat{\sigma}_t}, \quad \forall t \in \mathcal{D}_{\text{val}}$$

2. **Empirical Left-Tail Quantile ($1 - \alpha$ level):**
   $$q_{1-\alpha} = \text{Quantile}_{1-\alpha}\left(\{e_t\}_{t \in \mathcal{D}_{\text{val}}}\right)$$

3. **Calibrated Threshold:**
   $$\text{Threshold}_{t+1}^{\text{calibrated}} = \hat{\sigma}_{t+1} \cdot q_{1-\alpha}$$

A **VaR breach** on trading day $t+1$ occurs if:
$$I_{t+1} = \mathbb{I}\left(R_{t+1} < \text{Threshold}_{t+1}^{\text{calibrated}}\right) = 1$$

---

### 4. Classical Benchmarks

* **EWMA (RiskMetrics, $\lambda = 0.94$):**
  $$\sigma_t^2 = \lambda \sigma_{t-1}^2 + (1 - \lambda) R_{t-1}^2$$

* **Rolling Standard Deviation ($W=21$):**
  $$\sigma_t = \sqrt{\frac{1}{W-1} \sum_{i=1}^W \left(R_{t-i} - \bar{R}\right)^2}$$

* **Historical Simulation:**
  $$\text{VaR}_{\alpha, t} = - \text{Quantile}_{1-\alpha}\left(\{R_{t-i}\}_{i=1}^{250}\right)$$

---

### 5. Statistical Backtesting

#### A. Kupiec Proportion of Failures (POF) Test
Tests whether empirical breach probability matches nominal coverage $p_0 = 1 - \alpha$:
$$H_0: p = p_0 \quad \text{vs.} \quad H_1: p \neq p_0$$

Given $N$ test days, observed breaches $x = \sum_{t=1}^N I_t$, and sample rate $\hat{p} = \frac{x}{N}$:
$$\text{LR}_{\text{POF}} = -2 \ln \left[ \frac{(1 - p_0)^{N - x} \, p_0^x}{(1 - \hat{p})^{N - x} \, \hat{p}^x} \right] \sim \chi^2(1)$$

If $p\text{-value} = 1 - F_{\chi^2(1)}(\text{LR}_{\text{POF}}) < 0.05$, the model is rejected as mis-calibrated.

#### B. Christoffersen Independence Test
Evaluates whether breaches occur independently without clustering using a two-state Markov chain transition matrix $\Pi = \begin{pmatrix} \pi_{00} & \pi_{01} \\ \pi_{10} & \pi_{11} \end{pmatrix}$:
$$\text{LR}_{\text{ind}} = -2 \ln \left[ \frac{\pi_0^{n_{00} + n_{10}} (1 - \pi_0)^{n_{01} + n_{11}}}{\hat{\pi}_{01}^{n_{01}} (1 - \hat{\pi}_{01})^{n_{00}} \hat{\pi}_{11}^{n_{11}} (1 - \hat{\pi}_{11})^{n_{10}}} \right] \sim \chi^2(1)$$

---

### 6. Security Defenses & Fail-Safe Mechanism

* **Rule 1 — Hard Limit Check (Large Price Dislocations):**
  $$\text{Flag}_{\text{hard\_limit}, t} = \mathbb{I}\left(|R_t| > \gamma_{\text{max}}\right)$$
  When triggered, the system replaces $P_t \leftarrow P_{t-1}$ (zero-order hold repair).

* **Rule 2 — Volatility Collapse Check (Stealth Calm-Injection):**
  $$\text{RMS}_{k, t} = \sqrt{\frac{1}{k}\sum_{i=0}^{k-1} R_{t-i}^2}$$
  $$\text{Flag}_{\text{collapse}, t} = \mathbb{I}\left(\text{RMS}_{k, t} < \beta \cdot \text{Median}\left(\{\sigma_{21, \tau}\}_{\tau = t - k - 252}^{t - k}\right)\right)$$

* **Fail-Safe Volatility Floor:**
  If an alert is active on day $t$ or within the preceding $H_{\text{hold}}$ days:
  $$\hat{\sigma}_t^{\text{secure}} = \max\left(\hat{\sigma}_t, \;\; \max_{i \in [\text{lag}, \, \text{lag} + W_{\text{ref}}]} \hat{\sigma}_{t-i}\right)$$

---

## 🔐 Threat Modeling & Attack Simulation

The framework evaluates system robustness under three synthetic data manipulation scenarios:

| Attack | Operation | Purpose |
|---|---|---|
| **Flash Crash** | Abruptly drops prices by 20–40% over 1–5 trading days. | Tests zero-order hold repair and breach signal handling. |
| **Fat Finger** | Injects an isolated +50% price spike on day $t$. | Tests single-day spike detection. |
| **Volatility Suppression** | Multiplies returns by a factor $\alpha \in (0, 1)$ during volatile periods. | Simulates an adversary masking turbulence to artificially lower capital buffers. |

---

## 📁 Repository Layout

```
deepvar-sentinel/
│
├── README.md                    ← Project documentation
├── pyproject.toml               ← Build setup and testing configurations
├── requirements.txt             ← Runtime package dependencies
├── Makefile                     ← CLI build/test shortcuts
│
├── configs/                     ← Asset-specific YAML configuration files
│   ├── default.yaml             ← BTC-USD configuration
│   ├── sp500.yaml               ← S&P 500 configuration
│   └── ...                      ← Additional configurations
│
├── src/deepvar/                 ← Core Python package
│   ├── config.py                ← Configuration loader
│   ├── pipeline.py              ← High-level orchestration pipeline
│   ├── data/                    ← Data ingestion, feature engineering, windowing
│   ├── models/                  ← LSTM PyTorch model & training routine
│   ├── risk/                    ← VaR estimation, calibration, and backtesting
│   └── security/                ← Attack simulations, validation gate, fail-safe
│
├── scripts/                     ← Executable CLI scripts
│   ├── download_all.py          ← Multi-asset data downloader
│   ├── train_all.py             ← Model training automation
│   ├── backtest_all.py          ← Full backtest suite
│   ├── fred_crosscheck.py       ← CBOE VIX cross-check utility
│   └── attack_study.py          ← Adversarial defense demonstration
│
├── app/
│   └── streamlit_app.py         ← Interactive dashboard
│
├── tests/                       ← Pytest unit and regression tests (32 tests)
│
├── artifacts/                   ← Model weights and scalers (local / git-ignored)
├── reports/                     ← Generated summary CSV reports (local / git-ignored)
└── data/                        ← Cached raw market data (local / git-ignored)
```

---

## ⚙️ Configuration Schema

Parameters are customized per asset in `configs/*.yaml`:

```yaml
data:
  ticker: "BTC-USD"
  start: "2015-01-01"
  train_end: "2021-12-31"      # Strict chronological split
  val_end:   "2023-12-31"
  vol_window: 21
  target_horizon: 1

model:
  window: 60                   # Lookback window length (days)
  hidden1: 64
  hidden2: 32
  dropout: 0.3
  lr: 0.001
  epochs: 60
  patience: 8

risk:
  confidence: 0.99             # 99% 1-day VaR
  distribution: "t"            # "t" or "normal"
  t_df: 5.0
  ewma_lambda: 0.94
  hist_window: 250

security:
  max_abs_return: 0.40
  collapse_window: 5
  collapse_ratio: 0.35
  hold_days: 21
```

---

## 🧪 Testing

The test suite runs 32 automated unit tests across data transformations, feature engineering, backtesting logic, and security rules:

```powershell
pytest -q
```

All tests operate using synthetic offline data without requiring network connections or GPUs.

---

## 📚 References

* McNeil, A. J., & Frey, R. (2000). *Estimation of tail-related risk measures for heteroscedastic financial time series: an extreme value approach.* Journal of Empirical Finance, 7(3–4), 271–300.
* Kupiec, P. H. (1995). *Techniques for verifying the accuracy of risk measurement models.* Journal of Derivatives, 3(2), 73–84.
* Christoffersen, P. F. (1998). *Evaluating interval forecasts.* International Economic Review, 39(4), 841–862.
* J.P. Morgan / Reuters (1996). *RiskMetrics — Technical Document.* 4th Edition.

---

## 📄 License

This project is licensed under the MIT License — see `LICENSE` for details.
*Educational and research framework. Not financial advice.*

