# STOCKUP-Deep-Learning-VaR-Volatility-Forecasting-System

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)
![Tests](https://img.shields.io/badge/Tests-32%20passed-brightgreen?logo=pytest)
![Assets](https://img.shields.io/badge/Assets-20%20real--world-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **M.Tech Research Project** — Deep learning for financial risk estimation, tested on 20 real-world assets, with built-in protection against data-feed attacks.

---

## What is this?

Every bank and trading firm needs to answer one question every single day:

> *"How much money could we lose tomorrow — in the worst case?"*

That number is called **Value-at-Risk (VaR)**. For example, if a bank says **"Our 1-day 99% VaR is \$5 million"**, it means: *"We are 99% confident we will not lose more than \$5 million tomorrow."*

**This project builds an AI system that:**
1. Looks at the last 60 days of market prices for a stock, crypto, or index
2. Uses an **LSTM neural network** to predict how volatile (risky) tomorrow will be
3. Converts that volatility estimate into a **VaR number** (how much money is at risk)
4. Checks whether the prediction was actually correct — across **20 different assets**
5. Protects the system from **data-feed attacks** (hackers feeding fake calm data to hide a real crash)

---

## What problem does this solve?

### Problem 1 — Classic VaR models are too simple

Old methods (like the ones banks used before 2008) assume markets move in predictable patterns. They missed the 2008 financial crisis because they were not designed to learn from complex patterns in data.

**Our solution:** An LSTM (a type of AI that remembers patterns over time) learns what "calm" and "stormy" markets look like, and adjusts the risk estimate accordingly.

### Problem 2 — Even good AI models have a calibration gap

An AI trained to predict average volatility does not automatically get the *extreme tails* right. The 1% worst days are what VaR is about — and those are exactly the hardest to predict.

**Our solution:** We use a technique from McNeil & Frey (2000) — after training, we measure how wrong the model is on a *separate* validation set and correct the threshold automatically. This is called **calibration**.

### Problem 3 — Nobody thinks about data security in finance AI

What if someone hacks your data feed and replaces real volatile prices with fake calm prices? Your AI would think everything is fine while the market is actually crashing.

**Our solution:** A simple but effective **security gate** checks every incoming data point for suspicious behaviour before it reaches the model.

---

## Key Results (Test Period: 2024 onwards)

The main test is called **Kupiec POF** — it checks whether the VaR prediction was actually correct the right number of times. For a 99% VaR, you expect exactly 1% of days to be worse than predicted (called "breaches"). Too many breaches = model is too optimistic. Too few = model is too conservative.

| Model | Assets that PASS | Out of | Pass Rate |
|-------|-----------------|--------|-----------|
| **LSTM (Calibrated)**  | **14** | 20 | **70%** |
| Historical Simulation | 14 | 20 | 70% |
| EWMA (RiskMetrics) | 13 | 20 | 65% |
| Rolling 21-day Std | 10 | 20 | 50% |
| LSTM (Raw — no calibration) | 2 | 20 | 10% |

**The key finding:** Raw LSTM only passes 2/20 without calibration. After applying McNeil & Frey calibration, it jumps to 14/20 — matching the best classical method without retraining the model.

**Cross-check with VIX (the official market fear index):** Our LSTM's volatility predictions correlate with the CBOE VIX at **r = +0.765** (Pearson), confirming the model genuinely tracks real-world market fear.

---

## How it works — Step by step

Think of it like a weather forecasting pipeline, but for financial risk.

```
STEP 1 — Collect data
   "Get the last few years of daily prices for the asset"
   Source: Yahoo Finance (free, automatic download)

        ↓

STEP 2 — Security gate (the digital bouncer)
   "Check if today's price looks suspicious before letting it in"
   Rule 1: If price moved more than 40% in one day → probably a data error, fix it
   Rule 2: If market suddenly became suspiciously calm → possible attack, raise alert

        ↓

STEP 3 — Feature engineering
   "Turn raw prices into signals the AI can learn from"
   - Daily log return   : How much did the price move today? (in %)
   - 21-day volatility  : How wild has the market been this month?
   - High-Low range     : How big was today's price swing?

        ↓

STEP 4 — Sliding window
   "Package the last 60 days of signals into one input block for the AI"
   Shape: (60 days) × (3 features) = one block

        ↓

STEP 5 — LSTM model
   "The AI reads the 60-day block and predicts tomorrow's volatility"
   Architecture: 2-layer LSTM (64 units → 32 units → 1 output)
   Training: Looks at 2015–2021 data. Stops early if it starts overfitting.

        ↓

STEP 6 — Calibration (McNeil & Frey 2000)
   "Correct the model's tail estimate using real observed errors"
   Uses 2022–2023 (validation set) to find the true 1% worst-case multiplier
   This one step takes the pass rate from 10% → 70%

        ↓

STEP 7 — VaR number
   VaR = Portfolio Value × (1 - e^(μ - cal_q × σ̂))
   Example: "With $1,000,000 invested, your max 1-day 99% loss is $23,400"

        ↓

STEP 8 — Backtest
   "Check if the prediction was correct on 2024 data the model never saw"
   Kupiec POF test: were there exactly 1% breach days?
```

---

## Quick Start

### What you need first
- Python 3.11 or newer — [download here](https://www.python.org/downloads/)
- `uv` package manager — [install here](https://github.com/astral-sh/uv) (much faster than pip)
- Internet connection (to download price data from Yahoo Finance)

### Setup (Windows)

```powershell
# 1. Download the project
git clone <your-repo-url> deepvar-sentinel
cd deepvar-sentinel

# 2. Create a virtual environment (a clean, isolated Python space)
uv venv .venv
.venv\Scripts\Activate.ps1

# 3. Install all required packages
uv pip install -r requirements.txt
uv pip install -e .

# 4. Run the tests first — makes sure everything is working (no internet needed)
pytest -q
# Expected output: 32 passed in ~9s
```

### Setup (Mac / Linux)
```bash
git clone <your-repo-url> deepvar-sentinel && cd deepvar-sentinel
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt && uv pip install -e .
pytest -q
```

### Run the full pipeline

```powershell
# Step 1 — Download all 20 assets from Yahoo Finance (~2 minutes, needs internet)
python scripts\download_all.py

# Step 2 — Train all 20 LSTM models (~30-45 minutes on a laptop CPU)
python scripts\train_all.py

# Step 3 — Run backtests: 5 models × 20 assets → saves results to reports/
python scripts\backtest_all.py

# Step 4 — Cross-check LSTM vs VIX (needs FRED API, free)
python scripts\fred_crosscheck.py

# Step 5 — Test security gate on NAB benchmark data
python scripts\nab_gate_eval.py

# Step 6 — Launch the interactive web dashboard
streamlit run app\streamlit_app.py
# Opens automatically at: http://localhost:8501
```

>  **No GPU needed.** The model is small and trains in a few minutes on any laptop CPU.

---

## The 5 Models — What they are and how they compare

We test 5 different approaches on every asset and compare them fairly.

### 1. LSTM (Raw)
- **What it is:** A neural network that reads 60 days of market data and predicts tomorrow's volatility
- **How it predicts VaR:** Uses a statistical formula with a t-distribution (a bell curve with fat tails)
- **Problem:** The formula assumes tails look a certain way — they don't in real markets
- **Result:** Only 2/20 assets pass. Too many breach days (model under-estimates risk)

### 2. LSTM (Calibrated)  Our main contribution
- **What it is:** Same LSTM, but the VaR threshold is corrected after training
- **How the correction works:**
  1. Run the model on 2022–2023 (validation data it hasn't seen)
  2. Compute the actual worst-case return for each day: `residual = actual_return / predicted_sigma`
  3. Find the true 1% quantile of those residuals
  4. Use that real number instead of the formula's assumed number
- **Result:** 14/20 pass. Huge improvement with zero extra training.

### 3. EWMA (RiskMetrics)
- **What it is:** An old 1990s formula from J.P. Morgan. Recent days count more than old ones (exponential weighting)
- **Formula:** `σ²_today = 0.94 × σ²_yesterday + 0.06 × return²_today`
- **Strength:** Very fast, no training needed, works well on quiet markets
- **Weakness:** Slow to react to sudden volatility spikes
- **Result:** 13/20 pass

### 4. Rolling 21-day Standard Deviation
- **What it is:** The simplest possible model. Just look at how much prices moved in the last 21 trading days
- **Strength:** Completely transparent, zero assumptions
- **Weakness:** Treats all 21 days equally (a crash 20 days ago counts the same as yesterday)
- **Result:** 10/20 pass

### 5. Historical Simulation
- **What it is:** Look at the last 250 trading days. Take the worst 1% of actual returns. Use that as your VaR.
- **Strength:** No model assumptions at all — uses real observed data
- **Weakness:** Cannot predict a crisis larger than anything in the last 250 days
- **Result:** 14/20 pass (best classical method)

---

## The Security Module

This is what makes this project different from a standard VaR project.

### Why does security matter?

Imagine you manage risk for a hedge fund. Your VaR system reads live price data from a data vendor every morning. What if a hacker replaced that live feed with fake prices? Your model would think the market is calm, and you'd hold less money as a safety buffer — right before a crash.

This is not just theoretical. Supply-chain attacks on financial data are a real and growing concern.

### The 3 attack types we simulate

| Attack | What it does | Real-world equivalent |
|--------|-------------|----------------------|
| **Flash crash** | Drops price by 20–40% for 1–5 days | A data vendor's system gets corrupted |
| **Fat finger** | Spikes price up 50% for a day | A rogue trader enters a wrong order in the feed |
| **Volatility suppression** | Silently scales all returns by 20% | A sophisticated attacker replays historical calm data |

The last one (volatility suppression) is the most dangerous — it's the stealthiest. Prices look plausible. Only the volatility is wrong.

### The 2-rule Security Gate

We use just 2 simple, fully explainable rules (no ML needed here — explainability matters for regulators):

**Rule 1 — Hard Limit:**
> "If the price moved more than 40% in one day, that's physically impossible for most assets. Reject it and use yesterday's price instead."
> → Catches flash crash and fat finger attacks

**Rule 2 — Collapse Detection:**
> "If the market has been suspiciously calm for 5 days in a row (much calmer than its own historical average), something is wrong."
> → Catches volatility suppression attacks

### The Fail-Safe Floor

When the gate fires an alert, the system activates a fail-safe:
> "While we suspect the data is compromised, never let our volatility estimate drop. Keep it at the last trustworthy level for at least 21 trading days."

This means even a perfect attacker who fools the model cannot reduce the VaR estimate during an attack window.

---

##  The 20 Assets

We test on 6 different types of financial assets to prove the system is not just tuned for one market.

| # | Ticker | Asset | Type |
|---|--------|-------|------|
| 1 | BTC-USD | Bitcoin | Crypto |
| 2 | ETH-USD | Ethereum | Crypto |
| 3 | XRP-USD | Ripple | Crypto |
| 4 | ^GSPC | S&P 500 | US Stock Index |
| 5 | ^DJI | Dow Jones | US Stock Index |
| 6 | ^IXIC | NASDAQ | US Stock Index |
| 7 | AAPL | Apple | US Tech Stock |
| 8 | MSFT | Microsoft | US Tech Stock |
| 9 | TSLA | Tesla | US Tech Stock |
| 10 | ^NSEI | Nifty 50 | India Index |
| 11 | HDFCBANK.NS | HDFC Bank | Indian Bank |
| 12 | ICICIBANK.NS | ICICI Bank | Indian Bank |
| 13 | INFY.NS | Infosys | Indian IT Stock |
| 14 | RELIANCE.NS | Reliance | Indian Conglomerate |
| 15 | TCS.NS | Tata Consultancy | Indian IT Stock |
| 16 | ^HSI | Hang Seng | Hong Kong Index |
| 17 | ^N225 | Nikkei 225 | Japan Index |
| 18 | ^FTSE | FTSE 100 | UK Index |
| 19 | GC=F | Gold Futures | Commodity |
| 20 | CL=F | Crude Oil (WTI) | Commodity |

### How we split the data — strictly no peeking

This is critical. If the model has ever seen the test data, results are meaningless.

```
Training     : Jan 2015 → Dec 2021   ← Model learns from this
Validation   : Jan 2022 → Dec 2023   ← Used for early stopping + VaR calibration
Test         : Jan 2024 → today      ← Never touched until final evaluation
```

---

##  Test Suite

We have **32 automated tests** that run on completely synthetic (fake) data — no internet connection needed.

```powershell
pytest -q         # run all tests (~9 seconds)
pytest -v         # verbose — shows each test name
pytest tests/test_var.py -v   # run just one test file
```

| Test file | What it checks |
|-----------|---------------|
| `test_attacks.py` | The 3 attack functions work correctly and only touch the right days |
| `test_defenses.py` | The gate catches spikes and suppression; the fail-safe floor never lowers sigma |
| `test_features.py` | Log returns are computed correctly; target is 1-day ahead, not leaking future data |
| `test_var.py` | VaR maths is correct; calibration gives a negative quantile; more sigma = more VaR |
| `test_kupiec.py` | Perfect calibration passes; 10% breaches at 99% VaR fails |
| `test_pipeline.py` | LSTM trains, saves, and loads back correctly with the right output shape |
| `test_splits.py` | Train/val/test splits are chronological with zero overlap |
| `test_resample.py` | Weekend and holiday gaps are handled correctly |

---

## Project Structure

```
deepvar-sentinel/
│
├── README.md                    ← You are here
├── pyproject.toml               ← Project settings and test config
├── requirements.txt             ← All Python packages needed
├── Makefile                     ← Shortcuts: make train, make test, make app
│
├── configs/                     ← One settings file per asset (20 total)
│   ├── default.yaml             ← Bitcoin (BTC-USD)
│   ├── sp500.yaml               ← S&P 500
│   └── ...18 more...
│
├── src/deepvar/                 ← The main code library
│   ├── config.py                ← Loads the YAML settings files
│   ├── pipeline.py              ← Connects everything end-to-end
│   ├── data/
│   │   ├── download.py          ← Downloads prices from Yahoo Finance
│   │   ├── features.py          ← Computes log returns, volatility, range
│   │   ├── windows.py           ← Packages data into 60-day sliding blocks
│   │   └── synthetic.py        ← Generates fake data for tests (no internet needed)
│   ├── models/
│   │   ├── lstm.py              ← The LSTM neural network architecture
│   │   └── train.py             ← Training loop with early stopping
│   ├── risk/
│   │   ├── var.py               ← VaR formulas + calibration (McNeil & Frey)
│   │   ├── backtest.py          ← Kupiec POF and Christoffersen independence test
│   │   └── baselines.py        ← EWMA, rolling std, historical simulation
│   └── security/
│       ├── attacks.py           ← Flash crash, fat finger, vol suppression
│       ├── defenses.py          ← 2-rule input gate + fail-safe floor
│       └── evaluate.py          ← Finds the best window to test attacks on
│
├── scripts/                     ← Run these from the command line
│   ├── download_all.py          ← Downloads all 20 assets
│   ├── train_all.py             ← Trains all 20 models
│   ├── backtest_all.py          ← Tests all 5 models × 20 assets
│   ├── fred_crosscheck.py       ← Compares LSTM σ with the VIX index
│   ├── nab_gate_eval.py         ← Tests gate on NAB benchmark datasets
│   └── attack_study.py          ← Demo: shows what happens under each attack
│
├── app/
│   └── streamlit_app.py         ← Interactive dashboard in your browser
│
├── tests/                       ← Automated tests (32 total, all pass)
│
├── artifacts/                   ← Trained models saved here (not on GitHub)
├── reports/                     ← Backtest results saved here (not on GitHub)
├── data/raw/                    ← Downloaded prices saved here (not on GitHub)
│
└── docs/                        ← Plain-English explanations of every concept
    ├── 01-project-explained.md  ← Start here if you're new to the project
    ├── 05-finance-concepts.md   ← What is VaR, Kupiec, EWMA? Explained simply.
    ├── 06-cybersecurity-concepts.md ← What is a data-feed attack?
    └── 09-thesis-and-viva-guide.md  ← Questions your examiner might ask + answers
```

---

## Configuration

Every setting is in a simple text file (YAML format). Here is what the main settings do:

```yaml
data:
  ticker: "BTC-USD"         # Which asset to download from Yahoo Finance
  start: "2015-01-01"       # How far back to go
  train_end: "2021-12-31"   # Training data ends here
  val_end:   "2023-12-31"   # Validation data ends here (test starts Jan 2024)

model:
  window: 60                # How many past days the LSTM looks at
  hidden1: 64               # Size of the first LSTM layer
  hidden2: 32               # Size of the second LSTM layer
  dropout: 0.3              # Randomly turn off 30% of neurons during training
                            #   to prevent the model from memorising the data
  lr: 0.001                 # How fast the model learns (learning rate)
  epochs: 60                # Maximum number of training rounds
  patience: 8               # Stop early if no improvement for 8 rounds

risk:
  confidence: 0.99          # We want 99% VaR (1-in-100 worst day)
  distribution: "t"         # Use a fat-tailed distribution (better than normal)
  t_df: 5.0                 # Degrees of freedom for the t-distribution
  ewma_lambda: 0.94         # RiskMetrics decay factor
  hist_window: 250          # Historical simulation: use last 250 trading days

security:
  max_abs_return: 0.40      # Flag any day where price moved more than 40%
  collapse_window: 5        # Check if last 5 days are suspiciously calm
  collapse_ratio: 0.35      # "Suspiciously calm" = below 35% of normal vol
  hold_days: 21             # Keep the fail-safe floor active for 21 days
```

---

## Honest Limitations

I want to be transparent about where the system does not work perfectly.

**Where it struggles:**

| Issue | Why |
|-------|-----|
| **Gold (GC=F) fails all models** | Gold surged from \$2,000 → \$2,700 in 2024. Nothing in 2015–2023 training data predicted this. No model can predict something it has never seen. |
| **NASDAQ, S&P 500 fail calibration** | The 2022–2023 Fed rate hikes fundamentally changed how tech stocks move. The validation data the model calibrates on is itself a regime-change period. |
| **Raw LSTM under-estimates tail risk** | The LSTM is trained to minimise average error. The worst 1% of days are outliers — the model naturally pays less attention to them during training. Calibration fixes this, but it cannot help if the new regime is entirely unlike history. |
| **Security gate has false alarms** | The gate is intentionally designed to prefer false alarms over missed attacks. In a real trading system, this means you hold slightly more capital on some normal days. This is by design — it is the safer choice. |
| **NAB F1 scores are low** | The NAB benchmark contains machine sensor anomalies, server metrics, and traffic data. Our gate is specifically designed for financial price data. Low cross-domain scores are expected and not a weakness of the gate on its intended use case. |

---

## Background Reading (for examiners and reviewers)

The two key academic papers behind this project:

**McNeil & Frey (2000)** — the calibration method we use:
> McNeil, A. J., & Frey, R. (2000). *Estimation of tail-related risk measures for heteroscedastic financial time series: an extreme value approach.* Journal of Empirical Finance, 7(3–4), 271–300.

**Kupiec (1995)** — the statistical test we use to evaluate VaR:
> Kupiec, P. H. (1995). *Techniques for verifying the accuracy of risk measurement models.* Journal of Derivatives, 3(2), 73–84.

---

## License

MIT License — free to use, modify, and share with attribution.

---

*Built as an M.Tech research project. All results are from real market data. Not financial advice.*
