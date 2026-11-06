.PHONY: setup data train backtest attack app test lint demo clean
PY ?= python

setup:      ## create venv + install
	$(PY) -m venv .venv && . .venv/bin/activate && pip install -r requirements-dev.txt && pip install -e .
data:       ## download + cache market data
	$(PY) scripts/download_data.py
train:      ## train the LSTM
	$(PY) scripts/train.py
backtest:   ## Kupiec / baselines table
	$(PY) scripts/backtest.py
attack:     ## adversarial study
	$(PY) scripts/attack_study.py
app:        ## launch dashboard
	streamlit run app/streamlit_app.py
test:
	pytest -q
lint:
	ruff check .
demo:       ## whole pipeline offline on synthetic data (no internet needed)
	$(PY) scripts/train.py --source synthetic --epochs 15
	$(PY) scripts/backtest.py --source synthetic
	$(PY) scripts/attack_study.py --source synthetic
clean:
	rm -rf artifacts/* reports/*.csv reports/figures/* .pytest_cache
