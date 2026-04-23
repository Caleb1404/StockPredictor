# 📈 StockPredictor

A machine-learning stock prediction dashboard built with Python and Streamlit. Uses XGBoost trained on technical indicators, macroeconomic data, and live news sentiment to predict next-day price direction for US equities and ETFs.

---

## Features

- **XGBoost classifier** trained on 20+ features across four categories
- **Walk-forward backtesting** — model is trained on historical data and tested strictly out-of-sample (no data leakage)
- **Live data** — always pulls the most recent available prices; test period updates daily
- **News sentiment** via yfinance headlines + Finnhub API, scored with VADER
- **Macro environment** dashboard — VIX, 10Y Treasury yield, S&P 500 regime, USD strength
- **Search any ticker** — not limited to the preset list

### Feature groups

| Category | Features |
|---|---|
| Price & momentum | Daily return, 5-day return, 20-day return |
| Volume | Volume change, volume vs 20-day average, OBV 5-day change |
| Trend | Distance from MA20/50/200, Golden Cross flag, MACD, ADX +DI/-DI |
| Oscillators | RSI (14), Stochastic %K/%D |
| Volatility | Bollinger Band width, BB %position, ATR (normalised) |
| Macro | VIX + change + fear regime, 10Y yield + change, S&P 500 return + trend, DXY change |
| Sentiment | Monthly news sentiment score (yfinance + Finnhub + VADER) |

---

## Getting started

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd StockPredictor
```

### 2. Install dependencies

```bash
pip install streamlit yfinance pandas numpy requests xgboost scikit-learn \
            vaderSentiment ta matplotlib
```

### 3. Run the dashboard

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

> **Note:** Run with `streamlit run app.py`, not `python app.py`.

---

## Usage

### Dashboard tabs

| Tab | What you see |
|---|---|
| **Backtest & Strategy** | Cumulative return of model strategy vs buy & hold, Sharpe ratio, max drawdown, win rate, recent signal history |
| **Technical Analysis** | Price + moving averages, RSI, Stochastic, MACD, ADX, Volume / OBV |
| **Macro Environment** | VIX fear index, 10Y Treasury yield, S&P 500 regime, USD strength |
| **Model Insights** | Feature importance chart, news sentiment timeline, model configuration |

### Signal banner

The top of the page shows the latest signal (**BUY** or **SELL**) with confidence score, latest close price, and — when sentiment is enabled — the current month's news sentiment and its rank in feature importance.

### News sentiment toggle

Enable it in the sidebar. On first load it takes ~5–10 seconds (yfinance live news is instant; Finnhub fills in historical months at ~60 req/min). Results are cached for 1 hour.

### Custom tickers

Type any valid ticker symbol into the **"Or search any ticker"** field — e.g. `AMZN`, `TSM`, `BTC-USD`, `COIN`. The preset list is just a shortcut.

---

## Batch runner

`stock_predictor.py` runs predictions for all tickers in sequence and saves results to `strategy_summary_v2.csv`.

```bash
python stock_predictor.py
```

Output columns: `Ticker`, `OOS_Accuracy`, `Strategy_Return`, `Market_Return`, `Sharpe`, `Max_Drawdown`, `Win_Rate`, `Top_Feature`, `N_Features`, `Train_Samples`, `Test_Samples`.

---

## Data sources

| Source | Used for | Cost |
|---|---|---|
| [Yahoo Finance](https://finance.yahoo.com) via `yfinance` | Price/OHLCV data, macro ETFs, live news | Free |
| [Finnhub](https://finnhub.io) | Historical monthly news headlines | Free tier (60 req/min) |
| Derived | All technical indicators computed locally via `ta` library | — |

---

## Project structure

```
StockPredictor/
├── app.py                          # Streamlit dashboard (main entry point)
├── stock_predictor.py              # Batch runner for all tickers
├── strategy_summary.csv            # Example output (v1)
├── strategy_summary_v2.csv         # Output from batch runner (v2)
└── README.md
```

---

## Configuration

Dates are dynamic and computed relative to today at startup:

| Variable | Default | Description |
|---|---|---|
| `END_DATE` | Today | Last date of data to fetch |
| `TEST_START` | 12 months ago | Start of out-of-sample test period |
| `TRAIN_START` | 3 years ago | Start of model training period |
| `DATA_START` | 4 years ago | Start of raw data download (extra year for 200-day MA warmup) |

To change them, edit the constants near the top of `app.py`.

---

## Limitations & notes

- **Prediction target is binary** — the model only predicts direction (up/down), not magnitude
- **Past performance does not guarantee future results** — backtested strategy returns look better than live trading will be due to transaction costs and slippage not being modelled
- **News sentiment is monthly** — a coarser signal than daily; most useful as a trend indicator
- **Yahoo Finance rate limiting** — if downloads fail on startup, wait 30–60 seconds and reload

---

## Dependencies

```
streamlit
yfinance
pandas
numpy
requests
xgboost
scikit-learn
vaderSentiment
ta
matplotlib
```
