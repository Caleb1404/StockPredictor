"""
Stock Prediction Dashboard — Enhanced Edition v2.0

Improvements over v1:
  - Walk-forward backtesting: predictions ONLY on held-out test set (no data leakage)
  - Macro features: VIX, 10Y Treasury yield, S&P 500 regime, USD strength (UUP)
  - Extended technical: ADX, Stochastic, OBV, ATR, MA distances (20/50/200),
    Golden Cross flag, multi-period returns, volume ratio
  - Multi-tab Streamlit UI: Backtest | Technical | Macro | Model
  - Risk metrics: Sharpe ratio, max drawdown, win rate
  - Feature importance visualization
  - No training-set predictions used in strategy simulation
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from time import sleep
from datetime import date, timedelta
import matplotlib.pyplot as plt

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "NVDA", "AMD", "PLTR",
    "TSLA", "META", "SPY", "QQQ", "ARKK", "GLD",
]

# Dynamic dates — always use the most recent data available
_today      = date.today()
END_DATE    = _today.strftime("%Y-%m-%d")
TEST_START  = (_today - timedelta(days=365)).strftime("%Y-%m-%d")    # last 12 months = test
TRAIN_START = (_today - timedelta(days=365 * 3)).strftime("%Y-%m-%d") # 2 years of training
DATA_START  = (_today - timedelta(days=365 * 4)).strftime("%Y-%m-%d") # +1 year MA warmup

COMPANY_MAP = {
    "AAPL": "Apple",         "MSFT": "Microsoft",    "GOOGL": "Google",
    "NVDA": "Nvidia",        "AMD":  "AMD",           "PLTR": "Palantir",
    "TSLA": "Tesla",         "META": "Meta",          "SPY":  "S&P 500",
    "QQQ":  "Nasdaq 100",    "ARKK": "ARK Innovation ETF",   "GLD": "Gold ETF",
}

# Macro series downloaded via yfinance
MACRO_SYMBOLS = {
    "^VIX":  "VIX",       # CBOE fear / volatility index
    "^TNX":  "Yield10Y",  # 10-year US Treasury yield
    "^GSPC": "SPX",       # S&P 500 (market regime proxy)
    "UUP":   "DXY",       # Invesco USD Bull ETF (dollar-strength proxy)
}

# All candidate feature names; any that don't exist in a given run are dropped
ALL_FEATURES = [
    # Price & momentum
    "Return", "Return_5d", "Return_20d",
    # Volume
    "VolumeChange", "Vol_ratio", "OBV_change",
    # Trend / moving averages
    "MA20_dist", "MA50_dist", "MA200_dist", "Golden_cross",
    # MACD
    "MACD", "MACD_diff",
    # ADX
    "ADX", "ADX_pos", "ADX_neg",
    # Oscillators
    "RSI", "RSI_oversold", "RSI_overbought",
    "Stoch_k", "Stoch_d",
    # Volatility
    "BB_width", "BB_pct", "ATR",
    # Macro
    "VIX", "VIX_change", "VIX_regime",
    "Yield10Y", "Yield_change",
    "SPX_return", "SPX_trend",
    "DXY_change",
    # Sentiment
    "Sentiment",
]

# ─────────────────────────────────────────────────────────────────────────────
# PAGE LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(layout="wide", page_title="Stock Predictor Pro")
st.title("📈 Stock Prediction Dashboard — Enhanced Edition")
st.caption(
    "XGBoost · RSI · MACD · ADX · Stochastic · OBV · ATR · Bollinger Bands · "
    "VIX · 10Y Yields · Dollar Index · News Sentiment · Walk-Forward Backtesting"
)

# ── Ticker selector: quick-pick list OR free-text search ─────────────────────
_col_sel, _col_custom = st.columns([2, 1])
with _col_sel:
    _preset = st.selectbox("Quick-pick", ["(type a custom ticker →)"] + TICKERS, index=1)
with _col_custom:
    _custom = st.text_input("Or search any ticker", placeholder="e.g. AMZN, COIN, TSM…").upper().strip()

# Custom input wins if filled; otherwise use the preset
ticker = _custom if _custom else (_preset if _preset != "(type a custom ticker →)" else "AAPL")

# ─────────────────────────────────────────────────────────────────────────────
# DATA HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _clean_yf(df: pd.DataFrame, sym: str = "") -> pd.DataFrame:
    """
    Normalise columns from both yf.download() (MultiIndex) and
    Ticker.history() (simple columns like 'Open', 'High', ..., 'Dividends').
    Always returns a DataFrame with columns: Open, High, Low, Close, Volume.
    """
    # Flatten MultiIndex (yf.download artefact)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(str(p) for p in parts if p).strip()
            for parts in df.columns.values
        ]
    # Strip ticker suffix
    if sym:
        sfx = f"_{sym}"
        df.columns = [c[: -len(sfx)] if c.endswith(sfx) else c for c in df.columns]
    # Title-case all columns
    df.columns = [c.strip().title() for c in df.columns]
    # Aliases for adj-close variants
    df.rename(
        columns={
            "Adj Close":  "Close",
            "Adj_Close":  "Close",
            "Stock Splits": "Stock_Splits",
        },
        inplace=True,
        errors="ignore",
    )
    # Drop Ticker.history() extra columns we don't need
    for extra in ("Dividends", "Stock_Splits", "Capital Gains"):
        if extra in df.columns:
            df.drop(columns=[extra], inplace=True, errors="ignore")
    return df


def _yf_download_with_retry(sym: str, start: str, end: str, retries: int = 3) -> pd.DataFrame:
    """
    Download via yf.Ticker.history() — less rate-limited than yf.download().
    Falls back to yf.download() on failure, with exponential back-off.
    """
    for attempt in range(retries):
        try:
            # Ticker.history() uses a different API path and is less throttled
            t  = yf.Ticker(sym)
            df = t.history(start=start, end=end, auto_adjust=True, timeout=15)
            if not df.empty:
                # Normalise index name
                df.index = pd.to_datetime(df.index)
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                return df
        except Exception:
            pass
        try:
            df = yf.download(sym, start=start, end=end, progress=False, auto_adjust=True)
            if not df.empty:
                return df
        except Exception:
            pass
        if attempt < retries - 1:
            sleep(3 + attempt * 2)
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_stock(sym: str, start: str, end: str) -> pd.DataFrame | None:
    raw = _yf_download_with_retry(sym, start, end)
    if raw.empty:
        return None
    df = _clean_yf(raw, sym)
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df


@st.cache_data(ttl=3600)
def load_macro(start: str, end: str) -> pd.DataFrame:
    """Download macro indicators and build derived features."""
    series: dict = {}
    for sym, name in MACRO_SYMBOLS.items():
        try:
            sleep(0.5)  # Spread requests to avoid rate limiting
            raw = _yf_download_with_retry(sym, start, end)
            if raw.empty:
                continue
            raw = _clean_yf(raw)
            close_col = next(
                (c for c in raw.columns if c.lower() in ("close", "adj_close", "adj close")),
                None,
            )
            if close_col:
                series[name] = raw[close_col].squeeze()
        except Exception:
            continue

    if not series:
        return pd.DataFrame()

    macro = pd.DataFrame(series)
    macro.index = pd.to_datetime(macro.index)

    # Derived macro features
    if "VIX" in macro.columns:
        macro["VIX_change"] = macro["VIX"].pct_change()
        macro["VIX_regime"] = (macro["VIX"] > 20).astype(int)  # 1 = elevated fear

    if "Yield10Y" in macro.columns:
        macro["Yield_change"] = macro["Yield10Y"].diff()

    if "SPX" in macro.columns:
        macro["SPX_return"] = macro["SPX"].pct_change()
        macro["SPX_trend"]  = (macro["SPX"] > macro["SPX"].rolling(50).mean()).astype(int)

    if "DXY" in macro.columns:
        macro["DXY_change"] = macro["DXY"].pct_change()

    return macro


FINNHUB_API_KEY = "cvs02bhr01qp7vitu2lgcvs02bhr01qp7vitu2m0"


@st.cache_data(ttl=3600)
def get_sentiment(sym: str, _monthly_dates: list) -> dict:
    """
    Fast two-stage sentiment pipeline:

    Stage 1 — yfinance live news (~1 s, no key needed)
        Fetches the latest ~40 headlines and scores them with VADER.
        Used to fill the most recent month(s).

    Stage 2 — Finnhub monthly historical (~3-5 s for 12 months, free key)
        One API call per month, no mandatory sleep.
        Falls back to 0 if the key is exhausted or returns nothing.

    Replaces GDELT (which required 5 s sleep per request = 60+ s total).
    """
    analyzer = SentimentIntensityAnalyzer()
    result: dict = {d: 0.0 for d in _monthly_dates}

    # ── Stage 1: yfinance live headlines (instant) ────────────────────────────
    try:
        news_items = yf.Ticker(sym).news or []
        live_scores: dict[str, list] = {}
        for item in news_items[:40]:
            text = item.get("title", "") + " " + item.get("summary", "")
            score = analyzer.polarity_scores(text)["compound"]
            ts = pd.Timestamp(item.get("providerPublishTime", 0), unit="s")
            key = ts.strftime("%Y-%m-%d")[:7]   # YYYY-MM
            live_scores.setdefault(key, []).append(score)

        # Map live scores onto the monthly date list
        for d in _monthly_dates:
            month_key = d[:7]
            if month_key in live_scores:
                result[d] = float(np.mean(live_scores[month_key]))
    except Exception:
        pass

    # ── Stage 2: Finnhub historical (one call per month, no forced sleep) ─────
    for d in _monthly_dates:
        if result[d] != 0.0:
            continue   # already filled by live news
        ts_start = pd.Timestamp(d)
        ts_end   = ts_start + pd.offsets.MonthEnd(0)
        try:
            url = (
                f"https://finnhub.io/api/v1/company-news"
                f"?symbol={sym}"
                f"&from={ts_start.strftime('%Y-%m-%d')}"
                f"&to={ts_end.strftime('%Y-%m-%d')}"
                f"&token={FINNHUB_API_KEY}"
            )
            r = requests.get(url, timeout=8)
            items = r.json() if r.status_code == 200 else []
            if isinstance(items, list) and items:
                scores = [
                    analyzer.polarity_scores(
                        a.get("headline", "") + " " + a.get("summary", "")
                    )["compound"]
                    for a in items[:10]
                ]
                result[d] = float(np.mean(scores))
        except Exception:
            pass
        sleep(0.15)   # gentle pace — Finnhub free = 60 req/min

    return result

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    close  = df["Close"].squeeze().astype(float)
    high   = df["High"].squeeze().astype(float)
    low    = df["Low"].squeeze().astype(float)
    volume = df["Volume"].squeeze().astype(float)

    # ── Returns & momentum ──────────────────────────────────────────────────
    df["Return"]     = close.pct_change()
    df["Return_5d"]  = close.pct_change(5)
    df["Return_20d"] = close.pct_change(20)

    # ── Volume ───────────────────────────────────────────────────────────────
    df["VolumeChange"] = volume.pct_change()
    vol_ma20           = volume.rolling(20).mean().replace(0, np.nan)
    df["Vol_ratio"]    = volume / vol_ma20

    # ── Moving averages & distances ──────────────────────────────────────────
    ma20  = close.rolling(20).mean()
    ma50  = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    df["MA20"]  = ma20
    df["MA50"]  = ma50
    df["MA200"] = ma200
    df["MA20_dist"]    = (close - ma20)  / ma20.replace(0, np.nan)
    df["MA50_dist"]    = (close - ma50)  / ma50.replace(0, np.nan)
    df["MA200_dist"]   = (close - ma200) / ma200.replace(0, np.nan)
    df["Golden_cross"] = (ma50 > ma200).astype(int)  # 1 = bullish regime

    # ── RSI ──────────────────────────────────────────────────────────────────
    rsi_ind             = RSIIndicator(close=close, window=14)
    df["RSI"]           = rsi_ind.rsi()
    df["RSI_oversold"]  = (df["RSI"] < 30).astype(int)
    df["RSI_overbought"]= (df["RSI"] > 70).astype(int)

    # ── MACD ─────────────────────────────────────────────────────────────────
    macd_ind        = MACD(close=close)
    df["MACD"]      = macd_ind.macd()
    df["MACD_diff"] = macd_ind.macd_diff()

    # ── Stochastic oscillator ────────────────────────────────────────────────
    stoch         = StochasticOscillator(high=high, low=low, close=close)
    df["Stoch_k"] = stoch.stoch()
    df["Stoch_d"] = stoch.stoch_signal()

    # ── Bollinger Bands ──────────────────────────────────────────────────────
    bb             = BollingerBands(close=close)
    df["BB_high"]  = bb.bollinger_hband()
    df["BB_low"]   = bb.bollinger_lband()
    df["BB_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / close.replace(0, np.nan)
    df["BB_pct"]   = bb.bollinger_pband()   # 0 = at lower band, 1 = at upper band

    # ── ADX (trend strength) ──────────────────────────────────────────────────
    adx_ind       = ADXIndicator(high=high, low=low, close=close)
    df["ADX"]     = adx_ind.adx()
    df["ADX_pos"] = adx_ind.adx_pos()   # +DI (bullish pressure)
    df["ADX_neg"] = adx_ind.adx_neg()   # -DI (bearish pressure)

    # ── ATR (normalized volatility) ───────────────────────────────────────────
    atr_ind  = AverageTrueRange(high=high, low=low, close=close)
    df["ATR"] = atr_ind.average_true_range() / close.replace(0, np.nan)

    # ── On-Balance Volume ─────────────────────────────────────────────────────
    obv_ind          = OnBalanceVolumeIndicator(close=close, volume=volume)
    df["OBV"]        = obv_ind.on_balance_volume()
    df["OBV_change"] = df["OBV"].pct_change(5)

    return df

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY METRICS
# ─────────────────────────────────────────────────────────────────────────────

def strategy_metrics(returns: pd.Series) -> dict:
    r = returns.dropna()
    if len(r) == 0:
        return {"sharpe": 0.0, "max_drawdown": 0.0, "win_rate": 0.0, "total_return": 0.0}
    sharpe     = r.mean() / (r.std() + 1e-10) * np.sqrt(252)
    cum        = (1 + r).cumprod()
    roll_max   = cum.cummax()
    max_dd     = ((cum - roll_max) / roll_max).min()
    win_rate   = (r > 0).mean()
    total_ret  = float(cum.iloc[-1]) - 1.0
    return {
        "sharpe":       float(sharpe),
        "max_drawdown": float(max_dd),
        "win_rate":     float(win_rate),
        "total_return": float(total_ret),
    }

# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def run_pipeline(ticker: str, use_sentiment: bool = True) -> dict:
    # ── 1. Load stock ─────────────────────────────────────────────────────────
    data = load_stock(ticker, DATA_START, END_DATE)
    if data is None:
        return {"error": f"Could not download data for {ticker}."}
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col not in data.columns:
            return {"error": f"Missing column '{col}'. Found: {list(data.columns)}"}

    # ── 2. Technical features ─────────────────────────────────────────────────
    data = add_technical_features(data)

    # ── 3. Macro data (join on date, forward-fill weekends / holidays) ────────
    macro = load_macro(DATA_START, END_DATE)
    if not macro.empty:
        data = data.join(macro, how="left")
        for col in macro.columns:
            if col in data.columns:
                data[col] = data[col].ffill()

    # ── 4. Sentiment (monthly GDELT + VADER) ──────────────────────────────────
    if use_sentiment:
        monthly  = pd.date_range(start=TEST_START, end=END_DATE, freq="MS").strftime("%Y-%m-%d").tolist()
        sent_map = get_sentiment(ticker, monthly)
        sent_df  = (
            pd.DataFrame({
                "Date":      pd.to_datetime(list(sent_map.keys())),
                "Sentiment": list(sent_map.values()),
            })
            .sort_values("Date")
        )
        data = data.reset_index()
        data = pd.merge_asof(data.sort_values("Date"), sent_df, on="Date", direction="backward")
        data.set_index("Date", inplace=True)
    else:
        data = data.reset_index()
        data.set_index("Date", inplace=True)

    data["Sentiment"] = data.get("Sentiment", pd.Series(0, index=data.index)).fillna(0)

    # ── 5. Binary target: 1 if next-day close is higher ──────────────────────
    data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)

    # ── 6. Keep only available features ──────────────────────────────────────
    feats = [f for f in ALL_FEATURES if f in data.columns]
    data.dropna(subset=feats + ["Target"], inplace=True)

    # ── 7. Walk-forward train / test split (no look-ahead leakage) ───────────
    train_mask = (data.index >= pd.Timestamp(TRAIN_START)) & (data.index < pd.Timestamp(TEST_START))
    test_mask  =  data.index >= pd.Timestamp(TEST_START)

    X_train = data.loc[train_mask, feats]
    y_train = data.loc[train_mask, "Target"]
    X_test  = data.loc[test_mask,  feats]
    y_test  = data.loc[test_mask,  "Target"]

    if X_train.shape[0] < 100 or X_test.shape[0] < 20:
        return {"error": "Not enough data for a reliable train / test split."}

    # ── 8. Train XGBoost ──────────────────────────────────────────────────────
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # ── 9. Out-of-sample predictions ONLY ────────────────────────────────────
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc    = float(accuracy_score(y_test, y_pred))

    data.loc[test_mask, "Prediction"] = y_pred.astype(float)
    data.loc[test_mask, "Confidence"] = y_prob

    # ── 10. Strategy simulation on test period only ───────────────────────────
    td = data.loc[test_mask].copy()
    td["StrategyReturn"] = td["Prediction"].shift(1) * td["Return"]
    td["CumStrategy"]    = (1 + td["StrategyReturn"]).cumprod()
    td["CumMarket"]      = (1 + td["Return"]).cumprod()

    # ── 11. Feature importance ────────────────────────────────────────────────
    importance = (
        pd.Series(model.feature_importances_, index=feats)
        .sort_values(ascending=False)
        .head(15)
    )

    return {
        "data":        data,
        "test_data":   td,
        "accuracy":    acc,
        "importance":  importance,
        "metrics":     strategy_metrics(td["StrategyReturn"]),
        "mkt_metrics": strategy_metrics(td["Return"]),
        "features":    feats,
        "macro":       macro,
    }

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR OPTIONS
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")
    use_sentiment = st.toggle(
        "Include news sentiment",
        value=False,
        help=(
            "Fetches monthly GDELT news articles and scores them with VADER sentiment analysis.\n\n"
            "⏱ Adds ~90 seconds to first load (18 months × 5 s rate limit).\n"
            "Results are cached for 1 hour after the first run."
        ),
    )
    st.caption(
        "**Tip:** Load without sentiment first (fast), then enable it once you're happy with the setup."
    )
    st.divider()
    st.caption(
        f"**Train:** {TRAIN_START} → {TEST_START}  \n"
        f"**Test:** {TEST_START} → {END_DATE}  \n"
        "All metrics are out-of-sample only.  \n"
        f"Data refreshes daily."
    )

# ─────────────────────────────────────────────────────────────────────────────
# RUN PIPELINE — with step-by-step progress
# ─────────────────────────────────────────────────────────────────────────────

progress_box = st.empty()

with progress_box.status("Loading data…", expanded=True) as status:
    st.write("📥 Downloading stock price data…")
    # Pre-warm the individual caches so the status messages are visible
    _stock_test = load_stock(ticker, DATA_START, END_DATE)
    if _stock_test is None:
        status.update(label="❌ Failed to download stock data", state="error")
        st.error(f"Could not download price data for **{ticker}**. Yahoo Finance may be temporarily unavailable — please try again in a minute.")
        st.stop()

    st.write("🌍 Downloading macro data (VIX, yields, S&P 500, USD)…")
    load_macro(DATA_START, END_DATE)

    if use_sentiment:
        n_months = len(pd.date_range(start=TEST_START, end=END_DATE, freq="MS"))
        st.write(f"📰 Fetching news sentiment for **{ticker}** (~{n_months * 2} s via yfinance + Finnhub)…")
        monthly_preview = pd.date_range(start=TEST_START, end=END_DATE, freq="MS").strftime("%Y-%m-%d").tolist()
        get_sentiment(ticker, monthly_preview)

    st.write("🤖 Training XGBoost model…")
    result = run_pipeline(ticker, use_sentiment=use_sentiment)

    if "error" in result:
        status.update(label="❌ Pipeline failed", state="error")
        st.error(result["error"])
        st.stop()

    status.update(label="✅ Ready!", state="complete", expanded=False)

progress_box.empty()

if "error" in result:
    st.error(result["error"])
    st.stop()

data      = result["data"]
test_data = result["test_data"]
accuracy  = result["accuracy"]
importance= result["importance"]
metrics   = result["metrics"]
mkt_m     = result["mkt_metrics"]
macro     = result["macro"]

# ─────────────────────────────────────────────────────────────────────────────
# HEADER KPIs
# ─────────────────────────────────────────────────────────────────────────────

latest     = test_data.iloc[-1]
is_buy     = latest.get("Prediction") == 1.0
signal_val = "BUY" if is_buy else "SELL"
signal_icon= "🟢" if is_buy else "🔴"
conf       = latest.get("Confidence", 0.5)

# ── Sentiment pill for the banner ────────────────────────────────────────────
sent_score   = float(data["Sentiment"].iloc[-1]) if "Sentiment" in data.columns else 0.0
if use_sentiment and sent_score != 0.0:
    if sent_score > 0.05:
        sent_label = f"📰 News: Positive ({sent_score:+.2f})"
        sent_css   = "color:#2ecc71"
    elif sent_score < -0.05:
        sent_label = f"📰 News: Negative ({sent_score:+.2f})"
        sent_css   = "color:#e74c3c"
    else:
        sent_label = f"📰 News: Neutral ({sent_score:+.2f})"
        sent_css   = "color:#aaa"
    sent_rank  = int(result["importance"].index.tolist().index("Sentiment") + 1) \
                 if "Sentiment" in result["importance"].index else None
    sent_rank_str = f" · #{sent_rank} most important feature" if sent_rank else ""
    sent_html  = f'<span style="{sent_css};font-size:0.85rem">{sent_label}{sent_rank_str}</span>'
else:
    sent_html  = '<span style="color:#666;font-size:0.85rem">📰 News sentiment: off</span>'

signal_color = "#1a7a1a" if is_buy else "#8b0000"
st.markdown(
    f"""
    <div style="
        background:{signal_color}22;
        border:2px solid {signal_color};
        border-radius:10px;
        padding:12px 20px;
        display:flex;
        align-items:center;
        gap:20px;
        margin-bottom:12px;
    ">
        <span style="font-size:2.6rem;line-height:1">{signal_icon}</span>
        <div style="flex:1">
            <div style="font-size:1.9rem;font-weight:800;color:{signal_color};line-height:1.1">
                {signal_val}
            </div>
            <div style="font-size:0.85rem;opacity:0.75;margin-top:3px">
                <b>{ticker}</b> · Confidence: <b>{conf:.1%}</b>
                · Close: <b>${latest['Close']:.2f}</b>
                · Data through: <b>{test_data.index[-1].strftime('%d %b %Y')}</b>
            </div>
            <div style="margin-top:4px">{sent_html}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("🎯 Confidence",        f"{conf:.1%}")
c2.metric("✅ OOS Accuracy",      f"{accuracy:.1%}")
c3.metric(
    "📈 Strategy Return",
    f"{metrics['total_return']:+.1%}",
    delta=f"vs market {mkt_m['total_return']:+.1%}",
)
c4.metric(
    "⚡ Sharpe Ratio",
    f"{metrics['sharpe']:.2f}",
    delta=f"mkt {mkt_m['sharpe']:.2f}",
)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Backtest & Strategy",
    "📉 Technical Analysis",
    "🌍 Macro Environment",
    "🤖 Model Insights",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Backtest
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Strategy vs Buy & Hold  (out-of-sample only)")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(
            test_data.index, test_data["CumStrategy"],
            label=f"Model Strategy  ({metrics['total_return']:+.1%})",
            color="royalblue", lw=2,
        )
        ax.plot(
            test_data.index, test_data["CumMarket"],
            label=f"Buy & Hold  ({mkt_m['total_return']:+.1%})",
            color="gray", lw=1.5, linestyle="--",
        )
        ax.set_title(f"{ticker}  —  Cumulative Returns  ({TEST_START} → {END_DATE})")
        ax.set_ylabel("Growth of $1")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    with col_r:
        st.subheader("Risk / Return Metrics")
        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Max Drawdown",
            f"{metrics['max_drawdown']:.1%}",
            delta=f"mkt {mkt_m['max_drawdown']:.1%}",
            delta_color="inverse",
        )
        m2.metric("Win Rate",       f"{metrics['win_rate']:.1%}")
        m3.metric("OOS Accuracy",   f"{accuracy:.1%}")

        st.divider()
        st.subheader("Recent Signals  (last 20 trading days)")
        recent = test_data[["Close", "Prediction", "Confidence"]].tail(20).copy()
        recent["Signal"]     = recent["Prediction"].map({1.0: "🟢 BUY", 0.0: "🔴 SELL"})
        recent["Confidence"] = recent["Confidence"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
        recent["Close"]      = recent["Close"].map(lambda x: f"${x:.2f}")
        st.dataframe(
            recent[["Close", "Signal", "Confidence"]].iloc[::-1],
            use_container_width=True,
        )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Technical Analysis
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    _cutoff = data.index[-1] - pd.Timedelta(days=180)
    plot_d  = data[data.index >= _cutoff]

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Price & Moving Averages")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(plot_d.index, plot_d["Close"], label="Close",  color="black",  lw=1.5)
        ax.plot(plot_d.index, plot_d["MA20"],  label="MA 20",  color="orange", lw=1, alpha=0.9)
        ax.plot(plot_d.index, plot_d["MA50"],  label="MA 50",  color="blue",   lw=1, alpha=0.9)
        ax.plot(plot_d.index, plot_d["MA200"], label="MA 200", color="red",    lw=1, alpha=0.9)
        # Overlay buy signals from the test period
        buy_mask = plot_d.get("Prediction") == 1.0 if "Prediction" in plot_d.columns else None
        if buy_mask is not None and buy_mask.any():
            ax.scatter(
                plot_d[buy_mask].index, plot_d[buy_mask]["Close"],
                marker="^", color="lime", s=40, zorder=5, label="Buy signal",
            )
        ax.set_title(f"{ticker}  —  Price & MAs (last 6 months)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    with col_r:
        st.subheader("RSI & Stochastic")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
        ax1.plot(plot_d.index, plot_d["RSI"], color="purple", lw=1.5)
        ax1.axhline(70, color="red",   linestyle="--", alpha=0.5, label="Overbought 70")
        ax1.axhline(30, color="green", linestyle="--", alpha=0.5, label="Oversold 30")
        ax1.set_ylim(0, 100)
        ax1.set_title("RSI (14)")
        ax1.legend(fontsize=7)
        ax1.grid(alpha=0.3)
        ax2.plot(plot_d.index, plot_d["Stoch_k"], label="%K", color="steelblue", lw=1)
        ax2.plot(plot_d.index, plot_d["Stoch_d"], label="%D", color="darkorange", lw=1)
        ax2.axhline(80, color="red",   linestyle="--", alpha=0.5)
        ax2.axhline(20, color="green", linestyle="--", alpha=0.5)
        ax2.set_ylim(0, 100)
        ax2.set_title("Stochastic Oscillator")
        ax2.legend(fontsize=7)
        ax2.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("MACD")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
        ax1.plot(plot_d.index, plot_d["MACD"], color="steelblue", lw=1.5, label="MACD")
        ax1.set_title("MACD Line")
        ax1.legend(fontsize=7)
        ax1.grid(alpha=0.3)
        colors = ["seagreen" if v >= 0 else "tomato" for v in plot_d["MACD_diff"]]
        ax2.bar(plot_d.index, plot_d["MACD_diff"], color=colors, width=1)
        ax2.axhline(0, color="black", lw=0.5)
        ax2.set_title("MACD Histogram (momentum)")
        ax2.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

    with col4:
        st.subheader("ADX — Trend Strength")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(plot_d.index, plot_d["ADX"],     label="ADX",  color="black",    lw=2)
        ax.plot(plot_d.index, plot_d["ADX_pos"], label="+DI",  color="seagreen", lw=1, alpha=0.8)
        ax.plot(plot_d.index, plot_d["ADX_neg"], label="-DI",  color="tomato",   lw=1, alpha=0.8)
        ax.axhline(25, color="gray", linestyle="--", alpha=0.6, label="Trend threshold (25)")
        ax.set_title("ADX  (> 25 = trending, < 25 = ranging)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        st.pyplot(fig)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Macro Environment
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    if macro.empty:
        st.warning("Macro data could not be loaded. Check your internet connection.")
    else:
        _mc = data.index[-1] - pd.Timedelta(days=365)
        pm  = macro[macro.index >= _mc]

        col_l, col_r = st.columns(2)

        with col_l:
            if "VIX" in pm.columns:
                st.subheader("VIX — Fear & Volatility Index")
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(pm.index, pm["VIX"], color="crimson", lw=1.5)
                ax.axhline(20, color="orange",  linestyle="--", alpha=0.6, label="Elevated (20)")
                ax.axhline(30, color="darkred", linestyle="--", alpha=0.6, label="Extreme (30)")
                ax.fill_between(pm.index, pm["VIX"], 20, where=pm["VIX"] > 20, alpha=0.15, color="red")
                ax.set_title("VIX  (higher = more fear)")
                ax.legend(fontsize=8)
                ax.grid(alpha=0.3)
                st.pyplot(fig)

        with col_r:
            if "Yield10Y" in pm.columns:
                st.subheader("10-Year Treasury Yield")
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(pm.index, pm["Yield10Y"], color="navy", lw=1.5)
                ax.set_title("10Y Treasury Yield (%)  —  rising = headwind for growth stocks")
                ax.set_ylabel("Yield (%)")
                ax.grid(alpha=0.3)
                st.pyplot(fig)

        col3, col4 = st.columns(2)

        with col3:
            if "SPX" in pm.columns:
                st.subheader("S&P 500 — Market Regime")
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(pm.index, pm["SPX"], color="steelblue", lw=1.5)
                ax.plot(pm.index, pm["SPX"].rolling(50).mean(),
                        color="orange", lw=1, linestyle="--", label="50-day MA")
                ax.set_title("S&P 500  (above 50MA = bull regime)")
                ax.legend(fontsize=8)
                ax.grid(alpha=0.3)
                st.pyplot(fig)

        with col4:
            if "DXY" in pm.columns:
                st.subheader("USD Strength (UUP ETF proxy)")
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(pm.index, pm["DXY"], color="darkgreen", lw=1.5)
                ax.set_title("USD Strength  (rising = headwind for commodities & multinationals)")
                ax.grid(alpha=0.3)
                st.pyplot(fig)

        st.divider()
        st.subheader("📊 Current Macro Snapshot")
        snap_cols = ["VIX", "Yield10Y", "SPX", "DXY"]
        snap      = {c: round(float(macro[c].dropna().iloc[-1]), 2) for c in snap_cols if c in macro.columns}
        s_cols    = st.columns(len(snap))
        for col_w, (k, v) in zip(s_cols, snap.items()):
            col_w.metric(k, v)

        st.info(
            "**How to read macro signals:**\n\n"
            "• **VIX > 20** → elevated fear; markets historically mean-revert  \n"
            "• **VIX > 30** → extreme fear; one of history's strongest buy signals  \n"
            "• **Rising 10Y yield** → headwind for high-growth / high-PE stocks  \n"
            "• **Rising USD (UUP)** → pressure on commodities, gold, and multinationals  \n"
            "• **S&P 500 above 50MA** → bull regime; trend-following strategies perform better  "
        )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — Model Insights
# ═════════════════════════════════════════════════════════════════════════════
with tab4:

    # ── Sentiment section (full width, prominent) ─────────────────────────────
    st.subheader("📰 News Sentiment (GDELT + VADER)")

    if not use_sentiment:
        st.info(
            "**Sentiment is currently turned off.**  \n"
            "Enable it in the sidebar to fetch monthly news headlines from GDELT, "
            "score them with VADER sentiment analysis, and feed the score into the model as a feature.  \n"
            "You'll then see a chart here showing whether the news has been positive or negative each month, "
            "and how much it influences predictions."
        )
    elif "Sentiment" in data.columns and data["Sentiment"].abs().sum() > 0:
        # Deduplicate to monthly points (sentiment is the same for all days in a month)
        sent_monthly = (
            data["Sentiment"]
            .resample("MS").first()
            .dropna()
        )
        sent_monthly = sent_monthly[sent_monthly != 0]

        if len(sent_monthly) > 0:
            # Score colours
            bar_colors = [
                "#2ecc71" if s > 0.05 else ("#e74c3c" if s < -0.05 else "#95a5a6")
                for s in sent_monthly
            ]
            fig, ax = plt.subplots(figsize=(14, 3))
            bars = ax.bar(sent_monthly.index, sent_monthly, color=bar_colors,
                          alpha=0.9, width=20, edgecolor="white", lw=0.5)
            ax.axhline(0, color="white", lw=0.8, alpha=0.5)
            ax.axhline( 0.05, color="#2ecc71", lw=0.8, linestyle="--", alpha=0.4, label="Positive threshold")
            ax.axhline(-0.05, color="#e74c3c", lw=0.8, linestyle="--", alpha=0.4, label="Negative threshold")

            # Annotate each bar with its score
            for bar, score in zip(bars, sent_monthly):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.005 if score >= 0 else -0.012),
                    f"{score:+.2f}",
                    ha="center", va="bottom" if score >= 0 else "top",
                    fontsize=7.5, color="white", alpha=0.85,
                )

            ax.set_title(
                f"{ticker} — Monthly News Sentiment  "
                f"(🟢 positive  🔴 negative  ⬜ neutral)  ·  "
                f"Current: {sent_monthly.iloc[-1]:+.2f}",
                fontsize=11,
            )
            ax.set_ylabel("VADER Compound Score  (–1 to +1)")
            ax.set_ylim(
                min(sent_monthly.min() - 0.05, -0.15),
                max(sent_monthly.max() + 0.05,  0.15),
            )
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.2)
            st.pyplot(fig)

            # Sentiment KPIs
            sk1, sk2, sk3, sk4 = st.columns(4)
            sk1.metric("Current month",  f"{sent_monthly.iloc[-1]:+.3f}")
            sk2.metric("3-month avg",    f"{sent_monthly.tail(3).mean():+.3f}")
            sk3.metric("Positive months",
                       f"{(sent_monthly > 0.05).sum()} / {len(sent_monthly)}")
            sk4.metric("Sentiment in model",
                       "✅ Active" if "Sentiment" in result["features"] else "❌ Not used")
        else:
            st.warning("Sentiment was enabled but no scores were returned from GDELT this run.")
    else:
        st.warning("No sentiment data available for this ticker yet.")

    st.divider()

    # ── Feature importance + model config ─────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Feature Importance  (Top 15)")

        # Highlight sentiment bar if present
        imp_colors = [
            "#e67e22" if f == "Sentiment" else "steelblue"
            for f in importance.sort_values().index
        ]
        fig, ax = plt.subplots(figsize=(10, 6))
        importance.sort_values().plot(kind="barh", ax=ax, color=imp_colors, edgecolor="white")
        ax.set_title("XGBoost Feature Importance  (orange = sentiment)")
        ax.set_xlabel("Importance Score")
        ax.grid(axis="x", alpha=0.3)
        st.pyplot(fig)

    with col_r:
        st.subheader("Model Configuration")
        st.json({
            "algorithm":    "XGBoost Classifier",
            "train_period": f"{TRAIN_START} → {TEST_START}",
            "test_period":  f"{TEST_START} → {END_DATE}",
            "n_features":   len(result["features"]),
            "oos_accuracy": f"{accuracy:.2%}",
            "sentiment":    "ON — GDELT monthly VADER scores" if use_sentiment else "OFF",
            "validation":   "Walk-forward (strict train/test split, no data leakage)",
            "feature_groups": {
                "price_momentum": ["Return", "Return_5d", "Return_20d"],
                "volume":         ["VolumeChange", "Vol_ratio", "OBV_change"],
                "trend":          ["MA20/50/200_dist", "Golden_cross", "MACD", "ADX +/-DI"],
                "oscillators":    ["RSI", "Stochastic %K/%D"],
                "volatility":     ["BB_width", "BB_pct", "ATR"],
                "macro":          ["VIX + change + regime", "10Y yield + change",
                                   "S&P return + trend", "DXY change"],
                "sentiment":      ["Monthly GDELT news — VADER compound score"],
            },
        })
