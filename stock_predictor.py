"""
Stock Predictor — Batch Runner v2.0

Runs walk-forward predictions for all tickers and saves an enhanced CSV summary
with Sharpe ratio, max drawdown, win rate, and feature importance rankings.

Improvements over v1:
  - Walk-forward train/test split (no data leakage)
  - Macro features: VIX, 10Y yield, S&P regime, USD strength
  - Extended technical: ADX, Stochastic, OBV, ATR, MA distances
  - Risk metrics: Sharpe, max drawdown, win rate
  - Finnhub sentiment still used for recent 30 days
"""

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from time import sleep

from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMD", "PLTR",
           "TSLA", "META", "SPY", "QQQ", "ARKK", "GLD"]

DATA_START  = "2021-01-01"
TRAIN_START = "2022-01-01"
TEST_START  = "2023-07-01"
END_DATE    = "2024-12-31"

FINNHUB_API_KEY = "cvs02bhr01qp7vitu2lgcvs02bhr01qp7vitu2m0"

MACRO_SYMBOLS = {
    "^VIX":  "VIX",
    "^TNX":  "Yield10Y",
    "^GSPC": "SPX",
    "UUP":   "DXY",
}

ALL_FEATURES = [
    "Return", "Return_5d", "Return_20d",
    "VolumeChange", "Vol_ratio", "OBV_change",
    "MA20_dist", "MA50_dist", "MA200_dist", "Golden_cross",
    "MACD", "MACD_diff",
    "ADX", "ADX_pos", "ADX_neg",
    "RSI", "RSI_oversold", "RSI_overbought",
    "Stoch_k", "Stoch_d",
    "BB_width", "BB_pct", "ATR",
    "VIX", "VIX_change", "VIX_regime",
    "Yield10Y", "Yield_change",
    "SPX_return", "SPX_trend",
    "DXY_change",
    "Sentiment",
]

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

analyzer = SentimentIntensityAnalyzer()

def _yf_download(sym: str, start: str, end: str, retries: int = 3) -> pd.DataFrame:
    """Download via Ticker.history() (less rate-limited), fallback to yf.download()."""
    for attempt in range(retries):
        try:
            t  = yf.Ticker(sym)
            df = t.history(start=start, end=end, auto_adjust=True, timeout=15)
            if not df.empty:
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


def _clean_yf(df: pd.DataFrame, sym: str = "") -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(str(p) for p in parts if p).strip()
            for parts in df.columns.values
        ]
    if sym:
        sfx = f"_{sym}"
        df.columns = [c[: -len(sfx)] if c.endswith(sfx) else c for c in df.columns]
    df.columns = [c.strip().title() for c in df.columns]
    df.rename(
        columns={"Adj Close": "Close", "Adj_Close": "Close"},
        inplace=True, errors="ignore",
    )
    for extra in ("Dividends", "Stock_Splits", "Capital Gains"):
        df.drop(columns=[extra], inplace=True, errors="ignore")
    return df


def load_macro() -> pd.DataFrame:
    series: dict = {}
    for sym, name in MACRO_SYMBOLS.items():
        try:
            sleep(0.5)
            raw = _yf_download(sym, DATA_START, END_DATE)
            if raw.empty:
                continue
            raw = _clean_yf(raw)
            close_col = next(
                (c for c in raw.columns if c.lower() in ("close", "adj_close")), None
            )
            if close_col:
                series[name] = raw[close_col].squeeze()
        except Exception:
            continue
    if not series:
        return pd.DataFrame()
    macro = pd.DataFrame(series)
    macro.index = pd.to_datetime(macro.index)
    if "VIX" in macro.columns:
        macro["VIX_change"] = macro["VIX"].pct_change()
        macro["VIX_regime"] = (macro["VIX"] > 20).astype(int)
    if "Yield10Y" in macro.columns:
        macro["Yield_change"] = macro["Yield10Y"].diff()
    if "SPX" in macro.columns:
        macro["SPX_return"] = macro["SPX"].pct_change()
        macro["SPX_trend"]  = (macro["SPX"] > macro["SPX"].rolling(50).mean()).astype(int)
    if "DXY" in macro.columns:
        macro["DXY_change"] = macro["DXY"].pct_change()
    return macro


def get_finnhub_sentiment(ticker: str, date: str) -> float:
    try:
        url = (
            f"https://finnhub.io/api/v1/company-news"
            f"?symbol={ticker}&from={date}&to={date}&token={FINNHUB_API_KEY}"
        )
        items = requests.get(url, timeout=8).json()
        if not isinstance(items, list) or not items:
            return 0.0
        scores = [
            analyzer.polarity_scores(a.get("headline", "") + " " + a.get("summary", ""))["compound"]
            for a in items[:5]
        ]
        return float(np.mean(scores))
    except Exception:
        return 0.0


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    close  = df["Close"].squeeze().astype(float)
    high   = df["High"].squeeze().astype(float)
    low    = df["Low"].squeeze().astype(float)
    volume = df["Volume"].squeeze().astype(float)

    df["Return"]     = close.pct_change()
    df["Return_5d"]  = close.pct_change(5)
    df["Return_20d"] = close.pct_change(20)
    df["VolumeChange"] = volume.pct_change()
    vol_ma20 = volume.rolling(20).mean().replace(0, np.nan)
    df["Vol_ratio"] = volume / vol_ma20

    ma20  = close.rolling(20).mean()
    ma50  = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    df["MA20"]       = ma20
    df["MA50"]       = ma50
    df["MA200"]      = ma200
    df["MA20_dist"]  = (close - ma20)  / ma20.replace(0, np.nan)
    df["MA50_dist"]  = (close - ma50)  / ma50.replace(0, np.nan)
    df["MA200_dist"] = (close - ma200) / ma200.replace(0, np.nan)
    df["Golden_cross"] = (ma50 > ma200).astype(int)

    rsi = RSIIndicator(close=close, window=14)
    df["RSI"]            = rsi.rsi()
    df["RSI_oversold"]   = (df["RSI"] < 30).astype(int)
    df["RSI_overbought"] = (df["RSI"] > 70).astype(int)

    macd_ind        = MACD(close=close)
    df["MACD"]      = macd_ind.macd()
    df["MACD_diff"] = macd_ind.macd_diff()

    stoch         = StochasticOscillator(high=high, low=low, close=close)
    df["Stoch_k"] = stoch.stoch()
    df["Stoch_d"] = stoch.stoch_signal()

    bb             = BollingerBands(close=close)
    df["BB_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / close.replace(0, np.nan)
    df["BB_pct"]   = bb.bollinger_pband()

    adx_ind       = ADXIndicator(high=high, low=low, close=close)
    df["ADX"]     = adx_ind.adx()
    df["ADX_pos"] = adx_ind.adx_pos()
    df["ADX_neg"] = adx_ind.adx_neg()

    atr_ind  = AverageTrueRange(high=high, low=low, close=close)
    df["ATR"] = atr_ind.average_true_range() / close.replace(0, np.nan)

    obv_ind          = OnBalanceVolumeIndicator(close=close, volume=volume)
    df["OBV"]        = obv_ind.on_balance_volume()
    df["OBV_change"] = df["OBV"].pct_change(5)

    return df


def strategy_metrics(returns: pd.Series) -> dict:
    r = returns.dropna()
    if len(r) == 0:
        return {"sharpe": 0.0, "max_drawdown": 0.0, "win_rate": 0.0, "total_return": 0.0}
    sharpe    = r.mean() / (r.std() + 1e-10) * np.sqrt(252)
    cum       = (1 + r).cumprod()
    roll_max  = cum.cummax()
    max_dd    = float(((cum - roll_max) / roll_max).min())
    win_rate  = float((r > 0).mean())
    total_ret = float(cum.iloc[-1]) - 1.0
    return {
        "sharpe":       float(sharpe),
        "max_drawdown": max_dd,
        "win_rate":     win_rate,
        "total_return": total_ret,
    }

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

print("\n🌍 Loading macro data…")
macro = load_macro()
if macro.empty:
    print("  ⚠️  Could not load macro data — macro features will be skipped.")

summary = []

for ticker in TICKERS:
    print(f"\n📥 Processing {ticker}…")
    try:
        # ── Stock data ────────────────────────────────────────────────────────
        raw = _yf_download(ticker, DATA_START, END_DATE)
        if len(raw) < 200:
            print(f"   ⚠️  Not enough data, skipping.")
            continue
        data = _clean_yf(raw, ticker)
        data.index = pd.to_datetime(data.index)

        # ── Technical features ────────────────────────────────────────────────
        data = add_features(data)

        # ── Macro merge ───────────────────────────────────────────────────────
        if not macro.empty:
            data = data.join(macro, how="left")
            for col in macro.columns:
                if col in data.columns:
                    data[col] = data[col].ffill()

        # ── Sentiment (last 30 days of training data) ─────────────────────────
        print(f"   📰 Fetching Finnhub sentiment…")
        recent_dates = (
            data[data.index < pd.Timestamp(TEST_START)]
            .tail(30).index.strftime("%Y-%m-%d")
        )
        sent_map = {d: get_finnhub_sentiment(ticker, d) for d in recent_dates}
        data["Sentiment"] = [sent_map.get(d, 0.0) for d in data.index.strftime("%Y-%m-%d")]

        # ── Target ────────────────────────────────────────────────────────────
        data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)

        # ── Feature selection ─────────────────────────────────────────────────
        feats = [f for f in ALL_FEATURES if f in data.columns]
        data.dropna(subset=feats + ["Target"], inplace=True)

        # ── Walk-forward train / test split ───────────────────────────────────
        train_mask = (data.index >= pd.Timestamp(TRAIN_START)) & (data.index < pd.Timestamp(TEST_START))
        test_mask  =  data.index >= pd.Timestamp(TEST_START)

        X_train = data.loc[train_mask, feats]
        y_train = data.loc[train_mask, "Target"]
        X_test  = data.loc[test_mask,  feats]
        y_test  = data.loc[test_mask,  "Target"]

        if X_train.shape[0] < 100 or X_test.shape[0] < 20:
            print(f"   ⚠️  Insufficient data for split, skipping.")
            continue

        # ── Train ─────────────────────────────────────────────────────────────
        model = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            use_label_encoder=False, eval_metric="logloss", random_state=42,
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        # ── OOS predictions only ──────────────────────────────────────────────
        y_pred    = model.predict(X_test)
        accuracy  = float(accuracy_score(y_test, y_pred))

        # ── Strategy simulation (test period only) ────────────────────────────
        td = data.loc[test_mask].copy()
        td["Prediction"]     = y_pred
        td["StrategyReturn"] = td["Prediction"].shift(1) * td["Return"]
        td["CumStrategy"]    = (1 + td["StrategyReturn"]).cumprod()
        td["CumMarket"]      = (1 + td["Return"]).cumprod()

        sm  = strategy_metrics(td["StrategyReturn"])
        mm  = strategy_metrics(td["Return"])

        # Top feature for this ticker
        top_feat = pd.Series(model.feature_importances_, index=feats).idxmax()

        print(
            f"   ✅  Accuracy={accuracy:.1%}  "
            f"Strategy={sm['total_return']:+.1%}  "
            f"Market={mm['total_return']:+.1%}  "
            f"Sharpe={sm['sharpe']:.2f}  "
            f"Top feature: {top_feat}"
        )

        summary.append({
            "Ticker":          ticker,
            "OOS_Accuracy":    round(accuracy, 3),
            "Strategy_Return": round(sm["total_return"], 3),
            "Market_Return":   round(mm["total_return"], 3),
            "Sharpe":          round(sm["sharpe"], 2),
            "Max_Drawdown":    round(sm["max_drawdown"], 3),
            "Win_Rate":        round(sm["win_rate"], 3),
            "Top_Feature":     top_feat,
            "N_Features":      len(feats),
            "Train_Samples":   X_train.shape[0],
            "Test_Samples":    X_test.shape[0],
        })

    except Exception as e:
        print(f"   ❌  Error: {e}")
        summary.append({"Ticker": ticker, "Error": str(e)})

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

df_out = pd.DataFrame(summary)
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(df_out.to_string(index=False))
print("=" * 70)

out_path = "strategy_summary_v2.csv"
df_out.to_csv(out_path, index=False)
print(f"\n✅ Results saved to {out_path}")
