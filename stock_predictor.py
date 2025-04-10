import yfinance as yf
import pandas as pd
import numpy as np
import requests
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta

# === CONFIGURATION ===
tickers = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMD', 'PLTR', 'TSLA', 'META', 'SPY', 'QQQ', 'ARKK', 'GLD']
start_date = "2023-01-01"
end_date = "2024-01-01"
finnhub_api_key = "cvs02bhr01qp7vitu2lgcvs02bhr01qp7vitu2m0"

analyzer = SentimentIntensityAnalyzer()
summary = []

# === Finnhub News Fetcher ===
def get_finnhub_sentiment(ticker, date):
    try:
        url = (
            f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={date}&to={date}&token={finnhub_api_key}"
        )
        response = requests.get(url)
        news_items = response.json()
        if not isinstance(news_items, list) or not news_items:
            return 0.0

        scores = []
        for article in news_items[:5]:  # Limit to first 5 per day
            text = article.get('headline', '') + " " + article.get('summary', '')
            score = analyzer.polarity_scores(text)["compound"]
            scores.append(score)

        return np.mean(scores)
    except:
        return 0.0

# === Main Loop ===
for ticker in tickers:
    try:
        print(f"\n📥 Downloading data for {ticker}...")
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if len(data) < 100:
            print(f"⚠️ Not enough data for {ticker}, skipping.")
            continue

        # Feature engineering
        data['Return'] = data['Close'].pct_change()
        data['VolumeChange'] = data['Volume'].pct_change()
        close_series = data[['Close']].astype(float).squeeze()
        data['RSI'] = RSIIndicator(close=close_series).rsi()
        macd = MACD(close=close_series)
        data['MACD'] = macd.macd()
        data['MACD_diff'] = macd.macd_diff()
        bb = BollingerBands(close=close_series)
        data['BB_high'] = bb.bollinger_hband()
        data['BB_low'] = bb.bollinger_lband()
        data['BB_width'] = data['BB_high'] - data['BB_low']

        # News sentiment (last 30 days only)
        sentiment_map = {}
        print("🧠 Fetching sentiment from Finnhub...")
        recent_dates = data.tail(30).index.strftime('%Y-%m-%d')

        for date in recent_dates:
            sentiment_map[date] = get_finnhub_sentiment(ticker, date)

        sentiment_scores = []
        for date in data.index.strftime('%Y-%m-%d'):
            sentiment_scores.append(sentiment_map.get(date, 0.0))

        data['Sentiment'] = sentiment_scores

        # Target column
        data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
        data.dropna(inplace=True)

        features = ['Return', 'VolumeChange', 'RSI', 'MACD', 'MACD_diff', 'BB_width', 'Sentiment']
        X = data[features]
        y = data['Target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=1.0,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Strategy simulation
        data['Prediction'] = model.predict(X)
        data['StrategyReturn'] = data['Prediction'].shift(1) * data['Return']
        data['CumulativeStrategy'] = (1 + data['StrategyReturn']).cumprod()
        data['CumulativeMarket'] = (1 + data['Return']).cumprod()

        summary.append({
            'Ticker': ticker,
            'Accuracy': round(accuracy, 3),
            'Strategy Growth': round(data['CumulativeStrategy'].iloc[-1], 2),
            'Market Growth': round(data['CumulativeMarket'].iloc[-1], 2)
        })

    except Exception as e:
        summary.append({
            'Ticker': ticker,
            'Error': str(e)
        })

# === Summary Output ===
summary_df = pd.DataFrame(summary)
print("\n✅ Final Model Summary:\n")
print(summary_df.to_string(index=False))
summary_df.to_csv("strategy_summary_finnhub_sentiment.csv", index=False)
