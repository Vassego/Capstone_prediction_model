# IMPORT PACKAGES
from flask import Flask, render_template, request, flash, redirect, url_for
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import math
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import warnings
import os
import nltk
nltk.download('punkt', quiet=True)

# NEW: sentiment tools
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote as urlquote
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon', quiet=True)

# Ignore Warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
NEWSAPI_KEY = "e1ba8709df88429b9201e168474256c5"

# FLASK
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for flash messages

# To control caching so as to save and retrieve plot figs on client side
@app.after_request
def add_header(response):
    response.headers['Pragma'] = 'no-cache'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Expires'] = '0'
    return response

@app.route('/')
def index():
    return render_template('index.html')

# ------------------------------ Data Utilities ------------------------------

def get_historical(quote):
    end = datetime.now()
    start = end - timedelta(days=730)  # 2 years
    try:
        data = yf.download(quote, start=start, end=end, progress=False, auto_adjust=True)
        if data.empty:
            raise ValueError("No data returned from Yahoo Finance")
        data.reset_index(inplace=True)
        # Save to CSV with proper format
        data.to_csv(f'{quote}.csv', index=False)
        return True
    except Exception as e:
        print(f"Error fetching data: {e}")
        return False

# ------------------------------ Fundamentals & Ratios ------------------------------

def fetch_finnhub_basic_financials(ticker):
    """Fetches current metrics/ratios from Finnhub's company_basic_financials (metric + series)."""
    if not FINNHUB_API_KEY:
        return {}
    try:
        url = f"https://finnhub.io/api/v1/stock/metric?symbol={urlquote(ticker)}&metric=all&token={FINNHUB_API_KEY}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json() or {}
        return data  # dict with keys: metric, metricType, series, symbol
    except Exception as e:
        print(f"Finnhub fundamentals error: {e}")
        return {}

def fetch_yf_proxy_fundamentals(ticker):
    """Fallback: derive a few ratios from yfinance info and quarterly statements."""
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
        # Some fields can be missing; guard with get
        pe = info.get('trailingPE') or info.get('forwardPE')
        pb = info.get('priceToBook')
        dy = info.get('trailingAnnualDividendYield')
        mcap = info.get('marketCap')
        shares = info.get('sharesOutstanding')
        roe = info.get('returnOnEquity')
        profit_margin = info.get('profitMargins')

        # Try quarterly financials for leverage-style metrics
        bs = tk.quarterly_balance_sheet
        is_ = tk.quarterly_financials
        current_ratio = None
        d2e = None
        if bs is not None and not bs.empty:
            # Take most recent column
            col = bs.columns[0]
            curr_assets = bs.get('Total Current Assets')
            curr_liab = bs.get('Total Current Liabilities')
            total_liab = bs.get('Total Liab')
            eq = bs.get("Total Stockholder Equity")
            if curr_assets is not None and curr_liab is not None:
                try:
                    current_ratio = float(curr_assets[col]) / float(curr_liab[col])
                except Exception:
                    pass
            if total_liab is not None and eq is not None:
                try:
                    d2e = float(total_liab[col]) / float(eq[col]) if float(eq[col]) != 0 else None
                except Exception:
                    pass

        return {
            "proxy": True,
            "pe": pe,
            "pb": pb,
            "dividendYield": dy,
            "marketCap": mcap,
            "sharesOutstanding": shares,
            "roe": roe,
            "profitMargin": profit_margin,
            "currentRatio": current_ratio,
            "debtToEquity": d2e
        }
    except Exception as e:
        print(f"yfinance proxy fundamentals error: {e}")
        return {"proxy": True}

def build_ratio_features(price_df, ticker):
    """
    Returns a DataFrame indexed by Date with engineered features from fundamentals/ratios.
    Strategy:
    - Prefer Finnhub current metrics (static over short windows) and series.
    - Broadcast most recent current metrics across dates (forward-fill).
    - Use yfinance proxy ratios as fallback.
    - Create lagged/rolling signals and normalized versions aligned to price history.
    """
    idx = price_df.index
    feat = pd.DataFrame(index=idx)

    data = fetch_finnhub_basic_financials(ticker)
    used_source = "finnhub"
    metrics = {}
    series = {}

    if data and isinstance(data, dict) and data.get("metric"):
        metrics = data.get("metric", {}) or {}
        series = (data.get("series", {}) or {})
    else:
        used_source = "yfinance_proxy"
        metrics = fetch_yf_proxy_fundamentals(ticker)

    # Select a compact set of informative metrics
    # Names per Finnhub docs: 'peBasicExclExtraTTM', 'pbAnnual', 'roeTTM', 'netProfitMarginTTM', 'currentRatioAnnual', 'debtToEquityAnnual', 'dividendYieldIndicatedAnnual', etc.
    candidates = [
        # Finnhub names
        'peBasicExclExtraTTM', 'pbAnnual', 'roeTTM', 'netProfitMarginTTM',
        'currentRatioAnnual', 'debtToEquityAnnual', 'dividendYieldIndicatedAnnual',
        'fcfMarginTTM', 'operatingMarginTTM',
        # proxy names
        'pe', 'pb', 'roe', 'profitMargin', 'currentRatio', 'debtToEquity', 'dividendYield'
    ]

    values = {}
    for k in candidates:
        v = metrics.get(k)
        if v is None and used_source == "yfinance_proxy":
            v = metrics.get(k)  # same dict
        if v is not None:
            values[k] = v

    # Create base features by broadcasting constant metrics (these are "current" snapshot)
    for k, v in values.items():
        try:
            feat[k] = float(v)
        except Exception:
            feat[k] = np.nan

    # From series (historical quarterly/annual), if available, we can add last few values deltas
    # Finnhub series layout: {'annual': {'peBasicExclExtraTTM': [{'period': '2023-12-31','v': 24.1}, ...]}, 'quarterly': {...}}
    def add_series_feature(key, serobj):
        try:
            # pick quarterly series first if exists; else annual
            arr = []
            if 'quarterly' in serobj and key in serobj['quarterly']:
                arr = serobj['quarterly'][key]
            elif 'annual' in serobj and key in serobj['annual']:
                arr = serobj['annual'][key]
            if not arr:
                return None
            s = pd.Series(
                {pd.to_datetime(x['period']): x.get('v') for x in arr if x.get('period') and x.get('v') is not None}
            ).sort_index()
            # Reindex to price dates with forward-fill
            s_aligned = s.reindex(idx.union(s.index)).sort_index().ffill().reindex(idx)
            feat[f"{key}_series"] = s_aligned.astype(float)
            return True
        except Exception:
            return None

    if used_source == "finnhub" and isinstance(series, dict):
        for key in ['peBasicExclExtraTTM', 'pbAnnual', 'roeTTM', 'netProfitMarginTTM', 'currentRatioAnnual', 'debtToEquityAnnual']:
            add_series_feature(key, series)

    # Engineer transformations: lags, z-scores
    base_cols = list(feat.columns)
    for c in base_cols:
        try:
            feat[f"{c}_lag5"] = feat[c].shift(5)
            feat[f"{c}_z20"] = (feat[c] - feat[c].rolling(20, min_periods=5).mean()) / (feat[c].rolling(20, min_periods=5).std())
        except Exception:
            pass

    # Forward-fill + back-fill to avoid NaNs breaking scikit
    feat = feat.ffill().bfill()

    # Keep a compact set to avoid overfitting small sample
    keep = [c for c in feat.columns if any(x in c for x in ['pe', 'pb', 'roe', 'profitMargin', 'currentRatio', 'debtToEquity'])]  # include series/lags/z
    # Also keep dividendYield and a couple of margin features if present
    keep += [c for c in feat.columns if any(x in c for x in ['dividendYield', 'operatingMargin', 'fcfMargin'])]
    # Deduplicate and ensure exist
    keep = sorted(list({c for c in keep if c in feat.columns}))
    feat = feat[keep].copy()

    # Scale-friendly: replace infs
    feat.replace([np.inf, -np.inf], np.nan, inplace=True)
    feat = feat.ffill().bfill()

    # Record source
    feat.attrs['fundamentals_source'] = used_source
    return feat

# ------------------------------ Models ------------------------------

def ARIMA_ALGO(df, quote):
    try:
        # Ensure data is numeric
        df_close = df[['Close']].copy().dropna()
        df_close['Close'] = pd.to_numeric(df_close['Close'], errors='coerce')
        df_close = df_close.dropna()
        
        df_close = df_close.sort_index(ascending=True)
        train_size = int(len(df_close) * 0.8)
        train, test = df_close[:train_size], df_close[train_size:]
        model = ARIMA(train, order=(5,1,0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=len(test))
        full_model = ARIMA(df_close, order=(5,1,0)).fit()
        arima_pred = full_model.forecast(steps=1).iloc[0]
        fig = plt.figure(figsize=(7.2,4.8), dpi=65)
        plt.plot(test.index, test, label='Actual Price')
        plt.plot(test.index, forecast, label='Predicted Price', color='orange')
        plt.legend(loc='best')
        plt.title(f'ARIMA Forecast for {quote}')
        plt.savefig('static/ARIMA.png')
        plt.close(fig)
        error_arima = math.sqrt(mean_squared_error(test, forecast))
        print(f"ARIMA Prediction: {arima_pred:.2f}, RMSE: {error_arima:.4f}")
        return round(arima_pred, 2), round(error_arima, 4)
    except Exception as e:
        print(f"ARIMA Error: {e}")
        return 0, 0

def LSTM_ALGO(df, quote):
    try:
        # Ensure data is numeric
        dataset = df[['Close']].copy()
        dataset['Close'] = pd.to_numeric(dataset['Close'], errors='coerce')
        dataset = dataset.dropna().values
        
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(dataset)
        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size]
        X_train, y_train = [], []
        for i in range(60, len(train_data)):
            X_train.append(train_data[i-60:i, 0])
            y_train.append(train_data[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(60,1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=25, batch_size=32, verbose=0)
        test_data = scaled_data[train_size-60:]
        X_test = []
        for i in range(60, len(test_data)):
            X_test.append(test_data[i-60:i, 0])
        X_test = np.array(X_test).reshape((-1, 60, 1))
        predictions = model.predict(X_test, verbose=0)
        predictions = scaler.inverse_transform(predictions)
        actual = df['Close'].values[train_size:]
        fig = plt.figure(figsize=(7.2,4.8), dpi=65)
        plt.plot(actual, label='Actual Price')
        plt.plot(predictions, label='Predicted Price', color='orange')
        plt.legend(loc='best')
        plt.title(f'LSTM Forecast for {quote}')
        plt.savefig('static/LSTM.png')
        plt.close(fig)
        error_lstm = math.sqrt(mean_squared_error(actual, predictions))
        last_60 = scaled_data[-60:].reshape((1,60,1))
        forecast = model.predict(last_60, verbose=0)
        forecast_price = scaler.inverse_transform(forecast)
        print(f"LSTM Prediction: {forecast_price[0][0]:.2f}, RMSE: {error_lstm:.4f}")
        return round(forecast_price[0][0], 2), round(error_lstm, 4)
    except Exception as e:
        print(f"LSTM Error: {e}")
        return 0, 0

def LIN_REG_ALGO_MULTIVAR(df, quote, ratio_feat: pd.DataFrame):
    """
    Linear Regression using Close + engineered ratio features.
    Target: 7-day ahead Close (same as your earlier setup).
    """
    try:
        df_lr = df[['Close']].copy()
        # Ensure data is numeric
        df_lr['Close'] = pd.to_numeric(df_lr['Close'], errors='coerce')
        df_lr = df_lr.dropna()
        
        # Merge features aligned by Date index
        X_feat = ratio_feat.copy()
        # Combine with raw Close as a feature too (helps baseline)
        X_all = pd.concat([df_lr[['Close']], X_feat], axis=1)
        # Create 7-day ahead target
        y = df_lr['Close'].shift(-7)
        # Drop rows with NaNs in X or y
        data = pd.concat([X_all, y.rename('Prediction')], axis=1).dropna()
        X = data.drop(columns=['Prediction']).values
        y_arr = data['Prediction'].values
        train_size = int(len(X) * 0.8)
        if train_size < 30:
            # Fallback to your univariate if too small
            df_lr, lr_pred, forecast_set, mean_forecast, error_lr = LIN_REG_ALGO_UNIVARIATE(df, quote)
            return df_lr, lr_pred, forecast_set, mean_forecast, error_lr, ['Close']
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y_arr[:train_size], y_arr[train_size:]
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        error_lr = math.sqrt(mean_squared_error(y_test, y_pred))
        # Forecast next 7 days iteratively using current last row
        # For simplicity, we predict 7 days by repeating the last available feature row
        last_row = X_all.iloc[-1:].values
        forecast_set = []
        for i in range(7):
            pred = lr.predict(last_row)
            forecast_set.append(float(pred))
        fig = plt.figure(figsize=(7.2,4.8), dpi=65)
        plt.plot(y_test, label='Actual Price')
        plt.plot(y_pred, label='Predicted Price', color='orange')
        plt.legend(loc='best')
        plt.title(f'Linear Regression (Ratios) Forecast for {quote}')
        plt.savefig('static/LR.png')
        plt.close(fig)
        lr_pred = forecast_set
        mean_forecast = sum(forecast_set) / len(forecast_set)
        print(f"Linear Regression Prediction: {mean_forecast:.2f}, RMSE: {error_lr:.4f}")
        return df_lr, lr_pred, forecast_set, round(mean_forecast,2), round(error_lr,4), list(X_all.columns)
    except Exception as e:
        print(f"Linear Regression (multivar) Error: {e}")
        # fallback
        df_lr, lr_pred, forecast_set, mean_forecast, error_lr = LIN_REG_ALGO_UNIVARIATE(df, quote)
        return df_lr, lr_pred, forecast_set, mean_forecast, error_lr, ['Close']

def LIN_REG_ALGO_UNIVARIATE(df, quote):
    try:
        df_lr = df[['Close']].copy()
        # Ensure data is numeric
        df_lr['Close'] = pd.to_numeric(df_lr['Close'], errors='coerce')
        df_lr = df_lr.dropna()
        
        df_lr['Prediction'] = df_lr[['Close']].shift(-7)
        X = np.array(df_lr.drop(['Prediction'], axis=1))[:-7]
        y = np.array(df_lr['Prediction'])[:-7]
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        error_lr = math.sqrt(mean_squared_error(y_test, y_pred))
        forecast_set = []
        x_input = df_lr.drop(['Prediction'], axis=1).values[-7:]
        for i in range(7):
            pred = lr.predict([x_input[i]])
            forecast_set.append(float(pred))
        fig = plt.figure(figsize=(7.2,4.8), dpi=65)
        plt.plot(y_test, label='Actual Price')
        plt.plot(y_pred, label='Predicted Price', color='orange')
        plt.legend(loc='best')
        plt.title(f'Linear Regression Forecast for {quote}')
        plt.savefig('static/LR.png')
        plt.close(fig)
        lr_pred = forecast_set
        mean_forecast = sum(forecast_set) / len(forecast_set)
        print(f"Linear Regression (Univariate) Prediction: {mean_forecast:.2f}, RMSE: {error_lr:.4f}")
        return df_lr, lr_pred, forecast_set, round(mean_forecast,2), round(error_lr,4)
    except Exception as e:
        print(f"Linear Regression Error: {e}")
        return df, 0, [], 0, 0

# ------------------------------ Sentiment (News) ------------------------------

def fetch_news_google_rss(ticker, max_items=25, days_lookback=7):
    try:
        query = f"{ticker} stock OR shares OR earnings OR results"
        rss_url = f"https://news.google.com/rss/search?q={urlquote(query)}&hl=en-US&gl=US&ceid=US:en"
        resp = requests.get(rss_url, timeout=10)
        resp.raise_for_status()
        # Install lxml parser if not available: pip install lxml
        soup = BeautifulSoup(resp.content, 'lxml-xml')
        items = soup.find_all('item')
        news = []
        seen_titles = set()
        cutoff = datetime.now() - timedelta(days=days_lookback)
        for it in items:
            title = (it.title.text or "").strip()
            link = (it.link.text or "").strip()
            pub_date = it.pubDate.text if it.pubDate else ""
            source_tag = it.find('source')
            source = source_tag.text.strip() if source_tag else "Unknown"
            if not title or title in seen_titles:
                continue
            seen_titles.add(title)
            try:
                published = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %Z")
            except Exception:
                published = datetime.now()
            if published < cutoff:
                continue
            news.append({
                "title": title,
                "published": published,
                "source": source,
                "url": link
            })
            if len(news) >= max_items:
                break
        return news
    except Exception as e:
        print(f"News fetch error (RSS): {e}")
        return []

def fetch_news_api(ticker, max_items=25, days_lookback=7):
    if not NEWSAPI_KEY:
        return []
    try:
        from_date = (datetime.utcnow() - timedelta(days=days_lookback)).strftime("%Y-%m-%d")
        url = ("https://newsapi.org/v2/everything?"
               f"q={urlquote(ticker)}&from={from_date}&language=en&sortBy=publishedAt&pageSize={max_items}&apiKey={NEWSAPI_KEY}")
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        articles = data.get("articles", [])
        news = []
        seen_titles = set()
        for a in articles:
            title = (a.get("title") or "").strip()
            if not title or title in seen_titles:
                continue
            seen_titles.add(title)
            published_str = a.get("publishedAt", "")
            try:
                published = datetime.strptime(published_str, "%Y-%m-%dT%H:%M:%SZ")
            except Exception:
                published = datetime.utcnow()
            news.append({
                "title": title,
                "published": published,
                "source": (a.get("source") or {}).get("name") or "Unknown",
                "url": a.get("url") or ""
            })
        return news
    except Exception as e:
        print(f"News fetch error (NewsAPI): {e}")
        return []

def fetch_news(ticker, max_items=25, days_lookback=7):
    news = fetch_news_api(ticker, max_items=max_items, days_lookback=days_lookback)
    if not news:
        news = fetch_news_google_rss(ticker, max_items=max_items, days_lookback=days_lookback)
    return news

def score_sentiment(news_items):
    if not news_items:
        return 0.0, []
    sia = SentimentIntensityAnalyzer()
    detailed = []
    weights = []
    scores = []
    now = datetime.utcnow()
    for item in news_items:
        title = item.get("title", "")
        if not title:
            continue
        compound = sia.polarity_scores(title)["compound"]
        published = item.get("published", now)
        age_days = max(0.0, (now - published).total_seconds() / 86400.0)
        weight = 0.5 ** (age_days / 3.0)
        detailed.append({**item, "compound": compound})
        scores.append(compound * weight)
        weights.append(weight)
    agg = (sum(scores) / sum(weights)) if sum(weights) > 0 else 0.0
    return round(agg, 4), detailed

def plot_sentiment_bars(detailed, quote):
    if not detailed:
        return
    titles = [d["title"][:60] + ("..." if len(d["title"]) > 60 else "") for d in detailed][:15]
    vals = [d["compound"] for d in detailed][:15]
    fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
    plt.barh(range(len(vals)), vals, color=['#2ca02c' if v>0 else '#d62728' for v in vals])
    plt.yticks(range(len(vals)), titles, fontsize=8)
    plt.axvline(0, color='black', linewidth=0.8)
    plt.title(f'News Sentiment (VADER) for {quote}')
    plt.tight_layout()
    plt.savefig('static/SENTIMENT.png')
    plt.close(fig)

# ------------------------------ Recommendation ------------------------------

def recommending(mean, today_close, sentiment_score):
    base_buy = today_close < mean
    pos_thr = 0.2
    neg_thr = -0.2
    if sentiment_score >= pos_thr:
        decision = "BUY" if base_buy or (mean - today_close) * 0.5 > 0 else "HOLD"
        idea = "RISE"
    elif sentiment_score <= neg_thr:
        decision = "HOLD"
        idea = "FALL"
    else:
        if today_close < mean:
            idea, decision = "RISE", "BUY"
        else:
            idea, decision = "FALL", "HOLD"
    print(f"Recommendation: {idea} => {decision} (sentiment={sentiment_score:.3f})")
    return idea, decision

# ------------------------------ Route ------------------------------

@app.route('/insertintotable', methods=['POST'])
def insertintotable():
    nm = request.form['nm'].upper()
    if not nm:
        flash('Please enter a stock symbol', 'error')
        return redirect(url_for('index'))

    quote = nm
    try:
        if not get_historical(quote):
            flash('Stock symbol not found or data unavailable', 'error')
            return redirect(url_for('index'))

        # Read CSV data
        df = pd.read_csv(f'{quote}.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # Ensure all numeric columns are properly converted
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with NaN values
        df = df.dropna()
        
        if df.empty:
            flash('No data available for this stock', 'error')
            return redirect(url_for('index'))

        print("\n" + "="*80)
        print(f"Analyzing {quote} stock data")
        print("="*80)
        today = df.iloc[-1]
        today_close = float(today['Close'])

        # Build ratio features (auto-load)
        ratio_features = build_ratio_features(df, quote)
        fundamentals_source = ratio_features.attrs.get('fundamentals_source', 'unknown')

        # Run models
        arima_pred, error_arima = ARIMA_ALGO(df, quote)
        lstm_pred, error_lstm = LSTM_ALGO(df, quote)
        df_lr, lr_pred, forecast_set, mean_forecast, error_lr, lr_features = LIN_REG_ALGO_MULTIVAR(df, quote, ratio_features)

        # Fetch news + sentiment
        news_items = fetch_news(quote, max_items=25, days_lookback=7)
        agg_sentiment, detailed_sentiment = score_sentiment(news_items)
        plot_sentiment_bars(detailed_sentiment, quote)

        # Recommendation
        idea, decision = recommending(mean_forecast, today_close, agg_sentiment)

        print("\nAnalysis Complete")
        print("="*80)

        # Prepare data for template
        forecast_dates = [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1,8)]
        forecast_data = list(zip(forecast_dates, [round(float(p),2) for p in forecast_set]))
        today_data = {
            'open': round(float(today['Open']),2),
            'high': round(float(today['High']),2),
            'low': round(float(today['Low']),2),
            'close': round(float(today['Close']),2),
            'volume': f"{int(today['Volume']):,}"
        }

        top_headlines = [{
            'title': d['title'],
            'source': d['source'],
            'published': d['published'].strftime('%Y-%m-%d %H:%M'),
            'score': round(d['compound'], 3),
            'url': d['url']
        } for d in detailed_sentiment[:10]]

        # Select a few key ratio feature names to display
        display_ratio_cols = [c for c in lr_features if any(k in c for k in ['pe', 'pb', 'roe', 'profitMargin', 'currentRatio', 'debtToEquity', 'dividendYield'])]
        display_ratio_cols = display_ratio_cols[:10]
        latest_ratios = {c: round(float(ratio_features.iloc[-1][c]), 4) if c in ratio_features.columns else None for c in display_ratio_cols}

        return render_template('results.html',
                               quote=quote,
                               arima_pred=arima_pred,
                               lstm_pred=lstm_pred,
                               lr_pred=mean_forecast,
                               open_s=today_data['open'],
                               close_s=today_data['close'],
                               high_s=today_data['high'],
                               low_s=today_data['low'],
                               vol=today_data['volume'],
                               idea=idea,
                               decision=decision,
                               forecast_set=forecast_data,
                               error_arima=error_arima,
                               error_lstm=error_lstm,
                               error_lr=error_lr,
                               sentiment=agg_sentiment,
                               headlines=top_headlines,
                               fundamentals_source=fundamentals_source,
                               ratio_features_used=display_ratio_cols,
                               latest_ratios=latest_ratios)
    except Exception as e:
        print(f"Main Error: {e}")
        import traceback
        traceback.print_exc()
        flash('An error occurred during processing. Please try another stock.', 'error')
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)