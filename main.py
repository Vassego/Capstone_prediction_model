# IMPORT PACKAGES
from flask import Flask, render_template, request, flash, redirect, url_for
# Add these imports after your existing keras imports
from keras.layers import Dense, Dropout, LSTM, Bidirectional, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from sklearn.preprocessing import RobustScaler

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

FINNHUB_API_KEY ="d2i70c9r01qgfkrksi6gd2i70c9r01qgfkrksi70"
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
    start = end - timedelta(days=1440)  # 4 years
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
    """
    Advanced ARIMA algorithm with automatic order selection, walk-forward validation,
    and overfitting prevention using cross-validation and model order constraints.
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.stats.diagnostic import acorr_ljungbox
        from sklearn.metrics import mean_squared_error
        import pandas as pd
        import numpy as np
        import math
        import matplotlib.pyplot as plt

        # 1. DATA PREPROCESSING
        df_close = df[['Close']].copy().dropna()
        df_close['Close'] = pd.to_numeric(df_close['Close'], errors='coerce')
        df_close = df_close.dropna()
        df_close = df_close.sort_index(ascending=True)

        # Split data
        train_size = int(len(df_close) * 0.8)
        train, test = df_close[:train_size], df_close[train_size:]

        # 2. AUTOMATIC ORDER SELECTION WITH CONSTRAINTS (to avoid overly complex models)
        print("Searching for optimal ARIMA parameters with overfitting prevention...")
        best_aic = float('inf')
        best_order = (1, 1, 1)  # Default fallback

        # Limit p, d=1, q with moderate ranges to prevent overfitting
        max_p = 3
        max_q = 3

        for p in range(max_p + 1):
            for q in range(max_q + 1):
                if p == 0 and q == 0:
                    continue  # skip trivial model
                try:
                    model = ARIMA(train['Close'], order=(p, 1, q))
                    model_fit = model.fit()
                    aic = model_fit.aic
                    if aic < best_aic:
                        best_aic = aic
                        best_order = (p, 1, q)
                except Exception:
                    continue

        print(f"Auto-selected ARIMA order: {best_order} (AIC: {best_aic:.2f})")

        # 3. MODEL VALIDATION: Check residuals
        model = ARIMA(train['Close'], order=best_order)
        model_fit = model.fit()
        try:
            residuals = model_fit.resid
            lb_stat, lb_pvalue = acorr_ljungbox(residuals, lags=10, return_df=False)
            if any(p < 0.05 for p in lb_pvalue):
                print("Warning: Residuals show autocorrelation, model may need adjustment")
        except Exception as e:
            print(f"Residual diagnostic warning: {e}")

        # 4. WALK-FORWARD VALIDATION WITH OVERFITTING CHECK
        forecast_values = []
        forecast_indices = []
        history = train['Close'].tolist()

        for i in range(len(test)):
            try:
                temp_model = ARIMA(history, order=best_order)
                temp_fit = temp_model.fit()
                next_pred = temp_fit.forecast(steps=1)
                pred_value = next_pred.iloc[0] if hasattr(next_pred, 'iloc') else float(next_pred)
                forecast_values.append(pred_value)
                forecast_indices.append(test.index[i])
                history.append(test['Close'].iloc[i])
            except Exception as e:
                forecast_values.append(history[-1] if history else 0)
                forecast_indices.append(test.index[i])
                history.append(test['Close'].iloc[i])

        forecast = pd.Series(forecast_values, index=forecast_indices)

        # Overfitting detection: compare in-sample and out-of-sample errors
        try:
            in_sample_pred = model_fit.fittedvalues
            in_sample_error = math.sqrt(mean_squared_error(
                train['Close'].values[1:], in_sample_pred.values[1:]
            ))
            out_sample_error = math.sqrt(mean_squared_error(test['Close'].values, forecast.values))
            overfitting_ratio = out_sample_error / in_sample_error if in_sample_error > 0 else float('inf')
            if overfitting_ratio > 2.0:
                print(f"Warning: Potential overfitting detected (ratio: {overfitting_ratio:.2f})")
                # Apply model order adjustment to reduce complexity
                reduced_order = (max(1, best_order[0]-1), 1, max(1, best_order[2]-1))
                print(f"Trying reduced order to prevent overfitting: {reduced_order}")
                try:
                    model_reduced = ARIMA(df_close['Close'], order=reduced_order)
                    model_reduced_fit = model_reduced.fit()
                    next_prediction = model_reduced_fit.forecast(steps=1)
                    arima_pred = next_prediction.iloc[0] if hasattr(next_prediction, 'iloc') else float(next_prediction)
                    best_order = reduced_order
                except Exception as e:
                    print(f"Reduced order ARIMA failed: {e}")
                    arima_pred = forecast.iloc[-1] if len(forecast) > 0 else df_close['Close'].iloc[-1]
            else:
                next_prediction = model_fit.forecast(steps=1)
                arima_pred = next_prediction.iloc[0] if hasattr(next_prediction, 'iloc') else float(next_prediction)
        except Exception:
            arima_pred = forecast.iloc[-1] if len(forecast) > 0 else df_close['Close'].iloc[-1]

        # 5. VISUALIZATION
        fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
        plt.plot(test.index, test['Close'].values, label='Actual Price', linewidth=2)
        plt.plot(forecast.index, forecast.values, label='Predicted Price', color='orange', linewidth=2)
        plt.legend(loc='best')
        plt.title(f'ARIMA Forecast for {quote} (Order: {best_order})')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('static/ARIMA.png', bbox_inches='tight', dpi=100)
        plt.close(fig)

        # Calculate RMSE
        error_arima = math.sqrt(mean_squared_error(test['Close'].values, forecast.values))

        print(f"ARIMA (order={best_order}) Prediction: {arima_pred:.2f}, RMSE: {error_arima:.4f}")
        return round(arima_pred, 2), round(error_arima, 4)

    except Exception as e:
        print(f"ARIMA Error: {e}")
        import traceback
        traceback.print_exc()
        return 0, 0


def LSTM_ALGO(df, quote):
    try:
        from sklearn.preprocessing import MinMaxScaler, RobustScaler
        from keras.models import Sequential
        from keras.layers import LSTM, Dense, Dropout, BatchNormalization
        from keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from keras.optimizers import Adam
        from keras.regularizers import l2
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        
        # **1. SIMPLIFIED AND ROBUST DATA PREPROCESSING**
        # Use only essential, proven features for financial prediction
        dataset = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        
        # Ensure numeric data
        for col in dataset.columns:
            dataset[col] = pd.to_numeric(dataset[col], errors='coerce')
        dataset = dataset.dropna()
        
        # Add only essential technical indicators
        dataset['Returns'] = dataset['Close'].pct_change()
        dataset['SMA_5'] = dataset['Close'].rolling(window=5).mean()
        dataset['SMA_20'] = dataset['Close'].rolling(window=20).mean()
        dataset['Volatility'] = dataset['Returns'].rolling(window=10).std()
        
        # Price position indicators (more stable than complex oscillators)
        dataset['High_Low_Ratio'] = dataset['High'] / dataset['Low']
        dataset['Volume_MA'] = dataset['Volume'].rolling(window=10).mean()
        dataset['Volume_Ratio'] = dataset['Volume'] / dataset['Volume_MA']
        
        # Forward fill and drop remaining NaNs
        dataset = dataset.fillna(method='ffill').dropna()
        
        if len(dataset) < 100:
            raise ValueError("Insufficient data for LSTM training")
        
        # **2. PROPER SCALING STRATEGY**
        # Use single scaler for all features to maintain relationships
        feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 
                          'SMA_5', 'SMA_20', 'Volatility', 'High_Low_Ratio', 
                          'Volume_Ratio', 'Returns']
        
        # Filter only available columns
        available_features = [col for col in feature_columns if col in dataset.columns]
        feature_data = dataset[available_features].values
        
        # Use RobustScaler to handle outliers better
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(feature_data)
        
        # **3. OPTIMIZED SEQUENCE CREATION**
        lookback = min(60, len(scaled_data) // 4)  # Adaptive lookback window
        
        X, y = [], []
        close_idx = available_features.index('Close')
        
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i])
            y.append(scaled_data[i, close_idx])  # Predict next day's close
        
        X, y = np.array(X), np.array(y)
        
        # **4. PROPER TRAIN/VALIDATION/TEST SPLIT**
        train_size = int(len(X) * 0.7)
        val_size = int(len(X) * 0.15)
        
        X_train = X[:train_size]
        X_val = X[train_size:train_size + val_size]
        X_test = X[train_size + val_size:]
        
        y_train = y[:train_size]
        y_val = y[train_size:train_size + val_size]
        y_test = y[train_size + val_size:]
        
        if len(X_test) == 0:
            # If not enough data for separate test, use validation as test
            X_test, y_test = X_val, y_val
            X_val, y_val = X_train[-len(X_test):], y_train[-len(y_test):]
        
        # **5. SIMPLIFIED BUT EFFECTIVE MODEL ARCHITECTURE**
        model = Sequential([
            # Single LSTM layer with moderate complexity
            LSTM(64, return_sequences=True, input_shape=(lookback, len(available_features)),
                 kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001)),
            Dropout(0.2),
            BatchNormalization(),
            
            # Second LSTM layer
            LSTM(32, return_sequences=False, 
                 kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001)),
            Dropout(0.2),
            BatchNormalization(),
            
            # Dense layers with gradual size reduction
            Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.1),
            Dense(8, activation='relu'),
            Dense(1, activation='linear')  # Linear output for regression
        ])
        
        # **6. CONSERVATIVE OPTIMIZER SETTINGS**
        optimizer = Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999)
        
        model.compile(
            optimizer=optimizer,
            loss='mse',  # Use MSE for clearer error interpretation
            metrics=['mae']
        )
        
        # **7. ROBUST TRAINING WITH PROPER CALLBACKS**
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                min_delta=0.0001
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=0.00001,
                verbose=0
            )
        ]
        
        # **8. TRAINING WITH VALIDATION**
        history = model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=100,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1,
            shuffle=False  # Important: Don't shuffle time series data
        )
        
        # **9. PREDICTION AND ERROR CALCULATION**
        if len(X_test) > 0:
            predictions_scaled = model.predict(X_test, verbose=0).flatten()
            actual_scaled = y_test.flatten()
            
            # Convert back to original price scale
            # Create dummy array for inverse transform
            dummy_data = np.zeros((len(predictions_scaled), len(available_features)))
            dummy_data[:, close_idx] = predictions_scaled
            predictions_original = scaler.inverse_transform(dummy_data)[:, close_idx]
            
            dummy_data_actual = np.zeros((len(actual_scaled), len(available_features)))
            dummy_data_actual[:, close_idx] = actual_scaled
            actual_original = scaler.inverse_transform(dummy_data_actual)[:, close_idx]
            
            # Calculate RMSE in original price scale
            error_lstm = np.sqrt(np.mean((actual_original - predictions_original)**2))
            
        else:
            # Fallback if no test data
            error_lstm = 0
            predictions_original = []
            actual_original = []
        
        # **10. FUTURE PREDICTION**
        if len(scaled_data) >= lookback:
            last_sequence = scaled_data[-lookback:].reshape(1, lookback, len(available_features))
            future_pred_scaled = model.predict(last_sequence, verbose=0)[0, 0]
            
            # Convert to original scale
            dummy_future = np.zeros((1, len(available_features)))
            dummy_future[0, close_idx] = future_pred_scaled
            future_pred_original = scaler.inverse_transform(dummy_future)[0, close_idx]
            lstm_pred = float(future_pred_original)
        else:
            lstm_pred = float(dataset['Close'].iloc[-1])
        
        # **11. VISUALIZATION**
        # **11. VISUALIZATION**
        fig = plt.figure(figsize=(7.2, 4.8), dpi=65)  # Standardized size to match LR
        if len(actual_original) > 0:
            plt.plot(actual_original, label='Actual Price', linewidth=2, alpha=0.8)
            plt.plot(predictions_original, label='Predicted Price', color='orange', linewidth=2, alpha=0.8)
            plt.legend(loc='best')
            plt.title(f'Improved LSTM Forecast for {quote}')
            plt.xlabel('Time Steps')  # Added for consistency with LR
            plt.ylabel('Price ($)')
            plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('static/LSTM.png', bbox_inches='tight', dpi=100)
        plt.close(fig)

        
        # **12. COMPREHENSIVE LOGGING**
        print(f"\n{'='*60}")
        print(f"IMPROVED LSTM RESULTS FOR {quote}")
        print(f"{'='*60}")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Lookback window: {lookback}")
        print(f"Features used: {len(available_features)}")
        print(f"Training epochs: {len(history.history['loss'])}")
        print(f"Final training loss: {history.history['loss'][-1]:.6f}")
        print(f"Final validation loss: {history.history['val_loss'][-1]:.6f}")
        print(f"Next Day Prediction: ${lstm_pred:.2f}")
        print(f"RMSE: {error_lstm:.4f}")
        print(f"{'='*60}")
        
        return round(lstm_pred, 2), round(error_lstm, 4)
        
    except Exception as e:
        print(f"Improved LSTM Error: {e}")
        import traceback
        traceback.print_exc()
        return 0, 0

# Helper functions
def true_range(df):
    """Calculate True Range for ATR"""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    return np.maximum(high_low, np.maximum(high_close, low_close))

def calculate_rsi(prices, window=14):
    """Enhanced RSI calculation"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_macd(prices, fast=12, slow=26):
    """Enhanced MACD calculation"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    return macd.fillna(0)


def LIN_REG_ALGO_MULTIVAR(df, quote, ratio_feat: pd.DataFrame):
    """
    Enhanced Linear Regression with better error handling and data validation.
    """
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.feature_selection import SelectKBest, f_regression
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge, Lasso
        from sklearn.model_selection import cross_val_score
        
        # Start with basic features from original data
        df_features = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        
        # Ensure all data is numeric
        for col in df_features.columns:
            df_features[col] = pd.to_numeric(df_features[col], errors='coerce')
        df_features = df_features.dropna()
        
        # Only add technical indicators if we have enough data
        if len(df_features) > 50:
            # Add basic technical indicators with minimal lookback
            df_features['SMA_5'] = df_features['Close'].rolling(window=5, min_periods=1).mean()
            df_features['SMA_10'] = df_features['Close'].rolling(window=10, min_periods=1).mean()
            df_features['EMA_12'] = df_features['Close'].ewm(span=12, min_periods=1).mean()
            
            # Price-based features
            df_features['Price_Change'] = df_features['Close'].pct_change().fillna(0)
            df_features['Volatility_10d'] = df_features['Close'].rolling(window=10, min_periods=1).std().fillna(0)
            
            # Volume indicators
            df_features['Volume_SMA_5'] = df_features['Volume'].rolling(window=5, min_periods=1).mean()
            df_features['Volume_Ratio'] = (df_features['Volume'] / df_features['Volume_SMA_5']).fillna(1)
            
            # Simple RSI calculation
            delta = df_features['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / loss
            df_features['RSI'] = (100 - (100 / (1 + rs))).fillna(50)
            
            # Lagged features (minimal)
            df_features['Close_Lag_1'] = df_features['Close'].shift(1).fillna(df_features['Close'])
            df_features['Close_Lag_2'] = df_features['Close'].shift(2).fillna(df_features['Close'])
        
        # Merge with ratio features if available and valid
        if not ratio_feat.empty and len(ratio_feat) > 0:
            try:
                # Align indices more carefully
                common_idx = df_features.index.intersection(ratio_feat.index)
                if len(common_idx) > 20:  # Only merge if we have enough common data
                    df_features = df_features.loc[common_idx]
                    ratio_feat_aligned = ratio_feat.loc[common_idx]
                    
                    # Only add ratio features that aren't all NaN
                    valid_ratio_cols = []
                    for col in ratio_feat_aligned.columns:
                        if not ratio_feat_aligned[col].isna().all():
                            valid_ratio_cols.append(col)
                    
                    if valid_ratio_cols:
                        X_all = pd.concat([df_features, ratio_feat_aligned[valid_ratio_cols]], axis=1)
                    else:
                        X_all = df_features.copy()
                else:
                    X_all = df_features.copy()
            except Exception as e:
                print(f"Ratio feature merge failed: {e}")
                X_all = df_features.copy()
        else:
            X_all = df_features.copy()
        
        # Drop any remaining NaN values
        X_all = X_all.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Create target variable (7-day ahead Close price)
        target = X_all['Close'].shift(-7)
        
        # Prepare final dataset
        data = pd.concat([X_all, target.rename('Target')], axis=1).dropna()
        
        # Check if we have enough data
        if len(data) < 30:
            print("Insufficient data after feature engineering, falling back to univariate")
            df_lr, lr_pred, forecast_set, mean_forecast, error_lr = LIN_REG_ALGO_UNIVARIATE(df, quote)
            return df_lr, lr_pred, forecast_set, mean_forecast, error_lr, ['Close']
        
        # Separate features and target
        X = data.drop(['Target'], axis=1)
        y = data['Target']
        
        # Feature selection with safety check
        n_features = min(10, X.shape[1] - 1, len(X) // 5)  # Conservative feature selection
        if n_features < 1:
            n_features = min(5, X.shape[1])
        
        try:
            feature_selector = SelectKBest(f_regression, k=n_features)
            X_selected = feature_selector.fit_transform(X, y)
            selected_features = X.columns[feature_selector.get_support()].tolist()
        except Exception as e:
            print(f"Feature selection failed: {e}, using all features")
            X_selected = X.values
            selected_features = X.columns.tolist()
        
        # Scale features
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_selected)
        except Exception as e:
            print(f"Scaling failed: {e}, using unscaled features")
            X_scaled = X_selected
            scaler = None
        
        # Split data
        train_size = int(len(X_scaled) * 0.8)
        if train_size < 20:
            print("Insufficient training data, falling back to univariate")
            df_lr, lr_pred, forecast_set, mean_forecast, error_lr = LIN_REG_ALGO_UNIVARIATE(df, quote)
            return df_lr, lr_pred, forecast_set, mean_forecast, error_lr, ['Close']
        
        X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        
        # Simple model selection
        models = {
            'Linear': LinearRegression(),
            'Ridge': Ridge(alpha=1.0)
        }
        
        best_model = LinearRegression()
        best_model_name = "Linear"
        best_score = float('-inf')
        
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                score = model.score(X_train, y_train)
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_model_name = name
            except Exception as e:
                print(f"Model {name} failed: {e}")
                continue
        
        # Make predictions
        try:
            y_pred = best_model.predict(X_test)
            error_lr = math.sqrt(mean_squared_error(y_test, y_pred))
        except Exception as e:
            print(f"Prediction failed: {e}")
            df_lr, lr_pred, forecast_set, mean_forecast, error_lr = LIN_REG_ALGO_UNIVARIATE(df, quote)
            return df_lr, lr_pred, forecast_set, mean_forecast, error_lr, ['Close']
        
        # Forecasting for next 7 days
        forecast_set = []
        try:
            # Get last available features
            if scaler is not None:
                if hasattr(feature_selector, 'transform'):
                    last_features = feature_selector.transform(X.iloc[-1:])
                else:
                    last_features = X.iloc[-1:].values
                last_features_scaled = scaler.transform(last_features)
            else:
                last_features_scaled = X_selected[-1:] if len(X_selected) > 0 else X.iloc[-1:].values
            
            # Simple forecasting - repeat last prediction
            next_pred = best_model.predict(last_features_scaled)[0]
            forecast_set = [float(next_pred)] * 7  # Simple approach
            
        except Exception as e:
            print(f"Forecasting failed: {e}")
            # Fallback forecast
            recent_prices = df['Close'].tail(7).mean()
            forecast_set = [float(recent_prices)] * 7
        
        # Visualization
        # Visualization
        try:
            fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
            plt.plot(range(len(y_test)), y_test.values, label='Actual Price', linewidth=2, alpha=0.8)
            plt.plot(range(len(y_pred)), y_pred, label='Predicted Price', color='orange', linewidth=2, alpha=0.8)
            plt.legend(loc='best')  # Consistent location
            plt.title(f'Enhanced Linear Regression Forecast for {quote}\nModel: {best_model_name}')
            plt.xlabel('Time Steps')
            plt.ylabel('Price ($)')
            plt.grid(True, alpha=0.3)  # Consistent grid
            plt.tight_layout()
            plt.savefig('static/LR.png', bbox_inches='tight', dpi=100)
            plt.close(fig)
        except Exception as e:
            print(f"Plotting failed: {e}")

        
        # Calculate mean forecast
        mean_forecast = sum(forecast_set) / len(forecast_set) if forecast_set else 0
        
        print(f"Enhanced Linear Regression ({best_model_name}) Results:")
        print(f"Mean Forecast: {mean_forecast:.2f}, RMSE: {error_lr:.4f}")
        print(f"Features used: {len(selected_features)}")
        
        return (data, forecast_set, forecast_set, round(mean_forecast, 2), 
                round(error_lr, 4), selected_features)
        
    except Exception as e:
        print(f"Enhanced Linear Regression Error: {e}")
        import traceback
        traceback.print_exc()
        # Final fallback to univariate
        try:
            df_lr, lr_pred, forecast_set, mean_forecast, error_lr = LIN_REG_ALGO_UNIVARIATE(df, quote)
            return df_lr, lr_pred, forecast_set, mean_forecast, error_lr, ['Close']
        except Exception as fallback_error:
            print(f"Univariate fallback also failed: {fallback_error}")
            return df[['Close']], [0]*7, *7, 0, 0, ['Close']


# Helper functions (add these if not already present)
# def calculate_rsi(prices, window=14):
#     """Calculate Relative Strength Index"""
#     delta = prices.diff()
#     gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
#     loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
#     rs = gain / loss
#     rsi = 100 - (100 / (1 + rs))
#     return rsi.fillna(50)

# def calculate_macd(prices, fast=12, slow=26):
#     """Calculate MACD indicator"""
#     ema_fast = prices.ewm(span=fast).mean()
#     ema_slow = prices.ewm(span=slow).mean()
#     macd = ema_fast - ema_slow
#     return macd.fillna(0)



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