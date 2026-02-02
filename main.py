# -*- coding: utf-8 -*-

"""
Hybrid XGBoost Model for Stock Return Prediction
=================================================

This script builds a complete pipeline for predicting stock returns (t+1)
using a hybrid XGBoost model. It integrates three sources of information:
1.  **Technical Indicators:** Standard metrics like RSI, MACD, etc.
2.  **Sentiment Analysis:** A simulated daily sentiment score for the asset.
3.  **Macroeconomic Data:** VIX index to gauge market volatility.

The pipeline covers data fetching, feature engineering, preprocessing, and
model training with time-series cross-validation.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
from lstm_predictor import generate_lstm_trend_feature

warnings.filterwarnings('ignore', category=FutureWarning)

# --- 1. CONFIGURATION ---
class Config:
    """Configuration class for the modeling pipeline."""
    # Data Parameters
    TICKER = "AAPL"
    VIX_TICKER = "^VIX"
    START_DATE = "2015-01-01"
    END_DATE = "2024-12-31"

    # Feature Engineering Parameters
    RSI_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    ATR_PERIOD = 14
    EMA_SHORT_PERIOD = 20
    EMA_LONG_PERIOD = 50
    SENTIMENT_MA_WINDOW = 72  # 3 days on hourly data simulation

    # Model Parameters
    XGB_PARAMS = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 3,
        'learning_rate': 0.01,
        'n_estimators': 500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'random_state': 42,
        'use_label_encoder': False
    }

    # Cross-validation Parameters
    N_SPLITS = 5

    # Feature List
    FEATURES_TO_SCALE = [
        'rsi', 'macd', 'macdh', 'macds', 'atr', 'ema_short', 'ema_long',
        'sentiment_ma', 'close_vix_ratio', 'lstm_trend_prediction'
    ]


# --- 2. DATA FETCHING ---
def fetch_data(ticker, vix_ticker, start, end):
    """
    Fetches OHLCV data for a stock and the VIX index.

    Args:
        ticker (str): The stock ticker symbol.
        vix_ticker (str): The VIX ticker symbol.
        start (str): Start date for data fetching.
        end (str): End date for data fetching.

    Returns:
        pd.DataFrame: A merged DataFrame containing stock and VIX data.
    """
    print(f"Fetching data for {ticker} and {vix_ticker} from {start} to {end}...")
    stock_data = yf.download(ticker, start=start, end=end, progress=False)
    vix_data = yf.download(vix_ticker, start=start, end=end, progress=False)

    # Flatten MultiIndex columns if they exist for both dataframes
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = stock_data.columns.get_level_values(0)
    if isinstance(vix_data.columns, pd.MultiIndex):
        vix_data.columns = vix_data.columns.get_level_values(0)

    # Rename all columns to be consistent (lowercase and underscores)
    stock_data.columns = [col.lower().replace(' ', '_') for col in stock_data.columns]
    vix_data.columns = [col.lower().replace(' ', '_') for col in vix_data.columns]

    # Select and rename VIX close column
    vix_data = vix_data[['close']].rename(columns={'close': 'vix_close'})

    # Merge dataframes
    df = stock_data.join(vix_data, how='left')
    df.index.name = 'date'
    return df

# --- 3. SENTIMENT ANALYSIS INTEGRATION (SIMULATED) ---
def simulate_sentiment(df):
    """
    Simulates a daily sentiment score.

    In a real-world scenario, this function would connect to an API like
    NewsAPI or Finnhub, pull news, and run sentiment analysis.

    Args:
        df (pd.DataFrame): The input DataFrame with a date index.

    Returns:
        pd.DataFrame: DataFrame with the added 'sentiment' column.
    """
    print("Simulating daily sentiment scores...")
    # Simulate a noisy sentiment signal that correlates slightly with returns
    np.random.seed(42)
    price_change = df['close'].pct_change().fillna(0)
    random_noise = np.random.normal(0, 0.05, len(df))
    sentiment = (price_change * 0.3 + random_noise)
    
    # Clip to be within [-1, 1]
    df['sentiment'] = np.clip(sentiment, -1, 1)
    return df

# --- 4. FEATURE ENGINEERING ---
def engineer_features(df):
    """
    Engineers technical, sentiment, and macro features.

    Args:
        df (pd.DataFrame): DataFrame with stock, VIX, and sentiment data.

    Returns:
        pd.DataFrame: DataFrame with all engineered features.
    """
    print("Engineering features...")
    # Technical Indicators using pandas_ta
    df.ta.rsi(length=Config.RSI_PERIOD, append=True, col_names=('rsi',))
    df.ta.macd(fast=Config.MACD_FAST, slow=Config.MACD_SLOW, signal=Config.MACD_SIGNAL, append=True, col_names=('macd', 'macdh', 'macds'))
    df.ta.atr(length=Config.ATR_PERIOD, append=True, col_names=('atr',))
    df.ta.ema(length=Config.EMA_SHORT_PERIOD, append=True, col_names=('ema_short',))
    df.ta.ema(length=Config.EMA_LONG_PERIOD, append=True, col_names=('ema_long',))
    
    # Sentiment Features
    df['sentiment_ma'] = df['sentiment'].rolling(
        window=Config.SENTIMENT_MA_WINDOW, min_periods=1
    ).mean()

    # Macro Features
    df['close_vix_ratio'] = df['close'] / df['vix_close']
    
    # Lagged features for sentiment
    for lag in [1, 2, 3]:
        df[f'sentiment_lag_{lag}'] = df['sentiment'].shift(lag)
        df[f'sentiment_ma_lag_{lag}'] = df['sentiment_ma'].shift(lag)

    print(f"Total features created: {len(df.columns)}")
    return df

# --- 5. TARGET DEFINITION ---
def define_target(df):
    """
    Defines the binary target variable (y).

    The target is 1 if the next day's return is positive, and 0 otherwise.

    Args:
        df (pd.DataFrame): DataFrame with features.

    Returns:
        pd.DataFrame: DataFrame with the 'target' column.
    """
    print("Defining target variable (t+1 return)...")
    df['return_t+1'] = df['close'].pct_change().shift(-1)
    df['target'] = (df['return_t+1'] > 0).astype(int)
    return df

# --- 6. PREPROCESSING ---
def preprocess_data(df):
    """
    Handles missing values and scales features.

    Args:
        df (pd.DataFrame): The complete DataFrame with features and target.

    Returns:
        pd.DataFrame: The preprocessed and cleaned DataFrame.
    """
    print("Preprocessing data (handling NaNs and scaling)...")
    
    # Drop rows with NaNs created by indicators and target shifting
    df.dropna(inplace=True)
    
    # Ensure all feature columns are numeric
    feature_cols = Config.FEATURES_TO_SCALE + [
        f'sentiment_lag_{lag}' for lag in [1, 2, 3]
    ] + [
        f'sentiment_ma_lag_{lag}' for lag in [1, 2, 3]
    ]
    
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Scaling
    scaler = RobustScaler()
    df[Config.FEATURES_TO_SCALE] = scaler.fit_transform(df[Config.FEATURES_TO_SCALE])
    
    print(f"Data shape after preprocessing: {df.shape}")
    return df

# --- 7. MODEL TRAINING & EVALUATION ---
def train_and_evaluate(df):
    """
    Trains and evaluates the XGBoost model using time-series cross-validation.

    Args:
        df (pd.DataFrame): The preprocessed DataFrame.
    """
    print("\n--- Starting Model Training and Evaluation ---")
    
    features = Config.FEATURES_TO_SCALE + [
        f'sentiment_lag_{lag}' for lag in [1, 2, 3]
    ] + [
        f'sentiment_ma_lag_{lag}' for lag in [1, 2, 3]
    ]
    target = 'target'
    
    X = df[features]
    y = df[target]

    tscv = TimeSeriesSplit(n_splits=Config.N_SPLITS)
    
    fold_results = []
    
    for i, (train_index, val_index) in enumerate(tscv.split(X)):
        print(f"\n--- Fold {i+1}/{Config.N_SPLITS} ---")
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        print(f"Train period: {X_train.index.min().date()} to {X_train.index.max().date()}")
        print(f"Validation period: {X_val.index.min().date()} to {X_val.index.max().date()}")

        model = xgb.XGBClassifier(**Config.XGB_PARAMS)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )

        preds = model.predict(X_val)
        proba_preds = model.predict_proba(X_val)[:, 1]

        auc = roc_auc_score(y_val, proba_preds)
        accuracy = accuracy_score(y_val, preds)
        
        fold_results.append({'fold': i+1, 'auc': auc, 'accuracy': accuracy})
        print(f"Fold {i+1} -> AUC: {auc:.4f}, Accuracy: {accuracy:.4f}")

    # Display final results
    results_df = pd.DataFrame(fold_results)
    print("\n--- Cross-Validation Summary ---")
    print(results_df)
    print("\n--- Average Performance ---")
    print(f"Average AUC: {results_df['auc'].mean():.4f} (+/- {results_df['auc'].std():.4f})")
    print(f"Average Accuracy: {results_df['accuracy'].mean():.4f} (+/- {results_df['accuracy'].std():.4f})")

    # Feature Importance
    try:
        final_model = xgb.XGBClassifier(**Config.XGB_PARAMS)
        final_model.fit(X, y)
        
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': final_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n--- Top 10 Feature Importances (from final model) ---")
        print(importance_df.head(10))
    except Exception as e:
        print(f"\nCould not generate feature importances: {e}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("🚀 Starting Hybrid XGBoost Pipeline...")
    
    # 1. Fetch data
    main_df = fetch_data(
        Config.TICKER, Config.VIX_TICKER, Config.START_DATE, Config.END_DATE
    )
    
    if not main_df.empty:
        # 2. Generate LSTM trend prediction feature
        # This is done before other features to use the raw price data
        main_df = generate_lstm_trend_feature(main_df.copy(), Config.TICKER)

        # 3. Simulate sentiment
        main_df = simulate_sentiment(main_df)
        
        # 4. Engineer other features
        main_df = engineer_features(main_df)
        
        # 5. Define target
        main_df = define_target(main_df)
        
        # 6. Preprocess data
        processed_df = preprocess_data(main_df.copy())
        
        # 7. Train and evaluate the final XGBoost model
        if not processed_df.empty:
            print("\n👑 --- Iniciando Módulo XGBoost (Director de Decisiones) --- 👑")
            train_and_evaluate(processed_df)
        else:
            print("❌ Error: No data left after preprocessing. Check data quality and date ranges.")
    else:
        print("❌ Error: Failed to fetch initial data. Check tickers and network connection.")
        
    print("\n✅ Pipeline finished.")
