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
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
from datetime import datetime, timedelta
from lstm_predictor import generate_lstm_regime_features
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings('ignore', category=FutureWarning)

# --- 1. CONFIGURATION ---
class Config:
    """Configuration class for the modeling pipeline."""
    # Data Parameters
    TICKER = "AAPL"
    VIX_TICKER = "^VIX"
    START_DATE = None  # Will be set dynamically
    END_DATE = None
    TRAINING_YEARS = 4 # Number of years to use for training data

    # Feature Engineering Parameters
    RSI_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    ATR_PERIOD = 14
    EMA_SHORT_PERIOD = 20
    EMA_LONG_PERIOD = 50

    # Model Parameters (optimized for small dataset)
    XGB_PARAMS = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 5,  # Increased for more complex patterns
        'learning_rate': 0.05,  # Faster learning
        'n_estimators': 300,  # Fewer trees to avoid overfitting
        'subsample': 0.7,  # More aggressive sampling
        'colsample_bytree': 0.7,
        'min_child_weight': 3,  # Prevent overfitting on small samples
        'gamma': 0.2,  # More conservative splits
        'reg_alpha': 0.1,  # L1 regularization
        'reg_lambda': 1.0,  # L2 regularization
        'random_state': 42,
        'use_label_encoder': False,
        'early_stopping_rounds': 30
    }

    # Cross-validation Parameters
    N_SPLITS = 5
    MONTHLY_MIN_TRAIN_MONTHS = 12  # 1 year minimum for training
    MAX_MONTHLY_FOLDS = 12  # Evaluate last 12 months
    MIN_SAMPLES_PER_FOLD = 5  # Skip folds with fewer samples

    # Feature List (LSTM + TOP technical indicators + LSTM interactions)
    # Reduced to most predictive features to avoid overfitting
    FEATURES_TO_SCALE = [
        # Momentum (most important)
        'rsi', 'macd', 'macdh', 'macds', 'roc_20',
        # Volatility
        'atr', 'volatility_20',
        # Trend
        'ema_short', 'ema_long', 'price_to_sma20', 'price_to_sma50', 'ADX_14', 'DMP_14', 'DMN_14',
        # Volume
        'volume_ratio', 'mfi',
        # Price patterns
        'high_low_ratio', 'close_open_ratio', 'price_to_52w_high',
        # Market
        'close_vix_ratio', 'vix_change',
        # LSTM (strongest signals)
        'lstm_price_5d', 'lstm_price_10d', 'lstm_price_20d',
        # LSTM interactions (new)
        'lstm_momentum', 'lstm_vs_ema', 'lstm_vs_rsi'
    ]


# --- 2. DATA FETCHING ---
def fetch_data(ticker, vix_ticker, start, end):
    """
    Fetches OHLCV data for a stock and the VIX index.

    Args:
        ticker (str): The stock ticker symbol.
        vix_ticker (str): The VIX ticker symbol.
        start (str): Start date for data fetching.
        end (str): End date for data fetching. If None, uses today's date.

    Returns:
        pd.DataFrame: A merged DataFrame containing stock and VIX data.
    """
    if end is None:
        end = pd.Timestamp.today().date().isoformat()
    print(f"Fetching data for {ticker} and {vix_ticker} from {start} to {end}...")
    
    try:
        # Use yfinance with timeout and retry settings
        stock_data = yf.download(
            ticker, 
            start=start, 
            end=end, 
            progress=False,
            timeout=30
        )
        vix_data = yf.download(
            vix_ticker, 
            start=start, 
            end=end, 
            progress=False,
            timeout=30
        )

        # Flatten MultiIndex columns if they exist
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = stock_data.columns.get_level_values(0)
        if isinstance(vix_data.columns, pd.MultiIndex):
            vix_data.columns = vix_data.columns.get_level_values(0)

        # Rename columns to lowercase with underscores
        stock_data.columns = [col.lower().replace(' ', '_') for col in stock_data.columns]
        vix_data.columns = [col.lower().replace(' ', '_') for col in vix_data.columns]

        # Select and rename VIX close column
        vix_data = vix_data[['close']].rename(columns={'close': 'vix_close'})

        # Merge dataframes
        df = stock_data.join(vix_data, how='left')
        df.index.name = 'date'
        
        print(f"✅ Datos obtenidos: {len(df)} registros")
        return df
        
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        raise

# --- 3. FEATURE ENGINEERING ---
def engineer_features(df):
    """
    Engineers advanced technical, sentiment, and macro features.

    Args:
        df (pd.DataFrame): DataFrame with stock, VIX, and sentiment data.

    Returns:
        pd.DataFrame: DataFrame with all engineered features.
    """
    print("Engineering features...")
    
    # === MOMENTUM INDICATORS ===
    df.ta.rsi(length=Config.RSI_PERIOD, append=True, col_names=('rsi',))
    df.ta.rsi(length=14, append=True, col_names=('rsi_14',))
    
    df.ta.macd(fast=Config.MACD_FAST, slow=Config.MACD_SLOW, signal=Config.MACD_SIGNAL, append=True, col_names=('macd', 'macdh', 'macds'))
    
    # Stochastic (K%, D%) - momentum oscillator
    df.ta.stoch(k=14, d=3, smooth_k=3, append=True)
    
    # Rate of Change (momentum)
    df.ta.roc(length=10, append=True, col_names=('roc_10',))
    df.ta.roc(length=20, append=True, col_names=('roc_20',))
    
    # === VOLATILITY INDICATORS ===
    df.ta.atr(length=Config.ATR_PERIOD, append=True, col_names=('atr',))
    
    # Bollinger Bands (volatility bands)
    df.ta.bbands(length=20, std=2, append=True)
    if 'BBL_20_2.0' in df.columns:
        df.rename(columns={
            'BBL_20_2.0': 'bb_lower',
            'BBM_20_2.0': 'bb_mid',
            'BBU_20_2.0': 'bb_upper',
            'BBB_20_2.0': 'bb_width',
            'BBP_20_2.0': 'bb_percent'
        }, inplace=True)
    
    # Keltner Channels (volatility bands using ATR)
    df.ta.kc(length=20, scalar=2, append=True)
    
    # Standard Deviation (volatility)
    df['volatility_20'] = df['close'].rolling(window=20).std()
    df['volatility_5'] = df['close'].rolling(window=5).std()
    
    # === TREND INDICATORS ===
    df.ta.ema(length=Config.EMA_SHORT_PERIOD, append=True, col_names=('ema_short',))
    df.ta.ema(length=Config.EMA_LONG_PERIOD, append=True, col_names=('ema_long',))
    df.ta.sma(length=50, append=True, col_names=('sma_50',))
    
    # ADX (Average Directional Index) - trend strength
    df.ta.adx(length=14, append=True)
    
    # === VOLUME INDICATORS ===
    # OBV (On Balance Volume)
    df.ta.obv(append=True, col_names=('obv',))
    
    # Volume Rate of Change
    df.ta.roc(close=df['volume'], length=10, append=True, col_names=('volume_roc_10',))
    
    # Money Flow Index
    df.ta.mfi(length=14, append=True, col_names=('mfi',))
    
    # Volume normalized by 20-day MA
    df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma_20']
    
    # === PRICE PATTERNS ===
    # High-Low range
    df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
    
    # Close-Open ratio
    df['close_open_ratio'] = (df['close'] - df['open']) / df['open']
    
    # Price to SMA ratios
    df['price_to_sma20'] = df['close'] / df['close'].rolling(window=20).mean()
    df['price_to_sma50'] = df['close'] / df['close'].rolling(window=50).mean()
    
    # 52-week high/low
    df['high_52w'] = df['high'].rolling(window=252).max()
    df['low_52w'] = df['low'].rolling(window=252).min()
    df['price_to_52w_high'] = df['close'] / df['high_52w']
    df['price_to_52w_low'] = df['close'] / df['low_52w']
    
    # === MARKET BREADTH / MACRO ===
    df['close_vix_ratio'] = df['close'] / df['vix_close']
    df['vix_change'] = df['vix_close'].pct_change()
    
    # Log returns (smoother than simple returns)
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # === ACCUMULATION LINES ===
    # Money Flow (High-Low-Close index)
    df.ta.hlc3(append=True)
    
    print(f"Total features created: {len(df.columns)}")
    return df

# --- 4. TARGET DEFINITION ---
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

# --- 5. PREPROCESSING ---
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
    feature_cols = [c for c in Config.FEATURES_TO_SCALE if c in df.columns]
    if len(feature_cols) < len(Config.FEATURES_TO_SCALE):
        missing = [c for c in Config.FEATURES_TO_SCALE if c not in df.columns]
        print(f"⚠️ Missing features skipped: {missing}")
    
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Scaling
    scaler = RobustScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    print(f"Data shape after preprocessing: {df.shape}")
    return df

# --- 6. MODEL TRAINING & EVALUATION ---
def train_and_evaluate(df):
    """
    Trains and evaluates the XGBoost model using weekly walk-forward validation.

    Args:
        df (pd.DataFrame): The preprocessed DataFrame.
    """
    print("\n--- Starting Model Training and Evaluation ---")
    print("Evaluating by monthly periods with daily granularity.")

    features = [c for c in Config.FEATURES_TO_SCALE if c in df.columns]
    if len(features) < len(Config.FEATURES_TO_SCALE):
        missing = [c for c in Config.FEATURES_TO_SCALE if c not in df.columns]
        print(f"⚠️ Missing features skipped for training: {missing}")
    target = 'target'

    X = df[features]
    y = df[target]

    month_starts = X.index.to_period("M").to_timestamp()
    unique_months = pd.Index(month_starts.unique()).sort_values()

    if len(unique_months) <= Config.MONTHLY_MIN_TRAIN_MONTHS:
        print("❌ Error: Not enough monthly periods for walk-forward evaluation.")
        return

    months_to_eval = unique_months[Config.MONTHLY_MIN_TRAIN_MONTHS:]
    if Config.MAX_MONTHLY_FOLDS and len(months_to_eval) > Config.MAX_MONTHLY_FOLDS:
        months_to_eval = months_to_eval[-Config.MAX_MONTHLY_FOLDS:]

    fold_results = []
    skipped_folds = 0

    for i, month_start in enumerate(months_to_eval, start=1):
        val_mask = month_starts == month_start
        train_mask = month_starts < month_start

        X_train, X_val = X.loc[train_mask], X.loc[val_mask]
        y_train, y_val = y.loc[train_mask], y.loc[val_mask]

        if X_val.empty or X_train.empty or len(X_val) < Config.MIN_SAMPLES_PER_FOLD:
            skipped_folds += 1
            continue

        # Check if validation set has both classes
        if len(y_val.unique()) < 2:
            skipped_folds += 1
            continue

        month_end = (month_start + pd.offsets.MonthEnd(0)).date()

        fold_num = i - skipped_folds
        print(f"\n--- Fold {fold_num}/{len(months_to_eval) - skipped_folds} ---")
        print(f"Train period: {X_train.index.min().date()} to {X_train.index.max().date()}")
        print(f"Validation period: {month_start.date()} to {month_end} ({len(X_val)} samples)")

        model = xgb.XGBClassifier(**Config.XGB_PARAMS)

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        preds = model.predict(X_val)
        proba_preds = model.predict_proba(X_val)[:, 1]

        auc = roc_auc_score(y_val, proba_preds)
        accuracy = accuracy_score(y_val, preds)

        fold_results.append({'fold': fold_num, 'auc': auc, 'accuracy': accuracy})
        print(f"Fold {fold_num} -> AUC: {auc:.4f}, Accuracy: {accuracy:.4f}")

    # Display final results
    if skipped_folds > 0:
        print(f"\n⚠️  Skipped {skipped_folds} folds (insufficient samples or single class)")
    
    results_df = pd.DataFrame(fold_results)
    print("\n--- Cross-Validation Summary ---")
    print(results_df)
    print("\n--- Average Performance ---")
    if results_df.empty:
        print("❌ No valid folds to evaluate. Check data splits and target distribution.")
        return
    else:
        print(f"Average AUC: {results_df['auc'].mean():.4f} (+/- {results_df['auc'].std():.4f})")
        print(f"Average Accuracy: {results_df['accuracy'].mean():.4f} (+/- {results_df['accuracy'].std():.4f})")

    # Feature Importance
    try:
        final_params = dict(Config.XGB_PARAMS)
        final_params.pop('early_stopping_rounds', None)
        final_model = xgb.XGBClassifier(**final_params)
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
    print("🚀 Starting Main Pipeline...")
    
    # 1. Fetch data for the last N years
    start_date = (datetime.today() - timedelta(days=Config.TRAINING_YEARS * 365)).strftime('%Y-%m-%d')
    
    main_df = fetch_data(
        Config.TICKER, Config.VIX_TICKER, start_date, Config.END_DATE
    )
    
    if not main_df.empty:
        # 2. Generate LSTM regime features (before other features)
        main_df = generate_lstm_regime_features(main_df.copy(), Config.TICKER)

        # 3. Engineer other features
        main_df = engineer_features(main_df)
        
        # 4. Create LSTM interaction features (boost LSTM importance)
        if 'lstm_price_5d' in main_df.columns:
            main_df['lstm_momentum'] = (main_df['lstm_price_10d'] - main_df['lstm_price_5d']) / main_df['close']
            main_df['lstm_vs_ema'] = (main_df['lstm_price_10d'] - main_df['ema_short']) / main_df['close']
            main_df['lstm_vs_rsi'] = main_df['lstm_price_10d'] / main_df['close'] * main_df['rsi']
        
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

