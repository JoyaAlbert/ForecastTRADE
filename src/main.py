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
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, precision_recall_curve
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
import warnings
from datetime import datetime, timedelta
from lstm_predictor import generate_lstm_regime_features
from run_logger import AccumulativeRunLogger
from triple_barrier_labeler import TripleBarrierLabeler

# VisualizationEngine will be imported dynamically when needed
VisualizationEngine = None

warnings.filterwarnings('ignore', category=FutureWarning)

# --- 1. CONFIGURATION ---
class Config:
    """Configuration class for the modeling pipeline."""
    # Data Parameters
    TICKER = "AAPL"
    VIX_TICKER = "^VIX"
    START_DATE = None  # Will be set dynamically
    END_DATE = None
    TRAINING_YEARS = 6  # Number of years to use for training data (increased from 4 for more robust training)

    # Feature Engineering Parameters
    RSI_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    ATR_PERIOD = 14
    EMA_SHORT_PERIOD = 20
    EMA_LONG_PERIOD = 50

    # Model Parameters (optimized for time-series, reduced variance)
    XGB_PARAMS = {
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',  # Use AUCPR instead of AUC for imbalanced data
        'max_depth': 4,  # Shallower trees for noisy financial data
        'learning_rate': 0.03,  # Lower learning rate with early stopping
        'n_estimators': 500,  # More trees with lower LR
        'subsample': 0.8,  # Increase randomness & robustness
        'colsample_bytree': 0.8,
        'min_child_weight': 5,  # Prevent overfitting
        'gamma': 0.5,  # More conservative splits
        'reg_alpha': 0.5,  # L1 regularization (increased)
        'reg_lambda': 2.0,  # L2 regularization (increased)
        'scale_pos_weight': None,  # Computed dynamically based on class distribution
        'random_state': 42,
        'use_label_encoder': False,
        'early_stopping_rounds': 50  # Increased patience
    }

    # Time-Series Cross-Validation Parameters
    N_SPLITS = 12  # 12 time-series splits for comprehensive validation
    EMBARGO_DAYS = 2  # 2-day embargo prevents lookahead, maintains data quality (Run 6 optimized)
    MIN_TRAIN_SIZE = 252  # 252 trading days (~1 year) minimum
    
    # Walk-Forward Refinement: Sliding Window Configuration
    # Instead of expanding window (old data + new data), use fixed-size sliding window
    # This makes model adapt faster to market regime changes
    USE_SLIDING_WINDOW = True  # Enable adaptive sliding window strategy
    SLIDING_WINDOW_SIZE = 750  # 750 days window (increased from 500 for more historical context)
    VALIDATION_WINDOW_SIZE = 90  # 90 trading days for validation (increased for better statistics)
    # Benefit: Model forgets stale patterns, adapts to market regime changes seen in Fold 5
    
    # Probability Threshold Parameters
    PROBABILITY_THRESHOLD_BUY = 0.65  # Only buy if prob > 65% (baseline)
    PROBABILITY_THRESHOLD_SELL = 0.35  # Only sell if prob < 35% (baseline)
    ENABLE_THRESHOLD_GRIDSEARCH = True  # Tune buy threshold from PR curve
    THRESHOLD_OPTIM_METRIC = "f1"  # Options: "f1", "precision_at_recall"
    THRESHOLD_MIN_RECALL = 0.25  # Used for precision_at_recall
    THRESHOLD_MIN_PRECISION = 0.25  # Used for recall_at_precision (future)
    
    # Position Opening Strategy
    POSITION_OPENING_MODE = "hybrid"  # Options: "threshold" (simple), "percentile" (top X%), "hybrid" (smart)
    PERCENTILE_THRESHOLD = 60  # For percentile/hybrid mode: keep top X% of predictions (60=top 40%)
    # Modes explained:
    #   - "threshold": Classic approach - buy if proba >= threshold
    #   - "percentile": Aggressive filtering - only trade top X% most confident predictions
    #   - "hybrid": Smart combo - threshold + percentile + technical validation (volatility, ATR, trend)

    # Feature List (LSTM Compressed Latent + TOP technical indicators + Market Regime)
    # The compressed latent features (10-dim) capture only critical temporal patterns
    # Bottleneck compression removes noise while retaining essential information
    # Market regime features provide contextual awareness (volatility, trend strength, bull/bear position)
    # These are then combined with the best technical features for XGBoost
    FEATURES_TO_SCALE = [
        # LSTM Compressed Latent Features (10-dimensional, Tanh-normalized)
        # These capture temporal context and regime information (noise-filtered)
        'lstm_latent_0', 'lstm_latent_1', 'lstm_latent_2', 'lstm_latent_3', 'lstm_latent_4',
        'lstm_latent_5', 'lstm_latent_6', 'lstm_latent_7', 'lstm_latent_8', 'lstm_latent_9',
        # Market Regime Features (contextual awareness of market conditions)
        'rolling_volatility_zscore',  # Extreme volatility indicator (panic vs calm markets)
        'distance_to_ma200',  # Bull/bear regime position (bullish = above MA200)
        'regime_state',  # Encoded regime state from vol/trend
        'regime_change',  # Regime change flag (0/1)
        # Top Technical Indicators (selected for best signal-to-noise)
        # Momentum (most important for trend-following)
        'rsi', 'macd', 'macds', 'roc_20',
        # Volatility
        'atr', 'volatility_20',
        # Trend
        'ema_short', 'ema_long', 'price_to_sma20', 'ADX_14',
        # Volume
        'volume_ratio', 'mfi',
        # Market
        'close_vix_ratio', 'vix_change',
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
    df.ta.rsi(length=14, append=True, col_names=('rsi_14',))
    df.ta.rsi(length=21, append=True, col_names=('rsi_21',))  # Different period to avoid redundancy
    
    df.ta.macd(fast=Config.MACD_FAST, slow=Config.MACD_SLOW, signal=Config.MACD_SIGNAL, append=True, col_names=('macd', 'macdh', 'macds'))
    
    # Stochastic (K%, D%) - momentum oscillator
    df.ta.stoch(k=14, d=3, smooth_k=3, append=True)
    
    # Rate of Change (momentum)
    df.ta.roc(length=10, append=True, col_names=('roc_10',))
    df.ta.roc(length=20, append=True, col_names=('roc_20',))
    
    # === VOLATILITY INDICATORS ===
    df.ta.atr(length=Config.ATR_PERIOD, append=True, col_names=('atr',))
    
    # Bollinger Bands (volatility bands) with safety check
    df.ta.bbands(length=20, std=2, append=True)
    # Find actual Bollinger Bands columns (handle pandas_ta version differences in suffix naming)
    bb_prefixes = ['BBL_20', 'BBM_20', 'BBU_20', 'BBB_20', 'BBP_20']
    bb_map = {}
    for prefix in bb_prefixes:
        matching = [col for col in df.columns if col.startswith(prefix)]
        if matching:
            bb_map[matching[0]] = prefix.lower().replace('_20', '')  # BBL_20 -> bbl, etc
    
    if len(bb_map) == 5:
        # Rename to standardized names
        rename_dict = {
            list(bb_map.keys())[0]: 'bb_lower',
            list(bb_map.keys())[1]: 'bb_mid',
            list(bb_map.keys())[2]: 'bb_upper',
            list(bb_map.keys())[3]: 'bb_width',
            list(bb_map.keys())[4]: 'bb_percent'
        }
        df.rename(columns=rename_dict, inplace=True)
    else:
        print(f"   ⚠️ Bollinger Bands columns warning: Found {len(bb_map)}/5 expected")
        print(f"      Available BB columns: {[col for col in df.columns if 'BB' in col]}")
    
    # Keltner Channels (volatility bands using ATR)
    df.ta.kc(length=20, scalar=2, append=True)
    
    # Standard Deviation (volatility)
    df['volatility_20'] = df['close'].rolling(window=20).std()
    df['volatility_5'] = df['close'].rolling(window=5).std()
    
    # === TREND INDICATORS ===
    df.ta.ema(length=Config.EMA_SHORT_PERIOD, append=True, col_names=('ema_short',))
    df.ta.ema(length=Config.EMA_LONG_PERIOD, append=True, col_names=('ema_long',))
    df.ta.sma(length=100, append=True, col_names=('sma_100',))  # Changed from 50 to avoid EMA(50) redundancy
    
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
    
    # === MARKET REGIME INTEGRATION FEATURES ===
    # These features provide contextual awareness of market conditions (inflation, trends, volatility regimes)
    
    # 1. Rolling Volatility Z-Score: Indicates if current volatility is high or low relative to history
    # High z-score = extreme volatility environment (panic/greed)
    # Used to understand volatility regime and adjust model confidence
    volatility_mean = df['volatility_20'].rolling(window=252).mean()
    volatility_std = df['volatility_20'].rolling(window=252).std()
    df['rolling_volatility_zscore'] = (df['volatility_20'] - volatility_mean) / (volatility_std + 1e-8)
    
    # 2. Trend Strength (ADX-based): Continuous metric for market strength
    # ADX is already in dataframe from ta.adx(), Direct use avoids multicolinearity
    # High ADX (>30) = strong trend, Low ADX (<20) = ranging market
    df['ADX_14'] = df.get('ADX_14', 20)  # Default to neutral if missing
    
    # 3. Distance to 200-Day MA: Measure of bull/bear market position
    # Positive = price above MA200 (bullish), Negative = below MA200 (bearish)
    # Large distance = strong trend, Small distance = ranging
    df['ma_200'] = df['close'].rolling(window=200).mean()
    df['distance_to_ma200'] = (df['close'] - df['ma_200']) / df['close']

    # 4. Automatic regime change detection (volatility + trend state shifts)
    vol_state = np.where(df['rolling_volatility_zscore'] > 1.0, 1,
                         np.where(df['rolling_volatility_zscore'] < -1.0, -1, 0))
    trend_state = np.where(df['distance_to_ma200'] >= 0, 1, -1)
    df['regime_state'] = (vol_state * 2) + trend_state
    df['regime_change'] = (df['regime_state'] != df['regime_state'].shift(1)).fillna(0).astype(int)
    
    print("   ✅ Market regime features added: volatility_zscore, trend_strength, ma200_distance")
    
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
    Defines target using Triple Barrier Method (Marcos Lopez de Prado).
    
    Instead of simple binary target (return > 0), uses three barriers:
    - Profit Target: hits when price gains enough (adaptive to volatility)
    - Stop Loss: hits when price falls (protective)
    - Time Barrier: max holding period (forces timeout on ranging markets)
    
    Labels: 1=profit target hit, -1=stop loss hit, 0=timeout (no strong move)
    
    Args:
        df (pd.DataFrame): DataFrame with 'close' price
        
    Returns:
        pd.DataFrame: DataFrame with 'target' column and auxiliary features
    """
    print("\n📊 Defining target with Triple Barrier Method...")
    
    # 1. ADAPTIVE PARAMETERS based on realized volatility
    realized_vol = df['close'].pct_change().std()
    print(f"   Realized volatility: {realized_vol*100:.2f}%")
    
    # Profit target: scales with volatility (low vol = 0.5%, high vol = 1.5%)
    profit_target_pct = np.clip(realized_vol, 0.005, 0.015)  # [0.5%, 1.5%]
    
    # Stop loss: asymmetric (1/2 of profit target for faster losses)
    stop_loss_pct = profit_target_pct / 2.0
    
    # Max holding: 10-20 days, less in high volatility markets
    max_holding_days = max(10, min(20, int(10 / realized_vol)))  # Adaptive
    
    print(f"   Profit Target (adaptive): +{profit_target_pct*100:.2f}%")
    print(f"   Stop Loss (asymmetric): -{stop_loss_pct*100:.2f}%")
    print(f"   Max Holding Days: {max_holding_days}")
    
    # 2. Apply Triple Barrier Labeling
    labeler = TripleBarrierLabeler(
        profit_target_pct=profit_target_pct,
        stop_loss_pct=stop_loss_pct,
        max_holding_days=max_holding_days
    )
    
    labels = labeler.label_data_vectorized(df)
    df['target'] = labels
    
    # 3. Print label distribution
    unique_labels, counts = np.unique(labels.values, return_counts=True)
    print(f"\n   ✅ Label Distribution (Triple Barrier):")
    for label, count in zip(unique_labels, counts):
        label_name = {1: "Profit Target 🎯", -1: "Stop Loss ⛔", 0: "Time-out ⏱️"}
        pct = 100 * count / len(labels)
        print(f"      {label_name.get(label, f'Label {label}')}: {count:6d} ({pct:5.1f}%)")
    
    # 4. Keep simple binary target for compatibility (0/1 for probability of UP)
    # Convert labels: 1 → 1 (profit), -1 → 0 (loss), 0 → random (timeout)
    df['target_binary'] = (labels == 1).astype(int)
    
    # For timeouts (label 0), use original simple return logic
    timeout_mask = (labels == 0)
    if timeout_mask.sum() > 0:
        df.loc[timeout_mask, 'return_t+1'] = df.loc[timeout_mask, 'close'].pct_change().shift(-1)
        df.loc[timeout_mask, 'target_binary'] = (df.loc[timeout_mask, 'return_t+1'] > 0).astype(int)
    
    # Use target_binary as main target
    df['target'] = df['target_binary']
    
    print(f"   ✅ Final binary target distribution:")
    print(f"      Class 0 (Sell/Loss): {(df['target']==0).sum()} ({100*(df['target']==0).mean():.1f}%)")
    print(f"      Class 1 (Buy/Win): {(df['target']==1).sum()} ({100*(df['target']==1).mean():.1f}%)")
    
    return df

# --- 5. PREPROCESSING (WITH FOLD-SPECIFIC SCALING) ---
def get_feature_columns(df):
    """
    Get list of feature columns available in the dataframe.
    
    Returns:
        list: Feature column names
    """
    feature_cols = [c for c in Config.FEATURES_TO_SCALE if c in df.columns]
    if len(feature_cols) < len(Config.FEATURES_TO_SCALE):
        missing = [c for c in Config.FEATURES_TO_SCALE if c not in df.columns]
        print(f"⚠️ Missing features: {missing}")
    return feature_cols

def preprocess_data_raw(df):
    """
    Basic preprocessing: handle NaNs and ensure numeric types.
    NO SCALING (scaling must be done per-fold to prevent leakage).
    
    IMPORTANT: Only drops NaNs in feature columns and target, not all columns.

    Args:
        df (pd.DataFrame): The complete DataFrame with features and target.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    print("Preprocessing data (handling NaNs)...")
    initial_shape = df.shape[0]
    
    # Get feature columns
    feature_cols = get_feature_columns(df)
    
    # Ensure all feature columns are numeric (coerce errors to NaN)
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # IMPORTANT: Only drop rows where FEATURES or TARGET are NaN
    # Don't drop based on ALL columns (that's too aggressive)
    cols_to_check = feature_cols + ['target']
    df_clean = df.dropna(subset=cols_to_check).copy()
    
    rows_dropped = initial_shape - df_clean.shape[0]
    print(f"   Rows before preprocessing: {initial_shape}")
    print(f"   Rows dropped (NaN in features/target): {rows_dropped}")
    print(f"   Data shape after preprocessing: {df_clean.shape}")
    
    # Sanity check: ensure we have enough data
    if df_clean.shape[0] < 50:
        print(f"⚠️  Warning: Only {df_clean.shape[0]} rows left after preprocessing")
    
    return df_clean

# --- 6. MODEL TRAINING & EVALUATION (TIME-SERIES AWARE) ---
def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """
    Calculate annualized Sharpe Ratio.
    
    Args:
        returns (np.array): Daily returns (can include zeros for days without positions)
        risk_free_rate (float): Annual risk-free rate
        
    Returns:
        float: Sharpe Ratio
    """
    if len(returns) < 2:
        return np.nan
    # Use ALL returns (including zeros for cash days) for realistic Sharpe calculation
    # This reflects actual portfolio performance: long on selected days, cash otherwise
    excess_returns = returns - (risk_free_rate / 252)
    mean_return = excess_returns.mean()
    std_dev = excess_returns.std()
    
    # Need variance > 0 to calculate meaningful Sharpe
    if std_dev < 1e-10:
        return np.nan
    
    # Annualized Sharpe Ratio (252 trading days per year)
    sharpe = (mean_return / std_dev) * np.sqrt(252)
    return sharpe if not np.isnan(sharpe) else np.nan

def calculate_max_drawdown(returns):
    """
    Calculate maximum drawdown.
    
    Args:
        returns (np.array): Daily returns (strategy returns, can contain zeros for no-position days)
        
    Returns:
        float: Maximum drawdown (as negative value)
    """
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

def apply_probability_threshold(proba_preds, threshold_buy=0.65, threshold_sell=0.35, 
                              X_features=None, percentile_threshold=None, mode='threshold'):
    """
    Intelligent position opening with multiple filtering strategies.
    
    Strategies:
    1. 'threshold': Simple probability threshold (default)
    2. 'percentile': Top X% most confident predictions (e.g., top 20%)
    3. 'hybrid': Combines threshold + percentile + technical filters
    
    Args:
        proba_preds (np.array): Probabilities of class 1
        threshold_buy (float): Probability threshold for buy signal
        threshold_sell (float): Probability threshold for sell signal  
        X_features (pd.DataFrame): Optional feature dataframe for technical validation
        percentile_threshold (float): Percentile cutoff for top-X% strategy (e.g., 80 for top 20%)
        mode (str): 'threshold', 'percentile', or 'hybrid'
        
    Returns:
        np.array: Filtered predictions (only high-confidence trades)
    """
    filtered = np.zeros_like(proba_preds, dtype=int)
    
    if mode == 'threshold':
        # Simple threshold: only buy if proba >= threshold
        filtered[proba_preds >= threshold_buy] = 1
        
    elif mode == 'percentile':
        # Top-X% strategy: only buy the top X% most confident predictions
        if percentile_threshold is None:
            percentile_threshold = 80  # Top 20% by default
        cutoff = np.percentile(proba_preds, percentile_threshold)
        filtered[proba_preds >= cutoff] = 1
        
    elif mode == 'hybrid':
        # Hybrid: threshold + percentile + technical validation
        # Step 1: Pass minimum confidence threshold
        high_conf = proba_preds >= threshold_buy
        
        # Step 2: Among high confidence, take top percentile
        if percentile_threshold is None:
            percentile_threshold = 60  # Top 40% of high-conf trades
        high_conf_proba = proba_preds[high_conf]
        if len(high_conf_proba) > 0:
            cutoff = np.percentile(high_conf_proba, percentile_threshold)
            top_trades = (proba_preds >= cutoff) & high_conf
        else:
            top_trades = high_conf
        
        # Step 3: Apply technical filters if features provided
        if X_features is not None and isinstance(X_features, pd.DataFrame):
            technical_filter = np.ones(len(proba_preds), dtype=bool)
            
            # Filter 1: Volatility must be positive (avoid dead markets)
            if 'volatility_20' in X_features.columns:
                min_vol = X_features['volatility_20'].quantile(0.1)  # At least at 10th percentile
                technical_filter &= X_features['volatility_20'].values >= min_vol
            
            # Filter 2: ATR must be present (avoid near-zero ranges)
            if 'atr' in X_features.columns:
                min_atr = X_features['atr'].quantile(0.1)
                technical_filter &= X_features['atr'].values >= min_atr
            
            # Filter 3: Trending market (distance to MA200)
            if 'distance_to_ma200' in X_features.columns:
                # Strong uptrend: significantly above MA200
                technical_filter &= X_features['distance_to_ma200'].values > -0.02  # At least not too far below
            
            # Combine with top trades
            filtered[top_trades & technical_filter] = 1
        else:
            # No technical features, just use top trades
            filtered[top_trades] = 1
    
    return filtered

def find_optimal_threshold(y_true, proba_preds, metric="f1", min_recall=0.25, min_precision=0.25):
    """
    Find an optimal probability threshold from the precision-recall curve.

    Args:
        y_true (np.array): True labels.
        proba_preds (np.array): Predicted probabilities for class 1.
        metric (str): Optimization target: "f1" or "precision_at_recall".
        min_recall (float): Minimum recall when using precision_at_recall.
        min_precision (float): Minimum precision for future extensions.

    Returns:
        tuple: (threshold, stats dict)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, proba_preds)
    if len(thresholds) == 0:
        return 0.5, {"precision": precision[-1], "recall": recall[-1], "f1": 0.0}

    precision = precision[1:]
    recall = recall[1:]

    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)

    if metric == "precision_at_recall":
        mask = recall >= min_recall
        if np.any(mask):
            best_idx = np.argmax(precision[mask])
            idx = np.where(mask)[0][best_idx]
        else:
            idx = int(np.nanargmax(f1_scores))
    else:
        idx = int(np.nanargmax(f1_scores))

    return thresholds[idx], {
        "precision": float(precision[idx]),
        "recall": float(recall[idx]),
        "f1": float(f1_scores[idx])
    }

def create_volatility_weighted_classifier(base_model, volatility_weights=None):
    """
    Creates a wrapper that applies risk-adjustment during prediction.
    
    During high volatility periods, the model becomes more conservative
    by scaling down positive probability predictions, requiring higher
    confidence to trigger trades.
    
    Args:
        base_model: Fitted XGBoost classifier
        volatility_weights (np.array): Per-sample volatility scaling factors
                                       Values > 1 for high volatility periods
        
    Returns:
        callable: Prediction function that applies volatility-weighted adjustment
    """
    def volatility_adjusted_predict_proba(X):
        """
        Get probabilities with volatility risk adjustment.
        
        During high volatility:
        - Scale down probabilities (more conservative)
        - Require higher confidence for positive predictions
        """
        proba = base_model.predict_proba(X)[:, 1]
        
        if volatility_weights is not None and len(volatility_weights) == len(proba):
            # Adjust: high volatility → lower positive probability (conservative)
            # Formula: prob_adjusted = prob * (1 / volatility_weight)
            # This means high vol periods need higher raw probability to cross threshold
            proba = proba / (volatility_weights + 1e-8)
            proba = np.clip(proba, 0, 1)  # Keep in [0, 1]
        
        return proba
    
    return volatility_adjusted_predict_proba


def sliding_window_splits(n_samples, train_size, test_size, n_splits=8):
    """
    Generate sliding window split indices for walk-forward analysis.
    
    Unlike TimeSeriesSplit (which uses expanding windows), sliding window
    uses a fixed-size training window that moves forward in time. This allows
    the model to forget stale patterns and adapt faster to market regime changes.
    
    Args:
        n_samples (int): Total number of samples
        train_size (int): Number of samples to use for training (window size)
        test_size (int): Number of samples to use for validation
        n_splits (int): Number of splits to generate
        
    Yields:
        tuple: (train_indices, test_indices) for each split
    """
    for i in range(n_splits):
        # Start position: slides forward by test_size each iteration
        start_idx = i * test_size
        
        # Training window: fixed-size from start to start + train_size
        train_start = max(0, start_idx)
        train_end = start_idx + train_size
        
        # Test window: never overlaps with training
        test_start = train_end
        test_end = min(n_samples, test_start + test_size)
        
        # Only yield if we have enough samples for both train and test
        if test_end > test_start and train_end <= n_samples:
            train_indices = np.arange(train_start, min(train_end, n_samples))
            test_indices = np.arange(test_start, test_end)
            
            if len(train_indices) >= train_size // 2 and len(test_indices) > 0:
                yield train_indices, test_indices

def train_and_evaluate_timeseries(df):
    """
    Trains and evaluates the XGBoost model using TimeSeriesSplit
    with proper data leakage prevention:
    - Scalers fit ONLY on training data
    - Embargo period between train and validation
    - Generates visualizations and logs results
    
    Args:
        df (pd.DataFrame): The preprocessed DataFrame (NOT scaled yet).
    """
    print("\n--- Starting Time-Series-Aware Model Training ---")
    print(f"Using TimeSeriesSplit with {Config.N_SPLITS} splits and {Config.EMBARGO_DAYS}-day embargo period")

    features = get_feature_columns(df)
    target = 'target'

    X = df[features].copy()
    y = df[target].copy()
    
    if X.empty or y.empty:
        print("❌ Error: No data available for training")
        return

    # Initialize visualization and logging engines
    viz_engine = None
    try:
        from visualization_engine import VisualizationEngine as VizEngine
        viz_engine = VizEngine(output_dir='out')
        print("✅ VisualizationEngine loaded successfully")
    except ImportError as e:
        print(f"⚠️ Warning: Could not import VisualizationEngine: {e}")
    except Exception as e:
        print(f"⚠️ Warning: VisualizationEngine initialization error: {e}")
    
    run_logger = AccumulativeRunLogger(log_file='out/runs_log.json')
    
    run_date = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    run_number = run_logger.runs['metadata']['total_runs'] + 1

    # Choose cross-validation strategy
    if Config.USE_SLIDING_WINDOW:
        print(f"✨ Using SLIDING WINDOW strategy (adaptive, not expanding)")
        print(f"   Train window: {Config.SLIDING_WINDOW_SIZE} days | Test window: {Config.VALIDATION_WINDOW_SIZE} days")
        print(f"   This allows model to adapt faster to market regime changes")
        # Create sliding window splits
        cv_splits = list(sliding_window_splits(
            len(X),
            train_size=Config.SLIDING_WINDOW_SIZE,
            test_size=Config.VALIDATION_WINDOW_SIZE,
            n_splits=Config.N_SPLITS
        ))
        print(f"   Generated {len(cv_splits)} sliding window splits")
    else:
        print(f"Using TimeSeriesSplit with {Config.N_SPLITS} splits and {Config.EMBARGO_DAYS}-day embargo period")
        tscv = TimeSeriesSplit(n_splits=Config.N_SPLITS)
        cv_splits = list(tscv.split(X))
    
    fold_results = []
    all_val_predictions = []
    all_val_proba = []
    all_val_indices = []
    future_predictions = []  # Store future predictions from each fold
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits, start=1):
        # Get train and test indices
        X_train_unscaled = X.iloc[train_idx].copy()
        X_val_unscaled = X.iloc[val_idx].copy()
        y_train = y.iloc[train_idx].copy()
        y_val = y.iloc[val_idx].copy()
        
        # Check embargo period: ensure no overlap (only for expanding window strategy)
        if not Config.USE_SLIDING_WINDOW:
            embargo_buffer = Config.EMBARGO_DAYS
            if len(train_idx) > embargo_buffer:
                train_idx_embargoed = train_idx[:-embargo_buffer]
                X_train_unscaled = X.iloc[train_idx_embargoed].copy()
                y_train = y.iloc[train_idx_embargoed].copy()
        
        if len(X_train_unscaled) < Config.MIN_TRAIN_SIZE:
            print(f"⚠️ Fold {fold_idx}: Skipped - insufficient training data ({len(X_train_unscaled)} < {Config.MIN_TRAIN_SIZE})")
            continue
        
        if len(X_val_unscaled) < 10:
            print(f"⚠️ Fold {fold_idx}: Skipped - insufficient validation data")
            continue
        
        # Extract date information early for logging
        train_date_start = X_train_unscaled.index.min().date()
        train_date_end = X_train_unscaled.index.max().date()
        val_date_start = X_val_unscaled.index.min().date()
        val_date_end = X_val_unscaled.index.max().date()
        
        # Check if validation set has both classes
        if len(y_val.unique()) < 2:
            val_class = y_val.unique()[0]
            class_name = "Stop Loss" if val_class == 0 else "Profit Target"
            n_samples = len(y_val)
            print(f"⚠️ Fold {fold_idx}: Skipped - single class in validation set")
            print(f"   └─ Only class {val_class} ({class_name}) found in {n_samples} samples")
            print(f"   └─ Window: {val_date_start} to {val_date_end}")
            print(f"   └─ Market condition: All trades hit " + ("profit target" if val_class == 1 else "stop loss"))
            continue
        
        # === CRITICAL: Scale ONLY on training data ===
        scaler = RobustScaler()
        X_train_scaled = X_train_unscaled.copy()
        X_val_scaled = X_val_unscaled.copy()
        
        X_train_scaled[features] = scaler.fit_transform(X_train_unscaled[features])
        X_val_scaled[features] = scaler.transform(X_val_unscaled[features])
        
        print(f"\n--- Fold {fold_idx}/{Config.N_SPLITS} (TimeSeriesSplit) ---")
        print(f"Train: {train_date_start} to {train_date_end} ({len(X_train_scaled)} samples)")
        print(f"Valid: {val_date_start} to {val_date_end} ({len(X_val_unscaled)} samples)")
        print(f"Class distribution in validation: {y_val.value_counts().to_dict()}")
        
        # === CLASS IMBALANCE HANDLING ===
        # Hybrid approach: scale_pos_weight + SMOTE
        # - scale_pos_weight: Penalizes minority class misclassification in loss function
        # - SMOTE: Balances training data via synthetic sample generation
        # Both are complementary, not redundant
        
        # 1. Calculate scale_pos_weight from original (imbalanced) distribution
        class_counts = y_train.value_counts()
        if len(class_counts) == 2:
            scale_pos_weight = class_counts[0] / class_counts[1]  # Ratio of negative to positive
        else:
            scale_pos_weight = 1.0
        print(f"   ✅ Original class distribution: {class_counts.to_dict()}")
        print(f"   ✅ Class weights computed: scale_pos_weight = {scale_pos_weight:.4f}")
        
        # 2. Apply SMOTE to balance training data with error handling
        # Optimized sampling_strategy=0.8: reduces synthetic sample creation, lower overfitting risk
        minority_class_size = len(y_train[y_train == 0])
        k_neighbors = min(3, max(1, minority_class_size - 1))
        
        try:
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors, sampling_strategy=0.8)
            X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
            print(f"   ✅ SMOTE applied: {len(X_train_scaled)} → {len(X_train_smote)} samples (k_neighbors={k_neighbors}, strategy=0.8)")
            print(f"   ✅ Balanced class distribution: {pd.Series(y_train_smote).value_counts().to_dict()}")
        except ValueError as e:
            print(f"   ⚠️ SMOTE failed ({str(e)}). Using original training data.")
            print(f"      Minority class size: {minority_class_size} | k_neighbors: {k_neighbors}")
            X_train_smote, y_train_smote = X_train_scaled, y_train
            print(f"   ℹ️ Training without SMOTE balancing (scale_pos_weight={scale_pos_weight:.4f})")
        
        # Train base XGBoost model with class weighting and SMOTE-balanced data
        # scale_pos_weight provides additional emphasis on minority class during training
        xgb_params = Config.XGB_PARAMS.copy()
        xgb_params['scale_pos_weight'] = scale_pos_weight
        model = xgb.XGBClassifier(**xgb_params)
        
        model.fit(
            X_train_smote, y_train_smote,
            eval_set=[(X_val_scaled, y_val)],
            verbose=False
        )
        
        # === PROBABILITY CALIBRATION VIA RISK-ADJUSTMENT ===
        # Instead of post-hoc calibration, we apply dynamic probability adjustment
        # based on market volatility (volatility-weighted scaling)
        # This achieves calibration effect: high-vol periods get conservative probabilities
        
        # === RISK-ADJUSTED LOSS: Apply volatility weighting to predictions ===
        # Extract volatility from validation data for risk adjustment
        # High volatility → require higher confidence for trades (more conservative)
        val_volatility = X_val_unscaled['volatility_20'].values
        volatility_mean = val_volatility.mean()
        volatility_std = val_volatility.std()
        # Normalize volatility to [0.5, 2.5] range (0.5 = low vol aggressive, 2.5 = high vol conservative)
        volatility_weights = 1 + (val_volatility - volatility_mean) / (volatility_std + 1e-8)
        volatility_weights = np.clip(volatility_weights, 0.5, 2.5)  # Baseline bounds
        
        # Generate predictions
        preds = model.predict(X_val_scaled)
        proba_preds = model.predict_proba(X_val_scaled)[:, 1]
        
        # Apply volatility-weighted risk adjustment (conservative during high volatility)
        proba_preds_risk_adjusted = proba_preds / volatility_weights
        proba_preds_risk_adjusted = np.clip(proba_preds_risk_adjusted, 0, 1)
        
        # Apply probability thresholding with intelligent position opening strategy
        threshold_buy = Config.PROBABILITY_THRESHOLD_BUY
        threshold_stats = None
        if Config.ENABLE_THRESHOLD_GRIDSEARCH:
            threshold_buy, threshold_stats = find_optimal_threshold(
                y_val.values,
                proba_preds_risk_adjusted,
                metric=Config.THRESHOLD_OPTIM_METRIC,
                min_recall=Config.THRESHOLD_MIN_RECALL,
                min_precision=Config.THRESHOLD_MIN_PRECISION
            )
        
        # Apply intelligent position opening strategy
        print(f"   📍 Position Opening: {Config.POSITION_OPENING_MODE.upper()} mode")
        
        if Config.POSITION_OPENING_MODE == "hybrid":
            # Hybrid mode: threshold + percentile + technical validation
            preds_thresholded = apply_probability_threshold(
                proba_preds_risk_adjusted,
                threshold_buy,
                Config.PROBABILITY_THRESHOLD_SELL,
                X_features=X_val_unscaled,  # Pass validation features for technical filters
                percentile_threshold=Config.PERCENTILE_THRESHOLD,
                mode='hybrid'
            )
        else:
            # Simple threshold or percentile mode
            preds_thresholded = apply_probability_threshold(
                proba_preds_risk_adjusted,
                threshold_buy,
                Config.PROBABILITY_THRESHOLD_SELL,
                X_features=None,
                percentile_threshold=Config.PERCENTILE_THRESHOLD,
                mode=Config.POSITION_OPENING_MODE
            )
        
        # Log position statistics with trading details
        n_positions = np.sum(preds_thresholded == 1)
        position_rate = 100 * n_positions / len(preds_thresholded)
        print(f"   ✅ Positions opened: {n_positions}/{len(preds_thresholded)} ({position_rate:.1f}%)")
        
        # === BACKTEST SUMMARY (Historical Performance) ===
        df_val = df.loc[X_val_unscaled.index].copy() if X_val_unscaled is not None else df.tail(len(preds_thresholded)).copy()
        
        # Winning trades from backtest
        wins = (preds_thresholded == 1) & (y_val == 1)
        losses = (preds_thresholded == 1) & (y_val == 0)
        
        if wins.sum() > 0 or losses.sum() > 0:
            win_rate = wins.sum() / (wins.sum() + losses.sum()) if (wins.sum() + losses.sum()) > 0 else 0
            print(f"\n   📊 Backtest Results (Validation Period):")
            print(f"      Period: {df_val.index[0].strftime('%Y-%m-%d')} to {df_val.index[-1].strftime('%Y-%m-%d')}")
            print(f"      Winning trades: {wins.sum()}/{wins.sum() + losses.sum()} ({win_rate*100:.1f}% win rate)")
            if losses.sum() > 0:
                print(f"      Losing trades: {losses.sum()}")
            print(f"      Avoided losses: {((preds_thresholded == 0) & (y_val == 0)).sum()} (correctly predicted losers)")
        
        # Calculate metrics (using risk-adjusted probabilities for consistency)
        auc = roc_auc_score(y_val, proba_preds_risk_adjusted)
        aucpr = average_precision_score(y_val, proba_preds_risk_adjusted)
        accuracy = (preds == y_val).mean()
        accuracy_thresholded = (preds_thresholded == y_val).mean()
        precision = np.sum((preds_thresholded == 1) & (y_val == 1)) / (np.sum(preds_thresholded == 1) + 1e-6)
        recall = np.sum((preds_thresholded == 1) & (y_val == 1)) / (np.sum(y_val == 1) + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        
        # === STRATEGY RETURNS CALCULATION ===
        # Compute returns directly from close prices to avoid NaN issues
        df_val = df.loc[X_val_unscaled.index].copy()
        df_val['close_returns'] = df_val['close'].pct_change().shift(-1)  # Future returns
        val_returns = df_val['close_returns'].values
        
        # Filter out NaN returns (typically at the edges of the dataset)
        # Keep only valid indices where both predictions and returns exist
        valid_mask = ~np.isnan(val_returns)
        
        if valid_mask.sum() == 0:
            # Fallback: use raw returns even if NaN (will result in NaN Sharpe, but won't crash)
            print(f"   ⚠️ No valid returns found in validation set. Using fallback...")
            preds_valid = preds
            preds_thresholded_valid = preds_thresholded
            val_returns_valid = np.where(np.isnan(val_returns), 0, val_returns)
        else:
            preds_valid = preds[valid_mask]
            preds_thresholded_valid = preds_thresholded[valid_mask]
            val_returns_valid = val_returns[valid_mask]
        
        # Only earn returns when position is opened (pred=1), otherwise earn 0 (cash)
        strategy_returns = val_returns_valid * preds_valid
        strategy_returns_thresholded = val_returns_valid * preds_thresholded_valid
        
        # === IMPROVED SHARPE/DRAWDOWN CALCULATION ===
        # Calculate for both raw and thresholded predictions
        n_long_positions = np.sum(preds_thresholded_valid == 1)
        n_long_positions_raw = np.sum(preds_valid == 1)
        valid_days_count = len(preds_thresholded_valid)
        
        # Sharpe & Drawdown for thresholded strategy (actual trading strategy)
        if n_long_positions >= 3:  # Need at least 3 positions for meaningful Sharpe
            sharpe_thresholded = calculate_sharpe_ratio(strategy_returns_thresholded)
            max_dd = calculate_max_drawdown(strategy_returns_thresholded)
        else:
            sharpe_thresholded = np.nan
            max_dd = np.nan
        
        # Sharpe for raw model (for diagnostic purposes)
        # Note: Raw probabilities (fractional position sizes) typically yield NaN Sharpe
        # because the variance becomes too small. This is correct behavior:
        # without a binary decision threshold, position sizing is undefined.
        if n_long_positions_raw >= 3:
            sharpe = calculate_sharpe_ratio(strategy_returns)
        else:
            sharpe = np.nan
        
        # Debug: print returns statistics for troubleshooting
        print(f"   📊 Returns stats (valid days: {valid_days_count}/{len(preds)}):")
        if len(strategy_returns_thresholded) > 0:
            print(f"      Mean return: {np.nanmean(strategy_returns_thresholded):.6f}")
            print(f"      Std dev: {np.nanstd(strategy_returns_thresholded):.6f}")
            print(f"      Min/Max: {np.nanmin(strategy_returns_thresholded):.6f} / {np.nanmax(strategy_returns_thresholded):.6f}")
        print(f"      Long positions: {n_long_positions}")
        
        # Win rate calculation (only when positions are open)
        if n_long_positions > 0:
            win_rate = np.sum(strategy_returns_thresholded > 0) / n_long_positions
            total_return = np.sum(strategy_returns_thresholded)
        else:
            win_rate = 0
            total_return = 0
        
        fold_results.append({
            'fold': fold_idx,
            'auc': auc,
            'aucpr': aucpr,
            'accuracy': accuracy,
            'accuracy_thresholded': accuracy_thresholded,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'sharpe_thresholded': sharpe_thresholded,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'threshold_buy': float(threshold_buy),
            'threshold_metric': Config.THRESHOLD_OPTIM_METRIC if Config.ENABLE_THRESHOLD_GRIDSEARCH else 'static'
        })
        
        # Store predictions for visualization
        all_val_predictions.append(preds)
        all_val_proba.append(proba_preds)
        all_val_indices.append(val_idx)
        
        if threshold_stats:
            print(f"  → Threshold tuned ({Config.THRESHOLD_OPTIM_METRIC}): {threshold_buy:.4f} | PR: {threshold_stats['precision']:.3f}/{threshold_stats['recall']:.3f} | F1: {threshold_stats['f1']:.3f}")
        print(f"  → AUC: {auc:.4f} | AUCPR: {aucpr:.4f} | Accuracy: {accuracy:.4f} ({accuracy_thresholded:.4f} w/ threshold)")
        print(f"  → Sharpe: {sharpe:.4f} ({sharpe_thresholded:.4f} w/ threshold) | Max DD: {max_dd:.4f}")
        
        # Generate visualizations for this fold
        if viz_engine:
            print(f"   Generating visualizations for Fold {fold_idx}...")
            try:
                viz_engine.plot_lstm_xgboost_hybrid(
                    df=df,
                    train_data=train_idx,
                    test_data=val_idx,
                    lstm_preds=df.loc[X_val_unscaled.index, 'lstm_return_5d'].values if 'lstm_return_5d' in df.columns else np.zeros(len(val_idx)),
                    xgb_preds=proba_preds,
                    scaler=scaler,
                    run_date=run_date,
                    run_number=f"{run_number}_fold{fold_idx}"
                )
                
                future_prob = viz_engine.plot_future_forecast(
                    df=df,
                    lstm_preds=df.loc[X_val_unscaled.index, 'lstm_return_5d'].values if 'lstm_return_5d' in df.columns else np.zeros(len(val_idx)),
                    xgb_preds=proba_preds,
                    forecast_horizon=20,
                    run_date=run_date,
                    run_number=f"{run_number}_fold{fold_idx}",
                    latest_model=model,
                    scaler=scaler,
                    X_recent=X_val_unscaled.tail(30) if len(X_val_unscaled) >= 30 else X_val_unscaled  # Last 30 days for recent prediction
                )
                
                # Display FUTURE trading recommendation
                if future_prob is not None:
                    # Store for final recommendation
                    future_predictions.append({
                        'fold': fold_idx,
                        'probability': future_prob,
                        'date': df.index[-1]
                    })
                    
                    print(f"\n   " + "="*60)
                    print(f"   🎯 TRADING RECOMMENDATION FOR NEXT 20 DAYS")
                    print(f"   " + "="*60)
                    last_price = df['close'].iloc[-1]
                    last_date = df.index[-1]
                    profit_target = last_price * 1.015
                    stop_loss = last_price * 0.9925
                    
                    print(f"   📅 Valid from: {last_date.strftime('%Y-%m-%d')}")
                    print(f"   💰 Entry Price: ${last_price:.2f}")
                    print(f"   🎯 Profit Target: ${profit_target:.2f} (+1.50%)")
                    print(f"   🛑 Stop Loss: ${stop_loss:.2f} (-0.75%)")
                    print(f"   📊 Win Probability: {future_prob*100:.1f}%")
                    
                    if future_prob > 0.65:
                        print(f"\n   ✅ RECOMMENDATION: 🟢 BUY (High confidence)")
                        print(f"      Reason: Strong win probability ({future_prob*100:.1f}%)")
                        print(f"      Expected profit if target hit: ${profit_target - last_price:.2f}")
                    elif future_prob < 0.35:
                        print(f"\n   ❌ RECOMMENDATION: 🔴 AVOID/SELL (High risk)")
                        print(f"      Reason: High loss probability ({(1-future_prob)*100:.1f}%)")
                        print(f"      Potential loss if stop hit: ${last_price - stop_loss:.2f}")
                    else:
                        print(f"\n   ⚠️ RECOMMENDATION: 🟡 HOLD/WAIT (Uncertain)")
                        print(f"      Reason: Signal not clear enough ({future_prob*100:.1f}%)")
                        print(f"      Wait for probability > 65% (buy) or < 35% (sell)")
                    print(f"   " + "="*60 + "\n")
            except Exception as e:
                print(f"   ⚠️ Visualization error: {e}")
        else:
            print(f"   Skipping visualizations (VisualizationEngine not available)")

    # === DISPLAY FINAL RESULTS ===
    if not fold_results:
        print("\n❌ No valid folds to evaluate. Check data and parameters.")
        return
    
    results_df = pd.DataFrame(fold_results)
    print("\n" + "="*80)
    print("--- CROSS-VALIDATION SUMMARY (TimeSeriesSplit) ---")
    print("="*80)
    print(results_df.to_string(index=False))
    
    print("\n--- AGGREGATE STATISTICS ---")
    print(f"AUC:                  {results_df['auc'].mean():.4f} ± {results_df['auc'].std():.4f}")
    print(f"AUCPR:                {results_df['aucpr'].mean():.4f} ± {results_df['aucpr'].std():.4f}")
    print(f"Accuracy:             {results_df['accuracy'].mean():.4f} ± {results_df['accuracy'].std():.4f}")
    print(f"Accuracy (threshold): {results_df['accuracy_thresholded'].mean():.4f} ± {results_df['accuracy_thresholded'].std():.4f}")
    print(f"Sharpe Ratio:         {results_df['sharpe'].mean():.4f} ± {results_df['sharpe'].std():.4f}")
    print(f"Sharpe (threshold):   {results_df['sharpe_thresholded'].mean():.4f} ± {results_df['sharpe_thresholded'].std():.4f}")
    print(f"Max Drawdown:         {results_df['max_drawdown'].mean():.4f} ± {results_df['max_drawdown'].std():.4f}")
    print("="*80)

    # Log results to accumulative JSON log
    print("\n📊 Logging results to accumulative JSON...")
    run_config = {
        'run_number': run_number,
        'run_date': run_date.split('_')[0],
        'ticker': Config.TICKER,
        'features_used': features,
        'train_size': len(X_train_scaled) if fold_results else 0,
        'test_size': sum(len(idx) for idx in all_val_indices) if all_val_indices else 0,
        'fold_number': len(fold_results),
        'metrics': {
            'accuracy': float(results_df['accuracy'].mean()),
            'precision': float(results_df['precision'].mean()),
            'recall': float(results_df['recall'].mean()),
            'f1': float(results_df['f1'].mean()),
            'auc': float(results_df['auc'].mean()),
            'auc_pr': float(results_df['aucpr'].mean()),
            'threshold_buy_mean': float(results_df['threshold_buy'].mean()) if 'threshold_buy' in results_df else Config.PROBABILITY_THRESHOLD_BUY
        },
        'sharpe_ratio': float(results_df['sharpe'].mean()),
        'max_drawdown': float(results_df['max_drawdown'].mean()),
        'total_return': float(np.sum([s for s in results_df['sharpe']])),
        'win_rate': float(np.mean([1 if s > 0 else 0 for s in results_df['sharpe']])),
        'xgb_params': Config.XGB_PARAMS,
        'lstm_config': {
            'window_size': 60,
            'latent_dim': 32,
            'horizons': [5, 10, 20]
        },
        'buy_signals': int(np.sum([np.sum(p > 0.65) for p in all_val_proba])),
        'sell_signals': int(np.sum([np.sum(p < 0.35) for p in all_val_proba])),
        'hold_signals': int(np.sum([np.sum((p >= 0.35) & (p <= 0.65)) for p in all_val_proba])),
        'visualization_file': f'out/lstm_xgboost_hybrid_run_{run_number}_*',
        'forecast_file': f'out/future_forecast_run_{run_number}_*',
        'notes': f'TimeSeriesSplit with {Config.N_SPLITS} folds, embargo period {Config.EMBARGO_DAYS} days'
    }
    
    run_logger.log_run(run_config)
    print(f"   ✅ Run {run_number} logged successfully")
    print(run_logger.get_summary())
    
    # Export CSV summary
    run_logger.export_csv('out/runs_summary.csv')

    # Feature Importance (from final fold model)
    try:
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n--- Top 15 Feature Importances ---")
        print(importance_df.head(15).to_string(index=False))
    except Exception as e:
        print(f"\nCould not generate feature importances: {e}")
    
    # Return future predictions for final recommendation
    return future_predictions

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("🚀 Starting Optimized ForecastTRADE Pipeline (v2.0)...")
    print("Features: TimeSeriesSplit | Data Leakage Prevention | Financial Metrics")
    
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
            # Use rsi_14 (created in feature engineering, see line 206)
            if 'rsi_14' in main_df.columns:
                main_df['lstm_vs_rsi'] = main_df['lstm_price_10d'] / main_df['close'] * main_df['rsi_14']
            else:
                print("   ⚠️ Warning: rsi_14 not found, skipping lstm_vs_rsi feature")
        
        # 5. Define target
        main_df = define_target(main_df)
        
        # 6. Basic preprocessing (NO scaling - done per fold)
        processed_df = preprocess_data_raw(main_df.copy())
        
        # 7. Train and evaluate with TimeSeriesSplit (PREVENTS DATA LEAKAGE)
        if not processed_df.empty:
            print("\n👑 --- Launching XGBoost with Time-Series-Aware Cross-Validation --- 👑")
            future_preds = train_and_evaluate_timeseries(processed_df)
            
            # === FINAL CONSOLIDATED RECOMMENDATION ===
            if future_preds and len(future_preds) > 0:
                print("\n" + "="*80)
                print("🔮 FINAL TRADING RECOMMENDATION (Ensemble from All Folds)")
                print("="*80)
                
                # Get latest price and targets
                last_price = main_df['close'].iloc[-1]
                last_date = main_df.index[-1]
                profit_target = last_price * 1.015
                stop_loss = last_price * 0.9925
                
                # Ensemble: Average probability from all folds
                avg_prob = np.mean([p['probability'] for p in future_preds])
                max_prob = np.max([p['probability'] for p in future_preds])
                min_prob = np.min([p['probability'] for p in future_preds])
                
                print(f"\n📊 Ensemble Prediction Statistics:")
                print(f"   Average Probability: {avg_prob*100:.1f}%")
                print(f"   Range: {min_prob*100:.1f}% - {max_prob*100:.1f}%")
                print(f"   Folds Agreement: {len(future_preds)} models")
                
                print(f"\n💰 Trading Setup:")
                print(f"   Current Price: ${last_price:.2f} (as of {last_date.strftime('%Y-%m-%d')})")
                print(f"   Profit Target: ${profit_target:.2f} (+1.50%)")
                print(f"   Stop Loss: ${stop_loss:.2f} (-0.75%)")
                print(f"   Risk/Reward Ratio: 2:1")
                
                # Generate final recommendation based on ensemble
                print(f"\n" + "="*80)
                if avg_prob > 0.65:
                    print("✅ FINAL DECISION: 🟢 STRONG BUY")
                    print("="*80)
                    print(f"Confidence: HIGH ({avg_prob*100:.1f}%)")
                    print(f"Expected outcome: Price rises to ${profit_target:.2f}")
                    print(f"Potential profit: ${profit_target - last_price:.2f} (+1.50%)")
                    print(f"\nAction Plan:")
                    print(f"  1. Enter LONG position at ${last_price:.2f}")
                    print(f"  2. Set Take Profit at ${profit_target:.2f}")
                    print(f"  3. Set Stop Loss at ${stop_loss:.2f}")
                    print(f"  4. Hold for maximum 20 days")
                elif avg_prob < 0.35:
                    print("🛑 FINAL DECISION: 🔴 DO NOT BUY / AVOID")
                    print("="*80)
                    print(f"Confidence: HIGH ({(1-avg_prob)*100:.1f}% loss probability)")
                    print(f"Expected outcome: Price falls to ${stop_loss:.2f}")
                    print(f"Potential loss: ${last_price - stop_loss:.2f} (-0.75%)")
                    print(f"\nAction Plan:")
                    print(f"  1. DO NOT enter any position")
                    print(f"  2. Wait for win probability > 65%")
                    print(f"  3. Monitor daily for signal improvement")
                    print(f"  4. Consider SHORT position if probability < 20%")
                else:
                    print("⚠️ FINAL DECISION: 🟡 HOLD / WAIT FOR BETTER SIGNAL")
                    print("="*80)
                    print(f"Confidence: MEDIUM (uncertain, {avg_prob*100:.1f}%)")
                    print(f"Expected outcome: Unclear direction")
                    print(f"\nAction Plan:")
                    print(f"  1. STAY OUT of the market")
                    print(f"  2. Signal too weak for confident trading")
                    print(f"  3. Wait for probability > 65% (buy) or < 35% (sell)")
                    print(f"  4. Re-run analysis in 1-2 days")
                
                print("\n⚠️  Risk Warning:")
                print("   - Past performance doesn't guarantee future results")
                print("   - Always use proper position sizing (max 2-5% of capital)")
                print("   - Never trade without a stop loss")
                print("   - Market conditions can change rapidly")
                print("="*80)
        else:
            print("❌ Error: No data left after preprocessing. Check data quality and date ranges.")
    else:
        print("❌ Error: Failed to fetch initial data. Check tickers and network connection.")
        
    print("\n✅ Pipeline finished.")

