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

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logs
os.environ['ABSL_LOG_SEVERITY_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'

import argparse
import random
import time
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, precision_recall_curve
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
try:
    from sklearn.frozen import FrozenEstimator
except Exception:  # pragma: no cover
    FrozenEstimator = None
from imblearn.over_sampling import SMOTE, ADASYN
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Optional
try:
    from .data_pipeline import load_market_data
    from .feature_pipeline import run_feature_pipeline
    from .labeling_pipeline import run_labeling_pipeline
    from .training_pipeline import run_training_pipeline
    from .reporting_pipeline import (
        build_dynamic_risk_config,
        latest_volatility_for_risk,
        summarize_future_predictions,
    )
    from .lstm_predictor import (
        generate_lstm_regime_features,
        generate_placeholder_lstm_features,
        generate_lstm_regime_features_train_only,
    )
    from .run_logger import AccumulativeRunLogger
    from .rich_ui import RichUI
    from .utils.runtime import (
        CacheManager,
        DataConfig,
        FeatureConfig,
        LabelingConfig,
        RiskConfig,
        RunConfig,
        RuntimeFlags,
        StageProfiler,
        TrainingConfig,
    )
    from .configuration import apply_config_overrides, load_external_config
    from .modeling.cv import PurgedEmbargoTimeSeriesSplit, sliding_window_splits as sliding_window_splits_v2
    from .modeling.thresholds import (
        find_optimal_buy_threshold_sharpe as find_optimal_buy_threshold_sharpe_v2,
        find_optimal_threshold as find_optimal_threshold_v2,
    )
    from .modeling.calibration import (
        calibration_metrics,
        fit_probability_calibrator as fit_probability_calibrator_v2,
        train_calibration_split,
    )
    from .backtest.metrics import (
        apply_trading_costs,
        calculate_max_drawdown as calculate_max_drawdown_v2,
        calculate_sharpe_ratio as calculate_sharpe_ratio_v2,
        evaluate_net_metrics,
    )
    from .recommendation.engine import conservative_recommendation
except ImportError:
    from data_pipeline import load_market_data
    from feature_pipeline import run_feature_pipeline
    from labeling_pipeline import run_labeling_pipeline
    from training_pipeline import run_training_pipeline
    from reporting_pipeline import (
        build_dynamic_risk_config,
        latest_volatility_for_risk,
        summarize_future_predictions,
    )
    from lstm_predictor import (
        generate_lstm_regime_features,
        generate_placeholder_lstm_features,
        generate_lstm_regime_features_train_only,
    )
    from run_logger import AccumulativeRunLogger
    from rich_ui import RichUI
    from utils.runtime import (
        CacheManager,
        DataConfig,
        FeatureConfig,
        LabelingConfig,
        RiskConfig,
        RunConfig,
        RuntimeFlags,
        StageProfiler,
        TrainingConfig,
    )
    from configuration import apply_config_overrides, load_external_config
    from modeling.cv import PurgedEmbargoTimeSeriesSplit, sliding_window_splits as sliding_window_splits_v2
    from modeling.thresholds import (
        find_optimal_buy_threshold_sharpe as find_optimal_buy_threshold_sharpe_v2,
        find_optimal_threshold as find_optimal_threshold_v2,
    )
    from modeling.calibration import (
        calibration_metrics,
        fit_probability_calibrator as fit_probability_calibrator_v2,
        train_calibration_split,
    )
    from backtest.metrics import (
        apply_trading_costs,
        calculate_max_drawdown as calculate_max_drawdown_v2,
        calculate_sharpe_ratio as calculate_sharpe_ratio_v2,
        evaluate_net_metrics,
    )
    from recommendation.engine import conservative_recommendation
# from triple_barrier_labeler import TripleBarrierLabeler  # DEPRECATED: now using percentile-based labeling

# VisualizationEngine will be imported dynamically when needed
VisualizationEngine = None

# Initialize Rich UI
ui = RichUI()

warnings.filterwarnings('ignore', category=FutureWarning)


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:
        pass

# --- 1. CONFIGURATION ---
class Config:
    """Configuration class for the modeling pipeline."""
    SEED = 42
    OBJECTIVE = "sharpe_net"
    RISK_PROFILE = "conservative"
    CV_SCHEME = "sliding"
    COST_BPS = 20.0
    SLIPPAGE_BPS = 5.0
    MIN_COVERAGE_RATIO = 0.5
    CONSERVATIVE_MIN_BUY_THRESHOLD = 0.66
    CONSERVATIVE_MAX_TRADE_RATIO = 0.22
    CONSERVATIVE_PERCENTILE_THRESHOLD = 80
    CONSERVATIVE_MIN_CONFIDENCE = 0.62
    CONSERVATIVE_UNCERTAINTY_BAND = 0.12
    CONSERVATIVE_VOL_PENALTY_MULT = 1.25

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

    # Dynamic Risk / Labeling Parameters (Asset-Agnostic)
    DYNAMIC_RISK_TYPE = "Volatility_Adjusted"
    DYNAMIC_RISK_K_TP = 2.5
    DYNAMIC_RISK_K_SL = 1.25
    DYNAMIC_RISK_VOL_METRIC = "rolling_std_20d"  # Options: rolling_std_20d, atr_14d
    ESTIMATED_TRANSACTION_COST = 0.0020  # 0.20% per trade (estimated)
    # Triple barrier labeling parameters (separate from trading execution risk)
    LABEL_TP_MULTIPLIER = 2.0
    LABEL_SL_MULTIPLIER = 1.5
    LABEL_HORIZON_DAYS = 20

    # Model Parameters (optimized for PERCENTILE LABELING + balanced classes)
    XGB_PARAMS = {
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',  # Use AUCPR instead of AUC for imbalanced data
        'max_depth': 6,  # Increased from 5: more data (1000 vs 750) allows deeper trees
        'learning_rate': 0.02,  # Reduced from 0.03: slower learning with more data
        'n_estimators': 600,  # Increased from 500: more iterations with lower LR
        'subsample': 0.85,  # Increased from 0.8: more stable with balanced classes
        'colsample_bytree': 0.85,  # Increased from 0.8: can use more features now
        'min_child_weight': 2,  # Reduced from 3: balanced classes allow finer splits
        'gamma': 0.2,  # Reduced from 0.3: less conservative with 50/50 balance
        'reg_alpha': 0.2,  # L1 regularization (reduced for more expressiveness)
        'reg_lambda': 1.0,  # L2 regularization (reduced)
        'max_delta_step': 1,  # Reduced from 2: balanced classes need less step limiting
        'scale_pos_weight': None,  # Computed dynamically (should be ~1.0 with balance)
        'random_state': 42,
        'use_label_encoder': False,
        'early_stopping_rounds': 50  # Increased patience
    }

    # Time-Series Cross-Validation Parameters
    N_SPLITS = 12  # 12 time-series splits for comprehensive validation
    EMBARGO_DAYS = 20  # Increased to 20 days to match prediction horizon (prevents overlap leakage)
    MIN_TRAIN_SIZE = 252  # 252 trading days (~1 year) minimum
    
    # Walk-Forward Refinement: Sliding Window Configuration
    # Instead of expanding window (old data + new data), use fixed-size sliding window
    # This makes model adapt faster to market regime changes
    USE_SLIDING_WINDOW = True  # Enable adaptive sliding window strategy
    SLIDING_WINDOW_SIZE = 400  # 400 days window (~1.5 years)
    VALIDATION_WINDOW_SIZE = 120  # Increased to 120 days (~6 months) for statistically significant validation
    MIN_VALIDATION_SIZE = 60  # Minimum validation samples to keep a fold
    MIN_TRADES_PER_FOLD = 3  # Skip unreliable folds with too few trades
    # Benefit: Model returns are more realistic and less prone to "lucky" small sample sizes
    # Note: Requires more data but yields robust metrics
    
    # Probability Threshold Parameters
    TRADING_MODE: Literal["long_only", "long_short"] = "long_only"
    PROBABILITY_THRESHOLD_BUY = 0.65  # Only buy if prob > 65% (baseline)
    PROBABILITY_THRESHOLD_SELL = 0.35  # Only sell if prob < 35% (baseline)
    ENABLE_THRESHOLD_GRIDSEARCH = True  # Tune buy threshold from PR curve
    THRESHOLD_OPTIM_METRIC = "f1"  # Options: "f1", "precision_at_recall"
    THRESHOLD_MIN_RECALL = 0.25  # Used for precision_at_recall
    THRESHOLD_MIN_PRECISION = 0.25  # Used for recall_at_precision (future)
    THRESHOLD_MIN_TRADES_RATIO = 0.08  # Minimum long-trade ratio when optimizing threshold
    
    # NEW: Sharpe-Optimized Threshold System
    OPTIMIZE_BY_SHARPE = True  # Optimize thresholds by Sharpe Ratio (better than F1)
    USE_ADAPTIVE_THRESHOLDS = True  # Use volatility-adaptive thresholds
    ADAPTIVE_THRESHOLD_MAX_SHIFT = 0.04  # Smoothed max shift (±4%, reduced from implicit ±10%)
    ADAPTIVE_THRESHOLD_SMOOTHING_ALPHA = 0.35  # EMA smoothing across folds (lower = smoother)
    ADAPTIVE_THRESHOLD_ZSCORE_CLIP = 2.5  # Cap extreme volatility impact

    # Stability-Oriented Hyperparameter Tuning
    ENABLE_STABILITY_TUNING = True  # Lightweight fold-level tuning for robust performance
    TUNE_ON_FIRST_VALID_FOLD_ONLY = True  # Tune once, reuse for remaining folds (faster + stabler)
    STABILITY_TUNING_MAX_CANDIDATES = 8
    STABILITY_TUNING_MIN_TRADES = 8
    USE_SYNTHETIC_SAMPLING = True  # ADASYN/SMOTE for class balancing
    ENABLE_PROBA_CALIBRATION = True
    CALIBRATION_METHOD: Literal["isotonic", "sigmoid"] = "isotonic"
    CALIBRATION_MIN_SAMPLES = 300
    LSTM_LEAKAGE_SAFE_MODE = True  # Refit LSTM per fold on train only
    ENABLE_REGIME_SPLIT_MODEL = False  # Experimental two-model regime router
    REGIME_VOL_Z_THRESHOLD = 1.0
    
    # Position Opening Strategy
    POSITION_OPENING_MODE = "hybrid"  # Options: "threshold" (simple), "percentile" (top X%), "hybrid" (smart)
    PERCENTILE_THRESHOLD = 60  # For percentile/hybrid mode: keep top X% of predictions (60=top 40%)
    ENABLE_POSITION_SIZING = True  # Dynamic exposure sizing by confidence/regime
    POSITION_SIZE_MIN_CONFIDENCE = 0.55  # Below this, skip trade even if threshold passes
    POSITION_SIZE_UNCERTAINTY_BAND = 0.08  # Skip trades near 50% probability
    POSITION_SIZE_VOL_RISK_PENALTY = 0.35  # Exposure reduction in high-vol regimes
    POSITION_SIZE_MAX = 1.0
    ENABLE_REGIME_HARD_GATE = True  # Block all long entries in hostile regime
    HARD_GATE_VOL_ZSCORE = 1.20  # High-volatility cutoff
    HARD_GATE_BEARISH_MA200_RATIO = 0.60  # If >60% samples are >2% below MA200
    # Modes explained:
    #   - "threshold": Classic approach - buy if proba >= threshold
    #   - "percentile": Aggressive filtering - only trade top X% most confident predictions
    #   - "hybrid": Smart combo - threshold + percentile + technical validation (volatility, ATR, trend)

    # Feature List (LSTM Compressed Latent + TOP technical indicators + Market Regime)
    # The compressed latent features (10-dim) capture only critical temporal patterns
    # Bottleneck compression removes noise while retaining essential information
    # Market regime features provide contextual awareness (volatility, trend strength, bull/bear position)
    # These are then combined with the best technical features for XGBoost
    FEATURES_TO_SCALE = [  # Seed list; final features are resolved dynamically
        # LSTM Compressed Latent Features (10-dimensional, linear activation)
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
        # LSTM direction confidence features
        'lstm_dir_5d', 'lstm_dir_10d', 'lstm_dir_20d', 'lstm_return_confidence',
    ]
    MIN_FEATURES = 15  # Minimum features required after contract resolution


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
        
        # Display using Rich UI
        ui.show_progress_fetching(ticker, len(df))
        return df
        
    except Exception as e:
        ui.show_error(f"Failed to fetch data: {e}")
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
    
    # Rename rsi_14 to rsi for compatibility with Config.FEATURES_TO_SCALE
    df['rsi'] = df['rsi_14']
    
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
    volatility_mean = df['volatility_20'].rolling(window=252).mean()
    volatility_std = df['volatility_20'].rolling(window=252).std()
    df['rolling_volatility_zscore'] = (df['volatility_20'] - volatility_mean) / (volatility_std + 1e-8)
    
    df['ADX_14'] = df.get('ADX_14', 20)
    
    df['ma_200'] = df['close'].rolling(window=200).mean()
    df['distance_to_ma200'] = (df['close'] - df['ma_200']) / df['close']

    vol_state = np.where(df['rolling_volatility_zscore'] > 1.0, 1,
                         np.where(df['rolling_volatility_zscore'] < -1.0, -1, 0))
    trend_state = np.where(df['distance_to_ma200'] >= 0, 1, -1)
    df['regime_state'] = (vol_state * 2) + trend_state
    df['regime_change'] = (df['regime_state'] != df['regime_state'].shift(1)).fillna(0).astype(int)
    
    # Use Rich UI to show progress
    ui.show_progress_engineering(len(df.columns))
    
    # Log returns (smoother than simple returns)
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # === ACCUMULATION LINES ===
    # Money Flow (High-Low-Close index)
    df.ta.hlc3(append=True)
    
    return df

# --- 3.1 ADVANCED FEATURE ENGINEERING (OPTIMIZED) ---
def remove_correlated_features(df, threshold=0.98, protected_features=None):
    """
    Removes highly correlated features to reduce redundancy.
    NOW WITH INTELLIGENCE: Protects high-importance features from elimination.
    
    Args:
        df (pd.DataFrame): DataFrame with features
        threshold (float): Correlation threshold (default 0.98 - very conservative)
        protected_features (list): Features that should never be dropped (high importance)
    
    Returns:
        pd.DataFrame: DataFrame with redundant features removed
        list: List of removed features
    """
    print(f"\n🔍 Analyzing feature correlations (threshold={threshold})...")
    
    # Protected columns that should never be dropped
    protected = ['date', 'open', 'high', 'low', 'close', 'volume', 'target', 
                 'vix_close', 'forward_return']
    
    # Add top importance features to protected list (from historical runs)
    # These are consistently high-importance across runs
    top_features = [
        'lstm_latent_0', 'lstm_latent_3', 'lstm_latent_4', 'lstm_latent_5',
        'ema_short', 'ema_long', 'distance_to_ma200', 'close_vix_ratio',
        'volatility_20', 'atr', 'ADX_14', 'macds', 'macd',
        'lstm_latent_1', 'lstm_latent_6', 'lstm_latent_7', 'lstm_latent_9',
        'price_to_sma20', 'volume_ratio', 'mfi'
    ]
    
    if protected_features:
        protected.extend(protected_features)
    protected.extend(top_features)
    protected = list(set(protected))  # Remove duplicates
    
    # Get numeric columns excluding protected ones
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in protected]
    
    if len(feature_cols) < 2:
        print("   ℹ️ Not enough droppable features for correlation analysis")
        return df, []
    
    # Compute correlation matrix
    corr_matrix = df[feature_cols].corr().abs()
    
    # Find highly correlated pairs
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Features to drop (keep first, drop second in correlated pair)
    to_drop = []
    for column in upper_triangle.columns:
        correlated_features = upper_triangle[column][upper_triangle[column] > threshold]
        if not correlated_features.empty:
            for corr_feature in correlated_features.index:
                if corr_feature not in to_drop and corr_feature not in protected:
                    to_drop.append(corr_feature)
                    print(f"   ⚠️ Dropping '{corr_feature}' (corr={upper_triangle[column][corr_feature]:.3f} with '{column}')")
                elif corr_feature in protected:
                    print(f"   🛡️ Protecting '{corr_feature}' (corr={upper_triangle[column][corr_feature]:.3f} with '{column}', but is high-importance)")
    
    if to_drop:
        df = df.drop(columns=to_drop, errors='ignore')
        print(f"   ✅ Removed {len(to_drop)} redundant features (protected {len([f for f in top_features if f in df.columns])} important ones)")
    else:
        print("   ✅ No highly correlated droppable features found")
    
    return df, to_drop


def create_advanced_features(df):
    """
    Creates OPTIMIZED advanced features:
    - Only high-impact technical ratios
    - Selective LSTM×Technical interactions (top LSTM latents only)
    - Reduced feature count to avoid overfitting
    
    Expected improvement: AUC +8-12% (0.66→0.74) with less overfitting
    """
    print("\n🚀 Creating Advanced Feature Interactions (Optimized)...")
    
    df = df.copy()
    
    # === HIGH-IMPACT TECHNICAL RATIOS ONLY ===
    print("   📊 Adding high-impact technical ratios...")
    
    # Volatility normalized by price (proven important)
    if 'atr' in df.columns and 'close' in df.columns:
        df['atr_price_ratio'] = df['atr'] / df['close']
    
    # Volume dynamics (simplified)
    if 'volume' in df.columns and 'volume_ma_20' in df.columns:
        df['volume_ma_ratio'] = df['volume'] / (df['volume_ma_20'] + 1e-8)
    
    # Price momentum (only 10d, more stable than 5d)
    df['price_momentum_10d'] = df['close'] / df['close'].shift(10) - 1
    
    # === SELECTIVE LSTM×TECHNICAL INTERACTIONS ===
    # Only create interactions for top LSTM latents (0, 3, 4) based on importance
    print("   🧠 Creating selective LSTM×Technical interactions...")
    
    top_lstm_latents = [0, 3, 4]  # Based on feature importance from previous runs
    
    for i in top_lstm_latents:
        if f'lstm_latent_{i}' not in df.columns:
            continue
            
        # LSTM×RSI (strong predictor)
        if 'rsi_14' in df.columns:
            df[f'lstm{i}_×_rsi'] = df[f'lstm_latent_{i}'] * df['rsi_14'] / 100.0
        
        # LSTM×MACD (momentum capture)
        if 'macd' in df.columns:
            df[f'lstm{i}_×_macd'] = df[f'lstm_latent_{i}'] * df['macd']
        
        # LSTM×Distance to MA200 (trend alignment)
        if 'distance_to_ma200' in df.columns:
            df[f'lstm{i}_×_trend'] = df[f'lstm_latent_{i}'] * df['distance_to_ma200']
    
    # === ESSENTIAL MULTI-INDICATOR COMBINATIONS ===
    print("   🔬 Adding essential combinations...")
    
    # Risk-adjusted momentum (high predictive power)
    if 'atr_price_ratio' in df.columns:
        df['risk_adjusted_momentum'] = df['price_momentum_10d'] / (df['atr_price_ratio'] + 1e-8)
    
    print(f"   ✅ Total features after optimized engineering: {len(df.columns)}")
    
    return df


def filter_low_importance_features(df, known_low_importance=None):
    """
    Filters out features known to have low importance from previous runs.
    This is an adaptive system that learns from past executions.
    
    Args:
        df (pd.DataFrame): DataFrame with features
        known_low_importance (list): List of features to remove (from past runs)
    
    Returns:
        pd.DataFrame: Filtered dataframe
        list: List of dropped features
    """
    print(f"\n🔍 Applying adaptive feature filtering...")
    
    # Protected columns that should never be dropped
    protected_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'target', 
                      'vix_close', 'forward_return']
    
    # Known low-importance features from Run 65-66 analysis (<1% importance)
    # These are consistently low across multiple runs
    default_low_importance = [
        'lstm_latent_2',  # Consistently low importance
        'rsi',  # Redundant with rsi_14
        'rolling_volatility_zscore',  # Low predictive power
        'regime_state',  # Not capturing useful signal
        'regime_change',  # Too noisy
        'vix_change',  # Redundant with close_vix_ratio
        'volume_roc_10',  # Redundant with volume_ma_ratio
        'volume_ratio',  # Often noisy in crypto/stocks
        'regime_change', # Identified as low importance
    ]
    
    # Merge with any provided low-importance features
    if known_low_importance:
        features_to_drop = list(set(default_low_importance + known_low_importance))
    else:
        features_to_drop = default_low_importance
    
    # Don't drop protected columns
    features_to_drop = [f for f in features_to_drop if f not in protected_cols and f in df.columns]
    
    if features_to_drop:
        print(f"   ⚠️ Removing {len(features_to_drop)} known low-importance features:")
        print(f"      {', '.join(features_to_drop[:10])}")
        df = df.drop(columns=features_to_drop, errors='ignore')
        print(f"   ✅ Features dropped successfully")
    else:
        print("   ℹ️ No low-importance features found to drop")
    
    return df, features_to_drop


def resolve_feature_contract(
    df: pd.DataFrame,
    seed_features: List[str],
    removed_by_corr: Optional[List[str]] = None,
    removed_by_importance: Optional[List[str]] = None,
    min_features: int = 15,
) -> Dict[str, Any]:
    """
    Resolve the final feature contract after all engineering/filtering steps.

    Returns a dictionary with requested/available/final feature lists and
    dropped-feature diagnostics.
    """
    removed_by_corr = removed_by_corr or []
    removed_by_importance = removed_by_importance or []

    requested = list(seed_features)
    available = [f for f in requested if f in df.columns]
    intentional_drops = set(removed_by_corr) | set(removed_by_importance)
    missing_unexpected = [f for f in requested if f not in df.columns and f not in intentional_drops]

    final_features = [f for f in available if pd.api.types.is_numeric_dtype(df[f])]
    if len(final_features) < min_features:
        raise ValueError(
            f"Feature contract too small: {len(final_features)} < {min_features}. "
            "Review feature engineering and filtering thresholds."
        )

    return {
        'requested': requested,
        'available': available,
        'removed_by_corr': sorted(set(removed_by_corr)),
        'removed_by_importance': sorted(set(removed_by_importance)),
        'missing_unexpected': missing_unexpected,
        'final': final_features,
    }

# --- 4. TARGET DEFINITION ---
def define_target(df):
    """
    Defines target using TRIPLE BARRIER METHOD (Volatility-Based).
    
    This eliminates look-ahead bias by using ONLY past volatility to define targets.
    - Dynamic Profit Target: Entry + (Volatility * K_TP)
    - Dynamic Stop Loss: Entry - (Volatility * K_SL)
    - Time Horizon: 20 days
    
    Labels:
    - 1 (BUY): Price hit Profit Target first
    - 0 (SELL): Price hit Stop Loss first
    - NaN (HOLD): Neither hit within 20 days (or timed out)
    
    Args:
        df (pd.DataFrame): DataFrame with 'close' price
        
    Returns:
        pd.DataFrame: DataFrame with 'target' column
    """
    print("\n📊 Defining target with Triple Barrier Method (No Leakage)...")

    # 1. DYNAMIC VOLATILITY REFERENCE
    if Config.DYNAMIC_RISK_VOL_METRIC == 'atr_14d' and 'atr' in df.columns:
        vol_series = pd.to_numeric(df['atr'], errors='coerce')
        vol_metric_used = 'atr_14d'
    elif 'volatility_20' in df.columns:
        vol_series = pd.to_numeric(df['volatility_20'], errors='coerce')
        vol_metric_used = 'rolling_std_20d'
    else:
        vol_series = pd.to_numeric(df['close'], errors='coerce').rolling(window=20).std()
        vol_metric_used = 'rolling_std_20d'

    vol_series = vol_series.fillna(method='ffill').fillna(method='bfill')
    
    # 2. CALCULATE BARRIERS FOR EACH TIMESTAMP
    # Note: These are "theoretical" barriers if we entered at this timestamp
    df['barrier_up'] = df['close'] + (vol_series * Config.LABEL_TP_MULTIPLIER)
    df['barrier_down'] = df['close'] - (vol_series * Config.LABEL_SL_MULTIPLIER)
    
    print(
        f"   Triple Barrier Config: Volatility={vol_metric_used} | "
        f"TP={Config.LABEL_TP_MULTIPLIER:.1f}x | SL={Config.LABEL_SL_MULTIPLIER:.1f}x "
        f"| Horizon={Config.LABEL_HORIZON_DAYS}d"
    )

    # 3. VECTORIZED BARRIER TOUCH DETECTION
    # We need to look forward 20 days to see what was hit first
    horizon = int(Config.LABEL_HORIZON_DAYS)
    
    # Use numpy for speed
    closes = df['close'].values
    highs = df['high'].values if 'high' in df.columns else closes
    lows = df['low'].values if 'low' in df.columns else closes
    ups = df['barrier_up'].values
    downs = df['barrier_down'].values
    
    labels = np.full(len(df), np.nan) # Default to NaN (HOLD/TIMEOUT)
    
    # Iterate through potential entry points
    # (Vectorization is hard for "first touch" logic, loop is safer for correctness here)
    # We stop at len(df) - horizon to avoid index errors
    
    hits_up = 0
    hits_down = 0
    timeouts = 0
    
    for i in range(len(df) - horizon):
        entry_price = closes[i]
        barrier_up = ups[i]
        barrier_down = downs[i]
        
        # Look at window [i+1 : i+1+horizon]
        window_highs = highs[i+1 : i+1+horizon]
        window_lows = lows[i+1 : i+1+horizon]
        
        # Check touches
        # Find indices where barriers are breached
        up_touches = np.where(window_highs >= barrier_up)[0]
        down_touches = np.where(window_lows <= barrier_down)[0]
        
        first_up = up_touches[0] if len(up_touches) > 0 else 9999
        first_down = down_touches[0] if len(down_touches) > 0 else 9999
        
        if first_up == 9999 and first_down == 9999:
            # Timeout (neither hit)
            labels[i] = np.nan
            timeouts += 1
        elif first_up < first_down:
            # Hit profit target first
            labels[i] = 1.0 # BUY
            hits_up += 1
        elif first_down < first_up:
            # Hit stop loss first
            labels[i] = 0.0 # SELL
            hits_down += 1
        else:
            # Simultaneous touch (rare, assume stop loss for safety)
            labels[i] = 0.0
            hits_down += 1
            
    df['target'] = labels
    
    # 4. REPORT DISTRIBUTION
    total_labeled = hits_up + hits_down + timeouts
    if total_labeled > 0:
        print(f"   Label Distribution:")
        print(f"      BUY  (Hit TP): {hits_up} ({hits_up/total_labeled*100:.1f}%)")
        print(f"      SELL (Hit SL): {hits_down} ({hits_down/total_labeled*100:.1f}%)")
        print(f"      HOLD (Timeout): {timeouts} ({timeouts/total_labeled*100:.1f}%)")
        print(f"      (HOLD samples will be dropped from training)")

    # Dynamic risk metadata for downstream recommendation modules
    df['dynamic_risk_type'] = Config.DYNAMIC_RISK_TYPE
    df['dynamic_risk_k_tp'] = Config.DYNAMIC_RISK_K_TP
    df['dynamic_risk_k_sl'] = Config.DYNAMIC_RISK_K_SL
    df['dynamic_risk_vol_metric'] = vol_metric_used
    
    return df

# --- 5. PREPROCESSING (WITH FOLD-SPECIFIC SCALING) ---
def get_feature_columns(df, feature_contract: Optional[Dict[str, Any]] = None):
    """Get the effective feature columns from the resolved contract."""
    if feature_contract is not None:
        return [c for c in feature_contract.get('final', []) if c in df.columns]
    return [c for c in Config.FEATURES_TO_SCALE if c in df.columns]

def preprocess_data_raw(df, feature_contract: Optional[Dict[str, Any]] = None):
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
    feature_cols = get_feature_columns(df, feature_contract=feature_contract)
    
    # Ensure all feature columns are numeric (coerce errors to NaN)
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # IMPORTANT: Only drop rows where FEATURES or TARGET are NaN
    # Don't drop based on ALL columns (that's too aggressive)
    cols_to_check = feature_cols + ['target']
    df_clean = df.dropna(subset=cols_to_check).copy()
    
    rows_dropped = initial_shape - df_clean.shape[0]
    print(f"   Rows before preprocessing: {initial_shape}")
    print(f"   Rows dropped (NaN in features/target + HOLD labels): {rows_dropped}")
    print(f"   Data shape after preprocessing: {df_clean.shape}")
    
    # Verify class balance after percentile labeling
    if 'target' in df_clean.columns:
        class_counts = df_clean['target'].value_counts()
        print(f"   Final class distribution: SELL={class_counts.get(0, 0)} ({class_counts.get(0, 0)/len(df_clean)*100:.1f}%), BUY={class_counts.get(1, 0)} ({class_counts.get(1, 0)/len(df_clean)*100:.1f}%)")
    
    # Sanity check: ensure we have enough data
    if df_clean.shape[0] < 50:
        print(f"⚠️  Warning: Only {df_clean.shape[0]} rows left after preprocessing")
    
    return df_clean

# --- 6. MODEL TRAINING & EVALUATION (TIME-SERIES AWARE) ---
def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    return calculate_sharpe_ratio_v2(returns, risk_free_rate=risk_free_rate)

def calculate_max_drawdown(returns):
    return calculate_max_drawdown_v2(returns)

def apply_probability_threshold(
    proba_preds,
    threshold_buy=0.65,
    threshold_sell=0.35,
    X_features=None,
    percentile_threshold=None,
    mode='threshold',
    trading_mode: Literal["long_only", "long_short"] = "long_only",
):
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
    
    if trading_mode == "long_short":
        filtered[proba_preds <= threshold_sell] = -1
    return filtered

def find_optimal_threshold(y_true, proba_preds, metric="f1", min_recall=0.25, min_precision=0.25):
    _ = min_precision
    return find_optimal_threshold_v2(y_true, proba_preds, metric=metric, min_recall=min_recall)


def find_optimal_buy_threshold_sharpe(
    y_true,
    proba_preds,
    returns,
    min_samples=10,
    min_trades_ratio=0.08,
    cost_rate=0.0,
    slippage_rate=0.0,
    max_trades_ratio=None,
):
    _ = y_true
    return find_optimal_buy_threshold_sharpe_v2(
        proba_preds=np.asarray(proba_preds),
        raw_returns=np.asarray(returns),
        min_samples=min_samples,
        min_trades_ratio=min_trades_ratio,
        cost_rate=cost_rate,
        slippage_rate=slippage_rate,
        max_trades_ratio=max_trades_ratio,
    )


def find_optimal_thresholds_long_short(y_true, proba_preds, returns, min_samples=10):
    """Reserved placeholder for future long/short mode."""
    return 0.65, 0.35, {}


def get_adaptive_thresholds(volatility_zscore, base_buy=0.65, base_sell=0.35):
    """
    Calculate adaptive thresholds based on market volatility regime.
    
    During high volatility → Higher thresholds (more conservative)
    During low volatility → Lower thresholds (more aggressive)
    
    Args:
        volatility_zscore (float): Z-score of current volatility vs historical
                                  >1 = high vol, <-1 = low vol, ~0 = normal
        base_buy (float): Base buy threshold
        base_sell (float): Base sell threshold
    
    Returns:
        tuple: (adaptive_buy, adaptive_sell)
    """
    # Smooth nonlinear adjustment factor based on volatility.
    # Uses tanh to avoid abrupt jumps and caps to ±4% by default.
    z = np.clip(volatility_zscore, -Config.ADAPTIVE_THRESHOLD_ZSCORE_CLIP, Config.ADAPTIVE_THRESHOLD_ZSCORE_CLIP)
    adjustment = np.tanh(z / 1.5) * Config.ADAPTIVE_THRESHOLD_MAX_SHIFT
    
    adaptive_buy = np.clip(base_buy + adjustment, 0.50, 0.85)
    adaptive_sell = np.clip(base_sell - adjustment, 0.15, 0.45)
    
    return adaptive_buy, adaptive_sell


def smooth_thresholds(current_buy, current_sell, prev_buy, prev_sell, alpha=0.35):
    """EMA smoothing for fold-to-fold threshold stability."""
    alpha = float(np.clip(alpha, 0.05, 1.0))
    smoothed_buy = alpha * current_buy + (1 - alpha) * prev_buy
    smoothed_sell = alpha * current_sell + (1 - alpha) * prev_sell
    return float(smoothed_buy), float(smoothed_sell)


def compute_position_sizes(
    proba_preds,
    threshold_buy,
    volatility_zscore=0.0,
    eligible_mask=None,
    min_confidence=None,
    uncertainty_band=None,
    vol_penalty=None,
):
    """
    Confidence- and regime-aware position sizing.

    Rules:
    - Skip uncertain signals near 50% probability.
    - Skip low-confidence buys below minimum confidence.
    - Reduce size during high-volatility regimes.
    """
    proba_preds = np.asarray(proba_preds)
    sizes = np.zeros_like(proba_preds, dtype=float)

    if eligible_mask is None:
        buy_mask = proba_preds >= threshold_buy
    else:
        buy_mask = np.asarray(eligible_mask).astype(bool)
    if not np.any(buy_mask):
        stats = {
            'n_candidates': 0,
            'n_skipped_uncertain': 0,
            'n_skipped_low_conf': 0,
            'n_sized': 0,
            'avg_size': 0.0
        }
        return sizes, stats

    p_buy = proba_preds[buy_mask]

    confidence_component = np.clip(
        (p_buy - threshold_buy) / (1 - threshold_buy + 1e-9),
        0,
        1
    )

    min_confidence = Config.POSITION_SIZE_MIN_CONFIDENCE if min_confidence is None else float(min_confidence)
    uncertainty_band = Config.POSITION_SIZE_UNCERTAINTY_BAND if uncertainty_band is None else float(uncertainty_band)
    vol_penalty_cfg = Config.POSITION_SIZE_VOL_RISK_PENALTY if vol_penalty is None else float(vol_penalty)

    center_distance = np.abs(p_buy - 0.5)
    uncertain = center_distance < uncertainty_band
    low_conf = p_buy < min_confidence

    # Volatility risk multiplier (smaller size in high-volatility regimes)
    vol_penalty = max(0.0, volatility_zscore)
    vol_multiplier = float(np.clip(1.0 - vol_penalty_cfg * (vol_penalty / 2.0), 0.25, 1.0))

    base_size = confidence_component * vol_multiplier
    base_size[uncertain | low_conf] = 0.0
    base_size = np.clip(base_size, 0.0, Config.POSITION_SIZE_MAX)

    sizes[buy_mask] = base_size

    non_zero = sizes[sizes > 0]
    stats = {
        'n_candidates': int(buy_mask.sum()),
        'n_skipped_uncertain': int(uncertain.sum()),
        'n_skipped_low_conf': int(low_conf.sum()),
        'n_sized': int((sizes > 0).sum()),
        'avg_size': float(non_zero.mean()) if len(non_zero) > 0 else 0.0
    }
    return sizes, stats


def diagnose_fold_market_conditions(fold_idx, X_train_unscaled, X_val_unscaled, y_val, proba_preds, preds_thresholded, position_sizes):
    """Diagnose market regime conditions behind weak/no-trade fold outcomes."""
    diagnostics = {}

    # Class regime
    class_dist = y_val.value_counts(normalize=True).to_dict()
    diagnostics['val_class_dist'] = {int(k): float(v) for k, v in class_dist.items()}

    # Volatility regime relative to training window (critical fix: no self-centering)
    if 'volatility_20' in X_train_unscaled.columns and 'volatility_20' in X_val_unscaled.columns:
        train_vol = X_train_unscaled['volatility_20'].values
        val_vol = X_val_unscaled['volatility_20'].values
        vol_mean = np.nanmean(train_vol)
        vol_std = np.nanstd(train_vol) + 1e-9
        val_vol_z = (val_vol - vol_mean) / vol_std
        diagnostics['val_vol_zscore_mean'] = float(np.nanmean(val_vol_z))
        diagnostics['val_vol_zscore_std'] = float(np.nanstd(val_vol_z))
    else:
        diagnostics['val_vol_zscore_mean'] = np.nan
        diagnostics['val_vol_zscore_std'] = np.nan

    # Trend regime
    if 'distance_to_ma200' in X_val_unscaled.columns:
        dist_ma = X_val_unscaled['distance_to_ma200'].values
        diagnostics['distance_to_ma200_mean'] = float(np.nanmean(dist_ma))
        diagnostics['pct_below_ma200_gt_2pct'] = float(np.mean(dist_ma < -0.02))

    # Signal confidence concentration
    mid_zone = (proba_preds >= 0.45) & (proba_preds <= 0.55)
    diagnostics['signal_midzone_ratio'] = float(np.mean(mid_zone))
    diagnostics['mean_probability'] = float(np.mean(proba_preds))
    diagnostics['std_probability'] = float(np.std(proba_preds))

    # Tradeability / sizing behavior
    n_opened = int(np.sum(preds_thresholded == 1))
    n_sized = int(np.sum(position_sizes > 0))
    diagnostics['opened_positions'] = n_opened
    diagnostics['sized_positions'] = n_sized

    is_problematic = (n_opened > 0 and n_sized == 0) or (n_sized == 0) or (fold_idx in (6, 7))
    if is_problematic:
        pass
        # print(f"   🔍 Fold {fold_idx} Regime Diagnostics:")
        # print(f"      • Validation class balance: {diagnostics['val_class_dist']}")
        # print(f"      • Volatility regime (z): mean={diagnostics['val_vol_zscore_mean']:.2f}, std={diagnostics['val_vol_zscore_std']:.2f}")
        # if 'distance_to_ma200_mean' in diagnostics:
        #     print(f"      • Trend regime: mean distance_to_ma200={diagnostics['distance_to_ma200_mean']:.3f}, below MA200>2%={diagnostics['pct_below_ma200_gt_2pct']:.1%}")
        # print(f"      • Signal concentration (0.45-0.55): {diagnostics['signal_midzone_ratio']:.1%}")
        # print(f"      • Positions opened={n_opened}, sized={n_sized}")

    return diagnostics


def apply_regime_hard_gate(preds_thresholded, X_val_unscaled, avg_vol_zscore):
    """Apply fold-level hard gate to skip trades in hostile market regimes."""
    gated = preds_thresholded.copy()
    gate_active = False
    gate_reason = ""

    bearish_ratio = 0.0
    if 'distance_to_ma200' in X_val_unscaled.columns:
        bearish_ratio = float(np.mean(X_val_unscaled['distance_to_ma200'].values < -0.02))

    if (
        Config.ENABLE_REGIME_HARD_GATE
        and avg_vol_zscore >= Config.HARD_GATE_VOL_ZSCORE
        and bearish_ratio >= Config.HARD_GATE_BEARISH_MA200_RATIO
    ):
        gate_active = True
        gated[:] = 0
        gate_reason = (
            f"vol_z={avg_vol_zscore:.2f} >= {Config.HARD_GATE_VOL_ZSCORE:.2f} "
            f"and bearish_ma200_ratio={bearish_ratio:.1%} >= {Config.HARD_GATE_BEARISH_MA200_RATIO:.1%}"
        )

    return gated, gate_active, gate_reason


def _fit_probability_calibrator(model, X_calib, y_calib, method="isotonic"):
    raw_proba = model.predict_proba(X_calib)[:, 1]
    return fit_probability_calibrator_v2(raw_proba, y_calib, method=method)


def _compute_regime_mask(X_train_unscaled, X_target_unscaled):
    """Compute high-volatility regime mask relative to training distribution."""
    if 'rolling_volatility_zscore' in X_target_unscaled.columns and 'rolling_volatility_zscore' in X_train_unscaled.columns:
        train_z = np.asarray(X_train_unscaled['rolling_volatility_zscore'].values, dtype=float)
        target_z = np.asarray(X_target_unscaled['rolling_volatility_zscore'].values, dtype=float)
        if np.isfinite(train_z).any() and np.isfinite(target_z).any():
            return target_z > Config.REGIME_VOL_Z_THRESHOLD

    train_vol = np.asarray(X_train_unscaled['volatility_20'].values, dtype=float)
    target_vol = np.asarray(X_target_unscaled['volatility_20'].values, dtype=float)
    mu = np.nanmean(train_vol)
    sigma = np.nanstd(train_vol) + 1e-9
    z = (target_vol - mu) / sigma
    return z > Config.REGIME_VOL_Z_THRESHOLD


def tune_xgb_params_for_stability(
    X_train,
    y_train,
    X_val,
    y_val,
    val_returns,
    base_params,
    scale_pos_weight,
    max_candidates: Optional[int] = None,
):
    """
    Lightweight hyperparameter tuning focused on fold stability.

    Objective combines AUCPR, AUC, Sharpe, and minimum trade activity,
    favoring robust behavior over peak single-metric performance.
    """
    candidates = [
        {'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 450, 'subsample': 0.80, 'colsample_bytree': 0.80, 'min_child_weight': 3, 'gamma': 0.30, 'reg_alpha': 0.25, 'reg_lambda': 1.10},
        {'max_depth': 5, 'learning_rate': 0.025, 'n_estimators': 500, 'subsample': 0.82, 'colsample_bytree': 0.82, 'min_child_weight': 3, 'gamma': 0.25, 'reg_alpha': 0.20, 'reg_lambda': 1.00},
        {'max_depth': 6, 'learning_rate': 0.02, 'n_estimators': 600, 'subsample': 0.85, 'colsample_bytree': 0.85, 'min_child_weight': 2, 'gamma': 0.20, 'reg_alpha': 0.20, 'reg_lambda': 1.00},
        {'max_depth': 4, 'learning_rate': 0.02, 'n_estimators': 700, 'subsample': 0.88, 'colsample_bytree': 0.86, 'min_child_weight': 4, 'gamma': 0.35, 'reg_alpha': 0.30, 'reg_lambda': 1.20},
        {'max_depth': 5, 'learning_rate': 0.018, 'n_estimators': 750, 'subsample': 0.84, 'colsample_bytree': 0.84, 'min_child_weight': 2, 'gamma': 0.22, 'reg_alpha': 0.22, 'reg_lambda': 1.05},
        {'max_depth': 3, 'learning_rate': 0.035, 'n_estimators': 400, 'subsample': 0.78, 'colsample_bytree': 0.78, 'min_child_weight': 5, 'gamma': 0.40, 'reg_alpha': 0.35, 'reg_lambda': 1.30},
        {'max_depth': 6, 'learning_rate': 0.015, 'n_estimators': 900, 'subsample': 0.80, 'colsample_bytree': 0.80, 'min_child_weight': 3, 'gamma': 0.28, 'reg_alpha': 0.28, 'reg_lambda': 1.15},
        {'max_depth': 5, 'learning_rate': 0.03, 'n_estimators': 450, 'subsample': 0.90, 'colsample_bytree': 0.88, 'min_child_weight': 2, 'gamma': 0.18, 'reg_alpha': 0.15, 'reg_lambda': 0.95},
    ]
    effective_max = Config.STABILITY_TUNING_MAX_CANDIDATES if max_candidates is None else int(max_candidates)
    candidates = candidates[:effective_max]

    best_score = -np.inf
    best_params = None
    best_meta = {}

    for candidate in candidates:
        params = base_params.copy()
        params.update(candidate)
        params['scale_pos_weight'] = scale_pos_weight

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, proba)
        aucpr = average_precision_score(y_val, proba)

        if Config.TRADING_MODE == "long_only":
            th_buy, th_stats = find_optimal_buy_threshold_sharpe(
                y_val.values,
                proba,
                val_returns,
                min_samples=Config.STABILITY_TUNING_MIN_TRADES,
                min_trades_ratio=Config.THRESHOLD_MIN_TRADES_RATIO,
            )
            preds = apply_probability_threshold(
                proba,
                th_buy,
                mode='threshold',
                trading_mode=Config.TRADING_MODE
            )
        else:
            th_buy, th_sell, th_stats = find_optimal_thresholds_long_short(
                y_val.values,
                proba,
                val_returns,
                min_samples=Config.STABILITY_TUNING_MIN_TRADES
            )
            preds = apply_probability_threshold(
                proba,
                th_buy,
                th_sell,
                mode='threshold',
                trading_mode=Config.TRADING_MODE
            )
        trades = int(np.sum(preds == 1))

        sharpe = th_stats.get('sharpe', 0.0) if th_stats else 0.0
        trade_ratio = trades / max(1, len(y_val))
        trade_quality = min(trade_ratio / 0.25, 1.0)  # full credit if ~25%+ trading activity

        # Stability score: robust ranking, penalize too-few trades and negative Sharpe
        score = (
            0.40 * aucpr
            + 0.20 * auc
            + 0.30 * np.tanh(max(sharpe, 0.0) / 3.0)
            + 0.10 * trade_quality
        )
        if trades < Config.STABILITY_TUNING_MIN_TRADES:
            score -= 0.15
        if sharpe < 0:
            score -= 0.10

        if score > best_score:
            best_score = score
            best_params = params
            best_meta = {
                'auc': float(auc),
                'aucpr': float(aucpr),
                'sharpe': float(sharpe),
                'trades': trades,
                'threshold_buy': float(th_buy),
                'threshold_sell': float(th_sell) if Config.TRADING_MODE == "long_short" else None,
                'score': float(score)
            }

    return best_params if best_params is not None else base_params.copy(), best_meta

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


def sliding_window_splits(n_samples, train_size, test_size, n_splits=8, min_test_size=1):
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
            
            if len(train_indices) >= train_size // 2 and len(test_indices) >= int(min_test_size):
                yield train_indices, test_indices

def summarize_cv_results(fold_results, skipped_folds):
    """Summarize CV outcomes with explicit valid/skipped fold accounting."""
    if not fold_results:
        return {
            'configured_folds': 0,
            'n_folds_total': len(skipped_folds),
            'n_folds_valid': 0,
            'n_folds_skipped': len(skipped_folds),
            'skip_reasons': [f.get('reason', 'unknown') for f in skipped_folds],
            'results_df': pd.DataFrame(),
        }

    results_df = pd.DataFrame(fold_results)
    n_total = len(fold_results) + len(skipped_folds)
    return {
        'configured_folds': n_total,
        'n_folds_total': n_total,
        'n_folds_valid': len(fold_results),
        'n_folds_skipped': len(skipped_folds),
        'skip_reasons': [f.get('reason', 'unknown') for f in skipped_folds],
        'results_df': results_df,
    }


def train_and_evaluate_timeseries(
    df,
    feature_contract: Optional[Dict[str, Any]] = None,
    runtime_flags: Optional[RuntimeFlags] = None,
    stage_profiler: Optional[StageProfiler] = None,
):
    """
    Trains and evaluates the XGBoost model using TimeSeriesSplit
    with proper data leakage prevention:
    - Scalers fit ONLY on training data
    - Embargo period between train and validation
    - Generates visualizations and logs results
    
    Args:
        df (pd.DataFrame): The preprocessed DataFrame (NOT scaled yet).
    """
    runtime_flags = runtime_flags or RuntimeFlags()
    stage_profiler = stage_profiler or StageProfiler(enabled=False)

    print("\n--- Starting Time-Series-Aware Model Training ---")
    print(
        f"Configured folds: {Config.N_SPLITS} | Min validation size: {Config.MIN_VALIDATION_SIZE} "
        f"| Trading mode: {Config.TRADING_MODE} | mode={runtime_flags.mode} | plots={runtime_flags.plots}"
    )

    features = get_feature_columns(df, feature_contract=feature_contract)
    target = 'target'

    X = df[features].copy()
    y = df[target].copy()
    
    if X.empty or y.empty:
        print("❌ Error: No data available for training")
        return {}

    # Initialize visualization and logging engines
    viz_engine = None
    if runtime_flags.plots != "none":
        try:
            try:
                from viewing.visualization_engine import VisualizationEngine as VizEngine
            except ImportError:
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
    use_sliding = runtime_flags.cv_scheme == "sliding"
    if use_sliding:
        print(f"✨ Using SLIDING WINDOW strategy (adaptive, not expanding)")
        print(f"   Train window: {Config.SLIDING_WINDOW_SIZE} days | Test window: {Config.VALIDATION_WINDOW_SIZE} days")
        print(f"   This allows model to adapt faster to market regime changes")
        cv_splits = list(sliding_window_splits_v2(
            len(X),
            train_size=Config.SLIDING_WINDOW_SIZE,
            test_size=Config.VALIDATION_WINDOW_SIZE,
            n_splits=Config.N_SPLITS,
            min_test_size=Config.MIN_VALIDATION_SIZE,
        ))
        print(f"   Generated {len(cv_splits)} sliding window splits")
    else:
        print(f"Using PURGED+EMBARGO TimeSeries CV with {Config.N_SPLITS} splits")
        purged = PurgedEmbargoTimeSeriesSplit(
            n_splits=Config.N_SPLITS,
            test_size=Config.VALIDATION_WINDOW_SIZE,
            purge_size=Config.LABEL_HORIZON_DAYS,
            embargo_size=Config.EMBARGO_DAYS,
            min_train_size=Config.MIN_TRAIN_SIZE,
        )
        cv_splits = list(purged.split(len(X)))
    
    fold_results = []
    skipped_folds = []
    all_val_predictions = []
    all_val_proba = []
    all_val_indices = []
    future_predictions = []  # Store future predictions from each fold
    prev_threshold_buy = Config.PROBABILITY_THRESHOLD_BUY
    prev_threshold_sell = Config.PROBABILITY_THRESHOLD_SELL
    total_return_agg = 0.0
    total_wins_agg = 0
    total_trades_agg = 0
    tuned_params_cache = None
    last_plot_payload = None
    effective_tuning_candidates = Config.STABILITY_TUNING_MAX_CANDIDATES
    if runtime_flags.mode == "fast":
        effective_tuning_candidates = min(3, Config.STABILITY_TUNING_MAX_CANDIDATES)
    calibration_allowed = Config.ENABLE_PROBA_CALIBRATION and runtime_flags.mode == "full"
    cost_rate = runtime_flags.cost_bps / 10000.0
    slippage_rate = runtime_flags.slippage_bps / 10000.0
    conservative_mode = runtime_flags.risk_profile == "conservative"
    leakage_safe_lstm = Config.LSTM_LEAKAGE_SAFE_MODE and runtime_flags.mode == "full"
    lstm_feature_cols = [c for c in features if c.startswith("lstm_")]
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits, start=1):
        fold_start = time.perf_counter()
        # Get train and test indices
        X_train_unscaled = X.iloc[train_idx].copy()
        X_val_unscaled = X.iloc[val_idx].copy()
        y_train = y.iloc[train_idx].copy()
        y_val = y.iloc[val_idx].copy()
        
        # Check embargo period: ensure no overlap when scheme is not sliding
        if not use_sliding:
            embargo_buffer = Config.EMBARGO_DAYS
            if len(train_idx) > embargo_buffer:
                train_idx_embargoed = train_idx[:-embargo_buffer]
                X_train_unscaled = X.iloc[train_idx_embargoed].copy()
                y_train = y.iloc[train_idx_embargoed].copy()
        
        if len(X_train_unscaled) < Config.MIN_TRAIN_SIZE:
            print(f"⚠️ Fold {fold_idx}: Skipped - insufficient training data ({len(X_train_unscaled)} < {Config.MIN_TRAIN_SIZE})")
            skipped_folds.append({'fold': fold_idx, 'reason': 'insufficient_train_size'})
            continue
        
        if len(X_val_unscaled) < Config.MIN_VALIDATION_SIZE:
            print(
                f"⚠️ Fold {fold_idx}: Skipped - insufficient validation data "
                f"({len(X_val_unscaled)} < {Config.MIN_VALIDATION_SIZE})"
            )
            skipped_folds.append({'fold': fold_idx, 'reason': 'insufficient_validation_size'})
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
            skipped_folds.append({'fold': fold_idx, 'reason': 'single_validation_class'})
            continue

        if leakage_safe_lstm and lstm_feature_cols:
            with stage_profiler.timed("lstm_fold"):
                try:
                    fold_idx_union = X_train_unscaled.index.union(X_val_unscaled.index).sort_values()
                    fold_block = df.loc[fold_idx_union].copy()
                    train_block = df.loc[X_train_unscaled.index].copy()
                    fold_lstm_df = generate_lstm_regime_features_train_only(
                        train_block,
                        fold_block,
                        ticker_name=Config.TICKER,
                    )
                    for col in lstm_feature_cols:
                        if col in fold_lstm_df.columns:
                            X_train_unscaled.loc[:, col] = fold_lstm_df.loc[X_train_unscaled.index, col].values
                            X_val_unscaled.loc[:, col] = fold_lstm_df.loc[X_val_unscaled.index, col].values
                except Exception as exc:
                    print(f"⚠️ Fold {fold_idx}: LSTM leakage-safe generation failed ({exc}); using cached LSTM features.")
        
        # === CRITICAL: Scale ONLY on training data ===
        scaler = RobustScaler()
        X_train_scaled = X_train_unscaled.copy()
        X_val_scaled = X_val_unscaled.copy()
        
        X_train_scaled[features] = scaler.fit_transform(X_train_unscaled[features])
        X_val_scaled[features] = scaler.transform(X_val_unscaled[features])
        
        print(f"\n--- Fold {fold_idx}/{len(cv_splits)} (TimeSeriesSplit) ---")
        print(f"Train: {train_date_start} to {train_date_end} ({len(X_train_scaled)} samples)")
        print(f"Valid: {val_date_start} to {val_date_end} ({len(X_val_unscaled)} samples)")
        print(f"Class distribution in validation: {y_val.value_counts().to_dict()}")

        # Validation forward returns for thresholding/tuning diagnostics
        df_val = df.loc[X_val_unscaled.index].copy()
        df_val['close_returns'] = df_val['close'].pct_change().shift(-1)
        val_returns = np.where(np.isnan(df_val['close_returns'].values), 0, df_val['close_returns'].values)
        
        # === CLASS IMBALANCE HANDLING ===
        # Hybrid approach: AGGRESSIVE scale_pos_weight + ADASYN
        # - scale_pos_weight: Penalizes minority class misclassification (amplified 1.5x)
        # - ADASYN: Intelligent synthetic sample generation (focuses on hard examples)
        # Both are complementary, not redundant
        
        # 1. Calculate AGGRESSIVE scale_pos_weight from original (imbalanced) distribution
        class_counts = y_train.value_counts()
        if len(class_counts) == 2:
            base_weight = class_counts[0] / class_counts[1]  # Ratio of negative to positive
            scale_pos_weight = base_weight * 1.5  # 1.5x amplification for extreme imbalance
        else:
            scale_pos_weight = 1.0
        # print(f"   ✅ Original class distribution: {class_counts.to_dict()}")
        # print(f"   ✅ Class weights computed: scale_pos_weight = {scale_pos_weight:.4f} (base={base_weight:.4f}, 1.5x amplified)")
        
        # 2. Optional synthetic balancing (can be disabled for strict time-series realism)
        use_synthetic_sampling = Config.USE_SYNTHETIC_SAMPLING and runtime_flags.risk_profile != "conservative"
        if use_synthetic_sampling:
            minority_class_size = len(y_train[y_train == 0])
            k_neighbors = min(4, max(1, minority_class_size - 1))
            try:
                adasyn = ADASYN(random_state=42, n_neighbors=k_neighbors, sampling_strategy=0.95)
                X_train_smote, y_train_smote = adasyn.fit_resample(X_train_scaled, y_train)
            except (ValueError, RuntimeError):
                try:
                    smote = SMOTE(random_state=42, k_neighbors=k_neighbors, sampling_strategy=0.9)
                    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
                except ValueError:
                    X_train_smote, y_train_smote = X_train_scaled, y_train
        else:
            X_train_smote, y_train_smote = X_train_scaled, y_train
        
        # === TRAIN MODEL WITH STABILITY-ORIENTED HYPERPARAMETERS ===
        xgb_params = Config.XGB_PARAMS.copy()
        xgb_params['scale_pos_weight'] = scale_pos_weight

        should_tune = Config.ENABLE_STABILITY_TUNING and (
            tuned_params_cache is None or not Config.TUNE_ON_FIRST_VALID_FOLD_ONLY
        )

        if should_tune:
            # print(f"   🔧 Stability tuning XGBoost hyperparameters (fold {fold_idx})...")
            tuned_params_cache, tuning_meta = tune_xgb_params_for_stability(
                X_train_smote,
                pd.Series(y_train_smote),
                X_val_scaled,
                y_val,
                val_returns,
                xgb_params,
                scale_pos_weight,
                max_candidates=effective_tuning_candidates,
            )
            xgb_params = tuned_params_cache.copy()
            # print(
            #     f"   ✅ Tuning selected: depth={xgb_params['max_depth']}, "
            #     f"lr={xgb_params['learning_rate']}, n_estimators={xgb_params['n_estimators']} "
            #     f"| score={tuning_meta.get('score', 0):.3f}, aucpr={tuning_meta.get('aucpr', 0):.3f}, "
            #     f"sharpe={tuning_meta.get('sharpe', 0):.2f}, trades={tuning_meta.get('trades', 0)}"
            # )
        elif tuned_params_cache is not None:
            xgb_params = tuned_params_cache.copy()
            xgb_params['scale_pos_weight'] = scale_pos_weight
        
        if fold_idx == 1:
            pass
            # print(f"   📊 Using optimized hyperparameters for percentile labeling:")
            # print(f"      max_depth={xgb_params['max_depth']}, lr={xgb_params['learning_rate']}, n_estimators={xgb_params['n_estimators']}")
        
        # Train final model(s)
        model = xgb.XGBClassifier(**xgb_params)
        calibration_enabled = False
        calibration_method_used = "none"
        calibration_reason = "disabled_by_mode_or_config"

        if Config.ENABLE_REGIME_SPLIT_MODEL:
            train_high_mask = _compute_regime_mask(X_train_unscaled, X_train_unscaled)
            n_high = int(np.sum(train_high_mask))
            n_low = int(len(train_high_mask) - n_high)

            if n_high >= Config.MIN_TRAIN_SIZE // 2 and n_low >= Config.MIN_TRAIN_SIZE // 2:
                model_low = xgb.XGBClassifier(**xgb_params)
                model_high = xgb.XGBClassifier(**xgb_params)

                X_train_low = X_train_scaled.loc[~train_high_mask]
                y_train_low = y_train.loc[~train_high_mask]
                X_train_high = X_train_scaled.loc[train_high_mask]
                y_train_high = y_train.loc[train_high_mask]

                model_low.fit(X_train_low, y_train_low, verbose=False)
                model_high.fit(X_train_high, y_train_high, verbose=False)

                val_high_mask = _compute_regime_mask(X_train_unscaled, X_val_unscaled)
                proba_preds = np.zeros(len(X_val_scaled), dtype=float)
                if np.any(~val_high_mask):
                    proba_preds[~val_high_mask] = model_low.predict_proba(X_val_scaled.loc[~val_high_mask])[:, 1]
                if np.any(val_high_mask):
                    proba_preds[val_high_mask] = model_high.predict_proba(X_val_scaled.loc[val_high_mask])[:, 1]
                model = model_low  # default model for downstream visualizations
            else:
                model.fit(
                    X_train_smote, y_train_smote,
                    eval_set=[(X_val_scaled, y_val)],
                    verbose=False
                )
                proba_preds = model.predict_proba(X_val_scaled)[:, 1]
        else:
            model.fit(
                X_train_smote, y_train_smote,
                eval_set=[(X_val_scaled, y_val)],
                verbose=False
            )
            proba_preds = model.predict_proba(X_val_scaled)[:, 1]

        fold_calibration_error = np.nan
        fold_brier = np.nan
        if calibration_allowed and not Config.ENABLE_REGIME_SPLIT_MODEL and len(X_train_scaled) >= Config.CALIBRATION_MIN_SAMPLES:
            method = Config.CALIBRATION_METHOD if len(X_train_scaled) >= 300 else "sigmoid"
            X_train_np = np.asarray(X_train_scaled[features], dtype=float)
            X_val_calib = np.asarray(X_val_scaled[features], dtype=float)
            raw_train_proba = model.predict_proba(X_train_np)[:, 1]
            split_data = train_calibration_split(X_train_np, y_train.values, raw_train_proba, ratio=0.2)
            if split_data is not None:
                proba_calib, y_calib = split_data
                calibrator, calibration_reason = fit_probability_calibrator_v2(proba_calib, y_calib, method=method)
                if calibrator is not None:
                    raw_val_proba = model.predict_proba(X_val_calib)[:, 1]
                    proba_preds = calibrator.predict_proba(raw_val_proba)[:, 1]
                    calibration_enabled = True
                    calibration_method_used = method
                    cal_stats = calibration_metrics(y_val.values, proba_preds)
                    fold_calibration_error = cal_stats["ece"]
                    fold_brier = cal_stats["brier"]
                else:
                    calibration_method_used = "none"
            else:
                calibration_reason = "invalid_calibration_split"

        # === RISK-ADJUSTED LOSS: Apply volatility weighting to predictions ===
        # Extract volatility from validation data for risk adjustment
        # High volatility → require higher confidence for trades (more conservative)
        val_volatility = X_val_unscaled['volatility_20'].values
        volatility_mean = val_volatility.mean()
        volatility_std = val_volatility.std()
        # Normalize volatility to [0.5, 2.5] range (0.5 = low vol aggressive, 2.5 = high vol conservative)
        volatility_weights = 1 + (val_volatility - volatility_mean) / (volatility_std + 1e-8)
        volatility_weights = np.clip(volatility_weights, 0.5, 2.5)  # Baseline bounds
        
        # Apply volatility-weighted risk adjustment (conservative during high volatility)
        proba_preds_risk_adjusted = proba_preds / volatility_weights
        proba_preds_risk_adjusted = np.clip(proba_preds_risk_adjusted, 0, 1)
        preds = (proba_preds_risk_adjusted >= 0.5).astype(int)
        
        # Apply probability thresholding with intelligent position opening strategy
        threshold_buy = Config.PROBABILITY_THRESHOLD_BUY
        threshold_sell = Config.PROBABILITY_THRESHOLD_SELL if Config.TRADING_MODE == "long_short" else None
        threshold_stats = None
        
        if Config.ENABLE_THRESHOLD_GRIDSEARCH:
            if Config.OPTIMIZE_BY_SHARPE:
                if Config.TRADING_MODE == "long_only":
                    threshold_buy, threshold_stats = find_optimal_buy_threshold_sharpe(
                        y_val.values,
                        proba_preds_risk_adjusted,
                        val_returns,
                        min_samples=10,
                        min_trades_ratio=Config.THRESHOLD_MIN_TRADES_RATIO,
                        cost_rate=cost_rate,
                        slippage_rate=slippage_rate,
                        max_trades_ratio=Config.CONSERVATIVE_MAX_TRADE_RATIO if conservative_mode else None,
                    )
                else:
                    threshold_buy, threshold_sell, threshold_stats = find_optimal_thresholds_long_short(
                        y_val.values,
                        proba_preds_risk_adjusted,
                        val_returns,
                        min_samples=10
                    )
            else:
                # Original F1 optimization
                threshold_buy, threshold_stats = find_optimal_threshold(
                    y_val.values,
                    proba_preds_risk_adjusted,
                    metric=Config.THRESHOLD_OPTIM_METRIC,
                    min_recall=Config.THRESHOLD_MIN_RECALL,
                    min_precision=Config.THRESHOLD_MIN_PRECISION
                )
        
        # Apply adaptive thresholds based on market volatility
        if Config.USE_ADAPTIVE_THRESHOLDS:
            # Get volatility regime relative to TRAINING window (prevents self-centering bug)
            train_volatility = X_train_unscaled['volatility_20'].values if 'volatility_20' in X_train_unscaled.columns else val_volatility
            train_vol_mean = np.nanmean(train_volatility)
            train_vol_std = np.nanstd(train_volatility) + 1e-8
            avg_vol_zscore = ((val_volatility - train_vol_mean) / train_vol_std).mean()

            adaptive_buy, adaptive_sell = get_adaptive_thresholds(
                avg_vol_zscore, 
                base_buy=threshold_buy, 
                base_sell=threshold_sell if threshold_sell is not None else Config.PROBABILITY_THRESHOLD_SELL
            )

            # Smooth abrupt fold-to-fold jumps via EMA
            threshold_buy, smoothed_sell = smooth_thresholds(
                adaptive_buy,
                adaptive_sell,
                prev_threshold_buy,
                prev_threshold_sell,
                alpha=Config.ADAPTIVE_THRESHOLD_SMOOTHING_ALPHA
            )
            # print(
            #     f"   🎯 Adaptive thresholds smoothed (vol_zscore={avg_vol_zscore:.2f}): "
            #     f"BUY={threshold_buy:.3f}, SELL={threshold_sell:.3f}"
            # )

            if Config.TRADING_MODE == "long_short":
                threshold_sell = smoothed_sell
                prev_threshold_sell = threshold_sell
            prev_threshold_buy = threshold_buy
        else:
            avg_vol_zscore = 0.0

        if conservative_mode:
            threshold_buy = max(float(threshold_buy), float(Config.CONSERVATIVE_MIN_BUY_THRESHOLD))

        # Apply intelligent position opening strategy
        # print(f"   📍 Position Opening: {Config.POSITION_OPENING_MODE.upper()} mode")
        effective_percentile_threshold = (
            Config.CONSERVATIVE_PERCENTILE_THRESHOLD if conservative_mode else Config.PERCENTILE_THRESHOLD
        )

        if Config.POSITION_OPENING_MODE == "hybrid":
            # Hybrid mode: threshold + percentile + technical validation
            preds_thresholded = apply_probability_threshold(
                proba_preds_risk_adjusted,
                threshold_buy,
                threshold_sell if threshold_sell is not None else Config.PROBABILITY_THRESHOLD_SELL,
                X_features=X_val_unscaled,  # Pass validation features for technical filters
                percentile_threshold=effective_percentile_threshold,
                mode='hybrid',
                trading_mode=Config.TRADING_MODE
            )
        else:
            # Simple threshold or percentile mode
            preds_thresholded = apply_probability_threshold(
                proba_preds_risk_adjusted,
                threshold_buy,
                threshold_sell if threshold_sell is not None else Config.PROBABILITY_THRESHOLD_SELL,
                X_features=None,
                percentile_threshold=effective_percentile_threshold,
                mode=Config.POSITION_OPENING_MODE,
                trading_mode=Config.TRADING_MODE
            )
        
        # Log position statistics with trading details
        preds_thresholded, gate_active, gate_reason = apply_regime_hard_gate(
            preds_thresholded,
            X_val_unscaled,
            avg_vol_zscore
        )
        if gate_active:
            print(f"   🛡️ Regime hard-gate active: {gate_reason}")

        n_positions = np.sum(preds_thresholded == 1)
        position_rate = 100 * n_positions / len(preds_thresholded)
        print(f"   ✅ Positions opened: {n_positions}/{len(preds_thresholded)} ({position_rate:.1f}%)")

        # Dynamic position sizing (skip uncertain regimes/signals)
        if Config.ENABLE_POSITION_SIZING:
            position_sizes, size_stats = compute_position_sizes(
                proba_preds_risk_adjusted,
                threshold_buy,
                volatility_zscore=avg_vol_zscore,
                eligible_mask=(preds_thresholded == 1),
                min_confidence=Config.CONSERVATIVE_MIN_CONFIDENCE if conservative_mode else Config.POSITION_SIZE_MIN_CONFIDENCE,
                uncertainty_band=Config.CONSERVATIVE_UNCERTAINTY_BAND if conservative_mode else Config.POSITION_SIZE_UNCERTAINTY_BAND,
                vol_penalty=(
                    Config.POSITION_SIZE_VOL_RISK_PENALTY * Config.CONSERVATIVE_VOL_PENALTY_MULT
                    if conservative_mode
                    else Config.POSITION_SIZE_VOL_RISK_PENALTY
                ),
            )
            print(
                f"   📏 Position sizing: sized={size_stats['n_sized']}/{size_stats['n_candidates']} "
                f"| skipped uncertain={size_stats['n_skipped_uncertain']} "
                f"| skipped low_conf={size_stats['n_skipped_low_conf']} "
                f"| avg size={size_stats['avg_size']:.3f}"
            )
        else:
            position_sizes = preds_thresholded.astype(float)
        
        # === BACKTEST SUMMARY (Historical Performance) ===
        df_val = df.loc[X_val_unscaled.index].copy() if X_val_unscaled is not None else df.tail(len(preds_thresholded)).copy()
        
        # Winning trades from backtest (only non-zero sized positions count)
        active_positions = position_sizes > 0
        wins = active_positions & (y_val == 1)
        losses = active_positions & (y_val == 0)
        
        if wins.sum() > 0 or losses.sum() > 0:
            win_rate = wins.sum() / (wins.sum() + losses.sum()) if (wins.sum() + losses.sum()) > 0 else 0
            print(f"\n   📊 Backtest Results (Validation Period):")
            print(f"      Period: {df_val.index[0].strftime('%Y-%m-%d')} to {df_val.index[-1].strftime('%Y-%m-%d')}")
            print(f"      Winning trades: {wins.sum()}/{wins.sum() + losses.sum()} ({win_rate*100:.1f}% win rate)")
            if losses.sum() > 0:
                print(f"      Losing trades: {losses.sum()}")
            print(f"      Avoided losses: {((preds_thresholded == 0) & (y_val == 0)).sum()} (correctly predicted losers)")

        diagnose_fold_market_conditions(
            fold_idx,
            X_train_unscaled,
            X_val_unscaled,
            y_val,
            proba_preds_risk_adjusted,
            preds_thresholded,
            position_sizes
        )
        
        # Calculate metrics (using risk-adjusted probabilities for consistency)
        auc = roc_auc_score(y_val, proba_preds_risk_adjusted)
        aucpr = average_precision_score(y_val, proba_preds_risk_adjusted)
        accuracy = (preds == y_val).mean()
        accuracy_thresholded = (preds_thresholded == y_val).mean()
        precision = np.sum((preds_thresholded == 1) & (y_val == 1)) / (np.sum(preds_thresholded == 1) + 1e-6)
        recall = np.sum((preds_thresholded == 1) & (y_val == 1)) / (np.sum(y_val == 1) + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        
        # === STRATEGY RETURNS CALCULATION ===
        # (val_returns already calculated above for Sharpe optimization)
        
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
        position_sizes_valid = position_sizes[valid_mask] if valid_mask.sum() > 0 else position_sizes
        strategy_returns_thresholded = val_returns_valid * position_sizes_valid
        net_strategy_returns, turnover_series = apply_trading_costs(
            val_returns_valid,
            position_sizes_valid,
            cost_rate=cost_rate,
            slippage_rate=slippage_rate,
        )
        net_stats = evaluate_net_metrics(
            val_returns_valid,
            position_sizes_valid,
            cost_rate=cost_rate,
            slippage_rate=slippage_rate,
        )
        
        # === IMPROVED SHARPE/DRAWDOWN CALCULATION ===
        # Calculate for both raw and thresholded predictions
        n_long_positions = np.sum(position_sizes_valid > 0)
        n_long_positions_raw = np.sum(preds_valid == 1)
        valid_days_count = len(preds_thresholded_valid)

        if n_long_positions < Config.MIN_TRADES_PER_FOLD:
            print(
                f"⚠️ Fold {fold_idx}: Skipped from aggregate metrics - insufficient trades "
                f"({n_long_positions} < {Config.MIN_TRADES_PER_FOLD})"
            )
            skipped_folds.append({'fold': fold_idx, 'reason': 'insufficient_trades'})
            stage_profiler.stages["cv_fold"] = stage_profiler.stages.get("cv_fold", 0.0) + (time.perf_counter() - fold_start)
            continue
        
        # Sharpe & Drawdown for thresholded strategy (actual trading strategy)
        if n_long_positions >= 3:  # Need at least 3 positions for meaningful Sharpe
            sharpe_thresholded = calculate_sharpe_ratio(strategy_returns_thresholded)
            max_dd = calculate_max_drawdown(strategy_returns_thresholded)
            net_sharpe = calculate_sharpe_ratio(net_strategy_returns)
        else:
            sharpe_thresholded = np.nan
            max_dd = np.nan
            net_sharpe = np.nan
        
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
            win_rate = np.sum(net_strategy_returns > 0) / n_long_positions
            total_return = np.sum(strategy_returns_thresholded)
            net_return = np.sum(net_strategy_returns)
        else:
            win_rate = 0
            total_return = 0
            net_return = 0
        total_return_agg += float(total_return)
        total_wins_agg += int(np.sum(strategy_returns_thresholded > 0))
        total_trades_agg += int(n_long_positions)
        
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
            'threshold_sell': float(threshold_sell) if threshold_sell is not None else np.nan,
            'threshold_metric': 'sharpe' if Config.OPTIMIZE_BY_SHARPE else Config.THRESHOLD_OPTIM_METRIC if Config.ENABLE_THRESHOLD_GRIDSEARCH else 'static',
            'adaptive_thresholds': Config.USE_ADAPTIVE_THRESHOLDS,
            'fold_total_return': float(total_return),
            'net_return': float(net_return),
            'net_sharpe': float(net_stats.get("net_sharpe", net_sharpe)),
            'turnover': float(net_stats.get("turnover", np.nansum(turnover_series))),
            'fold_win_rate': float(win_rate),
            'fold_trades': int(n_long_positions),
            'n_trades_at_threshold': int(threshold_stats.get('n_trades', 0)) if threshold_stats else int(n_long_positions),
            'trade_ratio_at_threshold': float(threshold_stats.get('trade_ratio', 0.0)) if threshold_stats else float(n_long_positions / max(1, len(preds_thresholded))),
            'calibration_enabled': bool(calibration_enabled),
            'calibration_method': calibration_method_used,
            'calibration_reason': calibration_reason,
            'calibration_error': float(fold_calibration_error) if np.isfinite(fold_calibration_error) else np.nan,
            'brier': float(fold_brier) if np.isfinite(fold_brier) else np.nan,
        })
        
        # Store predictions for visualization
        all_val_predictions.append(preds)
        all_val_proba.append(proba_preds)
        all_val_indices.append(val_idx)
        
        if threshold_stats:
            if Config.OPTIMIZE_BY_SHARPE:
                if Config.TRADING_MODE == "long_short":
                    print(
                        f"  → Thresholds (sharpe): BUY={threshold_buy:.4f}, SELL={threshold_sell:.4f} "
                        f"| Sharpe: {threshold_stats['sharpe']:.2f} | WinRate: {threshold_stats['win_rate']:.1%}"
                    )
                else:
                    print(
                        f"  → Threshold (sharpe): BUY={threshold_buy:.4f} "
                        f"| Sharpe: {threshold_stats['sharpe']:.2f} | WinRate: {threshold_stats['win_rate']:.1%}"
                    )
            else:
                print(f"  → Threshold tuned ({Config.THRESHOLD_OPTIM_METRIC}): {threshold_buy:.4f} | PR: {threshold_stats['precision']:.3f}/{threshold_stats['recall']:.3f} | F1: {threshold_stats['f1']:.3f}")
        print(f"  → AUC: {auc:.4f} | AUCPR: {aucpr:.4f} | Accuracy: {accuracy:.4f} ({accuracy_thresholded:.4f} w/ threshold)")
        print(f"  → Sharpe: {sharpe:.4f} ({sharpe_thresholded:.4f} w/ threshold) | Max DD: {max_dd:.4f}")

        # Always compute fold-level future probability so ensemble is stable
        # regardless of plotting mode (none/final/all).
        X_recent = X_val_unscaled.tail(30) if len(X_val_unscaled) >= 30 else X_val_unscaled
        X_recent_scaled = scaler.transform(X_recent)
        recent_proba = model.predict_proba(X_recent_scaled)[:, 1]
        if len(recent_proba) > 0:
            future_predictions.append({
                'fold': fold_idx,
                'probability': float(recent_proba[-1]),
                'date': df.index[-1]
            })
        
        # Generate visualizations for this fold (all) or queue last payload (final)
        if viz_engine and runtime_flags.plots in {"all", "final"}:
            plot_payload = {
                "fold_idx": fold_idx,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "run_number": run_number,
                "run_date": run_date,
                "model": model,
                "scaler": scaler,
                "X_val_unscaled": X_val_unscaled,
                "proba_preds": proba_preds,
            }
            if runtime_flags.plots == "all":
                print(f"   Generating visualizations for Fold {fold_idx}...")
            else:
                last_plot_payload = plot_payload
            if runtime_flags.plots == "all":
                plot_start = time.perf_counter()
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
                        X_recent=X_val_unscaled.tail(30) if len(X_val_unscaled) >= 30 else X_val_unscaled
                    )
                except Exception as e:
                    print(f"   ⚠️ Visualization error: {e}")
                finally:
                    stage_profiler.stages["plots"] = stage_profiler.stages.get("plots", 0.0) + (time.perf_counter() - plot_start)
        else:
            print(f"   Skipping visualizations (mode={runtime_flags.plots})")

        stage_profiler.stages["cv_fold"] = stage_profiler.stages.get("cv_fold", 0.0) + (time.perf_counter() - fold_start)

        # Keep original recommendation prints only in full plot mode.
        if viz_engine and runtime_flags.plots == "all":
            try:
                if future_predictions:
                    future_prob = future_predictions[-1]["probability"]
                    print(f"\n   " + "="*60)
                    print(f"   🎯 TRADING RECOMMENDATION FOR NEXT 20 DAYS")
                    print(f"   " + "="*60)
                    last_price = df['close'].iloc[-1]
                    last_date = df.index[-1]
                    if Config.DYNAMIC_RISK_VOL_METRIC == 'atr_14d' and 'atr' in df.columns:
                        latest_volatility = float(pd.to_numeric(df['atr'], errors='coerce').ffill().iloc[-1])
                        vol_metric_used = 'atr_14d'
                    elif 'volatility_20' in df.columns:
                        latest_volatility = float(pd.to_numeric(df['volatility_20'], errors='coerce').ffill().iloc[-1])
                        vol_metric_used = 'rolling_std_20d'
                    else:
                        latest_volatility = float(pd.to_numeric(df['close'], errors='coerce').rolling(window=20).std().ffill().iloc[-1])
                        vol_metric_used = 'rolling_std_20d'

                    profit_target = last_price + (latest_volatility * Config.DYNAMIC_RISK_K_TP)
                    stop_loss = last_price - (latest_volatility * Config.DYNAMIC_RISK_K_SL)
                    tp_pct = ((profit_target - last_price) / last_price) * 100
                    sl_pct = ((stop_loss - last_price) / last_price) * 100
                    
                    print(f"   📅 Valid from: {last_date.strftime('%Y-%m-%d')}")
                    print(f"   💰 Entry Price: ${last_price:.2f}")
                    print(f"   🎚️ Dynamic Risk: {Config.DYNAMIC_RISK_TYPE} | metric={vol_metric_used} | k_tp={Config.DYNAMIC_RISK_K_TP} | k_sl={Config.DYNAMIC_RISK_K_SL}")
                    print(f"   🎯 Profit Target: ${profit_target:.2f} ({tp_pct:+.2f}%)")
                    print(f"   🛑 Stop Loss: ${stop_loss:.2f} ({sl_pct:+.2f}%)")
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

    if viz_engine and runtime_flags.plots == "final" and last_plot_payload is not None:
        print(f"   Generating final visualization (fold {last_plot_payload['fold_idx']})...")
        plot_start = time.perf_counter()
        try:
            val_idx = last_plot_payload["val_idx"]
            X_val_unscaled = last_plot_payload["X_val_unscaled"]
            proba_preds = last_plot_payload["proba_preds"]
            scaler = last_plot_payload["scaler"]
            model = last_plot_payload["model"]
            fold_idx = last_plot_payload["fold_idx"]
            run_number_plot = last_plot_payload["run_number"]
            run_date_plot = last_plot_payload["run_date"]

            viz_engine.plot_lstm_xgboost_hybrid(
                df=df,
                train_data=last_plot_payload["train_idx"],
                test_data=val_idx,
                lstm_preds=df.loc[X_val_unscaled.index, 'lstm_return_5d'].values if 'lstm_return_5d' in df.columns else np.zeros(len(val_idx)),
                xgb_preds=proba_preds,
                scaler=scaler,
                run_date=run_date_plot,
                run_number=f"{run_number_plot}_fold{fold_idx}"
            )
            future_prob = viz_engine.plot_future_forecast(
                df=df,
                lstm_preds=df.loc[X_val_unscaled.index, 'lstm_return_5d'].values if 'lstm_return_5d' in df.columns else np.zeros(len(val_idx)),
                xgb_preds=proba_preds,
                forecast_horizon=20,
                run_date=run_date_plot,
                run_number=f"{run_number_plot}_fold{fold_idx}",
                latest_model=model,
                scaler=scaler,
                X_recent=X_val_unscaled.tail(30) if len(X_val_unscaled) >= 30 else X_val_unscaled
            )
        except Exception as e:
            print(f"   ⚠️ Final visualization error: {e}")
        finally:
            stage_profiler.stages["plots"] = stage_profiler.stages.get("plots", 0.0) + (time.perf_counter() - plot_start)

    # === DISPLAY FINAL RESULTS ===
    if not fold_results:
        print("\n❌ No valid folds to evaluate. Check data and parameters.")
        return {
            "future_predictions": [],
            "metrics": {},
            "cv": {"n_folds_valid": 0, "n_folds_skipped": len(skipped_folds)},
            "artifacts": {},
            "timings": stage_profiler.summary(),
        }
    
    cv_summary = summarize_cv_results(fold_results, skipped_folds)
    results_df = cv_summary['results_df']
    coverage_ratio = float(cv_summary['n_folds_valid'] / max(1, cv_summary['configured_folds']))
    if coverage_ratio < Config.MIN_COVERAGE_RATIO:
        print(
            f"⚠️ Coverage ratio below target: {coverage_ratio:.3f} < {Config.MIN_COVERAGE_RATIO:.3f}. "
            "Treat recommendation as low reliability."
        )
    print(
        f"\nConfigured folds: {cv_summary['configured_folds']} | "
        f"Executed valid folds: {cv_summary['n_folds_valid']} | "
        f"Skipped: {cv_summary['n_folds_skipped']}"
    )
    print("\n" + "="*80)
    print("--- CROSS-VALIDATION SUMMARY (Valid Folds) ---")
    print("="*80)
    print(results_df.to_string(index=False))

    if skipped_folds:
        print("\n--- SKIPPED FOLDS REPORT ---")
        for skipped in skipped_folds:
            print(f"Fold {skipped.get('fold')}: {skipped.get('reason', 'unknown')}")
    
    print("\n--- AGGREGATE STATISTICS ---")
    print(f"AUC:                  {results_df['auc'].mean():.4f} ± {results_df['auc'].std():.4f}")
    print(f"AUCPR:                {results_df['aucpr'].mean():.4f} ± {results_df['aucpr'].std():.4f}")
    print(f"Accuracy:             {results_df['accuracy'].mean():.4f} ± {results_df['accuracy'].std():.4f}")
    print(f"Accuracy (threshold): {results_df['accuracy_thresholded'].mean():.4f} ± {results_df['accuracy_thresholded'].std():.4f}")
    print(f"Sharpe Ratio:         {results_df['sharpe'].mean():.4f} ± {results_df['sharpe'].std():.4f}")
    print(f"Sharpe (threshold):   {results_df['sharpe_thresholded'].mean():.4f} ± {results_df['sharpe_thresholded'].std():.4f}")
    if 'net_sharpe' in results_df:
        print(f"Net Sharpe:           {results_df['net_sharpe'].mean():.4f} ± {results_df['net_sharpe'].std():.4f}")
    if 'net_return' in results_df:
        print(f"Net Return:           {results_df['net_return'].mean():.4f} ± {results_df['net_return'].std():.4f}")
    if 'turnover' in results_df:
        print(f"Turnover:             {results_df['turnover'].mean():.4f} ± {results_df['turnover'].std():.4f}")
    print(f"Coverage Ratio:       {coverage_ratio:.4f}")
    print(f"Max Drawdown:         {results_df['max_drawdown'].mean():.4f} ± {results_df['max_drawdown'].std():.4f}")
    print("="*80)

    # Log results to accumulative JSON log
    print("\n📊 Logging results to accumulative JSON...")
    recommendation_quality = np.nan
    if future_predictions:
        pvals = np.array([p.get("probability", 0.5) for p in future_predictions], dtype=float)
        recommendation_quality = float(np.clip(1.0 - (np.std(pvals) * 4.0), 0.0, 1.0))
    run_config = {
        'run_number': run_number,
        'run_date': run_date.split('_')[0],
        'ticker': Config.TICKER,
        'features_used': features,
        'train_size': len(X_train_scaled) if fold_results else 0,
        'test_size': sum(len(idx) for idx in all_val_indices) if all_val_indices else 0,
        'fold_number': cv_summary['n_folds_valid'],
        'cv': {
            'n_folds_configured': cv_summary['configured_folds'],
            'n_folds_total': cv_summary['n_folds_total'],
            'n_folds_valid': cv_summary['n_folds_valid'],
            'n_folds_skipped': cv_summary['n_folds_skipped'],
            'skip_reasons': cv_summary['skip_reasons'],
            'coverage_ratio': coverage_ratio,
        },
        'features_contract': feature_contract or {},
        'metrics': {
            'accuracy': float(results_df['accuracy'].mean()),
            'accuracy_std': float(results_df['accuracy'].std()),
            'precision': float(results_df['precision'].mean()),
            'recall': float(results_df['recall'].mean()),
            'f1': float(results_df['f1'].mean()),
            'auc': float(results_df['auc'].mean()),
            'auc_std': float(results_df['auc'].std()),
            'auc_pr': float(results_df['aucpr'].mean()),
            'auc_pr_std': float(results_df['aucpr'].std()),
            'sharpe_thresholded_mean': float(results_df['sharpe_thresholded'].mean()),
            'sharpe_thresholded_std': float(results_df['sharpe_thresholded'].std()),
            'threshold_buy_mean': float(results_df['threshold_buy'].mean()) if 'threshold_buy' in results_df else Config.PROBABILITY_THRESHOLD_BUY,
            'min_fold_trades': int(results_df['fold_trades'].min()) if 'fold_trades' in results_df else 0,
            'median_fold_trades': float(results_df['fold_trades'].median()) if 'fold_trades' in results_df else 0.0,
            'worst_fold_auc': float(results_df['auc'].min()) if 'auc' in results_df else 0.0,
            'worst_fold_aucpr': float(results_df['aucpr'].min()) if 'aucpr' in results_df else 0.0,
            'std_fold_sharpe_thresholded': float(results_df['sharpe_thresholded'].std()) if 'sharpe_thresholded' in results_df else 0.0,
            'calibration_enabled': bool(results_df['calibration_enabled'].any()) if 'calibration_enabled' in results_df else False,
            'calibration_method': str(results_df['calibration_method'].mode().iloc[0]) if 'calibration_method' in results_df and not results_df['calibration_method'].mode().empty else 'none',
            'calibration_error': float(results_df['calibration_error'].mean()) if 'calibration_error' in results_df else np.nan,
            'brier': float(results_df['brier'].mean()) if 'brier' in results_df else np.nan,
        },
        'sharpe_ratio': float(results_df['sharpe'].mean()),
        'net_sharpe': float(results_df['net_sharpe'].mean()) if 'net_sharpe' in results_df else np.nan,
        'max_drawdown': float(results_df['max_drawdown'].mean()),
        'total_return': float(total_return_agg),
        'net_return': float(results_df['net_return'].sum()) if 'net_return' in results_df else np.nan,
        'turnover': float(results_df['turnover'].sum()) if 'turnover' in results_df else np.nan,
        'recommendation_quality': recommendation_quality,
        'win_rate': float(total_wins_agg / total_trades_agg) if total_trades_agg > 0 else 0.0,
        'xgb_params': Config.XGB_PARAMS,
        'lstm_config': {
            'window_size': 60,
            'latent_dim': 32,
            'horizons': [5, 10, 20]
        },
        'buy_signals': int(np.sum([np.sum(p > 0.65) for p in all_val_proba])),
        'sell_signals': int(np.sum([np.sum(p < 0.35) for p in all_val_proba])) if Config.TRADING_MODE == "long_short" else 0,
        'hold_signals': int(np.sum([np.sum((p >= 0.35) & (p <= 0.65)) for p in all_val_proba])),
        'visualization_file': f'out/lstm_xgboost_hybrid_run_{run_number}_*' if runtime_flags.plots != "none" else '',
        'forecast_file': f'out/future_forecast_run_{run_number}_*' if runtime_flags.plots != "none" else '',
        'runtime': {
            'mode': runtime_flags.mode,
            'plots': runtime_flags.plots,
            'cache': runtime_flags.cache,
            'profile': runtime_flags.profile,
            'objective': runtime_flags.objective,
            'risk_profile': runtime_flags.risk_profile,
            'cost_bps': runtime_flags.cost_bps,
            'slippage_bps': runtime_flags.slippage_bps,
            'cv_scheme': runtime_flags.cv_scheme,
        },
        'timings': stage_profiler.summary(),
        'notes': (
            f"Configured folds={cv_summary['configured_folds']}, valid={cv_summary['n_folds_valid']}, "
            f"min_validation={Config.MIN_VALIDATION_SIZE}, trading_mode={Config.TRADING_MODE}, "
            f"mode={runtime_flags.mode}, plots={runtime_flags.plots}"
        )
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
        
        # Show low-importance features that could be eliminated
        low_importance = importance_df[importance_df['importance'] < 0.01]
        if not low_importance.empty:
            print(f"\n⚠️ Features with <1% importance ({len(low_importance)} total):")
            print(f"   {', '.join(low_importance['feature'].head(10).tolist())}...")
            print(f"   💡 Consider removing these in future runs for better performance")
    except Exception as e:
        print(f"\nCould not generate feature importances: {e}")
    
    return {
        "future_predictions": future_predictions,
        "metrics": run_config.get("metrics", {}),
        "cv": run_config.get("cv", {}),
        "artifacts": {
            "visualization_file": run_config.get("visualization_file", ""),
            "forecast_file": run_config.get("forecast_file", ""),
        },
        "timings": stage_profiler.summary(),
    }

def _build_run_config(runtime: RuntimeFlags, ticker: str, start_date: str) -> RunConfig:
    return RunConfig(
        data=DataConfig(
            ticker=ticker,
            vix_ticker=Config.VIX_TICKER,
            start_date=start_date,
            end_date=Config.END_DATE,
        ),
        features=FeatureConfig(min_features=Config.MIN_FEATURES),
        labeling=LabelingConfig(
            tp_multiplier=Config.LABEL_TP_MULTIPLIER,
            sl_multiplier=Config.LABEL_SL_MULTIPLIER,
            horizon_days=Config.LABEL_HORIZON_DAYS,
        ),
        training=TrainingConfig(
            n_splits=Config.N_SPLITS,
            min_validation_size=Config.MIN_VALIDATION_SIZE,
            use_sliding_window=(runtime.cv_scheme == "sliding"),
            sliding_window_size=Config.SLIDING_WINDOW_SIZE,
            validation_window_size=Config.VALIDATION_WINDOW_SIZE,
        ),
        risk=RiskConfig(
            dynamic_risk_type=Config.DYNAMIC_RISK_TYPE,
            dynamic_risk_k_tp=Config.DYNAMIC_RISK_K_TP,
            dynamic_risk_k_sl=Config.DYNAMIC_RISK_K_SL,
            dynamic_risk_vol_metric=Config.DYNAMIC_RISK_VOL_METRIC,
            cost_bps=runtime.cost_bps,
            slippage_bps=runtime.slippage_bps,
        ),
        runtime=runtime,
    )


def run_pipeline(run_config: RunConfig, selected_name: Optional[str] = None):
    cache = CacheManager(enabled=run_config.runtime.cache_enabled)
    profiler = StageProfiler(enabled=run_config.runtime.profile_enabled)
    Config.TICKER = run_config.data.ticker

    main_df = load_market_data(
        fetch_fn=fetch_data,
        ticker=run_config.data.ticker,
        vix_ticker=run_config.data.vix_ticker,
        start_date=run_config.data.start_date,
        end_date=run_config.data.end_date,
        runtime=run_config.runtime,
        cache=cache,
        profiler=profiler,
    )
    if main_df.empty:
        ui.show_error("Failed to fetch initial data. Check tickers and network connection.")
        return {}

    main_df, feature_contract = run_feature_pipeline(
        df=main_df,
        ticker=run_config.data.ticker,
        runtime=run_config.runtime,
        cache=cache,
        profiler=profiler,
        generate_lstm_fn=generate_lstm_regime_features,
        engineer_features_fn=engineer_features,
        create_advanced_features_fn=create_advanced_features,
        remove_correlated_features_fn=remove_correlated_features,
        filter_low_importance_features_fn=filter_low_importance_features,
        resolve_feature_contract_fn=resolve_feature_contract,
        seed_features=Config.FEATURES_TO_SCALE,
        min_features=run_config.features.min_features,
        leakage_safe_lstm=(Config.LSTM_LEAKAGE_SAFE_MODE and run_config.runtime.mode == "full"),
        placeholder_lstm_fn=generate_placeholder_lstm_features,
    )

    print(f"\n📊 Feature engineering summary:")
    print(f"   • Requested seed features: {len(feature_contract['requested'])}")
    print(f"   • Available from seed: {len(feature_contract['available'])}")
    print(f"   • Correlation-based drops: {len(feature_contract['removed_by_corr'])}")
    print(f"   • Low-importance drops: {len(feature_contract['removed_by_importance'])}")
    print(f"   • Final contract features: {len(feature_contract['final'])}")
    if feature_contract['missing_unexpected']:
        print(f"   • Unexpected missing: {feature_contract['missing_unexpected']}")

    labeled_df, processed_df = run_labeling_pipeline(
        df=main_df,
        feature_contract=feature_contract,
        runtime=run_config.runtime,
        cache=cache,
        profiler=profiler,
        define_target_fn=define_target,
        preprocess_data_raw_fn=preprocess_data_raw,
        label_signature={
            "tp": run_config.labeling.tp_multiplier,
            "sl": run_config.labeling.sl_multiplier,
            "horizon": run_config.labeling.horizon_days,
        },
    )

    if processed_df.empty:
        ui.show_error("No data left after preprocessing. Check data quality and date ranges.")
        return {}

    print("\n👑 --- Launching XGBoost with Time-Series-Aware Cross-Validation --- 👑")
    train_result = run_training_pipeline(
        train_fn=train_and_evaluate_timeseries,
        df=processed_df,
        feature_contract=feature_contract,
        runtime=run_config.runtime,
        profiler=profiler,
    )

    future_preds = train_result.get("future_predictions", [])
    if future_preds:
        last_price = labeled_df['close'].iloc[-1]
        last_date = labeled_df.index[-1]
        vol_info = latest_volatility_for_risk(labeled_df, run_config.risk.dynamic_risk_vol_metric)
        profit_target = last_price + (vol_info["volatility"] * run_config.risk.dynamic_risk_k_tp)
        stop_loss = last_price - (vol_info["volatility"] * run_config.risk.dynamic_risk_k_sl)
        dynamic_risk_config = build_dynamic_risk_config(
            run_config.risk.dynamic_risk_type,
            run_config.risk.dynamic_risk_k_tp,
            run_config.risk.dynamic_risk_k_sl,
            vol_info["metric"],
        )
        prob_summary = summarize_future_predictions(future_preds)
        tp_pct = (profit_target - last_price) / max(1e-9, last_price)
        sl_pct = (stop_loss - last_price) / max(1e-9, last_price)
        dynamic_threshold = {
            "conservative": 0.68,
            "balanced": 0.62,
            "aggressive": 0.55,
        }.get(run_config.runtime.risk_profile, 0.68)
        cost_pct = (run_config.runtime.cost_bps + run_config.runtime.slippage_bps) / 10000.0
        decision = conservative_recommendation(
            probability=prob_summary["avg_prob"],
            dynamic_threshold=dynamic_threshold,
            tp_pct=float(tp_pct),
            sl_pct=float(sl_pct),
            cost_pct=cost_pct,
            regime_allowed=True,
        )

        ui.show_final_summary(run_config.data.ticker, future_preds)
        ui.show_recommendation(
            prob_summary["avg_prob"],
            last_price,
            profit_target,
            stop_loss,
            last_date.strftime('%Y-%m-%d'),
            dynamic_risk_config=dynamic_risk_config
        )
        print(
            f"   🧭 Decision state: {decision.state} | EV={decision.expected_value:.5f} | "
            f"Quality={decision.recommendation_quality:.3f} | reason={decision.reasons}"
        )
        ui.show_success("Pipeline execution completed successfully!")

    if run_config.runtime.profile_enabled:
        print("\n⏱️ Stage profiling summary:")
        for stage, seconds in profiler.summary().items():
            print(f"   - {stage}: {seconds:.3f}s")
    ui.show_success("Pipeline finished.")

    return {
        "train_result": train_result,
        "profile": profiler.summary(),
    }


def _parse_cli_args(argv=None):
    parser = argparse.ArgumentParser(description="ForecastTRADE runtime")
    parser.add_argument("--ticker", default=None, help="Ticker symbol (if omitted, interactive UI is used)")
    parser.add_argument("--mode", default="full", choices=["full", "fast"], help="Runtime mode")
    parser.add_argument("--plots", default="all", choices=["none", "final", "all"], help="Plot generation mode")
    parser.add_argument("--cache", default="on", choices=["on", "off"], help="Enable/disable local cache")
    parser.add_argument("--profile", default="on", choices=["on", "off"], help="Enable/disable profiling")
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    parser.add_argument("--objective", default=None, choices=["sharpe_net", "return", "max_winrate"], help="Primary optimization objective")
    parser.add_argument("--risk-profile", default=None, choices=["conservative", "balanced", "aggressive"], help="Risk policy profile")
    parser.add_argument("--cost-bps", default=None, type=float, help="Transaction cost in bps")
    parser.add_argument("--slippage-bps", default=None, type=float, help="Slippage in bps")
    parser.add_argument("--cv-scheme", default=None, choices=["sliding", "purged"], help="Cross-validation scheme")
    parser.add_argument("--no-ui", action="store_true", help="Disable interactive ticker selection")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_cli_args(argv)
    external = load_external_config(args.config)
    apply_config_overrides(Config, external)
    set_global_seed(int(getattr(Config, "SEED", 42)))
    runtime = RuntimeFlags.normalize(
        args.mode,
        args.plots,
        args.cache,
        args.profile,
        objective=args.objective or getattr(Config, "OBJECTIVE", "sharpe_net"),
        risk_profile=args.risk_profile or getattr(Config, "RISK_PROFILE", "conservative"),
        cost_bps=float(args.cost_bps if args.cost_bps is not None else getattr(Config, "COST_BPS", 20.0)),
        slippage_bps=float(args.slippage_bps if args.slippage_bps is not None else getattr(Config, "SLIPPAGE_BPS", 5.0)),
        cv_scheme=args.cv_scheme or getattr(Config, "CV_SCHEME", "sliding"),
        config_path=args.config,
    )

    selected_ticker = args.ticker
    selected_name = args.ticker

    if not args.no_ui:
        ui.display_header()
        if not selected_ticker:
            ui.select_asset()
            selected_ticker = ui.selected_ticker
            selected_name = ui.selected_name

    if not selected_ticker:
        selected_ticker = Config.TICKER
        selected_name = selected_ticker

    start_date = (datetime.today() - timedelta(days=Config.TRAINING_YEARS * 365)).strftime('%Y-%m-%d')
    run_config = _build_run_config(runtime, selected_ticker, start_date)
    return run_pipeline(run_config, selected_name=selected_name)


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    main()
