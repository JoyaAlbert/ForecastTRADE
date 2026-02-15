# -*- coding: utf-8 -*-
"""
Advanced Feature Engineering Module
====================================

Implements:
1. Fractional Differentiation (preserves memory while achieving stationarity)
2. Market Regime Features (volatility clusters, VWAP deviations)
3. Recursive Feature Elimination (RFE) for optimal feature selection
"""

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')


def fractional_differentiation(series, d=0.5, size=10):
    """
    Fractional differentiation to achieve stationarity while preserving memory.
    
    This addresses a key issue in financial ML: standard differencing (d=1) removes
    all memory, while raw prices are non-stationary. Fractional d finds the sweet spot.
    
    Args:
        series (pd.Series): Input time series (e.g., log prices)
        d (float): Fractional differentiation order (0 < d < 1)
                  d=0.5 is a good default balancing stationarity and memory
        size (int): Size of the differentiation window
        
    Returns:
        pd.Series: Fractionally differentiated series
    """
    # Weights for fractional differentiation
    weights = [1.0]
    for i in range(1, size):
        weight = -weights[-1] * (d - i + 1) / i
        weights.append(weight)
    
    weights = np.array(weights)
    
    # Apply convolution
    diff_series = pd.Series(index=series.index, dtype=np.float64)
    for i in range(len(series)):
        if i < len(weights):
            # Not enough history yet
            diff_series.iloc[i] = np.nan
        else:
            diff_series.iloc[i] = np.dot(weights, series.iloc[i-len(weights)+1:i+1].values)
    
    return diff_series


def add_fractional_differentiation_features(df):
    """
    Add fractionally differentiated price features to detect trends
    while preserving short-term memory.
    
    Args:
        df (pd.DataFrame): Input dataframe with 'close' column
        
    Returns:
        pd.DataFrame: Dataframe with new fd_* columns
    """
    print("   Applying fractional differentiation to prices...")
    
    log_prices = np.log(df['close'])
    
    # Create fractionally differentiated versions with different d values
    for d in [0.3, 0.5, 0.7]:
        fd_series = fractional_differentiation(log_prices, d=d)
        df[f'fd_price_{int(d*10)}'] = fd_series
    
    return df


def calculate_vwap_deviation(df, window=20):
    """
    Calculate Volume Weighted Average Price (VWAP) and its deviation.
    Used to detect when price deviates significantly from volume-weighted average.
    
    Args:
        df (pd.DataFrame): Input dataframe with 'close' and 'volume' columns
        window (int): Rolling window for VWAP calculation
        
    Returns:
        pd.DataFrame: Dataframe with new vwap_* columns
    """
    print("   Calculating VWAP deviation features...")
    
    # Standard VWAP
    df['vwap'] = (df['close'] * df['volume']).rolling(window).sum() / df['volume'].rolling(window).sum()
    df['vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap']
    
    # VWAP with different windows
    for w in [5, 10, 20]:
        vwap_w = (df['close'] * df['volume']).rolling(w).sum() / df['volume'].rolling(w).sum()
        df[f'vwap_{w}d_deviation'] = (df['close'] - vwap_w) / vwap_w
    
    return df


def calculate_garch_volatility(series, window=20, alpha=0.1, beta=0.85):
    """
    Simple GARCH(1,1) volatility estimation.
    Captures volatility clustering - a key regime feature.
    
    Args:
        series (pd.Series): Log returns series
        window (int): Initial window for estimation
        alpha (float): GARCH alpha parameter
        beta (float): GARCH beta parameter
        
    Returns:
        np.array: Conditional volatility series
    """
    returns = series.values
    sigma = np.zeros_like(returns)
    
    # Initialize with rolling window volatility
    sigma[:window] = returns[:window].std()
    
    # GARCH(1,1) recursion
    variance = sigma[window]**2
    for t in range(window, len(returns)):
        variance = alpha * returns[t-1]**2 + beta * variance
        sigma[t] = np.sqrt(np.abs(variance))  # Abs to avoid numerical issues
    
    return sigma


def add_volatility_regime_features(df):
    """
    Add features that capture volatility regimes and clustering.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with new vol_regime_* columns
    """
    print("   Calculating volatility regime features...")
    
    log_returns = np.log(df['close'] / df['close'].shift(1))
    
    # GARCH volatility
    garch_vol = calculate_garch_volatility(log_returns.dropna(), window=20)
    df['garch_volatility'] = np.nan
    df.iloc[1:len(garch_vol)+1, df.columns.get_loc('garch_volatility')] = garch_vol
    
    # Volatility mean reversion indicator
    vol_rolling = df['volatility_20'].rolling(window=20).mean()
    df['vol_vs_mean'] = df['volatility_20'] / vol_rolling
    
    # Volatility regime (high/low/medium)
    vol_percentile = df['volatility_20'].rolling(window=60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
    )
    df['vol_regime'] = vol_percentile
    
    return df


def perform_rfe_feature_selection(X, y, n_features_to_select=12, cv_folds=5):
    """
    Perform Recursive Feature Elimination using XGBoost estimator.
    Reduces feature set to the most predictive features.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target labels
        n_features_to_select (int): Number of features to keep
        cv_folds (int): Number of CV folds for stability
        
    Returns:
        list: Names of selected features
    """
    print(f"   Performing RFE to select top {n_features_to_select} features...")
    
    # Use RandomForest as RFE estimator (efficient and interpretable)
    estimator = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    
    try:
        # RFE with cross-validation
        rfe = RFE(estimator, n_features_to_select=n_features_to_select, step=1)
        rfe.fit(X, y)
        
        selected_features = X.columns[rfe.support_].tolist()
        
        print(f"   ✅ RFE selected {len(selected_features)} features")
        print(f"      Selected: {', '.join(selected_features[:5])}...")
        
        return selected_features
    except Exception as e:
        print(f"   ⚠️ RFE failed: {e}. Returning all features.")
        return X.columns.tolist()


def apply_advanced_feature_engineering(df):
    """
    Applies all advanced feature engineering techniques.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with all advanced features
    """
    print("\n🔬 --- Advanced Feature Engineering Module --- 🔬")
    
    df = df.copy()
    
    # 1. Fractional Differentiation
    df = add_fractional_differentiation_features(df)
    
    # 2. VWAP Deviation (Market Structure)
    df = calculate_vwap_deviation(df)
    
    # 3. Volatility Regime Features
    df = add_volatility_regime_features(df)
    
    # Handle NaNs created by new features
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    print(f"   ✅ Advanced features generated. Total columns: {len(df.columns)}")
    
    return df
