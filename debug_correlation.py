#!/usr/bin/env python3
"""
Debug: Check correlation between LSTM features and top XGBoost features
"""
import pandas as pd
import numpy as np
import yfinance as yf
from src.lstm_predictor import generate_lstm_regime_features

# Fetch data
ticker = "NVDA"
df = yf.download(ticker, start="2020-01-01", progress=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df.columns = [col.lower() for col in df.columns]
df = df[['open', 'high', 'low', 'close', 'volume']]

# Generate LSTM features
df_lstm = generate_lstm_regime_features(df.copy(), ticker)

# Calculate EMA features (like in main.py)
df_lstm['ema_short'] = df_lstm['close'].ewm(span=12).mean()
df_lstm['ema_long'] = df_lstm['close'].ewm(span=26).mean()

# Get VIX data
vix = yf.download("^VIX", start="2020-01-01", progress=False)
if isinstance(vix.columns, pd.MultiIndex):
    vix.columns = vix.columns.get_level_values(0)
vix.columns = [col.lower() + '_vix' for col in vix.columns]
df_lstm = df_lstm.join(vix['close_vix'], how='left')
df_lstm['close_vix'].fillna(method='ffill', inplace=True)
df_lstm['close_vix_ratio'] = df_lstm['close'] / df_lstm['close_vix']

# Select features to analyze
top_xgb_features = ['close_vix_ratio', 'ema_short', 'ema_long']
lstm_features = [f'lstm_latent_{i}' for i in range(10)]

# Drop NaNs
df_clean = df_lstm[top_xgb_features + lstm_features].dropna()

print("\n🔍 Correlation Analysis: LSTM vs Top XGBoost Features")
print("="*70)

# Correlation matrix
print("\n📊 Correlation Matrix (LSTM features vs Top XGBoost features):")
corr_matrix = df_clean.corr().loc[lstm_features, top_xgb_features]
print(corr_matrix.round(3))

print("\n⚠️ High correlations (|r| > 0.7):")
for lstm_feat in lstm_features:
    for xgb_feat in top_xgb_features:
        corr = corr_matrix.loc[lstm_feat, xgb_feat]
        if abs(corr) > 0.7:
            print(f"  {lstm_feat} <-> {xgb_feat}: {corr:.3f}")

print("\n📈 Variance comparison:")
for feat in top_xgb_features + lstm_features:
    print(f"  {feat:25s}: std={df_clean[feat].std():.6f}, range={df_clean[feat].max()-df_clean[feat].min():.6f}")

# Check if LSTM features change over time
print("\n📅 Time-series analysis of LSTM features (first 100 vs last 100):")
for feat in lstm_features:
    first_100_mean = df_clean[feat].iloc[:100].mean()
    last_100_mean = df_clean[feat].iloc[-100:].mean()
    change = abs(last_100_mean - first_100_mean)
    print(f"  {feat}: first_100={first_100_mean:.6f}, last_100={last_100_mean:.6f}, change={change:.6f}")
