#!/usr/bin/env python3
"""
Debug script to investigate why LSTM features have zero importance
"""
import pandas as pd
import numpy as np
import yfinance as yf
from src.lstm_predictor import generate_lstm_regime_features

# Fetch data
ticker = "NVDA"
df = yf.download(ticker, start="2020-01-01", progress=False)
# Handle multi-level columns
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df.columns = [col.lower() for col in df.columns]
df = df[['open', 'high', 'low', 'close', 'volume']]

print(f"\n📊 Initial data shape: {df.shape}")
print(f"Date range: {df.index[0]} to {df.index[-1]}")

# Generate LSTM features
df_with_lstm = generate_lstm_regime_features(df.copy(), ticker)

# Check LSTM latent features
latent_cols = [col for col in df_with_lstm.columns if col.startswith('lstm_latent_')]
print(f"\n🔍 LSTM Latent Features Found: {len(latent_cols)}")
print(f"Columns: {latent_cols}")

# Check statistics for each latent feature
print("\n📈 Statistics for LSTM Latent Features:")
for col in latent_cols:
    data = df_with_lstm[col].dropna()
    print(f"\n{col}:")
    print(f"  Count: {len(data)}")
    print(f"  Mean: {data.mean():.6f}")
    print(f"  Std: {data.std():.6f}")
    print(f"  Min: {data.min():.6f}")
    print(f"  Max: {data.max():.6f}")
    print(f"  Unique values: {data.nunique()}")
    print(f"  All same? {data.nunique() == 1}")
    
    # Check for zeros or constant values
    if data.std() < 1e-6:
        print(f"  ⚠️ WARNING: {col} has near-zero variance!")
    
    # Show sample values
    print(f"  Sample (last 5): {data.tail().values}")

# Check correlation between latent features
print("\n🔗 Correlation Matrix (first 5 latent features):")
if len(latent_cols) >= 5:
    corr = df_with_lstm[latent_cols[:5]].corr()
    print(corr.round(3))

# Compare with other features
print("\n📊 Comparison with other features:")
other_features = ['close', 'volume']
for feat in other_features:
    if feat in df_with_lstm.columns:
        data = df_with_lstm[feat].dropna()
        print(f"\n{feat}:")
        print(f"  Mean: {data.mean():.6f}")
        print(f"  Std: {data.std():.6f}")
        print(f"  Unique values: {data.nunique()}")
