# -*- coding: utf-8 -*-
"""
LSTM Price Predictor - Feature Generation
===========================================

This script trains an LSTM model to predict multi-horizon returns.
Instead of forecasting, it uses these predictions to generate features
for a downstream model like XGBoost.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2

class LstmConfig:
    """Configuration for the LSTM feature generator."""
    WINDOW_SIZE = 60  # Shorter window for faster processing, better for feature gen
    HORIZONS = [5, 10, 20]  # Predict 5, 10, 20 days ahead
    EPOCHS = 50
    BATCH_SIZE = 32
    REGULARIZATION_L2 = 0.001

def engineer_lstm_features(df):
    """Prepares a focused set of features for the LSTM model."""
    df = df.copy()
    
    # Use a smaller, more focused feature set for the LSTM
    # to reduce noise and computational load.
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility_20'] = df['log_returns'].rolling(window=20).std()
    df['price_to_sma20'] = df['close'] / df['close'].rolling(window=20).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    features = ['returns', 'volatility_20', 'price_to_sma20', 'rsi']
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    return df[features].copy(), features

def create_return_targets(df, horizons):
    """Create future return targets for training."""
    targets = {}
    for h in horizons:
        targets[f'return_{h}d'] = df['returns'].shift(-h)
    return pd.DataFrame(targets)

def create_sequences(data, targets, window_size):
    """Create X and y sequences for the LSTM."""
    X, y = [], []
    for i in range(len(data) - window_size - max(LstmConfig.HORIZONS)):
        X.append(data[i:(i + window_size)])
        y.append(targets[i + window_size])
    return np.array(X), np.array(y)

def build_feature_lstm(input_shape, n_outputs):
    """Builds a multi-output LSTM optimized for feature generation."""
    inputs = Input(shape=input_shape)
    x = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=L2(LstmConfig.REGULARIZATION_L2)))(inputs)
    x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(32, return_sequences=False, kernel_regularizer=L2(LstmConfig.REGULARIZATION_L2)))(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(n_outputs, name='return_predictions')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def generate_lstm_regime_features(df, ticker_name='AAPL'):
    """
    Trains an LSTM on the input data and uses it to generate features across
    the entire dataframe.
    """
    print("🧠 --- LSTM Module: Generating Features --- 🧠")
    
    # 1. Prepare data for LSTM
    df_lstm, feature_names = engineer_lstm_features(df)
    if len(df_lstm) < LstmConfig.WINDOW_SIZE + max(LstmConfig.HORIZONS):
        print("⚠️ Not enough data for LSTM feature generation. Skipping.")
        for h in LstmConfig.HORIZONS:
            df[f'lstm_price_{h}d'] = df['close']
        return df

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_lstm)
    
    targets = create_return_targets(df_lstm, LstmConfig.HORIZONS)
    X, y = create_sequences(scaled_data, targets.values, LstmConfig.WINDOW_SIZE)

    if len(X) < 100:
        print("⚠️ Not enough sequences for LSTM. Skipping.")
        for h in LstmConfig.HORIZONS:
            df[f'lstm_price_{h}d'] = df['close']
        return df

    # 2. Build and train LSTM
    print(f"   Training LSTM feature generator on {len(X)} sequences...")
    model = build_feature_lstm(input_shape=(LstmConfig.WINDOW_SIZE, len(feature_names)), n_outputs=len(LstmConfig.HORIZONS))
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    ]
    
    model.fit(X, y, epochs=LstmConfig.EPOCHS, batch_size=LstmConfig.BATCH_SIZE,
              validation_split=0.1, callbacks=callbacks, verbose=0)
    print("   ✅ LSTM training completed.")

    # 3. Generate features for the entire dataset
    # We create sequences for all data points to generate features for them.
    full_sequences = []
    for i in range(len(scaled_data) - LstmConfig.WINDOW_SIZE + 1):
        full_sequences.append(scaled_data[i:i+LstmConfig.WINDOW_SIZE])
    
    if not full_sequences:
        print("⚠️ Could not create sequences for prediction. Skipping LSTM features.")
        for h in LstmConfig.HORIZONS:
            df[f'lstm_price_{h}d'] = df['close']
        return df

    # Predict returns for all sequences
    predicted_returns = model.predict(np.array(full_sequences), verbose=0)
    
    # 4. Map predictions back to the original dataframe
    # The predictions align with the end of each window.
    pred_start_idx = LstmConfig.WINDOW_SIZE - 1
    
    for i, h in enumerate(LstmConfig.HORIZONS):
        # Create columns for predicted returns and prices
        return_col_name = f'lstm_return_{h}d'
        price_col_name = f'lstm_price_{h}d'
        
        # Initialize columns with NaN
        df[return_col_name] = np.nan
        
        # Place predictions into the correct rows
        df.iloc[pred_start_idx:pred_start_idx + len(predicted_returns), df.columns.get_loc(return_col_name)] = predicted_returns[:, i]
        
        # Calculate predicted price based on the return
        # The predicted return for day T is for day T+h. So we base the price on day T's close.
        df[price_col_name] = df['close'] * (1 + df[return_col_name])

    # The LSTM's purpose is to add features. The main df is now enriched.
    # We forward-fill the generated features to handle NaNs at the beginning.
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)

    print("   ✅ LSTM features generated and added to dataframe.")
    return df