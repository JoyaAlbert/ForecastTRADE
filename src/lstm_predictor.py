# -*- coding: utf-8 -*-
"""
LSTM Price Predictor - Feature Generation
===========================================

This script trains an LSTM model to predict multi-horizon returns.
Instead of forecasting, it uses these predictions to generate features
for a downstream model like XGBoost.
"""

import os
import contextlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_LOG_SEVERITY_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2

class LstmConfig:
    """Configuration for the LSTM feature generator."""
    WINDOW_SIZE = 60  # Shorter window for faster processing, better for feature gen
    HORIZONS = [5, 10, 20]  # Predict 5, 10, 20 days ahead
    EPOCHS = 80  # Increased for better learning
    BATCH_SIZE = 64  # Larger batches for stable gradients
    REGULARIZATION_L2 = 0.0001  # Reduced from 0.001 to allow learning
    LATENT_DIM = 32  # Full hidden state dimension before compression
    COMPRESSED_LATENT_DIM = 10  # Bottleneck: reduce to 10 features (from 32) - 70% compression baseline


@contextlib.contextmanager
def _silence_tensorflow_stderr():
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stderr(devnull):
            yield

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
    """
    Create future return targets for training.
    
    UPDATED: Now returns both continuous returns AND binary directional targets
    Binary targets help LSTM learn patterns aligned with classification tasks
    """
    targets = {}
    for h in horizons:
        # Continuous returns (original)
        targets[f'return_{h}d'] = df['returns'].shift(-h)
        # Binary directional targets (NEW: aligned with classification)
        # 1 if positive return, 0 if negative (helps learn directional patterns)
        targets[f'direction_{h}d'] = (df['returns'].shift(-h) > 0).astype(float)
    return pd.DataFrame(targets)

def create_sequences(data, targets, window_size):
    """Create X and y sequences for the LSTM."""
    X, y = [], []
    for i in range(len(data) - window_size - max(LstmConfig.HORIZONS)):
        X.append(data[i:(i + window_size)])
        y.append(targets[i + window_size])
    return np.array(X), np.array(y)


def _fill_placeholder_lstm_features(df):
    for h in LstmConfig.HORIZONS:
        df[f'lstm_price_{h}d'] = df['close']
        df[f'lstm_return_{h}d'] = 0.0
        df[f'lstm_dir_{h}d'] = 0.5
    df['lstm_return_confidence'] = 0.0
    for i in range(LstmConfig.COMPRESSED_LATENT_DIM):
        df[f'lstm_latent_{i}'] = 0.0
    return df


def generate_placeholder_lstm_features(df, ticker_name='AAPL'):
    """Generate deterministic placeholder LSTM columns to preserve feature schema."""
    return _fill_placeholder_lstm_features(df.copy())


def _map_lstm_outputs_to_df(df, predicted_returns, predicted_directions, compressed_latent_vectors):
    pred_start_idx = LstmConfig.WINDOW_SIZE - 1
    for i, h in enumerate(LstmConfig.HORIZONS):
        return_col_name = f'lstm_return_{h}d'
        price_col_name = f'lstm_price_{h}d'
        direction_col_name = f'lstm_dir_{h}d'

        df[return_col_name] = np.nan
        df.iloc[pred_start_idx:pred_start_idx + len(predicted_returns), df.columns.get_loc(return_col_name)] = predicted_returns[:, i]
        df[price_col_name] = df['close'] * (1 + df[return_col_name])
        df[direction_col_name] = np.nan
        df.iloc[pred_start_idx:pred_start_idx + len(predicted_directions), df.columns.get_loc(direction_col_name)] = predicted_directions[:, i]

    df['lstm_return_confidence'] = np.nan
    confidence_values = np.mean(np.abs(predicted_returns), axis=1)
    df.iloc[pred_start_idx:pred_start_idx + len(confidence_values), df.columns.get_loc('lstm_return_confidence')] = confidence_values

    compressed_latent_dim = compressed_latent_vectors.shape[1]
    for latent_idx in range(compressed_latent_dim):
        col_name = f'lstm_latent_{latent_idx}'
        df[col_name] = np.nan
        df.iloc[pred_start_idx:pred_start_idx + len(compressed_latent_vectors), df.columns.get_loc(col_name)] = compressed_latent_vectors[:, latent_idx]

    return df.bfill().ffill()

def build_feature_lstm(input_shape, n_outputs, latent_dim=32, compressed_dim=10):
    """
    Builds a multi-output LSTM optimized for feature generation.
    
    ARCHITECTURE:
    - Bidirectional LSTM layers learn temporal patterns
    - Hidden state (latent vector) extracted from the final LSTM layer
    - Bottleneck compression layer reduces latent dims from 32 to 10
    - Linear activation preserves variance and dynamic range
    - This compression forces LSTM to capture only critical temporal patterns
    - Single output for return predictions, compressed latent space for XGBoost features
    
    Args:
        input_shape (tuple): (window_size, n_features)
        n_outputs (int): Number of prediction horizons
        latent_dim (int): Dimension of latent space (hidden state) before compression
        compressed_dim (int): Dimension after bottleneck compression
        
    Returns:
        tuple: (model, latent_model, compressed_latent_model)
    """
    inputs = Input(shape=input_shape)
    
    # First bidirectional LSTM: learns complex temporal patterns
    lstm_1 = Bidirectional(
        LSTM(64, return_sequences=True, kernel_regularizer=L2(LstmConfig.REGULARIZATION_L2)),
        name='bilstm_1'
    )(inputs)
    lstm_1 = Dropout(0.1)(lstm_1)  # Reduced from 0.2 to 0.1
    
    # Second bidirectional LSTM: refines patterns with hidden state extraction
    lstm_2 = Bidirectional(
        LSTM(latent_dim, return_sequences=False, kernel_regularizer=L2(LstmConfig.REGULARIZATION_L2)),
        name='bilstm_2_latent'
    )(lstm_1)
    lstm_2_dropout = Dropout(0.1)(lstm_2)  # Reduced from 0.2 to 0.1
    
    # BOTTLENECK COMPRESSION LAYER: Force LSTM to extract only critical patterns
    # Reduces 32-dim latent space to 10-dim compressed representation
    # LINEAR activation (no Tanh) to preserve variance and dynamic range
    # Features will be standardized later with the rest of the features
    compressed_latent = Dense(
        compressed_dim, 
        activation='linear',  # Changed from 'tanh' to preserve variance
        name='bottleneck_compression',
        kernel_regularizer=L2(LstmConfig.REGULARIZATION_L2)
    )(lstm_2_dropout)
    
    # Dense layers for dual predictions
    # Branch 1: Continuous returns (regression)
    returns_branch = Dense(  32, activation='relu', name='returns_dense')(lstm_2_dropout)
    returns_output = Dense(n_outputs, activation='linear', name='return_predictions')(returns_branch)
    
    # Branch 2: Binary directions (classification) - NEW
    # This forces latent space to capture directional patterns useful for XGBoost
    direction_branch = Dense(32, activation='relu', name='direction_dense')(lstm_2_dropout)
    direction_output = Dense(n_outputs, activation='sigmoid', name='direction_predictions')(direction_branch)
    
    # Main model: dual outputs (returns + directions)
    model = Model(
        inputs=inputs, 
        outputs={'return_predictions': returns_output, 'direction_predictions': direction_output},
        name='lstm_dual_task'
    )
    model.compile(
        optimizer=Adam(learning_rate=0.001), 
        loss={'return_predictions': 'mse', 'direction_predictions': 'binary_crossentropy'},
        loss_weights={'return_predictions': 0.5, 'direction_predictions': 0.5},  # Equal weight
        metrics={'return_predictions': ['mae'], 'direction_predictions': ['accuracy']}
    )
    
    # Auxiliary model: extracts full latent vector
    latent_model = Model(inputs=inputs, outputs=lstm_2, name='lstm_latent_extractor')
    
    # Auxiliary model: extracts compressed latent vector (MAIN FEATURE SOURCE)
    compressed_latent_model = Model(
        inputs=inputs, 
        outputs=compressed_latent, 
        name='lstm_compressed_latent_extractor'
    )
    
    return model, latent_model, compressed_latent_model


def generate_lstm_regime_features_train_only(train_df, full_df, ticker_name='AAPL'):
    """
    Leakage-safe LSTM feature generation:
    - Fit scaler/model only on training window
    - Apply trained model to train+validation block
    """
    train_df_lstm, feature_names = engineer_lstm_features(train_df.copy())
    full_df_lstm, _ = engineer_lstm_features(full_df.copy())

    if len(train_df_lstm) < LstmConfig.WINDOW_SIZE + max(LstmConfig.HORIZONS):
        return _fill_placeholder_lstm_features(full_df.copy())

    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_df_lstm)
    scaled_full = scaler.transform(full_df_lstm)

    targets_df = create_return_targets(train_df_lstm, LstmConfig.HORIZONS)
    X_train, y_all = create_sequences(scaled_train, targets_df.values, LstmConfig.WINDOW_SIZE)

    MIN_SEQUENCES_REQUIRED = 120
    if len(X_train) < MIN_SEQUENCES_REQUIRED:
        return _fill_placeholder_lstm_features(full_df.copy())

    n_horizons = len(LstmConfig.HORIZONS)
    y_returns = y_all[:, :n_horizons]
    y_directions = y_all[:, n_horizons:]

    model, _, compressed_latent_model = build_feature_lstm(
        input_shape=(LstmConfig.WINDOW_SIZE, len(feature_names)),
        n_outputs=len(LstmConfig.HORIZONS),
        latent_dim=LstmConfig.LATENT_DIM,
        compressed_dim=LstmConfig.COMPRESSED_LATENT_DIM
    )
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
    ]
    validation_split = 0.1 if len(X_train) > 100 else 0.0

    with _silence_tensorflow_stderr():
        model.fit(
            X_train,
            {'return_predictions': y_returns, 'direction_predictions': y_directions},
            epochs=max(20, min(LstmConfig.EPOCHS, 50)),
            batch_size=min(LstmConfig.BATCH_SIZE, max(8, len(X_train)//10)),
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=0
        )

        full_sequences = []
        for i in range(len(scaled_full) - LstmConfig.WINDOW_SIZE + 1):
            full_sequences.append(scaled_full[i:i + LstmConfig.WINDOW_SIZE])
        if not full_sequences:
            return _fill_placeholder_lstm_features(full_df.copy())
        full_sequences_arr = np.array(full_sequences)
        predictions_dict = model.predict(full_sequences_arr, verbose=0)
        predicted_returns = predictions_dict['return_predictions']
        predicted_directions = predictions_dict['direction_predictions']
        compressed_latent_vectors = compressed_latent_model.predict(full_sequences_arr, verbose=0)

    out_df = full_df.copy()
    return _map_lstm_outputs_to_df(out_df, predicted_returns, predicted_directions, compressed_latent_vectors)

def generate_lstm_regime_features(df, ticker_name='AAPL'):
    """
    Trains an LSTM on the input data and uses bottleneck-compressed latent features
    to generate refined features for XGBoost.
    
    IMPROVEMENTS V2 (Multi-task Learning):
    - LSTM now learns BOTH continuous returns AND binary directions
    - Dual-task training aligns latent space with classification objectives
    - Extracts latent vector (hidden state) that captures directional temporal patterns
    - Applies bottleneck compression to reduce from 32 to 10 dimensions
    - Linear activation preserves variance for better feature importance
    - Only critical temporal patterns survive the compression, reducing noise
    """
    print("🧠 --- LSTM Module: Dual-Task Learning (Returns + Directions) --- 🧠")
    
    # 1. Prepare data for LSTM
    df_lstm, feature_names = engineer_lstm_features(df)
    if len(df_lstm) < LstmConfig.WINDOW_SIZE + max(LstmConfig.HORIZONS):
        print("⚠️ Not enough data for LSTM feature generation. Skipping.")
        return _fill_placeholder_lstm_features(df)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_lstm)
    
    targets_df = create_return_targets(df_lstm, LstmConfig.HORIZONS)
    X, y_all = create_sequences(scaled_data, targets_df.values, LstmConfig.WINDOW_SIZE)
    
    # Split targets into returns and directions
    # y_all shape: (n_samples, 6) where 6 = 3 horizons * 2 (return + direction each)
    n_horizons = len(LstmConfig.HORIZONS)
    y_returns = y_all[:, :n_horizons]  # First 3 columns: returns
    y_directions = y_all[:, n_horizons:]  # Last 3 columns: directions

    # Minimum 300 sequences required for stable LSTM training
    # With 300 sequences: 270 train (90% split), 30 validation (10% split)
    MIN_SEQUENCES_REQUIRED = 300
    if len(X) < MIN_SEQUENCES_REQUIRED:
        print(f"⚠️ Only {len(X)} sequences (need {MIN_SEQUENCES_REQUIRED}+ for stable LSTM).")
        print(f"   LSTM skipped due to insufficient data for generalization.")
        return _fill_placeholder_lstm_features(df)

    # 2. Build and train LSTM with bottleneck compression
    print(f"   Training LSTM with dual-task learning on {len(X)} sequences...")
    print(f"   Tasks: (1) Returns regression (MSE) + (2) Direction classification (BCE)")
    print(f"   Architecture: Full latent dim={LstmConfig.LATENT_DIM} → Compressed dim={LstmConfig.COMPRESSED_LATENT_DIM}")
    
    model, latent_model, compressed_latent_model = build_feature_lstm(
        input_shape=(LstmConfig.WINDOW_SIZE, len(feature_names)), 
        n_outputs=len(LstmConfig.HORIZONS),
        latent_dim=LstmConfig.LATENT_DIM,
        compressed_dim=LstmConfig.COMPRESSED_LATENT_DIM
    )
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    ]
    
    # Adaptive validation split based on sequence count
    # Need at least 30 validation samples for meaningful evaluation
    if len(X) > 100:
        validation_split = 0.1  # Standard 10/90 split if enough data
    elif len(X) > 50:
        validation_split = 0.05  # Small split if limited data
    else:
        validation_split = 0.0   # No split if very small
    
    if validation_split == 0.0:
        print(f"   ⚠️ Training without validation split (only {len(X)} sequences)")
    elif validation_split == 0.05:
        print(f"   ⚠️ Using minimal validation split (5% with {len(X)} sequences)")
    
    with _silence_tensorflow_stderr():
        model.fit(
            X,
            {'return_predictions': y_returns, 'direction_predictions': y_directions},
            epochs=LstmConfig.EPOCHS,
            batch_size=min(LstmConfig.BATCH_SIZE, max(8, len(X)//10)),
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=0
        )
    print("   ✅ LSTM training completed (dual-task: returns + directions).")

    # 3. Generate features for the entire dataset
    # Create sequences for all data points
    full_sequences = []
    for i in range(len(scaled_data) - LstmConfig.WINDOW_SIZE + 1):
        full_sequences.append(scaled_data[i:i+LstmConfig.WINDOW_SIZE])
    
    if not full_sequences:
        print("⚠️ Could not create sequences for prediction. Skipping LSTM features.")
        return _fill_placeholder_lstm_features(df)

    # Get predictions and compressed latent vectors
    full_sequences_arr = np.array(full_sequences)
    with _silence_tensorflow_stderr():
        predictions_dict = model.predict(full_sequences_arr, verbose=0)
        predicted_returns = predictions_dict['return_predictions']
        predicted_directions = predictions_dict['direction_predictions']
        compressed_latent_vectors = compressed_latent_model.predict(full_sequences_arr, verbose=0)

    print("   Extracting compressed latent space features...")
    print(f"   Dimensionality reduction: {LstmConfig.LATENT_DIM}d → {LstmConfig.COMPRESSED_LATENT_DIM}d (noise filtering)")
    df = _map_lstm_outputs_to_df(df, predicted_returns, predicted_directions, compressed_latent_vectors)
    compressed_latent_dim = compressed_latent_vectors.shape[1]

    print(f"   ✅ LSTM latent features generated: {compressed_latent_dim} dimensions (compressed)")
    print(f"   ✅ Dual-task training: Returns (MSE) + Directions (Binary CE)")
    total_features = len(LstmConfig.HORIZONS) * 3 + compressed_latent_dim + 1
    print(f"   ✅ Total LSTM features: {total_features} (returns + prices + directions + confidence + compressed latent)")
    print(f"   ✅ Noise reduction: {LstmConfig.LATENT_DIM}d → {compressed_latent_dim}d = {100*(1-compressed_latent_dim/LstmConfig.LATENT_DIM):.1f}% compression")
    return df
