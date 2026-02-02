# -*- coding: utf-8 -*-
"""
LSTM Predictor for Hybrid Financial Model
===========================================

This script defines the LSTM component of the hybrid forecasting model.
Its role is to act as a "pattern analyst" by learning from sequences of
price data (OHLCV).

The main function, `generate_lstm_trend_feature`, orchestrates the
entire process:
1.  Scales the relevant features (OHLCV).
2.  Creates time-series sequences (sliding windows).
3.  Builds and trains the Stacked LSTM model.
4.  Uses the trained model to predict the next day's price.
5.  Returns these predictions as a new feature for the main model.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION FOR LSTM ---
class LstmConfig:
    """Configuration class for the LSTM model."""
    # Data Preparation
    WINDOW_SIZE = 60  # 60 days of historical data to predict the next day
    FEATURES = ['open', 'high', 'low', 'close', 'volume']
    TRAIN_SPLIT_RATIO = 0.8

    # Model Architecture
    LSTM_UNITS_1 = 50
    LSTM_UNITS_2 = 50
    DROPOUT_RATE = 0.2
    
    # Training
    EPOCHS = 50
    BATCH_SIZE = 32
    LOSS_FUNCTION = 'mean_squared_error'

# --- 2. DATA PREPARATION ---
def create_sequences(data, window_size):
    """
    Transforms time-series data into sequences for LSTM.

    Args:
        data (np.array): The scaled time-series data.
        window_size (int): The number of time steps in each sequence.

    Returns:
        tuple: A tuple containing:
            - np.array: The input sequences (X).
            - np.array: The target values (y).
    """
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i, 3])  # Target is the 'close' price (index 3)
    return np.array(X), np.array(y)

# --- 3. MODEL ARCHITECTURE ---
def build_lstm_model(input_shape):
    """
    Builds the Stacked LSTM model as per the requirements.

    Args:
        input_shape (tuple): The shape of the input data (time_steps, features).

    Returns:
        tensorflow.keras.Model: The compiled LSTM model.
    """
    model = Sequential([
        LSTM(
            LstmConfig.LSTM_UNITS_1,
            return_sequences=True,
            input_shape=input_shape
        ),
        Dropout(LstmConfig.DROPOUT_RATE),
        LSTM(LstmConfig.LSTM_UNITS_2, return_sequences=False),
        Dropout(LstmConfig.DROPOUT_RATE),
        Dense(25), # Intermediate layer
        Dense(1)   # Final prediction layer
    ])

    optimizer = Adam()
    model.compile(optimizer=optimizer, loss=LstmConfig.LOSS_FUNCTION)
    model.summary()
    return model

# --- 4. VISUALIZATION ---
def plot_predictions(dates, real, predicted, ticker):
    """
    Generates a plot comparing real vs. predicted prices.

    Args:
        dates (pd.Index): The dates for the x-axis.
        real (np.array): The actual closing prices.
        predicted (np.array): The LSTM's predicted prices.
        ticker (str): The stock ticker for the plot title.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(dates, real, color='blue', label='Precio Real')
    plt.plot(dates, predicted, color='red', label='Predicción LSTM', alpha=0.7)
    plt.title(f'Predicción LSTM vs. Precio Real para {ticker}')
    plt.xlabel('Fecha')
    plt.ylabel('Precio de Cierre (USD)')
    plt.legend()
    plt.grid(True)
    # Save the plot instead of showing it to avoid blocking execution
    plt.savefig(f'lstm_prediction_vs_real_{ticker}.png')
    print(f"\n📈 Gráfico de predicción guardado como 'lstm_prediction_vs_real_{ticker}.png'")
    plt.close()

# --- 5. MAIN ORCHESTRATOR ---
def generate_lstm_trend_feature(df, ticker_name):
    """
    The main function to generate the LSTM trend prediction feature.

    Args:
        df (pd.DataFrame): The input dataframe with OHLCV data.
        ticker_name (str): The name of the stock ticker.

    Returns:
        pd.DataFrame: The original DataFrame augmented with the
                      'lstm_trend_prediction' column.
    """
    print("\n🧠 --- Iniciando Módulo LSTM --- 🧠")
    print("Generando la característica de tendencia con LSTM...")

    # 1. Prepare data for LSTM
    data_for_lstm = df[LstmConfig.FEATURES].values
    
    # 2. Scale features
    scaler = MinMaxScaler(feature_range=(0, 1))
    # We fit the scaler on the entire dataset to ensure consistency,
    # but a stricter approach would be to fit only on the training set.
    # For this hybrid model's purpose, this is a reasonable simplification.
    scaled_data = scaler.fit_transform(data_for_lstm)
    
    # Create a separate scaler for the target variable ('close') to inverse transform later
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler.fit(df[['close']])

    # 3. Create training data for the LSTM
    train_size = int(len(scaled_data) * LstmConfig.TRAIN_SPLIT_RATIO)
    train_data = scaled_data[:train_size]
    
    X_train, y_train = create_sequences(train_data, LstmConfig.WINDOW_SIZE)
    if X_train.shape[0] == 0:
        print("❌ Error en LSTM: No se pudieron crear secuencias de entrenamiento. El dataset es muy pequeño.")
        return df.assign(lstm_trend_prediction=df['close']) # Return unmodified

    # 4. Build and train the LSTM model
    print(f"Entrenando LSTM con {X_train.shape[0]} secuencias de entrenamiento...")
    lstm_model = build_lstm_model(
        (X_train.shape[1], X_train.shape[2])
    )
    history = lstm_model.fit(
        X_train, y_train,
        epochs=LstmConfig.EPOCHS,
        batch_size=LstmConfig.BATCH_SIZE,
        verbose=0  # Set to 1 to see training progress
    )
    print("LSTM entrenado con éxito.")
    
    # 5. Generate predictions for the entire dataset
    all_sequences, _ = create_sequences(scaled_data, LstmConfig.WINDOW_SIZE)
    predicted_prices_scaled = lstm_model.predict(all_sequences)
    
    # Inverse transform predictions to their original price scale
    predicted_prices = target_scaler.inverse_transform(predicted_prices_scaled)
    
    # 6. Align predictions with the main DataFrame
    # The predictions start from the end of the first window
    prediction_start_index = LstmConfig.WINDOW_SIZE
    
    # Create the new column, filling initial NaNs
    df['lstm_trend_prediction'] = np.nan
    df.iloc[prediction_start_index:prediction_start_index + len(predicted_prices), df.columns.get_loc('lstm_trend_prediction')] = predicted_prices.flatten()

    # To simplify, we can back-fill the first few values where there's no prediction
    df['lstm_trend_prediction'] = df['lstm_trend_prediction'].bfill()
    
    # 7. Evaluate and visualize on the test set part
    test_set_start_index = train_size
    
    # The first prediction corresponds to index WINDOW_SIZE in the original df.
    # The first test prediction corresponds to index train_size.
    num_train_predictions = X_train.shape[0]
    
    # Align real and predicted values for the plot
    # The real prices for the test set start after the training data
    real_prices_for_plot = df['close'].iloc[test_set_start_index:].values
    
    # The corresponding predictions are the ones after the training predictions
    predicted_prices_for_plot = predicted_prices[num_train_predictions:]
    
    # The dates for the test set plot
    test_dates = df.index[test_set_start_index:]

    # Due to windowing, the number of predictions is smaller than the test set size.
    # We need to align them by plotting only the dates for which we have predictions.
    if len(predicted_prices_for_plot) > 0:
        plot_dates = test_dates[-len(predicted_prices_for_plot):]
        plot_real = real_prices_for_plot[-len(predicted_prices_for_plot):]
        
        plot_predictions(
            plot_dates,
            plot_real,
            predicted_prices_for_plot,
            ticker_name
        )
    else:
        print("⚠️ Advertencia: No hay suficientes predicciones en el conjunto de prueba para generar un gráfico.")

    print("✅ Módulo LSTM completado.")
    return df
