# -*- coding: utf-8 -*-
"""TCN-based temporal feature generator with LSTM-compatible output schema."""

import contextlib
import hashlib
import json
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_LOG_SEVERITY_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Add, Conv1D, Dense, Dropout, Input, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

try:
    from .lstm_predictor import (
        LstmConfig,
        create_return_targets,
        create_sequences,
        engineer_lstm_features,
        generate_placeholder_lstm_features,
    )
except ImportError:
    from lstm_predictor import (
        LstmConfig,
        create_return_targets,
        create_sequences,
        engineer_lstm_features,
        generate_placeholder_lstm_features,
    )


@contextlib.contextmanager
def _silence_tensorflow_stderr():
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stderr(devnull):
            yield


def _tcn_block(x, channels: int, kernel_size: int, dilation_rate: int, dropout: float, name: str):
    residual = x
    y = Conv1D(
        filters=channels,
        kernel_size=kernel_size,
        padding="causal",
        dilation_rate=dilation_rate,
        activation="relu",
        name=f"{name}_conv1",
    )(x)
    y = LayerNormalization(name=f"{name}_ln1")(y)
    y = Dropout(dropout, name=f"{name}_drop1")(y)
    y = Conv1D(
        filters=channels,
        kernel_size=kernel_size,
        padding="causal",
        dilation_rate=dilation_rate,
        activation="relu",
        name=f"{name}_conv2",
    )(y)
    y = LayerNormalization(name=f"{name}_ln2")(y)
    y = Dropout(dropout, name=f"{name}_drop2")(y)

    if int(residual.shape[-1]) != channels:
        residual = Conv1D(channels, kernel_size=1, padding="same", name=f"{name}_res")(residual)
    return Add(name=f"{name}_add")([residual, y])


def _build_tcn_feature_model(input_shape, n_outputs: int, params: dict):
    channels = int(params.get("channels", 64))
    kernel_size = int(params.get("kernel_size", 3))
    dilation_levels = params.get("dilation_levels", [1, 2, 4, 8])
    dropout = float(params.get("dropout", 0.15))
    latent_dim = int(params.get("latent_dim", 10))

    inputs = Input(shape=input_shape, name="tcn_input")
    x = inputs
    for i, d in enumerate(dilation_levels):
        x = _tcn_block(
            x,
            channels=channels,
            kernel_size=kernel_size,
            dilation_rate=int(d),
            dropout=dropout,
            name=f"tcn_block_{i}",
        )

    last_step = x[:, -1, :]
    latent = Dense(latent_dim, activation="linear", name="tcn_latent")(last_step)

    returns_h = Dense(32, activation="relu", name="returns_dense")(latent)
    returns_out = Dense(n_outputs, activation="linear", name="return_predictions")(returns_h)

    dirs_h = Dense(32, activation="relu", name="direction_dense")(latent)
    dirs_out = Dense(n_outputs, activation="sigmoid", name="direction_predictions")(dirs_h)

    model = Model(
        inputs=inputs,
        outputs={"return_predictions": returns_out, "direction_predictions": dirs_out},
        name="tcn_dual_task",
    )
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={"return_predictions": "mse", "direction_predictions": "binary_crossentropy"},
        loss_weights={"return_predictions": 0.5, "direction_predictions": 0.5},
        metrics={"return_predictions": ["mae"], "direction_predictions": ["accuracy"]},
    )
    latent_model = Model(inputs=inputs, outputs=latent, name="tcn_latent_extractor")
    return model, latent_model


def _to_schema_columns(base_df, predicted_returns, predicted_directions, latent_vectors):
    df = base_df.copy()
    pred_start_idx = LstmConfig.WINDOW_SIZE - 1

    for i, h in enumerate(LstmConfig.HORIZONS):
        ret_col = f"tcn_return_{h}d"
        price_col = f"tcn_price_{h}d"
        dir_col = f"tcn_dir_{h}d"
        df[ret_col] = np.nan
        df.iloc[pred_start_idx:pred_start_idx + len(predicted_returns), df.columns.get_loc(ret_col)] = predicted_returns[:, i]
        df[price_col] = df["close"] * (1 + df[ret_col])
        df[dir_col] = np.nan
        df.iloc[pred_start_idx:pred_start_idx + len(predicted_directions), df.columns.get_loc(dir_col)] = predicted_directions[:, i]

    df["tcn_return_confidence"] = np.nan
    conf = np.mean(np.abs(predicted_returns), axis=1)
    df.iloc[pred_start_idx:pred_start_idx + len(conf), df.columns.get_loc("tcn_return_confidence")] = conf

    for i in range(latent_vectors.shape[1]):
        col = f"tcn_latent_{i}"
        df[col] = np.nan
        df.iloc[pred_start_idx:pred_start_idx + len(latent_vectors), df.columns.get_loc(col)] = latent_vectors[:, i]

    return df.bfill().ffill()


def _adapt_tcn_to_lstm_schema(df: pd.DataFrame, latent_dim: int = 10) -> pd.DataFrame:
    out = df.copy()
    for h in LstmConfig.HORIZONS:
        out[f"lstm_return_{h}d"] = out.get(f"tcn_return_{h}d", 0.0)
        out[f"lstm_price_{h}d"] = out.get(f"tcn_price_{h}d", out["close"])
        out[f"lstm_dir_{h}d"] = out.get(f"tcn_dir_{h}d", 0.5)
    out["lstm_return_confidence"] = out.get("tcn_return_confidence", 0.0)
    for i in range(latent_dim):
        out[f"lstm_latent_{i}"] = out.get(f"tcn_latent_{i}", 0.0)
    return out


def _placeholder_with_tcn_columns(df: pd.DataFrame, latent_dim: int) -> pd.DataFrame:
    base = generate_placeholder_lstm_features(df.copy())
    for h in LstmConfig.HORIZONS:
        base[f"tcn_return_{h}d"] = 0.0
        base[f"tcn_price_{h}d"] = base["close"]
        base[f"tcn_dir_{h}d"] = 0.5
    base["tcn_return_confidence"] = 0.0
    for i in range(latent_dim):
        base[f"tcn_latent_{i}"] = 0.0
    return base


def _train_and_project(train_df: pd.DataFrame, full_df: pd.DataFrame, params: dict):
    train_lstm_df, feature_names = engineer_lstm_features(train_df.copy())
    full_lstm_df, _ = engineer_lstm_features(full_df.copy())

    if len(train_lstm_df) < LstmConfig.WINDOW_SIZE + max(LstmConfig.HORIZONS):
        return _placeholder_with_tcn_columns(full_df.copy(), int(params.get("latent_dim", 10)))

    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_lstm_df)
    scaled_full = scaler.transform(full_lstm_df)

    targets_df = create_return_targets(train_lstm_df, LstmConfig.HORIZONS)
    X_train, y_all = create_sequences(scaled_train, targets_df.values, LstmConfig.WINDOW_SIZE)
    if len(X_train) < 120:
        return _placeholder_with_tcn_columns(full_df.copy(), int(params.get("latent_dim", 10)))

    n_horizons = len(LstmConfig.HORIZONS)
    y_returns = y_all[:, :n_horizons]
    y_dirs = y_all[:, n_horizons:]
    model, latent_model = _build_tcn_feature_model(
        input_shape=(LstmConfig.WINDOW_SIZE, len(feature_names)),
        n_outputs=n_horizons,
        params=params,
    )
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]
    validation_split = 0.1 if len(X_train) > 100 else 0.0

    with _silence_tensorflow_stderr():
        model.fit(
            X_train,
            {"return_predictions": y_returns, "direction_predictions": y_dirs},
            epochs=max(20, min(int(params.get("epochs", 60)), 60)),
            batch_size=min(int(params.get("batch_size", 64)), max(8, len(X_train) // 10)),
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=0,
        )
        full_sequences = []
        for i in range(len(scaled_full) - LstmConfig.WINDOW_SIZE + 1):
            full_sequences.append(scaled_full[i:i + LstmConfig.WINDOW_SIZE])
        if not full_sequences:
            return _placeholder_with_tcn_columns(full_df.copy(), int(params.get("latent_dim", 10)))
        full_sequences = np.asarray(full_sequences)
        pred_dict = model.predict(full_sequences, verbose=0)
        pred_returns = pred_dict["return_predictions"]
        pred_dirs = pred_dict["direction_predictions"]
        latent = latent_model.predict(full_sequences, verbose=0)

    with_tcn = _to_schema_columns(full_df.copy(), pred_returns, pred_dirs, latent)
    with_adapter = _adapt_tcn_to_lstm_schema(with_tcn, latent_dim=int(params.get("latent_dim", 10)))
    return with_adapter


def generate_tcn_regime_features(df, ticker_name: str = "AAPL", tcn_params: dict | None = None):
    _ = ticker_name
    params = tcn_params or {}
    return _train_and_project(df.copy(), df.copy(), params=params)


def generate_tcn_regime_features_train_only(train_df, full_df, ticker_name: str = "AAPL", tcn_params: dict | None = None):
    _ = ticker_name
    params = tcn_params or {}
    return _train_and_project(train_df.copy(), full_df.copy(), params=params)


def generate_placeholder_tcn_features(df, tcn_params: dict | None = None):
    params = tcn_params or {}
    return _placeholder_with_tcn_columns(df.copy(), int(params.get("latent_dim", 10)))


def fold_cache_key(
    ticker: str,
    seq_encoder: str,
    train_index,
    val_index,
    params: dict,
) -> str:
    payload = {
        "ticker": ticker,
        "encoder": seq_encoder,
        "train_start": str(train_index.min()) if len(train_index) else "",
        "train_end": str(train_index.max()) if len(train_index) else "",
        "val_start": str(val_index.min()) if len(val_index) else "",
        "val_end": str(val_index.max()) if len(val_index) else "",
        "n_train": int(len(train_index)),
        "n_val": int(len(val_index)),
        "params": params,
    }
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()
