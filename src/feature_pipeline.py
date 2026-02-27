from typing import Callable, Dict, Tuple

import pandas as pd

try:
    from .utils.runtime import CacheManager, RuntimeFlags, StageProfiler
except ImportError:
    from utils.runtime import CacheManager, RuntimeFlags, StageProfiler


def run_feature_pipeline(
    df: pd.DataFrame,
    ticker: str,
    runtime: RuntimeFlags,
    cache: CacheManager,
    profiler: StageProfiler,
    generate_lstm_fn: Callable[[pd.DataFrame, str], pd.DataFrame],
    engineer_features_fn: Callable[[pd.DataFrame], pd.DataFrame],
    create_advanced_features_fn: Callable[[pd.DataFrame], pd.DataFrame],
    remove_correlated_features_fn: Callable[[pd.DataFrame, float], Tuple[pd.DataFrame, list]],
    filter_low_importance_features_fn: Callable[[pd.DataFrame], Tuple[pd.DataFrame, list]],
    resolve_feature_contract_fn: Callable[..., Dict],
    seed_features: list,
    min_features: int,
    leakage_safe_lstm: bool = False,
    placeholder_lstm_fn: Callable[[pd.DataFrame], pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict]:
    fingerprint = CacheManager.fingerprint_df(df, columns=["open", "high", "low", "close", "volume", "vix_close"])
    cache_key = CacheManager.hash_payload(
        {
            "schema": "feature_pipeline_v1",
            "ticker": ticker,
            "df": fingerprint,
            "seed_features": seed_features,
            "min_features": min_features,
            "leakage_safe_lstm": leakage_safe_lstm,
        }
    )

    if runtime.cache_enabled:
        cached = cache.load_pickle(cache_key)
        if cached is not None:
            return cached["df"], cached["feature_contract"]

    with profiler.timed("lstm"):
        if leakage_safe_lstm and placeholder_lstm_fn is not None:
            out_df = placeholder_lstm_fn(df.copy())
        else:
            out_df = generate_lstm_fn(df.copy(), ticker)

    with profiler.timed("features"):
        out_df = engineer_features_fn(out_df)
        out_df = create_advanced_features_fn(out_df)
        out_df, dropped_corr = remove_correlated_features_fn(out_df, threshold=0.98)
        out_df, dropped_low_imp = filter_low_importance_features_fn(out_df)

    feature_contract = resolve_feature_contract_fn(
        out_df,
        seed_features=seed_features,
        removed_by_corr=dropped_corr,
        removed_by_importance=dropped_low_imp,
        min_features=min_features,
    )

    if runtime.cache_enabled:
        cache.save_pickle(cache_key, {"df": out_df, "feature_contract": feature_contract})

    return out_df, feature_contract
