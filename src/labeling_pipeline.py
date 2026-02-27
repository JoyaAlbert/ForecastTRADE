from typing import Callable, Dict, Tuple

import pandas as pd

try:
    from .utils.runtime import CacheManager, RuntimeFlags, StageProfiler
except ImportError:
    from utils.runtime import CacheManager, RuntimeFlags, StageProfiler


def run_labeling_pipeline(
    df: pd.DataFrame,
    feature_contract: Dict,
    runtime: RuntimeFlags,
    cache: CacheManager,
    profiler: StageProfiler,
    define_target_fn: Callable[[pd.DataFrame], pd.DataFrame],
    preprocess_data_raw_fn: Callable[[pd.DataFrame, Dict], pd.DataFrame],
    label_signature: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fingerprint = CacheManager.fingerprint_df(df, columns=["open", "high", "low", "close", "volume", "vix_close"])
    cache_key = CacheManager.hash_payload(
        {
            "schema": "labeling_pipeline_v1",
            "df": fingerprint,
            "label_signature": label_signature,
            "feature_contract": feature_contract.get("final", []),
        }
    )

    if runtime.cache_enabled:
        cached = cache.load_pickle(cache_key)
        if cached is not None:
            return cached["labeled_df"], cached["processed_df"]

    with profiler.timed("target"):
        labeled_df = define_target_fn(df.copy())
        processed_df = preprocess_data_raw_fn(labeled_df.copy(), feature_contract=feature_contract)

    if runtime.cache_enabled and not processed_df.empty:
        cache.save_pickle(cache_key, {"labeled_df": labeled_df, "processed_df": processed_df})
    return labeled_df, processed_df
