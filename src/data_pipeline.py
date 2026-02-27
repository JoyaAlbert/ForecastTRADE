from typing import Callable, Optional

import pandas as pd

try:
    from .utils.runtime import CacheManager, RuntimeFlags, StageProfiler
except ImportError:
    from utils.runtime import CacheManager, RuntimeFlags, StageProfiler


def load_market_data(
    fetch_fn: Callable[[str, str, str, Optional[str]], pd.DataFrame],
    ticker: str,
    vix_ticker: str,
    start_date: str,
    end_date: Optional[str],
    runtime: RuntimeFlags,
    cache: CacheManager,
    profiler: StageProfiler,
) -> pd.DataFrame:
    cache_key = CacheManager.hash_payload(
        {
            "schema": "market_data_v1",
            "ticker": ticker,
            "vix_ticker": vix_ticker,
            "start_date": start_date,
            "end_date": end_date,
        }
    )

    if runtime.cache_enabled:
        cached = cache.load_pickle(cache_key)
        if cached is not None:
            return cached

    with profiler.timed("fetch"):
        df = fetch_fn(ticker, vix_ticker, start_date, end_date)

    if runtime.cache_enabled and not df.empty:
        cache.save_pickle(cache_key, df)
    return df
