from typing import Any, Dict

import pandas as pd

try:
    from .utils.runtime import RuntimeFlags, StageProfiler
except ImportError:
    from utils.runtime import RuntimeFlags, StageProfiler


def run_training_pipeline(
    train_fn,
    df: pd.DataFrame,
    feature_contract: Dict[str, Any],
    runtime: RuntimeFlags,
    profiler: StageProfiler,
) -> Dict[str, Any]:
    with profiler.timed("train_eval"):
        return train_fn(
            df,
            feature_contract=feature_contract,
            runtime_flags=runtime,
            stage_profiler=profiler,
        )
