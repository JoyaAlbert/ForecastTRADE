import hashlib
import json
import os
import pickle
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd


@dataclass(frozen=True)
class DataConfig:
    ticker: str
    vix_ticker: str
    start_date: str
    end_date: Optional[str]


@dataclass(frozen=True)
class FeatureConfig:
    min_features: int


@dataclass(frozen=True)
class LabelingConfig:
    tp_multiplier: float
    sl_multiplier: float
    horizon_days: int


@dataclass(frozen=True)
class TrainingConfig:
    n_splits: int
    min_validation_size: int
    use_sliding_window: bool
    sliding_window_size: int
    validation_window_size: int


@dataclass(frozen=True)
class RiskConfig:
    dynamic_risk_type: str
    dynamic_risk_k_tp: float
    dynamic_risk_k_sl: float
    dynamic_risk_vol_metric: str
    cost_bps: float = 20.0
    slippage_bps: float = 5.0


@dataclass(frozen=True)
class RuntimeFlags:
    mode: str = "full"
    plots: str = "all"
    cache: str = "on"
    profile: str = "on"
    objective: str = "sharpe_net"
    risk_profile: str = "conservative"
    cost_bps: float = 20.0
    slippage_bps: float = 5.0
    cv_scheme: str = "sliding"
    seq_encoder: str = "tcn"
    target_coverage: float = 0.16
    threshold_search: str = "quantile"
    trade_ratio_floor: float = 0.09
    conservative_th_min: float = 0.60
    conservative_th_max: float = 0.64
    xgb_stochastic_mode: str = "off"
    xgb_candidate_budget: str = "medium"
    encoder_ablation: str = "off"
    benchmark_tickers: str = "MSFT,AAPL,NVDA"
    robust_std_weight: float = 0.5
    net_return_floor_fold: float = -0.002
    config_path: Optional[str] = None

    @property
    def cache_enabled(self) -> bool:
        return self.cache == "on"

    @property
    def profile_enabled(self) -> bool:
        return self.profile == "on"

    @staticmethod
    def normalize(
        mode: str,
        plots: str,
        cache: str,
        profile: str,
        objective: str = "sharpe_net",
        risk_profile: str = "conservative",
        cost_bps: float = 20.0,
        slippage_bps: float = 5.0,
        cv_scheme: str = "sliding",
        seq_encoder: str = "tcn",
        target_coverage: float = 0.16,
        threshold_search: str = "quantile",
        trade_ratio_floor: float = 0.09,
        conservative_th_min: float = 0.60,
        conservative_th_max: float = 0.64,
        xgb_stochastic_mode: str = "off",
        xgb_candidate_budget: str = "medium",
        encoder_ablation: str = "off",
        benchmark_tickers: str = "MSFT,AAPL,NVDA",
        robust_std_weight: float = 0.5,
        net_return_floor_fold: float = -0.002,
        config_path: Optional[str] = None,
    ) -> "RuntimeFlags":
        mode = mode.lower()
        plots = plots.lower()
        cache = cache.lower()
        profile = profile.lower()
        objective = (objective or "sharpe_net").lower()
        risk_profile = (risk_profile or "conservative").lower()
        cv_scheme = (cv_scheme or "sliding").lower()
        seq_encoder = (seq_encoder or "tcn").lower()
        threshold_search = (threshold_search or "quantile").lower()
        xgb_stochastic_mode = (xgb_stochastic_mode or "off").lower()
        xgb_candidate_budget = (xgb_candidate_budget or "medium").lower()
        encoder_ablation = (encoder_ablation or "off").lower()
        benchmark_tickers = str(benchmark_tickers or "MSFT,AAPL,NVDA")

        if mode not in {"full", "fast"}:
            mode = "full"
        if plots not in {"none", "final", "all"}:
            plots = "all"
        if cache not in {"on", "off"}:
            cache = "on"
        if profile not in {"on", "off"}:
            profile = "on"
        if objective not in {"sharpe_net", "return", "max_winrate"}:
            objective = "sharpe_net"
        if risk_profile not in {"conservative", "balanced", "aggressive"}:
            risk_profile = "conservative"
        if cv_scheme not in {"sliding", "purged"}:
            cv_scheme = "sliding"
        if seq_encoder not in {"lstm", "tcn", "off"}:
            seq_encoder = "tcn"
        if threshold_search not in {"grid", "quantile"}:
            threshold_search = "quantile"
        if xgb_stochastic_mode not in {"on", "off"}:
            xgb_stochastic_mode = "off"
        if xgb_candidate_budget not in {"low", "medium", "high"}:
            xgb_candidate_budget = "medium"
        if encoder_ablation not in {"on", "off"}:
            encoder_ablation = "off"

        # Fast defaults: no fold-by-fold plotting unless explicitly requested.
        if mode == "fast" and plots == "all":
            plots = "none"

        return RuntimeFlags(
            mode=mode,
            plots=plots,
            cache=cache,
            profile=profile,
            objective=objective,
            risk_profile=risk_profile,
            cost_bps=float(cost_bps),
            slippage_bps=float(slippage_bps),
            cv_scheme=cv_scheme,
            seq_encoder=seq_encoder,
            target_coverage=float(target_coverage),
            threshold_search=threshold_search,
            trade_ratio_floor=float(trade_ratio_floor),
            conservative_th_min=float(conservative_th_min),
            conservative_th_max=float(conservative_th_max),
            xgb_stochastic_mode=xgb_stochastic_mode,
            xgb_candidate_budget=xgb_candidate_budget,
            encoder_ablation=encoder_ablation,
            benchmark_tickers=benchmark_tickers,
            robust_std_weight=float(robust_std_weight),
            net_return_floor_fold=float(net_return_floor_fold),
            config_path=config_path,
        )


@dataclass(frozen=True)
class RunConfig:
    data: DataConfig
    features: FeatureConfig
    labeling: LabelingConfig
    training: TrainingConfig
    risk: RiskConfig
    runtime: RuntimeFlags


@dataclass
class StageProfiler:
    enabled: bool = True
    stages: Dict[str, float] = field(default_factory=dict)

    @contextmanager
    def timed(self, stage: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            if self.enabled:
                self.stages[stage] = self.stages.get(stage, 0.0) + (time.perf_counter() - start)

    def summary(self) -> Dict[str, float]:
        return {k: round(v, 6) for k, v in self.stages.items()}


class CacheManager:
    def __init__(self, root_dir: str = "out/cache", enabled: bool = True):
        self.root_dir = root_dir
        self.enabled = enabled
        if self.enabled:
            os.makedirs(self.root_dir, exist_ok=True)

    @staticmethod
    def hash_payload(payload: Dict[str, Any]) -> str:
        raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha1(raw).hexdigest()

    @staticmethod
    def fingerprint_df(df: pd.DataFrame, columns: Optional[list] = None) -> str:
        cols = columns if columns else list(df.columns)
        safe_cols = [c for c in cols if c in df.columns]
        if not safe_cols:
            safe_cols = list(df.columns[: min(3, len(df.columns))])
        sample = df[safe_cols].copy()
        sample["__index__"] = sample.index.astype(str)
        hashed = pd.util.hash_pandas_object(sample, index=False).values
        return hashlib.sha1(hashed.tobytes()).hexdigest()

    def _path(self, key: str, ext: str) -> str:
        return os.path.join(self.root_dir, f"{key}.{ext}")

    def load_pickle(self, key: str) -> Optional[Any]:
        if not self.enabled:
            return None
        path = self._path(key, "pkl")
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    def save_pickle(self, key: str, obj: Any) -> None:
        if not self.enabled:
            return
        path = self._path(key, "pkl")
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
