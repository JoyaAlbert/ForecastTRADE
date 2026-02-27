from typing import Dict, List

import numpy as np
import pandas as pd


def build_dynamic_risk_config(dynamic_type: str, k_tp: float, k_sl: float, vol_metric: str) -> Dict:
    return {
        "dynamic_risk": {
            "type": dynamic_type,
            "params": {
                "k_tp": k_tp,
                "k_sl": k_sl,
                "vol_metric": vol_metric,
            },
        }
    }


def summarize_future_predictions(predictions: List[Dict]) -> Dict[str, float]:
    if not predictions:
        return {"avg_prob": 0.5, "max_prob": 0.5, "min_prob": 0.5}
    probs = np.array([p["probability"] for p in predictions], dtype=float)
    return {
        "avg_prob": float(np.mean(probs)),
        "max_prob": float(np.max(probs)),
        "min_prob": float(np.min(probs)),
    }


def latest_volatility_for_risk(df: pd.DataFrame, vol_metric_pref: str) -> Dict[str, float]:
    if vol_metric_pref == "atr_14d" and "atr" in df.columns:
        vol = float(pd.to_numeric(df["atr"], errors="coerce").ffill().iloc[-1])
        metric = "atr_14d"
    elif "volatility_20" in df.columns:
        vol = float(pd.to_numeric(df["volatility_20"], errors="coerce").ffill().iloc[-1])
        metric = "rolling_std_20d"
    else:
        vol = float(pd.to_numeric(df["close"], errors="coerce").rolling(window=20).std().ffill().iloc[-1])
        metric = "rolling_std_20d"
    return {"volatility": vol, "metric": metric}
