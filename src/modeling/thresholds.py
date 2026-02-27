from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import precision_recall_curve


def find_optimal_threshold(
    y_true: np.ndarray,
    proba_preds: np.ndarray,
    metric: str = "f1",
    min_recall: float = 0.25,
) -> Tuple[float, Dict[str, float]]:
    precision, recall, thresholds = precision_recall_curve(y_true, proba_preds)
    if len(thresholds) == 0:
        return 0.5, {"precision": float(precision[-1]), "recall": float(recall[-1]), "f1": 0.0}

    precision = precision[1:]
    recall = recall[1:]
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    if metric == "precision_at_recall":
        mask = recall >= min_recall
        idx = int(np.where(mask)[0][np.argmax(precision[mask])]) if np.any(mask) else int(np.nanargmax(f1_scores))
    else:
        idx = int(np.nanargmax(f1_scores))
    return float(thresholds[idx]), {
        "precision": float(precision[idx]),
        "recall": float(recall[idx]),
        "f1": float(f1_scores[idx]),
    }


def find_optimal_buy_threshold_sharpe(
    proba_preds: np.ndarray,
    raw_returns: np.ndarray,
    min_samples: int = 10,
    min_trades_ratio: float = 0.08,
    cost_rate: float = 0.0,
    slippage_rate: float = 0.0,
    max_trades_ratio: float | None = None,
) -> Tuple[float, Dict[str, float]]:
    threshold_buy_range = np.arange(0.45, 0.80, 0.05)
    best_sharpe = -np.inf
    best_threshold_buy = 0.65
    best_stats: Dict[str, float] = {}
    min_required = max(int(len(proba_preds) * float(min_trades_ratio)), int(min_samples))
    raw_returns = np.asarray(raw_returns, dtype=float)
    trade_cost = float(cost_rate + slippage_rate)

    for th_buy in threshold_buy_range:
        trade_mask = proba_preds >= th_buy
        n_trades = int(trade_mask.sum())
        if n_trades < min_required:
            continue
        trade_ratio = float(n_trades / max(1, len(proba_preds)))
        if max_trades_ratio is not None and trade_ratio > float(max_trades_ratio):
            continue

        positions = trade_mask.astype(float)
        prev = np.concatenate([[0.0], positions[:-1]])
        turnover_series = np.abs(positions - prev)
        strategy_returns = (raw_returns * positions) - (turnover_series * trade_cost)
        active_returns = strategy_returns[positions > 0]

        if len(active_returns) > 0 and strategy_returns.std() > 0:
            sharpe = float(strategy_returns.mean() / strategy_returns.std() * np.sqrt(252))
        else:
            sharpe = 0.0
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_threshold_buy = float(th_buy)
            win_rate = float((active_returns > 0).sum() / len(active_returns)) if len(active_returns) > 0 else 0.0
            avg_return = float(active_returns.mean()) if len(active_returns) > 0 else 0.0
            best_stats = {
                "sharpe": sharpe,
                "n_trades": n_trades,
                "trade_ratio": trade_ratio,
                "turnover": float(np.sum(turnover_series)),
                "win_rate": win_rate,
                "avg_return": avg_return,
                "threshold_buy": float(th_buy),
            }
    return float(best_threshold_buy), best_stats
