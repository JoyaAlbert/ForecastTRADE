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
    search_mode: str = "grid",
    target_coverage_ratio: float = 0.12,
    turnover_penalty: float = 0.0,
    trade_ratio_floor: float = 0.08,
) -> Tuple[float, Dict[str, float]]:
    search_mode = (search_mode or "grid").lower()
    if search_mode == "quantile":
        # Higher quantiles => stricter thresholds.
        quantiles = np.linspace(0.50, 0.95, 10)
        threshold_buy_range = np.unique(np.quantile(proba_preds, quantiles))
    else:
        threshold_buy_range = np.arange(0.45, 0.80, 0.05)

    best_sharpe = -np.inf
    best_threshold_buy = 0.65
    best_stats: Dict[str, float] = {}
    min_required = max(int(len(proba_preds) * float(min_trades_ratio)), int(min_samples))
    raw_returns = np.asarray(raw_returns, dtype=float)
    trade_cost = float(cost_rate + slippage_rate)
    target_coverage_ratio = float(np.clip(target_coverage_ratio, 0.01, 1.0))
    turnover_penalty = float(max(0.0, turnover_penalty))
    trade_ratio_floor = float(np.clip(trade_ratio_floor, 0.01, 1.0))

    for th_buy in threshold_buy_range:
        trade_mask = proba_preds >= th_buy
        n_trades = int(trade_mask.sum())
        if n_trades < min_required:
            continue
        trade_ratio = float(n_trades / max(1, len(proba_preds)))
        if trade_ratio < trade_ratio_floor:
            continue
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
        coverage_penalty = abs(trade_ratio - target_coverage_ratio)
        turnover = float(np.sum(turnover_series))
        objective_score = sharpe - turnover_penalty * turnover - (coverage_penalty * 1.20)

        if objective_score > best_sharpe:
            best_sharpe = objective_score
            best_threshold_buy = float(th_buy)
            win_rate = float((active_returns > 0).sum() / len(active_returns)) if len(active_returns) > 0 else 0.0
            avg_return = float(active_returns.mean()) if len(active_returns) > 0 else 0.0
            best_stats = {
                "sharpe": sharpe,
                "objective_score": float(objective_score),
                "n_trades": n_trades,
                "trade_ratio": trade_ratio,
                "turnover": turnover,
                "win_rate": win_rate,
                "avg_return": avg_return,
                "coverage_penalty": float(coverage_penalty),
                "trade_ratio_floor": float(trade_ratio_floor),
                "threshold_buy": float(th_buy),
                "search_mode": search_mode,
            }
    return float(best_threshold_buy), best_stats
