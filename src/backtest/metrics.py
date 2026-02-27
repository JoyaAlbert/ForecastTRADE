from typing import Dict

import numpy as np


def calculate_sharpe_ratio(returns, risk_free_rate: float = 0.02):
    returns = np.asarray(returns, dtype=float)
    if len(returns) < 2:
        return np.nan
    excess_returns = returns - (risk_free_rate / 252)
    mean_return = excess_returns.mean()
    std_dev = excess_returns.std()
    if std_dev < 1e-10:
        return np.nan
    return float((mean_return / std_dev) * np.sqrt(252))


def calculate_max_drawdown(returns):
    returns = np.asarray(returns, dtype=float)
    if len(returns) == 0:
        return np.nan
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return float(drawdown.min())


def apply_trading_costs(raw_returns, position_sizes, cost_rate: float, slippage_rate: float):
    raw_returns = np.asarray(raw_returns, dtype=float)
    position_sizes = np.asarray(position_sizes, dtype=float)
    prev = np.concatenate([[0.0], position_sizes[:-1]])
    turnover = np.abs(position_sizes - prev)
    total_cost = turnover * (cost_rate + slippage_rate)
    net_returns = raw_returns * position_sizes - total_cost
    return net_returns, turnover


def evaluate_net_metrics(raw_returns, position_sizes, cost_rate: float, slippage_rate: float) -> Dict[str, float]:
    net_returns, turnover = apply_trading_costs(raw_returns, position_sizes, cost_rate, slippage_rate)
    return {
        "net_sharpe": float(calculate_sharpe_ratio(net_returns)),
        "net_return": float(np.nansum(net_returns)),
        "turnover": float(np.nansum(turnover)),
    }
