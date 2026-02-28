import unittest

import numpy as np

from src.modeling.thresholds import find_optimal_buy_threshold_sharpe


class NetSharpeThresholdTests(unittest.TestCase):
    def test_quantile_search_returns_stats(self):
        proba = np.array([0.40, 0.55, 0.61, 0.65, 0.70, 0.74, 0.80, 0.85, 0.90, 0.95])
        returns = np.array([-0.01, 0.0, 0.003, 0.004, -0.002, 0.006, 0.01, 0.015, 0.008, 0.02])
        th, stats = find_optimal_buy_threshold_sharpe(
            proba_preds=proba,
            raw_returns=returns,
            min_samples=2,
            min_trades_ratio=0.2,
            search_mode="quantile",
            target_coverage_ratio=0.3,
            turnover_penalty=0.01,
        )
        self.assertIsInstance(th, float)
        self.assertIn("objective_score", stats)
        self.assertEqual(stats.get("search_mode"), "quantile")
        self.assertGreaterEqual(stats.get("trade_ratio", 0.0), 0.2)

    def test_trade_ratio_floor_is_enforced(self):
        proba = np.array([0.95, 0.94, 0.93, 0.60, 0.59, 0.58, 0.57, 0.56, 0.55, 0.54])
        returns = np.array([0.01, 0.008, 0.012, -0.002, 0.001, 0.0, 0.001, -0.001, 0.002, 0.001])
        th, stats = find_optimal_buy_threshold_sharpe(
            proba_preds=proba,
            raw_returns=returns,
            min_samples=1,
            min_trades_ratio=0.1,
            search_mode="quantile",
            target_coverage_ratio=0.4,
            trade_ratio_floor=0.3,
        )
        self.assertIsInstance(th, float)
        self.assertGreaterEqual(stats.get("trade_ratio", 0.0), 0.3)


if __name__ == "__main__":
    unittest.main()
