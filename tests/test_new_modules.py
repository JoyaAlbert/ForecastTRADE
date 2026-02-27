import unittest

import numpy as np

from src.modeling.calibration import calibration_metrics, expected_calibration_error
from src.modeling.cv import PurgedEmbargoTimeSeriesSplit
from src.modeling.thresholds import find_optimal_buy_threshold_sharpe
from src.recommendation.engine import conservative_recommendation


class PurgedCVTests(unittest.TestCase):
    def test_purged_split_has_no_overlap(self):
        splitter = PurgedEmbargoTimeSeriesSplit(
            n_splits=3,
            test_size=20,
            purge_size=5,
            embargo_size=5,
            min_train_size=40,
        )
        splits = list(splitter.split(140))
        self.assertTrue(len(splits) > 0)
        for train_idx, test_idx in splits:
            self.assertTrue(np.max(train_idx) < np.min(test_idx))


class CalibrationTests(unittest.TestCase):
    def test_ece_and_brier_exist(self):
        y = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        p = np.array([0.1, 0.2, 0.8, 0.7, 0.9, 0.4, 0.6, 0.3])
        stats = calibration_metrics(y, p)
        self.assertIn("brier", stats)
        self.assertIn("ece", stats)
        self.assertGreaterEqual(stats["ece"], 0.0)
        self.assertAlmostEqual(stats["ece"], expected_calibration_error(y, p), places=7)


class ThresholdTests(unittest.TestCase):
    def test_sharpe_threshold_returns_stats(self):
        proba = np.array([0.8, 0.6, 0.7, 0.2, 0.9, 0.55, 0.51, 0.49, 0.76, 0.61])
        net_returns = np.array([0.01, -0.005, 0.02, -0.004, 0.015, 0.0, -0.002, 0.0, 0.018, -0.001])
        threshold, stats = find_optimal_buy_threshold_sharpe(proba, net_returns, min_samples=2, min_trades_ratio=0.2)
        self.assertIsInstance(threshold, float)
        self.assertIn("n_trades", stats)
        self.assertGreaterEqual(stats["trade_ratio"], 0.2)


class RecommendationTests(unittest.TestCase):
    def test_conservative_recommendation_states(self):
        decision = conservative_recommendation(
            probability=0.72,
            dynamic_threshold=0.68,
            tp_pct=0.08,
            sl_pct=-0.04,
            cost_pct=0.002,
            regime_allowed=True,
        )
        self.assertIn(decision.state, {"ENTER_SMALL", "ENTER_FULL", "WATCHLIST"})
        blocked = conservative_recommendation(
            probability=0.9,
            dynamic_threshold=0.68,
            tp_pct=0.08,
            sl_pct=-0.04,
            cost_pct=0.002,
            regime_allowed=False,
        )
        self.assertEqual(blocked.state, "NO_TRADE")


if __name__ == "__main__":
    unittest.main()
