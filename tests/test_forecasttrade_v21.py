import math
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

try:
    from src.main import (
        find_optimal_buy_threshold_sharpe,
        resolve_feature_contract,
        summarize_cv_results,
    )
    MAIN_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    MAIN_IMPORT_ERROR = exc

from src.run_logger import AccumulativeRunLogger


@unittest.skipIf(MAIN_IMPORT_ERROR is not None, f"src.main import failed: {MAIN_IMPORT_ERROR}")
class MainLogicTests(unittest.TestCase):
    def test_threshold_optimization_long_only_returns_single_threshold(self):
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        proba = np.array([0.80, 0.20, 0.77, 0.30, 0.73, 0.33, 0.69, 0.40])
        returns = np.array([0.02, -0.01, 0.03, -0.02, 0.01, -0.015, 0.02, -0.01])

        threshold_buy, stats = find_optimal_buy_threshold_sharpe(
            y_true,
            proba,
            returns,
            min_samples=2,
        )

        self.assertIsInstance(threshold_buy, float)
        self.assertGreaterEqual(threshold_buy, 0.45)
        self.assertLessEqual(threshold_buy, 0.75)
        self.assertIn("threshold_buy", stats)
        self.assertNotIn("threshold_sell", stats)

    def test_resolve_feature_contract_handles_drops(self):
        df = pd.DataFrame(
            {
                "lstm_latent_0": [0.1, 0.2],
                "macd": [1.0, 1.1],
                "atr": [2.0, 2.1],
                "ema_short": [3.0, 3.1],
            }
        )
        seed = ["lstm_latent_0", "lstm_latent_2", "macd", "atr", "ema_short"]

        contract = resolve_feature_contract(
            df,
            seed_features=seed,
            removed_by_corr=["lstm_latent_2"],
            removed_by_importance=[],
            min_features=3,
        )

        self.assertEqual(contract["requested"], seed)
        self.assertIn("lstm_latent_2", contract["removed_by_corr"])
        self.assertEqual(contract["missing_unexpected"], [])
        self.assertEqual(set(contract["final"]), {"lstm_latent_0", "macd", "atr", "ema_short"})

    def test_threshold_optimization_respects_min_trades_ratio(self):
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        proba = np.array([0.99, 0.10, 0.98, 0.11, 0.97, 0.12, 0.55, 0.20, 0.54, 0.30])
        returns = np.array([0.02, -0.01, 0.03, -0.02, 0.01, -0.015, 0.02, -0.01, 0.01, -0.01])

        threshold_buy, stats = find_optimal_buy_threshold_sharpe(
            y_true,
            proba,
            returns,
            min_samples=1,
            min_trades_ratio=0.3,
        )

        self.assertIsInstance(threshold_buy, float)
        self.assertGreaterEqual(stats.get("trade_ratio", 0.0), 0.3)

    def test_summarize_cv_results_counts_valid_and_skipped(self):
        fold_results = [{"fold": 1, "auc": 0.5}, {"fold": 2, "auc": 0.6}]
        skipped = [{"fold": 3, "reason": "insufficient_validation_size"}]

        summary = summarize_cv_results(fold_results, skipped)

        self.assertEqual(summary["n_folds_total"], 3)
        self.assertEqual(summary["n_folds_valid"], 2)
        self.assertEqual(summary["n_folds_skipped"], 1)
        self.assertEqual(summary["skip_reasons"], ["insufficient_validation_size"])


class RunLoggerTests(unittest.TestCase):
    def test_average_sharpe_ignores_nan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "runs_log.json")
            logger = AccumulativeRunLogger(log_file=log_file)

            logger.log_run(
                {
                    "run_number": 1,
                    "run_date": "2026-02-27",
                    "ticker": "MSFT",
                    "features_used": ["macd"],
                    "metrics": {"accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5, "auc": 0.5, "auc_pr": 0.5},
                    "sharpe_ratio": float("nan"),
                    "max_drawdown": -0.1,
                }
            )
            logger.log_run(
                {
                    "run_number": 2,
                    "run_date": "2026-02-27",
                    "ticker": "MSFT",
                    "features_used": ["macd"],
                    "metrics": {"accuracy": 0.6, "precision": 0.6, "recall": 0.6, "f1": 0.6, "auc": 0.6, "auc_pr": 0.6},
                    "sharpe_ratio": 1.2,
                    "max_drawdown": -0.05,
                }
            )

            avg_sharpe = logger.runs["metadata"]["average_sharpe"]
            self.assertTrue(math.isfinite(avg_sharpe))
            self.assertAlmostEqual(avg_sharpe, 1.2, places=6)


if __name__ == "__main__":
    unittest.main()
