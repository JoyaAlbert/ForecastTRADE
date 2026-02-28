import argparse
import unittest

import pandas as pd

from scripts.benchmark_defaults import choose_best


class RobustScoreSelectionTests(unittest.TestCase):
    def test_choose_best_respects_constraints(self):
        df = pd.DataFrame(
            [
                {
                    "candidate_id": "c1",
                    "robust_score": 0.30,
                    "net_sharpe_mean": 0.25,
                    "coverage_mean": 0.85,
                    "valid_folds_ratio_mean": 0.84,
                },
                {
                    "candidate_id": "c2",
                    "robust_score": 0.40,
                    "net_sharpe_mean": 0.10,
                    "coverage_mean": 0.90,
                    "valid_folds_ratio_mean": 0.90,
                },
            ]
        )
        args = argparse.Namespace(min_net_sharpe=0.20, min_coverage=0.83, min_valid_folds_ratio=0.83)
        best, relaxed = choose_best(df, args)
        self.assertFalse(relaxed)
        self.assertEqual(best["candidate_id"], "c1")

    def test_choose_best_relaxes_when_no_candidate_meets_constraints(self):
        df = pd.DataFrame(
            [
                {
                    "candidate_id": "c1",
                    "robust_score": 0.15,
                    "net_sharpe_mean": 0.05,
                    "coverage_mean": 0.80,
                    "valid_folds_ratio_mean": 0.82,
                }
            ]
        )
        args = argparse.Namespace(min_net_sharpe=0.20, min_coverage=0.83, min_valid_folds_ratio=0.83)
        best, relaxed = choose_best(df, args)
        self.assertTrue(relaxed)
        self.assertEqual(best["candidate_id"], "c1")


if __name__ == "__main__":
    unittest.main()
