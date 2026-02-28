import unittest

import numpy as np
import pandas as pd

from src.main import tune_xgb_params_for_stability


class XgbReproducibilityTests(unittest.TestCase):
    def _make_data(self):
        rng = np.random.default_rng(321)
        x_train = pd.DataFrame(
            {
                "f1": rng.normal(0, 1, 120),
                "f2": rng.normal(0, 1, 120),
                "f3": rng.normal(0, 1, 120),
            }
        )
        y_train = pd.Series((x_train["f1"] - 0.1 * x_train["f2"] > 0).astype(int).values)
        x_val = pd.DataFrame(
            {
                "f1": rng.normal(0, 1, 60),
                "f2": rng.normal(0, 1, 60),
                "f3": rng.normal(0, 1, 60),
            }
        )
        y_val = pd.Series((x_val["f1"] - 0.1 * x_val["f2"] > 0).astype(int).values)
        val_returns = np.where(y_val.values == 1, 0.01, -0.008)
        return x_train, y_train, x_val, y_val, val_returns

    def test_same_seed_same_result(self):
        x_train, y_train, x_val, y_val, val_returns = self._make_data()
        base_params = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "max_depth": 4,
            "learning_rate": 0.04,
            "n_estimators": 100,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 1,
            "gamma": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "use_label_encoder": False,
            "early_stopping_rounds": 10,
        }
        params_a, meta_a = tune_xgb_params_for_stability(
            x_train,
            y_train,
            x_val,
            y_val,
            val_returns,
            base_params,
            scale_pos_weight=1.0,
            max_candidates=5,
            target_coverage_ratio=0.15,
            trade_ratio_floor=0.05,
            fold_idx=1,
            base_seed=77,
            stochastic_mode=True,
            candidate_budget="low",
        )
        params_b, meta_b = tune_xgb_params_for_stability(
            x_train,
            y_train,
            x_val,
            y_val,
            val_returns,
            base_params,
            scale_pos_weight=1.0,
            max_candidates=5,
            target_coverage_ratio=0.15,
            trade_ratio_floor=0.05,
            fold_idx=1,
            base_seed=77,
            stochastic_mode=True,
            candidate_budget="low",
        )
        self.assertEqual(params_a.get("random_state"), params_b.get("random_state"))
        self.assertEqual(meta_a.get("seed_eff"), meta_b.get("seed_eff"))
        self.assertAlmostEqual(float(meta_a.get("score", 0.0)), float(meta_b.get("score", 0.0)), places=9)


if __name__ == "__main__":
    unittest.main()
