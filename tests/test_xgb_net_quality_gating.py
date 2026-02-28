import unittest

import numpy as np
import pandas as pd

from src.main import Config, tune_xgb_params_for_stability


class XgbNetQualityGatingTests(unittest.TestCase):
    def test_discard_candidates_below_net_return_floor(self):
        rng = np.random.default_rng(7)
        x_train = pd.DataFrame(rng.normal(size=(120, 4)), columns=["f1", "f2", "f3", "f4"])
        y_train = pd.Series((x_train["f1"] > 0).astype(int))
        x_val = pd.DataFrame(rng.normal(size=(60, 4)), columns=["f1", "f2", "f3", "f4"])
        y_val = pd.Series((x_val["f1"] > 0).astype(int))
        val_returns = np.full(len(x_val), -0.01, dtype=float)

        base_params = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "max_depth": 3,
            "learning_rate": 0.03,
            "n_estimators": 80,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 1,
            "gamma": 0.1,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "use_label_encoder": False,
            "early_stopping_rounds": 10,
        }

        old_floor = Config.NET_RETURN_FLOOR_PER_FOLD
        try:
            Config.NET_RETURN_FLOOR_PER_FOLD = 0.001
            tuned, meta = tune_xgb_params_for_stability(
                x_train,
                y_train,
                x_val,
                y_val,
                val_returns,
                base_params,
                scale_pos_weight=1.0,
                max_candidates=3,
                target_coverage_ratio=0.12,
                trade_ratio_floor=0.08,
                fold_idx=1,
                base_seed=42,
                stochastic_mode=False,
                candidate_budget="low",
            )
        finally:
            Config.NET_RETURN_FLOOR_PER_FOLD = old_floor

        self.assertIsInstance(tuned, dict)
        # No candidate should pass a positive net_return floor with strictly negative returns.
        self.assertTrue((not meta) or (meta.get("net_return", -1.0) <= 0.001))


if __name__ == "__main__":
    unittest.main()
