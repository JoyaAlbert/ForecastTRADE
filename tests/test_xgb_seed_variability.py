import unittest

import numpy as np
import pandas as pd

from src.main import tune_xgb_params_for_stability


class XgbSeedVariabilityTests(unittest.TestCase):
    def _make_data(self):
        rng = np.random.default_rng(123)
        x_train = pd.DataFrame(
            {
                "f1": rng.normal(0, 1, 90),
                "f2": rng.normal(0, 1, 90),
                "f3": rng.normal(0, 1, 90),
            }
        )
        y_train = pd.Series((x_train["f1"] + 0.2 * x_train["f2"] > 0).astype(int).values)
        x_val = pd.DataFrame(
            {
                "f1": rng.normal(0, 1, 45),
                "f2": rng.normal(0, 1, 45),
                "f3": rng.normal(0, 1, 45),
            }
        )
        y_val = pd.Series((x_val["f1"] + 0.2 * x_val["f2"] > 0).astype(int).values)
        val_returns = np.where(y_val.values == 1, 0.012, -0.009)
        return x_train, y_train, x_val, y_val, val_returns

    def test_stochastic_mode_changes_effective_seed(self):
        x_train, y_train, x_val, y_val, val_returns = self._make_data()
        base_params = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "max_depth": 3,
            "learning_rate": 0.05,
            "n_estimators": 80,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 1,
            "gamma": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "use_label_encoder": False,
            "early_stopping_rounds": 10,
        }
        _, meta_a = tune_xgb_params_for_stability(
            x_train,
            y_train,
            x_val,
            y_val,
            val_returns,
            base_params,
            scale_pos_weight=1.0,
            max_candidates=4,
            target_coverage_ratio=0.15,
            trade_ratio_floor=0.05,
            fold_idx=2,
            base_seed=42,
            stochastic_mode=True,
            candidate_budget="low",
        )
        _, meta_b = tune_xgb_params_for_stability(
            x_train,
            y_train,
            x_val,
            y_val,
            val_returns,
            base_params,
            scale_pos_weight=1.0,
            max_candidates=4,
            target_coverage_ratio=0.15,
            trade_ratio_floor=0.05,
            fold_idx=2,
            base_seed=43,
            stochastic_mode=True,
            candidate_budget="low",
        )
        self.assertNotEqual(meta_a.get("seed_eff"), meta_b.get("seed_eff"))


if __name__ == "__main__":
    unittest.main()
