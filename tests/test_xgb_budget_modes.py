import unittest

from src.main import build_xgb_tuning_candidates, get_xgb_candidate_budget_size


class XgbBudgetModeTests(unittest.TestCase):
    def test_budget_sizes_increase(self):
        low_n = len(build_xgb_tuning_candidates("low"))
        med_n = len(build_xgb_tuning_candidates("medium"))
        high_n = len(build_xgb_tuning_candidates("high"))
        self.assertGreaterEqual(low_n, 1)
        self.assertGreater(med_n, low_n)
        self.assertGreater(high_n, med_n)

    def test_budget_size_resolver(self):
        self.assertGreaterEqual(get_xgb_candidate_budget_size("low"), 1)
        self.assertEqual(
            len(build_xgb_tuning_candidates("medium")),
            get_xgb_candidate_budget_size("medium"),
        )

    def test_budget_filters_high_risk_combo(self):
        candidates = build_xgb_tuning_candidates("high")
        risky = [
            c for c in candidates
            if int(c.get("max_depth", 0)) >= 7 and float(c.get("learning_rate", 0.0)) >= 0.03
        ]
        self.assertEqual(len(risky), 0)


if __name__ == "__main__":
    unittest.main()
