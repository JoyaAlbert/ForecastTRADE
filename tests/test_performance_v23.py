import unittest

import pandas as pd

from src.main import drop_redundant_seq_equivalents, get_feature_columns
from src.modeling.calibration import train_calibration_split


class PerformanceV23Tests(unittest.TestCase):
    def test_drop_redundant_seq_equivalents_prefers_lstm(self):
        df = pd.DataFrame(
            {
                "close": [100.0, 101.0, 102.0],
                "lstm_price_5d": [101.0, 102.0, 103.0],
                "tcn_price_5d": [101.0, 102.0, 103.0],
                "lstm_return_5d": [0.01, 0.02, 0.01],
                "tcn_return_5d": [0.01, 0.02, 0.01],
            }
        )
        out, dropped = drop_redundant_seq_equivalents(df)
        self.assertIn("tcn_price_5d", dropped)
        self.assertIn("tcn_return_5d", dropped)
        self.assertIn("lstm_price_5d", out.columns)
        self.assertNotIn("tcn_price_5d", out.columns)

    def test_get_feature_columns_filters_tcn_duplicates(self):
        df = pd.DataFrame(
            {
                "lstm_return_5d": [0.1, 0.2],
                "tcn_return_5d": [0.1, 0.2],
                "macd": [1.0, 1.1],
            }
        )
        feature_contract = {"final": ["lstm_return_5d", "tcn_return_5d", "macd"]}
        cols = get_feature_columns(df, feature_contract=feature_contract)
        self.assertIn("lstm_return_5d", cols)
        self.assertIn("macd", cols)
        self.assertNotIn("tcn_return_5d", cols)

    def test_train_calibration_split_for_small_fold(self):
        n = 120
        y = [0, 1] * (n // 2)
        proba = [0.2, 0.8] * (n // 2)
        split = train_calibration_split(X_train=None, y_train=y, raw_train_proba=proba, ratio=0.30)
        self.assertIsNotNone(split)
        p_cal, y_cal = split
        self.assertGreaterEqual(len(p_cal), 30)
        self.assertGreaterEqual(len(set(y_cal)), 2)


if __name__ == "__main__":
    unittest.main()
