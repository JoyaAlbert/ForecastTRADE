import unittest

import numpy as np
import pandas as pd

from src.tcn_predictor import (
    generate_placeholder_tcn_features,
    generate_tcn_regime_features_train_only,
)


class TcnPredictorTests(unittest.TestCase):
    def _build_df(self, n=80):
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        close = np.linspace(100, 120, n)
        return pd.DataFrame(
            {
                "open": close * 0.99,
                "high": close * 1.01,
                "low": close * 0.98,
                "close": close,
                "volume": np.full(n, 1000.0),
                "vix_close": np.linspace(18, 22, n),
            },
            index=idx,
        )

    def test_placeholder_has_tcn_and_lstm_contract(self):
        df = self._build_df(40)
        out = generate_placeholder_tcn_features(df)
        self.assertIn("tcn_return_5d", out.columns)
        self.assertIn("tcn_latent_0", out.columns)
        self.assertIn("lstm_return_5d", out.columns)
        self.assertIn("lstm_latent_0", out.columns)
        self.assertEqual(len(out), len(df))

    def test_train_only_handles_short_data_without_failure(self):
        train_df = self._build_df(80)
        full_df = self._build_df(90)
        out = generate_tcn_regime_features_train_only(train_df, full_df, ticker_name="MSFT")
        self.assertEqual(len(out), len(full_df))
        self.assertIn("lstm_return_confidence", out.columns)
        self.assertIn("tcn_return_confidence", out.columns)


if __name__ == "__main__":
    unittest.main()
