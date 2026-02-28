import unittest

import pandas as pd

from scripts.benchmark_defaults import select_encoder_modes


class EncoderAblationSwitchTests(unittest.TestCase):
    def test_select_tcn_only_when_deltas_pass_thresholds(self):
        combo = pd.DataFrame(
            [
                {"candidate_id": "c1", "ticker": "MSFT", "encoder_mode": "off", "robust_score": 0.10, "net_sharpe_mean": 0.12},
                {"candidate_id": "c1", "ticker": "MSFT", "encoder_mode": "tcn", "robust_score": 0.14, "net_sharpe_mean": 0.19},
                {"candidate_id": "c2", "ticker": "MSFT", "encoder_mode": "off", "robust_score": 0.20, "net_sharpe_mean": 0.15},
                {"candidate_id": "c2", "ticker": "MSFT", "encoder_mode": "tcn", "robust_score": 0.21, "net_sharpe_mean": 0.16},
            ]
        )
        selected = select_encoder_modes(
            combo,
            encoder_required_delta_robust_score=0.02,
            encoder_required_delta_net_sharpe=0.05,
        )
        sel = {(r["candidate_id"], r["encoder_mode"]) for _, r in selected.iterrows()}
        self.assertIn(("c1", "tcn"), sel)
        self.assertIn(("c2", "off"), sel)


if __name__ == "__main__":
    unittest.main()
