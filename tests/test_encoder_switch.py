import unittest

from src.main import _parse_cli_args
from src.utils.runtime import RuntimeFlags


class EncoderSwitchTests(unittest.TestCase):
    def test_cli_parser_accepts_seq_encoder_and_threshold_search(self):
        args = _parse_cli_args(
            [
                "--ticker",
                "MSFT",
                "--seq-encoder",
                "off",
                "--threshold-search",
                "quantile",
                "--target-coverage",
                "0.18",
                "--trade-ratio-floor",
                "0.09",
                "--conservative-th-min",
                "0.66",
                "--conservative-th-max",
                "0.69",
                "--xgb-stochastic-mode",
                "on",
                "--xgb-candidate-budget",
                "high",
                "--encoder-ablation",
                "on",
                "--benchmark-tickers",
                "MSFT,AAPL,NVDA",
                "--robust-std-weight",
                "0.5",
                "--net-return-floor-fold",
                "-0.002",
                "--no-ui",
            ]
        )
        self.assertEqual(args.seq_encoder, "off")
        self.assertEqual(args.threshold_search, "quantile")
        self.assertAlmostEqual(args.target_coverage, 0.18, places=6)
        self.assertAlmostEqual(args.trade_ratio_floor, 0.09, places=6)
        self.assertAlmostEqual(args.conservative_th_min, 0.66, places=6)
        self.assertAlmostEqual(args.conservative_th_max, 0.69, places=6)
        self.assertEqual(args.xgb_stochastic_mode, "on")
        self.assertEqual(args.xgb_candidate_budget, "high")
        self.assertEqual(args.encoder_ablation, "on")
        self.assertEqual(args.benchmark_tickers, "MSFT,AAPL,NVDA")
        self.assertAlmostEqual(args.robust_std_weight, 0.5, places=6)
        self.assertAlmostEqual(args.net_return_floor_fold, -0.002, places=6)

    def test_runtime_flags_normalize_new_fields(self):
        runtime = RuntimeFlags.normalize(
            mode="full",
            plots="none",
            cache="on",
            profile="off",
            seq_encoder="lstm",
            target_coverage=0.2,
            threshold_search="grid",
            trade_ratio_floor=0.09,
            conservative_th_min=0.66,
            conservative_th_max=0.69,
            xgb_stochastic_mode="on",
            xgb_candidate_budget="low",
            encoder_ablation="on",
            benchmark_tickers="MSFT,AAPL,NVDA",
            robust_std_weight=0.5,
            net_return_floor_fold=-0.002,
        )
        self.assertEqual(runtime.seq_encoder, "lstm")
        self.assertEqual(runtime.threshold_search, "grid")
        self.assertAlmostEqual(runtime.target_coverage, 0.2, places=6)
        self.assertAlmostEqual(runtime.trade_ratio_floor, 0.09, places=6)
        self.assertAlmostEqual(runtime.conservative_th_min, 0.66, places=6)
        self.assertAlmostEqual(runtime.conservative_th_max, 0.69, places=6)
        self.assertEqual(runtime.xgb_stochastic_mode, "on")
        self.assertEqual(runtime.xgb_candidate_budget, "low")
        self.assertEqual(runtime.encoder_ablation, "on")
        self.assertEqual(runtime.benchmark_tickers, "MSFT,AAPL,NVDA")
        self.assertAlmostEqual(runtime.robust_std_weight, 0.5, places=6)
        self.assertAlmostEqual(runtime.net_return_floor_fold, -0.002, places=6)


if __name__ == "__main__":
    unittest.main()
