import tempfile
import unittest

import pandas as pd

from src.data_pipeline import load_market_data
from src.utils.runtime import CacheManager, RuntimeFlags, StageProfiler


class RuntimeFlagsTests(unittest.TestCase):
    def test_fast_mode_defaults_to_no_plots(self):
        flags = RuntimeFlags.normalize("fast", "all", "on", "on")
        self.assertEqual(flags.mode, "fast")
        self.assertEqual(flags.plots, "none")

    def test_full_mode_respects_plot_choice(self):
        flags = RuntimeFlags.normalize("full", "final", "on", "off")
        self.assertEqual(flags.mode, "full")
        self.assertEqual(flags.plots, "final")
        self.assertFalse(flags.profile_enabled)


class DataCacheTests(unittest.TestCase):
    def test_load_market_data_uses_cache(self):
        calls = {"n": 0}

        def fake_fetch(ticker, vix, start, end):
            calls["n"] += 1
            idx = pd.date_range("2024-01-01", periods=3, freq="D")
            return pd.DataFrame({"close": [1.0, 2.0, 3.0], "vix_close": [10.0, 11.0, 12.0]}, index=idx)

        with tempfile.TemporaryDirectory() as tmp:
            runtime = RuntimeFlags.normalize("full", "none", "on", "off")
            cache = CacheManager(root_dir=tmp, enabled=True)
            profiler = StageProfiler(enabled=False)

            df1 = load_market_data(fake_fetch, "MSFT", "^VIX", "2024-01-01", "2024-02-01", runtime, cache, profiler)
            df2 = load_market_data(fake_fetch, "MSFT", "^VIX", "2024-01-01", "2024-02-01", runtime, cache, profiler)

            self.assertEqual(calls["n"], 1)
            self.assertEqual(len(df1), 3)
            self.assertEqual(len(df2), 3)


if __name__ == "__main__":
    unittest.main()
