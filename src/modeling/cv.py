from dataclasses import dataclass
from typing import Generator, Tuple

import numpy as np


@dataclass(frozen=True)
class PurgedEmbargoTimeSeriesSplit:
    n_splits: int
    test_size: int
    purge_size: int = 0
    embargo_size: int = 0
    min_train_size: int = 252

    def split(self, n_samples: int) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        if n_samples <= 0 or self.n_splits <= 0 or self.test_size <= 0:
            return
        step = self.test_size
        for split_idx in range(self.n_splits):
            test_start = split_idx * step + self.min_train_size
            test_end = min(test_start + self.test_size, n_samples)
            if test_start >= n_samples or test_end <= test_start:
                break

            train_end = max(0, test_start - self.purge_size)
            train_start = 0
            embargo_end = min(n_samples, test_end + self.embargo_size)

            train_idx = np.arange(train_start, train_end, dtype=int)
            # Remove future leakage zone by excluding [test_start, embargo_end)
            train_idx = train_idx[train_idx < test_start]
            test_idx = np.arange(test_start, test_end, dtype=int)
            if len(train_idx) >= self.min_train_size and len(test_idx) > 0:
                yield train_idx, test_idx


def sliding_window_splits(
    n_samples: int,
    train_size: int,
    test_size: int,
    n_splits: int = 8,
    min_test_size: int = 1,
):
    for i in range(n_splits):
        start_idx = i * test_size
        train_start = max(0, start_idx)
        train_end = start_idx + train_size
        test_start = train_end
        test_end = min(n_samples, test_start + test_size)
        if test_end > test_start and train_end <= n_samples:
            train_indices = np.arange(train_start, min(train_end, n_samples))
            test_indices = np.arange(test_start, test_end)
            if len(train_indices) >= train_size // 2 and len(test_indices) >= int(min_test_size):
                yield train_indices, test_indices
