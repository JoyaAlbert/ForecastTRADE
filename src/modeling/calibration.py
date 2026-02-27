from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


class LocalCalibrator:
    def __init__(self, backend, kind: str):
        self.backend = backend
        self.kind = kind

    def predict_proba(self, raw_scores: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw_scores, dtype=float).reshape(-1)
        if self.kind == "isotonic":
            p1 = self.backend.predict(raw)
        else:
            p1 = self.backend.predict_proba(raw.reshape(-1, 1))[:, 1]
        p1 = np.clip(p1, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p1, p1])


def fit_probability_calibrator(raw_train_proba, y_calib, method: str = "isotonic"):
    if len(raw_train_proba) < 60 or len(np.unique(y_calib)) < 2:
        return None, "insufficient_data_or_single_class"
    try:
        raw_train_proba = np.asarray(raw_train_proba, dtype=float)
        y_calib = np.asarray(y_calib).astype(int)
        if method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(raw_train_proba, y_calib)
            return LocalCalibrator(iso, "isotonic"), "ok"
        lr = LogisticRegression(solver="lbfgs", max_iter=500, random_state=42)
        lr.fit(raw_train_proba.reshape(-1, 1), y_calib)
        return LocalCalibrator(lr, "sigmoid"), "ok"
    except Exception as exc:
        msg = str(exc).replace("\n", " ").strip()
        return None, f"fit_error:{exc.__class__.__name__}:{msg}"


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (proba >= lo) & (proba <= hi)
        else:
            mask = (proba >= lo) & (proba < hi)
        if not np.any(mask):
            continue
        acc = np.mean(y_true[mask])
        conf = np.mean(proba[mask])
        ece += abs(acc - conf) * (np.sum(mask) / len(y_true))
    return float(ece)


def calibration_metrics(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    return {
        "brier": float(brier_score_loss(y_true, proba)),
        "ece": expected_calibration_error(y_true, proba, n_bins=10),
    }


def train_calibration_split(X_train, y_train, raw_train_proba, ratio: float = 0.2) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    n = len(y_train)
    if n < 120:
        return None
    split = int(n * (1.0 - ratio))
    y_cal = np.asarray(y_train)[split:]
    proba_cal = np.asarray(raw_train_proba)[split:]
    if len(np.unique(y_cal)) < 2:
        return None
    return proba_cal, y_cal
