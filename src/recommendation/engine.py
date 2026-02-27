from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class RecommendationDecision:
    state: str
    expected_value: float
    recommendation_quality: float
    reasons: str


def compute_expected_value(probability: float, tp_pct: float, sl_pct: float, cost_pct: float) -> float:
    p = float(probability)
    return float((p * tp_pct) - ((1.0 - p) * abs(sl_pct)) - cost_pct)


def conservative_recommendation(
    probability: float,
    dynamic_threshold: float,
    tp_pct: float,
    sl_pct: float,
    cost_pct: float,
    regime_allowed: bool = True,
) -> RecommendationDecision:
    ev = compute_expected_value(probability, tp_pct, sl_pct, cost_pct)
    if not regime_allowed:
        return RecommendationDecision("NO_TRADE", ev, 0.0, "blocked_by_regime")
    if ev <= 0:
        return RecommendationDecision("WATCHLIST", ev, max(0.0, probability - 0.5), "non_positive_ev")
    if probability < dynamic_threshold:
        return RecommendationDecision("WATCHLIST", ev, max(0.0, probability - 0.5), "below_dynamic_threshold")

    quality = min(1.0, max(0.0, (probability - dynamic_threshold) / max(1e-6, 1.0 - dynamic_threshold)))
    if quality >= 0.7:
        return RecommendationDecision("ENTER_FULL", ev, quality, "high_confidence_positive_ev")
    if quality >= 0.35:
        return RecommendationDecision("ENTER_SMALL", ev, quality, "moderate_confidence_positive_ev")
    return RecommendationDecision("WATCHLIST", ev, quality, "low_margin_positive_ev")


def decision_to_payload(decision: RecommendationDecision) -> Dict[str, float]:
    return {
        "state": decision.state,
        "expected_value": decision.expected_value,
        "recommendation_quality": decision.recommendation_quality,
        "reasons": decision.reasons,
    }
