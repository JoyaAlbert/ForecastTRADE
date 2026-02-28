#!/usr/bin/env python3
"""Benchmark multiple configuration candidates across multiple seeds and tickers."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def default_candidates() -> list[dict[str, Any]]:
    return [
        {
            "id": "c1_balanced_sigmoid",
            "target_coverage": 0.16,
            "trade_ratio_floor": 0.09,
            "conservative_th_min": 0.60,
            "conservative_th_max": 0.64,
            "calibration_method": "sigmoid",
            "conservative_percentile_threshold": 70,
        },
        {
            "id": "c2_run23_like",
            "target_coverage": 0.18,
            "trade_ratio_floor": 0.10,
            "conservative_th_min": 0.58,
            "conservative_th_max": 0.64,
            "calibration_method": "isotonic",
            "conservative_percentile_threshold": 70,
        },
        {
            "id": "c3_run23_like_sigmoid",
            "target_coverage": 0.18,
            "trade_ratio_floor": 0.10,
            "conservative_th_min": 0.58,
            "conservative_th_max": 0.64,
            "calibration_method": "sigmoid",
            "conservative_percentile_threshold": 70,
        },
        {
            "id": "c4_conservative_mid",
            "target_coverage": 0.14,
            "trade_ratio_floor": 0.08,
            "conservative_th_min": 0.62,
            "conservative_th_max": 0.66,
            "calibration_method": "sigmoid",
            "conservative_percentile_threshold": 72,
        },
        {
            "id": "c5_safer_isotonic",
            "target_coverage": 0.15,
            "trade_ratio_floor": 0.08,
            "conservative_th_min": 0.61,
            "conservative_th_max": 0.65,
            "calibration_method": "isotonic",
            "conservative_percentile_threshold": 72,
        },
        {
            "id": "c6_more_selective",
            "target_coverage": 0.13,
            "trade_ratio_floor": 0.08,
            "conservative_th_min": 0.63,
            "conservative_th_max": 0.67,
            "calibration_method": "sigmoid",
            "conservative_percentile_threshold": 74,
        },
    ]


def normalize_candidate(cand: dict[str, Any], idx: int) -> dict[str, Any]:
    out = dict(cand)
    out.setdefault("id", f"candidate_{idx+1}")
    out["target_coverage"] = float(out.get("target_coverage", 0.16))
    out["trade_ratio_floor"] = float(out.get("trade_ratio_floor", 0.09))
    out["conservative_th_min"] = float(out.get("conservative_th_min", 0.60))
    out["conservative_th_max"] = float(out.get("conservative_th_max", 0.64))
    if out["conservative_th_min"] > out["conservative_th_max"]:
        out["conservative_th_min"], out["conservative_th_max"] = out["conservative_th_max"], out["conservative_th_min"]
    out["calibration_method"] = str(out.get("calibration_method", "sigmoid"))
    out["conservative_percentile_threshold"] = int(out.get("conservative_percentile_threshold", 70))
    out["xgb_stochastic_mode"] = bool(out.get("xgb_stochastic_mode", True))
    out["xgb_candidate_budget"] = str(out.get("xgb_candidate_budget", "medium")).lower()
    if out["xgb_candidate_budget"] not in {"low", "medium", "high"}:
        out["xgb_candidate_budget"] = "medium"
    return out


def parse_tickers(raw: str | None, base_cfg: dict, fallback: str) -> list[str]:
    if raw:
        tickers = [t.strip().upper() for t in str(raw).split(",") if t.strip()]
        return tickers if tickers else [fallback.upper()]
    cfg_tickers = base_cfg.get("BENCHMARK_TICKERS", None)
    if isinstance(cfg_tickers, list) and cfg_tickers:
        return [str(t).strip().upper() for t in cfg_tickers if str(t).strip()]
    return [fallback.upper()]


def load_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.candidates_json:
        with open(args.candidates_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("candidates json must be a list of candidate objects")
        candidates = [normalize_candidate(c, i) for i, c in enumerate(data)]
    else:
        candidates = [normalize_candidate(c, i) for i, c in enumerate(default_candidates())]

    if args.xgb_stochastic_mode is not None:
        forced_mode = args.xgb_stochastic_mode == "on"
        for c in candidates:
            c["xgb_stochastic_mode"] = forced_mode
    if args.xgb_candidate_budget is not None:
        forced_budget = args.xgb_candidate_budget
        for c in candidates:
            c["xgb_candidate_budget"] = forced_budget
    return candidates


def apply_candidate(base_cfg: dict, seed: int, cand: dict[str, Any], encoder_mode: str) -> dict:
    cfg = dict(base_cfg)
    cfg["SEED"] = int(seed)
    cfg["XGB_STOCHASTIC_MODE"] = bool(cand.get("xgb_stochastic_mode", True))
    cfg["XGB_CANDIDATE_BUDGET"] = str(cand.get("xgb_candidate_budget", "medium"))
    cfg["TARGET_COVERAGE_RATIO"] = float(cand["target_coverage"])
    cfg["TRADE_RATIO_FLOOR"] = float(cand["trade_ratio_floor"])
    cfg["CONSERVATIVE_MIN_BUY_THRESHOLD_MIN"] = float(cand["conservative_th_min"])
    cfg["CONSERVATIVE_MIN_BUY_THRESHOLD_MAX"] = float(cand["conservative_th_max"])
    cfg["CALIBRATION_METHOD"] = str(cand["calibration_method"])
    cfg["CONSERVATIVE_PERCENTILE_THRESHOLD"] = int(cand["conservative_percentile_threshold"])
    cfg["SEQ_ENCODER"] = str(encoder_mode)
    return cfg


def run_one(seed: int, ticker: str, encoder_mode: str, candidate: dict[str, Any], base_cfg: dict, args: argparse.Namespace) -> dict:
    cfg = apply_candidate(base_cfg, seed, candidate, encoder_mode=encoder_mode)
    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "bench_config.yaml"
        save_yaml(cfg_path, cfg)
        cmd = [
            sys.executable,
            "run.py",
            "--mode",
            args.mode,
            "--plots",
            args.plots,
            "--cache",
            args.cache,
            "--profile",
            args.profile,
            "--ticker",
            ticker,
            "--seq-encoder",
            encoder_mode,
            "--no-ui",
            "--risk-profile",
            args.risk_profile,
            "--objective",
            args.objective,
            "--config",
            str(cfg_path),
        ]
        subprocess.run(cmd, check=True)

    df = pd.read_csv(args.summary_csv)
    last = df.iloc[-1]
    configured_folds = float(last.get("cv_folds_configured", 0.0))
    valid_folds = float(last.get("cv_folds_valid", 0.0))
    valid_fold_ratio = (valid_folds / configured_folds) if configured_folds > 0 else 0.0
    result = {
        "candidate_id": candidate["id"],
        "ticker": str(ticker),
        "encoder_mode": str(encoder_mode),
        "seed": int(seed),
        "run_number": int(last.get("run_number", -1)),
        "auc": float(last.get("auc_roc", float("nan"))),
        "accuracy": float(last.get("accuracy", float("nan"))),
        "net_sharpe": float(last.get("net_sharpe", float("nan"))),
        "coverage_ratio": float(last.get("coverage_ratio", float("nan"))),
        "trade_coverage": float(last.get("trade_coverage", float("nan"))),
        "turnover": float(last.get("turnover", float("nan"))),
        "valid_folds_ratio": float(valid_fold_ratio),
    }
    result["score"] = (
        (result["net_sharpe"] * args.w_net_sharpe)
        + (result["coverage_ratio"] * args.w_coverage)
        + (result["auc"] * args.w_auc)
        - (result["turnover"] * args.w_turnover)
    )
    return result


def summarize_per_combo(rows: list[dict[str, Any]], candidates: list[dict[str, Any]], std_weight: float) -> pd.DataFrame:
    per_run = pd.DataFrame(rows)
    agg = (
        per_run.groupby(["candidate_id", "ticker", "encoder_mode"], as_index=False)
        .agg(
            score_mean=("score", "mean"),
            score_std=("score", "std"),
            net_sharpe_mean=("net_sharpe", "mean"),
            net_sharpe_std=("net_sharpe", "std"),
            coverage_mean=("coverage_ratio", "mean"),
            coverage_std=("coverage_ratio", "std"),
            valid_folds_ratio_mean=("valid_folds_ratio", "mean"),
            auc_mean=("auc", "mean"),
            accuracy_mean=("accuracy", "mean"),
            turnover_mean=("turnover", "mean"),
            runs=("run_number", "count"),
        )
    )
    w = float(std_weight)
    agg["robust_score"] = agg["score_mean"] - (w * agg["score_std"].fillna(0.0))
    agg["robust_net_sharpe"] = agg["net_sharpe_mean"] - (w * agg["net_sharpe_std"].fillna(0.0))
    params_df = pd.DataFrame(candidates)
    return agg.merge(params_df, left_on="candidate_id", right_on="id", how="left")


def select_encoder_modes(
    combo_summary: pd.DataFrame,
    encoder_required_delta_robust_score: float,
    encoder_required_delta_net_sharpe: float,
) -> pd.DataFrame:
    selections: list[dict[str, Any]] = []
    keys = combo_summary[["candidate_id", "ticker"]].drop_duplicates()
    for _, key in keys.iterrows():
        cid = key["candidate_id"]
        ticker = key["ticker"]
        sub = combo_summary[(combo_summary["candidate_id"] == cid) & (combo_summary["ticker"] == ticker)]
        row_off = sub[sub["encoder_mode"] == "off"]
        row_tcn = sub[sub["encoder_mode"] == "tcn"]
        if row_tcn.empty and row_off.empty:
            continue
        if row_tcn.empty:
            chosen = row_off.iloc[0].to_dict()
            reason = "only_off_available"
        elif row_off.empty:
            chosen = row_tcn.iloc[0].to_dict()
            reason = "only_tcn_available"
        else:
            off = row_off.iloc[0]
            tcn = row_tcn.iloc[0]
            robust_delta = float(tcn["robust_score"] - off["robust_score"])
            net_delta = float(tcn["net_sharpe_mean"] - off["net_sharpe_mean"])
            if robust_delta >= float(encoder_required_delta_robust_score) and net_delta >= float(encoder_required_delta_net_sharpe):
                chosen = tcn.to_dict()
                reason = "tcn_improves_robust_and_net"
            else:
                chosen = off.to_dict()
                reason = "off_is_more_robust"
        chosen["encoder_selection_reason"] = reason
        selections.append(chosen)
    return pd.DataFrame(selections)


def summarize_selected(rows: list[dict[str, Any]], selected_encoders: pd.DataFrame, candidates: list[dict[str, Any]], std_weight: float) -> pd.DataFrame:
    per_run = pd.DataFrame(rows)
    if per_run.empty or selected_encoders.empty:
        return pd.DataFrame()

    selection_keys = selected_encoders[["candidate_id", "ticker", "encoder_mode"]].drop_duplicates()
    selected_runs = per_run.merge(selection_keys, on=["candidate_id", "ticker", "encoder_mode"], how="inner")

    agg = (
        selected_runs.groupby("candidate_id", as_index=False)
        .agg(
            score_mean=("score", "mean"),
            score_std=("score", "std"),
            net_sharpe_mean=("net_sharpe", "mean"),
            net_sharpe_std=("net_sharpe", "std"),
            coverage_mean=("coverage_ratio", "mean"),
            coverage_std=("coverage_ratio", "std"),
            valid_folds_ratio_mean=("valid_folds_ratio", "mean"),
            auc_mean=("auc", "mean"),
            accuracy_mean=("accuracy", "mean"),
            turnover_mean=("turnover", "mean"),
            runs=("run_number", "count"),
            tickers=("ticker", "nunique"),
        )
    )
    w = float(std_weight)
    agg["robust_score"] = agg["score_mean"] - (w * agg["score_std"].fillna(0.0))
    agg["robust_net_sharpe"] = agg["net_sharpe_mean"] - (w * agg["net_sharpe_std"].fillna(0.0))
    agg = agg.sort_values("robust_score", ascending=False)
    params_df = pd.DataFrame(candidates)
    return agg.merge(params_df, left_on="candidate_id", right_on="id", how="left")


def choose_best(summary_df: pd.DataFrame, args: argparse.Namespace) -> tuple[dict[str, Any], bool]:
    constrained = summary_df[
        (summary_df["net_sharpe_mean"] >= float(args.min_net_sharpe))
        & (summary_df["coverage_mean"] >= float(args.min_coverage))
        & (summary_df["valid_folds_ratio_mean"] >= float(args.min_valid_folds_ratio))
    ]
    if not constrained.empty:
        best_row = constrained.sort_values("robust_score", ascending=False).iloc[0].to_dict()
        return best_row, False
    best_row = summary_df.sort_values("robust_score", ascending=False).iloc[0].to_dict()
    return best_row, True


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark config candidates over multiple seeds and tickers.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--summary-csv", default="out/runs_summary.csv")
    parser.add_argument("--seeds", default="42,77,101,123,777")
    parser.add_argument("--ticker", default="MSFT", help="Fallback single ticker")
    parser.add_argument("--tickers", default=None, help="Comma-separated benchmark tickers; overrides config BENCHMARK_TICKERS")
    parser.add_argument("--mode", default="full", choices=["full", "fast"])
    parser.add_argument("--plots", default="none", choices=["none", "final", "all"])
    parser.add_argument("--cache", default="on", choices=["on", "off"])
    parser.add_argument("--profile", default="off", choices=["on", "off"])
    parser.add_argument("--risk-profile", default="conservative")
    parser.add_argument("--objective", default="sharpe_net")
    parser.add_argument("--candidates-json", default=None, help="Path to json list of candidate configs")
    parser.add_argument("--min-net-sharpe", type=float, default=0.0)
    parser.add_argument("--min-coverage", type=float, default=0.83)
    parser.add_argument("--min-valid-folds-ratio", type=float, default=0.83)
    parser.add_argument("--w-net-sharpe", type=float, default=0.60)
    parser.add_argument("--w-coverage", type=float, default=0.30)
    parser.add_argument("--w-auc", type=float, default=0.15)
    parser.add_argument("--w-turnover", type=float, default=0.01)
    parser.add_argument("--xgb-stochastic-mode", choices=["on", "off"], default="on")
    parser.add_argument("--xgb-candidate-budget", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--encoder-ablation", choices=["on", "off"], default="on")
    parser.add_argument("--seq-encoder", choices=["off", "tcn", "lstm"], default="tcn", help="Used only when --encoder-ablation=off")
    parser.add_argument("--robust-std-weight", type=float, default=0.5)
    parser.add_argument("--encoder-required-delta-robust-score", type=float, default=0.02)
    parser.add_argument("--encoder-required-delta-net-sharpe", type=float, default=0.05)
    parser.add_argument("--out-json", default="out/benchmark_defaults_grid.json")
    parser.add_argument("--out-csv", default="out/benchmark_defaults_grid.csv")
    parser.add_argument("--best-config-out", default="out/recommended_default_config.yaml")
    args = parser.parse_args()

    base_cfg = load_yaml(Path(args.config))
    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    tickers = parse_tickers(args.tickers, base_cfg, fallback=args.ticker)
    candidates = load_candidates(args)
    encoder_modes = [args.seq_encoder] if args.encoder_ablation == "off" else ["off", "tcn"]

    rows: list[dict[str, Any]] = []
    for cand in candidates:
        print(f"\n### candidate={cand['id']} ###")
        for ticker in tickers:
            for encoder_mode in encoder_modes:
                print(f"  -> ticker={ticker} encoder={encoder_mode}")
                for seed in seeds:
                    row = run_one(seed, ticker, encoder_mode, cand, base_cfg, args)
                    rows.append(row)
                    print(
                        f"seed={seed} run={row['run_number']} score={row['score']:.4f} "
                        f"net_sharpe={row['net_sharpe']:.4f} coverage={row['coverage_ratio']:.4f} "
                        f"valid_folds_ratio={row['valid_folds_ratio']:.3f}"
                    )

    combo_summary = summarize_per_combo(rows, candidates, std_weight=args.robust_std_weight)
    selected_encoders = select_encoder_modes(
        combo_summary,
        encoder_required_delta_robust_score=args.encoder_required_delta_robust_score,
        encoder_required_delta_net_sharpe=args.encoder_required_delta_net_sharpe,
    )
    final_summary = summarize_selected(rows, selected_encoders, candidates, std_weight=args.robust_std_weight)

    if final_summary.empty:
        raise RuntimeError("No benchmark rows available after encoder selection")

    best, relaxed = choose_best(final_summary, args)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "params": {
                    "seeds": seeds,
                    "tickers": tickers,
                    "encoder_modes": encoder_modes,
                    "constraints": {
                        "min_net_sharpe": args.min_net_sharpe,
                        "min_coverage": args.min_coverage,
                        "min_valid_folds_ratio": args.min_valid_folds_ratio,
                    },
                    "weights": {
                        "w_net_sharpe": args.w_net_sharpe,
                        "w_coverage": args.w_coverage,
                        "w_auc": args.w_auc,
                        "w_turnover": args.w_turnover,
                    },
                    "robust_std_weight": args.robust_std_weight,
                    "encoder_gates": {
                        "required_delta_robust_score": args.encoder_required_delta_robust_score,
                        "required_delta_net_sharpe": args.encoder_required_delta_net_sharpe,
                    },
                    "constraint_relaxed": relaxed,
                    "promote_default": (not relaxed),
                },
                "per_run": rows,
                "per_combo_summary": combo_summary.to_dict(orient="records"),
                "encoder_selection": selected_encoders.to_dict(orient="records"),
                "summary": final_summary.to_dict(orient="records"),
                "best_candidate": best,
            },
            f,
            indent=2,
        )
    final_summary.to_csv(args.out_csv, index=False)

    best_cfg = dict(base_cfg)
    if not relaxed:
        chosen_rows = selected_encoders[selected_encoders["candidate_id"] == (best.get("candidate_id") or best.get("id"))]
        encoder_mode = "tcn"
        if not chosen_rows.empty:
            counts = chosen_rows["encoder_mode"].value_counts()
            encoder_mode = str(counts.index[0])
            best_cfg["SELECTED_ENCODER_BY_TICKER"] = {
                str(r["ticker"]): str(r["encoder_mode"])
                for _, r in chosen_rows.iterrows()
            }
        best_cfg = apply_candidate(best_cfg, int(best_cfg.get("SEED", 42)), best, encoder_mode=encoder_mode)
    save_yaml(Path(args.best_config_out), best_cfg)

    print("\n=== Best candidate ===")
    print(
        f"id={best.get('candidate_id') or best.get('id')} "
        f"score_mean={best.get('score_mean', float('nan')):.4f} "
        f"robust_score={best.get('robust_score', float('nan')):.4f} "
        f"net_sharpe_mean={best.get('net_sharpe_mean', float('nan')):.4f} "
        f"net_sharpe_std={best.get('net_sharpe_std', float('nan')):.4f} "
        f"coverage_mean={best.get('coverage_mean', float('nan')):.4f} "
        f"valid_folds_ratio_mean={best.get('valid_folds_ratio_mean', float('nan')):.4f}"
    )
    if relaxed:
        print("⚠️ Constraints not met: default promotion blocked, best-config-out keeps baseline config")
    print(f"saved {args.out_json}")
    print(f"saved {args.out_csv}")
    print(f"saved {args.best_config_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
