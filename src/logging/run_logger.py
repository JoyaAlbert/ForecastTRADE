# -*- coding: utf-8 -*-
"""
Run Logger - Cumulative JSON Logging System
===========================================

Maintains a cumulative log of all model runs with metrics, features,
accuracy, AUC, and timestamps. Enables tracking of model performance
evolution over time.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any


class RunLogger:
    """Manages cumulative JSON logging of model runs."""
    
    def __init__(self, log_dir: str = "out"):
        """
        Initialize the run logger.
        
        Args:
            log_dir (str): Directory where logs will be stored
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "runs.json"
        self.runs = self._load_existing_runs()
    
    def _load_existing_runs(self) -> List[Dict[str, Any]]:
        """Load existing runs from JSON file if it exists."""
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r') as f:
                    data = json.load(f)
                    return data.get('runs', [])
            except (json.JSONDecodeError, IOError):
                return []
        return []
    
    def log_run(self, 
                run_name: str,
                model_type: str,
                accuracy: float,
                auc: float,
                precision: float,
                recall: float,
                f1_score: float,
                features_used: List[str],
                n_features: int,
                fold_scores: Dict[str, float],
                sharpe_ratio: float = None,
                max_drawdown: float = None,
                parameters: Dict[str, Any] = None,
                notes: str = "") -> int:
        """
        Log a single model run.
        
        Args:
            run_name (str): Name/description of the run
            model_type (str): Type of model (lstm, xgboost, hybrid)
            accuracy (float): Model accuracy
            auc (float): Area under ROC curve
            precision (float): Precision score
            recall (float): Recall score
            f1_score (float): F1 score
            features_used (List[str]): List of feature names used
            n_features (int): Number of features
            fold_scores (Dict[str, float]): CV fold scores
            sharpe_ratio (float): Sharpe ratio (optional)
            max_drawdown (float): Maximum drawdown (optional)
            parameters (Dict[str, Any]): Model hyperparameters
            notes (str): Additional notes
            
        Returns:
            int: Run ID (sequential)
        """
        run_id = len(self.runs) + 1
        
        run_record = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "run_name": run_name,
            "model_type": model_type,
            "metrics": {
                "accuracy": round(accuracy, 6),
                "auc": round(auc, 6),
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1_score": round(f1_score, 6),
                "sharpe_ratio": round(sharpe_ratio, 6) if sharpe_ratio else None,
                "max_drawdown": round(max_drawdown, 6) if max_drawdown else None
            },
            "features": {
                "total_count": n_features,
                "names": features_used,
                "engineered_count": len([f for f in features_used if 'lstm' in f or 'fractional' in f])
            },
            "cross_validation": fold_scores,
            "parameters": parameters or {},
            "notes": notes
        }
        
        self.runs.append(run_record)
        self._save_runs()
        
        print(f"✅ Run #{run_id} logged: {run_name}")
        return run_id
    
    def _save_runs(self):
        """Save all runs to JSON file."""
        data = {
            "metadata": {
                "last_updated": datetime.now().isoformat(),
                "total_runs": len(self.runs)
            },
            "runs": self.runs
        }
        
        with open(self.log_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_run(self, run_id: int) -> Dict[str, Any]:
        """Get a specific run by ID."""
        for run in self.runs:
            if run['run_id'] == run_id:
                return run
        return None
    
    def get_all_runs(self) -> List[Dict[str, Any]]:
        """Get all logged runs."""
        return self.runs
    
    def get_runs_by_model(self, model_type: str) -> List[Dict[str, Any]]:
        """Get all runs of a specific model type."""
        return [r for r in self.runs if r['model_type'] == model_type]
    
    def get_best_run(self, metric: str = 'auc') -> Dict[str, Any]:
        """Get the best run by a specific metric."""
        if not self.runs:
            return None
        return max(self.runs, key=lambda r: r['metrics'].get(metric, 0))
    
    def get_comparison_table(self) -> str:
        """Generate a text table comparing all runs."""
        if not self.runs:
            return "No runs logged yet."
        
        lines = []
        lines.append("=" * 120)
        lines.append(f"{'ID':<4} {'Run Name':<20} {'Model':<10} {'Accuracy':<10} {'AUC':<10} {'F1':<10} {'Features':<10} {'Sharpe':<10}")
        lines.append("=" * 120)
        
        for run in self.runs:
            metrics = run['metrics']
            lines.append(
                f"{run['run_id']:<4} "
                f"{run['run_name']:<20} "
                f"{run['model_type']:<10} "
                f"{metrics['accuracy']:<10.6f} "
                f"{metrics['auc']:<10.6f} "
                f"{metrics['f1_score']:<10.6f} "
                f"{run['features']['total_count']:<10} "
                f"{str(metrics['sharpe_ratio']):<10}"
            )
        
        lines.append("=" * 120)
        return "\n".join(lines)
    
    def print_summary(self):
        """Print a summary of all runs."""
        print("\n" + self.get_comparison_table())
        
        if self.runs:
            best = self.get_best_run('auc')
            print(f"\n🏆 Best Run (AUC): #{best['run_id']} - {best['run_name']}")
            print(f"   AUC: {best['metrics']['auc']:.6f}, Accuracy: {best['metrics']['accuracy']:.6f}")


if __name__ == "__main__":
    # Example usage
    logger = RunLogger()
    
    # Log a sample run
    logger.log_run(
        run_name="LSTM Feature Extraction v1",
        model_type="lstm",
        accuracy=0.62,
        auc=0.68,
        precision=0.65,
        recall=0.60,
        f1_score=0.62,
        features_used=['lstm_latent_0', 'lstm_latent_1', 'rsi', 'volatility_20'],
        n_features=4,
        fold_scores={'fold_1': 0.67, 'fold_2': 0.69, 'fold_3': 0.66},
        sharpe_ratio=0.85,
        notes="Initial LSTM run"
    )
    
    logger.print_summary()
