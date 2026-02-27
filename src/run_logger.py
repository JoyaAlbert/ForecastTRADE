# -*- coding: utf-8 -*-
"""
Accumulative Run Logger for Model Performance Tracking
======================================================

Maintains a JSON log of all model runs including:
- Features used in each run
- Accuracy, AUC, Precision-Recall metrics
- Sharpe Ratio and Maximum Drawdown
- Training parameters and hyperparameters
- Timestamp and run metadata
"""

import json
import os
from datetime import datetime
import numpy as np
import pandas as pd

class AccumulativeRunLogger:
    """Logs and accumulates results from all model runs."""
    
    def __init__(self, log_file='out/runs_log.json'):
        """Initialize logger with file path."""
        self.log_file = log_file
        self.log_dir = os.path.dirname(log_file)
        
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        # Load existing log or create new one
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                self.runs = json.load(f)
        else:
            self.runs = {
                'metadata': {
                    'created': datetime.now().isoformat(),
                    'description': 'Accumulative log of LSTM-XGBoost hybrid model runs',
                    'total_runs': 0,
                'best_auc': 0.0,
                'best_accuracy': 0.0,
                'best_sharpe': -np.inf,
                'best_net_sharpe': -np.inf,
            },
                'runs': []
            }
    
    def log_run(self, run_config):
        """
        Log a complete model run with all metrics and configuration.
        
        Args:
            run_config (dict): Complete run configuration with:
                - features_used (list): Feature names
                - train_size (int): Training set size
                - test_size (int): Test set size
                - metrics (dict): Accuracy, AUC, Precision, Recall, F1
                - sharpe_ratio (float): Risk-adjusted return metric
                - max_drawdown (float): Maximum loss metric
                - xgb_params (dict): XGBoost hyperparameters
                - lstm_config (dict): LSTM configuration
                - fold_number (int): Cross-validation fold number
                - ticker (str): Stock ticker
                - run_date (str): Date of the run
                - run_number (int): Unique run identifier
        """
        run_entry = {
            'run_number': run_config.get('run_number', len(self.runs['runs']) + 1),
            'timestamp': datetime.now().isoformat(),
            'ticker': run_config.get('ticker', 'UNKNOWN'),
            'date': run_config.get('run_date', datetime.now().strftime('%Y-%m-%d')),
            'fold': run_config.get('fold_number', 0),
            
            # Features
            'features': {
                'count': len(run_config.get('features_used', [])),
                'names': run_config.get('features_used', []),
                'lstm_latent_dims': sum(1 for f in run_config.get('features_used', []) 
                                        if 'lstm_latent' in f),
                'technical_features': sum(1 for f in run_config.get('features_used', []) 
                                         if 'lstm' not in f)
            },
            'features_contract': run_config.get('features_contract', {}),
            
            # Dataset Info
            'dataset': {
                'train_size': run_config.get('train_size', 0),
                'test_size': run_config.get('test_size', 0),
                'total_samples': run_config.get('train_size', 0) + run_config.get('test_size', 0)
            },
            'cv': run_config.get('cv', {}),
            
            # Classification Metrics
            'metrics': {
                'accuracy': float(run_config.get('metrics', {}).get('accuracy', 0)),
                'precision': float(run_config.get('metrics', {}).get('precision', 0)),
                'recall': float(run_config.get('metrics', {}).get('recall', 0)),
                'f1_score': float(run_config.get('metrics', {}).get('f1', 0)),
                'auc_roc': float(run_config.get('metrics', {}).get('auc', 0)),
                'auc_pr': float(run_config.get('metrics', {}).get('auc_pr', 0))
            },
            
            # Financial Metrics
            'financial_metrics': {
                'sharpe_ratio': float(run_config.get('sharpe_ratio', 0)),
                'net_sharpe': float(run_config.get('net_sharpe', np.nan)),
                'max_drawdown': float(run_config.get('max_drawdown', 0)),
                'total_return': float(run_config.get('total_return', 0)),
                'net_return': float(run_config.get('net_return', np.nan)),
                'turnover': float(run_config.get('turnover', np.nan)),
                'win_rate': float(run_config.get('win_rate', 0))
            },
            'coverage_ratio': float(run_config.get('cv', {}).get('coverage_ratio', np.nan)),
            'calibration_error': float(run_config.get('metrics', {}).get('calibration_error', np.nan)),
            'recommendation_quality': float(run_config.get('recommendation_quality', np.nan)),
            
            # Model Configuration
            'xgboost_config': run_config.get('xgb_params', {}),
            'lstm_config': run_config.get('lstm_config', {}),
            
            # Predictions
            'predictions': {
                'buy_signals': int(run_config.get('buy_signals', 0)),
                'sell_signals': int(run_config.get('sell_signals', 0)),
                'hold_signals': int(run_config.get('hold_signals', 0))
            },
            
            # Outputs
            'output_files': {
                'visualization': run_config.get('visualization_file', ''),
                'forecast': run_config.get('forecast_file', '')
            },
            'runtime': run_config.get('runtime', {}),
            'timings': run_config.get('timings', {}),
            
            # Notes
            'notes': run_config.get('notes', '')
        }
        
        # Add to runs list
        self.runs['runs'].append(run_entry)
        
        # Update metadata
        self._update_metadata(run_entry)
        
        # Save to file
        self.save()
        
        return run_entry
    
    def _update_metadata(self, run_entry):
        """Update aggregate metadata after adding a run."""
        self.runs['metadata']['total_runs'] = len(self.runs['runs'])
        
        # Best metrics
        auc = run_entry['metrics']['auc_roc']
        accuracy = run_entry['metrics']['accuracy']
        sharpe = run_entry['financial_metrics']['sharpe_ratio']
        net_sharpe = run_entry['financial_metrics'].get('net_sharpe', np.nan)
        
        if auc > self.runs['metadata'].get('best_auc', 0):
            self.runs['metadata']['best_auc'] = auc
        if accuracy > self.runs['metadata'].get('best_accuracy', 0):
            self.runs['metadata']['best_accuracy'] = accuracy
        if sharpe > self.runs['metadata'].get('best_sharpe', -np.inf):
            self.runs['metadata']['best_sharpe'] = sharpe
        if np.isfinite(net_sharpe) and net_sharpe > self.runs['metadata'].get('best_net_sharpe', -np.inf):
            self.runs['metadata']['best_net_sharpe'] = float(net_sharpe)
        
        # Average metrics
        all_aucs = [r['metrics']['auc_roc'] for r in self.runs['runs']]
        all_accuracy = [r['metrics']['accuracy'] for r in self.runs['runs']]
        all_sharpe = [r['financial_metrics']['sharpe_ratio'] for r in self.runs['runs']]
        finite_sharpe = [s for s in all_sharpe if np.isfinite(s)]
        
        self.runs['metadata']['average_auc'] = float(np.mean(all_aucs)) if all_aucs else 0.0
        self.runs['metadata']['average_accuracy'] = float(np.mean(all_accuracy)) if all_accuracy else 0.0
        self.runs['metadata']['average_sharpe'] = float(np.nanmean(finite_sharpe)) if finite_sharpe else 0.0
        
        # Variance (stability metric)
        self.runs['metadata']['auc_variance'] = float(np.var(all_aucs)) if all_aucs else 0.0
        self.runs['metadata']['accuracy_variance'] = float(np.var(all_accuracy)) if all_accuracy else 0.0
        self.runs['metadata']['last_updated'] = datetime.now().isoformat()
    
    def save(self):
        """Save log to JSON file."""
        # Make all values JSON-serializable
        log_data = self._make_serializable(self.runs)
        
        with open(self.log_file, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)
    
    def get_summary(self):
        """Get summary statistics of all runs."""
        if not self.runs['runs']:
            return "No runs logged yet."
        
        df = pd.DataFrame(self.runs['runs'])
        
        summary = f"""
╔════════════════════════════════════════════════════════════╗
║        ACCUMULATIVE RUN LOG SUMMARY                        ║
╠════════════════════════════════════════════════════════════╣
║ Total Runs: {self.runs['metadata']['total_runs']:3d}                                    ║
║ Best AUC: {self.runs['metadata'].get('best_auc', 0):.4f}                              ║
║ Best Accuracy: {self.runs['metadata'].get('best_accuracy', 0):.4f}                      ║
║ Best Sharpe Ratio: {self.runs['metadata'].get('best_sharpe', 0):.4f}                  ║
║ Best Net Sharpe: {self.runs['metadata'].get('best_net_sharpe', 0):.4f}                   ║
║                                                            ║
║ Average AUC: {self.runs['metadata'].get('average_auc', 0):.4f}                         ║
║ Average Accuracy: {self.runs['metadata'].get('average_accuracy', 0):.4f}                 ║
║ Average Sharpe Ratio: {self.runs['metadata'].get('average_sharpe', 0):.4f}             ║
║                                                            ║
║ AUC Variance: {self.runs['metadata'].get('auc_variance', 0):.6f}                      ║
║ Accuracy Variance: {self.runs['metadata'].get('accuracy_variance', 0):.6f}             ║
╚════════════════════════════════════════════════════════════╝
        """
        return summary
    
    def get_run_details(self, run_number):
        """Get detailed information for a specific run."""
        for run in self.runs['runs']:
            if run['run_number'] == run_number:
                return run
        return None
    
    def export_csv(self, output_file='out/runs_summary.csv'):
        """Export run metrics to CSV for analysis."""
        if not self.runs['runs']:
            print("No runs to export.")
            return
        
        # Flatten the runs data for CSV
        rows = []
        for run in self.runs['runs']:
            row = {
                'run_number': run['run_number'],
                'timestamp': run['timestamp'],
                'ticker': run['ticker'],
                'fold': run['fold'],
                'cv_folds_configured': run.get('cv', {}).get('n_folds_configured'),
                'cv_folds_valid': run.get('cv', {}).get('n_folds_valid'),
                'cv_folds_skipped': run.get('cv', {}).get('n_folds_skipped'),
                'features_count': run['features']['count'],
                'lstm_dims': run['features']['lstm_latent_dims'],
                'train_size': run['dataset']['train_size'],
                'test_size': run['dataset']['test_size'],
                'accuracy': run['metrics']['accuracy'],
                'auc_roc': run['metrics']['auc_roc'],
                'auc_pr': run['metrics']['auc_pr'],
                'precision': run['metrics']['precision'],
                'recall': run['metrics']['recall'],
                'f1': run['metrics']['f1_score'],
                'sharpe_ratio': run['financial_metrics']['sharpe_ratio'],
                'net_sharpe': run['financial_metrics'].get('net_sharpe'),
                'max_drawdown': run['financial_metrics']['max_drawdown'],
                'total_return': run['financial_metrics']['total_return'],
                'net_return': run['financial_metrics'].get('net_return'),
                'turnover': run['financial_metrics'].get('turnover'),
                'coverage_ratio': run.get('coverage_ratio'),
                'calibration_error': run.get('calibration_error'),
                'recommendation_quality': run.get('recommendation_quality'),
                'buy_signals': run['predictions']['buy_signals'],
                'sell_signals': run['predictions']['sell_signals']
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
        print(f"   ✅ Run summary exported to {output_file}")
        
        return df
    
    @staticmethod
    def _make_serializable(obj):
        """Convert numpy/special types to JSON-serializable types."""
        if isinstance(obj, dict):
            return {k: AccumulativeRunLogger._make_serializable(v) 
                   for k, v in obj.items()}
        elif isinstance(obj, list):
            return [AccumulativeRunLogger._make_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, (np.ndarray, pd.Series)):
            return obj.tolist()
        else:
            return obj
