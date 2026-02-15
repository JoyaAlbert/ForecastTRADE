#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Results Viewer - Visualize and Analyze Model Runs
==================================================

Quick script to view:
- All logged runs with metrics
- Aggregated performance statistics
- Feature usage across runs
- CSV export for external analysis
"""

import json
import os
import pandas as pd
from tabulate import tabulate
from datetime import datetime

class ResultsViewer:
    """Display and analyze accumulated model run results."""
    
    def __init__(self, log_file='out/runs_log.json', csv_file='out/runs_summary.csv'):
        self.log_file = log_file
        self.csv_file = csv_file
        self.runs = None
        self.df = None
        
        self._load_data()
    
    def _load_data(self):
        """Load JSON log and CSV data."""
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                data = json.load(f)
                self.runs = data.get('runs', [])
                self.metadata = data.get('metadata', {})
        
        if os.path.exists(self.csv_file):
            self.df = pd.read_csv(self.csv_file)
    
    def print_summary(self):
        """Print high-level summary of all runs."""
        if not self.runs:
            print("No runs logged yet.")
            return
        
        print("\n" + "="*80)
        print("                    ACCUMULATIVE RUN SUMMARY")
        print("="*80)
        
        print(f"\n📊 Total Runs: {len(self.runs)}")
        print(f"   Created: {self.metadata.get('created', 'Unknown')}")
        print(f"   Last Updated: {self.metadata.get('last_updated', 'Unknown')}")
        
        print(f"\n🏆 BEST PERFORMANCE:")
        print(f"   Best AUC: {self.metadata.get('best_auc', 0):.4f}")
        print(f"   Best Accuracy: {self.metadata.get('best_accuracy', 0):.4f}")
        print(f"   Best Sharpe Ratio: {self.metadata.get('best_sharpe', 0):.4f}")
        
        print(f"\n📈 AVERAGE METRICS:")
        print(f"   Avg AUC: {self.metadata.get('average_auc', 0):.4f}")
        print(f"   Avg Accuracy: {self.metadata.get('average_accuracy', 0):.4f}")
        print(f"   Avg Sharpe Ratio: {self.metadata.get('average_sharpe', 0):.4f}")
        
        print(f"\n📊 STABILITY (Lower is Better):")
        print(f"   AUC Variance: {self.metadata.get('auc_variance', 0):.6f}")
        print(f"   Accuracy Variance: {self.metadata.get('accuracy_variance', 0):.6f}")
        
        print("\n" + "="*80 + "\n")
    
    def print_latest_runs(self, n=5):
        """Print latest N runs with key metrics."""
        if not self.runs:
            print("No runs logged yet.")
            return
        
        latest_runs = self.runs[-n:] if len(self.runs) >= n else self.runs
        
        print("\n" + "="*80)
        print(f"                    LATEST {len(latest_runs)} RUNS")
        print("="*80 + "\n")
        
        headers = ['Run #', 'Date', 'Ticker', 'Folds', 'Accuracy', 'AUC', 'Sharpe', 'Max DD']
        rows = []
        
        for run in latest_runs:
            rows.append([
                run['run_number'],
                run['date'],
                run['ticker'],
                run['fold'],
                f"{run['metrics']['accuracy']:.4f}",
                f"{run['metrics']['auc_roc']:.4f}",
                f"{run['financial_metrics']['sharpe_ratio']:.4f}",
                f"{run['financial_metrics']['max_drawdown']:.4f}"
            ])
        
        print(tabulate(rows, headers=headers, tablefmt='grid'))
        print()
    
    def print_feature_usage(self):
        """Analyze which features appear most in runs."""
        if not self.runs:
            print("No runs logged yet.")
            return
        
        feature_counts = {}
        
        for run in self.runs:
            for feature in run['features']['names']:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
        
        sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
        
        print("\n" + "="*80)
        print("                    FEATURE USAGE FREQUENCY")
        print("="*80 + "\n")
        
        headers = ['Feature', 'Used in Runs', 'Frequency (%)']
        rows = []
        
        total_runs = len(self.runs)
        for feature, count in sorted_features[:20]:  # Top 20 features
            percentage = (count / total_runs) * 100
            rows.append([feature, count, f"{percentage:.1f}%"])
        
        print(tabulate(rows, headers=headers, tablefmt='grid'))
        print()
    
    def print_csv_snippet(self, rows=10):
        """Print a snippet of the CSV summary."""
        if self.df is None:
            print("CSV file not found. Generate it with run_logger.export_csv()")
            return
        
        print("\n" + "="*80)
        print(f"                    CSV SUMMARY (First {rows} rows)")
        print("="*80 + "\n")
        
        print(self.df.head(rows).to_string())
        print(f"\n... Total {len(self.df)} rows ...\n")
    
    def get_best_run(self, metric='auc_roc'):
        """Get the best run by a specific metric."""
        if not self.df or metric not in self.df.columns:
            return None
        
        best_idx = self.df[metric].idxmax()
        return self.df.iloc[best_idx]
    
    def print_all_runs_table(self):
        """Print all runs in a table format."""
        if self.df is None:
            print("CSV file not found.")
            return
        
        print("\n" + "="*80)
        print("                    ALL RUNS - DETAILED TABLE")
        print("="*80 + "\n")
        
        # Select key columns
        key_cols = ['run_number', 'ticker', 'fold', 'accuracy', 'auc_roc', 'auc_pr', 
                   'sharpe_ratio', 'max_drawdown', 'buy_signals', 'sell_signals']
        
        available_cols = [col for col in key_cols if col in self.df.columns]
        
        print(self.df[available_cols].to_string())
        print()

def main():
    """Main viewer interface."""
    print("\n🎯 ForecastTRADE - Results Viewer")
    print("="*80)
    
    viewer = ResultsViewer()
    
    # Display information
    viewer.print_summary()
    viewer.print_latest_runs(n=5)
    viewer.print_feature_usage()
    viewer.print_csv_snippet(rows=5)
    
    print("\n📁 Output Files:")
    print("   - out/runs_log.json           : Complete JSON log of all runs")
    print("   - out/runs_summary.csv        : CSV export for analysis")
    print("   - out/*.png                   : Visualization images for each run")
    print("\n✅ Results viewer complete. Check 'out/' directory for all outputs.\n")

if __name__ == '__main__':
    main()
