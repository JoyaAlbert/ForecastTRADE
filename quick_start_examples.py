#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Start Examples - Como usar los logs y visualizaciones
===========================================================

Ejemplos prácticos de cómo acceder y analizar los datos generados.
"""

import json
import pandas as pd
import os
from pathlib import Path

def example_1_read_json():
    """Ejemplo 1: Leer el JSON log directamente."""
    print("\n" + "="*80)
    print("EJEMPLO 1: Leer JSON Log")
    print("="*80)
    
    log_file = 'out/runs_log.json'
    
    if not os.path.exists(log_file):
        print("⚠️  El archivo out/runs_log.json no existe aún.")
        print("   Ejecuta: python3 main.py")
        return
    
    with open(log_file, 'r') as f:
        data = json.load(f)
    
    # Metadata
    print("\n📊 METADATA:")
    print(f"   Total Runs: {data['metadata']['total_runs']}")
    print(f"   Best AUC: {data['metadata']['best_auc']:.4f}")
    print(f"   Best Accuracy: {data['metadata']['best_accuracy']:.4f}")
    print(f"   Best Sharpe: {data['metadata']['best_sharpe']:.4f}")
    
    # Último run
    if data['runs']:
        latest = data['runs'][-1]
        print(f"\n🔄 ÚLTIMO RUN (#{latest['run_number']}):")
        print(f"   Fecha: {latest['date']}")
        print(f"   Ticker: {latest['ticker']}")
        print(f"   Folds: {latest['fold']}")
        print(f"   Accuracy: {latest['metrics']['accuracy']:.4f}")
        print(f"   AUC: {latest['metrics']['auc_roc']:.4f}")
        print(f"   Sharpe: {latest['financial_metrics']['sharpe_ratio']:.4f}")
        print(f"   Features: {latest['features']['count']}")

def example_2_read_csv():
    """Ejemplo 2: Leer CSV con Pandas."""
    print("\n" + "="*80)
    print("EJEMPLO 2: Leer CSV y Análisis Rápido")
    print("="*80)
    
    csv_file = 'out/runs_summary.csv'
    
    if not os.path.exists(csv_file):
        print("⚠️  El archivo out/runs_summary.csv no existe aún.")
        print("   Ejecuta: python3 main.py")
        return
    
    df = pd.read_csv(csv_file)
    
    print(f"\n📈 Estadísticas Rápidas:")
    print(f"   Total runs: {len(df)}")
    print(f"\n   Accuracy:")
    print(f"      Min: {df['accuracy'].min():.4f}")
    print(f"      Max: {df['accuracy'].max():.4f}")
    print(f"      Mean: {df['accuracy'].mean():.4f}")
    print(f"      Std: {df['accuracy'].std():.4f}")
    
    print(f"\n   AUC ROC:")
    print(f"      Min: {df['auc_roc'].min():.4f}")
    print(f"      Max: {df['auc_roc'].max():.4f}")
    print(f"      Mean: {df['auc_roc'].mean():.4f}")
    
    print(f"\n   Sharpe Ratio:")
    print(f"      Min: {df['sharpe_ratio'].min():.4f}")
    print(f"      Max: {df['sharpe_ratio'].max():.4f}")
    print(f"      Mean: {df['sharpe_ratio'].mean():.4f}")
    
    # Best run
    best_idx = df['auc_roc'].idxmax()
    print(f"\n🏆 BEST RUN (by AUC):")
    print(f"   Run: {int(df.loc[best_idx, 'run_number'])}")
    print(f"   Accuracy: {df.loc[best_idx, 'accuracy']:.4f}")
    print(f"   AUC: {df.loc[best_idx, 'auc_roc']:.4f}")
    print(f"   Sharpe: {df.loc[best_idx, 'sharpe_ratio']:.4f}")
    print(f"   Buy Signals: {int(df.loc[best_idx, 'buy_signals'])}")

def example_3_compare_runs():
    """Ejemplo 3: Comparar dos grupos de runs."""
    print("\n" + "="*80)
    print("EJEMPLO 3: Comparar Primeros vs Últimos Runs")
    print("="*80)
    
    csv_file = 'out/runs_summary.csv'
    
    if not os.path.exists(csv_file):
        print("⚠️  El archivo out/runs_summary.csv no existe aún.")
        return
    
    df = pd.read_csv(csv_file)
    
    if len(df) < 2:
        print("⚠️  Se necesitan al menos 2 runs para comparar.")
        return
    
    # Divide en early y recent
    split_idx = len(df) // 2
    early = df.iloc[:split_idx]
    recent = df.iloc[split_idx:]
    
    print(f"\n📊 EARLY RUNS ({len(early)} runs):")
    print(f"   Accuracy: {early['accuracy'].mean():.4f} ± {early['accuracy'].std():.4f}")
    print(f"   AUC: {early['auc_roc'].mean():.4f} ± {early['auc_roc'].std():.4f}")
    print(f"   Sharpe: {early['sharpe_ratio'].mean():.4f} ± {early['sharpe_ratio'].std():.4f}")
    
    print(f"\n📈 RECENT RUNS ({len(recent)} runs):")
    print(f"   Accuracy: {recent['accuracy'].mean():.4f} ± {recent['accuracy'].std():.4f}")
    print(f"   AUC: {recent['auc_roc'].mean():.4f} ± {recent['auc_roc'].std():.4f}")
    print(f"   Sharpe: {recent['sharpe_ratio'].mean():.4f} ± {recent['sharpe_ratio'].std():.4f}")
    
    # Improvement
    acc_improve = (recent['accuracy'].mean() - early['accuracy'].mean()) / early['accuracy'].mean() * 100
    auc_improve = (recent['auc_roc'].mean() - early['auc_roc'].mean()) / early['auc_roc'].mean() * 100
    sharpe_improve = (recent['sharpe_ratio'].mean() - early['sharpe_ratio'].mean()) / early['sharpe_ratio'].mean() * 100
    
    print(f"\n🎯 IMPROVEMENT (Recent vs Early):")
    print(f"   Accuracy: {acc_improve:+.2f}%")
    print(f"   AUC: {auc_improve:+.2f}%")
    print(f"   Sharpe: {sharpe_improve:+.2f}%")

def example_4_visualizations():
    """Ejemplo 4: Listar todas las visualizaciones generadas."""
    print("\n" + "="*80)
    print("EJEMPLO 4: Archivos de Visualización Generados")
    print("="*80)
    
    out_dir = Path('out')
    
    if not out_dir.exists():
        print("⚠️  El directorio out/ no existe aún.")
        return
    
    # Buscar PNGs
    hybrid_plots = list(out_dir.glob('lstm_xgboost_hybrid_*.png'))
    forecast_plots = list(out_dir.glob('future_forecast_*.png'))
    
    print(f"\n🖼️  HYBRID PLOTS (LSTM + XGBoost):")
    print(f"   Total: {len(hybrid_plots)}")
    if hybrid_plots:
        for i, f in enumerate(hybrid_plots[:3]):
            print(f"   {i+1}. {f.name}")
        if len(hybrid_plots) > 3:
            print(f"   ... y {len(hybrid_plots) - 3} más")
    
    print(f"\n📈 FORECAST PLOTS:")
    print(f"   Total: {len(forecast_plots)}")
    if forecast_plots:
        for i, f in enumerate(forecast_plots[:3]):
            print(f"   {i+1}. {f.name}")
        if len(forecast_plots) > 3:
            print(f"   ... y {len(forecast_plots) - 3} más")

def example_5_feature_analysis():
    """Ejemplo 5: Análisis de features usadas."""
    print("\n" + "="*80)
    print("EJEMPLO 5: Análisis de Features Utilizadas")
    print("="*80)
    
    log_file = 'out/runs_log.json'
    
    if not os.path.exists(log_file):
        print("⚠️  El archivo out/runs_log.json no existe aún.")
        return
    
    with open(log_file, 'r') as f:
        data = json.load(f)
    
    if not data['runs']:
        print("⚠️  No hay runs en el log.")
        return
    
    # Contar frecuencia de features
    feature_freq = {}
    for run in data['runs']:
        for feature in run['features']['names']:
            feature_freq[feature] = feature_freq.get(feature, 0) + 1
    
    # Top 10
    top_features = sorted(feature_freq.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print(f"\n⭐ TOP 10 FEATURES (apariciones en runs):")
    for i, (feature, count) in enumerate(top_features, 1):
        percentage = (count / len(data['runs'])) * 100
        print(f"   {i:2d}. {feature:30s} {count:2d}x ({percentage:5.1f}%)")
    
    # Estadísticas
    print(f"\n📊 FEATURE STATISTICS:")
    print(f"   Features únicos: {len(feature_freq)}")
    print(f"   LSTM latent features: {sum(1 for f in feature_freq if 'lstm_latent' in f)}")
    print(f"   Technical features: {sum(1 for f in feature_freq if 'lstm' not in f)}")

def example_6_trading_signals():
    """Ejemplo 6: Análisis de señales de trading."""
    print("\n" + "="*80)
    print("EJEMPLO 6: Análisis de Señales de Trading")
    print("="*80)
    
    csv_file = 'out/runs_summary.csv'
    
    if not os.path.exists(csv_file):
        print("⚠️  El archivo out/runs_summary.csv no existe aún.")
        return
    
    df = pd.read_csv(csv_file)
    
    print(f"\n🚀 SEÑALES DE COMPRA (Buy Signals):")
    print(f"   Total: {int(df['buy_signals'].sum())}")
    print(f"   Por run: {df['buy_signals'].mean():.1f} ± {df['buy_signals'].std():.1f}")
    print(f"   Rango: {int(df['buy_signals'].min())} - {int(df['buy_signals'].max())}")
    
    print(f"\n🔴 SEÑALES DE VENTA (Sell Signals):")
    print(f"   Total: {int(df['sell_signals'].sum())}")
    print(f"   Por run: {df['sell_signals'].mean():.1f} ± {df['sell_signals'].std():.1f}")
    print(f"   Rango: {int(df['sell_signals'].min())} - {int(df['sell_signals'].max())}")
    
    print(f"\n⚖️  RATIO DE SEÑALES (Buy/Sell):")
    ratio = (df['buy_signals'].sum() + 0.1) / (df['sell_signals'].sum() + 0.1)
    print(f"   Ratio: {ratio:.2f} (>1 = bullish, <1 = bearish)")

def main():
    """Ejecutar ejemplos."""
    print("\n" + "="*100)
    print(" " * 25 + "🚀 QUICK START EXAMPLES")
    print("="*100)
    
    examples = [
        ("Leer JSON Log", example_1_read_json),
        ("Leer CSV y Análisis", example_2_read_csv),
        ("Comparar Primeros vs Últimos", example_3_compare_runs),
        ("Listar Visualizaciones", example_4_visualizations),
        ("Análisis de Features", example_5_feature_analysis),
        ("Análisis de Señales de Trading", example_6_trading_signals),
    ]
    
    print("\n📋 Ejemplos disponibles:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"   {i}. {name}")
    
    print("\n" + "-"*100 + "\n")
    
    # Ejecutar todos los ejemplos
    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"\n❌ Error en {name}: {e}")
    
    print("\n" + "="*100)
    print("✅ Ejemplos completados. Revisa los outputs arriba.\n")
    print("💡 TIPS:")
    print("   - Ejecuta: python3 main.py  (para generar datos)")
    print("   - Luego: python3 view_results.py  (para ver resumen bonito)")
    print("   - Los PNGs están en: out/  (abrirlos con visor de imágenes)")
    print("="*100 + "\n")

if __name__ == '__main__':
    main()
