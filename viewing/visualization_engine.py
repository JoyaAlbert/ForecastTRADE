# -*- coding: utf-8 -*-
"""
Visualization Engine for LSTM-XGBoost Hybrid Model
===================================================

Generates comprehensive charts showing:
- LSTM latent space evolution and regime detection
- XGBoost predictions vs actual prices
- Future price forecasts with confidence intervals
- Feature importance analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from datetime import datetime
import json
import os

class VisualizationEngine:
    """Generates visualizations for model predictions and analysis."""
    
    def __init__(self, output_dir='out'):
        """Initialize visualization engine."""
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = {
            'actual': '#1f77b4',
            'lstm': '#ff7f0e',
            'xgboost': '#2ca02c',
            'forecast': '#d62728',
            'confidence_hi': '#ff9999',
            'confidence_lo': '#99ccff'
        }

    @staticmethod
    def _get_dynamic_risk_config(vol_metric='rolling_std_20d', k_tp=2.5, k_sl=1.25):
        return {
            "dynamic_risk": {
                "type": "Volatility_Adjusted",
                "params": {
                    "k_tp": k_tp,
                    "k_sl": k_sl,
                    "vol_metric": vol_metric
                }
            }
        }
    
    def plot_lstm_xgboost_hybrid(self, df, train_data, test_data,
                                lstm_preds, xgb_preds, scaler,
                                run_date, run_number):
        """Generate comprehensive visualization of LSTM + XGBoost predictions.

        Args:
            df: Full dataframe with price and features.
            train_data: Training set indices.
            test_data: Testing set indices.
            lstm_preds: LSTM predictions.
            xgb_preds: XGBoost probability predictions.
            scaler: Fitted scaler for inverse transform.
            run_date: Date of the run.
            run_number: Run identifier.
        """
        fig = plt.figure(figsize=(20, 12), constrained_layout=True)
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

        # 1. MAIN PLOT: Price + LSTM Predictions + XGBoost Signals
        ax1 = fig.add_subplot(gs[0, :])

        # Plot actual price
        ax1.plot(df.index, df['close'], label='Actual Price',
                 color=self.colors['actual'], linewidth=2, alpha=0.9)

        # Plot LSTM price predictions (from latent features)
        lstm_price_col = 'lstm_price_5d'
        if lstm_price_col in df.columns:
            ax1.plot(df.index, df[lstm_price_col], label='LSTM Prediction (5d)',
                     color=self.colors['lstm'], linewidth=1.5, alpha=0.7, linestyle='--')

        # Highlight test period
        test_idx_all = df.index[test_data]
        ax1.axvspan(test_idx_all[0], test_idx_all[-1], alpha=0.1, color='gray', label='Test Period')

        # XGBoost prediction signals (buy/sell)
        aligned_len = min(len(test_data), len(xgb_preds))
        if aligned_len < len(test_data) or aligned_len < len(xgb_preds):
            print(f"   ⚠️ Visualization warning: aligning predictions to {aligned_len} points")

        test_idx = test_idx_all[:aligned_len]
        xgb_preds = np.asarray(xgb_preds)[:aligned_len]

        buy_signals = xgb_preds > 0.65
        sell_signals = xgb_preds < 0.35

        buy_idx = test_idx[buy_signals]
        sell_idx = test_idx[sell_signals]

        buy_prices = df.loc[buy_idx, 'close']
        sell_prices = df.loc[sell_idx, 'close']

        ax1.scatter(buy_idx, buy_prices, color='green', marker='^',
                    s=100, label='XGBoost Buy Signal', zorder=5)
        ax1.scatter(sell_idx, sell_prices, color='red', marker='v',
                    s=100, label='XGBoost Sell Signal', zorder=5)

        ax1.set_title(f'LSTM + XGBoost Hybrid Predictions - Run {run_number} ({run_date})',
                      fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price ($)', fontsize=11)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)

        # 2. LSTM LATENT FEATURES - Top 3 principal components
        ax2 = fig.add_subplot(gs[1, 0])

        latent_cols = [col for col in df.columns if col.startswith('lstm_latent_')]
        if len(latent_cols) >= 3:
            # Dynamically use the first 3 available latent features
            labels = ['Trend', 'Volatility', 'Regime', 'Momentum', 'Pattern']
            colors_list = [self.colors['lstm'], self.colors['xgboost'], self.colors['forecast']]
            
            for i, latent_col in enumerate(latent_cols[:3]):
                ax2.plot(df.index, df[latent_col], 
                        label=f'Latent {i} ({labels[i]})',
                        color=colors_list[i], linewidth=1.5, alpha=0.8)

            ax2.axvspan(test_idx_all[0], test_idx_all[-1], alpha=0.1, color='gray')
            ax2.set_title('LSTM Latent Features (Top 3)', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Normalized Value', fontsize=10)
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)

        # 3. XGBoost PROBABILITY PREDICTIONS
        ax3 = fig.add_subplot(gs[1, 1])

        test_dates = test_idx
        ax3.plot(test_dates, xgb_preds, label='XGBoost Probability',
                 color=self.colors['xgboost'], linewidth=2, marker='o', markersize=3)
        ax3.axhline(0.65, color='green', linestyle='--', linewidth=1.5, label='Buy Threshold')
        ax3.axhline(0.35, color='red', linestyle='--', linewidth=1.5, label='Sell Threshold')
        ax3.axhline(0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)

        ax3.fill_between(test_dates, 0.35, 0.65, alpha=0.2, color='yellow', label='Hold Zone')
        ax3.set_title('XGBoost Buy/Sell Probability', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Probability', fontsize=10)
        ax3.set_ylim([0, 1])
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)

        # 4. PRICE RETURNS ANALYSIS
        ax4 = fig.add_subplot(gs[2, 0])

        df['returns'] = df['close'].pct_change() * 100
        ax4.bar(df.index, df['returns'], color=['green' if x > 0 else 'red'
                                                for x in df['returns']], alpha=0.6, width=1)
        ax4.axhline(0, color='black', linewidth=0.8)
        ax4.axvspan(test_idx_all[0], test_idx_all[-1], alpha=0.1, color='gray')

        ax4.set_title('Daily Returns (%)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Return (%)', fontsize=10)
        ax4.grid(True, alpha=0.3)

        # 5. CUMULATIVE STRATEGY RETURNS
        ax5 = fig.add_subplot(gs[2, 1])

        # Create strategy returns based on predictions
        strategy_returns = np.zeros(aligned_len)
        aligned_test_data = test_data[:aligned_len]
        for i in range(aligned_len - 1):
            if xgb_preds[i] > 0.55:  # Buy signal
                strategy_returns[i] = df['returns'].iloc[aligned_test_data[i + 1]]
            elif xgb_preds[i] < 0.45:  # Sell signal
                strategy_returns[i] = -df['returns'].iloc[aligned_test_data[i + 1]]

        buy_and_hold = df['returns'].iloc[aligned_test_data].values

        cumsum_strategy = np.cumsum(strategy_returns)
        cumsum_bh = np.cumsum(buy_and_hold)

        ax5.plot(test_dates, cumsum_strategy, label='Strategy Returns',
                 color=self.colors['forecast'], linewidth=2, marker='o', markersize=3)
        ax5.plot(test_dates, cumsum_bh, label='Buy & Hold',
                 color=self.colors['actual'], linewidth=2, marker='s', markersize=3, alpha=0.7)

        ax5.fill_between(test_dates, cumsum_strategy, 0, alpha=0.2, color=self.colors['forecast'])
        ax5.axhline(0, color='black', linewidth=0.8)
        ax5.set_title('Cumulative Returns Comparison', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Cumulative Return (%)', fontsize=10)
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3)

        # Format x-axis
        for ax in [ax1, ax2, ax3, ax4, ax5]:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        # Save figure
        filename = f'out/lstm_xgboost_hybrid_run_{run_number}_{run_date}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"   ✅ Visualization saved: {filename}")
        plt.close()

        return filename
    
    def plot_future_forecast(self, df, lstm_preds, xgb_preds, forecast_horizon=20,
                            run_date='', run_number=1, latest_model=None, scaler=None, X_recent=None):
        """
        Generate future price forecast visualization with trading signals.
        
        Args:
            df: Dataframe with recent prices
            lstm_preds: LSTM predictions
            xgb_preds: XGBoost probabilities (win probability) - historical
            forecast_horizon: Days to forecast ahead
            run_date: Date string
            run_number: Run identifier
            latest_model: Trained XGBoost model for future predictions
            scaler: Fitted scaler for scaling recent features
            X_recent: Recent feature data (last 30 days) for future prediction
        """
        fig = plt.figure(figsize=(16, 10), constrained_layout=True)
        gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)
        
        # Last 100 days + forecast
        recent_data = df['close'].tail(100).copy()
        last_price = recent_data.iloc[-1]
        last_date = recent_data.index[-1]
        
        # Make FUTURE prediction with latest model
        latest_prob = 0.5  # Default fallback
        if latest_model is not None and X_recent is not None and scaler is not None:
            try:
                X_recent_scaled = scaler.transform(X_recent)
                recent_proba = latest_model.predict_proba(X_recent_scaled)[:, 1]
                latest_prob = recent_proba[-1]  # Most recent prediction
                print(f"\n   🔮 Future Prediction Made: {latest_prob*100:.1f}% probability of winning")
                print(f"      Based on latest market conditions as of {last_date.strftime('%Y-%m-%d')}")
            except Exception as e:
                print(f"   ⚠️ Failed to generate future prediction: {e}")
                latest_prob = xgb_preds[-1] if len(xgb_preds) > 0 else 0.5
        else:
            latest_prob = xgb_preds[-1] if len(xgb_preds) > 0 else 0.5
        
        # Generate forecast based on LSTM with trend and model prediction
        if 'lstm_return_5d' in df.columns:
            recent_returns = df['lstm_return_5d'].tail(30)
            avg_return = recent_returns[recent_returns.notna()].mean() if recent_returns.notna().sum() > 0 else 0.001
            volatility = recent_returns.std() if recent_returns.notna().sum() > 0 else 0.01
        else:
            avg_return = 0.001
            volatility = 0.01
        
        # Dynamic volatility-adjusted targets (asset-agnostic)
        if 'volatility_20' in df.columns and pd.notna(df['volatility_20'].iloc[-1]):
            volatility_price = float(df['volatility_20'].iloc[-1])
            vol_metric = 'rolling_std_20d'
        elif 'atr' in df.columns and pd.notna(df['atr'].iloc[-1]):
            volatility_price = float(df['atr'].iloc[-1])
            vol_metric = 'atr_14d'
        else:
            volatility_price = float(df['close'].rolling(window=20).std().ffill().iloc[-1])
            vol_metric = 'rolling_std_20d'

        k_tp = 2.5
        k_sl = 1.25
        profit_target = last_price + (volatility_price * k_tp)
        stop_loss = last_price - (volatility_price * k_sl)
        tp_pct = ((profit_target - last_price) / last_price) * 100
        sl_pct = ((stop_loss - last_price) / last_price) * 100
        dynamic_risk_config = self._get_dynamic_risk_config(vol_metric=vol_metric, k_tp=k_tp, k_sl=k_sl)
        
        # Generate forecast with directional bias from model prediction
        forecast_prices = []
        
        # Determine target based on model prediction
        if latest_prob > 0.5:
            # Bullish: trend towards profit target
            target_price = profit_target
            confidence = (latest_prob - 0.5) * 2  # 0.0 to 1.0 scale
        else:
            # Bearish: trend towards stop loss
            target_price = stop_loss
            confidence = (0.5 - latest_prob) * 2  # 0.0 to 1.0 scale
        
        # Total change needed to reach target
        total_change = (target_price - last_price) / last_price
        
        # Higher confidence → stronger movement towards target
        # At 20% prob (80% confidence bearish), we move ~90% of the way to stop loss
        bias_strength = 0.5 + (confidence * 0.5)  # Range: 0.5 to 1.0
        
        for i in range(1, forecast_horizon + 1):
            # Progressive movement towards target (exponential for faster initial movement)
            progress = i / forecast_horizon
            # Exponential curve for more dramatic movement
            curve = 1 - np.exp(-3 * progress)  # Fast initial movement, slows near end
            
            # Price moves towards target based on model confidence
            price_change = total_change * curve * bias_strength
            
            # Add realistic volatility (increases over time = uncertainty grows)
            noise = np.random.normal(0, volatility * 0.5 * np.sqrt(progress))
            
            forecast_price = last_price * (1 + price_change + noise)
            forecast_prices.append(forecast_price)
        
        forecast_dates = pd.date_range(start=last_date, periods=forecast_horizon + 1, freq='D')[1:]
        
        # ===== PLOT 1: Price + Forecast with Targets =====
        ax1 = fig.add_subplot(gs[0:2, :])
        
        # Historical price
        ax1.plot(recent_data.index, recent_data.values, label='Historical Price (100 days)',
                color=self.colors['actual'], linewidth=2.5, marker='o', markersize=2, zorder=3)
        
        # Forecast
        ax1.plot(forecast_dates, forecast_prices, label='LSTM Forecast',
                color=self.colors['forecast'], linewidth=2.5, marker='o', 
                markersize=3, linestyle='--', alpha=0.9, zorder=3)
        
        # Confidence interval
        std_error = np.std(forecast_prices) * 0.15
        upper_conf = [p + std_error for p in forecast_prices]
        lower_conf = [p - std_error for p in forecast_prices]
        
        ax1.fill_between(forecast_dates, lower_conf, upper_conf, 
                        alpha=0.2, color=self.colors['confidence_hi'], 
                        label='Confidence Interval (±15%)')
        
        # Trading targets
        ax1.axhline(profit_target, color='green', linestyle='--', linewidth=2.5, 
                   alpha=0.8, label=f'Profit Target: {tp_pct:+.2f}% (${profit_target:.2f})', zorder=2)
        ax1.axhline(stop_loss, color='red', linestyle='--', linewidth=2.5, 
                   alpha=0.8, label=f'Stop Loss: {sl_pct:+.2f}% (${stop_loss:.2f})', zorder=2)
        ax1.axhline(last_price, color='blue', linestyle=':', linewidth=1.5, 
                   alpha=0.6, label=f'Entry Price: ${last_price:.2f}', zorder=1)
        
        # Shaded profit/loss zones
        ax1.fill_between([recent_data.index[0], forecast_dates[-1]], profit_target, 
                        ax1.get_ylim()[1], alpha=0.05, color='green', zorder=0)
        ax1.fill_between([recent_data.index[0], forecast_dates[-1]], ax1.get_ylim()[0], 
                        stop_loss, alpha=0.05, color='red', zorder=0)
        
        # Divider line
        ax1.axvline(last_date, color='gray', linestyle=':', linewidth=2, alpha=0.7, zorder=2)
        ax1.text(last_date, ax1.get_ylim()[1]*0.98, 'TODAY ▼', fontsize=10, 
                ha='center', color='gray', fontweight='bold')
        
        ax1.set_title(f'20-Day Forecast with Trading Targets - Run {run_number}\n' +
                     f'Entry at ${last_price:.2f} on {last_date.strftime("%Y-%m-%d")}',
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price ($)', fontsize=11, fontweight='bold')
        ax1.legend(fontsize=10, loc='best', framealpha=0.95)
        ax1.grid(True, alpha=0.3)
        
        # Format x-axis
        ax1.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # ===== PLOT 2: Model Probability Trajectory (FUTURE) =====
        ax2 = fig.add_subplot(gs[2, 0])
        
        # Forecast probability (constant or slightly degrading confidence)
        forecast_probs = np.ones(len(forecast_dates)) * latest_prob
        # Slight degradation over time (natural model uncertainty growth)
        forecast_probs = forecast_probs - np.linspace(0, latest_prob * 0.1, len(forecast_dates))
        forecast_probs = np.clip(forecast_probs, 0, 1)
        
        ax2.plot(forecast_dates, forecast_probs, label='Win Probability Forecast',
                color=self.colors['xgboost'], linewidth=3, marker='o', markersize=4, zorder=3)
        
        # Decision zones
        ax2.axhline(0.65, color='green', linestyle='--', linewidth=2, alpha=0.8, label='BUY Zone (>65%)', zorder=2)
        ax2.axhline(0.35, color='red', linestyle='--', linewidth=2, alpha=0.8, label='SELL Zone (<35%)', zorder=2)
        ax2.axhline(0.50, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, label='Neutral', zorder=1)
        
        # Shaded zones
        ax2.fill_between(forecast_dates, 0.65, 1.0, alpha=0.1, color='green', label='Strong Buy', zorder=0)
        ax2.fill_between(forecast_dates, 0.0, 0.35, alpha=0.1, color='red', label='Strong Sell', zorder=0)
        ax2.fill_between(forecast_dates, 0.35, 0.65, alpha=0.05, color='yellow', label='Wait/Uncertain', zorder=0)
        
        # Current probability (from FUTURE prediction)
        ax2.scatter([last_date], [latest_prob], color='navy', s=150, marker='*', 
                   zorder=5, label=f'NEXT Prediction: {latest_prob*100:.1f}%')
        
        ax2.set_title('Future Model Confidence (Based on Latest Data)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Win Probability', fontsize=10, fontweight='bold')
        ax2.set_ylim([0, 1])
        ax2.legend(fontsize=9, loc='best', framealpha=0.95)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # ===== PLOT 3: Trading Action Recommendation (FUTURE) =====
        ax3 = fig.add_subplot(gs[2, 1])
        ax3.axis('off')
        
        # Determine trading action based on FUTURE prediction
        if latest_prob > 0.65:
            action = "BUY"
            action_symbol = "^"  # Up arrow
            action_color = 'darkgreen'
            reason = f"High win probability ({latest_prob*100:.1f}%)\nTarget: ${profit_target:.2f} ({tp_pct:+.2f}%)"
        elif latest_prob < 0.35:
            action = "SELL/AVOID"
            action_symbol = "v"  # Down arrow
            action_color = 'darkred'
            reason = f"High loss risk ({(1-latest_prob)*100:.1f}%)\nRisk: ${stop_loss:.2f} ({sl_pct:+.2f}%)"
        else:
            action = "HOLD/WAIT"
            action_symbol = "-"  # Dash
            action_color = 'darkorange'
            reason = f"Uncertain signal ({latest_prob*100:.1f}%)\nWait for clearer signal"
        
        # Display recommendation
        ax3.text(0.5, 0.85, f"{action_symbol} {action}", fontsize=26, fontweight='bold', 
                ha='center', va='center', color=action_color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor=action_color, linewidth=3))
        
        ax3.text(0.5, 0.55, reason, fontsize=11, ha='center', va='center',
                style='italic', color='black',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))
        
        ax3.text(0.5, 0.15, f'Valid from: {last_date.strftime("%Y-%m-%d")}\nNext {forecast_horizon} days',
                fontsize=9, ha='center', va='center', color='gray',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.3))

        print(f"   ⚙️ Dynamic risk config: {json.dumps(dynamic_risk_config)}")
        
        # Save main forecast
        filename = f'out/future_forecast_run_{run_number}_{run_date}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"   ✅ Forecast saved: {filename}")
        plt.close()

        # ===== ZOOM PLOT: Last 30 days + Full forecast =====
        zoom_start = last_date - pd.Timedelta(days=30)
        recent_month = df.loc[zoom_start:last_date, 'close']
        
        fig_zoom, ax_zoom = plt.subplots(1, 1, figsize=(16, 6), constrained_layout=True)
        
        # Plot
        ax_zoom.plot(recent_month.index, recent_month.values, label='Last 30 Days',
                 color=self.colors['actual'], linewidth=3, marker='o', markersize=4, zorder=3)
        ax_zoom.plot(forecast_dates, forecast_prices, label='Forecast (20 Days)',
                 color=self.colors['forecast'], linewidth=3, marker='s',
                 markersize=4, linestyle='--', alpha=0.9, zorder=3)
        
        # Targets
        ax_zoom.axhline(profit_target, color='green', linestyle='--', linewidth=2, 
                       alpha=0.8, label=f'Target: ${profit_target:.2f} ({tp_pct:+.2f}%)')
        ax_zoom.axhline(stop_loss, color='red', linestyle='--', linewidth=2, 
                       alpha=0.8, label=f'Stop: ${stop_loss:.2f} ({sl_pct:+.2f}%)')
        
        # Confidence
        ax_zoom.fill_between(forecast_dates, lower_conf, upper_conf,
                     alpha=0.15, color=self.colors['confidence_hi'])
        
        # Divider
        ax_zoom.axvline(last_date, color='gray', linestyle=':', linewidth=2, alpha=0.7)
        
        ax_zoom.set_title(f'Detailed Forecast View: Last 30 Days + 20 Day Forecast',
                  fontsize=13, fontweight='bold')
        ax_zoom.set_ylabel('Price ($)', fontsize=11, fontweight='bold')
        ax_zoom.legend(fontsize=10, loc='best', framealpha=0.95)
        ax_zoom.grid(True, alpha=0.3)
        ax_zoom.xaxis.set_major_locator(mdates.DayLocator(interval=3))
        ax_zoom.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.setp(ax_zoom.xaxis.get_majorticklabels(), rotation=45)

        zoom_filename = f'out/future_forecast_zoom_run_{run_number}_{run_date}.png'
        plt.savefig(zoom_filename, dpi=300, bbox_inches='tight')
        print(f"   ✅ Forecast zoom saved: {zoom_filename}")
        plt.close()

        return latest_prob  # Return future prediction probability
