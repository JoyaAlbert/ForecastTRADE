# -*- coding: utf-8 -*-
"""
Triple Barrier Method for Label Generation
===========================================

Implements Marcos Lopez de Prado's Triple Barrier Method.
Instead of predicting simple Up/Down, creates 3 labels:
- 1: Hit Profit Target (positive move)
- -1: Hit Stop Loss (negative move)  
- 0: Time-out (no significant move within time horizon)

This filters out market noise and focuses model on high-conviction trades.
"""

import numpy as np
import pandas as pd
from datetime import timedelta


class TripleBarrierLabeler:
    """
    Implements the Triple Barrier Method for creating robust trade labels.
    
    The method places 3 barriers:
    1. Upper barrier: profit target (e.g., +1%)
    2. Lower barrier: stop loss (e.g., -0.5%)
    3. Time barrier: max holding period (e.g., 10 days)
    
    A trade is labeled based on which barrier is hit first.
    """
    
    def __init__(self, profit_target_pct=0.01, stop_loss_pct=0.005, max_holding_days=10):
        """
        Initialize Triple Barrier parameters.
        
        Args:
            profit_target_pct (float): Profit target as % of entry price (default: 1%)
            stop_loss_pct (float): Stop loss as % of entry price (default: 0.5%)
            max_holding_days (int): Maximum holding period in days (default: 10)
        """
        self.profit_target_pct = profit_target_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_holding_days = max_holding_days
    
    def label_data(self, df):
        """
        Create triple barrier labels for all rows in dataframe.
        
        Args:
            df (pd.DataFrame): Must contain 'close' price and a datetime index
            
        Returns:
            pd.Series: Labels (1=profit target, -1=stop loss, 0=timeout)
        """
        labels = pd.Series(0, index=df.index, dtype=int)
        close_prices = df['close'].values
        dates = df.index
        
        for i in range(len(df) - 1):
            # Entry price (current bar)
            entry_price = close_prices[i]
            
            # Barrier prices
            profit_target = entry_price * (1 + self.profit_target_pct)
            stop_loss = entry_price * (1 - self.stop_loss_pct)
            
            # Time barrier
            entry_date = dates[i]
            max_date = entry_date + timedelta(days=self.max_holding_days)
            
            # Find which barrier is hit first
            label = 0  # Default: timeout
            
            for j in range(i + 1, len(df)):
                price = close_prices[j]
                date = dates[j]
                
                # Check time barrier first
                if date > max_date:
                    # Time-out: check which direction we're closer to
                    if price > entry_price:
                        label = 1  # Slight upward bias
                    else:
                        label = -1 if price < entry_price else 0
                    break
                
                # Check profit target
                if price >= profit_target:
                    label = 1
                    break
                
                # Check stop loss
                if price <= stop_loss:
                    label = -1
                    break
            
            labels.iloc[i] = label
        
        return labels
    
    def label_data_vectorized(self, df):
        """
        Faster vectorized implementation of triple barrier labeling.
        
        Args:
            df (pd.DataFrame): Must contain 'close' price and a datetime index
            
        Returns:
            pd.Series: Labels (1=profit target, -1=stop loss, 0=timeout)
        """
        close_prices = df['close'].values
        dates = df.index.values
        labels = np.zeros(len(df), dtype=int)
        
        for i in range(len(df) - 1):
            entry_price = close_prices[i]
            profit_target = entry_price * (1 + self.profit_target_pct)
            stop_loss = entry_price * (1 - self.stop_loss_pct)
            entry_date = dates[i]
            max_date = entry_date + np.timedelta64(self.max_holding_days, 'D')
            
            # Look ahead window
            future_prices = close_prices[i+1:min(i+self.max_holding_days*5, len(df))]
            future_dates = dates[i+1:min(i+self.max_holding_days*5, len(df))]
            
            # Check if profit target is hit
            if np.any(future_prices >= profit_target):
                labels[i] = 1
                continue
            
            # Check if stop loss is hit
            if np.any(future_prices <= stop_loss):
                labels[i] = -1
                continue
            
            # Check if time barrier is hit
            time_barrier_hit = np.where(future_dates > max_date)[0]
            if len(time_barrier_hit) > 0:
                idx = time_barrier_hit[0]
                final_price = future_prices[idx]
                if final_price > entry_price:
                    labels[i] = 1
                elif final_price < entry_price:
                    labels[i] = -1
                # else: labels[i] remains 0 (timeout sideways)
            else:
                # Ran out of data, use last price
                if future_prices[-1] > entry_price:
                    labels[i] = 1
                elif future_prices[-1] < entry_price:
                    labels[i] = -1
        
        return pd.Series(labels, index=df.index, dtype=int)


def apply_triple_barrier_labeling(df, profit_target_pct=0.01, stop_loss_pct=0.005, max_holding_days=10):
    """
    Apply triple barrier method to create robust labels.
    
    Args:
        df (pd.DataFrame): Input dataframe with 'close' price
        profit_target_pct (float): Profit target threshold
        stop_loss_pct (float): Stop loss threshold
        max_holding_days (int): Maximum holding period
        
    Returns:
        pd.Series: Triple barrier labels
    """
    print("\n📊 --- Triple Barrier Method Labeling --- 📊")
    print(f"   Profit Target: +{profit_target_pct*100:.2f}%")
    print(f"   Stop Loss: -{stop_loss_pct*100:.2f}%")
    print(f"   Max Holding: {max_holding_days} days")
    
    labeler = TripleBarrierLabeler(
        profit_target_pct=profit_target_pct,
        stop_loss_pct=stop_loss_pct,
        max_holding_days=max_holding_days
    )
    
    labels = labeler.label_data_vectorized(df)
    
    # Print label distribution
    unique_labels, counts = np.unique(labels.values, return_counts=True)
    print(f"\n   Label Distribution:")
    for label, count in zip(unique_labels, counts):
        label_name = {1: "Profit Target 🎯", -1: "Stop Loss ⛔", 0: "Time-out ⏱️"}
        pct = 100 * count / len(labels)
        print(f"      {label_name[label]}: {count:6d} ({pct:5.1f}%)")
    
    return labels
