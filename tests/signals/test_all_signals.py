#!/usr/bin/env python3
"""
TEST ALL SIGNALS PERFORMANCE ON SP500, GOLD, AND OIL

Comprehensive testing of all available signals against financial assets.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_asset_data(asset_name):
    """Load asset price data"""
    if asset_name == 'SP500':
        df = pd.read_csv('Data/SP500 Historical Data.csv', parse_dates=['Date'])
        df = df.rename(columns={'Date': 'date', 'Price': 'price'})
    elif asset_name == 'Gold':
        # Fix for Gold data format
        df = pd.read_csv('Data/Gold Futures Historical Data.csv', parse_dates=['Date'])
        # Check column names
        if 'Price' in df.columns:
            df = df.rename(columns={'Date': 'date', 'Price': 'price'})
        elif 'Close' in df.columns:
            df = df.rename(columns={'Date': 'date', 'Close': 'price'})
        else:
            # Update column names based on actual data
            df = df.rename(columns={'Date': 'date', df.columns[1]: 'price'})
    elif asset_name == 'Oil':
        df = pd.read_csv('Data/Crude Oil WTI Futures Historical Data.csv', parse_dates=['Date'])
        if 'Price' in df.columns:
            df = df.rename(columns={'Date': 'date', 'Price': 'price'})
        elif 'Close' in df.columns:
            df = df.rename(columns={'Date': 'date', 'Close': 'price'})
        else:
            df = df.rename(columns={'Date': 'date', df.columns[1]: 'price'})
    else:
        raise ValueError(f"Unknown asset: {asset_name}")

    df = df.sort_values('date')
    df['returns'] = df['price'].pct_change()
    return df[['date', 'price', 'returns']]

def load_signal(signal_path, signal_name):
    """Load signal data"""
    if not Path(signal_path).exists():
        return None

    df = pd.read_csv(signal_path, parse_dates=['date'])
    if 'adjusted_signal' in df.columns:
        df = df.rename(columns={'adjusted_signal': signal_name})
    elif 'signal' in df.columns:
        df = df.rename(columns={'signal': signal_name})
    elif 'master_signal' in df.columns:
        df = df.rename(columns={'master_signal': signal_name})
    else:
        # Find numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1 and 'fwd_return' in numeric_cols[0].lower():
            df = df.rename(columns={numeric_cols[1]: signal_name})
        else:
            df = df.rename(columns={numeric_cols[0]: signal_name})

    # Keep only date and signal columns
    return df[['date', signal_name]]

def merge_data(asset_df, signal_df):
    """Merge asset data with signal data"""
    if signal_df is None or signal_df.empty:
        return pd.DataFrame()

    merged = pd.merge(asset_df, signal_df, on='date', how='inner')
    # Handle missing returns
    merged = merged.dropna()

    return merged

def calculate_correlations(merged_df, signal_name, asset_name):
    """Calculate correlations between signal and returns"""
    if merged_df.empty:
        return None

    # Basic correlations
    current_corr = merged_df[signal_name].corr(merged_df['returns'])

    # Lagged correlations
    merged_df[f'{signal_name}_lag1'] = merged_df[signal_name].shift(1)
    merged_df[f'{signal_name}_lag3'] = merged_df[signal_name].shift(3)
    merged_df[f'{signal_name}_lag5'] = merged_df[signal_name].shift(5)

    lag1_corr = merged_df[f'{signal_name}_lag1'].corr(merged_df['returns'])
    lag3_corr = merged_df[f'{signal_name}_lag3'].corr(merged_df['returns'])
    lag5_corr = merged_df[f'{signal_name}_lag5'].corr(merged_df['returns'])

    return {
        'signal': signal_name,
        'asset': asset_name,
        'periods': len(merged_df),
        'date_range': f"{merged_df['date'].min():%Y-%m-%d} to {merged_df['date'].max():%Y-%m-%d}",
        'correlation_current': current_corr,
        'correlation_lag1': lag1_corr,
        'correlation_lag3': lag3_corr,
        'correlation_lag5': lag5_corr
    }

def run_signals_comparison():
    """Test all available signals against financial assets"""

    print("🚀 COMPREHENSIVE SIGNALS TESTING: ALL VS SP500/OIL")
    print("="*70)

    # Define signals and assets
    signals_config = {
        'Military_Actions': ('sector_signals/military_actions_signal.csv', 'military_signal'),
        'Geopolitical_Uncertainty': ('sector_signals/geopolitical_uncertainty_signal.csv', 'geopolitical_signal'),
        'Master_Geopolitics': ('sector_signals/signal_master_geopolitics.csv', 'geopolitics_master'),
        'Master_Inflation': ('sector_signals/signal_master_inflation.csv', 'inflation_master'),
        'PM_VWEV_Inflation': ('sector_signals/signal_vwev_inflation.csv', 'vwev_inflation'),
        'PM_VWP_China_Trade': ('sector_signals/signal_vwp_china_trade.csv', 'vwp_china_trade'),
        'PM_VWP_Politics': ('sector_signals/signal_vwp_politics.csv', 'vwp_politics')
    }

    assets = ['SP500', 'Oil']  # Focus on SP500 and Oil, Gold had format issue

    # Collect all results
    all_results = []

    print("🎯 TESTING SIGNALS:")

    for signal_display_name, (signal_path, signal_col) in signals_config.items():
        signal_df = load_signal(signal_path, signal_col)

        if signal_df is None or signal_df.empty:
            print(f"   ❌ {signal_display_name}: Signal file not found or empty")
            continue

        success_count = 0

        for asset in assets:
            try:
                asset_df = load_asset_data(asset)
                merged_df = merge_data(asset_df, signal_df)

                if merged_df.empty:
                    print(f"      ⚠️ {signal_display_name} vs {asset}: No overlapping dates")
                    continue

                # Calculate correlations
                corr_result = calculate_correlations(merged_df, signal_col, asset)
                if corr_result:
                    all_results.append(corr_result)
                    success_count += 1

                    print(f"      ✅ {signal_display_name} vs {asset} ({corr_result['periods']} days): "
                          f"Current={corr_result['correlation_current']:.3f}, "
                          f"Lag1={corr_result['correlation_lag1']:.3f}")

            except Exception as e:
                print(f"      ❌ {signal_display_name} vs {asset}: Error - {e}")

        if success_count > 0:
            signal_df_dates = f"{signal_df['date'].min():%Y-%m-%d} to {signal_df['date'].max():%Y-%m-%d}"
            print(f"      📅 {signal_display_name} data span: {signal_df_dates}")

    # Create final comparison
    print(f"\n{'='*70}")
    print("🎯 CROSS-ASSET SIGNAL PERFORMANCE COMPARISON")
    print(f"{'='*70}")

    if not all_results:
        print("❌ No valid signal correlations found")
        return

    # Create summary table
    summary_data = []
    for r in all_results:
        summary_data.append({
            'Signal': r['signal'].replace('_', ' ').title(),
            'Asset': r['asset'],
            'Days': r['periods'],
            'Current': f"{r['correlation_current']:.3f}",
            'Lag1': f"{r['correlation_lag1']:.3f}",
            'Best_Lag': f"{max(r['correlation_current'], r['correlation_lag1'], r['correlation_lag3'], r['correlation_lag5'], key=lambda x: abs(x) if x == x else 0):.3f}",
        })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_markdown(index=False))

    # Rank signals by predictive power (highest absolute lag1 correlation)
    print("\n🎖️ TOP PREDICTIVE SIGNALS (by 1-day lag correlation):")
    signal_ranking = []
    for result in all_results:
        best_lag = max(result['correlation_lag1'], result['correlation_lag3'], result['correlation_lag5'],
                      key=lambda x: abs(x) if x == x else 0)  # Handle NaN
        signal_ranking.append({
            'signal': result['signal'],
            'asset': result['asset'],
            'best_lag_corr': best_lag,
            'strength': abs(best_lag)
        })

    # Sort by strength
    signal_ranking.sort(key=lambda x: x['strength'], reverse=True)

    for i, rank in enumerate(signal_ranking[:5], 1):
        signal_name = rank['signal'].replace('_', ' ').title()
        print(f"   {i}. {signal_name} → {rank['asset']}: {rank['best_lag_corr']:.3f}")

    print("\n🗝️ LEGEND:")
    print("   • Current = Same-day correlation")
    print("   • Lag1 = 1-day predictive correlation (most valuable)")
    print("   • Positive = Risk assets rise with signal")
    print("   • Negative = Safe-haven behavior")

    print(f"\n✅ ALL SIGNALS TESTING COMPLETE! 🚀")

if __name__ == "__main__":
    run_signals_comparison()
