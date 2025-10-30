#!/usr/bin/env python3
"""
TEST MILITARY SIGNAL PERFORMANCE ON SP500, GOLD, AND OIL

Direct testing of military actions signal correlation with financial assets.
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
        df = pd.read_csv('Data/Gold Futures Historical Data.csv', parse_dates=['Date'])
        df = df.rename(columns={'Date': 'date', 'Price': 'price'})
    elif asset_name == 'Oil':
        df = pd.read_csv('Data/Crude Oil WTI Futures Historical Data.csv', parse_dates=['Date'])
        df = df.rename(columns={'Date': 'date', 'Price': 'price'})
    else:
        raise ValueError(f"Unknown asset: {asset_name}")

    df = df.sort_values('date')
    df['returns'] = df['price'].pct_change()
    return df[['date', 'price', 'returns']]

def load_military_signal():
    """Load military actions signal"""
    df = pd.read_csv('sector_signals/military_actions_signal.csv', parse_dates=['date'])
    df = df.rename(columns={'adjusted_signal': 'military_signal'})
    return df[['date', 'military_signal']]

def merge_data(asset_df, signal_df):
    """Merge asset data with signal data"""
    merged = pd.merge(asset_df, signal_df, on='date', how='inner')

    # Handle missing returns
    merged = merged.dropna()

    return merged

def calculate_correlations(merged_df, asset_name, window=90):
    """Calculate rolling correlations between signal and returns"""
    # Rolling correlation
    merged_df[f'military_signal_lag1'] = merged_df['military_signal'].shift(1)
    merged_df[f'military_signal_lag3'] = merged_df['military_signal'].shift(3)
    merged_df[f'military_signal_lag5'] = merged_df['military_signal'].shift(5)

    # Correlations
    current_corr = merged_df['military_signal'].corr(merged_df['returns'])
    lag1_corr = merged_df[f'military_signal_lag1'].corr(merged_df['returns'])
    lag3_corr = merged_df[f'military_signal_lag3'].corr(merged_df['returns'])
    lag5_corr = merged_df[f'military_signal_lag5'].corr(merged_df['returns'])

    return {
        'asset': asset_name,
        'periods': len(merged_df),
        'date_range': f"{merged_df['date'].min():%Y-%m-%d} to {merged_df['date'].max():%Y-%m-%d}",
        'correlation_current': current_corr,
        'correlation_lag1': lag1_corr,
        'correlation_lag3': lag3_corr,
        'correlation_lag5': lag5_corr
    }

def plot_signal_vs_returns(merged_df, asset_name):
    """Create plot of signal vs returns"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # Signal and returns over time
    ax1.plot(merged_df['date'], merged_df['military_signal'], 'b-', label='Military Signal', alpha=0.7)
    ax1.set_ylabel('Military Signal', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_title(f'Military Signal vs {asset_name} Performance')
    ax1.legend(loc='upper left')

    # Create twin axis for returns
    ax1_twin = ax1.twinx()
    ax1_twin.plot(merged_df['date'], merged_df['returns'].rolling(5).mean(), 'r-', label='5-Day Avg Returns', alpha=0.8)
    ax1_twin.set_ylabel('5-Day Avg Returns', color='r')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    ax1_twin.legend(loc='upper right')

    # Scatter plot
    ax2.scatter(merged_df['military_signal'], merged_df['returns'], alpha=0.6, s=20)
    ax2.set_xlabel('Military Signal')
    ax2.set_ylabel(f'{asset_name} Returns')
    ax2.set_title(f'Signal vs Returns Correlation: {merged_df["military_signal"].corr(merged_df["returns"]):.3f}')

    plt.tight_layout()
    plt.savefig(f'analysis_results/military_{asset_name.lower()}_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()

def analyze_signal_effectiveness(merged_df, asset_name):
    """Analyze signal effectiveness during high vs low tension periods"""

    # Define high tension periods (signal > 75th percentile)
    signal_75th = merged_df['military_signal'].quantile(0.75)
    low_75th = merged_df['military_signal'].quantile(0.25)

    high_tension = merged_df[merged_df['military_signal'] > signal_75th]
    low_tension = merged_df[merged_df['military_signal'] < low_75th]

    # Calculate average returns and volatility
    high_return_avg = high_tension['returns'].mean() * 100
    low_return_avg = low_tension['returns'].mean() * 100
    high_volatility = high_tension['returns'].std() * 100
    low_volatility = low_tension['returns'].std() * 100

    return {
        'asset': asset_name,
        'high_tension_periods': len(high_tension),
        'high_tension_avg_return_pct': high_return_avg,
        'high_tension_volatility_pct': high_volatility,
        'low_tension_periods': len(low_tension),
        'low_tension_avg_return_pct': low_return_avg,
        'low_tension_volatility_pct': low_volatility,
        'risk_adjusted_return': high_return_avg / high_volatility if high_volatility > 0 else 0
    }

def run_comprehensive_test():
    """Run comprehensive testing of military signal on multiple assets"""

    print("🚀 TESTING MILITARY ACTIONS SIGNAL ON FINANCIAL ASSETS")
    print("="*60)

    # Test assets
    assets = ['SP500', 'Gold', 'Oil']
    results = []

    # Analysis period
    print("📊 Analysis Period: Military signal span + asset data overlap")

    for asset in assets:
        try:
            print(f"\n{'='*50}")
            print(f"📈 ANALYZING {asset}")
            print(f"{'='*50}")

            # Load data
            asset_df = load_asset_data(asset)
            signal_df = load_military_signal()
            merged_df = merge_data(asset_df, signal_df)

            print(f"📅 Date range: {merged_df['date'].min():%Y-%m-%d} to {merged_df['date'].max():%Y-%m-%d}")
            print(f"📊 Trading days: {len(merged_df)}")

            # Correlation analysis
            corr_results = calculate_correlations(merged_df, asset)
            results.append(corr_results)

            print("\n📈 SIGNAL-RETURN CORRELATIONS:")
            print(f"   Current day: {corr_results['correlation_current']:.3f}")
            print(f"   1-day lagged: {corr_results['correlation_lag1']:.3f}")
            print(f"   3-day lagged: {corr_results['correlation_lag3']:.3f}")
            print(f"   5-day lagged: {corr_results['correlation_lag5']:.3f}")
            # Effectiveness analysis
            effectiveness = analyze_signal_effectiveness(merged_df, asset)
            print("\n🎯 SIGNAL EFFECTIVENESS ANALYSIS:")
            print(f"   High tension periods ({effectiveness['high_tension_periods']} days):")
            print(f"       Avg return: {effectiveness['high_tension_avg_return_pct']:.2f}%")
            print(f"       Volatility: {effectiveness['high_tension_volatility_pct']:.2f}%")
            print(f"   Low tension periods ({effectiveness['low_tension_periods']} days):")
            print(f"       Avg return: {effectiveness['low_tension_avg_return_pct']:.2f}%")
            print(f"       Volatility: {effectiveness['low_tension_volatility_pct']:.2f}%")
            print(f"       Risk-adjusted return: {effectiveness['risk_adjusted_return']:.2f}")
            # Create plots
            plot_signal_vs_returns(merged_df, asset)
            print(f"📊 Chart saved: analysis_results/military_{asset.lower()}_correlation.png")

        except Exception as e:
            print(f"❌ Error analyzing {asset}: {e}")

    # Summary comparison
    print(f"\n{'='*60}")
    print("🎯 MILITARY SIGNAL PERFORMANCE SUMMARY")
    print(f"{'='*60}")

    summary_data = []
    for r in results:
        summary_data.append({
            'Asset': r['asset'],
            'Periods': r['periods'],
            'Current_Corr': ".3f",
            'Lag1_Corr': ".3f",
            'Lag3_Corr': ".3f",
            'Date_Range': r['date_range']
        })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_markdown(index=False))

    print("\n🎖️ KEY INSIGHTS:")
    print("   • Positive correlation = Risk premium during military tension")
    print("   • Negative correlation = Safe-haven behavior (Gold/Oil)")
    print("   • Stronger lag correlations = Predictable market reactions")
    print("   • Charts saved in analysis_results/ folder")

    print(f"\n✅ MILITARY SIGNAL TESTING COMPLETE! 🚀")

if __name__ == "__main__":
    run_comprehensive_test()
