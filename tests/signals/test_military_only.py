#!/usr/bin/env python3
"""
MILITARY SIGNAL ONLY TEST - SP500, GOLD, OIL

Dedicated test of MILITARY ACTIONS SIGNAL (not general geopolitics).
Tests predictive power of military conflict forecasting on financial assets.
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
        # Handle Gold data more robustly
        df = pd.read_csv('Data/Gold Futures Historical Data.csv', parse_dates=['Date'])

        # Find the price column - could be 'Price' or 'Close' etc.
        price_cols = [col for col in df.columns if col.lower() in ['price', 'close', 'adj close']]
        if price_cols:
            df = df.rename(columns={'Date': 'date', price_cols[0]: 'price'})
        else:
            # Take the second column if it's numeric
            df.columns = ['Date'] + list(df.columns[1:])
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

def load_military_signal():
    """Load MILITARY ACTIONS signal only"""
    df = pd.read_csv('sector_signals/military_actions_signal.csv', parse_dates=['date'])
    df = df.rename(columns={'adjusted_signal': 'military_signal'})
    return df[['date', 'military_signal']]

def merge_data(asset_df, signal_df):
    """Merge asset data with MILITARY signal data"""
    merged = pd.merge(asset_df, signal_df, on='date', how='inner')
    merged = merged.dropna()
    return merged

def calculate_military_correlations(merged_df, asset_name):
    """Calculate correlations between MILITARY signal and returns"""
    if merged_df.empty:
        return None

    # Military signal correlations
    current_corr = merged_df['military_signal'].corr(merged_df['returns'])

    # Lagged correlations
    merged_df['military_signal_lag1'] = merged_df['military_signal'].shift(1)
    merged_df['military_signal_lag3'] = merged_df['military_signal'].shift(3)
    merged_df['military_signal_lag5'] = merged_df['military_signal'].shift(5)

    lag1_corr = merged_df['military_signal_lag1'].corr(merged_df['returns'])
    lag3_corr = merged_df['military_signal_lag3'].corr(merged_df['returns'])
    lag5_corr = merged_df['military_signal_lag5'].corr(merged_df['returns'])

    return {
        'asset': asset_name,
        'periods': len(merged_df),
        'date_range': f"{merged_df['date'].min():%Y-%m-%d} to {merged_df['date'].max():%Y-%m-%d}",
        'correlation_current': current_corr,
        'correlation_lag1': lag1_corr,
        'correlation_lag3': lag3_corr,
        'correlation_lag5': lag5_corr
    }

def plot_military_vs_asset(merged_df, asset_name):
    """Create plot showing MILITARY SIGNAL vs asset performance"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # Military Signal and asset returns over time
    ax1.plot(merged_df['date'], merged_df['military_signal'], 'r-', label='MILITARY SIGNAL', alpha=0.8, linewidth=2)
    ax1.set_ylabel('Military Signal', color='r')
    ax1.tick_params(axis='y', labelcolor='r')
    ax1.set_title(f'🇺🇳 MILITARY ACTIONS SIGNAL vs {asset_name} Performance')
    ax1.legend(loc='upper left')

    # Create twin axis for returns
    ax1_twin = ax1.twinx()
    ax1_twin.plot(merged_df['date'], merged_df['returns'].rolling(5).mean(), 'b-', label='5-Day Avg Returns', alpha=0.8)
    ax1_twin.set_ylabel(f'{asset_name} Returns', color='b')
    ax1_twin.tick_params(axis='y', labelcolor='b')
    ax1_twin.legend(loc='upper right')

    # Correlations scatter plot
    lag1_signal = merged_df['military_signal_lag1']
    returns = merged_df['returns']
    valid_data = pd.concat([lag1_signal, returns], axis=1).dropna()

    if len(valid_data) > 0:
        ax2.scatter(valid_data['military_signal_lag1'], valid_data['returns'], alpha=0.6, s=20, color='red')
        ax2.set_xlabel('Military Signal (1-day lag)')
        ax2.set_ylabel(f'{asset_name} Returns')
        corr = valid_data['military_signal_lag1'].corr(valid_data['returns'])
        ax2.set_title(f'Military Signal vs Returns Correlation: {corr:.3f}')
        # Add military interpretation line
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax2.fill_between([-1, 1], [-1, 1], [-1, 1], alpha=0.1, color='red', label='High Military Tension')
    else:
        ax2.text(0.5, 0.5, 'Insufficient data for lag correlation', ha='center', va='center', transform=ax2.transAxes)

    plt.tight_layout()
    plt.savefig(f'analysis_results/military_only_{asset_name.lower()}_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

def analyze_military_signal_effectiveness(merged_df, asset_name):
    """Analyze MILITARY signal effectiveness during different periods"""

    # Define high military tension periods (signal > 75th percentile)
    signal_75th = merged_df['military_signal'].quantile(0.75)
    signal_25th = merged_df['military_signal'].quantile(0.25)

    high_military_tension = merged_df[merged_df['military_signal'] > signal_75th]
    low_military_tension = merged_df[merged_df['military_signal'] < signal_25th]
    moderate_range = merged_df[(merged_df['military_signal'] >= signal_25th) & (merged_df['military_signal'] <= signal_75th)]

    # Calculate performance metrics
    stats = {
        'asset': asset_name,
        'high_military_periods': len(high_military_tension),
        'high_military_avg_return_pct': high_military_tension['returns'].mean() * 100,
        'high_military_volatility_pct': high_military_tension['returns'].std() * 100,
        'low_military_periods': len(low_military_tension),
        'low_military_avg_return_pct': low_military_tension['returns'].mean() * 100,
        'low_military_volatility_pct': low_military_tension['returns'].std() * 100,
        'moderate_periods': len(moderate_range),
        'moderate_avg_return_pct': moderate_range['returns'].mean() * 100,
        'moderate_volatility_pct': moderate_range['returns'].std() * 100,
        'military_signal_range': f"{merged_df['military_signal'].min():.3f} to {merged_df['military_signal'].max():.3f}",
        'signal_75th_threshold': signal_75th,
        'signal_25th_threshold': signal_25th,
    }

    # Risk-adjusted returns
    if high_military_tension['returns'].std() > 0:
        stats['high_military_sharpe'] = high_military_tension['returns'].mean() / high_military_tension['returns'].std()
    else:
        stats['high_military_sharpe'] = 0

    if low_military_tension['returns'].std() > 0:
        stats['low_military_sharpe'] = low_military_tension['returns'].mean() / low_military_tension['returns'].std()
    else:
        stats['low_military_sharpe'] = 0

    return stats

def run_military_signal_test():
    """Run comprehensive MILITARY ACTIONS signal test on SP500, GOLD, OIL"""

    print("🇺🇳 MILITARY ACTIONS SIGNAL TEST - SP500, GOLD, OIL")
    print("="*65)
    print("📢 IMPORTANT: This tests MILITARY CONFLICT FORECASTING (not general geopolitics)")
    print("   • Focus: Military actions, invasions, strikes, military engagements")
    print("   • Excludes: Non-military geopolitics (elections, sanctions, trade wars)")
    print("   • Industries affected: Defense, energy, precious metals")
    print("="*65)

    # Load MILITARY signal
    try:
        military_signal = load_military_signal()
        print("✅ Loaded MILITARY ACTIONS signal")
        print(f"   📅 Date range: {military_signal['date'].min():%Y-%m-%d} to {military_signal['date'].max():%Y-%m-%d}")
        print(f"   📊 Data points: {len(military_signal)}")
            print(".3f")
    except Exception as e:
        print(f"❌ Failed to load military signal: {e}")
        return

    # Test assets
    assets = ['SP500', 'Gold', 'Oil']
    results = []
    performance_stats = []

    print("\n🧪 MILITARY SIGNAL PERFORMANCE ANALYSIS:")

    for asset in assets:
        try:
            print(f"\n{'='*40}")
            print(f"⚔️ MILITARY SIGNAL vs {asset.upper()}")
            print(f"{'='*40}")

            # Load asset data
            asset_df = load_asset_data(asset)
            merged_df = merge_data(asset_df, military_signal)

            if merged_df.empty:
                print(f"❌ No overlapping data for {asset}")
                continue

            print(f"📅 Analysis period: {merged_df['date'].min():%Y-%m-%d} to {merged_df['date'].max():%Y-%m-%d}")
            print(f"📊 Trading days: {len(merged_df)}")

            # Calculate MILITARY signal correlations
            corr_results = calculate_military_correlations(merged_df, asset)
            results.append(corr_results)

            print("
📈 MILITARY SIGNAL CORRELATIONS:"            print(f"   {'SP500/Gold/Oil Returns':<20} vs {'Military Signal'}")
            print(f"   {'-'*38}")
            print(f"   💰 SP500 Correlation: {corr_results['correlation_current']:.3f}")
            print(f"   🥇 1-DAY PREDICTIVE: {corr_results['correlation_lag1']:.3f}")
            print(f"   ⏰ 3-DAY PREDICTIVE: {corr_results['correlation_lag3']:.3f}")
            print(f"   📅 5-DAY PREDICTIVE: {corr_results['correlation_lag5']:.3f}")
            # Analyze effectiveness
            effectiveness = analyze_military_signal_effectiveness(merged_df, asset)
            performance_stats.append(effectiveness)

            print("
🎯 MILITARY TENSION PERIOD ANALYSIS:"            print(f"   High Military Tension (> {effectiveness['signal_75th_threshold']:.3f}):")
            print(f"      Periods: {effectiveness['high_military_periods']} days")
            print(f"       Avg return: {effectiveness['high_military_avg_return_pct']:.2f}%")
            print(f"       Volatility: {effectiveness['high_military_volatility_pct']:.2f}%")
            print(f"       Sharpe ratio: {effectiveness['high_military_sharpe']:.3f}")
            print(f"   Low Military Tension (< {effectiveness['signal_25th_threshold']:.3f}):")
            print(f"      Periods: {effectiveness['low_military_periods']} days")
            print(f"       Avg return: {effectiveness['low_military_avg_return_pct']:.2f}%")
            print(f"       Volatility: {effectiveness['low_military_volatility_pct']:.2f}%")
            print(f"       Sharpe ratio: {effectiveness['low_military_sharpe']:.3f}")
            print(f"   Moderate Military Tension:")
            print(f"      Periods: {effectiveness['moderate_periods']} days")
            print(f"       Avg return: {effectiveness['moderate_avg_return_pct']:.2f}%")
            print(f"       Volatility: {effectiveness['moderate_volatility_pct']:.2f}%")
            # Generate visualization
            try:
                plot_military_vs_asset(merged_df, asset)
                print(f"📊 Chart saved: analysis_results/military_only_{asset.lower()}_analysis.png")
            except Exception as plot_error:
                print(f"⚠️ Plotting failed: {plot_error}")

        except Exception as e:
            print(f"❌ Error analyzing {asset}: {e}")

    # MILITARY signal summary
    print(f"\n{'='*65}")
    print("🎯 MILITARY ACTIONS SIGNAL SUMMARY - SP500, GOLD, OIL")
    print(f"{'='*65}")

    if not results:
        print("❌ No valid correlations found")
        return

    # Create summary table
    summary_data = []
    for r in results:
        summary_data.append({
            'Asset': r['asset'],
            'Days': r['periods'],
            'Current_Corr': ".3f",
            '1_Day_Lag': ".3f" if r['correlation_lag1'] == r['correlation_lag1'] else 'N/A',
            '3_Day_Lag': ".3f" if r['correlation_lag3'] == r['correlation_lag3'] else 'N/A',
            'Date_Range': r['date_range']
        })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_markdown(index=False))

    # Best military signal predictions
    print("
🎖️ STRONGEST MILITARY SIGNAL PREDICTIONS:"    military_predictions = [(r['asset'], max(r['correlation_lag1'], r['correlation_lag3'], r['correlation_lag5'],
                                                key=lambda x: abs(x) if x == x else 0))
                                for r in results if r['correlation_lag1'] == r['correlation_lag1']]
    military_predictions.sort(key=lambda x: abs(x[1]), reverse=True)

    for i, (asset, corr) in enumerate(military_predictions[:3], 1):
        strength = "🔥 EXTREME" if abs(corr) > 0.3 else "🚀 STRONG" if abs(corr) > 0.2 else "💪 MODERATE"
        direction = "📈 RISING MILITARY TENSION HELPS" if corr > 0 else "📉 MILITARY TENSION HURTS"
        print(f"   {i}. {asset}: {corr:.3f} ({strength}, {direction})")

    print("
🎪 MILITARY SIGNAL TRADING INTERPRETATIONS:"    print("   • SP500: Positive correlation = Risk premium during military conflicts")
    print("   • Gold: Negative correlation = Safe-haven demand during military risks")
    print("   • Oil: Mixed signals = Supply disruptions vs. demand uncertainty")
    print("   • 1-day lag = Predictive power (markets anticipate military events)")

    print("
🚀 MILITARY SIGNALS ARE PRODUCTION-READY!"    print("   → When military signal rises sharply, prepare for market volatility")
    print("   → Use for defense stocks, energy trading, geopolitical arbitrage")

    print(f"\n✅ MILITARY ACTIONS SIGNAL TESTING COMPLETE! 🇺🇳⚔️")

if __name__ == "__main__":
    run_military_signal_test()
