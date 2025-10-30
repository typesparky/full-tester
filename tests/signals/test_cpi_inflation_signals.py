#!/usr/bin/env python3
"""
TEST CPI INFLATION SIGNALS

Analysis of CPI inflation signals vs S&P 500 using correlation analysis
(same methodology as military signal tests)
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CORE.unified_analyzer import CorrelationAnalyzer
from CORE.preprocessor import AssetPreprocessor

def load_cpi_inflation_signals():
    """Load CPI inflation signals (equivalent to military signal loading)"""
    signal_file = Path("final_inflation_forecasts_daily.csv")
    if not signal_file.exists():
        raise FileNotFoundError(f"CPI inflation signal file not found: {signal_file}")

    signal_df = pd.read_csv(signal_file)
    signal_df['date'] = pd.to_datetime(signal_df['date'])
    signal_df = signal_df.set_index('date')

    print(f"✅ Loaded CPI inflation signals: {len(signal_df)} daily observations")
    print(f"   Date range: {signal_df.index.min().date()} to {signal_df.index.max().date()}")
    print(".3f"    print(".3f"    return signal_df

def load_asset_data():
    """Load S&P 500 data (equivalent to military asset loading)"""
    asset_file = Path("Data/SP500 Historical Data.csv")
    if not asset_file.exists():
        raise FileNotFoundError(f"S&P 500 data file not found: {asset_file}")

    asset_df = pd.read_csv(asset_file)
    asset_df['date'] = pd.to_datetime(asset_df['Date'])
    asset_df = asset_df[['date', 'Price']].rename(columns={'Price': 'sp500'})
    asset_df['date'] = pd.to_datetime(asset_df['date'])
    asset_df = asset_df.set_index('date')

    print(f"✅ Loaded S&P 500 data: {len(asset_df)} observations")

    return asset_df

def run_cpi_inflation_correlation_analysis():
    """Run correlation analysis on CPI signals vs S&P 500 (same as military)"""

    print("🇺🇸 CPI INFLATION SIGNALS - CORRELATION ANALYSIS")
    print("=" * 60)
    print("Testing same correlation methodology as military signals")
    print()

    try:
        # Load data (equivalent to signal.loading in military tests)
        print("📊 Loading CPI inflation and S&P 500 data...")
        signal_df = load_cpi_inflation_signals()
        asset_df = load_asset_data()

        # Preprocess assets (same preprocessing as military analysis)
        preprocessor = AssetPreprocessor()

        # Process S&P 500 (same as military test)
        sp500_processed = preprocessor.process_asset_data(
            asset_df[['sp500']],
            asset_name='SP500',
            calculate_log_returns=True
        )

        print("✅ Data preprocessing completed")

        # Run correlation analysis (same methodology as military)
        analyzer = CorrelationAnalyzer()

        # Analyze monthly inflation signal
        print("\n📅 ANALYZING MONTHLY INFLATION SIGNAL...")
        monthly_results = analyzer.analyze_correlation(
            signal_df[['monthly_inflation_estimate']],
            sp500_processed,
            lags_to_test=[-5, -3, -1, 0, 1, 3, 5],
            rolling_window=60,
            use_signal_changes=True,
            check_stationarity=True
        )

        print("Monthly inflation correlation results:")
        key_metrics_monthly = analyzer.get_key_correlation_metrics(monthly_results)
        for metric, value in key_metrics_monthly.items():
            print(".3f")

        # Analyze annual inflation signal
        print("\n📊 ANALYZING ANNUAL INFLATION SIGNAL...")
        annual_results = analyzer.analyze_correlation(
            signal_df[['annual_inflation_estimate']],
            sp500_processed,
            lags_to_test=[-5, -3, -1, 0, 1, 3, 5],
            rolling_window=60,
            use_signal_changes=True,
            check_stationarity=True
        )

        print("Annual inflation correlation results:")
        key_metrics_annual = analyzer.get_key_correlation_metrics(annual_results)
        for metric, value in key_metrics_annual.items():
            print(".3f")

        # Comparative analysis
        print("\n📈 COMPARATIVE ANALYSIS:")
        print("-" * 30)
        monthly_corr = monthly_results.get('zero_lag_correlation', 0)
        annual_corr = annual_results.get('zero_lag_correlation', 0)

        print(".3f"        print(".3f"
        # Predictive power comparison
        monthly_predictive = abs(key_metrics_monthly.get('max_lagged_correlation', 0))
        annual_predictive = abs(key_metrics_annual.get('max_lagged_correlation', 0))

        if monthly_predictive > annual_predictive:
            print(f"   ➤ Monthly signal more predictive (|r| = {monthly_predictive:.3f})")
        else:
            print(f"   ➤ Annual signal more predictive (|r| = {annual_predictive:.3f})")

        # Save results to analysis_results (same output as military)
        output_dir = Path("analysis_results")
        output_dir.mkdir(exist_ok=True)

        # Save comparative summary
        summary_results = {
            'signal_type': 'CPI_INFLATION',
            'monthly_zero_lag_corr': monthly_corr,
            'annual_zero_lag_corr': annual_corr,
            'monthly_max_predictive': monthly_predictive,
            'annual_max_predictive': annual_predictive,
            'analysis_date': pd.Timestamp.now().isoformat(),
            'test_period_days': len(signal_df),
            'methodology': 'same_as_military_signals'
        }

        summary_df = pd.DataFrame([summary_results])
        summary_df.to_csv(output_dir / "cpi_inflation_correlation_summary.csv", index=False)

        print("
📁 Results saved to:"        print(f"   analysis_results/cpi_inflation_correlation_summary.csv")

        print("
✓ CPI inflation signal analysis completed!"        print("✓ Same rigorous methodology as military signal testing"
        return True

    except Exception as e:
        print(f"❌ CPI inflation analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_cpi_inflation_correlation_analysis()
    if success:
        print("\n🎉 CPI inflation signals validated using military analysis framework!")
    else:
        print("\n⚠️ CPI inflation analysis encountered issues but was processed.")
    print("📊 Check analysis_results/ for complete correlation analysis.")
