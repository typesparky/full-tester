#!/usr/bin/env python3
"""
RUN ANALYSIS

Run correlation and statistical analysis on signals vs assets.
"""

import sys
import os
import yaml
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_yaml_config(config_path: str) -> dict:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_asset_data(asset_config: dict) -> dict:
    """Load asset data from configuration (supports local CSV and yfinance)"""
    assets_data = {}

    default_source = asset_config.get('default_data_source', 'local_csv')

    for asset in asset_config.get('assets', []):
        asset_name = asset['name']
        data_source = asset.get('data_source', default_source)

        try:
            if data_source == 'local_csv':
                data_file = asset['data_file']
                if Path(data_file).exists():
                    try:
                        # Handle BOM encoding and parse CSV properly
                        df = pd.read_csv(data_file, encoding='utf-8-sig', thousands=',', decimal='.')

                        # Try multiple date formats
                        date_formats = [
                            asset_config.get('date_column_format', '%m/%d/%Y'),
                            '%m/%d/%Y',  # MM/DD/YYYY
                            '%b %d, %Y'  # Mon DD, YYYY
                        ]

                        for fmt in date_formats:
                            try:
                                df['Date'] = pd.to_datetime(df['Date'], format=fmt)
                                break
                            except:
                                continue
                        else:
                            # Fallback to automatic parsing
                            df['Date'] = pd.to_datetime(df['Date'])

                        # Set date as index and get price column
                        df = df.set_index('Date')
                        price_col = asset.get('price_column', 'Price')

                        if price_col in df.columns:
                            # Convert price to numeric (remove commas and convert to float)
                            df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
                            assets_data[asset_name] = df[price_col]
                            print(f"✅ Loaded {asset_name} from local CSV: {len(df)} days from {df.index[0].date()} to {df.index[-1].date()}")
                        else:
                            print(f"❌ Price column '{price_col}' not found in {data_file}")
                            print(f"   Available columns: {list(df.columns)}")

                    except Exception as e:
                        print(f"❌ Error parsing {asset_name} CSV: {e}")
                        print(f"   File: {data_file}")
                        # Let's try to read the first few lines for debugging
                        try:
                            with open(data_file, 'r', encoding='utf-8-sig') as f:
                                lines = [next(f).strip() for _ in range(3)]
                                print(f"   First lines: {lines[:2]}")
                        except:
                            pass
                else:
                    print(f"❌ Asset data file not found: {data_file}")

            elif data_source == 'yfinance':
                import yfinance as yf

                ticker = asset['ticker']
                period = asset.get('period', '2y')
                print(f"📡 Fetching {asset_name} ({ticker}) from yfinance (period: {period})...")

                # Fetch data with specified period
                ticker_data = yf.Ticker(ticker)
                df = ticker_data.history(period=period, auto_adjust=True)

                if not df.empty:
                    assets_data[asset_name] = df[asset.get('price_column', 'Close')]
                    print(f"✅ Loaded {asset_name}: {len(df)} days from {df.index[0].date()} to {df.index[-1].date()}")
                else:
                    print(f"❌ Failed to fetch {ticker} from yfinance")

        except Exception as e:
            print(f"❌ Error loading {asset_name}: {e}")

    return assets_data

def run_correlation_analysis(analysis_config: dict, signal_df: pd.DataFrame, assets_data: dict):
    """
    Run basic correlation analysis between signal and assets.
    """
    print(f"\n🔍 Running {analysis_config.get('analysis_name', 'Correlation Analysis')}")
    print("="*60)

    lags_to_test = analysis_config.get('parameters', {}).get('lags_to_test', [0])
    rolling_window = analysis_config.get('parameters', {}).get('rolling_window', 30)

    if signal_df.empty:
        print("❌ No signal data available")
        return

    # Prepare signal data
    signal_df['date'] = pd.to_datetime(signal_df['date'])
    signal_df = signal_df.set_index('date')

    results = {}

    for asset_name, asset_series in assets_data.items():
        print(f"\n📈 Analyzing {asset_name} vs Geopolitical Uncertainty Signal")
        print("-"*50)

        # Calculate returns
        asset_returns = asset_series.pct_change().dropna()
        signal_changes = signal_df['signal'].diff().dropna()

        # Align time periods
        common_dates = asset_returns.index.intersection(signal_changes.index)
        if len(common_dates) == 0:
            print(f"❌ No overlapping dates for {asset_name}")
            continue

        asset_sample = asset_returns.loc[common_dates]
        signal_sample = signal_changes.loc[common_dates]

        print(f"Common dates: {len(common_dates)}")
        print(f"Asset returns: {asset_sample.mean():.4f}")
        print(f"Asset volatility: {asset_sample.std():.4f}")
        print(f"Signal changes: {len(signal_sample)} observations")

        # Calculate correlations at different lags
        correlations = {}
        for lag in lags_to_test:
            if lag == 0:
                corr = asset_sample.corr(signal_sample)
            else:
                shifted_signal = signal_sample.shift(lag).dropna()
                aligned_asset = asset_sample.loc[shifted_signal.index]
                corr = aligned_asset.corr(shifted_signal)

            correlations[f'lag_{lag}'] = corr
            print(f"Correlation lag {lag}: {corr:.4f}")
        # Find best lag
        best_lag = max(correlations.keys(), key=lambda k: abs(correlations[k]))
        best_corr = correlations[best_lag]

        results[asset_name] = {
            'correlations': correlations,
            'best_lag': best_lag,
            'best_corr': best_corr,
            'sample_size': len(common_dates)
        }

        # Calculate rolling correlation
        if rolling_window and len(signal_sample) > rolling_window:
            aligned_data = pd.concat([asset_sample, signal_sample], axis=1).dropna()
            if len(aligned_data) > rolling_window:
                rolling_corr = aligned_data.iloc[:, 0].rolling(rolling_window).corr(aligned_data.iloc[:, 1])
                print(f"Average rolling correlation: {rolling_corr.mean():.4f}")
                print(f"Rolling correlation above 0: {((rolling_corr > 0).sum()) / len(rolling_corr):.1%}")

    # Save results
    output_dir = Path(analysis_config.get('output_dir', 'analysis_results'))
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / 'correlation_results.yml', 'w') as f:
        yaml.dump(results, f, default_flow_style=False)

    print(f"\n💾 Results saved to {output_dir}/correlation_results.yml")
    print("🎉 Correlation analysis completed!")

def main():
    parser = argparse.ArgumentParser(description="Run signal analysis")
    parser.add_argument('--analysis-config', required=True,
                       help='Path to analysis configuration YAML file')

    args = parser.parse_args()

    # Load analysis configuration
    analysis_config = load_yaml_config(args.analysis_config)

    # Load asset configuration
    asset_config = load_yaml_config(analysis_config['asset_config'])

    # Load signal data
    signal_file = Path(analysis_config['signal_file'])
    if not signal_file.exists():
        print(f"❌ Signal file not found: {signal_file}")
        return

    signal_df = pd.read_csv(signal_file)

    # Load asset data
    assets_data = load_asset_data(asset_config)

    if not assets_data:
        print("❌ No asset data loaded")
        return

    # Run analysis based on type
    analysis_type = analysis_config.get('analyzer_type', 'correlation').lower()

    if analysis_type == 'correlation':
        run_correlation_analysis(analysis_config, signal_df, assets_data)
    else:
        print(f"❌ Unknown analysis type: {analysis_type}")

if __name__ == "__main__":
    main()
