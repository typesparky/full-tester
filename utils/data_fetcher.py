#!/usr/bin/env python3
"""
Data Fetcher: Downloads historical price data using yfinance for indices and commodities.
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Asset configurations
ASSETS = {
    'SP500': {
        'ticker': '^GSPC',
        'filename': 'SP500 Historical Data.csv',
        'name': 'S&P 500'
    },
    'NASDAQ': {
        'ticker': '^IXIC',
        'filename': 'NASDAQ Historical Data.csv',
        'name': 'Nasdaq Composite'
    },
    'OIL': {
        'ticker': 'CL=F',
        'filename': 'Oil Historical Data.csv',
        'name': 'Crude Oil (WTI)'
    },
    'GOLD': {
        'ticker': 'GC=F',
        'filename': 'Gold Historical Data.csv',
        'name': 'Gold Futures'
    }
}

def fetch_asset_data(asset_key: str, start_date: str = '2020-01-01', end_date: str = None):
    """
    Fetch historical data for a specified asset.

    Args:
        asset_key: Key from ASSETS dict (e.g., 'SP500')
        start_date: Start date for data (YYYY-MM-DD)
        end_date: End date for data (YYYY-MM-DD) - defaults to today
    """
    if asset_key not in ASSETS:
        raise ValueError(f"Unknown asset: {asset_key}. Available: {list(ASSETS.keys())}")

    asset_config = ASSETS[asset_key]
    ticker = asset_config['ticker']
    filename = asset_config['filename']
    name = asset_config['name']

    logger.info(f"Fetching {name} data ({ticker}) from {start_date} to {end_date or 'today'}...")

    # Download data
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            logger.error(f"No data received for {name}")
            return False

        logger.info(f"Downloaded {len(data)} records for {name}")
        logger.info(f"Date range: {data.index.min()} to {data.index.max()}")

        # Format similar to existing CSVs
        data.reset_index(inplace=True)

        # For indices and commodities, use 'Adj Close' or 'Close'
        # Handle column naming in yfinance (multi-index columns)
        if ('Adj Close', ticker) in data.columns:
            price_series = data[('Adj Close', ticker)].squeeze()
        elif 'Adj Close' in data.columns:
            price_series = data['Adj Close'].squeeze()
        else:
            price_series = data['Close'].squeeze()

        if ('Open', ticker) in data.columns:
            open_series = data[('Open', ticker)].squeeze()
            high_series = data[('High', ticker)].squeeze()
            low_series = data[('Low', ticker)].squeeze()
            volume_series = data[('Volume', ticker)].squeeze()
        else:
            open_series = data['Open'].squeeze()
            high_series = data['High'].squeeze()
            low_series = data['Low'].squeeze()
            volume_series = data['Volume'].squeeze()

        # Handle volume (set to 0 for indices since it's not meaningful)
        volume_series = pd.Series([0] * len(data), dtype=int)

        formatted_df = pd.DataFrame({
            'Date': data['Date'].dt.strftime('%b %d, %Y'),  # e.g., "Oct 23, 2023"
            'Price': price_series.round(2),  # 2 decimals for prices
            'Open': open_series.round(2),
            'High': high_series.round(2),
            'Low': low_series.round(2),
            'Volume': volume_series
        })

        # Save to Data directory
        data_dir = Path('./Data')
        data_dir.mkdir(exist_ok=True)
        filepath = data_dir / filename

        formatted_df.to_csv(filepath, index=False)
        logger.info(f"Saved data to {filepath}")

        # Show some stats
        logger.info(f"Price range: ${formatted_df['Price'].min():.2f} - ${formatted_df['Price'].max():.2f}")
        logger.info(f"Average price: ${formatted_df['Price'].mean():.2f}")

        return True

    except Exception as e:
        logger.error(f"Failed to fetch data for {name}: {e}")
        return False

def fetch_all_assets():
    """Fetch data for all assets defined in ASSETS."""
    logger.info("Fetching data for all assets...")

    success_count = 0
    for asset_key in ASSETS.keys():
        success = fetch_asset_data(asset_key, start_date='2020-01-01')
        if success:
            success_count += 1

    logger.info(f"Successfully fetched data for {success_count}/{len(ASSETS)} assets")

def main():
    """Main function to run data fetching."""
    import argparse

    parser = argparse.ArgumentParser(description='Fetch historical asset data')
    parser.add_argument('--asset', choices=list(ASSETS.keys()), help='Specific asset to fetch')
    parser.add_argument('--start', default='2020-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--all', action='store_true', help='Fetch all assets')

    args = parser.parse_args()

    if args.all:
        fetch_all_assets()
    elif args.asset:
        fetch_asset_data(args.asset, start_date=args.start)
    else:
        print(f"Available assets: {list(ASSETS.keys())}")
        print("Use --all to fetch all, or --asset ASSET_KEY for specific asset")
        print("\nExamples:")
        print("  python data_fetcher.py --all")
        print("  python data_fetcher.py --asset SP500")
        print("  python data_fetcher.py --asset GOLD --start 2022-01-01")

if __name__ == '__main__':
    main()
