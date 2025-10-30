#!/usr/bin/env python3

"""
Script to generate multiple Polymarket signal time series for different categories.
"""

import os
from pathlib import Path
from src.signals.signal_processor import SignalProcessor

def main():
    """Generate signals for all categories."""

    # Define categories and their parameters - now with separate manifests
    categories = [
        {
            'name': 'inflation',
            'keyword': 'inflation',
            'signal_type': 'vwev',  # quantitative markets
            'output_file': 'data/processed_signals/signal_vwev_inflation.csv',
            'manifest': 'polymarket_data/manifest/_manifest_inflation.csv',
            'data_dir': 'polymarket_data/inflation'
        },
        {
            'name': 'gdp',
            'keyword': 'gdp',
            'signal_type': 'vwev',  # quantitative markets
            'output_file': 'data/processed_signals/signal_vwev_gdp.csv',
            'manifest': 'polymarket_data/manifest/_manifest_gdp.csv',
            'data_dir': 'polymarket_data/us_gdp'
        },
        {
            'name': 'politics',
            'keyword': 'election',
            'signal_type': 'vwp',   # binary markets
            'output_file': 'data/processed_signals/signal_vwp_politics.csv',
            'manifest': 'polymarket_data/manifest/_manifest_politics_(us_election).csv',
            'data_dir': 'polymarket_data/politics_(us_election)'
        },
        {
            'name': 'fed_interest_rates',
            'keyword': 'rate',
            'signal_type': 'vwev',  # quantitative markets
            'output_file': 'data/processed_signals/signal_vwev_fed_interest_rates.csv',
            'manifest': 'polymarket_data/manifest/_manifest_fed_&_interest_rates.csv',
            'data_dir': 'polymarket_data/fed_&_interest_rates'
        },
        {
            'name': 'treasury_yields',
            'keyword': 'yield',
            'signal_type': 'vwev',  # quantitative markets
            'output_file': 'data/processed_signals/signal_vwev_treasury_yields.csv',
            'manifest': 'polymarket_data/manifest/_manifest_treasury_&_yields_(2025).csv',
            'data_dir': 'polymarket_data/treasury_&_yields_(2025)'
        },
        {
            'name': 'us_jobs_unemployment',
            'keyword': 'unemployment',
            'signal_type': 'vwev',  # quantitative markets
            'output_file': 'data/processed_signals/signal_vwev_us_jobs_unemployment.csv',
            'manifest': 'polymarket_data/manifest/_manifest_us_jobs_&_unemployment.csv',
            'data_dir': 'polymarket_data/us_jobs_&_unemployment'
        }
    ]

    print("Generating signals for multiple categories...")

    # Generate signals for each category
    for category in categories:
        print(f"\nGenerating signal for {category['name']}...")
        print(f"  Using manifest: {category['manifest']}")
        print(f"  Using data dir: {category['data_dir']}")

        try:
            # Create processor with category-specific manifest and data directory
            processor = SignalProcessor(category['manifest'], category['data_dir'])

            if category['signal_type'] == 'vwev':
                df_signal = processor.generate_quantitative_signal(
                    sector=f"us {category['name']}",  # e.g., "us gdp", "us inflation"
                    output_path=category['output_file']
                )
            else:
                df_signal = processor.generate_signal_time_series(
                    keyword=category['keyword'],
                    output_path=category['output_file']
                )

            if df_signal.empty:
                print(f"  ⚠️  No data found for {category['name']}")
            else:
                print(f"  ✅ Generated {len(df_signal)} data points for {category['name']}")
                print(f"     Output saved to: {category['output_file']}")

        except Exception as e:
            print(f"  ❌ Error generating signal for {category['name']}: {e}")

    print("\nSignal generation complete!")

if __name__ == "__main__":
    main()
