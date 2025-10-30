#!/usr/bin/env python3
"""
ANALYZE MILITARY MARKET DATES AND VOLUME STATISTICS

Provide comprehensive analysis of military markets including:
- Start and end dates for each market
- Volume distribution over time
- Active vs closed market status
- Daily volume statistics
"""

import pandas as pd
import os
import glob
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_included_markets():
    """Load the list of markets included in the military signal"""
    try:
        included_df = pd.read_csv("markets_for_signal/military_actions/markets_included_in_signal_66.csv")
        included_df['volume'] = pd.to_numeric(included_df['volume'], errors='coerce')
        return included_df
    except FileNotFoundError:
        print("❌ Included markets file not found")
        return pd.DataFrame()

def get_market_date_range_and_volume(market_slug, market_name):
    """Get start date, end date, and daily volume for a specific market"""

    # Find the market data file
    market_pattern = f"polymarket_data/**/{market_slug}.csv"
    market_files = glob.glob(market_pattern, recursive=True)

    if not market_files:
        return None, None, None, 0, 0

    try:
        market_path = market_files[0]
        market_df = pd.read_csv(market_path)

        # Check for date column (might be 'date' or 'created_at' or other)
        date_col = None
        for col in ['date', 'created_at', 'timestamp', 'Date']:
            if col in market_df.columns:
                date_col = col
                break

        if date_col is None:
            print(f"⚠️ No date column found for {market_name}")
            return None, None, None, market_df.get('volume', 0), len(market_df) if len(market_df) > 0 else 0

        # Convert date column
        try:
            market_df[date_col] = pd.to_datetime(market_df[date_col])
        except:
            # Try different formats
            for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%Y-%m-%d %H:%M:%S']:
                try:
                    market_df[date_col] = pd.to_datetime(market_df[date_col], format=fmt)
                    break
                except:
                    continue
            else:
                # Last resort - parse without format
                market_df[date_col] = pd.to_datetime(market_df[date_col], errors='coerce')

        market_df.dropna(subset=[date_col], inplace=True)

        if market_df.empty:
            return None, None, None, 0, 0

        start_date = market_df[date_col].min()
        end_date = market_df[date_col].max()

        # Try to get volume data
        volume_data = None
        for vol_col in ['volume', 'Volume', 'total_volume', 'trading_volume']:
            if vol_col in market_df.columns:
                volume_data = market_df[vol_col]
                break

        if volume_data is None:
            # Some markets might have price history as volume proxy
            daily_points = len(market_df)
        else:
            volume_data = pd.to_numeric(volume_data, errors='coerce')
            daily_points = len(volume_data.dropna()) if volume_data is not None else len(market_df)

        return start_date, end_date, volume_data, daily_points, len(market_df)

    except Exception as e:
        print(f"❌ Error processing {market_name}: {e}")
        return None, None, None, 0, 0

def analyze_military_markets_comprehensive():
    """Comprehensive analysis of military markets with dates and volumes"""

    print("🇺🇳 COMPREHENSIVE MILITARY MARKET DATE & VOLUME ANALYSIS")
    print("=" * 70)

    # Load included markets
    included_df = load_included_markets()
    if included_df.empty:
        print("❌ No included markets data found")
        return

    print(f"Analyzing {len(included_df)} markets included in military signal...\n")

    # Collect detailed market data
    market_details = []

    for idx, row in included_df.iterrows():
        market_name = row['question'][:50] + "..." if len(row['question']) > 50 else row['question']
        market_slug = row['slug']
        manifest_volume = row['volume'] if pd.notna(row['volume']) else 0

        print(f"📊 [{idx+1:2d}/{len(included_df)}] Analyzing: {market_name}")

        start_date, end_date, volume_series, daily_points, total_records = get_market_date_range_and_volume(
            market_slug, market_name)

        market_info = {
            'market_name': row['question'],
            'slug': market_slug,
            'manifest_volume': manifest_volume,
            'status': 'Active' if str(row.get('active', 'true')).lower() == 'true' else 'Closed',
            'start_date': start_date,
            'end_date': end_date,
            'date_range_days': (end_date - start_date).days if (start_date and end_date) else None,
            'total_records': total_records,
            'daily_data_points': daily_points,
            'data_completeness': daily_points / total_records if total_records > 0 else 0,
            'source_file': row.get('source_file', 'Unknown')
        }

        market_details.append(market_info)

    # Create detailed DataFrame
    details_df = pd.DataFrame(market_details)

    # Save comprehensive analysis
    output_dir = Path("markets_for_signal/military_actions")
    output_dir.mkdir(parents=True, exist_ok=True)

    comprehension_path = output_dir / "military_markets_detailed_analysis_with_dates.csv"
    details_df.to_csv(comprehension_path, index=False)
    print(f"✅ Saved detailed analysis: {comprehension_path}")

    # Generate summary statistics
    generate_volume_date_summary(details_df, output_dir)

    # Show key findings
    print("\n" + "="*80)
    print("🎯 KEY MARKET DATE & STATUS FINDINGS")
    print("="*80)

    # Active vs Closed status
    active_markets = len(details_df[details_df['status'] == 'Active'])
    closed_markets = len(details_df[details_df['status'] == 'Closed'])
    print(f"📊 Active Markets: {active_markets} ({active_markets/len(details_df)*100:.1f}%)")
    print(f"🔒 Closed Markets: {closed_markets} ({closed_markets/len(details_df)*100:.1f}%)")

    # Date range analysis
    valid_dates = details_df.dropna(subset=['start_date', 'end_date'])
    if not valid_dates.empty:
        avg_duration = valid_dates['date_range_days'].mean()
        min_duration = valid_dates['date_range_days'].min()
        max_duration = valid_dates['date_range_days'].max()

        print("
📅 Market Duration Statistics:"        print(f"   Average duration: {avg_duration:.0f} days")
        print(f"   Shortest market: {min_duration:.0f} days")
        print(f"   Longest market:  {max_duration:.0f} days")

        # Date range coverage
        earliest_start = valid_dates['start_date'].min()
        latest_end = valid_dates['end_date'].max()
        print("
📆 Overall Date Coverage:"        print(f"   Earliest start: {earliest_start.strftime('%Y-%m-%d')}")
        print(f"   Latest end:     {latest_end.strftime('%Y-%m-%d')}")
        print(f"   Total span:     {(latest_end - earliest_start).days:,} days")

    # Data completeness
    avg_completeness = details_df['data_completeness'].mean()
    print(f"\n💾 Data Completeness: {avg_completeness:.1%} average")

    # Show top markets by different metrics
    print("
🏆 Top Markets by Volume:"    volume_sorted = details_df.nlargest(10, 'manifest_volume')
    for idx, (_, market) in enumerate(volume_sorted.iterrows(), 1):
        name_short = market['market_name'][:45] + "..." if len(market['market_name']) > 45 else market['market_name']
        volume = market['manifest_volume']
        status = market['status']
        start_str = market['start_date'].strftime('%Y-%m-%d') if pd.notna(market['start_date']) else 'Unknown'
        end_str = market['end_date'].strftime('%Y-%m-%d') if pd.notna(market['end_date']) else 'Unknown'
        print("4d")

    print("\n🎉 Analysis complete!"
def generate_volume_date_summary(details_df, output_dir):
    """Generate summary report of volume and date statistics"""

    summary_path = output_dir / "volume_and_date_statistics_summary.txt"

    with open(summary_path, 'w') as f:
        f.write("MILITARY MARKETS - VOLUME & DATE STATISTICS SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        # Overall statistics
        f.write("OVERALL MARKET STATISTICS:\n")
        f.write(f"Total markets analyzed: {len(details_df)}\n")
        f.write(f"Active markets: {len(details_df[details_df['status'] == 'Active'])}\n")
        f.write(f"Closed markets: {len(details_df[details_df['status'] == 'Closed'])}\n")
        f.write(".0f"
        f.write(".0f"
        f.write(".0f"
        f.write(".2f")

        # Date-based analysis
        valid_dates = details_df.dropna(subset=['start_date', 'end_date'])
        if not valid_dates.empty:
            f.write("\nMARKET DURATION ANALYSIS:\n")
            f.write("Average market duration: {})
            f.write("Shortest market: {})
            f.write("Longest market: {})

            # Monthly breakdown
            f.write("\nMONTHLY MARKET STARTS:\n")
            start_months = valid_dates['start_date'].dt.to_period('M').value_counts().sort_index()
            for month, count in start_months.items():
                f.write("15")

            # Markets by year
            f.write("\nMARKETS BY YEAR (Start Date):\n")
            start_years = valid_dates['start_date'].dt.year.value_counts().sort_index()
            for year, count in start_years.items():
                f.write(f"Year {year}: {count} markets\n")

        # Top volume markets detail
        f.write("\nTOP 10 MARKETS BY VOLUME:\n")
        top_volume = details_df.nlargest(10, 'manifest_volume')
        for idx, (_, market) in enumerate(top_volume.iterrows(), 1):
            f.write("2d"                if pd.notna(market['start_date']):
                    f.write(" - {} to {}".format(
                        market['start_date'].strftime('%Y-%m-%d'),
                        market['end_date'].strftime('%Y-%m-%d') if pd.notna(market['end_date']) else 'Ongoing'
                    ))
                f.write(f" ({market['status']})\n")

    print(f"✅ Generated summary report: {summary_path}")

if __name__ == "__main__":
    analyze_military_markets_comprehensive()
