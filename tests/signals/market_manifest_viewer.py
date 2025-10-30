#!/usr/bin/env python3
"""
MARKET MANIFEST VIEWER

Displays filtered manifest data with clear market details including start/end dates.
Shows exactly what markets feed into each signal.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import pandas as pd

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CORE.unified_signal_generator import UnifiedSignalGenerator
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MarketInfo:
    """Clean market information"""
    filename: str
    question: str
    volume: float
    active: bool
    start_date: str = None
    end_date: str = None

class MarketManifestViewer:
    """Displays clean, filtered market manifests"""

    def __init__(self):
        self.generator = UnifiedSignalGenerator(debug=False)

    def load_market_data(self, manifest_path: Path) -> List[MarketInfo]:
        """Load market data from manifest file"""
        if not manifest_path.exists():
            return []

        try:
            df = pd.read_csv(manifest_path, dtype=str)  # Make sure all columns are read as strings

            markets = []
            for _, row in df.iterrows():
                filename = row['filename']

                # Try to load individual market data if path exists
                sector_name = str(row.get('sector', 'unknown')).replace(' ', '_').lower()
                market_data_path = manifest_path.parent / sector_name / filename
                start_date = None
                end_date = None

                if market_data_path.exists():
                    try:
                        market_df = pd.read_csv(market_data_path)
                        if not market_df.empty and 'date' in market_df.columns:
                            dates = pd.to_datetime(market_df['date'].dropna())
                            if len(dates) > 0:
                                start_date = dates.min().strftime('%Y-%m-%d')
                                end_date = dates.max().strftime('%Y-%m-%d')
                    except Exception:
                        pass

                market = MarketInfo(
                    filename=filename,
                    question=row['question'],
                    volume=float(row.get('volume', 0)),
                    active=row.get('active', True) != False,
                    start_date=start_date,
                    end_date=end_date
                )
                markets.append(market)

            return markets

        except Exception as e:
            logger.error(f"Could not load {manifest_path}: {e}")
            return []

    def display_manifest_by_signal(self, signal_name: str = None):
        """Display markets grouped by signal"""
        print("\n" + "="*100)
        print(f"🎯 SIGNAL MARKET MANIFESTS")
        print("="*100)

        # Load all signal configs
        signals_dir = Path("config/signals")
        signal_configs = list(signals_dir.glob("*.yaml")) if signals_dir.exists() else []

        for config_file in signal_configs:
            if signal_name and signal_name.lower() not in config_file.stem.lower():
                continue

            try:
                import yaml
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)

                signal_display_name = config.get('signal_name', config_file.stem.upper())
                input_manifests = config.get('input_manifest_files', [])

                print(f"\n🎯 SIGNAL: {signal_display_name}")
                print(f"📁 Config: {config_file.name}")
                print("-" * 60)

                total_markets = 0
                active_markets = 0

                for manifest_name in input_manifests:
                    manifest_path = Path(manifest_name)
                    print(f"\n📄 {manifest_name}:")
                    print("-" * 40)

                    try:
                        if manifest_path.exists():
                            markets = self.load_market_data(manifest_path)

                            if not markets:
                                print("❌ No data found")
                                continue

                            # Sort by volume descending and show summary info
                            markets.sort(key=lambda x: x.volume, reverse=True)
                            total_in_file = len(markets)
                            active_in_file = sum(1 for m in markets if m.active)
                            top_volume = max([m.volume for m in markets]) if markets else 0

                            print(f"   📊 {total_in_file} markets ({active_in_file} active), Top volume: ${top_volume:,.0f}")
                            print(f"   📁 All markets unfiltered (active + inactive / closed)")

                        else:
                            print(f"❌ Manifest file not found: {manifest_path}")
                            continue

                    except Exception as e:
                        print(f"❌ Error loading manifest: {e}")
                        continue

                    total_markets += total_in_file if 'total_in_file' in locals() else 0
                    active_markets += active_in_file if 'active_in_file' in locals() else 0

                print(f"\n📊 SUMMARY:")
                print(f"   Total Markets: {total_markets}")
                print(f"   Active Markets: {active_markets}")
                print(f"   Manifest Files: {len(input_manifests)}")
                # Calculate average volume
                total_volume = sum([sum([m.volume for m in self.load_market_data(Path(manifest))]) for manifest in input_manifests])
                avg_volume = total_volume / max(total_markets, 1)
                print(f"   Average Volume per Market: ${avg_volume:,.0f}")

            except Exception as e:
                print(f"❌ Error reading config {config_file}: {e}")

    def display_concept_breakdown(self, concept_name: str = None):
        """Show markets by concept with clean formatting"""
        print("\n" + "="*100)
        print(f"🏛️ MARKET BREAKDOWN BY CONCEPT")
        print("="*100)

        manifest_dir = Path("polymarket_data")
        manifest_files = list(manifest_dir.glob("_manifest_*.csv"))

        concept_groups = {}

        for manifest_file in manifest_files:
            # Classify manifest
            if "geopolitics" in manifest_file.name.lower():
                group = "Geopolitics/War"
            elif "custom_world_affairs" in manifest_file.name.lower():
                group = "World Affairs (Mixed)"
            elif "politics" in manifest_file.name.lower():
                group = "US Politics"
            elif "fed" in manifest_file.name.lower() or "interest" in manifest_file.name.lower():
                group = "Fed/Economy"
            elif "gdp" in manifest_file.name.lower():
                group = "US Economic Growth"
            else:
                group = "Other"

            if concept_name and concept_name.lower() not in group.lower():
                continue

            if group not in concept_groups:
                concept_groups[group] = []

            concept_groups[group].append(manifest_file)

        for group_name, manifests in concept_groups.items():
            print(f"\n🎯 {group_name.upper()}")
            print("=" * len(group_name) + "=======================")

            group_total = 0
            group_active = 0

            for manifest_file in manifests:
                markets = self.load_market_data(manifest_file)
                group_total += len(markets)
                group_active += sum(1 for m in markets if m.active)

                print(f"\n📄 {manifest_file.name}:")
                print("-" * (len(manifest_file.name) + 8))

                # Show ALL markets (remove volume filter)
                all_markets = sorted(markets, key=lambda x: x.volume, reverse=True)

                for i, market in enumerate(all_markets[:10], 1):  # Show top 10 by volume
                    volume_str = f"${market.volume:,.0f}"
                    date_range = ""
                    if market.start_date and market.end_date:
                        date_range = f" ({market.start_date} → {market.end_date})"

                    print(f"  {i}. ${market.volume:,.0f} {market.question[:60]}...{date_range}")

                if len(all_markets) > 10:
                    print(f"   ...and {len(all_markets) - 10} more markets (including inactive)")

            print(f"\n📊 GROUP Total: {group_total} markets ({group_active} active)")

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Market Manifest Viewer - Shows clean market data by signal")
    parser.add_argument('--signal', '-s', type=str, help='Show markets for specific signal (e.g., geopolitics)')
    parser.add_argument('--concept', '-c', type=str, help='Show markets by concept (e.g., geopolitics)')
    parser.add_argument('--all-signals', action='store_true', help='Show all signals')

    # Default: show all if no args provided
    if len(sys.argv) == 1:
        args = parser.parse_args(['--all-signals'])
    else:
        args = parser.parse_args()

    viewer = MarketManifestViewer()

    if args.signal:
        viewer.display_manifest_by_signal(args.signal)
    elif args.concept:
        viewer.display_concept_breakdown(args.concept)
    else:
        print("🌍 MARKET MANIFEST VIEWER")
        print("=" * 80)
        print()
        print("Commands:")
        print("  --signal SIGNAL     Show markets used by specific signal")
        print("  --concept CONCEPT   Show markets by concept category")
        print("  --all-signals       Show markets for all signals")
        print()
        viewer.display_manifest_by_signal()

if __name__ == "__main__":
    main()
