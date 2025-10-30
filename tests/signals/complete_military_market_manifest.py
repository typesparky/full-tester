#!/usr/bin/env python3
"""
COMPLETE MILITARY MARKET MANIFEST

Creates a comprehensive CSV of all markets processed for military signal,
including both included and excluded markets with filtering details.
"""

import sys
import os
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CORE.unified_signal_generator import UnifiedSignalGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_complete_military_manifest():
    """Create comprehensive CSV of all military signal markets"""

    print("🇺🇳 CREATING COMPLETE MILITARY MARKET MANIFEST")
    print("="*60)

    # Define military keywords
    military_keywords = [
        'military action', 'military strike', 'invade', 'strike on', 'attack on',
        'attack by', 'strike by', 'military engagement', 'military clash',
        'military response', 'declare war', 'military airstrike', 'invasion',
        'capture by', 'seize by', 'escalat', 'coalition strike', 'air raid',
        'bombard', 'bombed', 'missile strike', 'missile attack', 'rocket strike',
        'rocket attack', 'drone strike', 'drone attack', 'artillery strike',
        'shelling', 'troop deployment', 'troop movement', 'offensive',
        'counterattack', 'counter strike', 'ground offensive', 'naval blockade',
        'siege', 'incursion', 'raid', 'ambush', 'assault', 'hostile action',
        'military intervention', 'occupy', 'territorial expansion', 'annex',
        'take control', 'seizure', 'hostilities begin', 'conflict escalates',
        'military build up'
    ]

    # Collect all markets and their processing results
    all_market_results = []

    # Military signal config
    config_path = Path("config/signals/military_actions_signal.yaml")
    with open(config_path, 'r') as f:
        military_config = yaml.safe_load(f)

    manifest_files = military_config['input_manifest_files']

    for manifest_file in manifest_files:
        manifest_path = Path(manifest_file)

        print(f"Processing: {manifest_file}")

        if not manifest_path.exists():
            continue

        df = pd.read_csv(manifest_path, dtype=str)

        for idx, row in df.iterrows():
            question = str(row.get('question', '')).lower()

            # Handle volume parsing
            volume_str = row.get('volume', '0')
            try:
                volume = float(volume_str.replace(',', '')) if volume_str else 0
            except (ValueError, TypeError):
                volume = 0

            # Basic filtering checks
            quality_filters = {
                'minimum_volume': 1000,
                'exclude_patterns': ['say', 'went to', 'speaker'],
                'military_keywords_found': [],
                'filter_reasons': []
            }

            # Check exclusion patterns
            excluded = False
            for exclude_pattern in quality_filters['exclude_patterns']:
                if exclude_pattern.lower() in question:
                    quality_filters['filter_reasons'].append(f'Excluded: Contains "{exclude_pattern}"')
                    excluded = True
                    break

            # Check volume
            if volume < quality_filters['minimum_volume'] and not excluded:
                quality_filters['filter_reasons'].append('Excluded: Below minimum volume 1000')
                excluded = True

            # Check military keywords
            military_match = False
            if not excluded:
                for keyword in military_keywords:
                    if keyword.replace('.*', ' ').lower() in question:
                        quality_filters['military_keywords_found'].append(keyword)
                        military_match = True

                if not military_match:
                    quality_filters['filter_reasons'].append('Excluded: No military keywords found')
                    excluded = True

            # Determine polarity and sentiment if included
            polarity_score = 0
            sentiment = 'EXCLUDED 🔴' if excluded else 'ANALYZED 🔵'

            if not excluded:
                escalation_count = sum(1 for k in military_keywords[:28] if k.replace('.*', ' ').replace(' ', '').lower() in question.replace(' ', ''))
                de_escalation_count = sum(1 for k in military_keywords[28:] if k.replace('.*', ' ').replace(' ', '').lower() in question.replace(' ', ''))

                if escalation_count > de_escalation_count:
                    polarity_score = 1
                    sentiment = 'ESCALATION ⚠️'
                elif de_escalation_count > escalation_count:
                    polarity_score = -1
                    sentiment = 'DE-ESCALATION 🛡️'

            # Create comprehensive market record
            market_record = {
                'source_manifest': manifest_file,
                'question': row.get('question', ''),
                'slug': row.get('slug', ''),
                'volume': volume,
                'volume_formatted': f"{volume:,.2f}",
                'active': row.get('active', 'true') == 'true',
                'included_in_signal': not excluded,
                'sentiment': sentiment,
                'polarity_score': polarity_score,
                'military_keywords_matched': len(quality_filters['military_keywords_found']),
                'military_keywords_list': '; '.join(quality_filters['military_keywords_found']),
                'filter_reasons': '; '.join(quality_filters['filter_reasons']) if quality_filters['filter_reasons'] else 'PASSED ALL FILTERS ✅',
                'question_length': len(question),
                'processed_timestamp': datetime.now().isoformat()
            }

            all_market_results.append(market_record)

    # Convert to DataFrame and save
    manifest_df = pd.DataFrame(all_market_results)

    # Sort by inclusion status, volume, and keywords
    manifest_df = manifest_df.sort_values(['included_in_signal', 'volume', 'military_keywords_matched'],
                                        ascending=[False, False, False])

    # Save comprehensive CSV
    output_file = 'sector_signals/complete_military_market_manifest.csv'
    manifest_df.to_csv(output_file, index=False)

    # Summary statistics
    total_markets = len(manifest_df)
    included_markets = len(manifest_df[manifest_df['included_in_signal']])
    excluded_markets = total_markets - included_markets
    total_volume_all = manifest_df['volume'].sum()
    total_volume_included = manifest_df[manifest_df['included_in_signal']]['volume'].sum()

    print("
📊 MILITARY MARKET MANIFEST SUMMARY:"    print(f"📄 Total Markets Processed: {total_markets:,}")
    print(f"✅ Markets Included: {included_markets:,} ({included_markets/total_markets*100:.1f}%)")
    print("
💵 VOLUME BREAKDOWN:"    print(f"💰 Total Volume (All): ${total_volume_all:,.2f}")
    print(f"💰 Included Volume: ${total_volume_included:,.2f} ({total_volume_included/total_volume_all*100:.1f}%)")

    print("
😊 SENTIMENT BREAKDOWN:"    sentiment_counts = manifest_df['sentiment'].value_counts()
    for sentiment, count in sentiment_counts.items():
        pct = count / total_markets * 100
        print(f"{sentiment}: {count:,} ({pct:.1f}%)")

    print("
📁 MANIFEST FILES PROCESSED:"    for source in manifest_df['source_manifest'].unique():
        source_count = len(manifest_df[manifest_df['source_manifest'] == source])
        included_from_source = len(manifest_df[(manifest_df['source_manifest'] == source) & manifest_df['included_in_signal']])
        print(f"• {source}: {source_count} markets, {included_from_source} included")

    print(f"\n🔗 COMPLETE MANIFEST SAVED: {output_file}")
    print(f"📊 Includes: {len(manifest_df.columns)} columns of detailed filtering information")
    print("\n✅ MILITARY MARKET MANIFEST COMPLETE!")

if __name__ == "__main__":
    import yaml
    create_complete_military_manifest()
