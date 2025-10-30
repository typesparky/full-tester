#!/usr/bin/env python3
"""
ANALYZE MARKET SELECTION IN UNIFIED SIGNALS

This script examines exactly which markets were selected for each concept,
their polarities, flags, and how they contribute to the final signal.
Useful for manual validation and building a cleaner, more accurate model.
"""

import sys
import os
import pandas as pd
from pathlib import Path
from enum import Enum
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CORE.unified_signal_generator import UnifiedSignalGenerator
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def analyze_market_selection():
    """Analyze which markets were selected for each concept"""

    logger.info("=" * 100)
    logger.info("MARKET SELECTION ANALYSIS FOR UNIFIED SIGNALS")
    logger.info("=" * 100)

    generator = UnifiedSignalGenerator(debug=False)

    # Load manual flags
    flags_config = generator.flagged_markets if generator.flagged_markets else {}
    logger.info(f"Manual flags loaded for {len(flags_config)} concepts")

    analysis_results = {}

    for concept_key, concept in generator.concepts.items():
        logger.info(f"\n{'='*80}")
        logger.info(f"ANALYZING CONCEPT: {concept.name.upper()}")
        logger.info(f"{'='*80}")

        concept_excludes = set(flags_config.get(concept_key, {}).get('excludes', []))
        concept_includes = set(flags_config.get(concept_key, {}).get('includes', []))

        logger.info(f"Concept keywords: {concept.keywords}")
        logger.info(f"Market type: {concept.market_type.value}")
        logger.info(f"Minimum volume: {concept.min_volume}")
        logger.info(f"Manual excludes ({len(concept_excludes)}): {list(concept_excludes) if concept_excludes else 'None'}")
        logger.info(f"Manual includes ({len(concept_includes)}): {list(concept_includes) if concept_includes else 'None'}")

        if concept.polarity_logic:
            logger.info("Polarity mappings:")
            for term, polarity in concept.polarity_logic.items():
                direction = "PRO-CONCEPT (+1)" if polarity.value == 1 else "ANTI-CONCEPT (-1)"
                logger.info(f"  '{term}' → {direction}")

        # Find all markets that match this concept (simulate Stage 1 logic)
        matched_markets = []

        # Check all manifest files
        for manifest_file in Path("polymarket_data").glob("_manifest_*.csv"):
            try:
                manifest_df = pd.read_csv(manifest_file)
            except Exception as e:
                logger.warning(f"Could not read {manifest_file}: {e}")
                continue

            for _, market_row in manifest_df.iterrows():
                question = market_row['question']

                # Check if market matches concept keywords
                matches_keywords = any(kw.lower() in question.lower() for kw in concept.keywords)

                if not matches_keywords:
                    continue

                # Manual flag checks
                flagged = {
                    'excluded_by_flag': market_row['filename'] in concept_excludes,
                    'included_by_flag': market_row['filename'] in concept_includes if concept_includes else False
                }

                # Volume check
                volume_check = market_row['volume'] >= concept.min_volume

                market_info = {
                    'filename': market_row['filename'],
                    'question': question[:100] + "..." if len(question) > 100 else question,
                    'volume': market_row['volume'],
                    'matched_keywords': [kw for kw in concept.keywords if kw.lower() in question.lower()],
                    'flagged': flagged,
                    'passes_volume': volume_check
                }

                matched_markets.append(market_info)

        logger.info(f"\nFound {len(matched_markets)} markets matching keywords:")

        # Categorize markets
        valid_markets = []
        excluded_markets = []
        low_volume_markets = []

        for market in matched_markets:
            if market['flagged']['excluded_by_flag']:
                excluded_markets.append(market)
            elif not market['passes_volume']:
                low_volume_markets.append(market)
            else:
                valid_markets.append(market)

        logger.info(f"\n{'='*60}")
        logger.info(f"SUMMARY: {len(valid_markets)} VALID, {len(excluded_markets)} EXCLUDED, {len(low_volume_markets)} LOW VOLUME")
        logger.info(f"{'='*60}")

        if valid_markets:
            logger.info("\nVALID MARKETS INCLUDED IN SIGNAL:")
            for i, market in enumerate(valid_markets, 1):
                logger.info(f"{i:2d}. {market['filename']:50} | Vol:{market['volume']:>8.0f} | {market['question'][:80]}")

                # Determine polarity for this market
                polarity = determine_market_polarity(market['question'], concept)
                logger.info(f"    └─ Polarity: λ = {polarity:+d} | Keywords matched: {market['matched_keywords']}")

        if excluded_markets:
            logger.info(f"\nEXCLUDED MARKETS ({len(excluded_markets)}):")
            for market in excluded_markets:
                logger.info(f"  ❌ {market['filename']:50} | {market['question'][:60]}")

        if low_volume_markets:
            logger.info(f"\nLOW VOLUME MARKETS ({len(low_volume_markets)}):")
            for market in low_volume_markets:
                logger.info(f"  💰 {market['filename']:50} | Vol:{market['volume']:>8.0f} | {market['question'][:60]}")

        # Store results
        analysis_results[concept_key] = {
            'concept_name': concept.name,
            'keywords': concept.keywords,
            'market_type': concept.market_type.value,
            'min_volume': concept.min_volume,
            'total_markets_found': len(matched_markets),
            'valid_markets': [{
                'filename': m['filename'],
                'question': m['question'],
                'volume': m['volume'],
                'polarity': determine_market_polarity(m['question'], concept),
                'matched_keywords': m['matched_keywords']
            } for m in valid_markets],
            'excluded_markets': excluded_markets,
            'low_volume_markets': low_volume_markets,
            'manual_flags': flags_config.get(concept_key, {})
        }

    # Save detailed analysis to file
    save_analysis_results(analysis_results)

    # Summary table
    logger.info(f"\n{'='*100}")
    logger.info("OVERALL SUMMARY TABLE")
    logger.info(f"{'='*100}")
    logger.info("Concept".ljust(20) + "Type".ljust(12) + "Total".rjust(8) + "Valid".rjust(8) + "Excluded".rjust(10) + "Low Vol".rjust(10))
    logger.info("-" * 100)

    total_valid = 0
    for concept_key, results in analysis_results.items():
        concept_name = results['concept_name'][:19]
        market_type = results['market_type'][:11]
        total = results['total_markets_found']
        valid = len(results['valid_markets'])
        excluded = len(results['excluded_markets'])
        low_vol = len(results['low_volume_markets'])

        logger.info(f"{concept_name:<20}{market_type:<12}{total:>8}{valid:>8}{excluded:>10}{low_vol:>10}")
        total_valid += valid

    logger.info("-" * 100)
    logger.info(f"Total valid markets across all concepts: {total_valid}")

    return analysis_results


def determine_market_polarity(question: str, concept) -> int:
    """Determine the polarity (λ_i) for a market based on question analysis"""
    if not concept.polarity_logic:
        return 1  # Default pro-concept

    question_lower = question.lower()

    for term, polarity in concept.polarity_logic.items():
        if term.lower() in question_lower:
            return +1 if polarity.value == 1 else -1

    return 1  # Default if no specific terms match


def save_analysis_results(results):
    """Save detailed analysis results to JSON for manual review"""
    output_file = Path("test_signals/market_selection_analysis.json")
    logger.info(f"\nSaving detailed analysis to: {output_file}")

    # Convert to JSON-serializable format
    json_results = {}
    for concept_key, data in results.items():
        json_results[concept_key] = {
            'concept_name': data['concept_name'],
            'keywords': data['keywords'],
            'market_type': data['market_type'],
            'min_volume': data['min_volume'],
            'total_markets_found': data['total_markets_found'],
            'manual_flags': data['manual_flags'],
            'valid_markets_detailed': data['valid_markets'],
            'excluded_files': [m['filename'] for m in data['excluded_markets']],
            'low_volume_files': [m['filename'] for m in data['low_volume_markets']]
        }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)

    logger.info(f"Analysis saved to {output_file}")

    # Create a CSV summary for easy review
    csv_file = Path("test_signals/market_selection_summary.csv")
    rows = []
    for concept_key, data in results.items():
        for market in data['valid_markets']:
            rows.append({
                'concept': concept_key,
                'filename': market['filename'],
                'question': market['question'],
                'volume': market['volume'],
                'polarity': market['polarity'],
                'matched_keywords': ', '.join(market['matched_keywords'])
            })

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(csv_file, index=False)
        logger.info(f"CSV summary saved to {csv_file} ({len(rows)} valid markets)")


def analyze_politcs_signal_in_depth():
    """Deep dive into the politics signal that worked well on SP500"""

    logger.info(f"\n{'='*80}")
    logger.info("DEEP ANALYSIS: POLITICS SIGNAL MARKETS")
    logger.info(f"{'='*80}")

    generator = UnifiedSignalGenerator(debug=False)

    # Get valid markets for politics
    politics_concept = generator.concepts['politics']

    # Load the generated politics signal to see what ranges it covers
    signal_df = pd.read_csv("test_signals/test_output_politics.csv", parse_dates=['date'])
    signal_range = signal_df['adjusted_signal'].describe()

    logger.info("Politics signal summary:")
    logger.info(f"  Days: {len(signal_df)}")
    logger.info(f"  Date range: {signal_df['date'].min()} to {signal_df['date'].max()}")
    logger.info("  Signal distribution:")
    logger.info(".4f")
    logger.info(".1%")
    logger.info("  Sample signals:")
    sample = signal_df.head(10).copy()
    sample['date'] = sample['date'].dt.strftime('%Y-%m-%d')
    for _, row in sample.iterrows():
        logger.info(".4f")


if __name__ == "__main__":
    results = analyze_market_selection()
    analyze_politcs_signal_in_depth()
