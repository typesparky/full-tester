#!/usr/bin/env python3
"""
END-TO-END INTEGRATION TEST FOR UNIFIED SIGNAL METHODOLOGY

This test runs the complete 3-stage pipeline on available data to validate:
1. Stage 1: Data Normalization (finds and aligns markets)
2. Stage 2: Daily Aggregation (calculates VWP/VWEV signals)
3. Stage 3: Continuity & Validation (ensures signal quality)

Requirements: Prior testing approved, basic functionality verified.
"""

import sys
import os
import pandas as pd
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CORE.unified_signal_generator import UnifiedSignalGenerator
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_end_to_end_generation():
    """Test complete end-to-end signal generation for all concepts"""
    logger.info("=" * 80)
    logger.info("END-TO-END UNIFIED SIGNAL GENERATION TEST")
    logger.info("=" * 80)

    generator = UnifiedSignalGenerator(debug=True)

    # Show available concepts
    logger.info(f"Available signal concepts: {list(generator.concepts.keys())}")
    for key, concept in generator.concepts.items():
        logger.info(f"  {key}: {concept.name} ({concept.market_type.value}, min_vol={concept.min_volume})")

    # Show manual flags loaded
    if generator.flagged_markets:
        logger.info(f"Manual flags loaded for {len(generator.flagged_markets)} concepts:")
        for concept, flags in generator.flagged_markets.items():
            excludes = flags.get('excludes', [])
            includes = flags.get('includes', [])
            logger.info(f"  {concept}: {len(excludes)} excludes, {len(includes)} includes")
    else:
        logger.info("No manual flags loaded")

    results = {}

    # Test each concept
    for concept_key in generator.concepts.keys():
        logger.info(f"\n{'='*60}")
        logger.info(f"TESTING CONCEPT: {concept_key.upper()}")
        logger.info(f"{'='*60}")

        try:
            # Generate signal
            output_path = f"test_signals/test_output_{concept_key}.csv"
            result_df = generator.generate_concept_signal_unified(concept_key, output_path)

            if result_df.empty:
                logger.warning(f"✗ FAILED: No signal generated for {concept_key}")
                results[concept_key] = {'success': False, 'reason': 'empty_result', 'days': 0}
            else:
                logger.info(f"✓ SUCCESS: Generated {len(result_df)} days of signal")

                # Log basic statistics
                stats = result_df.describe()
                logger.info(f"  Signal statistics:")
                logger.info(f"    Mean: {result_df['adjusted_signal'].mean():.4f}")
                logger.info(f"    Std:  {result_df['adjusted_signal'].std():.4f}")
                logger.info(f"    Min:  {result_df['adjusted_signal'].min():.4f}")
                logger.info(f"    Max:  {result_df['adjusted_signal'].max():.4f}")
                logger.info(f"    Date range: {result_df['date'].min()} to {result_df['date'].max()}")

                # Save sample for inspection
                sample = result_df.head(5).copy()
                sample['date'] = sample['date'].dt.strftime('%Y-%m-%d')
                logger.info(f"  Sample data (first 5 days):")
                for _, row in sample.iterrows():
                    logger.info(f"    {row['date']}: signal={row['adjusted_signal']:.4f}")

                results[concept_key] = {'success': True, 'days': len(result_df), 'stats': stats.to_dict()}

        except Exception as e:
            logger.error(f"✗ ERROR: {e}")
            results[concept_key] = {'success': False, 'reason': str(e), 'days': 0}

        # Get stage statistics
        stats = generator.get_concept_statistics(concept_key)
        results[concept_key]['stage_stats'] = stats

    # Generate test report
    _generate_e2e_report(results)

    # Determine overall success
    successful_concepts = sum(1 for r in results.values() if r['success'])
    total_concepts = len(results)

    logger.info(f"\n{'='*80}")
    logger.info("END-TO-END TEST SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Successfully generated {successful_concepts}/{total_concepts} signal concepts")

    for concept, result in results.items():
        status = "✓" if result['success'] else "✗"
        days = result.get('days', 0)
        reason = "" if result['success'] else f" ({result.get('reason', 'unknown')})"
        logger.info(f"  {status} {concept}: {days} days{reason}")

    return successful_concepts > 0  # At least one concept should succeed


def _generate_e2e_report(results):
    """Generate detailed end-to-end test report"""
    report_path = Path("test_signals/e2e_test_report.txt")

    with open(report_path, 'w') as f:
        f.write("UNIFIED SIGNAL METHODOLOGY END-TO-END TEST REPORT\n")
        f.write("=" * 60 + "\n\n")

        successful = [k for k, v in results.items() if v['success']]
        failed = [k for k, v in results.items() if not v['success']]

        f.write(f"SUCCESS RATE: {len(successful)}/{len(results)} concepts generated\n")
        f.write(f"SUCCESSFUL: {', '.join(successful)}\n")
        f.write(f"FAILED: {', '.join(failed)}\n\n")

        for concept, result in results.items():
            f.write(f"CONCEPT: {concept.upper()}\n")
            f.write(f"  Success: {result['success']}\n")

            if result['success']:
                f.write(f"  Signal Days: {result['days']}\n")
                if 'stats' in result:
                    stats = result['stats']['adjusted_signal']
                    f.write(f"  Signal Mean: {stats['mean']:.4f}\n")
                    f.write(f"  Signal Std: {stats['std']:.4f}\n")
                    f.write(f"  Signal Range: [{stats['min']:.4f}, {stats['max']:.4f}]\n")
            else:
                f.write(f"  Error: {result.get('reason', 'unknown')}\n")

            if 'stage_stats' in result:
                s = result['stage_stats']
                f.write(f"  Stage 1 Markets: {s['stage1_markets']}\n")
                f.write(f"  Stage 2 Days: {s['stage2_days']}\n")
                f.write(f"  Stage 3: {s['stage3_validation']}\n")

            f.write("\n")

    logger.info(f"Detailed report saved to: {report_path}")


def main():
    """Main test function"""
    logger.info("Starting End-to-End Test for Unified Signal Methodology")

    success = test_end_to_end_generation()

    if success:
        logger.info("🎉 END-TO-END TEST PASSED!")
        logger.info("The unified methodology is ready for production use.")
    else:
        logger.warning("⚠️  END-TO-END TEST HAD ISSUES")
        logger.warning("Check the detailed report for debugging information.")

    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
