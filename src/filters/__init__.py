"""
FILTERING MODULE

Manages application of filtering rules from master_filters.yaml
"""

import yaml
import pandas as pd
import re
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class FilterManager:
    """Loads and applies filters from master_filters.yaml"""

    def __init__(self, config_path='config/master_filters.yaml', save_filtered_manifests=True):
        """Initialize with master filters config"""
        with open(config_path, 'r') as f:
            self.filters = yaml.safe_load(f)['master_filters']
        self.save_filtered_manifests = save_filtered_manifests
        self.filtering_stats: Dict[str, Dict] = {}
        self.manifest_outputs = []
            
    def apply_global_filters(self, manifest_df):
        """Apply global exclusion rules to manifest DataFrame"""
        # Create a copy to avoid modifying original
        original_count = len(manifest_df)
        filtered_df = manifest_df.copy()
        filter_stats = {}

        logger.info(f"Starting filtering on {original_count} markets")

        for category, conf in self.filters['categories'].items():
            if not conf.get('enabled', False):
                continue

            # Check patterns
            patterns = conf.get('detect_patterns', [])
            if isinstance(patterns, str):
                patterns = [patterns]

            # If volume threshold
            volume_threshold = conf.get('volume_threshold', None)

            # Apply exclusion logic
            mask_to_exclude = pd.Series(False, index=filtered_df.index)
            exclusion_reasons = {}

            for pattern in patterns:
                # Check filename column
                mask_filename = filtered_df['filename'].astype(str).str.contains(pattern, case=False, na=False, regex=True)
                mask_question = filtered_df['question'].astype(str).str.contains(pattern, case=False, na=False, regex=True)
                mask_to_exclude |= (mask_filename | mask_question)

                # Track reasons
                exclusion_reason = f"Pattern: {pattern}"
                if exclusion_reason not in exclusion_reasons:
                    exclusion_reasons[exclusion_reason] = 0
                exclusion_reasons[exclusion_reason] += (mask_filename | mask_question).sum()

            # Volume filtering: exclude low-volume active markets but keep closed markets
            if volume_threshold and 'volume' in filtered_df.columns:
                # Keep volume = 0 (closed markets with historical data)
                # Exclude 0 < volume < 100 (low-volume active markets)
                mask_volume = (filtered_df['volume'] > 0) & (filtered_df['volume'] < volume_threshold)
                mask_to_exclude |= mask_volume

                if mask_volume.sum() > 0:
                    exclusion_reasons[f"Volume: 0 < vol < {volume_threshold}"] = mask_volume.sum()

            # Apply exclusion
            excluded_count = mask_to_exclude.sum()
            filtered_df = filtered_df[~mask_to_exclude]

            filter_stats[category] = {
                'excluded': excluded_count,
                'remaining': len(filtered_df),
                'reasons': exclusion_reasons
            }

            logger.info(f"Applied {category}: excluded {excluded_count} markets, {len(filtered_df)} remaining")
            if exclusion_reasons:
                for reason, count in exclusion_reasons.items():
                    logger.info(f"  └─ {reason}: {count} markets")

        # Final stats
        final_count = len(filtered_df)
        total_excluded = original_count - final_count

        self.filtering_stats = {
            'original_count': original_count,
            'final_count': final_count,
            'total_excluded': total_excluded,
            'filter_categories': filter_stats,
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"✅ Filtering complete: {final_count} markets remaining ({total_excluded} excluded)")

        return filtered_df
        
    def filter_manifests(self, manifest_files, min_quality_score=None):
        """Load manifests, apply filters, return filtered file paths"""
        all_filtered_df = []
        
        for manifest_path in manifest_files:
            # Load manifest CSV
            manifest_df = pd.read_csv(manifest_path)
            
            # Apply global filters
            filtered_df = self.apply_global_filters(manifest_df)
            
            # Apply quality score filter
            if min_quality_score is not None and 'quality_score' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['quality_score'] >= min_quality_score]
                
            # Exclude already flagged as excluded
            if 'is_excluded' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['is_excluded'] != True]
                
            all_filtered_df.append(filtered_df)
            
        if all_filtered_df:
            combined_df = pd.concat(all_filtered_df, ignore_index=True)

            # Add market data range analysis if data exists
            combined_df = self._add_market_date_ranges(combined_df)

            # Save filtered manifest if requested
            if self.save_filtered_manifests:
                self._save_filtered_manifest(combined_df)

            # Extract full_path (assuming filename gives relative path)
            # Construct full path from filename if full_path not present
            if 'full_path' in combined_df.columns:
                file_paths = combined_df['full_path'].tolist()
            else:
                # Construct from filename: e.g., "geopolitics__foreign_policy_comprehensive/file.csv" -> "polymarket_data/geopolitics__foreign_policy_comprehensive/file.csv"
                base_path = str(Path(manifest_path).parent)  # polymorphic_data/
                file_paths = [f"{base_path}/{fname}" for fname in combined_df['filename']]

            # Clean file paths - remove any None/NaN/invalid entries
            clean_file_paths = [p for p in file_paths if p and str(p) != 'nan' and str(p).strip()]
            if len(clean_file_paths) != len(file_paths):
                logger.warning(f"Removed {len(file_paths) - len(clean_file_paths)} invalid file paths")
            return clean_file_paths
        else:
            return []

    def _add_market_date_ranges(self, df):
        """Add date range information for each market by analyzing CSV files"""
        enhanced_df = df.copy()

        date_ranges = []
        data_points = []
        volume_totals = []

        for idx, row in enhanced_df.iterrows():
            filename = row['filename']
            try:
                # Try to find and read the market data CSV
                market_path = None
                if 'polymarket_data' in filename:
                    market_path = Path(filename)
                else:
                    # Try searching in common directories
                    possible_paths = [
                        Path('polymarket_data') / filename,
                        Path('polymarket_data/geopolitics__foreign_policy_comprehensive') / filename,
                        Path('polymarket_data/geopolitics__war') / filename,
                        Path('polymarket_data/custom_world_affairs') / filename
                    ]
                    for path in possible_paths:
                        if path.exists():
                            market_path = path
                            break

                if market_path and market_path.exists():
                    market_df = pd.read_csv(market_path, parse_dates=['timestamp'])
                    if not market_df.empty and 'timestamp' in market_df.columns:
                        min_date = market_df['timestamp'].min()
                        max_date = market_df['timestamp'].max()
                        date_ranges.append(f"{min_date.date()} to {max_date.date()}")
                        data_points.append(len(market_df))
                        volume_totals.append(row.get('volume', 0))
                    else:
                        date_ranges.append("No valid data")
                        data_points.append(0)
                        volume_totals.append(0)
                else:
                    date_ranges.append("File not found")
                    data_points.append(0)
                    volume_totals.append(0)

            except Exception as e:
                date_ranges.append(f"Error: {str(e)}")
                data_points.append(0)
                volume_totals.append(0)

        enhanced_df['date_range'] = date_ranges
        enhanced_df['data_points'] = data_points
        enhanced_df['volume_total'] = volume_totals

        return enhanced_df

    def _save_filtered_manifest(self, df):
        """Save the filtered manifest with statistics and date ranges"""
        output_dir = Path("test_signals")
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"filtered_manifest_{timestamp}.csv"

        # Save the filtered manifest
        df.to_csv(output_file, index=False)
        logger.info(f"💾 Saved filtered manifest to {output_file}")

        # Save filtering statistics
        stats_file = output_dir / f"filtering_stats_{timestamp}.yaml"
        with open(stats_file, 'w') as f:
            yaml.dump(self.filtering_stats, f, default_flow_style=False)
        logger.info(f"💾 Saved filtering statistics to {stats_file}")

        # Print comprehensive statistics
        self._print_filtering_summary(df)

        self.manifest_outputs.append({
            'manifest_file': str(output_file),
            'stats_file': str(stats_file),
            'timestamp': timestamp
        })

    def _print_filtering_summary(self, final_df):
        """Print comprehensive filtering summary"""
        print("\n" + "="*80)
        print("📊 MANIFEST FILTERING STATISTICS")
        print("="*80)

        print("\n📈 OVERVIEW:")
        print(f"  Original markets: {self.filtering_stats['original_count']}")
        print(f"  Filtered markets: {self.filtering_stats['final_count']}")
        print(f"  Total excluded: {self.filtering_stats['total_excluded']}")

        print()
        print("🔍 FILTRATION BREAKDOWN:")
        for category, stats in self.filtering_stats['filter_categories'].items():
            print(f"\n  {category.upper()}:")
            print(f"    Excluded: {stats['excluded']}")
            if stats['reasons']:
                print("    Reasons:")
                for reason, count in stats['reasons'].items():
                    print(f"      • {reason}: {count}")

        # Final market statistics
        if len(final_df) > 0:
            print()
            print("📊 FINAL MARKET STATISTICS:")
            total_volume = final_df['volume_total'].sum()
            avg_volume = final_df['volume_total'].mean()
            median_volume = final_df['volume_total'].median()

            print(f"  Markets with data: {len(final_df[final_df['data_points'] > 0])}")
            print(f"  Markets without data: {len(final_df[final_df['data_points'] == 0])}")
            print(f"  Total volume: {total_volume:,.2f}")
            print(f"  Avg volume: {avg_volume:,.0f}")
            print(f"  Median volume: {median_volume:,.0f}")

            # Date range summary
            date_range_mask = (final_df['date_range'] != 'File not found') & (final_df['date_range'] != 'No valid data')
            valid_ranges = final_df[date_range_mask]
            if len(valid_ranges) > 0:
                date_ranges = valid_ranges['date_range']
                print()
                print("📅 DATE RANGES:")
                earliest = sorted(date_ranges)[0].split(' to ')[0] if len(date_ranges) > 0 else 'N/A'
                latest = sorted(date_ranges)[-1].split(' to ')[-1] if len(date_ranges) > 0 else 'N/A'
                print(f"  Earliest start: {earliest}")
                print(f"  Latest end: {latest}")

        print(f"\n💾 Results timestamped: {self.filtering_stats['timestamp']}")
        print("="*80)

    def get_stats_summary(self):
        """Return filtering statistics for external use"""
        return self.filtering_stats
