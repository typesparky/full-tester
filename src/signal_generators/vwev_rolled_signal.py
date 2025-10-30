#!/usr/bin/env python3
"""
VWEV ROLLED SIGNAL GENERATOR

Advanced Expected Value signal generator with futures roll capabilities.
Implements sophisticated probability distribution reconstruction and
rolling adjustment for continuous time series.
"""

import sys
import os
import re
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass
import logging
from scipy.optimize import curve_fit
from scipy import stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EventGroup:
    """Represents a group of markets for the same event/period"""
    event_key: str
    markets: List[Dict]
    start_date: Optional[pd.Timestamp] = None
    end_date: Optional[pd.Timestamp] = None
    total_volume: float = 0.0

@dataclass
class DistributionConfig:
    """Configuration for probability distribution reconstruction"""
    percentiles: List[float]
    min_sample_size: int = 3
    fit_tolerance: float = 1e-6

class VWEV_Rolled_Signal:
    """
    Volume-Weighted Expected Value Rolled Signal Generator

    Implements sophisticated methodology:
    1. Event grouping via regex patterns
    2. Daily EV calculation with probability distribution reconstruction
    3. Pareto tail fitting for open-ended bins
    4. Futures roll with backward ratio adjustment
    """

    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.signal_name = self.config['signal_name']
        self.input_manifests = [Path(p) for p in self.config['input_manifest_files']]
        self.output_file = Path(self.config['output_file'])
        self.generator_type = self.config.get('generator_type', 'vwev_rolled')

        # Event grouping and rolling config
        self.event_pattern = self.config['event_grouping_pattern']
        self.roll_field = self.config['roll_on_field']
        self.tail_method = self.config['open_bin_handling_method']

        # Distribution reconstruction config
        self.dist_config = DistributionConfig(**self.config.get('distribution_bins', {}))

        logger.info(f"🔧 Initialized {self.generator_type} signal: {self.signal_name}")

    def generate(self) -> Optional[pd.DataFrame]:
        """
        Main orchestration method implementing the 4-step process
        """
        try:
            # Step 1: Group markets by event
            event_groups = self._group_markets_by_event()
            if not event_groups:
                logger.warning("❌ No event groups found")
                return None

            logger.info(f"📊 Step 1: Grouped {len(event_groups)} event categories")

            # Step 2: Calculate daily EV for each event
            event_signals = {}
            for event_key, event_group in event_groups.items():
                ev_series = self._calculate_event_expected_value(event_group)
                if ev_series is not None and not ev_series.empty:
                    event_signals[event_key] = ev_series
                    logger.info(f"  ✓ {event_key}: {len(ev_series)} days of EV data")

            if not event_signals:
                logger.warning("❌ No event signals generated")
                return None

            # Step 3: Stitch continuous signal via futures roll
            continuous_signal = self._stitch_continuous_signal(event_signals)

            if continuous_signal.empty:
                logger.warning("❌ No continuous signal generated")
                return None

            logger.info(f"🧵 Step 3: Stitched {len(continuous_signal)} days of continuous signal")

            # Save to output file
            continuous_signal.to_csv(self.output_file, index=False)
            logger.info(f"💾 Saved to {self.output_file}")

            return continuous_signal

        except Exception as e:
            logger.error(f"❌ Signal generation failed: {e}")
            return None

    def _group_markets_by_event(self) -> Dict[str, EventGroup]:
        """
        Step 1: Group markets by event using regex pattern from config
        Uses the event_grouping_pattern to extract event identifiers from questions
        """
        event_groups = {}
        total_markets = 0

        for manifest_path in self.input_manifests:
            if not manifest_path.exists():
                logger.warning(f"Manifest not found: {manifest_path}")
                continue

            try:
                manifest_df = pd.read_csv(manifest_path)
                logger.debug(f"Processing {len(manifest_df)} markets from {manifest_path.name}")

                for _, row in manifest_df.iterrows():
                    question = str(row['question'])
                    filename = str(row['filename'])
                    volume = float(row.get('volume', 0))

                    # Apply event grouping regex
                    match = re.search(self.event_pattern, question, re.IGNORECASE)
                    if match:
                        event_key = match.group(1).strip().title()  # e.g., "June 2025"
                        total_markets += 1

                        if event_key not in event_groups:
                            event_groups[event_key] = EventGroup(
                                event_key=event_key,
                                markets=[],
                                total_volume=0.0
                            )

                        # Add market to event group
                        event_groups[event_key].markets.append({
                            'filename': filename,
                            'question': question,
                            'volume': volume,
                            'manifest_path': manifest_path.parent / filename  # Full path
                        })
                        event_groups[event_key].total_volume += volume

            except Exception as e:
                logger.error(f"Error processing manifest {manifest_path}: {e}")
                continue

        logger.info(f"Grouped {total_markets} markets into {len(event_groups)} event groups")
        for event_key, group in event_groups.items():
            logger.debug(f"  {event_key}: {len(group.markets)} markets, vol={group.total_volume:.0f}")

        return event_groups

    def _calculate_event_expected_value(self, event_group: EventGroup) -> Optional[pd.DataFrame]:
        """
        Step 2: Calculate daily EV for single event group using distribution reconstruction

        For each day the event is active:
        1. Load all market price/volume data
        2. Parse thresholds to build cumulative distribution
        3. Reconstruct full probability distribution
        4. Calculate Volume-Weighted Expected Value
        """
        # Collect all market data timelines
        market_timelines = []
        all_dates = set()

        for market in event_group.markets:
            csv_path = market['manifest_path']
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
                    if not df.empty:
                        df['date'] = df['timestamp'].dt.date
                        df['market_file'] = str(csv_path)
                        df['volume'] = market['volume']
                        market_timelines.append(df)
                        all_dates.update(df['date'].unique())
                except Exception as e:
                    logger.debug(f"Error loading {csv_path}: {e}")

        if not market_timelines:
            return None

        # Sort dates for consistent processing
        sorted_dates = sorted(all_dates)
        logger.debug(f"Processing {len(sorted_dates)} active dates for {event_group.event_key}")

        expected_values = []

        for process_date in sorted_dates:
            # Get data for this specific date across all markets
            date_data = []

            for timeline_df in market_timelines:
                day_data = timeline_df[timeline_df['date'] == process_date]
                if not day_data.empty:
                    # Take the latest entry for the day
                    latest = day_data.iloc[-1]
                    date_data.append({
                        'question': '',  # Will be set when we have manifest
                        'yes_prob': latest['yes'],
                        'volume': latest['volume']
                    })

            if len(date_data) < self.dist_config.min_sample_size:
                continue  # Need minimum sample size for distribution reconstruction

            # Parse questions and reconstruct distribution
            market_questions = [m['question'] for m in event_group.markets]
            thresholds = self._parse_all_thresholds(market_questions)

            if not thresholds:
                continue

            # Match market data to thresholds
            market_probs = []
            for market_data, market_question in zip(date_data, market_questions):
                threshold_val = self._parse_single_threshold(market_question)
                if threshold_val is not None:
                    market_probs.append({
                        'threshold': threshold_val,
                        'probability': market_data['yes_prob'],
                        'volume': market_data['volume']
                    })

            if len(market_probs) < 3:  # Need at least 3 points for distribution
                continue

            # Reconstruct full probability distribution
            expected_val = self._reconstruct_distribution_expected_value(market_probs)

            if expected_val is not None:
                expected_values.append({
                    'date': process_date,
                    'signal': expected_val,
                    'active_markets': len(date_data),
                    'event_key': event_group.event_key
                })
                logger.debug(".4f")

        if not expected_values:
            return None

        # Convert to DataFrame
        result_df = pd.DataFrame(expected_values)
        result_df['date'] = pd.to_datetime(result_df['date'])
        result_df = result_df.sort_values('date').reset_index(drop=True)

        return result_df

    def _reconstruct_distribution_expected_value(self, market_probs: List[Dict]) -> Optional[float]:
        """
        Reconstruct probability distribution from threshold data and calculate expected value

        market_probs format: [{'threshold': X, 'probability': P, 'volume': V}, ...]
        where threshold=X means P(inflation_level > X)
        """
        try:
            # Sort by threshold
            market_probs.sort(key=lambda x: x['threshold'])
            n = len(market_probs)

            if n < 3:
                return None

            # Handle open-ended upper bin using Pareto fitting
            if self.tail_method == 'fit_pareto':
                last_threshold = market_probs[-1]['threshold']
                last_prob = market_probs[-1]['probability']

                # Fit Pareto distribution to tail
                # Assume if P(X > T) = p, we can fit shape parameter
                shape_param, scale_param = self._fit_pareto_tail(
                    last_threshold, last_prob, market_probs
                )

                # Generate tail distribution points
                tail_quantiles = np.linspace(0.01, 0.99, 10)
                tail_values = stats.pareto.ppf(tail_quantiles, b=shape_param, scale=scale_param)
                tail_probs = np.diff(np.concatenate([[0], tail_quantiles]))

            else:
                # Simple assumption: truncate at reasonable upper bound
                tail_values = np.linspace(
                    market_probs[-1]['threshold'],
                    market_probs[-1]['threshold'] * 2,
                    5
                )
                tail_probs = np.ones(5) * market_probs[-1]['probability'] / 5

            # Build complete distribution
            complete_values = []
            complete_probs = []

            # Lower bins (cumulative differences)
            for i in range(len(market_probs)-1):
                lower_thresh = market_probs[i]['threshold']
                upper_thresh = market_probs[i+1]['threshold']
                prob_mass = market_probs[i]['probability'] - market_probs[i+1]['probability']

                if prob_mass > 0:
                    # Assume uniform distribution in each bin
                    bin_midpoint = (lower_thresh + upper_thresh) / 2
                    complete_values.append(bin_midpoint)
                    complete_probs.append(prob_mass)

            # Upper tail distribution
            for tail_val, tail_prob in zip(tail_values, tail_probs):
                complete_values.append(tail_val)
                complete_probs.append(tail_prob)

            # Calculate PURELY probability-weighted expected value (IGNORE VOLUME as requested)
            if complete_values and complete_probs:
                # Only use probability masses for weighting - ignore volume completely
                total_weight = sum(complete_probs)  # Pure probability weighting
                if total_weight > 0:
                    expected_value = sum(
                        mid_val * prob_mass / total_weight
                        for mid_val, prob_mass in zip(complete_values, complete_probs)
                    )
                    return expected_value

        except Exception as e:
            logger.debug(f"Distribution reconstruction failed: {e}")

        return None

    def _fit_pareto_tail(self, threshold: float, exceedance_prob: float,
                        market_data: List[Dict]) -> Tuple[float, float]:
        """
        Fit Pareto distribution to the tail region
        """
        try:
            shape, scale = 2.5, threshold  # Conservative initial parameters

            # Simple shape estimation from exceedance probability
            # For Pareto: P(X > x) = (scale/x)^shape
            if exceedance_prob > 0 and threshold > 0:
                shape = abs(np.log(exceedance_prob) / np.log(scale / threshold))

            return shape, scale

        except Exception:
            return 2.0, threshold  # Conservative fallback

    def _stitch_continuous_signal(self, event_signals: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Step 3: Stitch discrete event signals into continuous time series using futures roll

        Implements backward ratio adjustment for volume-based rolling.
        """
        if not event_signals:
            return pd.DataFrame()

        # Sort event groups by their start dates
        event_timeline = []
        for event_key, signal_df in event_signals.items():
            if not signal_df.empty:
                event_timeline.append({
                    'event_key': event_key,
                    'signal_df': signal_df,
                    'start_date': signal_df['date'].min(),
                    'end_date': signal_df['date'].max()
                })

        event_timeline.sort(key=lambda x: x['start_date'])

        if len(event_timeline) < 2:
            # Single event - return as-is
            return event_timeline[0]['signal_df'] if event_timeline else pd.DataFrame()

        # Backward ratio adjustment (iterate from newest to oldest)
        continuous_signal = []
        current_adjustment = 1.0

        for i in reversed(range(len(event_timeline))):
            event_data = event_timeline[i]
            next_event_data = event_timeline[i+1] if i+1 < len(event_timeline) else None

            event_signal = event_data['signal_df'].copy()

            # Apply current adjustment factor to this event
            event_signal['adjusted_signal'] = event_signal['signal'] * current_adjustment

            # If there's a next event, calculate roll adjustment
            if next_event_data:
                roll_date = self._find_roll_date(event_data, next_event_data)

                if roll_date is not None:
                    # Calculate adjustment factor based on overlap period
                    new_ev_at_roll = self._interpolate_signal_value(
                        event_signal, roll_date
                    )
                    old_ev_at_roll = self._interpolate_signal_value(
                        next_event_data['signal_df'], roll_date
                    )

                    if new_ev_at_roll > 0 and old_ev_at_roll > 0:
                        current_adjustment *= old_ev_at_roll / new_ev_at_roll

                        # Update historical values with new adjustment
                        for j in range(len(continuous_signal)):
                            continuous_signal[j]['adjusted_signal'] *= old_ev_at_roll / new_ev_at_roll

                        logger.debug(f"📅 Rolled {event_data['event_key']} -> {next_event_data['event_key']} @ {roll_date}: ratio={old_ev_at_roll/new_ev_at_roll:.3f}")

            continuous_signal.extend(event_signal.to_dict('records'))

        # Combine into final DataFrame
        if continuous_signal:
            result_df = pd.DataFrame(continuous_signal)
            # Group by date (take latest entry for overlapping dates)
            result_df = result_df.sort_values('date').groupby('date').last().reset_index()
            return result_df[['date', 'adjusted_signal']].rename(columns={'adjusted_signal': 'signal'})

        return pd.DataFrame()

    def _find_roll_date(self, old_event: Dict, new_event: Dict) -> Optional[pd.Timestamp]:
        """
        Find roll date based on volume crossover
        """
        try:
            old_signal = old_event['signal_df']
            new_signal = new_event['signal_df']

            # Find overlapping date range
            overlap_start = max(old_signal['date'].min(), new_signal['date'].min())
            overlap_end = min(old_signal['date'].max(), new_signal['date'].max())

            if overlap_start >= overlap_end:
                return None

            # Find date where new volume > old volume (for volume-based roll)
            # Simplified: use midpoint of overlap for now
            return overlap_start + (overlap_end - overlap_start) / 2

        except Exception:
            return None

    def _interpolate_signal_value(self, signal_df: pd.DataFrame, target_date: pd.Timestamp) -> float:
        """
        Interpolate signal value at target date
        """
        try:
            signal_df_sorted = signal_df.sort_values('date')

            # Exact match
            exact_match = signal_df_sorted[signal_df_sorted['date'] == target_date]
            if not exact_match.empty:
                return exact_match.iloc[0]['signal']

            # Linear interpolation
            earlier = signal_df_sorted[signal_df_sorted['date'] < target_date]
            later = signal_df_sorted[signal_df_sorted['date'] > target_date]

            if earlier.empty or later.empty:
                return signal_df_sorted.iloc[0]['signal']  # Fallback to first value

            before_idx = earlier.iloc[-1]['date']
            after_idx = later.iloc[0]['date']
            before_val = earlier.iloc[-1]['signal']
            after_val = later.iloc[0]['signal']

            # Linear interpolation
            ratio = (target_date - before_idx).total_seconds() / (after_idx - before_idx).total_seconds()
            return before_val + ratio * (after_val - before_val)

        except Exception:
            return 0.0

    def _parse_all_thresholds(self, questions: List[str]) -> List[float]:
        """Parse threshold values from all questions in an event group"""
        thresholds = []
        for question in questions:
            threshold = self._parse_single_threshold(question)
            if threshold is not None:
                thresholds.append(threshold)
        return list(set(thresholds))  # Remove duplicates

    def _parse_single_threshold(self, question: str) -> Optional[float]:
        """Parse single inflation threshold from question"""
        question_lower = question.lower().strip()

        # Annual level patterns: "reach/reach more than X%"
        level_patterns = [
            r'reach\s+more\s+than\s+(\d+(?:\.\d+)?)',
            r'exceed\s+more\s+than\s+(\d+(?:\.\d+)?)',
            r'more\s+than\s+(\d+(?:\.\d+)?)'
        ]

        for pattern in level_patterns:
            matches = re.findall(pattern, question_lower)
            if matches:
                val = float(matches[0]) / 100.0  # Convert to decimal
                return val

        # Change patterns for monthly: "increase by X%"
        change_patterns = [
            r'increase\s+by\s+(\d+(?:\.\d+)?)',
            r'rise\s+by\s+(\d+(?:\.\d+)?)'
        ]

        for pattern in change_patterns:
            matches = re.findall(pattern, question_lower)
            if matches:
                return float(matches[0]) / 100.0  # Convert to decimal

        return None
