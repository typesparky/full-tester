#!/usr/bin/env python3
"""
VWP SIGNAL GENERATOR

Generates Volume-Weighted Probability signals with configurable polarity rules.
Supports both simple VWP aggregation and polarity-adjusted uncertainty signals.
"""

import sys
import os
import re
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PolarityRules:
    """Configuration for polarity classification (+1/-1 rules)"""
    increase_uncertainty: List[str]
    decrease_uncertainty: List[str]
    concept_description: str = ""
    market_polarity_overrides: Optional[Dict[str, int]] = None

    def classify_polarity(self, question: str) -> int:
        """
        Classify market polarity based on question keywords.

        Returns:
            +1: Increases uncertainty (escalation, risk)
            -1: Decreases uncertainty (de-escalation, stability)
             0: Unclear/unknown polarity
        """
        question_lower = question.lower()

        # Check for escalation/increase uncertainty keywords (regex matching)
        for keyword in self.increase_uncertainty:
            try:
                if re.search(keyword, question_lower, re.IGNORECASE):
                    return 1
            except re.error:
                # Fallback to substring match if regex fails
                if keyword.lower() in question_lower:
                    return 1

        # Check for de-escalation/decrease uncertainty keywords (regex matching)
        for keyword in self.decrease_uncertainty:
            try:
                if re.search(keyword, question_lower, re.IGNORECASE):
                    return -1
            except re.error:
                # Fallback to substring match if regex fails
                if keyword.lower() in question_lower:
                    return -1

        # Check for market-specific overrides
        if self.market_polarity_overrides:
            # Could match on filename or other identifiers
            pass

        return 0  # Unclear polarity

class VWP_Signal:
    """
    Volume-Weighted Probability Signal Generator

    Supports polarity-adjusted signals for uncertainty modeling.
    Reads input markets from specified directories and applies
    configurable polarity rules for +1/-1 classification.
    """

    def __init__(self, filtered_file_paths: Optional[List[str]] = None,
                 input_dirs: List[str] = None, polarity_rules: Dict = None,
                 output_file: str = None, min_volume: float = 100.0,
                 min_quality_score: int = 60):
        if filtered_file_paths is not None:
            self.filtered_file_paths = [Path(p) for p in filtered_file_paths]
            self.input_dirs = []  # Not used when filtered paths provided
            self.use_filtered_paths = True
            logger.info(f"Using {len(filtered_file_paths)} pre-filtered file paths")
        else:
            # Legacy mode: scan directories
            self.input_dirs = [Path(d) for d in (input_dirs or [])]
            self.filtered_file_paths = []
            self.use_filtered_paths = False
            logger.info(f"Using {len(self.input_dirs)} input directories (legacy mode)")

        self.polarity_rules = PolarityRules(**polarity_rules) if polarity_rules else PolarityRules(concept_description="", increase_uncertainty=[], decrease_uncertainty=[])
        self.output_file = Path(output_file) if output_file else None
        self.min_volume = min_volume
        self.min_quality_score = min_quality_score

        if self.output_file:
            self.output_file.parent.mkdir(parents=True, exist_ok=True)

        if not self.use_filtered_paths:
            # Legacy pre-filtering for backwards compatibility
            self.pre_filtered_markets = self._pre_filter_manifests()
            logger.info(f"🚀 Pre-filtered to {len(self.pre_filtered_markets)} high-quality markets from manifests")

    def load_market_data(self, min_quality_score=60) -> pd.DataFrame:
        """
        Load and aggregate market data from filtered file paths or input directories.
        Applies basic filtering and polarity classification.

        Args:
            min_quality_score: Minimum quality score for market inclusion (default 60)
        """
        all_market_data = []

        if self.use_filtered_paths:
            logger.info("Loading market data from filtered file paths...")
            market_files = self.filtered_file_paths
        else:
            logger.info("Loading market data from input directories (legacy mode)...")
            market_files = []
            for input_dir in self.input_dirs:
                if not input_dir.exists():
                    logger.warning(f"Input directory not found: {input_dir}")
                    continue

                # Load market files (CSV files)
                market_files.extend(list(input_dir.glob("*.csv")))
                logger.info(f"Found {len(market_files)} market files in {input_dir}")

                # Filter to pre-approved if using legacy mode
                if not self.use_filtered_paths:
                    market_files = [f for f in market_files if str(f) in self.pre_filtered_markets]
                    logger.info(f"After pre-filtering: {len(market_files)} approved markets")

        logger.info(f"Processing {len(market_files)} market files")

        for market_file in market_files:
            try:
                if not self.use_filtered_paths and str(market_file) not in self.pre_filtered_markets:
                    continue  # Skip in legacy mode

                # Load market data
                market_df = pd.read_csv(market_file, parse_dates=['timestamp'])

                if market_df.empty:
                    continue

                # Skip markets with very low data points (but allow closed/historical markets)
                # Reduce threshold for single-market signals - allow markets with any data
                if len(market_df) < 1:  # Only skip completely empty markets
                    continue

                # Extract question: from manifest if available, otherwise filename
                if 'question' in market_df.columns and not market_df['question'].isna().all():
                    question = market_df['question'].iloc[0]
                else:
                    question = self._filename_to_question(market_file.name)

                # Classify polarity
                polarity = self.polarity_rules.classify_polarity(question)
                logger.debug(f"Market: {market_file.name} -> Question: '{question}' -> Polarity: {polarity}")

                # Skip markets with unclear polarity (could be filtered differently)
                if polarity == 0:
                    logger.debug(f"    └─ Skipping unclear polarity market")
                    continue

                # Add metadata to dataframe
                market_df['market_file'] = str(market_file)
                market_df['question'] = question
                market_df['polarity'] = polarity

                # Calculate aligned prices (P' = P_yes if polarity=1, 1-P_yes if polarity=-1)
                market_df['aligned_price'] = np.where(
                    market_df['polarity'] == 1,
                    market_df['yes'],
                    1.0 - market_df['yes']
                )

                # Convert timestamp to date for daily aggregation
                market_df['date'] = pd.to_datetime(market_df['timestamp']).dt.date

                all_market_data.append(market_df)

                logger.debug(f"Loaded market: {market_file.name} (Polarity: {polarity})")

            except Exception as e:
                logger.warning(f"Error loading market {market_file}: {e}")
                continue

        if not all_market_data:
            logger.warning("No valid market data found")
            return pd.DataFrame()

        # Combine all market data
        combined_df = pd.concat(all_market_data, ignore_index=True)
        logger.info(f"Total market records loaded: {len(combined_df)}")
        logger.info(f"Unique markets: {combined_df['market_file'].nunique()}")

        return combined_df

    def generate_daily_signal(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate daily aggregated VWP signal.
        VWP = Σ(aligned_price_i * volume_factor_i) / Σ(volume_factor_i)
        """
        logger.info("Generating daily VWP signals...")

        # Create volume weighting (using record count as proxy for volume)
        # In real implementation, would use actual traded volume
        market_data['volume_weight'] = market_data.groupby('market_file')['market_file'].transform('count')

        # Get unique market-date combinations
        market_days = market_data[['market_file', 'date', 'aligned_price', 'volume_weight']].drop_duplicates()

        # Aggregate by date: VWP = Σ(aligned_price * weight) / Σ(weight)
        daily_signals = []

        date_range = market_data['date'].sort_values().unique()

        # Use ALL available dates (remove artificial limit for full testing)
        for date in date_range:
            day_data = market_days[market_days['date'] == date]

            if day_data.empty:
                continue

            # Volume-weighted VWP calculation
            total_weighted_price = (day_data['aligned_price'] * day_data['volume_weight']).sum()
            total_weight = day_data['volume_weight'].sum()

            if total_weight > 0:
                vwp_signal = total_weighted_price / total_weight

                daily_signals.append({
                    'date': date,
                    'signal': float(vwp_signal),
                    'active_markets': len(day_data),
                    'total_weight': total_weight
                })

        signal_df = pd.DataFrame(daily_signals)
        logger.info(f"Generated {len(signal_df)} days of VWP signals")

        return signal_df

    def generate(self) -> Optional[pd.DataFrame]:
        """
        Main generation method.
        Returns the final signal DataFrame.
        """
        try:
            # Load and process market data
            market_data = self.load_market_data(self.min_quality_score)

            if market_data.empty:
                logger.error("No market data available for signal generation")
                return None

            # Generate daily signals
            signal_df = self.generate_daily_signal(market_data)

            if signal_df.empty:
                logger.error("No signal data generated")
                return None

            logger.info("✅ Signal generation completed successfully")
            return signal_df

        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            return None

    def save_signal(self, signal_df: pd.DataFrame):
        """Save the generated signal to output file"""
        if signal_df is not None and not signal_df.empty:
            signal_df.to_csv(self.output_file, index=False)
            logger.info(f"💾 Saved signal to {self.output_file}")

    def _filename_to_question(self, filename: str) -> str:
        """
        Extract question from filename by cleaning up common patterns.
        Keep the question readable for polarity matching.
        """
        # Remove file extension
        name = filename.replace('.csv', '')

        # Replace underscores with spaces
        name = name.replace('_', ' ')

        # Remove trailing numbers (like market IDs) but keep dates if part of question
        # Remove patterns that look like " 12345" at end
        name = re.sub(r'\s+\d{4,}$', '', name)

        return name.strip()

    def _pre_filter_manifests(self) -> Set[str]:
        """
        Pre-filter manifests using quality scoring before CSV parsing.
        Uses identical logic to interactive_market_reviewer.py for consistency.

        Returns set of approved market file paths (str) for CSV parsing.
        """
        logger.info("🚀 Pre-filtering manifests for high-quality geopolitics markets...")
        approved_files = set()

        # First build complete set of available CSV files
        available_csv_files = set()
        for input_dir in self.input_dirs:
            for csv_file in input_dir.glob("*.csv"):
                available_csv_files.add(str(csv_file))

        # Scan manifest files in geopolitics directories
        manifest_patterns = ["_manifest_geopolitics__", "_manifest_custom_world_affairs"]

        for input_dir in self.input_dirs:
            parent_dir = input_dir.parent

            for manifest_file in parent_dir.glob("*_manifest_*.csv"):
                # Only process geopolitics manifests
                if not any(pattern in manifest_file.name for pattern in manifest_patterns):
                    continue

                try:
                    manifest_df = pd.read_csv(manifest_file)
                    logger.debug(f"Processing manifest: {manifest_file.name} ({len(manifest_df)} markets)")

                    for _, row in manifest_df.iterrows():
                        question = str(row['question']).strip()
                        volume = float(row.get('volume', 0))
                        manifest_filename = str(row.get('filename', '')).strip()

                        if not manifest_filename:
                            continue

                        # Apply quality scoring (identical to interactive_review.ipynb)
                        score = self._score_market_quality(question, volume)

                        if score >= self.min_quality_score:  # Use configurable threshold
                            # Need to find the actual file path that matches this manifest entry
                            # The manifest filename is relative, so try to find the matching file
                            for available_file in available_csv_files:
                                if available_file.endswith(f"/{manifest_filename}"):
                                    approved_files.add(available_file)
                                    logger.debug(f"✅ APPROVED: {manifest_filename} (score: {score})")
                                    break
                        else:
                            logger.debug(f"❌ FILTERED: {manifest_filename} (score: {score})")

                except Exception as e:
                    logger.warning(f"Error processing manifest {manifest_file}: {e}")
                    continue

        logger.info(f"Manifest pre-filtering complete: {len(approved_files)} markets approved out of {len(available_csv_files)} available")
        return approved_files

    def _score_market_quality(self, question: str, volume: float) -> int:
        """
        Score market quality using identical logic to interactive_market_reviewer.py
        Returns 0-100 quality score.
        """
        score = 0
        flags = []
        question_lower = question.lower()

        # === SUBSTANTIVE VS RHETORICAL (0-25) ===
        substantive_score = 25

        # Check for irrelevant/ironic content (like Nobel Peace Prize wars which are not real geopolitical signals)
        irrelevant_patterns = [
            (r'nobel\s+peace\s+prize', "nobel peace prize market"),  # Nobel Peace Prize markets are irrelevant for geopolitics
            (r'celebrity.*out', "celebrity leadership change"),     # Celebrity politics (not real)
            (r'rapper.*president', "entertainment politics"),       # Rapper politics
            (r'out\s+in\s+2025', "specific year leader change"),    # Generic "out in 2025" predictions
            (r'be\s+the\s+first\s+leader\s+out\s+in\s+\d{4}', "first leader out pattern"),  # "first leader out" pattern
        ]

        # Check for placeholder/template markets - zero score immediately
        placeholder_patterns = [
            (r'\b(person|party)\s+[a-z]\b', "person/party template"),  # Person A, Party B
            (r'\bcountry\s+([a-z]{1}|[a-z]{3})\b', "country template"), # Country X, Country ABC
            (r'\b(candidate|option)\s+[a-z0-9]+\b', "candidate/option template"), # Candidate 1, Option A
            (r'\b(person|party)\s+[a-z]\b.*\b(win|election)', "person/party template - election"), # Any person/party template in election context
        ]

        # Also check for specific template phrases we've seen
        template_phrases = [
            "will another person win",
            "will another party",
            "hold the second most seats",
            "win 2nd place",
            "win 2nd place",
            "person ",
            "party "
        ]

        # First check irrelevant content
        for pattern, flag_reason in irrelevant_patterns:
            if re.search(pattern, question_lower, re.IGNORECASE):
                flags.append(f"Irrelevant: {flag_reason}")
                substantive_score = 0  # Irrelevant markets get zero score
                break
        else:
            # Then check placeholder templates
            for pattern, flag_reason in placeholder_patterns:
                if re.search(pattern, question_lower, re.IGNORECASE):
                    flags.append(f"Template: {flag_reason}")
                    substantive_score = 0  # Template markets get zero score
                    break
            else:
                # Check for template phrases if regex didn't match
                for phrase in template_phrases:
                    if phrase in question_lower and len(question_lower.split()) < 15:  # Short template questions
                        flags.append("Template: placeholder phrase")
                        substantive_score = 0  # Template markets get zero score
                        break

                # Multi-pass pattern detection for speech markets - ZERO SCORE FOR SPEECH
                speech_patterns = [
                    (r"\bwill\s+\w+\s+say\s+", "speech pattern"),
                    (r"\bsay\s+['\"]?[^'\"]*(['\"])?\s+during\s+", "speech during event"),
                    (r"\bsay\s+['\"]?[^'\"]*['\"]?\s+times?\b", "word count pattern"),
                    (r"\bmention\b", "mention pattern"),
                    (r"\bphrase\b", "phrase pattern"),
                ]

                speech_detected = False
                for pattern, flag_reason in speech_patterns:
                    if re.search(pattern, question_lower, re.IGNORECASE):
                        flags.append(f"Rhetorical: {flag_reason}")
                        speech_detected = True
                        substantive_score = 0  # ZERO SCORE FOR SPEECH
                        break

        # If speech detected, force zero score
        if substantive_score == 0:
            return 0

        # === POLITICAL IMPACT (0-25) ===
        high_impact = ["election", "wins", "policy", "war", "sanctions", "government"]
        impact_score = min(25, sum(1 for word in high_impact if word in question_lower) * 5)

        # === VOLUME VALIDATION (0-25) ===
        volume_score = min(25, volume / 1000)

        # === SIGNAL RELEVANCE (0-25) ===
        relevance_score = 25

        # Calculate overall (but don't override forced zero for speech markets)
        score = substantive_score + impact_score + volume_score + relevance_score

        return min(100, score)  # Cap at 100

    def analyze_signal_quality(self, signal_df: pd.DataFrame) -> Dict:
        """
        Basic analysis of signal characteristics.
        """
        if signal_df.empty:
            return {}

        stats = {
            'total_days': len(signal_df),
            'signal_range': [signal_df['signal'].min(), signal_df['signal'].max()],
            'avg_signal': signal_df['signal'].mean(),
            'signal_volatility': signal_df['signal'].std(),
            'avg_active_markets': signal_df['active_markets'].mean(),
            'valid_data_ratio': len(signal_df) / len(signal_df.dropna())
        }

        return stats
