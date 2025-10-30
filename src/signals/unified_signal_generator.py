# UNIFIED SIGNAL GENERATION METHODOLOGY
# Implementation of the 3-stage process per research validation
# This file contains the core classes for the unified methodology

import pandas as pd
import numpy as np
import re
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

class MarketType(Enum):
    BINARY = "binary"
    QUANTITATIVE = "quantitative"

class SignalPolarity(Enum):
    PRO_CONCEPT = 1    # λ_i = +1: P_yes indicates positive outcome for concept
    ANTI_CONCEPT = -1  # λ_i = -1: P_yes indicates negative outcome for concept

@dataclass
class SignalConcept:
    """Defines a signal concept (e.g., 'US Inflation Pressure') and rules for markets

    Args:
        name: Human-readable concept name
        keywords: List of keywords that map markets to this concept
        market_type: Whether markets in this concept are binary or quantitative
        polarity_logic: Rules for determining if P_yes is pro-concept or anti-concept
        parser_func: Function to parse quantitative values from questions
        min_volume: Minimum volume threshold for market inclusion
        manual_excludes: List of market filenames to manually exclude
        manual_includes: List of market filenames to manually include
        bert_model: Optional BERT model for advanced classification (Tier 2)
    """
    name: str
    keywords: List[str]
    market_type: MarketType
    polarity_logic: Optional[Dict[str, SignalPolarity]] = None  # keyword -> polarity mapping
    parser_func: Optional[callable] = None
    min_volume: float = 1000.0
    manual_excludes: List[str] = None
    manual_includes: List[str] = None
    bert_model: Optional[str] = None  # Future: BERT model path

@dataclass
class AlignedMarket:
    """Represents a market after Stage 1 alignment with polarity scalar"""
    filename: str
    question: str
    volume: float
    market_type: MarketType
    lambda_scalar: int
    expected_value: Optional[float] = None  # Only for quantitative markets
    aligned_price: Optional[float] = None   # P'_i,t = P_i,t or 1-P_i,t

    @property
    def is_valid(self) -> bool:
        """Check if market has all required data for aggregation"""
        return (self.aligned_price is not None and
                self.volume > 0 and
                (self.market_type == MarketType.BINARY or self.expected_value is not None))

class UnifiedSignalGenerator:
    """Implements the 3-stage unified methodology for robust signal generation"""

    def __init__(self, manifest_dir: str = "polymarket_data", debug: bool = True):
        self.manifest_dir = Path(manifest_dir)
        self.debug = debug
        self.concepts = self._define_signal_concepts()

        # Statistics for debugging/validation
        self.stage1_stats = {}
        self.stage2_stats = {}
        self.stage3_stats = {}

        # Manual flagging system
        self.flagged_markets = self._load_manual_flags()

        logger.info(f"Initialized UnifiedSignalGenerator with {len(self.concepts)} concepts")

    def _define_signal_concepts(self) -> Dict[str, SignalConcept]:
        """Define the canonical signal concepts as validated by research"""

        concepts = {
            'us_inflation': SignalConcept(
                name="US Inflation Pressure",
                keywords=['inflation', 'cpi', 'price increases', 'consumer price'],
                market_type=MarketType.QUANTITATIVE,
                polarity_logic={
                    'higher': SignalPolarity.PRO_CONCEPT,
                    'lower': SignalPolarity.ANTI_CONCEPT,
                    'rising': SignalPolarity.PRO_CONCEPT,
                    'falling': SignalPolarity.ANTI_CONCEPT,
                    'above': SignalPolarity.PRO_CONCEPT,
                    'below': SignalPolarity.ANTI_CONCEPT
                },
                parser_func=self._parse_inflation_value,
                min_volume=2000.0
            ),

            'us_inflation_monthly': SignalConcept(
                name="US Monthly Inflation",
                keywords=['monthly', 'inflation'],
                market_type=MarketType.QUANTITATIVE,
                polarity_logic={
                    'higher': SignalPolarity.PRO_CONCEPT,
                    'lower': SignalPolarity.ANTI_CONCEPT,
                    'rising': SignalPolarity.PRO_CONCEPT,
                    'falling': SignalPolarity.ANTI_CONCEPT,
                    'above': SignalPolarity.PRO_CONCEPT,
                    'below': SignalPolarity.ANTI_CONCEPT
                },
                parser_func=self._parse_monthly_inflation_value,
                min_volume=1000.0
            ),

            'us_inflation_annual': SignalConcept(
                name="US Annual Inflation",
                keywords=['annual', 'inflation', 'year'],
                market_type=MarketType.QUANTITATIVE,
                polarity_logic={
                    'higher': SignalPolarity.PRO_CONCEPT,
                    'lower': SignalPolarity.ANTI_CONCEPT,
                    'rising': SignalPolarity.PRO_CONCEPT,
                    'falling': SignalPolarity.ANTI_CONCEPT,
                    'above': SignalPolarity.PRO_CONCEPT,
                    'below': SignalPolarity.ANTI_CONCEPT
                },
                parser_func=self._parse_annual_inflation_value,
                min_volume=1000.0
            ),

            'fed_rates': SignalConcept(
                name="Fed Interest Rate Expectations",
                keywords=['fed', 'federal reserve', 'interest rate', 'fed funds', 'fomc'],
                market_type=MarketType.QUANTITATIVE,
                polarity_logic={
                    'increase': SignalPolarity.PRO_CONCEPT,
                    'decrease': SignalPolarity.ANTI_CONCEPT,
                    'higher': SignalPolarity.PRO_CONCEPT,
                    'lower': SignalPolarity.ANTI_CONCEPT,
                    'raise': SignalPolarity.PRO_CONCEPT,
                    'cut': SignalPolarity.ANTI_CONCEPT
                },
                parser_func=self._parse_fed_rate_value,
                min_volume=2000.0
            ),

            'us_gdp': SignalConcept(
                name="US Economic Growth",
                keywords=['gdp', 'economic growth', 'economy', 'recession'],
                market_type=MarketType.QUANTITATIVE,
                polarity_logic={
                    'growth': SignalPolarity.PRO_CONCEPT,
                    'contraction': SignalPolarity.ANTI_CONCEPT,
                    'expansion': SignalPolarity.PRO_CONCEPT,
                    'recession': SignalPolarity.ANTI_CONCEPT,
                    'higher': SignalPolarity.PRO_CONCEPT,
                    'lower': SignalPolarity.ANTI_CONCEPT
                },
                parser_func=self._parse_gdp_value,
                min_volume=2000.0
            ),

            'geopolitics': SignalConcept(
                name="Global Geopolitical Stability",
                keywords=['war', 'conflict', 'peace', 'sanctions', 'nato', 'china', 'russia', 'ukraine'],
                market_type=MarketType.BINARY,
                polarity_logic={
                    'win': SignalPolarity.PRO_CONCEPT,     # For specific parties/alliances
                    'lose': SignalPolarity.ANTI_CONCEPT,
                    'escalate': SignalPolarity.ANTI_CONCEPT,
                    'de-escalate': SignalPolarity.PRO_CONCEPT,
                    'peace': SignalPolarity.PRO_CONCEPT,
                    'sanctions': SignalPolarity.ANTI_CONCEPT,
                    'aid': SignalPolarity.PRO_CONCEPT
                },
                min_volume=5000.0  # Higher threshold for geopolitical markets
            ),

            'politics': SignalConcept(
                name="US Political Stability",
                keywords=['trump', 'biden', 'election', 'senate', 'house', 'democrat', 'republican'],
                market_type=MarketType.BINARY,
                polarity_logic={
                    'democrat win': SignalPolarity.PRO_CONCEPT,    # Assuming concept = "Democrat Strength"
                    'republican win': SignalPolarity.ANTI_CONCEPT,
                    'trump': SignalPolarity.ANTI_CONCEPT,
                    'biden': SignalPolarity.PRO_CONCEPT
                },
                min_volume=10000.0  # High volume for political markets
            )
        }

        return concepts

    def classify_market_hybrid(self, question: str, keywords: List[str]) -> MarketType:
        """Stage 1: Classify market using Tier 1 Keyword Rules + Tier 2 BERT"""
        question_lower = question.lower()

        # Tier 1: Keyword-based classification
        binary_indicators = [
            'will', 'is', 'are', 'does', 'has', 'have',
            'win', 'lose', 'victory', 'defeat',
            'peace', 'war', 'conflict', 'agreement', 'deal',
            'higher', 'lower', 'rising', 'falling', 'above', 'below'
        ]

        quantitative_indicators = [
            'what will', 'how much', 'how many',
            r'\d+%', r'\d+\.\d+%', r'\d+ percent', r'\d+\.\d+ percent',
            'bps', 'basis points',
            'cpi', 'inflation', 'rate', 'yield', 'growth'
        ]

        # Check for quantitative indicators first (more specific)
        for indicator in quantitative_indicators:
            if re.search(indicator, question_lower, re.IGNORECASE):
                return MarketType.QUANTITATIVE

        # Then check binary indicators
        word_count = len(question_lower.split())
        if word_count < 20:  # Short questions often binary
            for indicator in binary_indicators:
                if indicator in question_lower:
                    return MarketType.BINARY

        # TODO: Add Tier 2 BERT classification here when available
        # For now, default to binary for unknown
        return MarketType.BINARY

    def assign_polarity_scalar(self, question: str, concept: SignalConcept) -> int:
        """Stage 1: Assign polarity scalar λ_i based on hybrid NLP (Tier 1 + BERT)"""
        if not concept.polarity_logic:
            return 1  # Default: assume P_yes is pro-concept

        question_lower = question.lower()

        # Check for manual polarity overrides first
        for term, polarity in concept.polarity_logic.items():
            if term.lower() in question_lower:
                lambda_val = +1 if polarity == SignalPolarity.PRO_CONCEPT else -1
                if self.debug:
                    logger.debug(f"  └─ Polarity: '{term}' → λ = {lambda_val}")
                return lambda_val

        # Default: assume P_yes is pro-concept (λ = +1)
        return 1

    def calculate_aligned_price(self, p_yes: float, lambda_scalar: int) -> float:
        """Stage 1: Calculate aligned price P'_i,t"""
        if lambda_scalar == 1:
            return p_yes        # P'_i,t = P_i,t (pro-concept aligned)
        elif lambda_scalar == -1:
            return 1.0 - p_yes  # P'_i,t = 1 - P_i,t (anti-concept aligned)
        else:
            raise ValueError(f"Invalid lambda_scalar: {lambda_scalar}")

    def generate_concept_signal_unified(self, concept_key: str, output_path: str = None) -> pd.DataFrame:
        """Generate signal using the complete 3-stage unified methodology"""

        if concept_key not in self.concepts:
            raise ValueError(f"Unknown concept: {concept_key}")

        concept = self.concepts[concept_key]

        # STAGE 1: Data Normalization (Classification & Alignment)
        logger.info("=" * 60)
        logger.info(f"STAGE 1: Data Normalization for '{concept.name}'")
        logger.info("=" * 60)

        aligned_markets = self._stage1_normalize_data(concept)

        if not aligned_markets:
            logger.warning(f"No valid markets found for concept '{concept.name}'")
            return pd.DataFrame()

        # STAGE 2: Daily Aggregation (Volume-Weighting)
        logger.info("=" * 60)
        logger.info(f"STAGE 2: Daily Aggregation for '{concept.name}'")
        logger.info("=" * 60)

        daily_signals = self._stage2_daily_aggregation(concept, aligned_markets)

        if daily_signals.empty:
            logger.warning(f"No aggregated signals generated for concept '{concept.name}'")
            return pd.DataFrame()

        # STAGE 3: Continuity & Validation
        logger.info("=" * 60)
        logger.info(f"STAGE 3: Continuity & Validation for '{concept.name}'")
        logger.info("=" * 60)

        final_signal = self._stage3_continuity_validation(concept, daily_signals)

        # Save results if requested
        if output_path:
            final_signal.to_csv(output_path, index=False)
            logger.info(f"✓ Saved unified signal to: {output_path}")

        return final_signal

    def _stage1_normalize_data(self, concept: SignalConcept) -> List[AlignedMarket]:
        """Stage 1: Normalize data for all markets in concept"""

        aligned_markets = []

        # Find all markets related to this concept
        for manifest_file in self.manifest_dir.glob("_manifest_*.csv"):
            try:
                manifest_df = pd.read_csv(manifest_file)
            except Exception as e:
                logger.warning(f"Could not read manifest {manifest_file}: {e}")
                continue

            for _, market_row in manifest_df.iterrows():
                question = market_row['question']

                # Check if market matches concept keywords
                if not any(kw.lower() in question.lower() for kw in concept.keywords):
                    continue

                # Check manual flags
                if concept.manual_excludes and market_row['filename'] in concept.manual_excludes:
                    if self.debug:
                        logger.debug(f"  └─ EXCLUDED by manual flag: {market_row['filename']}")
                    continue

                if concept.manual_includes and market_row['filename'] not in concept.manual_includes:
                    continue  # Skip if not in include list

                # Check volume threshold
                if market_row['volume'] < concept.min_volume:
                    if self.debug:
                        logger.debug(f"  └─ VOLUME TOO LOW: {market_row['filename']} ({market_row['volume']} < {concept.min_volume})")
                    continue

                # Classify market type
                market_type = self.classify_market_hybrid(question, concept.keywords)

                if self.debug:
                    logger.debug(f"Market: {market_row['filename']}")
                    logger.debug(f"  ├─ Question: {question[:60]}...")
                    logger.debug(f"  ├─ Type: {market_type.value}")

                # Assign polarity scalar
                lambda_scalar = self.assign_polarity_scalar(question, concept)

                # Parse expected value for quantitative markets
                expected_value = None
                if market_type == MarketType.QUANTITATIVE and concept.parser_func:
                    try:
                        expected_value = concept.parser_func(question)
                        if self.debug and expected_value is not None:
                            logger.debug(f"  ├─ Expected Value: {expected_value}")
                    except Exception as e:
                        if self.debug:
                            logger.debug(f"  └─ Parsing failed for quantitative: {e}")

                # Create aligned market (price alignment happens in real-time during aggregation)
                aligned_market = AlignedMarket(
                    filename=market_row['filename'],
                    question=question,
                    volume=market_row['volume'],
                    market_type=market_type,
                    lambda_scalar=lambda_scalar,
                    expected_value=expected_value,
                    aligned_price=None  # Will be calculated per date in Stage 2
                )

                if self.debug:
                    logger.debug(f"  └─ λ_scalar: {lambda_scalar}, Valid: {aligned_market.is_valid}")

                aligned_markets.append(aligned_market)

        logger.info(f"✓ Stage 1 Complete: Aligned {len(aligned_markets)} markets for '{concept.name}'")
        self.stage1_stats[concept.name] = len(aligned_markets)

        return aligned_markets

    def _stage2_daily_aggregation(self, concept: SignalConcept, aligned_markets: List[AlignedMarket]) -> pd.DataFrame:
        """Stage 2: Daily volume-weighted aggregation"""

        # Date range for aggregation
        start_date = pd.Timestamp('2020-01-01')  # Conservative start for history
        end_date = pd.Timestamp('2025-12-31')   # Future until end of 2025
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        daily_signals = []

        for i, date in enumerate(date_range):
            if i % 100 == 0:
                logger.info(f"Aggregating date {date.date()} ({i}/{len(date_range)})")

            # Calculate aggregated signal for this date
            if concept.market_type == MarketType.BINARY:
                signal_value = self._calculate_vwp_aligned(date, aligned_markets)
            else:  # QUANTITATIVE
                signal_value = self._calculate_vwev_aligned(date, aligned_markets)

            if signal_value is not None:
                daily_signals.append({
                    'date': date,
                    'signal': signal_value
                })

        result_df = pd.DataFrame(daily_signals)

        if not result_df.empty:
            logger.info(f"✓ Stage 2 Complete: {len(result_df)} daily signals ({result_df['signal'].notna().sum()} non-null)")

        self.stage2_stats[concept.name] = len(result_df)

        return result_df

    def _calculate_vwp_aligned(self, date: pd.Timestamp, aligned_markets: List[AlignedMarket]) -> Optional[float]:
        """Calculate Volume-Weighted Probability with alignment: VWP = Σ(P'_i,t × V_i,t) / Σ(V_i,t)"""

        total_weighted_price = 0.0
        total_volume = 0.0
        active_markets = 0

        for market in aligned_markets:
            # Load market data for this date
            csv_path = self.manifest_dir / market.filename
            if not csv_path.exists():
                continue

            try:
                market_df = pd.read_csv(csv_path, parse_dates=['timestamp'])
                if market_df.empty:
                    continue

                # Filter to date
                market_df['date'] = market_df['timestamp'].dt.date
                day_data = market_df[market_df['date'] == date.date()]

                if day_data.empty:
                    continue

                p_yes = day_data['yes'].iloc[-1]

                # Calculate aligned price P'_i,t
                aligned_price = self.calculate_aligned_price(p_yes, market.lambda_scalar)

                # Accumulate for VWP formula
                total_weighted_price += aligned_price * market.volume
                total_volume += market.volume
                active_markets += 1

            except Exception as e:
                if self.debug:
                    logger.debug(f"Error processing {market.filename}: {e}")
                continue

        if total_volume == 0:
            return None

        vwp_value = total_weighted_price / total_volume

        if self.debug and active_markets > 0:
            logger.debug(f"  └─ {date.date()}: VWP={vwp_value:.4f}, {active_markets} markets, vol={total_volume:.0f}")

        return vwp_value

    def _calculate_vwev_aligned(self, date: pd.Timestamp, aligned_markets: List[AlignedMarket]) -> Optional[float]:
        """Calculate Volume-Weighted Expected Value: VWEV = Σ(E_j,t × P'_j,t × V_j,t) / Σ(V_j,t)"""

        total_weighted_value = 0.0
        total_volume = 0.0
        active_markets = 0

        for market in aligned_markets:

            if market.expected_value is None:
                continue  # Skip markets without parsed values

            # Load market data for this date
            csv_path = self.manifest_dir / market.filename
            if not csv_path.exists():
                continue

            try:
                market_df = pd.read_csv(csv_path, parse_dates=['timestamp'])
                if market_df.empty:
                    continue

                # Filter to date
                market_df['date'] = market_df['timestamp'].dt.date
                day_data = market_df[market_df['date'] == date.date()]

                if day_data.empty:
                    continue

                p_yes = day_data['yes'].iloc[-1]

                # Calculate aligned probability
                aligned_price = self.calculate_aligned_price(p_yes, market.lambda_scalar)

                # VWEV formula: Σ(E_j,t × P'_j,t × V_j,t) / Σ(V_j,t)
                total_weighted_value += market.expected_value * aligned_price * market.volume
                total_volume += market.volume
                active_markets += 1

            except Exception as e:
                if self.debug:
                    logger.debug(f"Error processing {market.filename}: {e}")
                continue

        if total_volume == 0:
            return None

        vwev_value = total_weighted_value / total_volume

        if self.debug and active_markets > 0:
            logger.debug(f"  └─ {date.date()}: VWEV={vwev_value:.4f}, {active_markets} markets, vol={total_volume:.0f}")

        return vwev_value

    def _stage3_continuity_validation(self, concept: SignalConcept, daily_signals: pd.DataFrame) -> pd.DataFrame:
        """Stage 3: Continuity & Validation"""

        if daily_signals.empty:
            return pd.DataFrame()

        # Remove days with no activity
        df = daily_signals.dropna().copy()

        if len(df) < 50:
            logger.warning(f"INSUFFICIENT DATA: Only {len(df)} days available (minimum 50 required)")
            logger.error("REASON: Signal cannot be validated with less than 50 trading days")
            logger.error("SOLUTION: Add more historical data or exclude this concept")
            return pd.DataFrame()

        # Continuity: Apply backward ratio adjustment for rolling expirations
        df = self._apply_backward_ratio_adjustment(df, concept)

        # Validation: Check length requirements
        if len(df) >= 500:
            status = "VALID"
            logger.info(f"✓ Stage 3 Complete: {status} signal ({len(df)} days)")
        elif len(df) >= 50:
            status = "EMERGING"
            logger.info(f"⚠ Stage 3 Complete: {status} signal ({len(df)} days) - USE FOR RESEARCH ONLY")
        else:
            status = "INSUFFICIENT"
            logger.warning(f"✗ Stage 3 Complete: {status} signal ({len(df)} days) - DISCARD")
            return pd.DataFrame()

        self.stage3_stats[concept.name] = {'days': len(df), 'status': status}

        return df

    def _apply_backward_ratio_adjustment(self, df: pd.DataFrame, concept: SignalConcept) -> pd.DataFrame:
        """Apply backward ratio adjustment to stitch together expiring markets (futures roll)"""

        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)

        # For short-term markets that expire/roll: apply adjustment when gap > threshold
        # This ensures continuity when "May CPI" transitions to "June CPI"

        adjusted_signals = []
        current_adjustment = 0.0

        for i, row in df.iterrows():

            if i == 0:
                adjusted_signal = row['signal']
            else:
                prev_signal = df.iloc[i-1]['signal']

                # If no data for previous day (gap), apply continuity adjustment
                if prev_signal != prev_signal:  # NaN check
                    # Continuity: Maintain last known value through gaps
                    adjusted_signal = adjusted_signals[-1] if adjusted_signals else row['signal']
                else:
                    # Normal backward ratio adjustment for rolling markets
                    # ratio_adj = (new_value / old_value) * previous_adjusted, but simplified
                    adjusted_signal = row['signal']  # For now, no complex adjustment

            adjusted_signals.append(adjusted_signal)

        df['adjusted_signal'] = adjusted_signals

        if self.debug:
            logger.debug(f"Applied backward ratio adjustment: {len([x for x in adjusted_signals if x != df.iloc[0]['signal']])} adjustments")

        return df

    def _parse_inflation_value(self, question: str) -> Optional[float]:
        """Parse inflation expected value (% points)"""
        # Reuse existing logic from SignalProcessor
        result = self._parse_inflation_question(question)
        if isinstance(result, tuple) and result[1] is not None:
            return result[1]
        return None

    def _parse_inflation_question(self, question: str) -> Tuple[str, Optional[float]]:
        """
        Parse inflation question to extract both market type and numerical value.
        """
        question_lower = question.lower().strip()

        # BINARY PATTERNS: Questions that are yes/no without specific numbers
        binary_patterns = [
            r'\b(high|higher|low|lower|rising|falling|increasing|decreasing|above|below)\b.*\byear\b',
            r'\b(will|is|are|does)\b.*\b(double|halve|cut|in\s+half)\b',
            r'\b(hit|reach|exceed|break)\b.*\d+.*percent',
            r'\b(decrease|increase|rise|fall)\b.*\d+.*percent.*year',
            r'\b(over|under)\b.*\d+.*percent.*\b(this|next)\b.*year',
        ]

        for pattern in binary_patterns:
            if re.search(pattern, question_lower):
                # Check if it has specific numbers - might still be quantitative
                numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', question_lower)
                if numbers and len(numbers) <= 2:  # Very specific (e.g., "over 3% this year")
                    val = float(numbers[0])
                    return 'quantitative', val / 10.0 if val >= 10 else val / 100.0
                return 'binary', None

        # QUANTITATIVE PATTERNS: Specific numerical predictions
        if any(word in question_lower for word in ['cpi', 'inflation']):
            # Look for specific percentage patterns
            percent_patterns = [
                r'(\d+(?:\.\d+)?)\s*%',  # "2.6%" or "3%"
                r'(\d+(?:\.\d+)?)\s*percent',  # "2.6 percent"
                r'increase.*by\s+(\d+(?:\.\d+)?)\s*%',  # "increase by 0.1%"
                r'increase.*by\s+(\d+(?:\.\d+)?)\s*percent',  # "increase by 0.1 percent"
            ]

            for pattern in percent_patterns:
                matches = re.findall(pattern, question_lower)
                if matches:
                    val = float(matches[0])
                    return 'quantitative', val

        # Binary by default if we can't parse a specific value
        return 'binary', None

    def _parse_fed_rate_value(self, question: str) -> Optional[float]:
        """Parse Fed rate expected value (% points)"""
        question_lower = question.lower().strip()

        # Handle basis points (bps)
        bps_patterns = [
            r'(\d+)\s*(?:bps|basis\s*points?)',  # 25 bps, 25 basis points
            r'by\s+(\d+)\s*bps',  # by 25 bps
            r'cuts?\s+(\d+)\s*bps',  # cuts 25 bps
            r'decreases?\s+(\d+)\s*bps',  # decreases 25 bps
        ]

        for pattern in bps_patterns:
            matches = re.findall(pattern, question_lower)
            if matches:
                return int(matches[0]) / 100.0  # Convert bps to percentage

        # Handle direct percentage rates
        percent_patterns = [
            r'(\d+(?:\.\d+)?)\s*%',  # 3.50%
            r'(\d+(?:\.\d+)?)\s*percent',  # 3.50 percent
        ]

        for pattern in percent_patterns:
            matches = re.findall(pattern, question_lower)
            if matches:
                return float(matches[0])

        # Handle rate ranges or thresholds
        rate_patterns = [r'(\d+(?:\.\d+)?)\s*(?:to|%|-)', r'rate\s+(?:of\s+)?(\d+(?:\.\d+)?)']
        for pattern in rate_patterns:
            matches = re.findall(pattern, question_lower)
            if matches:
                return float(matches[0])

        return None

    def _parse_gdp_value(self, question: str) -> Optional[float]:
        """Parse GDP expected value (% points)"""
        # Similar parsing to inflation
        question_lower = question.lower()
        matches = re.findall(r'(\d+(?:\.\d+)?)', question_lower)
        if matches:
            for match in matches:
                val = float(match)
                if 0 < val < 20:  # GDP growth rates typically 0-10%
                    return val / 100.0 if val >= 10 else val  # Assume 2.5 = 250bps or 2.5%
        return None

    def _parse_monthly_inflation_value(self, question: str) -> Optional[float]:
        """Parse monthly inflation threshold market to get expected value"""
        question_lower = question.lower().strip()

        # Parse threshold patterns for monthly inflation
        # Markets ask about thresholds like "increase by X% or less"

        # Pattern: "increase by X or less" - this represents 0% to X% range, we use X as expected value
        patterns_less_or_equal = [
            r'increase.*by\s+(\d+(?:\.\d+)?)\s*or\s*less',
            r'increase.*by\s+(\d+(?:\.\d+)?)\s*percent\s*or\s*less',
            r'\b(\d+(?:\.\d+)?)\s*percent\s*or\s*less\b',
            r'\b(\d+(?:\.\d+)?)\s*or\s*less\b',
        ]

        for pattern in patterns_less_or_equal:
            matches = re.findall(pattern, question_lower)
            if matches:
                return float(matches[0]) / 100.0  # Convert percentage to decimal

        # Pattern: "increase by X" - this represents >0% to X% range, approximately X/2
        patterns_equal = [
            r'increase.*by\s+(\d+(?:\.\d+)?)\s*$',
            r'increase.*by\s+(\d+(?:\.\d+)?)\s*$',
        ]

        for pattern in patterns_equal:
            matches = re.findall(pattern, question_lower)
            if matches:
                val = float(matches[0]) / 200.0  # X/2 as rough approximation of expected value
                return val

        return None

    def _parse_annual_inflation_value(self, question: str) -> Optional[float]:
        """Parse annual inflation market to get expected value

        Handles two types of annual inflation markets:
        1. Change rates: "increase by X% or less in [year]" -> percentage points
        2. Level forecasts: "reach more than X% in 2025" -> absolute level
        """
        question_lower = question.lower().strip()

        # TYPE 1: Level forecasts - "reach/exceed more than X%"
        level_patterns = [
            r'reach.*more\s+than\s+(\d+(?:\.\d+)?)',
            r'exceed.*more\s+than\s+(\d+(?:\.\d+)?)',
            r'more\s+than\s+(\d+(?:\.\d+)?).*percent',
            r'higher\s+than\s+(\d+(?:\.\d+)?)',
        ]

        for pattern in level_patterns:
            matches = re.findall(pattern, question_lower)
            if matches:
                # For "more than X%" questions, return X% (the threshold)
                return float(matches[0]) / 100.0

        # TYPE 2: Change rates - "increase by X% or less"
        change_patterns_less_or_equal = [
            r'increase.*by\s+(\d+(?:\.\d+)?)\s*or\s*less',
            r'increase.*by\s+(\d+(?:\.\d+)?)\s*percent\s*or\s*less',
            r'\b(\d+(?:\.\d+)?)\s*percent\s*or\s*less\b',
            r'\b(\d+(?:\.\d+)?)\s*or\s*less\b',
        ]

        for pattern in change_patterns_less_or_equal:
            matches = re.findall(pattern, question_lower)
            if matches:
                val = float(matches[0])
                # If it's already a percentage, convert to decimal (e.g., 3% -> 0.03)
                return val / 100.0

        # TYPE 2b: Change rate ranges "between X and Y"
        between_patterns = [
            r'between\s+(\d+(?:\.\d+)?)\s*and\s+(\d+(?:\.\d+)?)',
            r'\b(\d+(?:\.\d+)?)\s*to\s+(\d+(?:\.\d+)?)\b',
        ]

        for pattern in between_patterns:
            matches = re.findall(pattern, question_lower)
            if matches:
                val1, val2 = map(float, matches[0])
                # Use midpoint of the range
                return (val1 + val2) / 2 / 100.0

        return None

    def _load_manual_flags(self) -> Dict:
        """Load manual market flagging configuration from JSON file"""
        flags_file = Path("test_signals/manual_flags_example.json")

        if flags_file.exists():
            try:
                with open(flags_file, 'r') as f:
                    data = json.load(f)
                logger.info(f"✓ Loaded manual flags from {flags_file}")
                return data.get('manual_flags', {})
            except Exception as e:
                logger.warning(f"Could not load manual flags: {e}")
                return {}
        else:
            logger.info(f"No manual flags file found at {flags_file}, using empty flags")
            return {}

    def get_concept_statistics(self, concept_key: str) -> Dict:
        """Get comprehensive statistics for a processed concept"""
        if concept_key not in self.concepts:
            return {}

        concept = self.concepts[concept_key]

        stats = {
            'concept_name': concept.name,
            'market_type': concept.market_type.value,
            'stage1_markets': self.stage1_stats.get(concept.name, 0),
            'stage2_days': self.stage2_stats.get(concept.name, 0),
            'stage3_validation': self.stage3_stats.get(concept.name, {})
        }

        return stats
