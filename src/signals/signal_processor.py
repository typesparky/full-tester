import pandas as pd
import numpy as np
import re
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

class SignalProcessor:
    """
    Processes Polymarket data into continuous time-series signals using VWEV/VWP methods.
    """

    def __init__(self, manifest_path: str, data_directory: str):
        self.manifest_path = Path(manifest_path)
        self.data_directory = Path(data_directory)
        self.manifest_df = None
        self.positive_keywords = [
            'pass', 'succeed', 'win', 'agreement', 'deal', 'normalize', 'positive',
            'peace', 'stabilize', 'settle', 'benefit', 'growth'
        ]
        self.negative_keywords = [
            'fail', 'war', 'conflict', 'sanctions', 'tariff', 'invade', 'attack',
            'crisis', 'fail', 'oust', 'negative', 'recession', 'decline'
        ]

    def is_placeholder_market(self, question_text: str) -> bool:
        """
        Checks if a market question contains common placeholder patterns.
        Automatically filters out template/example markets.
        """
        # Case-insensitive matching
        question_lower = question_text.lower()

        # Pattern for "Person [Letter]" or "Party [Letter]" - templates
        if re.search(r'\b(person|party)\s+[a-z]\b', question_lower):
            logger.debug(f"Filtered placeholder market: {question_text} (Person/Party template)")
            return True

        # Pattern for "Country [Letter(s)]" e.g., Country X, Country ABC
        if re.search(r'\bcountry\s+([a-z]{1}|[a-z]{3})\b', question_lower):
            logger.debug(f"Filtered placeholder market: {question_text} (Country template)")
            return True

        # Pattern for "Candidate [Number]" or "Option [Letter/Number]"
        if re.search(r'\b(candidate|option)\s+[a-z0-9]+\b', question_lower):
            logger.debug(f"Filtered placeholder market: {question_text} (Candidate/Option template)")
            return True

        # Additional specific placeholder patterns observed
        if "person " in question_lower and len(question_lower.split()) < 10:
            # Very short questions with "person" are often templates
            logger.debug(f"Filtered placeholder market: {question_text} (Person template)")
            return True

        if "party " in question_lower and len(question_lower.split()) < 10:
            # Very short questions with "party" are often templates
            logger.debug(f"Filtered placeholder market: {question_text} (Party template)")
            return True

        return False

    def load_manifest(self) -> pd.DataFrame:
        """Load the manifest CSV file and filter out placeholder markets."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")
        self.manifest_df = pd.read_csv(self.manifest_path)
        initial_count = len(self.manifest_df)

        # Filter out placeholder markets
        filtered_markets = []
        for _, market in self.manifest_df.iterrows():
            question = str(market.get('question', ''))
            if not self.is_placeholder_market(question):
                filtered_markets.append(market)

        self.manifest_df = pd.DataFrame(filtered_markets).reset_index(drop=True)
        filtered_count = len(self.manifest_df)
        placeholder_count = initial_count - filtered_count

        logger.info(f"Loaded manifest: {initial_count} markets total")
        if placeholder_count > 0:
            logger.info(f"Filtered out {placeholder_count} placeholder/template markets")
        logger.info(f"Kept {filtered_count} valid markets for processing")

        return self.manifest_df

    def group_markets(self, keyword: str) -> pd.DataFrame:
        """Group markets containing the keyword in their question text."""
        if self.manifest_df is None:
            self.load_manifest()

        # Filter markets containing keyword (case-insensitive)
        mask = self.manifest_df['question'].str.contains(keyword, case=False, na=False)
        group_df = self.manifest_df[mask].copy()
        logger.info(f"Grouped {len(group_df)} markets matching keyword '{keyword}'")
        return group_df

    def parse_inflation_question(self, question: str) -> Tuple[str, Optional[float]]:
        """
        Parse inflation question to extract both market type and numerical value.

        Returns: (market_type, parsed_value)
        - 'quantitative': value is percentage (2.6, 5.0, etc.)
        - 'binary': value is None (yes/no outcome only)

        Inflation Examples:
        - "Will annual inflation increase by 2.6% in 2025?" → ('quantitative', 2.6)
        - "Will inflation reach more than 3% this year?" → ('quantitative', 3.0)
        - "Will inflation be higher than last year?" → ('binary', None)
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

    def parse_fed_rates_question(self, question: str) -> Optional[float]:
        """
        Parse Fed interest rates questions to extract the rate value.

        Handles basis points formats like:
        - "fed decreases interest rates by 25 bps" -> 0.25
        - "fed cuts rates by 50 basis points" -> 0.50
        - "federal funds rate of 3.50%" -> 3.50
        """
        question = question.lower().strip()

        # Handle basis points (bps)
        bps_patterns = [
            r'(\d+)\s*(?:bps|basis\s*points?)',  # 25 bps, 25 basis points
            r'by\s+(\d+)\s*bps',  # by 25 bps
            r'cuts?\s+(\d+)\s*bps',  # cuts 25 bps
            r'decreases?\s+(\d+)\s*bps',  # decreases 25 bps
        ]

        for pattern in bps_patterns:
            matches = re.findall(pattern, question)
            if matches:
                return int(matches[0]) / 100.0  # Convert bps to percentage

        # Handle direct percentage rates
        percent_patterns = [
            r'(\d+(?:\.\d+)?)\s*%',  # 3.50%
            r'(\d+(?:\.\d+)?)\s*percent',  # 3.50 percent
        ]

        for pattern in percent_patterns:
            matches = re.findall(pattern, question)
            if matches:
                return float(matches[0])

        # Handle rate ranges or thresholds
        rate_patterns = [r'(\d+(?:\.\d+)?)\s*(?:to|%|-)', r'rate\s+(?:of\s+)?(\d+(?:\.\d+)?)']
        for pattern in rate_patterns:
            matches = re.findall(pattern, question)
            if matches:
                return float(matches[0])

        return None

    def parse_generic_quantitative(self, question: str) -> Optional[float]:
        """
        Generic parser for quantitative markets - tries to extract any reasonable number.
        """
        question = question.lower().strip()

        # Look for meaningful numbers (prioritize larger ones)
        all_numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', question)
        if all_numbers:
            # Try to find context clues
            if any(word in question for word in ['rate', 'yield', 'gdp', 'growth', 'fed', 'funds']):
                # For financial rates, numbers around 1-10 are likely percentages
                numbers = [float(n) for n in all_numbers if 0 < float(n) <= 100]
                if numbers:
                    return max(numbers)  # Take the largest reasonable rate

            # Default: take any reasonable number
            numbers = [float(n) for n in all_numbers if float(n) > 0]
            if numbers:
                return numbers[0]  # Take first number

        return None

    def parse_gdp_question(self, question: str) -> Optional[float]:
        """
        Parse GDP question to extract expected growth value (as decimal).
        Handles growth percentage markets like "between X% and Y%" correctly.

        Returns: Float value representing expected GDP growth rate (0.02 for 2%)
        """
        question_lower = question.lower().strip()

        # PATTERN 1: Between X% and Y% - take midpoint
        between_pattern = r'between\s+(\d+(?:\.\d+)?)\s*%\s*and\s+(\d+(?:\.\d+)?)\s*%'
        matches = re.findall(between_pattern, question_lower)
        if matches:
            val1, val2 = map(float, matches[0])
            midpoint = (val1 + val2) / 2
            return midpoint / 100.0  # Convert percentage to decimal

        # PATTERN 2: Greater than X%
        greater_pattern = r'greater\s+than\s+(\d+(?:\.\d+)?)\s*%'
        matches = re.findall(greater_pattern, question_lower)
        if matches:
            val = float(matches[0])
            # For "greater than X%", we assume slightly above X as expected value
            return (val + 1.0) / 100.0  # Add 1% buffer

        # PATTERN 3: Less than X%
        less_pattern = r'less\s+than\s+(\d+(?:\.\d+)?)\s*%'
        matches = re.findall(less_pattern, question_lower)
        if matches:
            val = float(matches[0])
            # For "less than X%", we assume slightly below X as expected value
            return max(0, (val - 1.0) / 100.0)  # Subtract 1% buffer, floor at 0

        # PATTERN 4: Exactly X%
        exact_pattern = r'(\d+(?:\.\d+)?)\s*%'
        matches = re.findall(exact_pattern, question_lower)
        if matches:
            val = float(matches[0])
            return val / 100.0  # Convert percentage to decimal

        # Fallback: use generic parser
        return self.parse_generic_quantitative(question)

    def calculate_vwev(self, market_group: pd.DataFrame, date: pd.Timestamp, parser_func, data_dir: Path = None) -> float:
        """
        Calculate Volume-Weighted Expected Value for quantitative markets.

        VWEV = Σ(P_yes_i * value_i * volume_i) / Σ(volume_i)
        """
        total_weighted_value = 0.0
        total_volume = 0.0

        base_dir = data_dir if data_dir else self.data_directory

        for _, market in market_group.iterrows():
            try:
                csv_filename = market['filename']
                csv_path = base_dir / csv_filename
                if not csv_path.exists():
                    continue

                market_df = pd.read_csv(csv_path, parse_dates=['timestamp'])
                if market_df.empty:
                    continue

                market_df['date'] = market_df['timestamp'].dt.date
                day_data = market_df[market_df['date'] == date.date()]

                if day_data.empty:
                    continue

                p_yes = day_data['yes'].iloc[-1]
                volume = market['volume']

                # Parse the expected value using parser
                value = parser_func(market['question'])
                if value is None:
                    continue

                total_weighted_value += p_yes * value * volume
                total_volume += volume

            except Exception as e:
                logger.warning(f"Error processing market {market['slug']}: {e}")
                continue

        if total_volume == 0:
            return np.nan

        return total_weighted_value / total_volume

    def calculate_unsigned_vwp(self, market_group: pd.DataFrame, date: pd.Timestamp) -> float:
        """
        Calculate Unsigned Volume-Weighted Probability for binary markets.

        VWP = Σ(P_yes_i * volume_i) / Σ(volume_i)
        """
        total_p_yes_weighted = 0.0
        total_volume = 0.0

        for _, market in market_group.iterrows():
            try:
                csv_filename = market['filename']
                csv_path = self.data_directory / csv_filename
                if not csv_path.exists():
                    continue

                market_df = pd.read_csv(csv_path, parse_dates=['timestamp'])
                if market_df.empty:
                    continue

                market_df['date'] = market_df['timestamp'].dt.date
                day_data = market_df[market_df['date'] == date.date()]

                if day_data.empty:
                    continue

                p_yes = day_data['yes'].iloc[-1]
                volume = market['volume']

                total_p_yes_weighted += p_yes * volume
                total_volume += volume

            except Exception as e:
                logger.warning(f"Error processing market {market['slug']}: {e}")
                continue

        if total_volume == 0:
            return np.nan

        return total_p_yes_weighted / total_volume

    def generate_concept_aggregated_signal(self, concept: str, question_keywords: List[str],
                                         value_parser_func, output_path: str,
                                         min_volume: float = 1000.0,
                                         use_historical: bool = True) -> pd.DataFrame:
        """
        Generate aggregated concept signals by combining related markets over time.

        Instead of individual market signals, creates ONE continuous signal for the concept
        by aggregating all related markets that ever existed.

        Args:
            concept: Concept name (e.g., "Inflation Expectations")
            question_keywords: Keywords to identify related markets
            value_parser_func: Function to extract values from market questions
            output_path: Output file path
            min_volume: Minimum volume threshold for credibility
            use_historical: Whether to include historically closed markets

        Returns:
            DataFrame with date and aggregated signal
        """
        logger.info(f"Generating AGGREGATED {concept} signal from concept keywords: {question_keywords}")

        # Find ALL markets related to this concept across ALL directories and manifests
        all_concept_markets = []

        # 1. Check all known manifests for related markets
        manifest_files = [
            "polymarket_data/manifest/_manifest_geopolitics__foreign_policy_comprehensive.csv",
            "polymarket_data/manifest/_manifest_geopolitics__war.csv",
            "polymarket_data/manifest/_manifest_geopolitics__war.csv",
            "polymarket_data/manifest/_manifest_custom_world_affairs.csv",
            "polymarket_data/manifest/_manifest_geopolitics__war.csv"
        ]

        for manifest_file in manifest_files:
            manifest_path = Path(manifest_file)
            if manifest_path.exists():
                try:
                    df = pd.read_csv(manifest_path)
                    # Filter for markets containing our keywords (case-insensitive)
                    mask = df['question'].str.lower().str.contains('|'.join(question_keywords), na=False)
                    relevant_markets = df[mask]

                    if not relevant_markets.empty:
                        logger.info(f"Found {len(relevant_markets)} markets in {manifest_file}")
                        for _, market in relevant_markets.iterrows():
                            all_concept_markets.append({
                                'filename': market['filename'],
                                'question': market['question'],
                                'volume': market['volume'],
                                'source': str(manifest_path.parent.name) + '/' + str(manifest_path.name)
                            })
                except Exception as e:
                    logger.warning(f"Could not read manifest {manifest_file}: {e}")

        # 2. Also scan all directories for additional markets (including potential closed markets)
        directories_to_scan = [
            "polymarket_data/inflation",
            "polymarket_data/us_gdp",
            "polymarket_data/fed_&_interest_rates",
            "polymarket_data/treasury_&_yields_(2025)",
            "polymarket_data/us_jobs_&_unemployment",
            "polymarket_data/politics_(us_election)",
            "polymarket_data/geopolitics__foreign_policy_comprehensive",
            "polymarket_data/geopolitics__war",
            "polymarket_data/geopolitics_&_war",
            "polymarket_data/custom_world_affairs"
        ]

        for dir_path in directories_to_scan:
            directory = Path(dir_path)
            if directory.exists():
                for csv_file in directory.glob("*.csv"):
                    filename = csv_file.name.lower()

                    # Check if filename contains our keywords
                    if any(keyword.lower() in filename for keyword in question_keywords):
                        # Extract question from filename
                        question = self._filename_to_question(csv_file.name)
                        if question and any(keyword.lower() in question.lower() for keyword in question_keywords):
                            all_concept_markets.append({
                                'filename': str(csv_file.name),
                                'question': question,
                                'volume': min_volume,  # Assume minimum volume for directory-scanned files
                                'source': f"{directory.name}/{csv_file.name}"
                            })

        # Remove duplicates based on filename
        unique_markets = []
        seen_filenames = set()
        for market in all_concept_markets:
            if market['filename'] not in seen_filenames:
                if market['volume'] >= min_volume:
                    unique_markets.append(market)
                    seen_filenames.add(market['filename'])

        logger.info(f"Found {len(unique_markets)} unique concept-related markets for {concept}")

        if not unique_markets:
            logger.warning(f"No markets found for concept '{concept}' with keywords {question_keywords}")
            return pd.DataFrame()

        # Convert to DataFrame for processing
        concept_df = pd.DataFrame(unique_markets)

        # Aggregate by date - create TIME SERIES for the entire concept
        start_date = pd.Timestamp('2020-01-01')  # Go back to beginning of SPY data
        end_date = pd.Timestamp('2025-12-31')    # Future date
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        aggregated_signals = []

        total_markets_checked = len(unique_markets)

        for i, date in enumerate(date_range):
            if i % 100 == 0:  # Progress logging
                logger.info(f"Processing date {date.date()} ({i}/{len(date_range)} completed)")

            # Calculate VWEV across ALL markets active on this date
            total_weighted_value = 0.0
            total_volume = 0.0

            active_markets = 0

            for _, market in concept_df.iterrows():
                filename = market['filename']
                volume = market['volume']

                # Find the correct directory for this market
                possible_dirs = [
                    Path("polymarket_data/inflation"),
                    Path("polymarket_data/us_gdp"),
                    Path("polymarket_data/fed_&_interest_rates"),
                    Path("polymarket_data/politics_(us_election)"),
                    Path("polymarket_data/geopolitics__foreign_policy_comprehensive"),
                    Path("polymarket_data/geopolitics__war"),
                    Path("polymarket_data/custom_world_affairs"),
                ]

                csv_path = None
                for dir_path in possible_dirs:
                    potential = dir_path / filename
                    if potential.exists():
                        csv_path = potential
                        break

                if csv_path is None:
                    continue  # Market file not found

                try:
                    # Load market data for this date
                    market_df = pd.read_csv(csv_path, parse_dates=['timestamp'])
                    if market_df.empty:
                        continue

                    market_df['date'] = market_df['timestamp'].dt.date
                    day_data = market_df[market_df['date'] == date.date()]

                    if day_data.empty:
                        continue  # No data for this date

                    p_yes = day_data['yes'].iloc[-1]

                    # Parse the value from the question
                    parsed_value = value_parser_func(market['question'])

                    if parsed_value is None:
                        continue  # Couldn't parse value

                    # Add to VWEV calculation
                    total_weighted_value += p_yes * parsed_value * volume
                    total_volume += volume
                    active_markets += 1

                except Exception as e:
                    logger.debug(f"Error processing market {filename} on {date}: {e}")
                    continue

            # Calculate final aggregated signal for this date
            if total_volume > 0:
                concept_signal = total_weighted_value / total_volume
            else:
                concept_signal = np.nan

            aggregated_signals.append({
                'date': date,
                'signal': concept_signal,
                'active_markets': active_markets,
                'total_volume': total_volume
            })

        # Create final DataFrame
        result_df = pd.DataFrame(aggregated_signals)

        # Only keep dates where we had at least some market activity
        result_df = result_df[result_df['active_markets'] > 0]

        logger.info(f"Generated {len(result_df)} days of AGGREGATED {concept} signal")
        logger.info(f"Date range: {result_df['date'].min()} to {result_df['date'].max()}")

        if output_path:
            result_df.to_csv(output_path, index=False)
            logger.info(f"Saved AGGREGATED {concept} signal to: {output_path}")

        return result_df

    def generate_quantitative_signal(self, sector: str, output_path: str) -> pd.DataFrame:
        """
        Legacy method - now delegates to concept-based aggregation.
        """
        logger.info(f"Legacy method called for {sector} - using concept aggregation instead")

        # Map sectors to concept aggregation
        sector_configs = {
            'us inflation': {
                'concept': 'US Inflation Expectations',
                'keywords': ['inflation', 'cpi'],
                'parser': self.parse_inflation_question,
                'min_volume': 500.0
            },
            'us gdp': {
                'concept': 'US GDP Growth',
                'keywords': ['gdp', 'growth', 'economic'],
                'parser': self.parse_gdp_question,  # Dedicated GDP parser
                'min_volume': 500.0
            },
            'fed interest rates': {
                'concept': 'Fed Interest Rate Expectations',
                'keywords': ['fed', 'interest', 'rate', 'bps', 'fomc', 'federal', 'reserve'],
                'parser': self.parse_fed_rates_question,
                'min_volume': 500.0
            },
            'politics': {
                'concept': 'Political Sentiment',
                'keywords': ['trump', 'biden', 'election', 'political', 'senate', 'house'],
                'parser': self.parse_generic_quantitative,
                'min_volume': 200.0
            }
        }

        if sector.lower() not in sector_configs:
            logger.warning(f"Unknown sector '{sector}', falling back to legacy processing")

            # Fallback to legacy method for unknown sectors
            return self._legacy_quantitative_signal(sector, output_path)

        config = sector_configs[sector.lower()]
        return self.generate_concept_aggregated_signal(
            concept=config['concept'],
            question_keywords=config['keywords'],
            value_parser_func=config['parser'],
            output_path=output_path,
            min_volume=config['min_volume']
        )

    def _legacy_quantitative_signal(self, sector: str, output_path: str) -> pd.DataFrame:
        """Original quantitative signal generation as fallback."""
        # [Original implementation unchanged for compatibility]
        logger.info(f"Using LEGACY quantitative processing for {sector}")

        # Select parser and directory based on sector
        if 'inflation' in sector.lower():
            parser_func = self.parse_inflation_question
            data_subdir = "inflation"
        elif 'gdp' in sector.lower():
            parser_func = self.parse_inflation_question
            data_subdir = "us_gdp"
        elif 'fed' in sector.lower() or 'interest' in sector.lower() or 'rates' in sector.lower():
            parser_func = self.parse_fed_rates_question
            data_subdir = "fed_&_interest_rates"
        elif 'politics' in sector.lower() or 'election' in sector.lower():
            parser_func = self.parse_generic_quantitative
            data_subdir = "politics_(us_election)"
        else:
            raise ValueError(f"Unknown sector: {sector}")

        # For quantitative, scan the directory directly since no manifest
        data_dir = Path(self.data_directory) / data_subdir
        if not data_dir.exists():
            logger.error(f"Data directory not found: {data_dir}")
            return pd.DataFrame()

        # Create market group from directory
        market_group = []
        for csv_file in data_dir.glob("*.csv"):
            question = self._filename_to_question(csv_file.name)
            if question and parser_func(question) is not None:
                market_group.append({
                    'filename': str(csv_file.relative_to(data_dir)),
                    'volume': 1.0,
                    'question': question
                })

        market_group_df = pd.DataFrame(market_group)

        if market_group_df.empty:
            logger.error(f"No markets found for sector '{sector}' in {data_subdir}")
            return pd.DataFrame()

        logger.info(f"Found {len(market_group_df)} markets in {sector}")

        # Date range - use historical dates for concept aggregation
        start_date = pd.Timestamp('2020-01-01')
        end_date = pd.Timestamp('2025-12-31')
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        signals = []
        for date in date_range:
            vwev_value = self.calculate_vwev(market_group_df, date, parser_func, data_dir)
            signals.append({
                'date': date,
                'signal': vwev_value
            })

        df = pd.DataFrame(signals).dropna()

        if output_path:
            df.to_csv(output_path, index=False)
            logger.info(f"Saved VWEV signal to: {output_path}")

        return df

    def _filename_to_question(self, filename: str) -> str:
        """
        Convert filename to a mock question for parsing.
        """
        # Remove extension and replace underscores with spaces
        name = filename.replace('.csv', '').replace('_', ' ').replace('  ', ' ')

        # Clean up some common prefixes/suffixes
        name = name.replace('will ', '').replace('what will ', '')

        return name.strip()

    def generate_binary_signal(self, sector: str, output_path: str) -> pd.DataFrame:
        """
        Generate unsigned VWP signal for binary markets in a sector.

        Args:
            sector: Sector name (e.g., "Geopolitics")
            output_path: Path to save the output CSV

        Returns:
            DataFrame with date and signal columns
        """
        logger.info(f"Generating binary unsigned VWP signal for sector: {sector}")

        # Map sector to keyword or use the whole manifest
        if 'geopolitics' in sector.lower():
            manifest_file = "_manifest_geopolitics__foreign_policy_comprehensive.csv"
            manifest_path = self.manifest_path.parent / "manifest" / manifest_file
            self.manifest_path = manifest_path  # Switch to geopolitics manifest

            # Load all markets from geopolitics manifest
            if self.manifest_df is None or not str(self.manifest_df) in str(manifest_path):
                self.manifest_df = pd.read_csv(manifest_path)
            market_group = self.manifest_df
        else:
            raise ValueError(f"Unknown sector: {sector}")

        if market_group.empty:
            logger.error(f"No markets found for sector '{sector}'")
            return pd.DataFrame()

        # Date range
        start_date = pd.Timestamp('2024-05-01')
        end_date = pd.Timestamp('2025-12-31')
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        signals = []
        for date in date_range:
            vwp_value = self.calculate_unsigned_vwp(market_group, date)
            signals.append({
                'date': date,
                'signal': vwp_value
            })

        df = pd.DataFrame(signals).dropna()

        if output_path:
            df.to_csv(output_path, index=False)
            logger.info(f"Saved unsigned VWP signal to: {output_path}")

        return df

    def calculate_daily_signal(self, market_group: pd.DataFrame, date: pd.Timestamp) -> float:
        """
        Calculate aggregated signal for a specific date across all markets in the group.

        Uses VWEV formula for quantitative markets, VWP for binary markets.
        """
        total_weighted_value = 0.0
        total_volume = 0.0

        date_str = date.strftime('%Y-%m-%d')

        for _, market in market_group.iterrows():
            try:
                # Construct CSV path
                csv_filename = market['filename']
                csv_path = self.data_directory / csv_filename
                if not csv_path.exists():
                    logger.warning(f"CSV not found: {csv_path}")
                    continue

                # Load market data
                market_df = pd.read_csv(csv_path, parse_dates=['timestamp'])
                if market_df.empty:
                    continue

                # Filter to specific date - take the last closing price of the day
                market_df['date'] = market_df['timestamp'].dt.date
                day_data = market_df[market_df['date'] == date.date()]

                if day_data.empty:
                    continue

                # Get the 'yes' probability (P_yes)
                p_yes = day_data['yes'].iloc[-1]  # Last closing price

                # Get volume (point-in-time from manifest)
                volume = market['volume']

                # Parse market type and info
                market_type, market_info = self.parse_market_info(market['question'])

                if market_type == 'quantitative':
                    # VWEV: Σ(P_yes * value * volume) / Σ(volume)
                    total_weighted_value += p_yes * market_info * volume
                elif market_type == 'binary':
                    # VWP: Σ(P_yes * direction * volume) / Σ(volume)
                    total_weighted_value += p_yes * market_info * volume

                total_volume += volume

            except Exception as e:
                logger.warning(f"Error processing market {market['slug']}: {e}")
                continue

        if total_volume == 0:
            return np.nan

        return total_weighted_value / total_volume

    def generate_signal_time_series(self, keyword: str, output_path: str = None) -> pd.DataFrame:
        """
        Generate continuous time series signal for markets matching the keyword.

        Args:
            keyword: Keyword to search for in market questions
            output_path: Optional path to save CSV output

        Returns:
            DataFrame with 'date' and 'signal' columns
        """
        logger.info(f"Generating signal time series for keyword: {keyword}")

        # Group markets
        market_group = self.group_markets(keyword)
        if market_group.empty:
            logger.error(f"No markets found matching '{keyword}'")
            return pd.DataFrame()

        # Determine date range - we'll use 2024-2025 as example
        # In practice, this would be determined from data availability
        start_date = pd.Timestamp('2024-05-01')
        end_date = pd.Timestamp('2025-12-31')
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        signals = []
        for date in date_range:
            signal_value = self.calculate_daily_signal(market_group, date)
            signals.append({
                'date': date,
                'signal': signal_value
            })

        # Create DataFrame
        df = pd.DataFrame(signals)
        # CRITICAL FIX: Do NOT forward-fill NaN values - these represent illiquid/zero-volume days
        # The backtesting system should handle NaN as missing signal data
        df = df.dropna()  # Only drop NaN from very beginning if no signal at start

        logger.info(f"Generated signal time series with {len(df)} data points")

        # Save to CSV if output_path provided
        if output_path:
            df.to_csv(output_path, index=False)
            logger.info(f"Saved signal time series to: {output_path}")

        return df

        # Legacy method - unified methodology now uses separate UnifiedSignalGenerator class
        logger.warning("Consider upgrading to use UnifiedSignalGenerator for new unified methodology")
        return None


def interactive_menu():
    """Interactive CLI for signal processing."""
    print("🌍 Polymarket Signal Engineering - Interactive Mode")
    print("=" * 55)

    # Available sectors
    sectors = {
        'quantitative': [
            '1. US Inflation (CPI)',
            '2. US GDP Growth',
            '3. Fed Interest Rates',
            '4. Politics & Elections'
        ],
        'binary': [
            '5. Geopolitics & Foreign Policy',
            '6. Geopolitics & War',
            '7. Trade Deals'
        ]
    }

    print("\n📊 Available Sectors:")
    for category, sector_list in sectors.items():
        print(f"\n{category.upper()} MARKETS:")
        for sector in sector_list:
            print(f"   {sector}")

    # Get sector choice
    while True:
        try:
            choice = input("\nSelect a sector (1-7): ").strip()
            choice_map = {
                '1': ('quantitative', 'US Inflation'),
                '2': ('quantitative', 'US GDP'),
                '3': ('quantitative', 'Fed Interest Rates'),
                '4': ('quantitative', 'Politics'),
                '5': ('binary', 'Geopolitics'),
                '6': ('binary', 'Geopolitics War'),
                '7': ('binary', 'Trade Deals')
            }

            if choice in choice_map:
                signal_type, sector = choice_map[choice]
                break
            else:
                print("❌ Invalid choice. Please select 1-7.")
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            return

    # Get volume filter
    while True:
        try:
            volume_filter = input(f"\n💰 Minimum volume filter (Polymarket units) [default: 10000]: ").strip()
            if volume_filter == '':
                volume_filter = 10000.0
            else:
                volume_filter = float(volume_filter)
            break
        except ValueError:
            print("❌ Please enter a valid number.")
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            return

    # Set output file
    default_output = f"raw_signal_{sector.lower().replace(' ', '_')}.csv"
    output_file = input(f"\n💾 Output file name [default: {default_output}]: ").strip()
    if output_file == '':
        output_file = default_output

    # Configure processor based on type
    if signal_type == 'quantitative':
        manifest_path = "polymarket_data/manifest/_manifest_geopolitics__foreign_policy_comprehensive.csv"
        data_dir = "polymarket_data"
    else:
        if 'war' in sector.lower():
            manifest_path = "polymarket_data/manifest/_manifest_geopolitics__war.csv"
            data_dir = "polymarket_data/geopolitics__war"
        elif 'trade' in sector.lower():
            manifest_path = "polymarket_data/manifest/_manifest_geopolitics__foreign_policy_comprehensive.csv"
            data_dir = "polymarket_data/custom_world_affairs"  # Contains trade deals
        else:
            manifest_path = "polymarket_data/manifest/_manifest_geopolitics__foreign_policy_comprehensive.csv"
            data_dir = "polymarket_data/geopolitics__foreign_policy_comprehensive"

    print(f"\n🔧 Configuration:")
    print(f"   Sector: {sector}")
    print(f"   Type: {signal_type.upper()}")
    print(f"   Min Volume: {volume_filter:,.0f}")
    print(f"   Manifest: {manifest_path}")
    print(f"   Data Dir: {data_dir}")
    print(f"   Output: {output_file}")

    # Confirm
    confirm = input(f"\n✅ Start processing? [Y/n]: ").strip().lower()
    if confirm in ['', 'y', 'yes']:
        # Initialize processor
        processor = SignalProcessor(manifest_path, data_dir)

        try:
            print(f"\n🚀 Processing {sector} signals...")

            if signal_type == 'quantitative':
                df = processor.generate_quantitative_signal(sector, output_file)
            elif signal_type == 'binary':
                df = processor.generate_binary_signal(sector, output_file)

            if not df.empty:
                print(f"\n✅ Success! Generated {len(df)} signal data points")
                print(f"   📁 Saved to: {output_file}")
                print(f"   📅 Date range: {df['date'].min()} to {df['date'].max()}")
                print(f"   📈 Signal range: {df['signal'].min():.4f} to {df['signal'].max():.4f}")
            else:
                print("❌ No signals generated - check configuration and data")

        except Exception as e:
            print(f"❌ Processing failed: {e}")

    print("\n👋 Thank you for using Polymarket Signal Engineering!")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Process Polymarket data into signals")
    parser.add_argument("--sector", help="Sector name (e.g., 'US Inflation', 'Geopolitics')")
    parser.add_argument("--type", choices=['quantitative', 'binary'],
                        help="Signal type: quantitative (VWEV) or binary (unsigned VWP)")
    parser.add_argument("--volume-filter", type=float, default=10000,
                        help="Minimum volume filter")
    parser.add_argument("--manifest", default="polymarket_data/manifest/_manifest_geopolitics__foreign_policy_comprehensive.csv",
                        help="Path to manifest CSV file")
    parser.add_argument("--data-dir", default="polymarket_data",
                        help="Directory containing the market data CSVs")
    parser.add_argument("--output", help="Output CSV file path")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode")

    args = parser.parse_args()

    # Interactive mode
    if args.interactive or not any([args.sector, args.type]):
        interactive_menu()
        return

    # Command-line mode
    if not args.sector or not args.type:
        print("❌ Error: --sector and --type are required for non-interactive mode")
        print("💡 Try: python CORE/signal_processor.py --interactive")
        return

    # Set default output path
    if not args.output:
        args.output = f"raw_signal_{args.sector.lower().replace(' ', '_')}.csv"

    # Initialize processor
    processor = SignalProcessor(args.manifest, args.data_dir)

    try:
        if args.type == 'quantitative':
            df = processor.generate_quantitative_signal(args.sector, args.output)
        elif args.type == 'binary':
            df = processor.generate_binary_signal(args.sector, args.output)

        if not df.empty:
            logger.info("✅ Signal processing completed successfully")
        else:
            logger.error("❌ No signals generated - check configuration and data")

    except Exception as e:
        logger.error(f"Signal processing failed: {e}")
        raise


if __name__ == "__main__":
    main()
