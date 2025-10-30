# Signal Development Template

This guide provides templates and examples for adding new signal types to the Unified Analysis System.

## Directory Structure

```
src/
├── signals/           # Core signal generation logic
│   ├── unified_signal_generator.py    # Main signal generation class
│   ├── signal_processor.py            # Signal processing utilities
│   └── polymarket_fetcher.py          # Data fetching for signals
└── analysis/          # Signal analysis and backtesting
    ├── unified_analyzer.py            # Main analysis API
    ├── feature_factory.py             # Signal combination logic
    └── data_preprocessor.py           # Data preparation utilities

tests/signals/         # Signal-specific tests
config/signals/        # Signal configuration files
```

## Adding a New Signal Type

### Step 1: Define the Signal Concept

Add your signal concept to `src/signals/unified_signal_generator.py` in the `_define_signal_concepts()` method:

```python
'your_signal_name': SignalConcept(
    name="Your Signal Description",
    keywords=['keyword1', 'keyword2', 'keyword3'],
    market_type=MarketType.BINARY,  # Use BINARY for polarity-based signals
    polarity_logic={  # ONLY for BINARY markets
        'positive_term': SignalPolarity.PRO_CONCEPT,
        'negative_term': SignalPolarity.ANTI_CONCEPT,
        'another_positive': SignalPolarity.PRO_CONCEPT,
    },
    # Note: parser_func is NOT used for quantitative signals (see Step 3)
    min_volume=1000.0
),
```

### Step 2: Define Polarity Mapping (BINARY Markets Only)

For binary markets, define how question wording maps to signal polarity:

```python
polarity_logic={
    'win': SignalPolarity.PRO_CONCEPT,      # P_yes indicates positive outcome
    'lose': SignalPolarity.ANTI_CONCEPT,    # P_yes indicates negative outcome
    'higher': SignalPolarity.PRO_CONCEPT,    # Increase = positive
    'lower': SignalPolarity.ANTI_CONCEPT,    # Decrease = negative
    'above': SignalPolarity.PRO_CONCEPT,     # Above threshold = positive
    'below': SignalPolarity.ANTI_CONCEPT,    # Below threshold = negative
}
```

### Step 3: Quantitative Signals - CORRECT Methodology

**CRITICAL: Do NOT use the parser_func approach below for quantitative signals.**

For quantitative signals, **you MUST use the Expected Value from Distribution methodology**. Example:

For inflation signals, instead of parsing static numbers from text:

```python
# ❌ INCORRECT - This throws away the market price!
def _parse_inflation_value(self, question: str) -> float:
    """WRONG: Returns static number from question, ignores price"""
    if "above 3%" in question:
        return 3.0  # Wrong - ignores market's actual probability

# ✅ CORRECT - Group markets and calculate Expected Value
def calculate_inflation_ev_from_markets(self, related_market_tickers: List[str]) -> float:
    """Correct: Use prices across related markets to build probability distribution"""
    # Group markets like: P(>2%), P(>3%), P(>4%) for same event
    # Derive bins: P(2-3%) = P(>2%) - P(>3%)
    # Calculate EV: Σ(midpoint × bin_probability)
    return expected_value  # e.g., 3.42% for June CPI
```

Then connect to the Rolling Signal Generator - this is **already implemented** in `src/signal_generators/vwev_rolled_signal.py`. Use a config file:

```yaml
# config/signals/inflation_vwev.yaml
signal_name: "US CPI Inflation Expected Value"
generator_type: "vwev_rolled"

event_grouping_pattern: r'(\w+ \d{4})'  # Extracts "June 2025", "July 2025"
roll_on_field: "volume"                 # Volume-based roll detection
open_bin_handling_method: "fit_pareto" # Handle open-ended distributions

input_manifest_files:
  - "polymarket_data/manifest/_manifest_inflation.csv"

distribution_bins:
  percentiles: [5, 10, 20, 30, 50, 70, 80, 90, 95]
  min_sample_size: 3
  fit_tolerance: 1e-6

output_file: "data/processed_signals/signal_vwev_inflation.csv"
```

### Step 4: Add SignalType Enum and Loading

Add your signal to the SignalType enum in `src/analysis/unified_analyzer.py`:

```python
class SignalType(Enum):
    # Existing signals...
    YOUR_SIGNAL = "your_signal_name"
```

### Step 5: Add Data Loading Logic

Add data loading logic in the `_load_signal_data` method:

```python
elif signal_type == SignalType.YOUR_SIGNAL:
    # Load your signal data
    filepath = root_dir / "your_signal_file.csv"
    df = pd.read_csv(filepath)
    df['y'] = df['signal_column_name']
```

### Step 6: Create Configuration File

Create `config/signals/your_signal.yaml`:

```yaml
signal_name: "Your Signal Name"
generator_type: "unified"  # or "vwp" for simplified generation

# For unified generation
concept_key: "your_signal_name"

# For manifest-based generation
input_manifest_files:
  - "polymarket_data/_manifest_your_category.csv"

min_quality_score: 0.6  # Filter out low-quality markets
output_file: "signal_your_signal.csv"

# Polarity rules for simplified generation
polarity_rules:
  positive_keywords: ["win", "higher", "above"]
  negative_keywords: ["lose", "lower", "below"]
```

### Step 7: Add Tests

Create `tests/signals/test_your_signal.py`:

```python
#!/usr/bin/env python3
"""
Tests for Your Signal implementation
"""

from src.signals.unified_signal_generator import UnifiedSignalGenerator
from src.analysis.unified_analyzer import SignalTesterAPI, AnalysisConfig, AssetType, SignalType
import pytest

class TestYourSignal:
    def test_concept_definition(self):
        """Test that your signal concept is properly defined"""
        generator = UnifiedSignalGenerator()
        concepts = generator.concepts

        assert 'your_signal_name' in concepts
        concept = concepts['your_signal_name']
        assert concept.name == "Your Signal Description"
        assert concept.market_type == MarketType.BINARY  # or QUANTITATIVE

    def test_signal_generation(self):
        """Test signal generation produces valid output"""
        generator = UnifiedSignalGenerator()
        signal_df = generator.generate_concept_signal_unified('your_signal_name')

        assert not signal_df.empty
        assert 'date' in signal_df.columns
        assert 'signal' in signal_df.columns

        # Check signal range
        assert signal_df['signal'].min() >= -1.0
        assert signal_df['signal'].max() <= 1.0

    def test_analysis_integration(self):
        """Test signal works with analysis system"""
        config = AnalysisConfig(
            asset_type=AssetType.SP500,
            signal_type=SignalType.YOUR_SIGNAL,
            strategy_type=StrategyType.MOMENTUM,
            strategy_direction=StrategyDirection.LONG_SHORT,
            strategy_entry_percentile=0.80
        )

        tester = SignalTesterAPI(config=config)
        # This would run the full analysis stack
        # For testing, you might want to mock or use sample data
```

### Step 8: Update Documentation

Update this template with any unique aspects of your signal implementation.

## Examples

### Binary Signal Example: Election Results

```python
'election_results': SignalConcept(
    name="Election Outcome Prediction",
    keywords=['election', 'president', 'candidate'],
    market_type=MarketType.BINARY,
    polarity_logic={
        'biden': SignalPolarity.PRO_CONCEPT,   # P_yes = Biden win
        'trump': SignalPolarity.ANTI_CONCEPT,  # P_yes = Trump win
        'democrat': SignalPolarity.PRO_CONCEPT,
        'republican': SignalPolarity.ANTI_CONCEPT,
    },
    min_volume=10000.0  # High volume for political markets
)
```

### Quantitative Signal Example: CPI Inflation

```python
'us_inflation': SignalConcept(
    name="US Inflation Pressure",
    keywords=['inflation', 'cpi', 'price increases'],
    market_type=MarketType.QUANTITATIVE,
    polarity_logic={
        'higher': SignalPolarity.PRO_CONCEPT,
        'lower': SignalPolarity.ANTI_CONCEPT,
        'rising': SignalPolarity.PRO_CONCEPT,
        'falling': SignalPolarity.ANTI_CONCEPT,
    },
    parser_func=self._parse_inflation_value,
    min_volume=2000.0
)
```

## Best Practices

1. **Keywords**: Use specific, non-overlapping keywords to avoid conflicts
2. **Volume Thresholds**: Set appropriate minimum volume based on market liquidity
3. **Polarity Logic**: Carefully define how question wording maps to signal direction
4. **Parser Functions**: Handle edge cases in quantitative value extraction
5. **Testing**: Add comprehensive tests for both generation and analysis
6. **Documentation**: Update this template with your specific implementation details

## Configuration Reference

### Signal Concept Parameters

- `name`: Human-readable description
- `keywords`: List of strings to match in market questions
- `market_type`: `BINARY` or `QUANTITATIVE`
- `polarity_logic`: Dict mapping question terms to signal polarity
- `parser_func`: Function to extract quantitative values (quantitative only)
- `min_volume`: Minimum market volume required
- `manual_excludes`: List of market filenames to exclude
- `manual_includes`: List of market filenames to force include
- `bert_model`: Future BERT model for advanced classification

### Analysis Configuration Parameters

- `signal_lookback_months`: Historical window for percentile calculation (default: 6)
- `strategy_entry_percentile`: Signal threshold for trade entry (default: 0.80)
- `forward_periods_map`: Forecast horizons to analyze (default: 1w, 1m, 3m, 6m)
- `initial_capital`: Starting portfolio value (default: $10,000)
- `min_signal_data_days`: Minimum historical data required (default: 252)
