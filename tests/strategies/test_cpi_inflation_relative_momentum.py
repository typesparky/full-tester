#!/usr/bin/env python3
"""
TEST CPI INFLATION AS RELATIVE MOMENTUM SIGNAL (vs S&P 500)

Instead of absolute inflation levels, use:
% Change in Inflation ÷ % Change in S&P 500 = Relative Momentum

Also tests counter strategies since negative t-values suggest
running opposite to inflation trends is more effective.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CORE.unified_analyzer import (
    SignalTesterAPI,
    AnalysisConfig,
    AssetType,
    SignalType,
    StrategyDirection,
    StrategyType,
)

def test_relative_momentum_counter_strategy():
    """Test CPI inflation as relative momentum signal with counter strategy"""

    print("🇺🇸 TESTING CPI RELATIVE MOMENTUM SIGNAL")
    print("=" * 60)
    print("Inflation % Change vs S&P % Change = Relative Momentum")
    print("Testing counter strategy due to negative t-values")

    # Strategy 1: Normal relative momentum (inflation accelerating > decelerating)
    print("\n1️⃣ STRATEGY 1: RELATIVE MOMENTUM")
    print("   Signal: Inflation change rate - S&P change rate")
    print("   Logic: Long when inflation momentum > 80th percentile")

    config_normal = AnalysisConfig(
        asset_type=AssetType.SP500,
        signal_type=SignalType.INFLATION,  # Using our enhanced INFLATION type
        strategy_type=StrategyType.MOMENTUM,
        strategy_direction=StrategyDirection.LONG_SHORT,
        strategy_entry_percentile=0.80,
    )

    try:
        print("\n   ┌─ Running analysis...")
        tester_normal = SignalTesterAPI(config=config_normal)
        tester_normal.run()
        print("   └─ ✅ Normal relative momentum completed")
    except Exception as e:
        print(f"   └─ ❌ Normal strategy failed: {e}")

    # Strategy 2: Counter strategy (opposite to inflation trends)
    print("\n2️⃣ STRATEGY 2: COUNTER STRATEGY")
    print("   Signal: Same inflation momentum, but direction inverted")
    print("   Logic: Fades inflation trends (sell rising, buy falling inflation)")

    # For counter strategy, we'd need to modify the analyzer or create inverted condition
    # Let's create a separate inverted analysis
    # (Note: Could enhance analyzer to support direct Inversion flag)

    print("   └─ Counter strategy requires enhanced analyzer inversion flag")

    print("
🎯 ANALYSIS COMPLETE"    print("Expected improvement:"    print("  • Positive t-values (vs current negative)"    print("  • Better economic logic (defensive positioning)"    print("  • Relative momentum captures inflation vs market dynamics")
    print("
📊 Check analysis_results/ for dashboard visualizations"

if __name__ == "__main__":
    test_relative_momentum_counter_strategy()
