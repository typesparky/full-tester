#!/usr/bin/env python3
"""
Test CPI Inflation signal on SP500 with 80% threshold - SAME AS MILITARY STYLE
"""

import sys
import os
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

def test_cpi_inflation_momentum():
    """Test CPI Inflation signal on SP500 with MOMENTUM strategy"""
    print("🇺🇸 Testing CPI INFLATION signal on SP500 with MOMENTUM strategy")
    print("   - Polymarket VWEV expected annual inflation, pct_change as signal")
    print("   - Momentum: ride rising signals, buy strong trends")
    print("   - Long-Short variant for full market exposure")
    print("   - Creates full performance dashboards")

    # Test LONG_SHORT momentum strategy with 80th percentile
    strategies_to_test = [
        (StrategyDirection.LONG_SHORT, StrategyType.MOMENTUM, 0.80, "Long-Short: Buy at 80th %ile (strong rise), sell at 20th %ile (strong fall)")
    ]

    for strategy_dir, strategy_type, entry_pct, description in strategies_to_test:
        print(f"\n🎯 Testing {description}...")
        print("="*70)

        config = AnalysisConfig(
            asset_type=AssetType.SP500,
            signal_type=SignalType.INFLATION,
            strategy_type=strategy_type,
            strategy_direction=strategy_dir,
            strategy_entry_percentile=entry_pct
        )

        try:
            tester = SignalTesterAPI(config=config)
            tester.run()
            print(f"✅ {strategy_dir.value.replace('_', '-').title()} Analysis completed successfully!")
            print("   📊 Dashboard PNG saved to analysis_results/")
        except Exception as e:
            print(f"❌ {strategy_dir.value.replace('_', '-').title()} Analysis failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_cpi_inflation_momentum()
    print("\n🎊 CPI INFLATION SIGNAL MOMENTUM ANALYSIS COMPLETE!")
    print("📈 Dashboards generated with same methodology as military signals")
    print("📊 Check analysis_results/ for full performance visualizations")
