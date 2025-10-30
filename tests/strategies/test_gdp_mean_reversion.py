#!/usr/bin/env python3
"""
Test GDP Growth signal on SP500 with MEAN REVERSION strategies
Similar setup to inflation analysis but for GDP growth expectations
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

def test_gdp_mean_reversion():
    """Test GDP Growth signal on SP500 with MEAN REVERSION strategies"""
    print("🌐 Testing GDP GROWTH signal on SP500 with MEAN REVERSION strategies")
    print("   - Polymarket VWEV expected GDP growth, pct_change as signal")
    print("   - Mean reversion: buy low (20th percentile), sell when normalized")
    print("   - Tests Long-Only and Long-Short variants for GDP markets")
    print("   - Creates full performance dashboards")

    # Test LONG_ONLY and LONG_SHORT mean reversion strategies with 20th/80th percentiles
    strategies_to_test = [
        (StrategyDirection.LONG_ONLY, StrategyType.MEAN_REVERSION, 0.20, "Long-Only: Buy at 20th %ile (moderate low), sell at 50th %ile"),
        (StrategyDirection.LONG_SHORT, StrategyType.MEAN_REVERSION, 0.20, "Long-Short: Buy at 20th %ile, sell at 80th %ile (moderate mean reversion)")
    ]

    for strategy_dir, strategy_type, entry_pct, description in strategies_to_test:
        print(f"\n🎯 Testing {description}...")
        print("="*70)

        config = AnalysisConfig(
            asset_type=AssetType.SP500,
            signal_type=SignalType.GDP,
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
    # Note: GDP signal is loaded from signal_vwev_gdp.csv which should be generated
    # using the unified signal generator with the GDP market manifest
    test_gdp_mean_reversion()
    print("\n🚀 GDP GROWTH SIGNAL MEAN REVERSION ANALYSIS COMPLETE!")
    print("📈 Dashboards generated with same methodology as inflation signals")
    print("📊 Check analysis_results/ for full performance visualizations")
    print("📝 Note: GDP signal uses Polymarket GDP market probabilities for expectations")
