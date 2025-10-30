#!/usr/bin/env python3
"""
Test MILITARY_ACTIONS signal on SP500 with 80% threshold - BOTH LONG_SHORT AND LONG_ONLY
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

def test_military_80pct():
    """Test MILITARY signal on SP500 with 80% threshold - both strategies"""
    print("🇺🇳 Testing MILITARY ACTIONS signal on SP500 with 80% threshold")
    print("   - 7-day forward returns with percentiles: 1,5,10,20,80,90,95,99")

    # Test both LONG_SHORT and LONG_ONLY
    for strategy_dir in [StrategyDirection.LONG_SHORT, StrategyDirection.LONG_ONLY]:
        print(f"\n🎯 Running {strategy_dir.value.replace('_', '-').title()} Analysis...")
        print("="*60)

        config = AnalysisConfig(
            asset_type=AssetType.SP500,
            signal_type=SignalType.MILITARY_ACTIONS,
            strategy_type=StrategyType.MOMENTUM,
            strategy_direction=strategy_dir,
            strategy_entry_percentile=0.80  # 80% threshold as requested
        )

        try:
            tester = SignalTesterAPI(config=config)
            tester.run()
            print(f"✅ {strategy_dir.value.replace('_', '-').title()} Analysis completed successfully!")
        except Exception as e:
            print(f"❌ {strategy_dir.value.replace('_', '-').title()} Analysis failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_military_80pct()
