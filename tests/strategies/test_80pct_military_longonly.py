#!/usr/bin/env python3
"""
Test MILITARY_ACTIONS signal on SP500 with 80% threshold - LONG ONLY
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

def test_military_longonly():
    """Test MILITARY signal on SP500 with 80% threshold, LONG ONLY"""
    print("🇺🇳 Testing MILITARY ACTIONS signal on SP500 with 80% threshold (LONG ONLY)")

    config = AnalysisConfig(
        asset_type=AssetType.SP500,
        signal_type=SignalType.MILITARY_ACTIONS,
        strategy_type=StrategyType.MOMENTUM,
        strategy_direction=StrategyDirection.LONG_ONLY,  # Long only as requested
        strategy_entry_percentile=0.80  # 80% threshold
    )

    try:
        tester = SignalTesterAPI(config=config)
        tester.run()
        print("✅ LONG ONLY Analysis completed successfully!")
    except Exception as e:
        print(f"❌ LONG ONLY Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_military_longonly()
