#!/usr/bin/env python3
"""
Multi-asset military signal analysis: BTC, GOLD, OIL, SP500
Testing momentum + ATR strategy across different assets
"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CORE.unified_analyzer import (
    SignalTesterAPI,
    AnalysisConfig,
    AssetType,
    SignalType,
    StrategyDirection,
    StrategyType,
    ExitLogic,
)

def run_multi_asset_test():
    """Run momentum + ATR strategy on all major assets"""

    assets_to_test = [AssetType.BTC, AssetType.GOLD, AssetType.OIL, AssetType.SP500]
    results = []

    print("🌍 MULTI-ASSET MILITARY SIGNAL ANALYSIS")
    print("🇺🇳 Strategy: Momentum (Long/Short) + ATR Trailing Stop")
    print("="*60)

    for asset_type in assets_to_test:
        asset_name = asset_type.value.upper()
        print(f"\n🚀 Analyzing {asset_name}...")
        print("-"*40)

        # Create configuration for momentum strategy with ATR exits
        config = AnalysisConfig(
            asset_type=asset_type,
            signal_type=SignalType.MILITARY_ACTIONS,
            strategy_type=StrategyType.MOMENTUM,
            strategy_direction=StrategyDirection.LONG_SHORT,
            strategy_entry_percentile=0.80,  # Long >80%, Short <20%
            exit_logic=ExitLogic.ATR_TRAILING_STOP,
            exit_params={
                'atr_period': 14,
                'atr_multiplier': 1.5,
                'atr_timeframe': 'DAILY'
            }
        )

        try:
            tester = SignalTesterAPI(config=config)
            tester.run()
            print(f"✅ {asset_name} completed successfully!")
            results.append(f"✅ {asset_name}")
        except Exception as e:
            print(f"❌ {asset_name} failed: {e}")
            results.append(f"❌ {asset_name}")

    print(f"\n{'='*60}")
    print("🎊 MULTI-ASSET RESULTS SUMMARY")
    print(f"{'='*60}")

    for result in results:
        print(result)

    print(f"\n📊 All dashboards saved to 'analysis_results/' folder")
    print(f"🔍 File patterns: *{SignalType.MILITARY_ACTIONS.value}_{StrategyType.MOMENTUM.value}_long_short_80pct_atr*")

if __name__ == "__main__":
    print("Launching multi-asset military signal analysis...")
    run_multi_asset_test()
    print("✅ Analysis complete!")
