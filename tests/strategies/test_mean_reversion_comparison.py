#!/usr/bin/env python3
"""
Mean Reversion Strategy Test on BTC and OIL
Testing if counter-trend works better on these assets
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

def test_mean_reversion_on_assets():
    """Test mean reversion strategy on BTC and OIL with military signal"""

    assets_to_test = [AssetType.BTC, AssetType.OIL]
    results = []

    print("🔄 MEAN REVERSION STRATEGY TEST")
    print("🇺🇳 Testing Long/Short Mean Reversion + ATR Trailing Stop")
    print("="*70)
    print("Strategy: Mean Reversion (Buy The Dip, Sell The Rip)")
    print("Entry: Long <10% signal, Short >90% signal")
    print("Exit: ATR Trailing Stop")
    print("="*70)

    for asset_type in assets_to_test:
        asset_name = asset_type.value.upper()
        print(f"\n🚀 Testing {asset_name}...")
        print("-"*40)

        # Mean reversion configuration
        config = AnalysisConfig(
            asset_type=asset_type,
            signal_type=SignalType.MILITARY_ACTIONS,
            strategy_type=StrategyType.MEAN_REVERSION,  # Mean reversion instead of momentum
            strategy_direction=StrategyDirection.LONG_SHORT,
            strategy_entry_percentile=0.10,  # Entry at extreme percentiles (reversed)
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
            print(f"✅ {asset_name} Mean Reversion completed!")
            results.append(f"✅ {asset_name}")
        except Exception as e:
            print(f"❌ {asset_name} failed: {e}")
            results.append(f"❌ {asset_name}")

    print(f"\n{'='*70}")
    print("🎊 MEAN REVERSATION RESULTS SUMMARY")
    print(f"{'='*70}")

    for result in results:
        print(result)

    print(f"\n📊 Files saved to 'analysis_results/' folder")
    print(f"🔍 Look for: *{SignalType.MILITARY_ACTIONS.value}_{StrategyType.MEAN_REVERSION.value}_long_short_10pct_atr*")

def compare_strategies():
    """Compare momentum vs mean reversion performance"""
    print("\n📊 PERFORMANCE COMPARISON:")
    print("-"*70)
    print("BTC Impulse Strategy:")
    print("  Momentum (Trend Following): 24.9/100")
    print("  Mean Reversion (Counter-Trend): ?/100 ← Current test")
    print()
    print("OIL Impulse Strategy:")
    print("  Momentum (Trend Following): 12.9/100")
    print("  Mean Reversion (Counter-Trend): ?/100 ← Current test")
    print("-"*70)

if __name__ == "__main__":
    print("Testing Mean Reversion Strategy on BTC & OIL...")
    print("Idea: Some assets might respond better to counter-trend trades")
    print("With the military signal acting as an 'over-reaction' indicator.")
    print()

    test_mean_reversion_on_assets()
    compare_strategies()

    print(f"\n✅ Mean Reversion analysis complete!")
    print("This will show if BTC & OIL respond better to counter-trend trading!")
