#!/usr/bin/env python3
"""
Simple test of one premade strategy with ATR exits
"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.unified_analyzer import (
    SignalTesterAPI,
    AnalysisConfig,
    AssetType,
    SignalType,
    StrategyDirection,
    StrategyType,
    ExitLogic,
)

def test_momentum_with_atr(asset_type):
    """Test momentum strategy with ATR exits on military signal"""
    asset_name = asset_type.value.upper()
    print(f"🇺🇳 Testing: Momentum (Long/Short) + ATR Trailing Stop on {asset_name}")

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
        print(f"✅ Momentum + ATR strategy completed successfully on {asset_name}!")
        print("📊 Check analysis_results/ for the dashboard")
        return True
    except Exception as e:
        print(f"❌ Strategy failed on {asset_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mean_reversion_with_signal_exit():
    """Test mean reversion strategy with signal neutral exits"""
    print("🇺🇳 Testing: Mean Reversion (Long/Short) + Signal Neutral Exit")

    config = AnalysisConfig(
        asset_type=AssetType.SP500,
        signal_type=SignalType.MILITARY_ACTIONS,
        strategy_type=StrategyType.MEAN_REVERSION,
        strategy_direction=StrategyDirection.LONG_SHORT,
        strategy_entry_percentile=0.10,  # Entry logic will be inverted for mean reversion
        exit_logic=ExitLogic.SIGNAL_NEUTRAL,
        exit_params={
            'neutral_band': [0.40, 0.60]
        }
    )

    try:
        tester = SignalTesterAPI(config=config)
        tester.run()
        print("✅ Mean Reversion + Signal strategy completed successfully!")
        print("📊 Check analysis_results/ for the dashboard")
    except Exception as e:
        print(f"❌ Strategy failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🎯 Premade Strategy Testing - Choose one:")
    print("1. Momentum (Long/Short) + ATR Trailing Stop")
    print("2. Mean Reversion (Long/Short) + Signal Neutral Exit")

    choice = input("Select strategy (1 or 2): ").strip()

    if choice == "1":
        test_momentum_with_atr()
    elif choice == "2":
        test_mean_reversion_with_signal_exit()
    else:
        print("❌ Invalid choice. Running default momentum test...")
        test_momentum_with_atr()
