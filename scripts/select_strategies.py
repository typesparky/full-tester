#!/usr/bin/env python3
"""
PREMADE MILITARY SIGNAL STRATEGY SELECTOR

Interactive menu to select from premade trading strategies defined in config/backtest_strategies.yaml.
Users can choose entry strategies and exit types to run comprehensive backtests.
"""

import sys
import os
import yaml
from pathlib import Path
from typing import List, Tuple

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

def load_strategies() -> dict:
    """Load strategy configurations from YAML file."""
    strategy_file = Path("config/backtest_strategies.yaml")
    if not strategy_file.exists():
        print("❌ Strategy configuration file not found: config/backtest_strategies.yaml")
        return {}

    try:
        with open(strategy_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"❌ Error reading strategy configuration: {e}")
        return {}

def create_config_from_strategy(strategy_key: str, exit_logic: str, config_data: dict) -> AnalysisConfig:
    """Create AnalysisConfig object from strategy definition."""
    if 'strategies' not in config_data or strategy_key not in config_data['strategies']:
        raise ValueError(f"Strategy '{strategy_key}' not found in configuration")

    strategy_def = config_data['strategies'][strategy_key]

    # Map strategy types
    strategy_type_map = {
        "momentum": StrategyType.MOMENTUM,
        "mean_reversion": StrategyType.MEAN_REVERSION
    }

    # Map direction types
    direction_map = {
        "long_short": StrategyDirection.LONG_SHORT,
        "long_only": StrategyDirection.LONG_ONLY,
        "short_only": StrategyDirection.SHORT_ONLY
    }

    # Map exit logic
    exit_logic_map = {
        "signal_neutral": ExitLogic.SIGNAL_NEUTRAL,
        "atr_trailing_stop": ExitLogic.ATR_TRAILING_STOP
    }

    # Get exit parameters
    exit_params = config_data.get('exit_logic_options', {}).get(exit_logic, {}).get('params', {})

    return AnalysisConfig(
        asset_type=AssetType.SP500,  # Hardcoded to SP500 for military signals
        signal_type=SignalType.MILITARY_ACTIONS,  # Hardcoded for this demo
        strategy_type=strategy_type_map[strategy_def['strategy_type']],
        strategy_direction=direction_map[strategy_def['strategy_direction']],
        strategy_entry_percentile=strategy_def['entry_percentile_high'],  # Using high threshold
        exit_logic=exit_logic_map[exit_logic],
        exit_params=exit_params
    )

def display_strategy_menu(strategies: dict) -> List[Tuple[str, str]]:
    """Display interactive menu for strategy selection."""
    print("\n" + "="*80)
    print("🎯 MILITARY ACTIONS SIGNAL - STRATEGY SELECTION")
    print("="*80)
    print("Available Entry Strategies:")
    print("-"*40)

    # Display entry strategies
    entry_strategies = list(strategies.get('strategies', {}).keys())
    for i, strategy_key in enumerate(entry_strategies, 1):
        strategy_def = strategies['strategies'][strategy_key]
        name = strategy_def.get('name', strategy_key)
        description = strategy_def.get('description', 'No description available')
        entry_high = strategy_def.get('entry_percentile_high', 0) * 100
        entry_low = strategy_def.get('entry_percentile_low', 0) * 100

        print(f"{i:3d} {name}")
        print(f"      {description}")
        print(f"      Entry: Long >{entry_high:.0f}% | Short <{max(entry_low, 1):.0f}%")
        print()

    # Display exit logic options
    exit_options = list(strategies.get('exit_logic_options', {}).keys())
    print("Available Exit Logics:")
    print("-"*40)

    for i, exit_key in enumerate(exit_options, 1):
        exit_def = strategies['exit_logic_options'][exit_key]
        name = exit_def.get('name', exit_key)
        description = exit_def.get('description', 'No description')
        print(f"{i}. {name}")
        print(f"   {description}")
        print()

    # Get user selections
    print("\n📝 SELECTION TIME")
    print("-"*30)

    selected_strategies = []
    selections = input("Enter entry strategy numbers (comma-separated, e.g., 1,3,5): ").strip()

    if selections.upper() == "ALL":
        selected_entry = entry_strategies
    else:
        try:
            indices = [int(x.strip()) - 1 for x in selections.split(',')]
            selected_entry = [entry_strategies[i] for i in indices if 0 <= i < len(entry_strategies)]
        except (ValueError, IndexError):
            print("❌ Invalid selection. Using first strategy as default.")
            selected_entry = [entry_strategies[0]]

    print(f"Selected entry strategies: {', '.join(selected_entry)}")

    # Exit logic selection
    exit_selection = input("Select exit logic (1=Signal Neutral, 2=ATR Trailing Stop) [default: 1]: ").strip()
    if exit_selection == "2":
        selected_exit = "atr_trailing_stop"
        print("Selected: ATR Trailing Stop")
    else:
        selected_exit = "signal_neutral"
        print("Selected: Signal Neutral Exit")

    # Combine all combinations
    combinations = []
    for entry in selected_entry:
        combinations.append((entry, selected_exit))

    print(f"\n🚀 Will run {len(combinations)} strategy combinations:")
    for entry, exit_logic in combinations:
        strategy_name = strategies['strategies'][entry].get('name', entry)
        exit_name = strategies['exit_logic_options'][exit_logic].get('name', exit_logic)
        print(f"   • {strategy_name} + {exit_name}")

    return combinations

def run_selected_strategies(combination_list: List[Tuple[str, str]], config_data: dict):
    """Run the selected strategy combinations."""
    print(f"\n{'='*80}")
    print("🚀 EXECUTING STRATEGY BACKTESTS")
    print(f"{'='*80}")

    results_summary = []

    for i, (strategy_key, exit_logic) in enumerate(combination_list, 1):
        strategy_name = config_data['strategies'][strategy_key].get('name', strategy_key)
        exit_name = config_data['exit_logic_options'][exit_logic].get('name', exit_logic)

        print(f"\n[{i}/{len(combination_list)}] Running: {strategy_name}")
        print(f"Exit Logic: {exit_name}")
        print("-" * 50)

        try:
            # Create configuration
            config = create_config_from_strategy(strategy_key, exit_logic, config_data)

            # Run analysis
            tester = SignalTesterAPI(config=config)
            tester.run()

            # Store results for summary
            signal_score = 75.0  # Placeholder - would extract from actual results
            results_summary.append({
                'strategy': f"{strategy_name}",
                'exit_logic': f"{exit_name}",
                'signal_score': signal_score
            })

            print(f"✅ Strategy {i} completed successfully!")

        except Exception as e:
            print(f"❌ Strategy {i} failed: {e}")
            continue

    # Display final summary
    print(f"\n{'='*80}")
    print("🎊 ANALYSIS SUMMARY - MILITARY ACTIONS STRATEGIES")
    print(f"{'='*80}")

    if results_summary:
        print("Strategy Performance Rankings:")
        print("-" * 50)

        # Sort by signal score (placeholder)
        sorted_results = sorted(results_summary, key=lambda x: x['signal_score'], reverse=True)

        for i, result in enumerate(sorted_results, 1):
            score = result['signal_score']
            strategy = result['strategy']
            exit_logic = result['exit_logic']
            print("2d")

        print("
📁 Dashboards and charts saved to 'analysis_results/' folder"        print("📊 Compare different strategies side-by-side for robust signal analysis!")

    print(f"\n🎉 COMPLETED: {len(results_summary)}/{len(combination_list)} strategies analyzed successfully!")

def main():
    """Main execution flow."""
    print("🇺🇳 MILITARY ACTIONS SIGNAL ANALYSIS SYSTEM")
    print("   - Premade strategy combinations for signal testing")
    print("   - ATR trailing stops and signal neutral exits")
    print("   - Side-by-side performance comparisons")

    # Load strategy configurations
    config_data = load_strategies()
    if not config_data:
        print("❌ No strategy configurations found. Exiting.")
        return

    # Display menu and get selections
    selected_combinations = display_strategy_menu(config_data)

    if not selected_combinations:
        print("❌ No strategies selected. Exiting.")
        return

    # Confirm and execute
    confirm = input(f"\n🚀 Ready to run {len(selected_combinations)} strategy backtests. Continue? (y/N): ").strip().lower()
    if confirm in ['y', 'yes']:
        run_selected_strategies(selected_combinations, config_data)
    else:
        print("❌ Execution cancelled by user.")

if __name__ == "__main__":
    main()
