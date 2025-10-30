# main_runner.py

import pandas as pd
import json
import logging
from pathlib import Path
from data_preprocessor import DataPreprocessor
from unified_analyzer import SignalTesterAPI, AnalysisConfig, AssetType, SignalType, StrategyDirection, StrategyType

# Set up logger for main_runner
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Step 1: Define Your Raw Data Loaders ---

def load_price_raw(asset_type: AssetType) -> pd.DataFrame:
    """Loads raw price data for a given asset."""
    logger.info(f"Attempting to load {asset_type.name} price data...")
    data_dir = Path("Data")

    try:
        if asset_type == AssetType.BTC:
            filepath = data_dir / "Bitcoin Historical Data.csv"
            df = pd.read_csv(filepath)
            logger.info(f"Loaded BTC CSV with columns: {list(df.columns)}")
            df.rename(columns={'Date': 'date', 'Price': 'price'}, inplace=True)
            df['date'] = pd.to_datetime(df['date'])
            df['price'] = df['price'].replace({',': ''}, regex=True).astype(float)

        elif asset_type == AssetType.ETH:
            filepath = data_dir / "eth_price_history.csv"
            df = pd.read_csv(filepath)
            logger.info(f"Loaded ETH CSV with columns: {list(df.columns)}")
            df.rename(columns={'Date': 'date', 'Close': 'price'}, inplace=True)
            df['date'] = pd.to_datetime(df['date'])
            df['price'] = df['price'].replace({r'\$': '', ',': ''}, regex=True).astype(float)
        else:
            raise NotImplementedError(f"Price loader not implemented for {asset_type.name}")

        if df['date'].dt.tz is not None:
            df['date'] = df['date'].dt.tz_localize(None)

        df.set_index('date', inplace=True)
        logger.info(f"Successfully loaded {len(df)} records from {filepath.name}")
        logger.info(f"Final DataFrame columns: {list(df.columns)}, shape: {df.shape}")

        # Ensure we return only the price column
        if 'price' not in df.columns:
            logger.error(f"ERROR: 'price' column not found after processing {asset_type.name} data!")
            logger.error(f"Available columns: {list(df.columns)}")
            return pd.DataFrame()

        return df[['price']]

    except FileNotFoundError:
        logger.error(f"FATAL: Price data file not found for {asset_type.name}. Aborting.")
        return pd.DataFrame()
    except KeyError as e:
        logger.error(f"FATAL: A required column is missing in the price data file for {asset_type.name}. Missing column: {e}. Aborting.")
        return pd.DataFrame()


def load_signal_raw(signal_type: SignalType) -> pd.DataFrame:
    """Loads raw signal data."""
    logger.info(f"Attempting to load {signal_type.name} signal data...")
    data_dir = Path("Data")

    try:
        if signal_type == SignalType.HASHRATE:
            filepath = data_dir / "hash-rate.json"
            with open(filepath, 'r') as f:
                data = json.load(f).get('hash-rate', [])
            logger.info(f"Loaded hashrate JSON data with {len(data)} records")
            df = pd.DataFrame(data, columns=['x', 'y'])
            logger.info(f"Created DataFrame with columns: {list(df.columns)}")
            df.rename(columns={'y': 'signal'}, inplace=True)
            df['date'] = pd.to_datetime(df['x'], unit='ms')
            logger.info(f"After renaming, columns: {list(df.columns)}")

        elif signal_type == SignalType.USDC_ISSUANCE:
            filepath = data_dir / "usdc-usd-max.csv"
            df = pd.read_csv(filepath)
            logger.info(f"Loaded USDC CSV with columns: {list(df.columns)}")
            df.rename(columns={'snapped_at': 'date', 'market_cap': 'signal'}, inplace=True)
            df['date'] = pd.to_datetime(df['date'])

        elif signal_type == SignalType.USDT_ISSUANCE:
            filepath = data_dir / "usdt-usd-max.csv"
            df = pd.read_csv(filepath)
            logger.info(f"Loaded USDT CSV with columns: {list(df.columns)}")
            df.rename(columns={'snapped_at': 'date', 'market_cap': 'signal'}, inplace=True)
            df['date'] = pd.to_datetime(df['date'])

        elif signal_type == SignalType.POLYMARKET_VWP:
            # Load generated Polymarket VWP signal
            filepath = Path("signal_vwp_china_trade.csv")
            df = pd.read_csv(filepath)
            logger.info(f"Loaded VWP signal CSV with columns: {list(df.columns)}")
            df.rename(columns={'date': 'date', 'signal': 'signal'}, inplace=True)
            df['date'] = pd.to_datetime(df['date'])

        elif signal_type == SignalType.POLYMARKET_VWEV:
            # Load generated Polymarket VWEV signal
            filepath = Path("signal_vwev_inflation.csv")
            df = pd.read_csv(filepath)
            logger.info(f"Loaded VWEV signal CSV with columns: {list(df.columns)}")
            df.rename(columns={'date': 'date', 'signal': 'signal'}, inplace=True)
            df['date'] = pd.to_datetime(df['date'])

        elif signal_type == SignalType.MASTER_MACRO:
            # Load Master Macro signal (generated by feature_factory.py)
            filepath = Path("signal_master_macro.csv")
            df = pd.read_csv(filepath)
            logger.info(f"Loaded Master Macro signal CSV with columns: {list(df.columns)}")
            df.rename(columns={'date': 'date', 'master_signal': 'signal'}, inplace=True)
            df['date'] = pd.to_datetime(df['date'])

        elif signal_type == SignalType.MASTER_INFLATION:
            # Load Master Inflation signal (generated by feature_factory.py)
            filepath = Path("sector_signals/signal_master_inflation.csv")
            df = pd.read_csv(filepath)
            logger.info(f"Loaded Master Inflation signal CSV with columns: {list(df.columns)}")
            df.rename(columns={'date': 'date', 'master_signal': 'signal'}, inplace=True)
            df['date'] = pd.to_datetime(df['date'])

        elif signal_type == SignalType.MASTER_GEOPOLITICS:
            # Load Master Geopolitics signal (generated by feature_factory.py)
            filepath = Path("sector_signals/signal_master_geopolitics.csv")
            df = pd.read_csv(filepath)
            logger.info(f"Loaded Master Geopolitics signal CSV with columns: {list(df.columns)}")
            df.rename(columns={'date': 'date', 'master_signal': 'signal'}, inplace=True)
            df['date'] = pd.to_datetime(df['date'])

        else:
            raise NotImplementedError(f"Signal loader not implemented for {signal_type.name}")

        if df['date'].dt.tz is not None:
            df['date'] = df['date'].dt.tz_localize(None)

        df.set_index('date', inplace=True)
        logger.info(f"Successfully loaded {len(df)} records from {filepath.name}")
        logger.info(f"Final DataFrame columns: {list(df.columns)}, shape: {df.shape}")

        # Ensure we return only the signal column
        if 'signal' not in df.columns:
            logger.error(f"ERROR: 'signal' column not found after processing {signal_type.name} data!")
            logger.error(f"Available columns: {list(df.columns)}")
            return pd.DataFrame()

        return df[['signal']]

    except FileNotFoundError:
        logger.error(f"FATAL: Signal data file not found for {signal_type.name}. Aborting.")
        return pd.DataFrame()
    except KeyError as e:
        logger.error(f"FATAL: A required column is missing in the signal data file for {signal_type.name}. Missing column: {e}. Aborting.")
        return pd.DataFrame()


# --- Main Execution Block ---

def run_multi_asset_analysis(signal_type: SignalType, assets=None):
    """Run analysis for multiple asset types."""
    if assets is None:
        assets = [AssetType.SP500]  # Only SP500 data available

    print(f"\n--- Running {signal_type.name} analysis for {', '.join([a.name for a in assets])} ---")

    # Use default values for non-interactive testing
    strategy_type, strategy_direction, entry_percentile = StrategyType.MOMENTUM, StrategyDirection.LONG_SHORT, 0.90

    for asset_type in assets:
        try:
            print("\n" + "="*25 + f" Analyzing on {asset_type.name.upper()} " + "="*25)

            # 3. Configure and Run Analysis
            config = AnalysisConfig(asset_type=asset_type, signal_type=signal_type,
                                     strategy_type=strategy_type, strategy_direction=strategy_direction,
                                     strategy_entry_percentile=entry_percentile)
            print_strategy_definition(config)

            print(f"Creating SignalTesterAPI with config: {asset_type.name}/{signal_type.name}")
            tester = SignalTesterAPI(config=config)

            print("Running SignalTesterAPI...")
            tester.run()
            print(f"✅ Successfully completed analysis for {asset_type.name}/{signal_type.name}")

        except Exception as e:
            print(f"\n❌ ERROR running analysis for {asset_type.name}/{signal_type.name}: {e}\n")
            import traceback
            print("Full traceback:")
            traceback.print_exc()

def test_single_signal(signal_type: SignalType, asset_type: AssetType = AssetType.SP500):
    """Test a single signal type."""
    print(f"\n--- Testing {signal_type.name} Signal on {asset_type.name.upper()} ---")

    config = AnalysisConfig(
        asset_type=asset_type,
        signal_type=signal_type,
        strategy_type=StrategyType.MOMENTUM,
        strategy_direction=StrategyDirection.LONG_SHORT,
        strategy_entry_percentile=0.90
    )
    print_strategy_definition(config)

    tester = SignalTesterAPI(config=config)
    try:
        tester.run()
        print(f"✓ {signal_type.name} analysis completed successfully")
    except Exception as e:
        print(f"✗ Error testing {signal_type.name}: {e}")

def run_individual_signals():
    """Test individual signals on SP500."""
    print("\n" + "="*60)
    print("🧪 TESTING INDIVIDUAL SIGNALS ON SP500")
    print("="*60)

    signals_to_test = [
        (SignalType.POLYMARKET_VWEV, "VWEV Inflation"),
        (SignalType.POLYMARKET_VWP, "VWP Geopolitics"),
        (SignalType.MASTER_MACRO, "Master Macro"),
        (SignalType.MASTER_INFLATION, "Master Inflation"),
        (SignalType.MASTER_GEOPOLITICS, "Master Geopolitics"),
    ]

    for signal_type, description in signals_to_test:
        try:
            test_single_signal(signal_type)
        except Exception as e:
            print(f"✗ Failed to test {description}: {e}")
            continue

    print("\n" + "="*60 + "\n📊 INDIVIDUAL SIGNAL COMPARISON COMPLETE\n" + "="*60)

def print_strategy_definition(config: AnalysisConfig):
    """Prints a clear definition of the backtest strategy being used."""
    print("\n" + "-"*60)
    print("🔬 Backtest Strategy Definition:")
    print(f"   - STRATEGY LOGIC: {config.strategy_type.name.replace('_', ' ').title()}")
    print(f"   - STRATEGY DIRECTION: {config.strategy_direction.name.replace('_', ' ').title()}")
    if config.strategy_type == StrategyType.MOMENTUM:
        if config.strategy_direction in [StrategyDirection.LONG_ONLY, StrategyDirection.LONG_SHORT]: print(f"   - ENTRY (LONG): Signal's rolling percentile > {config.strategy_entry_percentile:.0%}")
        if config.strategy_direction in [StrategyDirection.SHORT_ONLY, StrategyDirection.LONG_SHORT]: print(f"   - ENTRY (SHORT): Signal's rolling percentile < {1 - config.strategy_entry_percentile:.0%}")
    else:
        if config.strategy_direction in [StrategyDirection.LONG_ONLY, StrategyDirection.LONG_SHORT]: print(f"   - ENTRY (LONG): Signal's rolling percentile < {config.strategy_entry_percentile:.0%} (Buy the Dip)")
        if config.strategy_direction in [StrategyDirection.SHORT_ONLY, StrategyDirection.LONG_SHORT]: print(f"   - ENTRY (SHORT): Signal's rolling percentile > {1 - config.strategy_entry_percentile:.0%} (Sell the Rip)")
    print(f"   - EXIT: Signal's rolling percentile enters the neutral zone (45%-55%).")
    print("-"*60 + "\n")

def interactive_analysis_ui():
    """Interactive terminal-based analysis UI"""
    print("\n" + "="*80)
    print("🎯 UNIFIED ANALYSIS SYSTEM - SIGNAL GENERATION & ANALYSIS")
    print("="*80)

    import inquirer
    import yaml
    from pathlib import Path

    # Step 1: Scan config/signals/ for available signals
    print("\n📊 STEP 1: SIGNAL GENERATION - AVAILABLE SIGNALS")
    print("-" * 50)

    signals_dir = Path("config/signals")
    signal_configs = list(signals_dir.glob("*.yaml"))

    if not signal_configs:
        print("❌ No signal configuration files found in config/signals/")
        print("Falling back to existing signals...")
        signal_configs = [
            {"name": "GEOPOLITICAL_UNCERTAINTY", "description": "Aggregated geopolitical uncertainty signal", "config_path": None}
        ]
    else:
        print("\n📁 SIGNALS FOUND IN CONFIG:")
        signal_configs = []
        for i, config_file in enumerate(signal_configs, 1):
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)

                signal_name = config.get('signal_name', config_file.stem.upper().replace('_', ' '))
                description = config.get('description', 'No description available')
                signal_configs.append({
                    "name": signal_name,
                    "description": description,
                    "config_path": config_file
                })

                print(f"  {i}. {signal_name}")
                print(f"     {description}")

            except Exception as e:
                print(f"  {i}. {config_file.stem.upper().replace('_', ' ')} (Error reading config: {e})")
                signal_configs.append({
                    "name": config_file.stem.upper().replace('_', ' '),
                    "description": "Configuration error",
                    "config_path": config_file
                })

    # Always offer fallback to existing signals
    signal_options = [sig["name"] for sig in signal_configs]
    signal_options.extend([
        "Use Existing Signals",
        "Skip Generation - Direct To Analysis"
    ])

    # Present signal selection menu
    signal_questions = [
        inquirer.Checkbox('signals',
                         message="Select signals to generate (or use options below):",
                         choices=signal_options,
                         default=[sig["name"] for sig in signal_configs[:1]])  # Default to first available signal
    ]

    selected_signals = inquirer.prompt(signal_questions)['signals']

    print(f"\n🔄 Selected: {', '.join(selected_signals)}")

    # Step 2: Generate signals for selected signals
    print("\n🚀 STEP 2: SIGNAL GENERATION")
    print("-" * 40)

    generated_signals = []

    # Check if user selected options
    if "Use Existing Signals" in selected_signals:
        print("📋 Using existing signals...")
        generated_signals = [
            "POLYMARKET_VWEV",
            "POLYMARKET_VWP",
            "MASTER_INFLATION",
            "MASTER_GEOPOLITICS",
            "MASTER_MACRO"
        ]
    elif "Skip Generation - Direct To Analysis" in selected_signals:
        print("⏭️ Skipping generation, going direct to analysis...")
        generated_signals = [
            "POLYMARKET_VWEV",
            "POLYMARKET_VWP",
            "MASTER_INFLATION",
            "MASTER_GEOPOLITICS",
            "MASTER_MACRO"
        ]
    else:
        # Generate selected signals
        try:
            from src.signals.unified_signal_generator import UnifiedSignalGenerator
            generator = UnifiedSignalGenerator(debug=False)

            for selected_signal in selected_signals:
                # Find the corresponding config
                config_info = None
                for sig_config in signal_configs:
                    if sig_config["name"] == selected_signal:
                        config_info = sig_config
                        break

                if config_info and config_info["config_path"]:
                    print(f"\n🔨 Generating {selected_signal} signal...")

                    try:
                        # Generate signal from config
                        # This would need to be implemented in UnifiedSignalGenerator
                        # For now, we call the concept generation
                        # signal = generator.generate_signal_from_config(config_info["config_path"])
                        print(f"⚠️ Signal generation from config not yet implemented, using existing signal")
                        generated_signals.append(selected_signal)

                    except Exception as e:
                        print(f"❌ Error generating {selected_signal}: {e}")
                        continue
                else:
                    print(f"ℹ️ {selected_signal} - using existing implementation")
                    generated_signals.append(selected_signal)

        except ImportError:
            print("⚠️ Signal generation not available. Using existing signals...")
            generated_signals = [
                "POLYMARKET_VWEV",
                "POLYMARKET_VWP",
                "MASTER_INFLATION",
                "MASTER_GEOPOLITICS",
                "MASTER_MACRO"
            ]

    if not generated_signals:
        print("❌ No signals were successfully generated. Exiting...")
        return

    print(f"✅ Generated/selected {len(generated_signals)} signals: {', '.join(generated_signals)}")

    # Step 3: Show signal set summary statistics
    print("\n📊 STEP 3: SIGNAL SET SUMMARY")
    print("-" * 40)

    signal_stats = {}

    for signal_id in generated_signals:
        try:
            signal_df = load_signal_raw(get_signal_type_from_id(signal_id))

            if not signal_df.empty:
                start_date = signal_df.index.min()
                end_date = signal_df.index.max()
                duration_days = len(signal_df)

                # Calculate signal range
                signal_min = signal_df['signal'].min()
                signal_max = signal_df['signal'].max()
                signal_range = f"{signal_min:.3f} to {signal_max:.3f}"

                # Calculate basic stats
                signal_mean = signal_df['signal'].mean()
                signal_std = signal_df['signal'].std()

                signal_stats[signal_id] = {
                    'duration_days': duration_days,
                    'start_date': start_date.date(),
                    'end_date': end_date.date(),
                    'signal_range': signal_range,
                    'signal_mean': signal_mean,
                    'signal_std': signal_std,
                    'total_points': len(signal_df)
                }

                print(f"📈 {signal_id}:")
                print(f"   Duration: {duration_days} days ({start_date.date()} to {end_date.date()})")
                print(f"   Range: {signal_range}")
                print(f"   Data Points: {len(signal_df)}")
                print(f"   Mean: {signal_mean:.4f} | Std: {signal_std:.4f}")
                print()

        except Exception as e:
            print(f"⚠️ Could not load stats for {signal_id}: {e}")

    # Step 4: Option to see all markets used in signals
    market_questions = [
        inquirer.Confirm('view_markets',
                        message="Would you like to see all markets used in the generated signals?",
                        default=False)
    ]

    market_answers = inquirer.prompt(market_questions)
    view_all_markets = market_answers['view_markets']

    if view_all_markets:
        print("\n🏛️ STEP 4: MARKETS USED IN SIGNAL GENERATION")
        print("-" * 50)

        from pathlib import Path
        import pandas as pd

        # Show markets by concept
        concept_mapping = {
            "VWEV": ["inflation"],
            "VWP": ["geopolitics", "world_affairs"],
            "MACRO": ["macro", "combined"]
        }

        market_sets = {}

        # Find and load manifest files
        manifest_dir = Path("polymarket_data")
        manifest_files = list(manifest_dir.glob("_manifest_*.csv"))

        total_unique_markets = 0

        for manifest_file in manifest_files:
            concept_name = None
            for key, concepts in concept_mapping.items():
                if any(concept.lower() in manifest_file.name.lower() for concept in concepts):
                    concept_name = key
                    break

            if concept_name:
                try:
                    df = pd.read_csv(manifest_file)

                    if concept_name not in market_sets:
                        market_sets[concept_name] = []

                    # Filter for active markets with minimum volume
                    active_markets = df[
                        (df.get('active', True) != False) &
                        (df.get('volume', 0) > 1000)
                    ]

                    if len(active_markets) > 0:
                        market_sets[concept_name].extend(active_markets.to_dict('records'))
                        total_unique_markets += len(active_markets)

                except Exception as e:
                    print(f"Error loading {manifest_file}: {e}")

        print(f"📊 Total unique markets used: {total_unique_markets}")
        print()

        # Show markets by signal - use YAML configs to show which manifests each signal uses
        signal_market_manifests = {}

        for sig_config in signal_configs:
            if sig_config.get('config_path'):
                try:
                    with open(sig_config['config_path'], 'r') as f:
                        config = yaml.safe_load(f)

                    signal_name = config.get('signal_name', 'Unknown')
                    input_manifests = config.get('input_manifest_files', [])

                    signal_market_manifests[signal_name] = input_manifests

                except Exception as e:
                    print(f"Could not read config for {sig_config['name']}: {e}")

        print("\n📋 SIGNAL MARKET MANIFESTS:")
        print("-" * 40)

        if signal_market_manifests:
            for signal_name, manifests in signal_market_manifests.items():
                print(f"\n🎯 SIGNAL: {signal_name}")
                print(f"   Input Manifest Files: {len(manifests)}")
                for manifest in manifests:
                    print(f"   📄 {manifest}")

                    # Load and show number of markets in each manifest
                    manifest_path = Path(manifest)
                    if manifest_path.exists():
                        try:
                            df = pd.read_csv(manifest_path)
                            active_markets = df[
                                (df.get('active', True) != False) &
                                (df.get('volume', 0) > 1000)
                            ]
                            print(f"      📊 Active markets (>1k volume): {len(active_markets)}")
                        except Exception as e:
                            print(f"      ❌ Error reading manifest: {e}")

        else:
            print("No signal configurations found. Showing category breakdown:")

        for concept, markets in market_sets.items():
            print(f"🔹 {concept} Markets ({len(markets)} total):")

            # Show top 3 by volume for each concept
            sorted_markets = sorted(markets, key=lambda x: x.get('volume', 0), reverse=True)[:3]

            for i, market in enumerate(sorted_markets, 1):
                question = market.get('question', 'N/A')[:60] + "..."
                volume = market.get('volume', 0)
                print(f"  {i}. 💰 ${volume:,.0f} - {question}")

            if len(markets) > 3:
                print(f"  ...and {len(markets) - 3} more markets")
            print()

    # Step 5: Ask if want to test the signals
    print("🧪 STEP 4: SIGNAL TESTING")
    print("-" * 40)

    test_questions = [
        inquirer.Confirm('test_signals',
                        message=f"Would you like to test the {len(signal_stats)} generated signals on real assets?",
                        default=True)
    ]

    test_answers = inquirer.prompt(test_questions)
    test_signals = test_answers['test_signals']

    if test_signals:
        # Ask which assets to test
        asset_choices = [
            'SP500',
            'BTC',
            'ETH'
        ]

        asset_questions = [
            inquirer.Checkbox('assets',
                             message="Select assets to test the signals on:",
                             choices=asset_choices,
                             default=['SP500'])
        ]

        asset_answers = inquirer.prompt(asset_questions)
        selected_assets = asset_answers['assets']

        if not selected_assets:
            print("No assets selected. Analysis cancelled.")
            return

        # Ask which signals to test - fallback to generated signals if stats failed to load
        available_signal_names = list(signal_stats.keys()) if signal_stats else generated_signals
        if len(available_signal_names) > 1:
            signal_questions = [
                inquirer.Checkbox('final_signals',
                                 message="Select signals to test:",
                                 choices=available_signal_names,
                                 default=available_signal_names[:2])
            ]

            signal_answers = inquirer.prompt(signal_questions)
            selected_signals = signal_answers['final_signals']
        else:
            selected_signals = available_signal_names
            if available_signal_names:
                print(f"Using signal: {selected_signals[0]}")
            else:
                print("No signals available for testing.")
                return

        # Backtest parameters
        print("\n⚙️ BACKTEST PARAMETERS")
        print("-" * 40)

        params_questions = [
            inquirer.select('strategy_type',
                          message="Select strategy type:",
                          choices=['momentum', 'reversion'],
                          default='momentum'),
            inquirer.select('direction',
                          message="Select strategy direction:",
                          choices=['long_short', 'long_only', 'short_only'],
                          default='long_short'),
            inquirer.select('percentile',
                          message="Select entry percentile:",
                          choices=[0.80, 0.85, 0.90, 0.95],
                          default=0.90)
        ]

        params_answers = inquirer.prompt(params_questions)

        # Execute analysis
        print("\n🚀 RUNNING ANALYSIS...")
        print("-" * 40)

        strategy_type = params_answers['strategy_type']
        strategy_direction = params_answers['direction']
        strategy_percentile = params_answers['percentile']

        # Convert to enums
        strategy_type_enum = StrategyType.MOMENTUM if strategy_type == 'momentum' else StrategyType.REVERSION
        direction_map = {
            'long_short': StrategyDirection.LONG_SHORT,
            'long_only': StrategyDirection.LONG_ONLY,
            'short_only': StrategyDirection.SHORT_ONLY
        }
        strategy_direction_enum = direction_map[strategy_direction]

        # Run analysis for selected combinations
        analysis_count = 0

        for signal_id in selected_signals:
            signal_type = get_signal_type_from_id(signal_id)

            for asset_name in selected_assets:
                asset_type = get_asset_type_from_name(asset_name)

                print(f"\n🔬 Testing {signal_id} on {asset_name.upper()}")
                analysis_count += 1

                try:
                    config = AnalysisConfig(
                        asset_type=asset_type,
                        signal_type=signal_type,
                        strategy_type=strategy_type_enum,
                        strategy_direction=strategy_direction_enum,
                        strategy_entry_percentile=strategy_percentile,
                        min_signal_data_days=10  # Override to allow testing
                    )

                    print_strategy_definition(config)
                    tester = SignalTesterAPI(config=config)
                    tester.run()

                    print(f"✅ Analysis {analysis_count} completed successfully")

                except Exception as e:
                    print(f"❌ Analysis {analysis_count} failed: {e}")

        print(f"Total analysis time: {analysis_count * 5:.0f} seconds (estimated)")
    print("\n🎉 Signal generation and testing workflow complete!")
    print("📁 Check 'analysis_results/' folder for detailed results and charts")

def get_signal_type_from_id(signal_id):
    """Get SignalType enum from signal ID string"""
    mapping = {
        'POLYMARKET_VWEV': SignalType.POLYMARKET_VWEV,
        'POLYMARKET_VWP': SignalType.POLYMARKET_VWP,
        'MASTER_MACRO': SignalType.MASTER_MACRO,
        'MASTER_INFLATION': SignalType.MASTER_INFLATION,
        'MASTER_GEOPOLITICS': SignalType.MASTER_GEOPOLITICS
    }
    return mapping.get(signal_id, SignalType.POLYMARKET_VWP)

def get_asset_type_from_name(asset_name):
    """Get AssetType enum from asset name string"""
    mapping = {
        'SP500': AssetType.SP500,
        'BTC': AssetType.BTC,
        'ETH': AssetType.ETH
    }
    return mapping.get(asset_name, AssetType.SP500)

if __name__ == "__main__":
    try:
        import inquirer
        interactive_analysis_ui()
    except ImportError:
        # Fallback to non-interactive mode
        print("🔄 Interactive mode requires 'inquirer' package. Running in standard mode...")
        print("To enable interactive mode: pip install inquirer")
        print("\nRunning Individual Signal Analysis for Comparison...")
        run_individual_signals()
