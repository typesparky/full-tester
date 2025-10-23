# main_runner.py

import pandas as pd
import json
import logging
from pathlib import Path
from data_preprocessor import DataPreprocessor
from unified_analyzer import SignalTesterAPI, AnalysisConfig, AssetType, SignalType, StrategyDirection, StrategyType

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

def run_dual_asset_analysis(signal_type: SignalType):
    """Helper function to run the same signal analysis for both BTC and ETH."""
    print(f"\n--- Running {signal_type.name} analysis for both BTC and ETH ---")
    
    # Use default values for non-interactive testing
    strategy_type, strategy_direction, entry_percentile = StrategyType.MOMENTUM, StrategyDirection.LONG_ONLY, 0.90
        
    for asset_type in [AssetType.BTC, AssetType.ETH]:
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

if __name__ == "__main__":
    # Run hashrate analysis directly for debugging
    print("Running Hashrate Analysis for debugging...")
    run_dual_asset_analysis(SignalType.HASHRATE)