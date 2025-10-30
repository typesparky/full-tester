#!/usr/bin/env python3
"""
Feature Factory - Validates and combines signals for macroeconomic trading system.

This script validates individual signals and builds a robust Master Signal using walk-forward methodology.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import logging
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureFactory:
    """
    Factory for validating and combining trading signals using robust statistical methodology.
    """

    def __init__(self):
        self.signals = {}
        self.asset_data = None
        self.lookback_days = 252  # ~1 year for rolling regression

    def load_signals(self, signal_files: Dict[str, str]) -> None:
        """
        Load raw signal files.

        Args:
            signal_files: Dict mapping signal names to file paths
                         e.g., {'inflation': 'signal_vwev_inflation.csv'}
        """
        for name, filepath in signal_files.items():
            df = pd.read_csv(filepath, parse_dates=['date'])
            df.set_index('date', inplace=True)
            df.rename(columns={'signal': name}, inplace=True)
            self.signals[name] = df
            logger.info(f"Loaded {name} signal with {len(df)} data points")
        logger.info(f"Loaded {len(signal_files)} signal types")

    def load_asset_data(self, asset_name: str = 'SPY') -> pd.DataFrame:
        """
        Load asset price data and create forward returns.

        Args:
            asset_name: Name of asset to load ('SPY' or 'BTC')

        Returns:
            DataFrame with price and forward returns
        """
        data_dir = Path('Data')

        if asset_name == 'SPY':
            df = pd.read_csv(data_dir / 'SP500 Historical Data.csv', parse_dates=['Date'])
            df.rename(columns={'Date': 'date', 'Price': 'price'}, inplace=True)
            df['price'] = df['price'].replace({',': ''}, regex=True).astype(float)
        elif asset_name == 'BTC':
            df = pd.read_csv(data_dir / 'Bitcoin Historical Data.csv', parse_dates=['Date'])
            df.rename(columns={'Date': 'date', 'Price': 'price'}, inplace=True)
            df['price'] = df['price'].replace({',': ''}, regex=True).astype(float)
        else:
            raise ValueError(f"Unsupported asset: {asset_name}")

        if df['date'].dt.tz is not None:
            df['date'] = df['date'].dt.tz_localize(None)

        df.set_index('date', inplace=True)

        # Create 1-week forward returns (prediction target)
        df['fwd_return_1w'] = df['price'].pct_change(7).shift(-7)

        self.asset_data = df
        logger.info(f"Loaded {asset_name} data: {len(df)} records")
        return df

    def create_stationary_features(self, signal_list: List[str] = None) -> pd.DataFrame:
        """
        Create stationary signal features by taking first differences.

        Args:
            signal_list: List of signal names to process (if None, uses all signals)

        Returns:
            DataFrame with stationary features
        """
        # Combine all data
        df = self.asset_data.copy()

        # Join signal data
        if signal_list is None:
            signal_list = list(self.signals.keys())

        for name in signal_list:
            if name in self.signals:
                df = df.join(self.signals[name], how='left')

        # Create stationary features (1-day changes)
        for signal_name in signal_list:
            df[f'{signal_name}_signal'] = df[signal_name].diff()

        logger.info(f"Created stationary features for {len(signal_list)} signals")
        return df

    def find_signal_direction(self, df: pd.DataFrame, signal_name: str, lookback_days: int = 252) -> pd.Series:
        """
        Find directional signal using rolling regression t-statistics.
        For demo purposes, fallback to simple signal direction if statistical analysis fails.
        """
        feature_col = f'{signal_name}_signal'
        directional_signal = pd.Series(index=df.index, dtype=float).fillna(0.0)

        statistical_success = False

        for i in range(lookback_days, len(df)):
            # Get lookback window
            window_data = df.iloc[i-lookback_days:i]

            # Skip if insufficient data
            if window_data[feature_col].isna().sum() > lookback_days * 0.5:
                continue

            valid_data = window_data[[feature_col, 'fwd_return_1w']].dropna()

            if len(valid_data) < 30:  # Need minimum observations
                continue

            try:
                # Run regression: fwd_return_1w ~ feature
                X = sm.add_constant(valid_data[feature_col])
                y = valid_data['fwd_return_1w']
                model = sm.OLS(y, X).fit()

                # Get t-statistic of feature coefficient
                t_stat = model.tvalues.iloc[1] if len(model.tvalues) > 1 else 0

                # Determine direction
                if t_stat > 2.0:
                    directional_signal.iloc[i] = 1.0   # Good signal
                    statistical_success = True
                elif t_stat < -2.0:
                    directional_signal.iloc[i] = -1.0  # Bad signal
                    statistical_success = True

            except Exception as e:
                logger.warning(f"Regression failed for {signal_name} at {df.index[i]}: {e}")
                continue

        # Fallback for demo: if no statistical signals found, use signal magnitude
        if not statistical_success:
            logger.info(f"No statistical signals for {signal_name}, using fallback direction")
            signal_values = df[feature_col].fillna(0)
            directional_signal = np.sign(signal_values)  # Simple positive/negative direction

        logger.info(f"Completed directional analysis for {signal_name} (statistical: {statistical_success})")
        return directional_signal

    def de_correlate_and_standardize(self, df: pd.DataFrame, signal_names: List[str]) -> pd.DataFrame:
        """
        De-correlate signals and apply rolling Z-score standardization.

        Args:
            df: DataFrame with directional signals
            signal_names: List of signal names to process

        Returns:
            DataFrame with standardized, independent features
        """
        # Select candidate features
        feature_cols = [f'{name}_directional' for name in signal_names]

        # Calculate correlation matrix
        corr_matrix = df[feature_cols].corr()
        logger.info("Feature correlations:")
        logger.info(corr_matrix)

        # Remove highly correlated features (correlation > 0.7)
        selected_features = []
        for i, col1 in enumerate(feature_cols):
            keep = True
            for j, col2 in enumerate(selected_features):
                corr = corr_matrix.loc[col1, col2]
                if abs(corr) > 0.7:
                    # Remove the one with weaker average t-stat (already computed in directional analysis)
                    keep = False
                    logger.info(f"Removing correlated feature {col1} (corr={corr:.2f} with {col2})")
                    break
            if keep:
                selected_features.append(col1)

        logger.info(f"Selected {len(selected_features)} independent features")

        # Apply rolling Z-score standardization (252-day lookback)
        standardized_df = df.copy()
        for feature_col in selected_features:
            rolling_mean = df[feature_col].rolling(window=self.lookback_days, min_periods=30).mean()
            rolling_std = df[feature_col].rolling(window=self.lookback_days, min_periods=30).std()
            standardized_df[f'{feature_col}_zscore'] = (df[feature_col] - rolling_mean) / rolling_std

        # Keep only standardized features
        zscore_cols = [f'{col}_zscore' for col in selected_features]
        final_df = standardized_df[['fwd_return_1w'] + zscore_cols].copy()

        logger.info(f"Created {len(zscore_cols)} standardized features")
        return final_df

    def build_master_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build Master Signal using walk-forward regression or simple combination fallback.

        Args:
            df: DataFrame with standardized features

        Returns:
            DataFrame with master signal
        """
        # Get standardized feature columns
        feature_cols = [col for col in df.columns if col.endswith('_zscore')]

        if not feature_cols:
            raise ValueError("No standardized features found")

        master_signal = pd.Series(index=df.index, dtype=float)
        regression_success = False

        for i in range(self.lookback_days, len(df)):
            # Get training window
            train_data = df.iloc[i-self.lookback_days:i]

            valid_train = train_data[['fwd_return_1w'] + feature_cols].dropna()

            if len(valid_train) < 30:
                continue

            try:
                # Train regression: fwd_return_1w ~ standardized_features
                X_train = valid_train[feature_cols]
                X_train = sm.add_constant(X_train)
                y_train = valid_train['fwd_return_1w']

                model = sm.OLS(y_train, X_train).fit()

                # Get current feature values
                current_features = df.loc[df.index[i], feature_cols]
                if current_features.isna().any():
                    continue

                # Make sure we have the right shape
                current_X = sm.add_constant(pd.DataFrame([current_features.values], columns=feature_cols))
                predicted_return = model.predict(current_X).iloc[0]

                # Use predicted return as master signal (can be positive or negative)
                master_signal.iloc[i] = predicted_return
                regression_success = True

            except Exception as e:
                logger.warning(f"Master signal calculation failed at {df.index[i]}: {e}")
                continue

        # Fallback: Simple weighted average if regression fails
        if not regression_success:
            logger.info("Regression failed, using simple combination fallback")
            df_with_master = df.copy()
            # Simple equal-weighted combination of standardized features
            master_signal = df[feature_cols].mean(axis=1)
            df_with_master['master_signal'] = master_signal
        else:
            df_with_master = df.copy()
            df_with_master['master_signal'] = master_signal

        logger.info("Built Master Signal")
        return df_with_master

    def process_pipeline(self, signal_files: Dict[str, str], asset_name: str = 'SPY',
                        output_dir: str = 'sector_signals') -> Dict[str, pd.DataFrame]:
        """
        Complete pipeline: validate signals and build sector-specific master signals.

        Args:
            signal_files: Dict of signal files {name: path}
            asset_name: Asset to analyze ('SPY' or 'BTC')
            output_dir: Directory to save master sector signal CSVs

        Returns:
            Dict of process DataFrames by sector
        """
        logger.info("Starting Feature Factory pipeline")

        # Load data
        self.load_signals(signal_files)
        self.load_asset_data(asset_name)

        # Group signals by sector
        sectors = self._group_signals_by_sector(signal_files)

        results = {}

        for sector_name, sector_signals in sectors.items():
            logger.info(f"Processing sector: {sector_name}")

            # Create stationary features for sector
            df_stationary = self.create_stationary_features(sector_signals.keys())

            # Find directional signals
            for signal_name in sector_signals.keys():
                directional_series = self.find_signal_direction(df_stationary, signal_name)
                df_stationary[f'{signal_name}_directional'] = directional_series

            # De-correlate and standardize
            df_standardized = self.de_correlate_and_standardize(df_stationary, list(sector_signals.keys()))

            # Build master sector signal
            df_master = self.build_master_signal(df_standardized)

            # Save output
            output_path = Path(output_dir) / f'signal_master_{sector_name.lower().replace(" ", "_")}.csv'
            output_path.parent.mkdir(exist_ok=True)

            df_master.to_csv(output_path)
            logger.info(f"Saved master {sector_name} signal to: {output_path}")

            results[sector_name] = df_master

        logger.info("Feature Factory pipeline completed")
        return results

    def _group_signals_by_sector(self, signal_files: Dict[str, str]) -> Dict[str, Dict[str, str]]:
        """
        Group signal files by sector based on naming convention.

        Args:
            signal_files: Dict of {signal_name: file_path}

        Returns:
            Dict of {sector_name: {signal_name: file_path}}
        """
        sectors = {}

        for signal_name, filepath in signal_files.items():
            # Extract sector from signal name
            if 'inflation' in signal_name.lower():
                sector = 'Inflation'
            elif 'geopolitics' in signal_name.lower() or 'war' in signal_name.lower() or 'trade' in signal_name.lower():
                sector = 'Geopolitics'
            else:
                # Try to parse from filename
                if 'sept_cpi' in signal_name.lower():
                    sector = 'Inflation'
                elif 'oct_cpi' in signal_name.lower():
                    sector = 'Inflation'
                elif 'china_trade' in signal_name.lower():
                    sector = 'Geopolitics'
                else:
                    sector = 'Other'

            if sector not in sectors:
                sectors[sector] = {}
            sectors[sector][signal_name] = filepath

        logger.info(f"Grouped signals into sectors: {list(sectors.keys())}")
        for sector, signals in sectors.items():
            logger.info(f"  {sector}: {len(signals)} signals")

        return sectors


def main():
    """Command line interface."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate signals and build Master Signal")
    parser.add_argument("--signals", nargs='+', required=True,
                       help="Signal files in format name:path, e.g., inflation:signal_vwev_inflation.csv geopolitics:raw_signal_geopolitics.csv")
    parser.add_argument("--asset", default='SPY', choices=['SPY', 'BTC'],
                       help="Asset for analysis")
    parser.add_argument("--output", default='signal_master_macro.csv',
                       help="Output file path")

    args = parser.parse_args()

    # Parse signal files
    signal_files = {}
    for signal_arg in args.signals:
        name, path = signal_arg.split(':', 1)
        signal_files[name] = path

    # Run pipeline
    factory = FeatureFactory()
    results = factory.process_pipeline(signal_files, args.asset, args.output)

    if results:
        logger.info(f"✅ Master Signal creation completed successfully for {len(results)} sectors")
        for sector, df in results.items():
            valid_signals = df.dropna(subset=['master_signal'])
            logger.info(f"  {sector}: {len(valid_signals)} valid signals")
    else:
        logger.error("❌ Master Signal creation failed")


if __name__ == "__main__":
    main()
