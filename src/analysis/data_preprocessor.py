# data_preprocessor.py

import pandas as pd
import numpy as np
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Prepares raw price and signal data for the Unified Analysis API.
    """
    def __init__(self, price_df: pd.DataFrame, signal_df: pd.DataFrame, lookback_months: int = 6):
        if not isinstance(price_df.index, pd.DatetimeIndex) or not isinstance(signal_df.index, pd.DatetimeIndex):
            raise TypeError("Input DataFrames must have a DatetimeIndex.")
            
        self.price_df = price_df
        self.signal_df = signal_df
        self.lookback_days = lookback_months * 30
        self.forward_periods = {'1w': 7, '1m': 30, '3m': 90, '6m': 180}

    def process(self) -> pd.DataFrame:
        """Runs the full preprocessing pipeline."""
        logger.info("Starting data preprocessing pipeline...")
        logger.info(f"Price DataFrame shape: {self.price_df.shape}, columns: {list(self.price_df.columns)}")
        logger.info(f"Signal DataFrame shape: {self.signal_df.shape}, columns: {list(self.signal_df.columns)}")

        # Validate required columns exist
        if 'price' not in self.price_df.columns:
            logger.error("ERROR: 'price' column missing from price DataFrame!")
            logger.error(f"Available columns: {list(self.price_df.columns)}")
            return pd.DataFrame()

        if 'signal' not in self.signal_df.columns:
            logger.error("ERROR: 'signal' column missing from signal DataFrame!")
            logger.error(f"Available columns: {list(self.signal_df.columns)}")
            return pd.DataFrame()

        df = self.price_df.join(self.signal_df, how='inner')
        logger.info(f"After join - DataFrame shape: {df.shape}, columns: {list(df.columns)}")

        if df.empty:
            logger.error("ERROR: Inner join resulted in empty DataFrame!")
            return pd.DataFrame()

        df.sort_index(inplace=True)
        
        logger.info("Calculating forward returns for 1w, 1m, 3m, 6m...")
        df['price_pct_change'] = df['price'].pct_change()
        for name, period in self.forward_periods.items():
            df[f'fwd_return_{name}'] = df['price'].pct_change(periods=period).shift(-period)

        logger.info(f"Calculating {self.lookback_days}-day rolling signal percentile...")
        df['signal_pct_change'] = df['signal'].pct_change()
        df['signal_percentile'] = df['signal_pct_change'].rolling(
            window=self.lookback_days,
            min_periods=self.lookback_days // 3
        ).apply(
            lambda x: stats.percentileofscore(pd.Series(x), pd.Series(x).iloc[-1]) / 100.0 if not np.isnan(pd.Series(x).iloc[-1]) else np.nan,
            raw=False
        )
        
        core_cols = ['price_pct_change', 'signal_pct_change', 'signal_percentile']
        processed_df = df.dropna(subset=core_cols)
        
        logger.info(f"Preprocessing complete. Final DataFrame shape: {processed_df.shape}")
        return processed_df