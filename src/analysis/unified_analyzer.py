# signal_analyzer.py

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, field
from enum import Enum
import warnings
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import ta  # For technical analysis including ATR

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

# --- Enums and Configuration ---

class AssetType(Enum):
    BTC = "btc"
    ETH = "eth"
    SP500 = "sp500"
    NASDAQ = "nasdaq"
    OIL = "oil"
    GOLD = "gold"

class SignalType(Enum):
    HASHRATE = "hashrate"
    USDC_ISSUANCE = "usdc_issuance"
    USDT_ISSUANCE = "usdt_issuance"
    POLYMARKET_VWP = "polymarket_vwp"
    POLYMARKET_VWEV = "polymarket_vwev"
    MASTER_MACRO = "master_macro"
    MASTER_INFLATION = "master_inflation"
    MASTER_GEOPOLITICS = "master_geopolitics"
    GEOPOLITICAL_UNCERTAINTY = "geopolitical_uncertainty"
    MILITARY_ACTIONS = "military_actions"
    INFLATION = "inflation"
    GDP = "gdp"

class StrategyDirection(Enum):
    LONG_ONLY = "long_only"
    SHORT_ONLY = "short_only"
    LONG_SHORT = "long_short"

class StrategyType(Enum):
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"

class ExitLogic(Enum):
    SIGNAL_NEUTRAL = "signal_neutral"
    ATR_TRAILING_STOP = "atr_trailing_stop"

@dataclass
class AnalysisConfig:
    """The main control panel for configuring an analysis run."""
    asset_type: AssetType
    signal_type: SignalType
    strategy_type: StrategyType = StrategyType.MOMENTUM
    strategy_direction: StrategyDirection = StrategyDirection.LONG_SHORT
    strategy_entry_percentile: float = 0.80
    signal_lookback_months: int = 6
    forward_periods_map: dict = field(default_factory=lambda: {'1w': 7, '1m': 30, '3m': 90, '6m': 180})
    percentile_thresholds: List[float] = field(default_factory=lambda: [1, 5, 10, 20, 80, 90, 95, 99])
    output_dir: str = "analysis_results"
    initial_capital: float = 10000.0

    # Exit logic configuration
    exit_logic: ExitLogic = ExitLogic.SIGNAL_NEUTRAL
    exit_params: dict = field(default_factory=lambda: {
        'atr_period': 14,
        'atr_multiplier': 1.5,
        'atr_timeframe': 'DAILY',
        'neutral_band': [0.40, 0.60]
    })

    # DATA QUALITY ENFORCEMENT - DISABLED for real-time signals
    min_signal_data_days: int = 0  # Remove minimum for military testing

    @property
    def output_filename(self) -> str:
        entry_pct_str = int(self.strategy_entry_percentile * 100)
        exit_suffix = "_atr" if self.exit_logic == ExitLogic.ATR_TRAILING_STOP else ""
        return f"{self.asset_type.value}_{self.signal_type.value}_{self.strategy_type.value}_{self.strategy_direction.value}_{entry_pct_str}pct{exit_suffix}.png"

# --- Main Analysis and Visualization API ---

class SignalTesterAPI:
    """The single, unified API for running signal analysis and generating visualizations."""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.df = None
        self.pnl_results = {}
        self.regression_results = {}
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def run(self):
        """Executes the full analysis pipeline."""
        title = f"{self.config.asset_type.name}/{self.config.signal_type.name} ({self.config.strategy_type.name}, {self.config.strategy_direction.name})"
        logger.info(f"--- Starting Analysis: {title} ---")
        try:
            self.df = self._prepare_data()
            if self.df.empty:
                logger.error("Data preparation resulted in an empty DataFrame. Cannot proceed.")
                return

            self.pnl_results = self._calculate_pnl()
            self.regression_results = self._calculate_regression()

            self._create_pro_dashboard()
            self.generate_specialized_visualizations()
            
            signal_score = calculate_signal_score(self.pnl_results, self.regression_results)
            logger.info(f"--- AUTOMATED SIGNAL SCORE: {signal_score:.1f} / 100 ---")
            
            logger.info(f"--- Analysis Complete. Dashboards saved to '{self.config.output_dir}' ---")
        except Exception as e:
            logger.error(f"Analysis pipeline failed: {e}", exc_info=True)

    def _prepare_data(self) -> pd.DataFrame:
        price_df = self._load_asset_data(self.config.asset_type)
        signal_df = self._load_signal_data(self.config.signal_type)

        if price_df.empty or signal_df.empty:
            return pd.DataFrame()

        df = price_df.join(signal_df, how='inner')
        logger.info(f"After join - DataFrame shape: {df.shape}, columns: {list(df.columns)}")

        if df.empty:
            logger.warning("Inner join of price and signal data resulted in an empty DataFrame. Check for date range overlap.")
            return pd.DataFrame()

        # Store full time series for visualization (before filtering)
        self.full_plot_df = df.copy()
        # Add pct_change columns for visualization
        self.full_plot_df['price_pct_change'] = self.full_plot_df['price'].pct_change()
        self.full_plot_df['signal_pct_change'] = self.full_plot_df['signal'].pct_change()

        df.sort_index(inplace=True)

        # Debug: Check if columns exist before accessing them
        logger.info(f"Columns available: {list(df.columns)}")
        if 'price' not in df.columns:
            logger.error("ERROR: 'price' column not found in DataFrame!")
            return pd.DataFrame()
        if 'signal' not in df.columns:
            logger.error("ERROR: 'signal' column not found in DataFrame!")
            return pd.DataFrame()

        # Filter for valid data
        df = df.dropna(subset=['signal'])  # Remove NaN signals
        df = df[df['price'] > 0]  # Keep only valid prices
        # Note: Polymarket signals can be negative (e.g., VWP in [-1, 1] range)

        df['price_pct_change'] = df['price'].pct_change()
        df['signal_pct_change'] = df['signal'].pct_change()

        # Adaptive rolling percentile calculation
        valid_pct_change = df.dropna(subset=['signal_pct_change'])
        num_valid = len(valid_pct_change)

        # Adaptive lookback: aim for at least 20 periods, but scale based on data
        min_lookback = min(num_valid // 3, 30)  # At least 1/3 of data, max 30 days
        min_lookback = max(min_lookback, 10)  # But at least 10 days

        logger.info(f"Using {min_lookback}-day adaptive lookback for rolling percentiles ({num_valid} valid pct_change points)")

        if num_valid >= min_lookback * 2:  # Need at least 2x lookback for meaningful rolling calc
            df['signal_percentile'] = df['signal_pct_change'].rolling(
                window=min_lookback, min_periods=min_lookback
            ).apply(lambda x: stats.percentileofscore(pd.Series(x), pd.Series(x).iloc[-1]) / 100.0, raw=False)
        else:
            logger.warning(f"Very short dataset ({num_valid} points) - using simple ranking approach")
            # Use simple percentile ranking
            df['signal_percentile'] = pd.qcut(df['signal_pct_change'], q=10, duplicates='drop', labels=False).astype(float) / 9.0

        # For MILITARY_ACTIONS signal, include 7-day forward returns with specified percentiles
        if self.config.signal_type == SignalType.MILITARY_ACTIONS:
            logger.info("MILITARY_ACTIONS: Using 7-day forward returns for percentile analysis")
            # Use 7-day forward period for dashboard percentiles (1,5,10,20,80,90,95,99)
            self.config.forward_periods_map = {'1w': 7}  # Exactly 7 trading days forward

        # For short datasets, skip forward period calculations to preserve data
        available_periods = {}
        if len(df) >= 100:
            for name, period in self.config.forward_periods_map.items():
                # Only calculate if we have enough data for the forward period
                min_required = period + 5  # Reduced buffer for military signals
                if len(df) >= min_required:
                    df[f'fwd_return_{name}'] = df['price'].pct_change(periods=period).shift(-period)
                    available_periods[name] = period

        # Update config with available periods
        self.config.forward_periods_map = available_periods

        # More selective dropping to preserve data
        processed_df = df.dropna(subset=['price_pct_change', 'signal_pct_change'])
        processed_df = processed_df.dropna(subset=['signal_percentile'])

        # For military signals, be more lenient with forward returns
        if not self.config.forward_periods_map:
            logger.warning("No forward periods calculated - skipping forward return drops")
        elif len(processed_df) > 100:
            processed_df = processed_df.dropna(subset=[f'fwd_return_{name}' for name in self.config.forward_periods_map.keys()])
        else:
            logger.warning("Short dataset detected - skipping forward return drops to retain data for correlation analysis")

        # ENFORCE MINIMUM DATA REQUIREMENTS
        if len(processed_df) < self.config.min_signal_data_days:
            logger.error(f"❌ SIGNAL QUALITY FAILURE: Only {len(processed_df)} trading days available, but minimum {self.config.min_signal_data_days} required.")
            logger.error("REASON: Insufficient historical data for robust statistical analysis.")
            logger.error("SOLUTION: Signal cannot be used for backtesting until more historical data is available (2020-2024).")
            return pd.DataFrame()  # Return empty to prevent analysis

        logger.info(f"✓ Data quality check PASSED: {len(processed_df)} trading days available (minimum {self.config.min_signal_data_days} required)")

        logger.info(f"Data preparation complete. Final shape for analysis: {processed_df.shape}")
        return processed_df

    def _calculate_pnl(self) -> Dict:
        df_pnl = self.df.copy()

        # Use simple correlation-based strategy for very short datasets
        if len(df_pnl) < 50:
            logger.info("Short dataset detected - using correlation-based entry logic")
            # Calculate correlation over rolling window
            corr_periods = min(30, len(df_pnl) // 3)
            df_pnl['rolling_corr'] = df_pnl['signal_pct_change'].rolling(corr_periods).corr(df_pnl['price_pct_change'])

            # Strategy: trade when correlation is extreme (adjusted for short datasets)
            if self.config.strategy_type == StrategyType.MOMENTUM:
                long_entry = df_pnl['rolling_corr'] > 0.3  # Moderate positive correlation
                short_entry = df_pnl['rolling_corr'] < -0.3  # Moderate negative correlation
            else:  # MEAN_REVERSAL
                long_entry = df_pnl['rolling_corr'] < -0.3  # Buy when correlation breaks down
                short_entry = df_pnl['rolling_corr'] > 0.3   # Short when correlation strengthens

            neutral_zone = abs(df_pnl['rolling_corr']) < 0.2  # Narrow neutral zone
            positions = np.zeros(len(df_pnl))
            current_pos = 0

            for i in range(len(df_pnl)):
                todays_signal = 0
                if long_entry.iloc[i]:
                    todays_signal = 1
                elif short_entry.iloc[i]:
                    todays_signal = -1
                if todays_signal != 0 and todays_signal != current_pos:
                    current_pos = todays_signal
                elif current_pos != 0 and neutral_zone.iloc[i]:
                    current_pos = 0
                positions[i] = current_pos
        else:
            # Use full percentile-based strategy for longer datasets
            if self.config.strategy_type == StrategyType.MOMENTUM:
                long_entry = df_pnl['signal_percentile'] > self.config.strategy_entry_percentile
                short_entry = df_pnl['signal_percentile'] < (1.0 - self.config.strategy_entry_percentile)
            else:  # MEAN_REVERSION
                long_entry = df_pnl['signal_percentile'] < self.config.strategy_entry_percentile
                short_entry = df_pnl['signal_percentile'] > (1.0 - self.config.strategy_entry_percentile)

            neutral_zone = (df_pnl['signal_percentile'] >= 0.45) & (df_pnl['signal_percentile'] <= 0.55)
            positions = np.zeros(len(df_pnl))
            current_pos = 0

            for i in range(len(df_pnl)):
                todays_signal = 0
                if self.config.strategy_direction in [StrategyDirection.LONG_ONLY, StrategyDirection.LONG_SHORT] and long_entry.iloc[i]:
                    todays_signal = 1
                elif self.config.strategy_direction in [StrategyDirection.SHORT_ONLY, StrategyDirection.LONG_SHORT] and short_entry.iloc[i]:
                    todays_signal = -1
                if todays_signal != 0:
                    current_pos = todays_signal
                elif current_pos != 0 and neutral_zone.iloc[i]:
                    current_pos = 0
                positions[i] = current_pos

        df_pnl['position'] = positions

        df_pnl['strategy_return'] = df_pnl['position'].shift(1) * df_pnl['price_pct_change']
        df_pnl.fillna({'strategy_return': 0}, inplace=True)

        df_pnl['strategy_equity'] = self.config.initial_capital * (1 + df_pnl['strategy_return']).cumprod()
        df_pnl['buy_hold_equity'] = self.config.initial_capital * (1 + df_pnl['price_pct_change'].fillna(0)).cumprod()
        df_pnl['strategy_drawdown'] = (df_pnl['strategy_equity'] / df_pnl['strategy_equity'].expanding().max()) - 1

        results = {'dataframe': df_pnl}
        total_days = len(df_pnl)
        results['total_return'] = (df_pnl['strategy_equity'].iloc[-1] / self.config.initial_capital - 1)
        results['annualized_return'] = (1 + results['total_return']) ** (365 / total_days) - 1 if total_days > 0 else 0
        strat_std = df_pnl['strategy_return'].std()
        # Correct Sharpe ratio calculation
        # Sharpe = (Annualized Portfolio Return - Annualized Risk-Free Rate) / Annualized Portfolio Volatility
        risk_free_rate_annual = 0.02
        annualized_volatility = strat_std * np.sqrt(365)  # Annualize daily volatility
        annualized_excess_return = results['annualized_return'] - risk_free_rate_annual
        results['sharpe_ratio'] = (annualized_excess_return / annualized_volatility) if annualized_volatility > 0 else 0
        results['max_drawdown'] = df_pnl['strategy_drawdown'].min() if not df_pnl['strategy_drawdown'].empty else 0
        results['num_trades'] = df_pnl['position'].diff().abs().sum()

        # Additional metrics for improved scoring system
        # Win rate: percentage of positive return days
        positive_returns = (df_pnl['strategy_return'] > 0).sum()
        total_returns = (df_pnl['strategy_return'] != 0).sum()
        results['win_rate'] = positive_returns / total_returns if total_returns > 0 else 0

        # Volatility: annualized standard deviation of returns
        results['volatility'] = strat_std * np.sqrt(365)  # Annualize daily volatility

        # Calmar ratio: annualized return / maximum drawdown (absolute value)
        if results['max_drawdown'] != 0:
            results['calmar_ratio'] = results['annualized_return'] / abs(results['max_drawdown'])
        else:
            results['calmar_ratio'] = 0

        return results

    def _calculate_regression(self) -> Dict:
        results = {}
        for name, period in self.config.forward_periods_map.items():
            subset = self.df[['signal_pct_change', f'fwd_return_{name}']].dropna()
            if len(subset) < 10:  # Lower threshold for short datasets
                continue

            slope, intercept, r_value, p_value, std_err = stats.linregress(subset['signal_pct_change'], subset[f'fwd_return_{name}'])
            t_statistic = slope / std_err if std_err > 0 else 0
            results[name] = {'r_squared': r_value**2, 'p_value': p_value, 't_statistic': t_statistic, 'coefficient': slope}
        return results
    
    def _create_pro_dashboard(self):
        """Creates the new, professional multi-panel dashboard."""
        fig = plt.figure(figsize=(24, 28))
        gs = GridSpec(5, 4, figure=fig, hspace=0.6, wspace=0.5)
        strat_type_str = self.config.strategy_type.name.replace("_", "-").title()
        strat_dir_str = self.config.strategy_direction.name.replace("_", "-").title()
        title = f'Signal Analysis: "{self.config.signal_type.name.replace("_", " ").title()}" on "{self.config.asset_type.name.upper()}" ({strat_type_str} / {strat_dir_str})'
        fig.suptitle(title, fontsize=32, fontweight='bold')

        self._plot_time_series(fig.add_subplot(gs[0, 0]), 'signal', self.config.signal_type.name.replace('_', ' ').title(), 'blue')
        self._plot_time_series(fig.add_subplot(gs[0, 1]), 'price', f"{self.config.asset_type.name.upper()} Price", 'orange')
        self._plot_distribution(fig.add_subplot(gs[0, 2]))
        self._plot_fwd_return_dist(fig.add_subplot(gs[0, 3]))

        self._plot_pnl_equity(fig.add_subplot(gs[1, :2]))
        self._plot_drawdown(fig.add_subplot(gs[1, 2:]))

        self._plot_trading_signals(fig.add_subplot(gs[2, 0]))
        self._plot_rolling_correlation(fig.add_subplot(gs[2, 1]))
        self._plot_correlation_lags(fig.add_subplot(gs[2, 2]))  # Replace regime returns with lags correlation
        self._plot_extreme_percentile_returns(fig.add_subplot(gs[2, 3]))

        self._plot_performance_table(fig.add_subplot(gs[3, :2]))
        self._plot_regression_table(fig.add_subplot(gs[3, 2:]))
        self._plot_summary_text(fig.add_subplot(gs[4, :]))
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.savefig(Path(self.config.output_dir) / self.config.output_filename, dpi=150)
        plt.show()

    def _plot_time_series(self, ax, column, title, color):
        self.full_plot_df[column].plot(ax=ax, color=color, lw=1.5), ax.set_title(title, fontsize=14, fontweight='bold'), ax.set_ylabel(title)

    def _plot_distribution(self, ax):
        sns.histplot(self.full_plot_df['signal_pct_change'], ax=ax, kde=True, bins=50, color='green')
        ax.set_title("Signal % Change Distribution", fontsize=14, fontweight='bold')
        
    def _plot_fwd_return_dist(self, ax):
        for name in self.config.forward_periods_map.keys():
            sns.kdeplot(self.df[f'fwd_return_{name}'].dropna(), ax=ax, label=name, lw=2)
        ax.set_title("Forward Return Distributions", fontsize=14, fontweight='bold'), ax.legend()

    def _plot_pnl_equity(self, ax):
        df_pnl = self.pnl_results['dataframe']
        ax.plot(df_pnl.index, df_pnl['strategy_equity'], label='Strategy', color='g', lw=2)
        ax.plot(df_pnl.index, df_pnl['buy_hold_equity'], label='Buy & Hold', color='b', linestyle='--', lw=1.5)
        ax.set_title("Portfolio Performance", fontsize=16, fontweight='bold'), ax.set_ylabel("Portfolio Value ($)"), ax.set_yscale('log'), ax.legend()

    def _plot_drawdown(self, ax):
        df_pnl = self.pnl_results['dataframe']
        ax.fill_between(df_pnl.index, df_pnl['strategy_drawdown'], 0, color='r', alpha=0.3, label='Strategy')
        ax.set_title("Strategy Drawdown", fontsize=16, fontweight='bold'), ax.set_ylabel("Drawdown (%)"), ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    def _plot_trading_signals(self, ax):
        df_pnl = self.pnl_results['dataframe']
        ax.plot(df_pnl.index, df_pnl['signal_pct_change'], color='gray', alpha=0.4, lw=1)
        signals = df_pnl[df_pnl['position'].diff() != 0]
        long_signals = signals[signals['position'] == 1]
        short_signals = signals[signals['position'] == -1]
        ax.plot(long_signals.index, long_signals['signal_pct_change'], 'go', markersize=7, alpha=0.8, linestyle='None', label='Long Entry')
        ax.plot(short_signals.index, short_signals['signal_pct_change'], 'rv', markersize=7, alpha=0.8, linestyle='None', label='Short Entry')
        ax.set_title("Trading Signals", fontsize=14, fontweight='bold'), ax.legend()

    def _plot_rolling_correlation(self, ax):
        correlation = self.df['price_pct_change'].rolling(90).corr(self.df['signal_pct_change'])
        correlation.plot(ax=ax, color='purple', lw=2)
        ax.set_title("90-Day Rolling Correlation", fontsize=14, fontweight='bold'), ax.axhline(0, color='black', linestyle='--', lw=1)

    def _plot_correlation_lags(self, ax):
        """Plot correlation across different lags (-5 to +5)"""
        lags = list(range(-5, 6))  # -5 to +5
        correlations = []

        for lag in lags:
            if lag == 0:
                corr = self.df['price_pct_change'].corr(self.df['signal_pct_change'])
            else:
                # Use shift for lag analysis (lag >0 means signal leads price)
                corr = self.df['price_pct_change'].corr(self.df['signal_pct_change'].shift(lag))
            correlations.append(corr)

        # Create lag labels
        lag_labels = [f't+{i}' if i >= 0 else f't{i}' for i in lags]

        bars = ax.bar(lag_labels, correlations, color=['red' if x < 0 else 'blue' for x in correlations], alpha=0.7)
        ax.set_title("Signal-Asset Correlation Across Lags (-5 to +5)", fontsize=14, fontweight='bold')
        ax.set_ylabel("Correlation Coefficient", fontsize=12)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # Add value labels on bars
        for bar, corr in zip(bars, correlations):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.01),
                   '03f', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)

        ax.tick_params(axis='x', rotation=45)

    def _plot_regime_returns(self, ax):
        # Check if we have any forward return columns
        fwd_cols = [col for col in self.df.columns if col.startswith('fwd_return_')]
        if not fwd_cols:
            ax.text(0.5, 0.5, 'No forward returns\n(Short dataset)', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, style='italic')
            ax.set_title("Returns by Signal Regime (N/A)", fontsize=14, fontweight='bold')
            return

        # Use the first available forward return column
        fwd_col = fwd_cols[0]

        extreme_returns = self.df[self.df['signal_percentile'] > 0.9][fwd_col].mean() * 100
        normal_returns = self.df[self.df['signal_percentile'] <= 0.9][fwd_col].mean() * 100

        if np.isnan(extreme_returns) or np.isnan(normal_returns):
            ax.text(0.5, 0.5, 'Insufficient data\nfor regime analysis', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, style='italic')
            ax.set_title("Returns by Signal Regime (N/A)", fontsize=14, fontweight='bold')
            return

        # Create series for plotting
        data = pd.Series({'Normal Signal (<90%)': normal_returns, 'Extreme Signal (>90%)': extreme_returns})
        data.plot(kind='bar', ax=ax, color=['blue', 'red'])

        ax.set_title(f"Returns by Signal Regime ({self.config.strategy_entry_percentile:.0%})", fontsize=14, fontweight='bold')
        ax.set_ylabel("Avg. Return (%)")
        # Add value labels
        for i, v in enumerate(data):
            ax.text(i, v + (0.001 if v >= 0 else -0.001), '.2f', ha='center', va='bottom' if v >= 0 else 'top')
    
    def _plot_extreme_percentile_returns(self, ax):
        results = {}
        # Check if we have any forward return columns
        fwd_cols = [col for col in self.df.columns if col.startswith('fwd_return_')]
        if not fwd_cols:
            ax.text(0.5, 0.5, 'No forward returns\n(Short dataset)', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, style='italic')
            ax.set_title("Forward Returns Analysis (N/A)", fontsize=14, fontweight='bold')
            return

        # Use the first available forward return column
        fwd_col = fwd_cols[0] if fwd_cols else 'fwd_return_1w'

        for p in self.config.percentile_thresholds:
            subset = self.df[self.df['signal_percentile'] > (p / 100.0)] if p >= 50 else self.df[self.df['signal_percentile'] < (p / 100.0)]
            if not subset.empty and fwd_col in subset.columns:
                results[f'{p}%'] = subset[fwd_col].mean()

        if not results:
            ax.text(0.5, 0.5, 'Insufficient data\nfor analysis', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, style='italic')
            ax.set_title("Forward Returns Analysis (N/A)", fontsize=14, fontweight='bold')
            return

        (pd.Series(results) * 100).plot(kind='bar', ax=ax, color='teal')
        ax.set_title(f"7-Day Forward Returns by Signal Percentile (Entry: {self.config.strategy_entry_percentile:.0%})", fontsize=14, fontweight='bold'), ax.tick_params(axis='x', rotation=45)

    def _plot_performance_table(self, ax):
        ax.axis('off'), ax.set_title("Performance Comparison", fontsize=16, fontweight='bold', pad=20)
        pnl = self.pnl_results
        df_pnl = pnl['dataframe']
        bh_ret = (df_pnl['buy_hold_equity'].iloc[-1] / self.config.initial_capital - 1) if not df_pnl.empty else 0
        table_data = [["Metric", "Strategy", "Buy & Hold"],
                      ["Total Return", f"{pnl.get('total_return', 0):.2%}", f"{bh_ret:.2%}"],
                      ["Annualized Return", f"{pnl.get('annualized_return', 0):.2%}", "-"],
                      ["Sharpe Ratio", f"{pnl.get('sharpe_ratio', 0):.2f}", "-"],
                      ["Max Drawdown", f"{pnl.get('max_drawdown', 0):.2%}", "-"]]
        table = ax.table(cellText=table_data, colLabels=None, cellLoc='center', loc='center')
        table.auto_set_font_size(False), table.set_fontsize(14), table.scale(1, 1.8)
        for (i, j), cell in table.get_celld().items():
            if i == 0: cell.set_text_props(weight='bold', color='white'), cell.set_facecolor('darkslategray')
    
    def _plot_regression_table(self, ax):
        ax.axis('off'), ax.set_title("Regression Significance", fontsize=16, fontweight='bold', pad=20)
        results = []
        for name, res in self.regression_results.items():
            significance = '***' if res['p_value'] < 0.01 else '**' if res['p_value'] < 0.05 else '*' if res['p_value'] < 0.1 else ''
            # Format numeric values to 2 decimal places for better readability
            results.append([name,
                          f"{res['r_squared']:.2f}",
                          f"{res['coefficient']:.2f}",
                          f"{res['t_statistic']:.2f}",
                          f"{res['p_value']:.2f}",
                          significance])
        if not results: return
        table_data = pd.DataFrame(results, columns=["Period", "R²", "Coef.", "t-stat", "p-value", "Sig."])
        table = ax.table(cellText=table_data.values, colLabels=table_data.columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False), table.set_fontsize(12), table.scale(1, 2)
        for (i, j), cell in table.get_celld().items():
            if i == 0: cell.set_text_props(weight='bold', color='white'), cell.set_facecolor('darkslategray')

    def _plot_summary_text(self, ax):
        ax.axis('off')
        ax.set_title("Strategy Rules & Summary", fontsize=16, fontweight='bold')
        entry_pct_str = f"{self.config.strategy_entry_percentile:.0%}"
        
        text = (f"STRATEGY RULES:\n"
                f"  - Logic: {self.config.strategy_type.name.replace('_', ' ').title()}\n"
                f"  - Direction: {self.config.strategy_direction.name.replace('_', ' ').title()}\n"
                f"  - Entry (Long): Signal > {entry_pct_str}\n"
                f"  - Exit: Signal < 50%\n\n"
                f"KEY METRICS:\n"
                f"  - Observations: {len(self.df)}\n"
                f"  - Trades Executed: {self.pnl_results.get('num_trades', 0):.0f}\n"
                f"  - Final Portfolio: ${self.pnl_results['dataframe']['strategy_equity'].iloc[-1]:,.2f}")
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=14, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=1', facecolor='wheat', alpha=0.5))

    def generate_specialized_visualizations(self):
        logger.info("Generating specialized visualizations...")
        signal_name = self.config.signal_type.name.replace('_', ' ').title()
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle(f"Specialized {signal_name} Visualizations for {self.config.asset_type.name.upper()}", fontsize=20)
        ax1, ax2, ax3, ax4 = axes.flatten()
        
        ax1_twin = ax1.twinx()
        ax1.plot(self.df.index, self.df['price'], color='orange', label='Price')
        ax1_twin.plot(self.df.index, self.df['signal'], color='blue', label=signal_name)
        ax1.set_ylabel("Price ($)"), ax1_twin.set_ylabel(signal_name), ax1.set_title("Price vs. Signal Over Time")
        ax1.legend(loc='upper left'), ax1_twin.legend(loc='upper right')

        self.df['signal_pct_change'].rolling(30).std().plot(ax=ax2)
        ax2.set_title("30-Day Rolling Volatility of Signal Change"), ax2.set_ylabel("Volatility")
        
        sns.histplot(self.df['signal_pct_change'], ax=ax3, kde=True, bins=50)
        ax3.set_title("Distribution of Signal % Change")
        
        heatmap_data = self.df.groupby(pd.qcut(self.df['signal_percentile'], 10, labels=False, duplicates='drop'))[
            [f'fwd_return_{name}' for name in self.config.forward_periods_map]
        ].mean()
        if self.config.forward_periods_map and heatmap_data.shape[1] > 0:
            sns.heatmap(heatmap_data, ax=ax4, annot=True, cmap='viridis', fmt='.2%')
            ax4.set_title("Mean Forward Returns by Signal Percentile Decile")
            ax4.set_xlabel("Forward Period"), ax4.set_ylabel("Signal Percentile Decile")
        else:
            ax4.text(0.5, 0.5, 'No forward returns\n(Short dataset)', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title("Forward Returns Heatmap (N/A)")
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(Path(self.config.output_dir) / f"specialized_viz_{self.config.asset_type.value}_{self.config.signal_type.value}.png")
        plt.show()

    def _load_asset_data(self, asset_type: AssetType):
        logger.info(f"Loading {asset_type.name} price data...")
        data_dir = Path("Data")
        try:
            if asset_type == AssetType.BTC:
                filepath = data_dir / "Bitcoin Historical Data.csv"
                df = pd.read_csv(filepath, parse_dates=['Date'])
                df.rename(columns={'Date': 'date', 'Price': 'price'}, inplace=True)
                df['price'] = df['price'].replace({',': ''}, regex=True).astype(float)
            elif asset_type == AssetType.ETH:
                filepath = data_dir / "eth_price_history.csv"
                df = pd.read_csv(filepath, parse_dates=['Date'])
                df.rename(columns={'Date': 'date', 'Close': 'price'}, inplace=True)
                df['price'] = df['price'].replace({r'\$': '', ',': ''}, regex=True).astype(float)
            elif asset_type == AssetType.SP500:
                filepath = data_dir / "SP500 Historical Data.csv"
                df = pd.read_csv(filepath, parse_dates=['Date'], date_format='%b %d, %Y')
                df.rename(columns={'Date': 'date', 'Price': 'price', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Vol.': 'volume'}, inplace=True)
            elif asset_type == AssetType.NASDAQ:
                filepath = data_dir / "NASDAQ Historical Data.csv"
                df = pd.read_csv(filepath, parse_dates=['Date'], date_format='%b %d, %Y')
                df.rename(columns={'Date': 'date', 'Price': 'price', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Vol.': 'volume'}, inplace=True)
            elif asset_type == AssetType.OIL:
                filepath = data_dir / "Crude Oil WTI Futures Historical Data.csv"
                df = pd.read_csv(filepath, parse_dates=['Date'])
                df.rename(columns={'Date': 'date', 'Price': 'price', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Vol.': 'volume'}, inplace=True)
            elif asset_type == AssetType.GOLD:
                filepath = data_dir / "Gold Futures Historical Data.csv"
                df = pd.read_csv(filepath, parse_dates=['Date'])
                df.rename(columns={'Date': 'date', 'Price': 'price', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Vol.': 'volume'}, inplace=True)
                df['price'] = df['price'].replace({',': ''}, regex=True).astype(float)
            else:
                raise NotImplementedError
        except FileNotFoundError as e:
            logger.error(f"FATAL: Price data file not found: {e}. Aborting.")
            return pd.DataFrame()

        if 'date' in df.columns and df['date'].dt.tz is not None:
            df['date'] = df['date'].dt.tz_localize(None)

        df.set_index('date', inplace=True)

        # Calculate ATR if OHLC data is available for exit logic
        if self.config.exit_logic == ExitLogic.ATR_TRAILING_STOP:
            required_cols = ['high', 'low']
            if all(col in df.columns for col in required_cols):
                logger.info("Calculating ATR for trailing stop exit logic...")
                atr_period = self.config.exit_params.get('atr_period', 14)
                df['atr'] = ta.volatility.average_true_range(
                    high=df['high'],
                    low=df['low'],
                    close=df['price'],  # Use price as close
                    window=atr_period
                )
                logger.info(f"ATR calculated with {atr_period}-day period for trailing stops")
            else:
                logger.warning("ATR trailing stop requested but OHLC data not available - falling back to signal neutral exits")
                self.config.exit_logic = ExitLogic.SIGNAL_NEUTRAL

        logger.info(f"Loaded {len(df)} records from {filepath.name}")
        return df[['price']]  # Only return price column for compatibility

    def _load_signal_data(self, signal_type: SignalType):
        logger.info(f"Loading {signal_type.name} signal data...")
        data_dir = Path("Data")
        root_dir = Path(".")
        try:
            if signal_type == SignalType.HASHRATE:
                filepath = data_dir / 'hash-rate.json'
                with open(filepath, 'r') as f: data = json.load(f).get('hash-rate', [])
                df = pd.DataFrame(data, columns=['x', 'y'])
                logger.info(f"Hashrate data loaded: {len(df)} records, columns: {list(df.columns)}")
                df['date'] = pd.to_datetime(df['x'], unit='ms')
            elif signal_type == SignalType.USDC_ISSUANCE:
                filepath = data_dir / "usdc-usd-max.csv"
                df = pd.read_csv(filepath, parse_dates=['snapped_at'])
                logger.info(f"USDC data loaded: {len(df)} records, columns: {list(df.columns)}")
                df.rename(columns={'snapped_at': 'date', 'market_cap': 'y'}, inplace=True)
                logger.info(f"After renaming: columns: {list(df.columns)}")
            elif signal_type == SignalType.USDT_ISSUANCE:
                filepath = data_dir / "usdt-usd-max.csv"
                df = pd.read_csv(filepath, parse_dates=['snapped_at'])
                df.rename(columns={'snapped_at': 'date', 'market_cap': 'y'}, inplace=True)
            elif signal_type == SignalType.POLYMARKET_VWP:
                # Load Polymarket VWP signal (generated CSV in data/processed_signals/)
                filepath = Path("data/processed_signals/signal_vwp_china_trade.csv")
                df = pd.read_csv(filepath, parse_dates=['date'])
                logger.info(f"VWP signal loaded: {len(df)} records, columns: {list(df.columns)}")
                df['y'] = df['signal']  # Rename to match expected column
            elif signal_type == SignalType.POLYMARKET_VWEV:
                # Load Polymarket VWEV signal (generated CSV in data/processed_signals/)
                filepath = Path("data/processed_signals/signal_vwev_inflation.csv")
                df = pd.read_csv(filepath, parse_dates=['date'])
                df['y'] = df['signal']
            elif signal_type == SignalType.MASTER_MACRO:
                # Load Master Macro signal (generated by feature_factory.py)
                filepath = Path("data/processed_signals/signal_master_macro.csv")
                df = pd.read_csv(filepath, parse_dates=['date'])
                df['y'] = df['master_signal']  # Column name from feature_factory output
            elif signal_type == SignalType.MASTER_INFLATION:
                # Load Master Inflation signal (generated by feature_factory.py)
                filepath = Path("data/processed_signals/signal_master_inflation.csv")
                df = pd.read_csv(filepath, parse_dates=['date'])
                df['y'] = df['master_signal']  # Column name from feature_factory output
            elif signal_type == SignalType.MASTER_GEOPOLITICS:
                # Load Master Geopolitics signal (generated by feature_factory.py)
                filepath = Path("data/processed_signals/signal_master_geopolitics.csv")
                df = pd.read_csv(filepath, parse_dates=['date'])
                df['y'] = df['master_signal']  # Column name from feature_factory output
            elif signal_type == SignalType.GEOPOLITICAL_UNCERTAINTY:
                # Load our custom geopolitical uncertainty signal
                filepath = Path("data/processed_signals/geopolitical_uncertainty_signal.csv")
                df = pd.read_csv(filepath, parse_dates=['date'])
                df['y'] = df['signal']  # Use the signal column directly
            elif signal_type == SignalType.MILITARY_ACTIONS:
                # Load our custom military actions signal
                filepath = Path("data/processed_signals/military_actions_signal.csv")
                df = pd.read_csv(filepath, parse_dates=['date'])
                df['y'] = df['adjusted_signal']  # Use the adjusted_signal column
            elif signal_type == SignalType.INFLATION:
                # Load GMMV Polymarket inflation VWEV signal - expected annual inflation from market probabilities
                filepath = Path("data/processed_signals/signal_vwev_inflation.csv")
                df = pd.read_csv(filepath, parse_dates=['date'])
                df['y'] = df['signal']  # VWEV derived expected max annual inflation
            elif signal_type == SignalType.GDP:
                # Load GMMV Polymarket GDP VWEV signal - expected GDP growth from market probabilities
                filepath = Path("data/processed_signals/signal_vwev_gdp.csv")
                df = pd.read_csv(filepath, parse_dates=['date'])
                df['y'] = df['signal']  # VWEV derived expected GDP growth
            else:
                raise NotImplementedError
        except FileNotFoundError as e:
            logger.error(f"FATAL: Signal data file not found: {e}. Aborting.")
            return pd.DataFrame()

        if 'date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['date']) and df['date'].dt.tz is not None:
            df['date'] = df['date'].dt.tz_localize(None)
        df.set_index('date', inplace=True)
        logger.info(f"Loaded {len(df)} records from {filepath.name}")
        return df[['y']].rename(columns={'y': 'signal'})

# --- Automated Scoring ---
def calculate_signal_score(pnl_results: Dict, regression_results: Dict) -> float:
    """
    Calculate comprehensive strategy score using weighted trading metrics.

    Scoring Methodology:
    - Sharpe ratio (25%): Risk-adjusted returns (higher is better)
    - Maximum drawdown (20%): Capital preservation (lower is better)
    - Win rate (15%): Trading accuracy (higher is better)
    - Total return (20%): Absolute performance (higher is better)
    - Volatility (10%): Risk control (lower is better)
    - Calmar ratio (10%): Return per unit of drawdown (higher is better)

    Each metric is normalized to 0-1 scale using appropriate benchmarks, then weighted and combined.
    """
    # Define weights as specified
    weights = {
        'sharpe_ratio': 0.25,    # 25%
        'max_drawdown': 0.20,    # 20%
        'win_rate': 0.15,        # 15%
        'total_return': 0.20,    # 20%
        'volatility': 0.10,      # 10%
        'calmar_ratio': 0.10     # 10%
    }

    # Calculate normalized scores for each metric
    # Sharpe ratio: benchmark 1.5 (good hedge fund level)
    sharpe = pnl_results.get('sharpe_ratio', 0)
    sharpe_score = np.clip(sharpe / 1.5, 0, 1)

    # Maximum drawdown: benchmark 50% (lower is better, so invert)
    max_dd = abs(pnl_results.get('max_drawdown', 1))
    drawdown_score = np.clip(1 - (max_dd / 0.5), 0, 1)

    # Win rate: benchmark 60% (good trading system)
    win_rate = pnl_results.get('win_rate', 0)
    win_rate_score = np.clip(win_rate / 0.6, 0, 1)

    # Total return: benchmark 30% annual (ambitious but achievable)
    total_return = pnl_results.get('total_return', 0)
    return_score = np.clip(total_return / 0.3, 0, 1)

    # Volatility: benchmark 40% annual (lower is better, so invert)
    volatility = pnl_results.get('volatility', 1)
    volatility_score = np.clip(1 - (volatility / 0.4), 0, 1)

    # Calmar ratio: benchmark 1.0 (return per unit of drawdown)
    calmar = pnl_results.get('calmar_ratio', 0)
    calmar_score = np.clip(calmar / 1.0, 0, 1)

    # Calculate weighted final score (0-100 scale)
    final_score = (sharpe_score * weights['sharpe_ratio'] +
                   drawdown_score * weights['max_drawdown'] +
                   win_rate_score * weights['win_rate'] +
                   return_score * weights['total_return'] +
                   volatility_score * weights['volatility'] +
                   calmar_score * weights['calmar_ratio']) * 100

    return final_score

# --- Main Execution Block ---

def run_dual_asset_analysis(signal_type: SignalType):
    """Helper function to run the same signal analysis for both BTC and ETH."""
    print(f"\n--- Running {signal_type.name} analysis for both BTC and ETH ---")
    
    try:
        print("\nSelect strategy logic:")
        print("  1: Momentum (Buy High)"); print("  2: Mean-Reversion (Buy the Dip)")
        type_choice = input("Enter choice [default: 1]: ").strip() or "1"
        strategy_type = StrategyType.MEAN_REVERSION if type_choice == "2" else StrategyType.MOMENTUM
        
        print("\nSelect strategy direction:")
        print("  1: Long-Only"); print("  2: Short-Only"); print("  3: Long/Short")
        direction_choice = input("Enter choice [default: 1]: ").strip() or "1"
        if direction_choice == "1": strategy_direction = StrategyDirection.LONG_ONLY
        elif direction_choice == "2": strategy_direction = StrategyDirection.SHORT_ONLY
        elif direction_choice == "3": strategy_direction = StrategyDirection.LONG_SHORT
        else: raise ValueError("Invalid direction choice.")

        if strategy_type == StrategyType.MOMENTUM:
            prompt, default_pct, validation, error_msg = "Enter entry percentile for MOMENTUM (e.g., 90 for top 10%) [default: 90]: ", 90.0, lambda p: p > 50 and p < 100, "Percentile must be between 50 and 100."
        else:
            prompt, default_pct, validation, error_msg = "Enter entry percentile for MEAN-REVERSION (e.g., 10 for bottom 10%) [default: 10]: ", 10.0, lambda p: p < 50 and p > 0, "Percentile must be between 0 and 50."
            
        percentile_input = input(prompt).strip() or str(default_pct)
        entry_percentile_val = float(percentile_input)
        if not validation(entry_percentile_val): raise ValueError(error_msg)
        entry_percentile = entry_percentile_val / 100.0
        
    except ValueError as e:
        print(f"\n❌ Invalid input: {e}. Using default values.")
        strategy_type, strategy_direction, entry_percentile = StrategyType.MOMENTUM, StrategyDirection.LONG_ONLY, 0.90
        
    for asset_type in [AssetType.BTC, AssetType.ETH]:
        try:
            print("\n" + "="*25 + f" Analyzing on {asset_type.name.upper()} " + "="*25)
            config = AnalysisConfig(asset_type=asset_type, signal_type=signal_type, 
                                    strategy_type=strategy_type, strategy_direction=strategy_direction,
                                    strategy_entry_percentile=entry_percentile)
            print_strategy_definition(config)
            tester = SignalTesterAPI(config)
            tester.run()
        except Exception as e:
            logger.error(f"Analysis for {asset_type.name}/{signal_type.name} failed: {e}", exc_info=True)

def print_strategy_definition(config: AnalysisConfig):
    """Prints a clear definition of the backtest strategy being used."""
    print("\n" + "-"*60 + "\n🔬 Backtest Strategy Definition:")
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
    while True:
        print("\n" + "="*50 + "\n🚀 SIGNAL ANALYZER MAIN MENU\n" + "="*50)
        print("  1: Run Hashrate Analysis (on BTC & ETH)")
        print("  2: Run USDC Issuance Analysis (on BTC & ETH)")
        print("  3: Run USDT Issuance Analysis (on BTC & ETH)")
        print("\n  q: Quit\n" + "="*50)
        
        choice = input("Enter your choice: ").strip()
        
        if choice == '1': run_dual_asset_analysis(SignalType.HASHRATE)
        elif choice == '2': run_dual_asset_analysis(SignalType.USDC_ISSUANCE)
        elif choice == '3': run_dual_asset_analysis(SignalType.USDT_ISSUANCE)
        elif choice.lower() == 'q': print("Exiting."); break
        else: print("\n❌ Invalid choice. Please try again.")
        
        input("\nPress Enter to return to the menu...")
