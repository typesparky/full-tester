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

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

# --- Enums and Configuration ---

class AssetType(Enum):
    BTC = "btc"
    ETH = "eth"

class SignalType(Enum):
    HASHRATE = "hashrate"
    USDC_ISSUANCE = "usdc_issuance"
    USDT_ISSUANCE = "usdt_issuance"

class StrategyDirection(Enum):
    LONG_ONLY = "long_only"
    SHORT_ONLY = "short_only"
    LONG_SHORT = "long_short"

class StrategyType(Enum):
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"

@dataclass
class AnalysisConfig:
    """The main control panel for configuring an analysis run."""
    asset_type: AssetType
    signal_type: SignalType
    strategy_type: StrategyType = StrategyType.MOMENTUM
    strategy_direction: StrategyDirection = StrategyDirection.LONG_SHORT
    strategy_entry_percentile: float = 0.90
    signal_lookback_months: int = 6
    forward_periods_map: dict = field(default_factory=lambda: {'1w': 7, '1m': 30, '3m': 90, '6m': 180})
    percentile_thresholds: List[float] = field(default_factory=lambda: [1, 5, 10, 20, 80, 90, 95, 99])
    output_dir: str = "analysis_results"
    initial_capital: float = 10000.0
    
    @property
    def output_filename(self) -> str:
        entry_pct_str = int(self.strategy_entry_percentile * 100)
        return f"{self.asset_type.value}_{self.signal_type.value}_{self.strategy_type.value}_{self.strategy_direction.value}_{entry_pct_str}pct.png"

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

        df.sort_index(inplace=True)

        # Debug: Check if columns exist before accessing them
        logger.info(f"Columns available: {list(df.columns)}")
        if 'price' not in df.columns:
            logger.error("ERROR: 'price' column not found in DataFrame!")
            return pd.DataFrame()
        if 'signal' not in df.columns:
            logger.error("ERROR: 'signal' column not found in DataFrame!")
            return pd.DataFrame()

        df = df[(df['price'] > 0) & (df['signal'] > 0)].dropna()
        
        df['price_pct_change'] = df['price'].pct_change()
        df['signal_pct_change'] = df['signal'].pct_change()

        for name, period in self.config.forward_periods_map.items():
            df[f'fwd_return_{name}'] = df['price'].pct_change(periods=period).shift(-period)
            
        signal_lookback_days = self.config.signal_lookback_months * 30
        df['signal_percentile'] = df['signal_pct_change'].rolling(
            window=signal_lookback_days, min_periods=signal_lookback_days // 3
        ).apply(lambda x: stats.percentileofscore(pd.Series(x), pd.Series(x).iloc[-1]) / 100.0, raw=False)
        
        processed_df = df.dropna(subset=['price_pct_change', 'signal_pct_change', 'signal_percentile'])
        logger.info(f"Data preparation complete. Final shape for analysis: {processed_df.shape}")
        return processed_df

    def _calculate_pnl(self) -> Dict:
        df_pnl = self.df.copy()
        
        if self.config.strategy_type == StrategyType.MOMENTUM:
            long_entry = df_pnl['signal_percentile'] > self.config.strategy_entry_percentile
            short_entry = df_pnl['signal_percentile'] < (1.0 - self.config.strategy_entry_percentile)
        else: # MEAN_REVERSION
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
        # Fixed Sharpe ratio calculation with proper risk-free rate (2% annual, converted to daily)
        risk_free_rate_annual = 0.02
        risk_free_rate_daily = (1 + risk_free_rate_annual) ** (1/365) - 1
        excess_return = results['annualized_return'] - risk_free_rate_annual
        results['sharpe_ratio'] = (excess_return / strat_std) if strat_std > 0 else 0
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
        for name in self.config.forward_periods_map.keys():
            subset = self.df[['signal_pct_change', f'fwd_return_{name}']].dropna()
            if len(subset) < 30: continue
            
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
        self._plot_regime_returns(fig.add_subplot(gs[2, 2]))
        self._plot_extreme_percentile_returns(fig.add_subplot(gs[2, 3]))

        self._plot_performance_table(fig.add_subplot(gs[3, :2]))
        self._plot_regression_table(fig.add_subplot(gs[3, 2:]))
        self._plot_summary_text(fig.add_subplot(gs[4, :]))
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.savefig(Path(self.config.output_dir) / self.config.output_filename, dpi=150)
        plt.show()

    def _plot_time_series(self, ax, column, title, color):
        self.df[column].plot(ax=ax, color=color, lw=1.5), ax.set_title(title, fontsize=14, fontweight='bold'), ax.set_ylabel(title)
        
    def _plot_distribution(self, ax):
        sns.histplot(self.df['signal_pct_change'], ax=ax, kde=True, bins=50, color='green')
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

    def _plot_regime_returns(self, ax):
        extreme_returns = self.df[self.df['signal_percentile'] > 0.9]['fwd_return_1m'].mean() * 100
        normal_returns = self.df[self.df['signal_percentile'] <= 0.9]['fwd_return_1m'].mean() * 100
        sns.barplot(x=['Normal Signal (<90%)', 'Extreme Signal (>90%)'], y=[normal_returns, extreme_returns], ax=ax)
        ax.set_title("Returns by Signal Regime (30d Fwd)", fontsize=14, fontweight='bold'), ax.set_ylabel("Avg. Return (%)")
        if not np.isnan(normal_returns) and not np.isnan(extreme_returns): ax.bar_label(ax.containers[0], fmt='%.2f%%')
    
    def _plot_extreme_percentile_returns(self, ax):
        results = {}
        for p in self.config.percentile_thresholds:
            subset = self.df[self.df['signal_percentile'] > (p / 100.0)] if p >= 50 else self.df[self.df['signal_percentile'] < (p / 100.0)]
            if not subset.empty: results[f'{p}%'] = subset['fwd_return_1m'].mean()
        if not results: return
        (pd.Series(results) * 100).plot(kind='bar', ax=ax, color='teal')
        ax.set_title("30d Fwd Returns at Extremes", fontsize=14, fontweight='bold'), ax.tick_params(axis='x', rotation=45)

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
        sns.heatmap(heatmap_data, ax=ax4, annot=True, cmap='viridis', fmt='.2%')
        ax4.set_title("Mean Forward Returns by Signal Percentile Decile")
        ax4.set_xlabel("Forward Period"), ax4.set_ylabel("Signal Percentile Decile")
        
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
            else: 
                raise NotImplementedError
        except FileNotFoundError as e:
            logger.error(f"FATAL: Price data file not found: {e}. Aborting.")
            return pd.DataFrame()
            
        if 'date' in df.columns and df['date'].dt.tz is not None:
            df['date'] = df['date'].dt.tz_localize(None)
        
        df.set_index('date', inplace=True)
        logger.info(f"Loaded {len(df)} records from {filepath.name}")
        return df[['price']]

    def _load_signal_data(self, signal_type: SignalType):
        logger.info(f"Loading {signal_type.name} signal data...")
        data_dir = Path("Data")
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