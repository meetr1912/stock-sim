"""
Post-Earnings Announcement Drift (PEAD) Strategy Implementation
Analyzes and backtests the tendency for stock prices to continue moving 
in the direction of earnings surprises after earnings announcements.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class PEADStrategy:
    def __init__(self, price_data: pd.DataFrame, earnings_data: pd.DataFrame):
        """
        Initialize PEAD strategy with price and earnings data.
        
        Args:
            price_data: DataFrame with columns [ticker, date, open, high, low, close, volume]
            earnings_data: DataFrame with columns [ticker, earnings_date, release_time, ...]
        """
        self.price_data = price_data.copy()
        self.earnings_data = earnings_data.copy()
        self.results = {}
        
        # Ensure date columns are datetime (handle timezone-aware data)
        self.price_data['date'] = pd.to_datetime(self.price_data['date'], utc=True)
        if not earnings_data.empty:
            self.earnings_data['earnings_date'] = pd.to_datetime(self.earnings_data['earnings_date'], utc=True)
        
    def calculate_returns(self, window: int = 1) -> pd.DataFrame:
        """Calculate price returns for specified window."""
        price_data = self.price_data.sort_values(['ticker', 'date']).copy()
        
        # Calculate returns
        price_data['return'] = price_data.groupby('ticker')['close'].pct_change(window)
        price_data['return_1d'] = price_data.groupby('ticker')['close'].pct_change(1)
        price_data['return_5d'] = price_data.groupby('ticker')['close'].pct_change(5)
        price_data['return_10d'] = price_data.groupby('ticker')['close'].pct_change(10)
        
        return price_data
    
    def calculate_earnings_surprise(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate earnings surprise using price reaction.
        This is a simplified approach using price gaps as proxy for earnings surprise.
        """
        if self.earnings_data.empty:
            # If no earnings data, create synthetic earnings surprise based on large price gaps
            price_data = price_data.sort_values(['ticker', 'date']).copy()
            
            # Calculate overnight gap (assuming earnings released after market close)
            price_data['prev_close'] = price_data.groupby('ticker')['close'].shift(1)
            price_data['gap'] = (price_data['open'] - price_data['prev_close']) / price_data['prev_close']
            
            # Identify potential earnings dates (large gaps > 2 standard deviations)
            price_data['gap_abs'] = price_data['gap'].abs()
            threshold = price_data['gap_abs'].quantile(0.95)  # Top 5% of gaps
            
            earnings_proxy = price_data[price_data['gap_abs'] > threshold].copy()
            earnings_proxy['earnings_surprise'] = earnings_proxy['gap']
            earnings_proxy['is_earnings_day'] = True
            
            return earnings_proxy
        else:
            # Use actual earnings data (to be implemented when earnings data is available)
            return pd.DataFrame()
    
    def identify_pead_opportunities(self, lookback_days: int = 1, forward_days: int = 10) -> pd.DataFrame:
        """
        Identify PEAD opportunities by finding earnings events and measuring subsequent drift.
        
        Args:
            lookback_days: Days to look back for initial reaction
            forward_days: Days to measure drift after earnings
        """
        price_data = self.calculate_returns()
        earnings_events = self.calculate_earnings_surprise(price_data)
        
        if earnings_events.empty:
            print("No earnings events identified")
            return pd.DataFrame()
        
        pead_signals = []
        
        for _, event in earnings_events.iterrows():
            ticker = event['ticker']
            event_date = event['date']
            surprise_direction = np.sign(event['earnings_surprise'])
            
            # Get price data for this ticker around the event
            ticker_prices = price_data[price_data['ticker'] == ticker].copy()
            ticker_prices = ticker_prices.sort_values('date')
            
            # Find the event in the price data
            event_idx = ticker_prices[ticker_prices['date'] == event_date].index
            if len(event_idx) == 0:
                continue
                
            event_idx = event_idx[0]
            ticker_idx = ticker_prices.index.get_loc(event_idx)
            
            # Check if we have enough data for forward measurement
            if ticker_idx + forward_days >= len(ticker_prices):
                continue
            
            # Calculate forward returns
            event_close = ticker_prices.iloc[ticker_idx]['close']
            forward_prices = ticker_prices.iloc[ticker_idx+1:ticker_idx+1+forward_days]
            
            if len(forward_prices) == 0:
                continue
            
            # Calculate cumulative returns over the drift period
            forward_returns = []
            for i, row in forward_prices.iterrows():
                days_after = (row['date'] - event_date).days
                cum_return = (row['close'] - event_close) / event_close
                forward_returns.append({
                    'ticker': ticker,
                    'event_date': event_date,
                    'days_after_earnings': days_after,
                    'cumulative_return': cum_return,
                    'surprise_direction': surprise_direction,
                    'initial_surprise': event['earnings_surprise']
                })
            
            pead_signals.extend(forward_returns)
        
        return pd.DataFrame(pead_signals)
    
    def backtest_strategy(self, holding_period: int = 5, min_surprise_threshold: float = 0.02) -> Dict:
        """
        Backtest the PEAD strategy.
        
        Args:
            holding_period: Number of days to hold position after earnings
            min_surprise_threshold: Minimum absolute surprise to trigger trade
        """
        pead_data = self.identify_pead_opportunities(forward_days=holding_period)
        
        if pead_data.empty:
            return {"status": "No PEAD opportunities found", "total_trades": 0}
        
        # Filter for significant surprises only
        significant_events = pead_data[
            abs(pead_data['initial_surprise']) >= min_surprise_threshold
        ].copy()
        
        # Get final returns for each holding period
        final_returns = significant_events[
            significant_events['days_after_earnings'] == holding_period
        ].copy()
        
        if final_returns.empty:
            return {"status": "No complete holding periods found", "total_trades": 0}
        
        # Calculate strategy performance
        final_returns['strategy_return'] = (
            final_returns['surprise_direction'] * final_returns['cumulative_return']
        )
        
        # Performance metrics
        total_trades = len(final_returns)
        win_rate = (final_returns['strategy_return'] > 0).mean()
        avg_return = final_returns['strategy_return'].mean()
        total_return = final_returns['strategy_return'].sum()
        sharpe_ratio = avg_return / final_returns['strategy_return'].std() if final_returns['strategy_return'].std() > 0 else 0
        
        # Monthly breakdown
        final_returns['year_month'] = final_returns['event_date'].dt.to_period('M')
        monthly_returns = final_returns.groupby('year_month')['strategy_return'].agg(['sum', 'count', 'mean'])
        
        results = {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'average_return_per_trade': avg_return,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'holding_period_days': holding_period,
            'min_surprise_threshold': min_surprise_threshold,
            'monthly_performance': monthly_returns,
            'trade_details': final_returns
        }
        
        self.results = results
        return results
    
    def plot_pead_analysis(self, max_days: int = 10):
        """Plot PEAD drift patterns."""
        pead_data = self.identify_pead_opportunities(forward_days=max_days)
        
        if pead_data.empty:
            print("No PEAD data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Average drift by surprise direction
        avg_drift = pead_data.groupby(['days_after_earnings', 'surprise_direction'])['cumulative_return'].mean().unstack()
        avg_drift.plot(ax=axes[0,0], title='Average Price Drift After Earnings by Surprise Direction')
        axes[0,0].set_xlabel('Days After Earnings')
        axes[0,0].set_ylabel('Cumulative Return')
        axes[0,0].legend(['Negative Surprise', 'Positive Surprise'])
        axes[0,0].grid(True)
        
        # 2. Distribution of initial surprises
        initial_surprises = pead_data[pead_data['days_after_earnings'] == 1]['initial_surprise']
        axes[0,1].hist(initial_surprises, bins=50, alpha=0.7)
        axes[0,1].set_title('Distribution of Earnings Surprises (Price Gaps)')
        axes[0,1].set_xlabel('Initial Surprise (Return)')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].grid(True)
        
        # 3. PEAD effect by holding period
        if hasattr(self, 'results') and self.results:
            holding_periods = range(1, max_days + 1)
            avg_returns = []
            for period in holding_periods:
                period_data = pead_data[pead_data['days_after_earnings'] == period]
                if not period_data.empty:
                    avg_return = (period_data['surprise_direction'] * period_data['cumulative_return']).mean()
                    avg_returns.append(avg_return)
                else:
                    avg_returns.append(0)
            
            axes[1,0].plot(holding_periods, avg_returns, marker='o')
            axes[1,0].set_title('Average Strategy Return by Holding Period')
            axes[1,0].set_xlabel('Holding Period (Days)')
            axes[1,0].set_ylabel('Average Return')
            axes[1,0].grid(True)
        
        # 4. Win rate by surprise magnitude
        if 'trade_details' in getattr(self, 'results', {}):
            trade_details = self.results['trade_details']
            trade_details['surprise_magnitude'] = trade_details['initial_surprise'].abs()
            trade_details['surprise_bin'] = pd.cut(trade_details['surprise_magnitude'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
            win_rate_by_magnitude = trade_details.groupby('surprise_bin')['strategy_return'].apply(lambda x: (x > 0).mean())
            
            win_rate_by_magnitude.plot(kind='bar', ax=axes[1,1])
            axes[1,1].set_title('Win Rate by Surprise Magnitude')
            axes[1,1].set_ylabel('Win Rate')
            axes[1,1].tick_params(axis='x', rotation=45)
            axes[1,1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self) -> str:
        """Generate a comprehensive PEAD strategy report."""
        if not hasattr(self, 'results') or not self.results:
            return "No backtest results available. Run backtest_strategy() first."
        
        results = self.results
        
        report = f"""
PEAD STRATEGY BACKTEST REPORT
{'='*50}

STRATEGY PARAMETERS:
- Holding Period: {results['holding_period_days']} days
- Minimum Surprise Threshold: {results['min_surprise_threshold']:.2%}

PERFORMANCE SUMMARY:
- Total Trades: {results['total_trades']}
- Win Rate: {results['win_rate']:.2%}
- Average Return per Trade: {results['average_return_per_trade']:.2%}
- Total Strategy Return: {results['total_return']:.2%}
- Sharpe Ratio: {results['sharpe_ratio']:.3f}

TRADE BREAKDOWN:
- Winning Trades: {int(results['total_trades'] * results['win_rate'])}
- Losing Trades: {int(results['total_trades'] * (1 - results['win_rate']))}

MONTHLY PERFORMANCE:
"""
        
        if 'monthly_performance' in results and not results['monthly_performance'].empty:
            monthly_df = results['monthly_performance']
            report += monthly_df.to_string()
        
        report += f"""

DATA COVERAGE:
- Price Data Records: {len(self.price_data)}
- Unique Tickers: {self.price_data['ticker'].nunique()}
- Date Range: {self.price_data['date'].min()} to {self.price_data['date'].max()}
- Earnings Events: {results['total_trades']} (identified via price gaps)

NOTES:
- This analysis uses price gaps as a proxy for earnings surprises
- For production use, integrate actual earnings estimates and results
- Consider transaction costs, slippage, and market impact in live trading
"""
        
        return report

# Example usage function
def run_pead_analysis(price_data: pd.DataFrame, earnings_data: pd.DataFrame = None) -> PEADStrategy:
    """Run complete PEAD analysis on provided data."""
    if earnings_data is None:
        earnings_data = pd.DataFrame()
    
    print("Initializing PEAD Strategy...")
    strategy = PEADStrategy(price_data, earnings_data)
    
    print("Running backtest...")
    results = strategy.backtest_strategy(holding_period=5, min_surprise_threshold=0.03)
    
    print("Generating visualizations...")
    strategy.plot_pead_analysis()
    
    print("Generating report...")
    report = strategy.generate_report()
    print(report)
    
    return strategy

if __name__ == "__main__":
    # This will be used when actual data is available
    print("PEAD Strategy module loaded successfully")
    print("Use run_pead_analysis(price_data) to execute the strategy") 