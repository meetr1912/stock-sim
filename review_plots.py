"""
Review Enhanced PEAD Strategy Plots
30-day holding periods with trailing stops vs original 5-day strategy
"""

from enhanced_pead_strategy import EnhancedPEADStrategy
from pead_strategy import PEADStrategy
from data_acquisition import DataAcquisition
import pandas as pd
import matplotlib.pyplot as plt

def main():
    print("="*70)
    print("REVIEWING ENHANCED 30-DAY PEAD STRATEGY PLOTS")
    print("="*70)
    
    # Load data
    data_fetcher = DataAcquisition(data_dir='data')
    price_data = data_fetcher.load_data('sp500_prices_2023-01-01_to_2024-12-31.csv')
    
    print(f'Data loaded: {len(price_data)} records for {price_data["ticker"].nunique()} tickers')
    
    # ENHANCED 30-DAY STRATEGY
    print("\n" + "="*50)
    print("ENHANCED 30-DAY STRATEGY WITH TRAILING STOPS")
    print("="*50)
    
    enhanced_strategy = EnhancedPEADStrategy(price_data, pd.DataFrame())
    
    # Run with optimal parameters
    enhanced_results = enhanced_strategy.backtest_enhanced_strategy(
        holding_period=30,           # 30-day max holding for institutional accumulation
        min_surprise_threshold=0.03, # 3% minimum surprise threshold
        trailing_stop_pct=0.15      # 15% trailing stop
    )
    
    print(f"ENHANCED STRATEGY RESULTS:")
    print(f"- Total Trades: {enhanced_results['total_trades']}")
    print(f"- Win Rate: {enhanced_results['win_rate']:.1%}")
    print(f"- Average Return per Trade: {enhanced_results['average_return_per_trade']:.2%}")
    print(f"- Total Strategy Return: {enhanced_results['total_return']:.2%}")
    print(f"- Sharpe Ratio: {enhanced_results['sharpe_ratio']:.3f}")
    print(f"- Trailing Stops Triggered: {enhanced_results['stops_triggered']} ({enhanced_results['stop_rate']:.1%})")
    print(f"- Average Days Held: {enhanced_results['avg_days_held']:.1f} days")
    print(f"- Institutional Performance: {enhanced_results['institutional_performance']:.2%}")
    
    # ORIGINAL 5-DAY STRATEGY FOR COMPARISON
    print("\n" + "="*50)
    print("ORIGINAL 5-DAY STRATEGY (FOR COMPARISON)")
    print("="*50)
    
    original_strategy = PEADStrategy(price_data, pd.DataFrame())
    original_results = original_strategy.backtest_strategy(
        holding_period=5,
        min_surprise_threshold=0.05
    )
    
    print(f"ORIGINAL STRATEGY RESULTS:")
    print(f"- Total Trades: {original_results['total_trades']}")
    print(f"- Win Rate: {original_results['win_rate']:.1%}")
    print(f"- Average Return per Trade: {original_results['average_return_per_trade']:.2%}")
    print(f"- Total Strategy Return: {original_results['total_return']:.2%}")
    print(f"- Sharpe Ratio: {original_results['sharpe_ratio']:.3f}")
    
    # COMPARISON SUMMARY
    print("\n" + "="*50)
    print("KEY IMPROVEMENTS WITH 30-DAY + TRAILING STOPS")
    print("="*50)
    
    total_return_improvement = ((enhanced_results['total_return'] - original_results['total_return']) / original_results['total_return']) * 100
    trade_increase = ((enhanced_results['total_trades'] - original_results['total_trades']) / original_results['total_trades']) * 100
    
    print(f"✅ Total Return Improvement: +{total_return_improvement:.1f}%")
    print(f"✅ More Trading Opportunities: +{trade_increase:.1f}%")
    print(f"✅ Risk Protection: {enhanced_results['stops_triggered']} positions protected")
    print(f"✅ Institutional Capture: {enhanced_results['avg_days_held']:.1f} day avg hold vs 5 day fixed")
    
    # Generate enhanced visualizations
    print(f"\nGenerating enhanced strategy visualizations...")
    enhanced_strategy.plot_enhanced_analysis()
    
    print(f"\n" + "="*70)
    print("PLOT DESCRIPTIONS:")
    print("="*70)
    print("1. Return Distribution - Shows spread of returns with trailing stops")
    print("2. Days Held vs Return - Red dots = stopped out, Green = full hold")
    print("3. Institutional Activity - Higher scores = better performance")
    print("4. Performance by Surprise Direction - Long vs short bias")
    print("5. Cumulative Performance - Strategy evolution over time")
    print("6. Trailing Stop Effectiveness - Max gain vs final return")
    
    plt.show()

if __name__ == "__main__":
    main() 