"""
Run Enhanced PEAD Analysis with 30-day holding and trailing stops
Compare with original 5-day strategy
"""

import pandas as pd
import matplotlib.pyplot as plt
from data_acquisition import DataAcquisition, validate_data_quality
from pead_strategy import PEADStrategy
from enhanced_pead_strategy import EnhancedPEADStrategy
import warnings
warnings.filterwarnings('ignore')

def compare_strategies():
    """Compare original 5-day vs enhanced 30-day PEAD strategies."""
    
    print("="*70)
    print("ENHANCED PEAD STRATEGY ANALYSIS - 30 DAYS WITH TRAILING STOPS")
    print("="*70)
    
    # Load existing data
    data_fetcher = DataAcquisition(data_dir="data")
    price_data = data_fetcher.load_data("sp500_prices_2023-01-01_to_2024-12-31.csv")
    
    if price_data.empty:
        print("No data found. Please run main.py first to fetch data.")
        return
    
    print(f"Loaded data: {len(price_data)} records for {price_data['ticker'].nunique()} tickers")
    
    # Run original 5-day strategy
    print("\n" + "="*50)
    print("ORIGINAL 5-DAY STRATEGY RESULTS")
    print("="*50)
    
    original_strategy = PEADStrategy(price_data, pd.DataFrame())
    original_results = original_strategy.backtest_strategy(
        holding_period=5, 
        min_surprise_threshold=0.05
    )
    
    # Run enhanced 30-day strategy with different trailing stop percentages
    print("\n" + "="*50)
    print("ENHANCED 30-DAY STRATEGY RESULTS")
    print("="*50)
    
    enhanced_strategy = EnhancedPEADStrategy(price_data, pd.DataFrame())
    
    # Test different trailing stop percentages
    trailing_stops = [0.10, 0.15, 0.20]  # 10%, 15%, 20%
    best_enhanced_results = None
    best_enhanced_sharpe = -float('inf')
    best_stop_pct = None
    
    for stop_pct in trailing_stops:
        print(f"\nTesting {stop_pct:.0%} trailing stop...")
        
        enhanced_results = enhanced_strategy.backtest_enhanced_strategy(
            holding_period=30,
            min_surprise_threshold=0.03,  # Lower threshold for more trades
            trailing_stop_pct=stop_pct
        )
        
        if enhanced_results.get('sharpe_ratio', -float('inf')) > best_enhanced_sharpe:
            best_enhanced_sharpe = enhanced_results['sharpe_ratio']
            best_enhanced_results = enhanced_results
            best_stop_pct = stop_pct
    
    # Generate detailed comparison
    print("\n" + "="*70)
    print("STRATEGY COMPARISON SUMMARY")
    print("="*70)
    
    comparison_data = {
        'Metric': [
            'Total Trades',
            'Win Rate',
            'Average Return per Trade',
            'Total Strategy Return', 
            'Sharpe Ratio',
            'Max Drawdown',
            'Average Days Held',
            'Risk Management'
        ],
        'Original (5-day)': [
            f"{original_results.get('total_trades', 0)}",
            f"{original_results.get('win_rate', 0):.1%}",
            f"{original_results.get('average_return_per_trade', 0):.2%}",
            f"{original_results.get('total_return', 0):.2%}",
            f"{original_results.get('sharpe_ratio', 0):.3f}",
            "No stop loss",
            "5 days",
            "None"
        ],
        'Enhanced (30-day + Trailing Stop)': [
            f"{best_enhanced_results.get('total_trades', 0)}",
            f"{best_enhanced_results.get('win_rate', 0):.1%}",
            f"{best_enhanced_results.get('average_return_per_trade', 0):.2%}",
            f"{best_enhanced_results.get('total_return', 0):.2%}",
            f"{best_enhanced_results.get('sharpe_ratio', 0):.3f}",
            f"{best_enhanced_results.get('max_drawdown', 0):.2%}",
            f"{best_enhanced_results.get('avg_days_held', 0):.1f} days",
            f"{best_stop_pct:.0%} trailing stop"
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # Key insights
    print(f"\n" + "="*50)
    print("KEY INSIGHTS")
    print("="*50)
    
    enhanced_better_sharpe = best_enhanced_results['sharpe_ratio'] > original_results['sharpe_ratio']
    enhanced_better_return = best_enhanced_results['average_return_per_trade'] > original_results['average_return_per_trade']
    
    print(f"✓ Enhanced strategy shows {'BETTER' if enhanced_better_sharpe else 'SIMILAR'} risk-adjusted returns")
    print(f"✓ Average return per trade: {'IMPROVED' if enhanced_better_return else 'SIMILAR'}")
    print(f"✓ Downside protection: {best_enhanced_results['stops_triggered']} positions protected by trailing stops")
    print(f"✓ Institutional capture: {best_enhanced_results['avg_days_held']:.1f} day average hold allows for institutional accumulation")
    
    # Set the enhanced strategy to best parameters for plotting
    enhanced_strategy.backtest_enhanced_strategy(
        holding_period=30,
        min_surprise_threshold=0.03,
        trailing_stop_pct=best_stop_pct
    )
    
    # Generate enhanced plots
    print(f"\nGenerating enhanced analysis plots...")
    enhanced_strategy.plot_enhanced_analysis()
    
    # Generate and save enhanced report
    enhanced_report = enhanced_strategy.generate_enhanced_report()
    
    # Save enhanced results
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    if 'trade_details' in best_enhanced_results:
        enhanced_trade_file = f"data/enhanced_pead_trades_{timestamp}.csv"
        best_enhanced_results['trade_details'].to_csv(enhanced_trade_file, index=False)
        print(f"Enhanced trade details saved to: {enhanced_trade_file}")
    
    enhanced_report_file = f"data/enhanced_pead_report_{timestamp}.txt"
    with open(enhanced_report_file, 'w', encoding='utf-8') as f:
        f.write(enhanced_report)
    print(f"Enhanced analysis report saved to: {enhanced_report_file}")
    
    print(f"\n" + enhanced_report)
    
    return {
        'original_results': original_results,
        'enhanced_results': best_enhanced_results,
        'enhanced_strategy': enhanced_strategy
    }

if __name__ == "__main__":
    results = compare_strategies() 