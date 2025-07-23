"""
Main execution script for PEAD Strategy Implementation
Orchestrates data acquisition, processing, and strategy execution
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd

# Import our modules
from data_acquisition import DataAcquisition, validate_data_quality
from pead_strategy import PEADStrategy, run_pead_analysis

def main():
    """Main execution function for PEAD strategy."""
    print("="*60)
    print("POST-EARNINGS ANNOUNCEMENT DRIFT (PEAD) STRATEGY")
    print("="*60)
    
    # Initialize data acquisition
    data_fetcher = DataAcquisition(data_dir="data")
    
    # Configuration
    START_DATE = "2023-01-01"
    END_DATE = "2024-12-31"
    SAMPLE_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'WMT']
    
    print(f"Configuration:")
    print(f"- Analysis Period: {START_DATE} to {END_DATE}")
    print(f"- Sample Tickers: {len(SAMPLE_TICKERS)} stocks")
    print(f"- Data Directory: {data_fetcher.data_dir}")
    
    # Check if data already exists
    price_file = f"sp500_prices_{START_DATE}_to_{END_DATE}.csv"
    price_data = data_fetcher.load_data(price_file)
    
    if price_data.empty:
        print(f"\nNo existing data found. Fetching new data...")
        
        # Fetch fresh data
        dataset = data_fetcher.fetch_full_dataset(
            start_date=START_DATE,
            end_date=END_DATE,
            tickers=SAMPLE_TICKERS
        )
        
        price_data = dataset['prices']
        
        if price_data.empty:
            print("ERROR: Failed to fetch price data. Please check your internet connection and try again.")
            return
            
        print(f"Successfully fetched data for {price_data['ticker'].nunique()} tickers")
    else:
        print(f"\nLoaded existing data from {price_file}")
        print(f"Data contains {len(price_data)} records for {price_data['ticker'].nunique()} tickers")
    
    # Validate data quality
    print(f"\nValidating data quality...")
    quality_report = validate_data_quality(price_data)
    
    print(f"Data Quality Status: {quality_report['status']}")
    if quality_report['status'] == 'FAILED':
        print("Data quality issues detected:")
        for key, value in quality_report.items():
            if key != 'status' and value:
                print(f"  {key}: {value}")
        
        response = input("\nContinue with analysis despite data quality issues? (y/n): ")
        if response.lower() != 'y':
            print("Analysis aborted due to data quality concerns.")
            return
    
    # Run PEAD Analysis
    print(f"\n{'='*40}")
    print("STARTING PEAD ANALYSIS")
    print(f"{'='*40}")
    
    try:
        # Create strategy instance
        strategy = PEADStrategy(price_data, pd.DataFrame())
        
        # Test different holding periods
        holding_periods = [3, 5, 10]
        surprise_thresholds = [0.02, 0.03, 0.05]
        
        best_results = None
        best_params = None
        best_sharpe = -float('inf')
        
        print(f"\nTesting parameter combinations...")
        
        for holding_period in holding_periods:
            for threshold in surprise_thresholds:
                print(f"Testing: {holding_period} days holding, {threshold:.1%} surprise threshold")
                
                results = strategy.backtest_strategy(
                    holding_period=holding_period,
                    min_surprise_threshold=threshold
                )
                
                if results.get('sharpe_ratio', -float('inf')) > best_sharpe:
                    best_sharpe = results['sharpe_ratio']
                    best_results = results
                    best_params = (holding_period, threshold)
        
        # Display best results
        if best_results and best_results.get('total_trades', 0) > 0:
            print(f"\n{'='*40}")
            print("BEST PARAMETER COMBINATION FOUND")
            print(f"{'='*40}")
            print(f"Holding Period: {best_params[0]} days")
            print(f"Surprise Threshold: {best_params[1]:.1%}")
            print(f"Sharpe Ratio: {best_sharpe:.3f}")
            
            # Set strategy to best parameters and generate report
            strategy.backtest_strategy(
                holding_period=best_params[0],
                min_surprise_threshold=best_params[1]
            )
            
            # Generate and display report
            report = strategy.generate_report()
            print(report)
            
            # Save results
            save_results(strategy, best_results, data_fetcher.data_dir)
            
            # Generate plots
            print(f"\nGenerating analysis plots...")
            strategy.plot_pead_analysis()
            
        else:
            print("\nNo profitable parameter combinations found.")
            print("This could indicate:")
            print("1. The PEAD effect is not present in this dataset")
            print("2. The time period or stocks selected don't exhibit PEAD")
            print("3. The proxy method for earnings surprise needs refinement")
            print("4. More comprehensive earnings data is needed")
    
    except Exception as e:
        print(f"Error during PEAD analysis: {e}")
        print("This might be due to insufficient data or parameter issues.")
        return
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print("Next steps:")
    print("1. Review the generated plots and reports")
    print("2. Consider acquiring more comprehensive earnings data")
    print("3. Test with longer time periods or different stock universes")
    print("4. Implement risk management and position sizing rules")

def save_results(strategy: PEADStrategy, results: dict, data_dir: str):
    """Save strategy results to files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed trade results
    if 'trade_details' in results:
        trade_file = os.path.join(data_dir, f"pead_trades_{timestamp}.csv")
        results['trade_details'].to_csv(trade_file, index=False)
        print(f"Trade details saved to: {trade_file}")
    
    # Save summary report
    report_file = os.path.join(data_dir, f"pead_report_{timestamp}.txt")
    with open(report_file, 'w') as f:
        f.write(strategy.generate_report())
    print(f"Analysis report saved to: {report_file}")

def setup_environment():
    """Check and setup the required environment."""
    required_packages = [
        'yfinance', 'pandas', 'numpy', 'matplotlib', 
        'seaborn', 'requests', 'bs4'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease install missing packages using:")
        print("pip install -r requirements.txt")
        return False
    
    return True

if __name__ == "__main__":
    # Check environment
    if not setup_environment():
        sys.exit(1)
    
    # Run main analysis
    main() 