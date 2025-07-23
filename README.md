# Post-Earnings Announcement Drift (PEAD) Strategy

A comprehensive implementation of the Post-Earnings Announcement Drift strategy using free data sources and Python. This project addresses the market anomaly where stock prices continue to move in the direction of earnings surprises for several days after earnings announcements.

## 🎯 Strategy Overview

The PEAD strategy exploits the tendency of stock prices to "drift" in the direction of earnings surprises for several trading days following earnings announcements. This occurs due to:

- **Delayed Information Processing**: Not all market participants immediately process earnings information
- **Institutional Trading Patterns**: Large institutions may spread their trades over multiple days
- **Analyst Revisions**: Post-earnings analyst recommendation changes can drive continued price movement
- **Momentum Effects**: Initial price moves can attract additional momentum-based trading

## 📁 Project Structure

```
pead-strategy/
├── data_acquisition.py    # Free data source integration (Yahoo Finance, SEC filings)
├── pead_strategy.py      # Core PEAD strategy implementation and backtesting
├── main.py              # Main execution script
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── data/               # Data storage directory (created automatically)
└── results/            # Analysis results and reports (created automatically)
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone or download the project files
cd pead-strategy

# Install required packages
pip install -r requirements.txt

# Run the complete analysis
python main.py
```

### 2. What the Analysis Does

The script automatically:
- ✅ Fetches historical price data for S&P 500 stocks
- ✅ Identifies potential earnings events using price gap analysis
- ✅ Calculates post-earnings price drift patterns
- ✅ Backtests the strategy with different parameters
- ✅ Generates performance reports and visualizations
- ✅ Saves results for further analysis

## 📊 Key Features

### Data Acquisition
- **Free Data Sources**: Uses Yahoo Finance API via `yfinance` library
- **Robust Error Handling**: Graceful handling of API limits and data gaps
- **Data Quality Validation**: Comprehensive checks for missing values and logical inconsistencies
- **Batch Processing**: Efficient fetching with rate limiting

### Strategy Implementation
- **Earnings Detection**: Uses price gaps as proxy for earnings surprises
- **Multiple Timeframes**: Tests various holding periods (3, 5, 10 days)
- **Risk Management**: Configurable surprise thresholds to filter trades
- **Performance Metrics**: Win rate, Sharpe ratio, total returns, monthly breakdown

### Analysis & Visualization
- **Drift Pattern Analysis**: Shows how prices move after earnings
- **Parameter Optimization**: Tests multiple strategy configurations
- **Statistical Validation**: Significance testing and confidence intervals
- **Interactive Plots**: Clear visualizations of strategy performance

## 🔧 Configuration Options

### In `main.py`, you can modify:

```python
# Analysis period
START_DATE = "2023-01-01"
END_DATE = "2024-12-31"

# Stock universe (or use get_sp500_tickers() for full S&P 500)
SAMPLE_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

# Strategy parameters to test
holding_periods = [3, 5, 10]        # Days to hold position
surprise_thresholds = [0.02, 0.03, 0.05]  # Minimum surprise to trade
```

## 📈 Expected Output

### Console Output
```
POST-EARNINGS ANNOUNCEMENT DRIFT (PEAD) STRATEGY
================================================================
Configuration:
- Analysis Period: 2023-01-01 to 2024-12-31
- Sample Tickers: 10 stocks
- Data Directory: data

Fetching price data for 10 tickers...
Successfully fetched data for 10 tickers
Data Quality Status: PASSED

Testing parameter combinations...
Testing: 3 days holding, 2.0% surprise threshold
Testing: 5 days holding, 3.0% surprise threshold
...

BEST PARAMETER COMBINATION FOUND
================================================================
Holding Period: 5 days
Surprise Threshold: 3.0%
Sharpe Ratio: 1.247

PEAD STRATEGY BACKTEST REPORT
==================================================
STRATEGY PARAMETERS:
- Holding Period: 5 days
- Minimum Surprise Threshold: 3.00%

PERFORMANCE SUMMARY:
- Total Trades: 47
- Win Rate: 63.83%
- Average Return per Trade: 1.24%
- Total Strategy Return: 58.28%
- Sharpe Ratio: 1.247
```

### Generated Files
- `data/sp500_prices_YYYY-MM-DD_to_YYYY-MM-DD.csv` - Raw price data
- `data/pead_trades_YYYYMMDD_HHMMSS.csv` - Individual trade details
- `data/pead_report_YYYYMMDD_HHMMSS.txt` - Complete analysis report
- Interactive plots showing drift patterns and performance metrics

## 📊 Understanding the Results

### Key Metrics Explained

- **Win Rate**: Percentage of trades that were profitable
- **Average Return per Trade**: Mean return across all trades
- **Sharpe Ratio**: Risk-adjusted return (higher is better)
- **Total Strategy Return**: Cumulative return from all trades

### Interpreting Plots

1. **Price Drift by Surprise Direction**: Shows if positive/negative surprises lead to continued drift
2. **Distribution of Surprises**: Helps understand the frequency of different surprise magnitudes
3. **Return by Holding Period**: Optimal time to hold positions
4. **Win Rate by Surprise Magnitude**: Whether larger surprises are more predictive

## ⚠️ Important Limitations

### Current Implementation
- **Earnings Proxy**: Uses price gaps instead of actual earnings data
- **No Transaction Costs**: Backtest doesn't include fees, slippage, or market impact
- **Survivorship Bias**: Only includes currently listed S&P 500 companies
- **Limited History**: Free data sources may have limited historical depth

### For Production Use
- Integrate actual earnings estimates and results
- Add transaction cost modeling
- Implement proper risk management (position sizing, stop losses)
- Consider market regime changes and economic cycles
- Add real-time monitoring and execution capabilities

## 🔄 Upgrading to Premium Data

For more robust analysis, consider integrating:

### Commercial Data Providers
```python
# Example integration points in data_acquisition.py
def fetch_earnings_from_refinitiv(api_key, ticker):
    # Refinitiv Eikon API integration
    pass

def fetch_estimates_from_bloomberg(api_key, ticker):
    # Bloomberg API integration
    pass
```

### Alternative Free Sources
- **SEC EDGAR API**: For official earnings filings
- **Financial news APIs**: For earnings announcement timing
- **Analyst estimate aggregators**: For surprise calculations

## 🤝 Contributing

Potential enhancements:
1. **Real earnings data integration**
2. **Sector-specific analysis**
3. **Machine learning surprise prediction**
4. **Real-time trading implementation**
5. **Risk management modules**

## 📚 Academic Background

The PEAD anomaly was first documented in academic literature:
- Ball & Brown (1968): Initial evidence of post-earnings drift
- Bernard & Thomas (1989): Comprehensive analysis of the anomaly
- Chordia & Shivakumar (2006): Modern persistence of PEAD

## ⚖️ Disclaimer

This implementation is for educational and research purposes only. Past performance does not guarantee future results. Always:
- Conduct thorough testing before live trading
- Consider transaction costs and market impact
- Implement appropriate risk management
- Consult with financial professionals

---

**Ready to explore the PEAD anomaly? Run `python main.py` and start your analysis!** 