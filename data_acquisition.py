"""
PEAD Strategy Data Acquisition Module
Fetches historical price data and earnings calendar from free sources
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
import os
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class DataAcquisition:
    def __init__(self, data_dir: str = "data"):
        """Initialize data acquisition with storage directory."""
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def get_sp500_tickers(self) -> List[str]:
        """Fetch current S&P 500 constituents from Wikipedia."""
        try:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tables = pd.read_html(url)
            sp500_table = tables[0]
            tickers = sp500_table['Symbol'].tolist()
            # Clean up ticker symbols (remove dots, etc.)
            tickers = [ticker.replace('.', '-') for ticker in tickers]
            print(f"Successfully fetched {len(tickers)} S&P 500 tickers")
            return tickers
        except Exception as e:
            print(f"Error fetching S&P 500 tickers: {e}")
            # Fallback to a subset of major tickers
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V']
    
    def fetch_price_data(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch historical price data for a single ticker."""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date)
            
            if data.empty:
                print(f"No data found for {ticker}")
                return None
                
            # Reset index to make date a column
            data = data.reset_index()
            data['ticker'] = ticker
            
            # Rename columns to lowercase for consistency
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            
            # Ensure we have required columns
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                print(f"Missing columns for {ticker}: {missing_cols}")
                return None
                
            return data[['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return None
    
    def fetch_earnings_calendar(self, ticker: str) -> Optional[pd.DataFrame]:
        """Fetch earnings calendar for a ticker using yfinance."""
        try:
            stock = yf.Ticker(ticker)
            calendar = stock.calendar
            
            if calendar is None or calendar.empty:
                return None
                
            # Convert to DataFrame if it's not already
            if hasattr(calendar, 'T'):
                calendar = calendar.T
                
            calendar = calendar.reset_index()
            calendar['ticker'] = ticker
            
            # Standardize column names
            calendar.columns = [col.lower().replace(' ', '_') for col in calendar.columns]
            
            return calendar
            
        except Exception as e:
            print(f"Error fetching earnings calendar for {ticker}: {e}")
            return None
    
    def batch_fetch_price_data(self, tickers: List[str], start_date: str, end_date: str, 
                             batch_size: int = 10, delay: float = 1.0) -> pd.DataFrame:
        """Fetch price data for multiple tickers with rate limiting."""
        all_data = []
        total_tickers = len(tickers)
        
        print(f"Fetching price data for {total_tickers} tickers...")
        
        for i in range(0, total_tickers, batch_size):
            batch = tickers[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(total_tickers-1)//batch_size + 1}: {batch}")
            
            for ticker in batch:
                data = self.fetch_price_data(ticker, start_date, end_date)
                if data is not None:
                    all_data.append(data)
                time.sleep(delay)  # Rate limiting
            
            # Longer delay between batches
            if i + batch_size < total_tickers:
                time.sleep(delay * 2)
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            print(f"Successfully fetched data for {len(combined_data['ticker'].unique())} tickers")
            return combined_data
        else:
            print("No data was successfully fetched")
            return pd.DataFrame()
    
    def save_data(self, data: pd.DataFrame, filename: str):
        """Save data to CSV file."""
        filepath = os.path.join(self.data_dir, filename)
        data.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
    
    def load_data(self, filename: str) -> pd.DataFrame:
        """Load data from CSV file."""
        filepath = os.path.join(self.data_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'], utc=True)
            return df
        else:
            print(f"File {filepath} not found")
            return pd.DataFrame()
    
    def fetch_full_dataset(self, start_date: str = "2020-01-01", 
                          end_date: Optional[str] = None, 
                          tickers: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """Fetch complete dataset for PEAD analysis."""
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        if tickers is None:
            tickers = self.get_sp500_tickers()
        
        print(f"Fetching data from {start_date} to {end_date}")
        
        # Fetch price data
        price_data = self.batch_fetch_price_data(tickers, start_date, end_date)
        
        if not price_data.empty:
            self.save_data(price_data, f"sp500_prices_{start_date}_to_{end_date}.csv")
        
        # Note: For earnings calendar, we'll need to implement a more sophisticated approach
        # as yfinance earnings calendar is limited. This is a placeholder for now.
        print("Note: Earnings calendar fetching requires additional implementation")
        print("Consider using SEC EDGAR API or financial news APIs for comprehensive earnings data")
        
        return {
            'prices': price_data,
            'earnings': pd.DataFrame()  # Placeholder
        }

# Example usage and data quality checks
def validate_data_quality(price_data: pd.DataFrame) -> Dict[str, any]:
    """Perform data quality checks on price data."""
    checks = {}
    
    if price_data.empty:
        return {"status": "FAILED", "reason": "No data provided"}
    
    # Check for required columns
    required_cols = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in price_data.columns]
    
    checks['missing_columns'] = missing_cols
    checks['total_records'] = len(price_data)
    checks['unique_tickers'] = price_data['ticker'].nunique()
    checks['date_range'] = {
        'start': price_data['date'].min(),
        'end': price_data['date'].max()
    }
    
    # Check for missing values
    checks['missing_values'] = price_data.isnull().sum().to_dict()
    
    # Check for invalid prices (negative or zero)
    price_cols = ['open', 'high', 'low', 'close']
    invalid_prices = {}
    for col in price_cols:
        if col in price_data.columns:
            invalid_count = (price_data[col] <= 0).sum()
            invalid_prices[col] = invalid_count
    
    checks['invalid_prices'] = invalid_prices
    
    # Check for logical price relationships (high >= low, etc.)
    if all(col in price_data.columns for col in ['high', 'low', 'open', 'close']):
        checks['illogical_prices'] = {
            'high_less_than_low': (price_data['high'] < price_data['low']).sum(),
            'close_above_high': (price_data['close'] > price_data['high']).sum(),
            'close_below_low': (price_data['close'] < price_data['low']).sum()
        }
    
    # Overall status
    has_errors = (
        bool(missing_cols) or
        any(count > 0 for count in invalid_prices.values()) or
        (checks.get('illogical_prices') and any(count > 0 for count in checks['illogical_prices'].values()))
    )
    
    checks['status'] = "FAILED" if has_errors else "PASSED"
    
    return checks

if __name__ == "__main__":
    # Example usage
    data_fetcher = DataAcquisition()
    
    # Fetch data for a small sample first
    sample_tickers = ['AAPL', 'MSFT', 'GOOGL']
    dataset = data_fetcher.fetch_full_dataset(
        start_date="2023-01-01",
        end_date="2024-01-01",
        tickers=sample_tickers
    )
    
    # Validate data quality
    if not dataset['prices'].empty:
        quality_report = validate_data_quality(dataset['prices'])
        print("\nData Quality Report:")
        for key, value in quality_report.items():
            print(f"{key}: {value}") 