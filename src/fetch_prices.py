"""
Fetch and store price data for the universe
"""

import pandas as pd
import yfinance as yf
from typing import List, Optional
from pathlib import Path
from tqdm import tqdm
import time
from datetime import datetime

import utils


def fetch_single_ticker(ticker: str, 
                        start_date: str, 
                        end_date: str) -> Optional[pd.DataFrame]:
    """
    Fetch price data for a single ticker
    
    Args:
        ticker: Ticker symbol
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD
        
    Returns:
        DataFrame with OHLCV data or None if failed
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        
        if len(df) == 0:
            print(f"No data for {ticker}")
            return None
        
        # Clean column names
        df.columns = [col.lower() for col in df.columns]
        
        # Ensure we have the columns we need
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            print(f"Missing columns for {ticker}")
            return None
        
        # Add ticker column
        df['ticker'] = ticker
        
        # Reset index to make date a column
        df = df.reset_index()
        df = df.rename(columns={'index': 'date', 'Date': 'date'})
        
        # Select and order columns
        cols = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
        df = df[cols]
        
        return df
        
    except Exception as e:
        print(f"Error fetching {ticker}: {str(e)}")
        return None


def fetch_all_prices(tickers: List[str],
                     start_date: str,
                     end_date: str,
                     output_path: str,
                     batch_size: int = 50) -> pd.DataFrame:
    """
    Fetch prices for all tickers and save to parquet
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date
        end_date: End date
        output_path: Path to save parquet file
        batch_size: Number of tickers to fetch before saving
        
    Returns:
        DataFrame with all price data
    """
    all_data = []
    failed_tickers = []
    
    print(f"\nFetching prices for {len(tickers)} tickers...")
    print(f"Date range: {start_date} to {end_date}")
    
    # Fetch with progress bar
    for i, ticker in enumerate(tqdm(tickers)):
        df = fetch_single_ticker(ticker, start_date, end_date)
        
        if df is not None:
            all_data.append(df)
        else:
            failed_tickers.append(ticker)
        
        # Rate limiting - be nice to Yahoo Finance
        if (i + 1) % 10 == 0:
            time.sleep(1)
        
        # Save intermediate results
        if (i + 1) % batch_size == 0 and len(all_data) > 0:
            print(f"\nSaving batch at {i+1}/{len(tickers)}...")
            temp_df = pd.concat(all_data, ignore_index=True)
            temp_df.to_parquet(output_path.replace('.parquet', f'_temp_{i+1}.parquet'))
    
    # Combine all data
    if len(all_data) == 0:
        print("No data fetched!")
        return pd.DataFrame()
    
    print("\nCombining all data...")
    df = pd.concat(all_data, ignore_index=True)
    
    # Sort by date and ticker
    df = df.sort_values(['date', 'ticker']).reset_index(drop=True)
    
    # Save final result
    utils.ensure_dir(Path(output_path).parent)
    df.to_parquet(output_path)
    
    print(f"\nData saved to: {output_path}")
    print(f"Total rows: {len(df):,}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Unique tickers: {df['ticker'].nunique()}")
    
    if failed_tickers:
        print(f"\nFailed tickers ({len(failed_tickers)}): {failed_tickers[:10]}...")
    
    return df


def load_prices(path: str) -> pd.DataFrame:
    """Load price data from parquet"""
    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} rows from {path}")
    return df


def update_prices(existing_path: str, 
                  tickers: List[str],
                  output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Update existing price data with latest prices
    
    Args:
        existing_path: Path to existing price data
        tickers: List of tickers to update
        output_path: Path to save updated data (default: overwrite existing)
        
    Returns:
        Updated DataFrame
    """
    if output_path is None:
        output_path = existing_path
    
    # Load existing data
    existing_df = load_prices(existing_path)
    
    # Get the last date in existing data
    last_date = existing_df['date'].max()
    
    # Fetch new data from last_date + 1 day
    start_date = (pd.to_datetime(last_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"\nUpdating from {start_date} to {end_date}")
    
    if start_date >= end_date:
        print("Data is already up to date!")
        return existing_df
    
    # Fetch new data
    new_df = fetch_all_prices(tickers, start_date, end_date, 
                              output_path.replace('.parquet', '_update.parquet'))
    
    if len(new_df) == 0:
        print("No new data to add")
        return existing_df
    
    # Combine
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=['date', 'ticker'])
    combined_df = combined_df.sort_values(['date', 'ticker']).reset_index(drop=True)
    
    # Save
    combined_df.to_parquet(output_path)
    
    print(f"\nUpdated data saved to: {output_path}")
    print(f"Added {len(new_df):,} new rows")
    print(f"Total rows: {len(combined_df):,}")
    
    return combined_df


def get_market_data(tickers: List[str], 
                    start_date: str,
                    end_date: Optional[str] = None,
                    cache_path: Optional[str] = None) -> pd.DataFrame:
    """
    Get market data with optional caching
    
    Args:
        tickers: List of tickers
        start_date: Start date
        end_date: End date (None = today)
        cache_path: Path to cache file (None = no caching)
        
    Returns:
        DataFrame with price data
    """
    start_date, end_date = utils.get_date_range(start_date, end_date)
    
    # Check cache
    if cache_path and Path(cache_path).exists():
        print(f"Loading cached data from {cache_path}")
        df = load_prices(cache_path)
        
        # Check if we need to update
        last_date = df['date'].max().strftime("%Y-%m-%d")
        if last_date < end_date:
            print(f"Cache outdated (last date: {last_date}), updating...")
            df = update_prices(cache_path, tickers, cache_path)
        
        return df
    
    # Fetch fresh data
    print("No cache found, fetching fresh data...")
    df = fetch_all_prices(tickers, start_date, end_date, cache_path or "temp_prices.parquet")
    
    return df