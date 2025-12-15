"""
Sector Rotation Analyzer - Phase 3
Identifies which sectors are outperforming/underperforming
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime, timedelta


# Sector ETFs for tracking
SECTOR_ETFS = {
    'Technology': 'XLK',
    'Healthcare': 'XLV',
    'Financials': 'XLF',
    'Energy': 'XLE',
    'Industrials': 'XLI',
    'Consumer Discretionary': 'XLY',
    'Consumer Staples': 'XLP',
    'Utilities': 'XLU',
    'Real Estate': 'XLRE',
    'Materials': 'XLB',
    'Communication Services': 'XLC'
}

# Stock to sector mapping (expanded)
STOCK_SECTORS = {
    # Technology
    'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'GOOG': 'Technology',
    'NVDA': 'Technology', 'AMD': 'Technology', 'INTC': 'Technology', 'CSCO': 'Technology',
    'ORCL': 'Technology', 'CRM': 'Technology', 'ADBE': 'Technology', 'AVGO': 'Technology',
    'QCOM': 'Technology', 'TXN': 'Technology', 'AMAT': 'Technology', 'LRCX': 'Technology',
    'KLAC': 'Technology', 'SNPS': 'Technology', 'CDNS': 'Technology', 'NXPI': 'Technology',
    
    # Healthcare
    'JNJ': 'Healthcare', 'UNH': 'Healthcare', 'PFE': 'Healthcare', 'ABBV': 'Healthcare',
    'TMO': 'Healthcare', 'LLY': 'Healthcare', 'MRK': 'Healthcare', 'ABT': 'Healthcare',
    'DHR': 'Healthcare', 'BMY': 'Healthcare', 'AMGN': 'Healthcare', 'GILD': 'Healthcare',
    'CVS': 'Healthcare', 'CI': 'Healthcare', 'HUM': 'Healthcare', 'ISRG': 'Healthcare',
    
    # Financials
    'JPM': 'Financials', 'BAC': 'Financials', 'WFC': 'Financials', 'GS': 'Financials',
    'MS': 'Financials', 'BLK': 'Financials', 'C': 'Financials', 'SCHW': 'Financials',
    'AXP': 'Financials', 'USB': 'Financials', 'PNC': 'Financials', 'TFC': 'Financials',
    'V': 'Financials', 'MA': 'Financials', 'PYPL': 'Financials', 'SQ': 'Financials',
    
    # Energy
    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'SLB': 'Energy',
    'EOG': 'Energy', 'PSX': 'Energy', 'MPC': 'Energy', 'VLO': 'Energy',
    'OXY': 'Energy', 'HAL': 'Energy', 'KMI': 'Energy', 'WMB': 'Energy',
    
    # Consumer Discretionary
    'AMZN': 'Consumer Discretionary', 'TSLA': 'Consumer Discretionary', 
    'HD': 'Consumer Discretionary', 'NKE': 'Consumer Discretionary',
    'LOW': 'Consumer Discretionary', 'SBUX': 'Consumer Discretionary',
    'TGT': 'Consumer Discretionary', 'TJX': 'Consumer Discretionary',
    'MCD': 'Consumer Discretionary', 'BKNG': 'Consumer Discretionary',
    'F': 'Consumer Discretionary', 'GM': 'Consumer Discretionary',
    
    # Consumer Staples
    'WMT': 'Consumer Staples', 'PG': 'Consumer Staples', 'KO': 'Consumer Staples',
    'PEP': 'Consumer Staples', 'COST': 'Consumer Staples', 'MDLZ': 'Consumer Staples',
    'CL': 'Consumer Staples', 'KHC': 'Consumer Staples', 'GIS': 'Consumer Staples',
    
    # Industrials
    'BA': 'Industrials', 'CAT': 'Industrials', 'GE': 'Industrials',
    'UPS': 'Industrials', 'MMM': 'Industrials', 'HON': 'Industrials',
    'LMT': 'Industrials', 'RTX': 'Industrials', 'DE': 'Industrials',
    
    # Communication Services
    'META': 'Communication Services', 'DIS': 'Communication Services',
    'NFLX': 'Communication Services', 'CMCSA': 'Communication Services',
    'T': 'Communication Services', 'VZ': 'Communication Services',
    'TMUS': 'Communication Services', 'CHTR': 'Communication Services',
    
    # Utilities
    'NEE': 'Utilities', 'DUK': 'Utilities', 'SO': 'Utilities',
    'D': 'Utilities', 'AEP': 'Utilities', 'EXC': 'Utilities',
    
    # Real Estate
    'AMT': 'Real Estate', 'PLD': 'Real Estate', 'CCI': 'Real Estate',
    'EQIX': 'Real Estate', 'PSA': 'Real Estate', 'SPG': 'Real Estate',
    
    # Materials
    'LIN': 'Materials', 'APD': 'Materials', 'SHW': 'Materials',
    'FCX': 'Materials', 'NEM': 'Materials', 'DOW': 'Materials'
}


def add_sector_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sector rotation features
    
    Args:
        df: Features dataframe with ticker and date columns
        
    Returns:
        DataFrame with sector features
    """
    print("  - Sector rotation features...")
    
    # Map tickers to sectors
    df['sector'] = df['ticker'].map(STOCK_SECTORS)
    df['sector'] = df['sector'].fillna('Other')
    
    # Calculate sector returns
    sector_returns = df.groupby(['date', 'sector'])['return_1d'].transform('mean')
    df['sector_return_1d'] = sector_returns
    
    sector_returns_5d = df.groupby(['date', 'sector'])['return_5d'].transform('mean')
    df['sector_return_5d'] = sector_returns_5d
    
    sector_returns_20d = df.groupby(['date', 'sector'])['return_20d'].transform('mean')
    df['sector_return_20d'] = sector_returns_20d
    
    # Relative strength vs sector
    df['vs_sector_1d'] = df['return_1d'] - df['sector_return_1d']
    df['vs_sector_5d'] = df['return_5d'] - df['sector_return_5d']
    df['vs_sector_20d'] = df['return_20d'] - df['sector_return_20d']
    
    # Sector momentum (is sector trending?)
    df['sector_momentum'] = df.groupby('sector')['sector_return_20d'].transform(
        lambda x: x.rolling(20, min_periods=1).mean()
    )
    
    # Sector rank (which sectors are hottest?)
    df['sector_rank'] = df.groupby('date')['sector_return_20d'].rank(pct=True)
    
    # Stock rank within sector
    df['within_sector_rank'] = df.groupby(['date', 'sector'])['return_20d'].rank(pct=True)
    
    return df


def get_sector_rotation_signal(df: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """
    Calculate sector rotation strength
    
    Args:
        df: DataFrame with sector features
        lookback: Days to look back for rotation
        
    Returns:
        DataFrame with rotation signals
    """
    # Get latest data for each ticker
    latest = df.groupby('ticker').last().reset_index()
    
    # Sector performance over lookback
    for sector in STOCK_SECTORS.values():
        sector_stocks = latest[latest['sector'] == sector]
        if len(sector_stocks) > 0:
            print(f"  {sector}: {sector_stocks['sector_return_20d'].mean():.2%} return")
    
    return latest


if __name__ == "__main__":
    # Test sector features
    import sys
    sys.path.insert(0, '.')
    
    from build_features import load_features
    
    df = load_features('data/processed/features.parquet')
    df = add_sector_features(df)
    
    print("\nSector Features:")
    print(df[['ticker', 'sector', 'sector_return_20d', 'vs_sector_20d', 'sector_rank']].head(10))
    
    print("\nSector Rotation Signal:")
    rotation = get_sector_rotation_signal(df)
    print(rotation.groupby('sector')['sector_return_20d'].mean().sort_values(ascending=False))