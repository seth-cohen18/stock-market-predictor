"""
Define the tradable universe of stocks
"""

import pandas as pd
import yfinance as yf
from typing import List
from pathlib import Path
import json


# Top 200 liquid stocks (SPY top holdings + high volume names)
DEFAULT_UNIVERSE = [
    # Tech giants
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'ORCL', 'ADBE',
    'CRM', 'AMD', 'INTC', 'CSCO', 'ACN', 'QCOM', 'TXN', 'INTU', 'IBM', 'AMAT',
    
    # Finance
    'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'SCHW', 'AXP',
    'C', 'USB', 'PNC', 'TFC', 'BK', 'COF', 'DFS', 'SPGI', 'CME', 'ICE',
    
    # Healthcare
    'UNH', 'JNJ', 'LLY', 'ABBV', 'MRK', 'TMO', 'ABT', 'AMGN', 'DHR', 'PFE',
    'BMY', 'GILD', 'CVS', 'CI', 'REGN', 'VRTX', 'HUM', 'BSX', 'ISRG', 'MDT',
    
    # Consumer
    'WMT', 'HD', 'PG', 'KO', 'PEP', 'COST', 'MCD', 'NKE', 'SBUX', 'TGT',
    'LOW', 'TJX', 'DG', 'DLTR', 'CMG', 'YUM', 'ORLY', 'AZO', 'RCL', 'MAR',
    
    # Industrials
    'BA', 'UNP', 'HON', 'UPS', 'RTX', 'CAT', 'DE', 'LMT', 'GE', 'MMM',
    'GD', 'NOC', 'WM', 'EMR', 'ETN', 'ITW', 'CSX', 'NSC', 'FDX', 'PCAR',
    
    # Energy
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'HAL',
    
    # Telecom/Media
    'T', 'VZ', 'CMCSA', 'DIS', 'NFLX', 'TMUS', 'CHTR', 'WBD',
    
    # Retail/E-commerce
    'BABA', 'JD', 'MELI', 'SE', 'SHOP', 'EBAY', 'ETSY',
    
    # Real Estate
    'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'SPG', 'O', 'WELL',
    
    # Materials
    'LIN', 'APD', 'SHW', 'ECL', 'NEM', 'FCX', 'NUE', 'DOW',
    
    # Utilities
    'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL',
    
    # Semiconductors
    'TSM', 'ASML', 'MU', 'LRCX', 'KLAC', 'MCHP', 'MRVL', 'SNPS', 'CDNS',
    
    # Software/Cloud
    'NOW', 'SNOW', 'WDAY', 'PANW', 'CRWD', 'ZS', 'DDOG', 'NET', 'TEAM',
    
    # Biotech
    'BIIB', 'MRNA', 'ILMN', 'ALNY', 'SGEN', 'EXAS',
    
    # Industrial Tech
    'UBER', 'LYFT', 'ABNB', 'DASH', 'RIVN', 'LCID',
    
    # Financial Services
    'PYPL', 'SQ', 'COIN', 'SOFI', 'AFRM',
    
    # Consumer Discretionary
    'F', 'GM', 'TSLA', 'NIO', 'XPEV', 'LI',
    
    # ETFs (important for regime)
    'SPY', 'QQQ', 'IWM', 'DIA', 'EEM', 'EFA', 'TLT', 'GLD', 'SLV', 'USO',
    'XLF', 'XLE', 'XLV', 'XLK', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB', 'XLRE'
]


def get_universe(method: str = "fixed_list", size: int = 200) -> List[str]:
    """
    Get the universe of tradable stocks
    
    Args:
        method: Method for universe selection
            - "fixed_list": Use predefined list
            - "sp500": Get S&P 500 constituents
            - "top_volume": Get highest volume stocks
        size: Target universe size
        
    Returns:
        List of ticker symbols
    """
    if method == "fixed_list":
        # Use our curated list
        universe = DEFAULT_UNIVERSE[:size]
        
    elif method == "sp500":
        # Get S&P 500 constituents
        try:
            # Read from Wikipedia
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tables = pd.read_html(url)
            sp500_table = tables[0]
            universe = sp500_table['Symbol'].str.replace('.', '-').tolist()
            universe = universe[:size]
        except Exception as e:
            print(f"Error fetching S&P 500 list: {e}")
            print("Falling back to default universe")
            universe = DEFAULT_UNIVERSE[:size]
            
    elif method == "top_volume":
        # This would require a separate data source
        # For now, fall back to default
        print("top_volume method not yet implemented, using fixed_list")
        universe = DEFAULT_UNIVERSE[:size]
        
    else:
        raise ValueError(f"Unknown universe method: {method}")
    
    # Remove duplicates and sort
    universe = sorted(list(set(universe)))
    
    print(f"Universe size: {len(universe)} tickers")
    return universe


def filter_by_liquidity(tickers: List[str], 
                        min_volume: float = 1_000_000,
                        min_price: float = 5.0,
                        lookback_days: int = 60) -> List[str]:
    """
    Filter tickers by liquidity criteria
    
    Args:
        tickers: List of ticker symbols
        min_volume: Minimum average daily volume
        min_price: Minimum price
        lookback_days: Days to look back for liquidity check
        
    Returns:
        Filtered list of tickers
    """
    filtered = []
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=f"{lookback_days}d")
            
            if len(hist) < lookback_days * 0.8:  # Need at least 80% of data
                continue
                
            avg_volume = hist['Volume'].mean()
            avg_price = hist['Close'].mean()
            
            if avg_volume >= min_volume and avg_price >= min_price:
                filtered.append(ticker)
                
        except Exception as e:
            print(f"Error checking {ticker}: {e}")
            continue
    
    return filtered


def save_universe(tickers: List[str], filepath: str = "data/universe.json"):
    """Save universe to file"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump({
            'tickers': tickers,
            'date': pd.Timestamp.now().strftime("%Y-%m-%d"),
            'count': len(tickers)
        }, f, indent=2)
    print(f"Saved universe ({len(tickers)} tickers) to {filepath}")


def load_universe(filepath: str = "data/universe.json") -> List[str]:
    """Load universe from file"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        print(f"Loaded universe ({data['count']} tickers) from {filepath}")
        print(f"Universe date: {data['date']}")
        return data['tickers']
    except FileNotFoundError:
        print(f"Universe file not found: {filepath}")
        return []


if __name__ == "__main__":
    # Test the universe
    print("Getting default universe...")
    universe = get_universe(method="fixed_list", size=200)
    
    print(f"\nFirst 20 tickers:")
    print(universe[:20])
    
    # Save it
    save_universe(universe)
    
    print("\nUniverse created successfully!")