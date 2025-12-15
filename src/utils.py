"""
Utility functions for the market AI system
"""

import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def ensure_dir(path: str) -> Path:
    """Create directory if it doesn't exist"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_date_range(start_date: str, end_date: Optional[str] = None) -> tuple:
    """
    Get date range for data fetching
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format, None = today
        
    Returns:
        tuple of (start_date, end_date) as strings
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    return start_date, end_date


def calculate_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    """
    Calculate log returns
    
    Args:
        prices: Price series
        periods: Number of periods for return calculation
        
    Returns:
        Log returns
    """
    return np.log(prices / prices.shift(periods))


def calculate_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate rolling volatility (annualized)
    
    Args:
        returns: Return series
        window: Rolling window size
        
    Returns:
        Annualized volatility
    """
    return returns.rolling(window).std() * np.sqrt(252)


def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """
    Winsorize series to clip extreme values
    
    Args:
        series: Input series
        lower: Lower percentile
        upper: Upper percentile
        
    Returns:
        Winsorized series
    """
    lower_bound = series.quantile(lower)
    upper_bound = series.quantile(upper)
    return series.clip(lower_bound, upper_bound)


def normalize_features(df: pd.DataFrame, exclude_cols: list = None) -> pd.DataFrame:
    """
    Z-score normalize features by date (cross-sectional)
    
    Args:
        df: DataFrame with features
        exclude_cols: Columns to exclude from normalization
        
    Returns:
        Normalized DataFrame
    """
    if exclude_cols is None:
        exclude_cols = ['date', 'ticker']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    normalized = df.copy()
    for col in feature_cols:
        grouped = df.groupby('date')[col]
        normalized[col] = (df[col] - grouped.transform('mean')) / grouped.transform('std')
        
    # Fill any remaining NaNs with 0
    normalized[feature_cols] = normalized[feature_cols].fillna(0)
    
    return normalized


def calculate_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate Sharpe ratio
    
    Args:
        returns: Return series
        periods_per_year: Number of periods in a year (252 for daily)
        
    Returns:
        Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0
    mean_return = returns.mean()
    std_return = returns.std()
    if std_return == 0:
        return 0.0
    return (mean_return / std_return) * np.sqrt(periods_per_year)


def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    Calculate maximum drawdown
    
    Args:
        returns: Return series
        
    Returns:
        Maximum drawdown as a decimal
    """
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    return drawdown.min()


def print_backtest_metrics(returns: pd.Series, name: str = "Strategy"):
    """Print backtest performance metrics"""
    total_return = (1 + returns).prod() - 1
    sharpe = calculate_sharpe(returns)
    max_dd = calculate_max_drawdown(returns)
    win_rate = (returns > 0).mean()
    
    print(f"\n{'='*50}")
    print(f"{name} Performance Metrics")
    print(f"{'='*50}")
    print(f"Total Return:    {total_return:>10.2%}")
    print(f"Sharpe Ratio:    {sharpe:>10.2f}")
    print(f"Max Drawdown:    {max_dd:>10.2%}")
    print(f"Win Rate:        {win_rate:>10.2%}")
    print(f"Num Trades:      {len(returns):>10}")
    print(f"{'='*50}\n")


def get_trading_days(start: str, end: str) -> pd.DatetimeIndex:
    """
    Get trading days between start and end dates
    
    Args:
        start: Start date
        end: End date
        
    Returns:
        DatetimeIndex of trading days
    """
    # Create a date range and filter to weekdays
    all_days = pd.date_range(start, end, freq='D')
    trading_days = all_days[all_days.weekday < 5]  # Monday=0, Friday=4
    return trading_days


def format_currency(amount: float) -> str:
    """Format number as currency"""
    return f"${amount:,.2f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format decimal as percentage"""
    return f"{value * 100:.{decimals}f}%"