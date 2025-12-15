"""
Stock Market Prediction System
Phase 1: Stocks & ETFs with ML-based recommendations
"""

__version__ = "1.0.0"
__author__ = "Market AI"
__phase__ = "1"

from . import utils
from . import universe
from . import fetch_prices
from . import build_features
from . import train
from . import backtest
from . import recommend

__all__ = [
    'utils',
    'universe',
    'fetch_prices',
    'build_features',
    'train',
    'backtest',
    'recommend'
]