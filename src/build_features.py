"""
Feature engineering for market prediction
UPDATED: Phase 3 - Maximum Accuracy Mode with comprehensive features
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from pathlib import Path
from tqdm import tqdm

import utils
from sector_rotation import add_sector_features
from news_sentiment import NewsSentimentAnalyzer


def add_return_features(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """Add return features for different windows"""
    for window in windows:
        df[f'return_{window}d'] = df.groupby('ticker')['close'].transform(
            lambda x: np.log(x / x.shift(window))
        )
    return df


def add_ma_features(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """Add moving average features"""
    for window in windows:
        df[f'ma_{window}'] = df.groupby('ticker')['close'].transform(
            lambda x: x.rolling(window).mean()
        )
        df[f'price_to_ma_{window}'] = df['close'] / (df[f'ma_{window}'] + 1e-10)
    return df


def add_volatility_features(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """Add volatility features"""
    for window in windows:
        df[f'volatility_{window}d'] = df.groupby('ticker')['return_1d'].transform(
            lambda x: x.rolling(window).std()
        )
    return df


def add_volume_features(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """Add volume-based features"""
    for window in windows:
        df[f'volume_ma_{window}'] = df.groupby('ticker')['volume'].transform(
            lambda x: x.rolling(window).mean()
        )
        df[f'volume_ratio_{window}'] = df['volume'] / (df[f'volume_ma_{window}'] + 1e-10)
    return df


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add momentum indicators"""
    for window in [5, 10, 20]:
        df[f'momentum_{window}'] = df.groupby('ticker')['close'].transform(
            lambda x: (x - x.shift(window)) / (x.shift(window) + 1e-10)
        )
    
    df['range_5d'] = df.groupby('ticker')['high'].transform(lambda x: x.rolling(5).max()) - \
                     df.groupby('ticker')['low'].transform(lambda x: x.rolling(5).min())
    df['range_20d'] = df.groupby('ticker')['high'].transform(lambda x: x.rolling(20).max()) - \
                      df.groupby('ticker')['low'].transform(lambda x: x.rolling(20).min())
    
    return df


def add_market_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add market regime features based on SPY"""
    print("  - Market regime features...")
    
    spy_data = df[df['ticker'] == 'SPY'][['date', 'close', 'volatility_20d']].copy()
    spy_data = spy_data.rename(columns={
        'close': 'spy_close',
        'volatility_20d': 'spy_volatility'
    })
    
    spy_data['spy_return_1d'] = np.log(spy_data['spy_close'] / spy_data['spy_close'].shift(1))
    spy_data['spy_return_5d'] = np.log(spy_data['spy_close'] / spy_data['spy_close'].shift(5))
    spy_data['spy_return_20d'] = np.log(spy_data['spy_close'] / spy_data['spy_close'].shift(20))
    
    # Market trend
    spy_data['spy_ma_50'] = spy_data['spy_close'].rolling(50).mean()
    spy_data['spy_ma_200'] = spy_data['spy_close'].rolling(200).mean()
    spy_data['spy_trend'] = (spy_data['spy_ma_50'] / (spy_data['spy_ma_200'] + 1e-10)) - 1
    
    df = df.merge(
        spy_data[['date', 'spy_return_1d', 'spy_return_5d', 'spy_return_20d', 
                  'spy_volatility', 'spy_trend']],
        on='date',
        how='left'
    )
    
    df['spy_volatility'] = df['spy_volatility'].fillna(df['spy_volatility'].mean())
    df['spy_trend'] = df['spy_trend'].fillna(0)
    
    # Beta to market
    df['beta'] = df.groupby('ticker').apply(
        lambda x: x['return_1d'].rolling(60).corr(x['spy_return_1d'])
    ).reset_index(level=0, drop=True)
    
    return df


def add_advanced_market_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    PHASE 3: Advanced market regime detection
    Identifies bull, bear, and sideways markets
    """
    print("  - Advanced market regime detection...")
    
    spy_full = df[df['ticker'] == 'SPY'][['date', 'close']].copy()
    
    if len(spy_full) == 0:
        df['regime_bull'] = 0
        df['regime_bear'] = 0
        df['regime_sideways'] = 1
        return df
    
    spy_full['spy_ma_50'] = spy_full['close'].rolling(50).mean()
    spy_full['spy_ma_200'] = spy_full['close'].rolling(200).mean()
    
    # Bull: Price > MA50 > MA200
    # Bear: Price < MA50 < MA200
    # Sideways: Mixed
    
    spy_full['regime_bull'] = (
        (spy_full['close'] > spy_full['spy_ma_50']) & 
        (spy_full['spy_ma_50'] > spy_full['spy_ma_200'])
    ).astype(int)
    
    spy_full['regime_bear'] = (
        (spy_full['close'] < spy_full['spy_ma_50']) & 
        (spy_full['spy_ma_50'] < spy_full['spy_ma_200'])
    ).astype(int)
    
    df = df.merge(
        spy_full[['date', 'regime_bull', 'regime_bear']],
        on='date',
        how='left'
    )
    
    df['regime_bull'] = df['regime_bull'].fillna(0)
    df['regime_bear'] = df['regime_bear'].fillna(0)
    df['regime_sideways'] = ((df['regime_bull'] == 0) & (df['regime_bear'] == 0)).astype(int)
    
    return df


def add_liquidity_features(df: pd.DataFrame) -> pd.DataFrame:
    """PHASE 3: Add liquidity and trading features"""
    print("  - Liquidity features...")
    
    # Dollar volume
    df['dollar_volume'] = df['close'] * df['volume']
    df['dollar_volume_ma_20'] = df.groupby('ticker')['dollar_volume'].transform(
        lambda x: x.rolling(20).mean()
    )
    
    # Spread proxy (high-low range as % of close)
    df['spread_proxy'] = (df['high'] - df['low']) / (df['close'] + 1e-10)
    df['spread_ma_20'] = df.groupby('ticker')['spread_proxy'].transform(
        lambda x: x.rolling(20).mean()
    )
    
    # Volume volatility
    df['volume_volatility'] = df.groupby('ticker')['volume'].transform(
        lambda x: x.rolling(20).std() / (x.rolling(20).mean() + 1e-10)
    )
    
    return df


def add_mean_reversion_features(df: pd.DataFrame) -> pd.DataFrame:
    """PHASE 3: Mean reversion indicators"""
    print("  - Mean reversion features...")
    
    # Z-score of price vs MA
    for window in [20, 60]:
        ma = df.groupby('ticker')['close'].transform(lambda x: x.rolling(window).mean())
        std = df.groupby('ticker')['close'].transform(lambda x: x.rolling(window).std())
        df[f'zscore_{window}'] = (df['close'] - ma) / (std + 1e-10)
    
    return df


def add_momentum_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """PHASE 3: Quality of momentum"""
    print("  - Momentum quality features...")
    
    # Consecutive up/down days
    df['price_change_sign'] = np.sign(df.groupby('ticker')['close'].transform(lambda x: x.diff()))
    
    # Streak of positive/negative days
    df['momentum_streak'] = df.groupby('ticker')['price_change_sign'].transform(
        lambda x: x.groupby((x != x.shift()).cumsum()).cumcount() + 1
    )
    
    # Smooth vs choppy
    df['returns_smoothness'] = df.groupby('ticker')['return_1d'].transform(
        lambda x: 1 / (x.rolling(20).std() + 1e-10)
    )
    
    return df


def add_sentiment_features(df: pd.DataFrame, use_news: bool = False, 
                           api_keys: Dict = None) -> pd.DataFrame:
    """
    PHASE 3: Add news sentiment features
    
    Args:
        df: Features dataframe
        use_news: Whether to fetch news sentiment
        api_keys: API keys dict
    """
    if not use_news:
        print("  - News sentiment: Skipped")
        df['sentiment_score'] = 0.0
        df['sentiment_magnitude'] = 0.0
        return df
    
    print("  - News sentiment analysis...")
    print("    (This may take a few minutes...)")
    
    try:
        analyzer = NewsSentimentAnalyzer(api_keys=api_keys)
        
        # Get unique tickers
        latest_date = df['date'].max()
        latest_tickers = df[df['date'] == latest_date]['ticker'].unique().tolist()
        
        # Get sentiment
        sentiment_df = analyzer.get_market_sentiment(latest_tickers)
        
        # Merge
        df = df.merge(
            sentiment_df[['ticker', 'sentiment_score', 'sentiment_magnitude']],
            on='ticker',
            how='left'
        )
        
        df['sentiment_score'] = df['sentiment_score'].fillna(0)
        df['sentiment_magnitude'] = df['sentiment_magnitude'].fillna(0)
        
        print("    ✓ Sentiment features added")
    except Exception as e:
        print(f"    ⚠️  Sentiment failed: {e}")
        print("    Continuing without sentiment...")
        df['sentiment_score'] = 0.0
        df['sentiment_magnitude'] = 0.0
    
    return df


def create_labels(df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    """Create forward-looking labels for different horizons"""
    for horizon in horizons:
        df[f'target_{horizon}d'] = df.groupby('ticker')['close'].transform(
            lambda x: np.log(x.shift(-horizon) / x)
        )
    return df


def build_features(df: pd.DataFrame, config: Dict, 
                  use_news: bool = False, api_keys: Dict = None) -> pd.DataFrame:
    """
    Build all features for model training
    UPDATED: Phase 3 - Maximum accuracy mode
    
    Args:
        df: Price dataframe
        config: Config dict
        use_news: Whether to fetch news sentiment (Phase 3)
        api_keys: API keys for news (Phase 3)
    """
    print("\n🚀 Building Features (Phase 3 - Maximum Accuracy Mode)...")
    print("="*60)
    
    # Core features
    print("\n📊 Core Features:")
    print("  - Return features...")
    df = add_return_features(df, windows=[1, 5, 10, 20, 60])
    
    print("  - Moving average features...")
    df = add_ma_features(df, windows=[5, 20, 60, 200])
    
    print("  - Volatility features...")
    df = add_volatility_features(df, windows=[5, 10, 20, 60])
    
    print("  - Volume features...")
    df = add_volume_features(df, windows=[5, 20])
    
    print("  - Momentum features...")
    df = add_momentum_features(df)
    
    # Market regime
    df = add_market_regime_features(df)
    
    # Phase 3: Advanced features
    print("\n🎯 Phase 3 Advanced Features:")
    df = add_sector_features(df)
    df = add_advanced_market_regime(df)
    df = add_liquidity_features(df)
    df = add_mean_reversion_features(df)
    df = add_momentum_quality_features(df)
    
    # Phase 3: News sentiment (optional)
    df = add_sentiment_features(df, use_news=use_news, api_keys=api_keys)
    
    # Create labels
    print("\n  - Creating prediction targets...")
    df = create_labels(df, horizons=config['model']['horizons'])
    
    print("\n✅ Feature engineering complete!")
    print(f"   Total columns: {len(df.columns)}")
    
    return df


def clean_features(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Clean features and remove warm-up period"""
    print("\nCleaning data...")
    
    # Remove warm-up period
    warmup_days = config['features'].get('warmup_days', 252)
    min_date = df['date'].min() + pd.Timedelta(days=warmup_days)
    df_clean = df[df['date'] >= min_date].copy()
    
    removed_rows = len(df) - len(df_clean)
    print(f"  Removed {removed_rows:,} warm-up rows")
    
    # Replace infinities with NaN
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    
    print(f"  Final shape: {df_clean.shape}")
    
    return df_clean


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Get list of feature columns
    UPDATED: Phase 3 - Comprehensive features
    """
    FEATURE_COLS = [
        # Returns
        'return_1d', 'return_5d', 'return_10d', 'return_20d', 'return_60d',
        
        # Moving averages
        'price_to_ma_5', 'price_to_ma_20', 'price_to_ma_60', 'price_to_ma_200',
        
        # Volatility
        'volatility_5d', 'volatility_10d', 'volatility_20d', 'volatility_60d',
        
        # Volume
        'volume_ratio_5', 'volume_ratio_20',
        
        # Momentum
        'momentum_5', 'momentum_10', 'momentum_20',
        'range_5d', 'range_20d',
        
        # Market regime
        'spy_return_1d', 'spy_return_5d', 'spy_return_20d',
        'spy_volatility', 'spy_trend', 'beta',
        'regime_bull', 'regime_bear', 'regime_sideways',
        
        # PHASE 3: Sector features
        'sector_return_1d', 'sector_return_5d', 'sector_return_20d',
        'vs_sector_1d', 'vs_sector_5d', 'vs_sector_20d',
        'sector_momentum', 'sector_rank', 'within_sector_rank',
        
        # PHASE 3: Liquidity
        'dollar_volume_ma_20', 'spread_ma_20', 'volume_volatility',
        
        # PHASE 3: Mean reversion
        'zscore_20', 'zscore_60',
        
        # PHASE 3: Momentum quality
        'momentum_streak', 'returns_smoothness',
        
        # PHASE 3: Sentiment
        'sentiment_score', 'sentiment_magnitude'
    ]
    
    # Only return columns that exist
    available_features = [col for col in FEATURE_COLS if col in df.columns]
    
    print(f"\n📊 Total Features: {len(available_features)}")
    
    return available_features


def save_features(df: pd.DataFrame, output_path: str):
    """Save features to parquet"""
    df.to_parquet(output_path, index=False)
    print(f"\nFeatures saved to: {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Tickers: {df['ticker'].nunique()}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")


def load_features(features_path: str) -> pd.DataFrame:
    """Load features from parquet"""
    return pd.read_parquet(features_path)


if __name__ == "__main__":
    # Test feature building
    import sys
    sys.path.insert(0, '.')
    
    config = utils.load_config('config.yaml')
    
    # Load sample data
    df = pd.read_parquet('data/raw/prices.parquet')
    
    # Build features (without news for testing)
    df = build_features(df, config, use_news=False)
    df = clean_features(df, config)
    
    # Get feature columns
    features = get_feature_columns(df)
    print(f"\nFeatures: {features}")
    
    # Save
    save_features(df, 'data/processed/features.parquet')