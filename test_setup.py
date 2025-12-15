"""
Quick test script to verify the system setup
This runs a minimal version of the pipeline with just a few stocks
"""

import sys
import os
import pandas as pd
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils import load_config, ensure_dir
from universe import DEFAULT_UNIVERSE
from fetch_prices import fetch_all_prices
from build_features import build_features, get_feature_columns
from train import train_horizon_model


def quick_test():
    """Run a quick test of the system with minimal data"""
    
    print("\n" + "="*60)
    print("QUICK SYSTEM TEST")
    print("="*60)
    print("\nThis will test the system with just 10 stocks")
    print("Full pipeline will use 200+ stocks and take longer")
    
    # Load config
    config = load_config('config.yaml')
    
    # Setup directories
    ensure_dir('data/raw')
    ensure_dir('data/processed')
    ensure_dir('models')
    
    # Use just 10 stocks for testing
    test_tickers = ['SPY', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 
                    'NVDA', 'TSLA', 'META', 'JPM', 'V']
    
    print(f"\n1. Testing data fetch with {len(test_tickers)} tickers...")
    print("   (This may take 1-2 minutes)")
    
    # Fetch 2 years of data (faster than full 5+ years)
    start_date = (pd.Timestamp.now() - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
    
    try:
        df_prices = fetch_all_prices(
            tickers=test_tickers,
            start_date=start_date,
            end_date=None,
            output_path='data/raw/test_prices.parquet'
        )
        
        print(f"\n✓ Data fetch successful!")
        print(f"  Shape: {df_prices.shape}")
        print(f"  Tickers: {df_prices['ticker'].nunique()}")
        
    except Exception as e:
        print(f"\n✗ Data fetch failed: {e}")
        return False
    
    print("\n2. Testing feature engineering...")
    
    try:
        df_features = build_features(df_prices, config)
        
        print(f"\n✓ Feature engineering successful!")
        print(f"  Shape: {df_features.shape}")
        print(f"  Features: {len(get_feature_columns(df_features))}")
        
    except Exception as e:
        print(f"\n✗ Feature engineering failed: {e}")
        return False
    
    print("\n3. Testing model training (1-day horizon only)...")
    
    try:
        feature_cols = get_feature_columns(df_features)
        
        result = train_horizon_model(
            df=df_features,
            horizon=1,
            config=config,
            feature_cols=feature_cols,
            output_dir='models'
        )
        
        print(f"\n✓ Model training successful!")
        print(f"  Test IC: {result['metrics']['ic']:.4f}")
        print(f"  Direction Accuracy: {result['metrics']['direction_accuracy']:.2%}")
        
    except Exception as e:
        print(f"\n✗ Model training failed: {e}")
        return False
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\nYour system is working correctly.")
    print("\nNext steps:")
    print("1. Run 'python main.py' for the full pipeline with all stocks")
    print("2. Then use 'python main.py recommend --capital 10000 --horizon 1w'")
    print("   to get recommendations")
    
    return True


if __name__ == "__main__":
    try:
        success = quick_test()
        if not success:
            print("\n❌ Test failed. Check error messages above.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)