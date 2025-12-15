"""
Main Ensemble Pipeline - Train and use ensemble models
Run this instead of main.py to use ensemble predictions
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils import load_config, ensure_dir
from universe import get_universe, save_universe
from fetch_prices import get_market_data
from build_features import build_features, save_features, get_feature_columns, load_features
from ensemble import train_ensemble_model, EnsembleModel
from performance_tracker import PerformanceTracker


def main():
    """Run ensemble pipeline"""
    
    print("\n" + "="*80)
    print("STOCK MARKET PREDICTION SYSTEM - ENSEMBLE MODE")
    print("="*80)
    
    config = load_config('config.yaml')
    
    # Create directories
    for directory in ['data/raw', 'data/processed', 'models', 'data/performance']:
        ensure_dir(directory)
    
    print("✓ Directories created")
    
    # Step 1: Data Pipeline
    print("\n" + "="*60)
    print("STEP 1: Data Pipeline")
    print("="*60)
    
    # Use existing data if available
    prices_path = f"{config['data']['raw_dir']}/prices.parquet"
    features_path = f"{config['data']['processed_dir']}/features.parquet"
    
    # Check if we have existing features
    from pathlib import Path
    if Path(features_path).exists():
        print("\n✓ Using existing features from data/processed/features.parquet")
        df_features = load_features(features_path)
        print(f"  Shape: {df_features.shape}")
        print(f"  Tickers: {df_features['ticker'].nunique()}")
        print(f"  Date range: {df_features['date'].min()} to {df_features['date'].max()}")
    else:
        print("\n⚠️  No existing features found. Building from scratch...")
        
        # 1.1 Get universe
        print("\n1.1 Getting stock universe...")
        universe = get_universe(
            method=config['universe']['method'],
            size=config['universe']['size']
        )
        print(f"Universe size: {len(universe)} tickers")
        save_universe(universe, f"{config['data']['processed_dir']}/universe.json")
        
        # 1.2 Get price data
        print("\n1.2 Fetching price data...")
        df_prices = get_market_data(universe, config)
        
        if len(df_prices) == 0:
            print("❌ No price data available. Please run 'python main.py' first to fetch data.")
            return
        
        # 1.3 Build features
        print("\n1.3 Building features...")
        df_features = build_features(df_prices, config)
        save_features(df_features, features_path)
    
    # Step 2: Train Ensemble Models
    print("\n" + "="*60)
    print("STEP 2: Ensemble Model Training")
    print("="*60)
    
    feature_cols = get_feature_columns(df_features)
    print(f"Number of features: {len(feature_cols)}")
    
    ensemble_metrics = {}
    
    for horizon in config['model']['horizons']:
        print(f"\n{'='*60}")
        print(f"Training {horizon}-day ensemble")
        print(f"{'='*60}")
        
        ensemble, metrics = train_ensemble_model(df_features, feature_cols, horizon, config)
        
        # Save ensemble
        ensemble.save('models', horizon)
        
        ensemble_metrics[f'{horizon}d'] = metrics
    
    # Step 3: Performance Summary
    print("\n" + "="*80)
    print("ENSEMBLE TRAINING COMPLETE")
    print("="*80)
    
    print("\n📊 Model Performance Summary:\n")
    
    for horizon_key, metrics in ensemble_metrics.items():
        print(f"{horizon_key}:")
        print(f"  Individual Models:")
        print(f"    LightGBM: {metrics['lgbm_accuracy']:.2%}")
        print(f"    XGBoost:  {metrics['xgb_accuracy']:.2%}")
        print(f"    Random Forest: {metrics['rf_accuracy']:.2%}")
        print(f"  ENSEMBLE: {metrics['ensemble_accuracy']:.2%}")
        print(f"  Improvement: {metrics['improvement']:+.2%}")
        print(f"  Test IC: {metrics['test_ic']:.4f}")
        print(f"  Test Direction Accuracy: {metrics['test_direction_accuracy']:.2%}")
        print()
    
    # Initialize performance tracker
    print("\n✓ Performance tracker initialized at data/performance/")
    tracker = PerformanceTracker()
    
    print("\n" + "="*80)
    print("✅ System ready! You can now:")
    print("  1. Run: python stock_predictor_gui.py (GUI will use ensemble models)")
    print("  2. Run: python track_performance.py (view performance stats)")
    print("  3. Generate recommendations with ensemble predictions")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()