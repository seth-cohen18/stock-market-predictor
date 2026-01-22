"""
Main training pipeline for stock market predictor
Runs the complete data → features → training workflow
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import yaml
import pandas as pd
from pathlib import Path

def load_config(config_path='config.yaml'):
    """Load configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    """Run the complete training pipeline"""
    
    print("=" * 70)
    print("STOCK MARKET PREDICTOR - TRAINING PIPELINE")
    print("=" * 70)
    
    # Import modules
    from fetch_prices import update_prices, fetch_all_prices, load_prices
    from build_features import build_features, clean_features, get_feature_columns, save_features, load_features
    from train import train_horizon_model
    
    # Load config
    config = load_config('config.yaml')
    
    # Step 1: Fetch/Update latest prices
    print("\n📊 Step 1: Fetching latest market data...")
    
    prices_path = f"{config['data']['raw_dir']}/prices.parquet"
    
    try:
        # Check if prices file exists
        if Path(prices_path).exists():
            print("Updating existing price data...")
            df_prices = update_prices(prices_path, config['data']['tickers'])
        else:
            print("Fetching fresh price data...")
            # Get date range from config or use defaults
            start_date = config.get('data', {}).get('start_date', '2019-01-01')
            end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
            df_prices = fetch_all_prices(
                tickers=config['data']['tickers'],
                start_date=start_date,
                end_date=end_date,
                output_path=prices_path
            )
        print("✅ Market data updated")
    except Exception as e:
        print(f"⚠️  Error updating prices: {e}")
        print("   Loading existing data...")
        df_prices = load_prices(prices_path)
    
    # Step 2: Build features
    print("\n🔧 Step 2: Building features...")
    try:
        # Build features (without news for speed)
        df_features = build_features(df_prices, config, use_news=False)
        
        # Clean features
        df_features = clean_features(df_features, config)
        
        # Save features
        features_path = f"{config['data']['processed_dir']}/features.parquet"
        save_features(df_features, features_path)
        
        print("✅ Features built")
    except Exception as e:
        print(f"❌ Error building features: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Step 3: Train models
    print("\n🤖 Step 3: Training models...")
    
    # Load features
    features_path = f"{config['data']['processed_dir']}/features.parquet"
    df = load_features(features_path)
    
    # Get feature columns
    feature_cols = get_feature_columns(df)
    print(f"Number of features: {len(feature_cols)}")
    
    # Create models directory
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    # Train models for each horizon
    results = {}
    
    for horizon in config['model']['horizons']:
        print(f"\nTraining {horizon}-day model...")
        result = train_horizon_model(
            df=df,
            horizon=horizon,
            config=config,
            feature_cols=feature_cols,
            output_dir=model_dir
        )
        results[f'{horizon}d'] = result
    
    # Summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    for horizon_name, result in results.items():
        print(f"\n{horizon_name}:")
        metrics = result['metrics']
        print(f"  Direction Accuracy: {metrics['direction_accuracy']:.2%}")
        print(f"  Information Coefficient: {metrics['ic']:.4f}")
        print(f"  R²: {metrics['r2']:.4f}")
    
    print("\n✅ Training pipeline complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()