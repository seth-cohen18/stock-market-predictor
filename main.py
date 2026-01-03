"""
Main pipeline script - runs the entire system
"""

import sys
import os
import argparse
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils import load_config, ensure_dir
from universe import get_universe, save_universe
from fetch_prices import get_market_data
from build_features import build_features, save_features, get_feature_columns
from train import train_horizon_model
from recommend import EnhancedRecommendationEngine
from ensemble import train_ensemble_model, EnsembleModel
from performance_tracker import PerformanceTracker


def setup_directories(config):
    """Create necessary directories"""
    ensure_dir(config['data']['raw_dir'])
    ensure_dir(config['data']['processed_dir'])
    ensure_dir('models')
    ensure_dir('results')
    print("✓ Directories created")


def run_data_pipeline(config):
    """Run the data collection and feature engineering pipeline"""
    print("\n" + "="*60)
    print("STEP 1: Data Pipeline")
    print("="*60)
    
    # Get universe
    print("\n1.1 Getting stock universe...")
    tickers = get_universe(
        method=config['universe']['method'],
        size=config['universe']['size']
    )
    save_universe(tickers)
    
    # Fetch prices
    print("\n1.2 Fetching price data...")
    prices_path = f"{config['data']['raw_dir']}/prices.parquet"
    df_prices = get_market_data(
        tickers=tickers,
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date'],
        cache_path=prices_path
    )
    
    # Build features
    print("\n1.3 Building features...")
    
    # Check if Phase 3 mode requested
    use_news = getattr(args, 'use_news', False) if 'args' in dir() else False
    api_key = getattr(args, 'api_key', None) if 'args' in dir() else None
    api_keys = {'alpha_vantage': api_key} if api_key else None
    
    df_features = build_features(df_prices, config, use_news=use_news, api_keys=api_keys)
    
    # Save features
    features_path = f"{config['data']['processed_dir']}/features.parquet"
    save_features(df_features, features_path)
    
    return df_features


def run_model_training(config, df_features):
    """Train models for all horizons with daily adaptive retraining"""
    print("\n" + "="*60)
    print("STEP 2: Model Training (Daily Adaptive Retraining)")
    print("="*60)
    
    # Import adaptive trainer
    from adaptive_trainer import AdaptiveTrainer
    
    feature_cols = get_feature_columns(df_features)
    print(f"Number of features: {len(feature_cols)}")
    
    # Initialize adaptive trainer (tracks improvement over time)
    trainer = AdaptiveTrainer()
    
    results = {}
    for horizon in config['model']['horizons']:
        print(f"\n2.{horizon} Training {horizon}-day model...")
        
        model_path = f'models/model_{horizon}d.txt'
        
        # Use adaptive training (proves daily learning)
        training_metrics = trainer.train_with_latest_data(
            df_features=df_features,
            feature_cols=feature_cols,
            horizon=horizon,
            model_path=model_path
        )
        
        results[f'{horizon}d'] = training_metrics
    
    return results



def run_full_pipeline(config_path='config.yaml'):
    """Run the complete pipeline"""
    print("\n" + "="*80)
    print("STOCK MARKET PREDICTION SYSTEM")
    print("="*80)
    
    # Load config
    config = load_config(config_path)
    
    # Setup
    setup_directories(config)
    
    # Run data pipeline
    df_features = run_data_pipeline(config)
    
    # Train models
    training_results = run_model_training(config, df_features)
    
    # Summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    
    print("\nModel Performance Summary:")
    for horizon_name, result in training_results.items():
        if 'metrics' in result:
            metrics = result['metrics']
            print(f"\n{horizon_name}:")
            print(f"  Test IC: {metrics['ic']:.4f}")
            print(f"  Direction Accuracy: {metrics['direction_accuracy']:.2%}")
            print(f"  R²: {metrics['r2']:.4f}")
    
    print("\n✅ System ready! You can now use recommend.py to get recommendations.")
    
    return training_results


def get_recommendations_cli():
    """CLI for getting recommendations"""
    parser = argparse.ArgumentParser(description='Get investment recommendations')
    parser.add_argument('--capital', type=float, required=True, 
                       help='Investment capital in dollars')
    parser.add_argument('--horizon', type=str, default='1w',
                       choices=['1d', '1w', '1m'],
                       help='Investment horizon')
    parser.add_argument('--risk', type=str, default='medium',
                       choices=['low', 'medium', 'high'],
                       help='Risk tolerance level')
    parser.add_argument('--goal', type=str, default='max_sharpe',
                       choices=['max_return', 'max_sharpe', 'prob_target'],
                       help='Optimization goal')
    parser.add_argument('--target', type=float, default=None,
                       help='Target ROI as decimal (e.g., 0.03 for 3%%)')
    parser.add_argument('--positions', type=int, default=None,
                       help='Number of positions to hold')
    parser.add_argument('--whole-shares', action='store_true',
                       help='Use whole shares only (default: fractional allowed)')
    
    args = parser.parse_args()
    
    # Initialize engine
    engine = EnhancedRecommendationEngine()
    
    # Get recommendations
    result = engine.recommend(
        capital=args.capital,
        horizon=args.horizon,
        risk_level=args.risk,
        goal=args.goal,
        target_roi=args.target,
        num_positions=args.positions,
        allow_fractional=not args.whole_shares
    )
    
    # Print results
    engine.print_recommendations(result)
    
    return result


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'recommend':
        # Run recommendations CLI
        sys.argv.pop(1)  # Remove 'recommend' from args
        get_recommendations_cli()
    else:
        # Check for Phase 3 arguments
        parser = argparse.ArgumentParser(description='Run full pipeline', add_help=False)
        parser.add_argument('--use-news', action='store_true',
                           help='Enable news sentiment analysis (Phase 3)')
        parser.add_argument('--api-key', type=str, default=None,
                           help='Alpha Vantage API key for news sentiment')
        parser.add_argument('--help', action='store_true', help='Show this help message')
        
        args, unknown = parser.parse_known_args()
        
        if args.help:
            print("\nStock Market Predictor - Main Pipeline")
            print("\nUsage:")
            print("  python main.py                    # Run with baseline features")
            print("  python main.py --use-news         # Run Phase 3 with news sentiment")
            print("  python main.py --use-news --api-key YOUR_KEY  # Phase 3 with API")
            print("\nOptions:")
            print("  --use-news        Enable Phase 3 news sentiment analysis")
            print("  --api-key KEY     Alpha Vantage API key (get free at alphavantage.co)")
            print("  --help            Show this message")
            sys.exit(0)
        
        # Run full pipeline with Phase 3 options
        globals()['args'] = args  # Make args available to run_data_pipeline
        run_full_pipeline('config.yaml')