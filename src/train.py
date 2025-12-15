"""
Model training with proper time-series validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
import joblib
from datetime import datetime
import json

import utils
import build_features


class TimeSeriesSplit:
    """Custom time series splitter for walk-forward validation"""
    
    def __init__(self, train_years: int = 5, test_years: int = 1):
        self.train_years = train_years
        self.test_years = test_years
    
    def split(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate train/test splits
        
        Args:
            df: DataFrame with date column
            
        Returns:
            List of (train_df, test_df) tuples
        """
        df = df.sort_values('date').reset_index(drop=True)
        
        min_date = df['date'].min()
        max_date = df['date'].max()
        
        # Calculate total data span in days
        total_days = (max_date - min_date).days
        
        # If we have less data than train_years, use a simple 80/20 split
        if total_days < (self.train_years * 365):
            print(f"  Warning: Only {total_days} days of data. Using 80/20 train/test split instead of walk-forward.")
            split_point = int(len(df) * 0.8)
            train_df = df.iloc[:split_point].copy()
            test_df = df.iloc[split_point:].copy()
            return [(train_df, test_df)]
        
        splits = []
        current_test_start = min_date + pd.DateOffset(years=self.train_years)
        
        while current_test_start < max_date:
            train_end = current_test_start
            test_end = current_test_start + pd.DateOffset(years=self.test_years)
            
            train_mask = (df['date'] < train_end)
            test_mask = (df['date'] >= current_test_start) & (df['date'] < test_end)
            
            train_df = df[train_mask].copy()
            test_df = df[test_mask].copy()
            
            if len(train_df) > 0 and len(test_df) > 0:
                splits.append((train_df, test_df))
            
            # Move forward by test_years
            current_test_start = test_end
        
        return splits


def train_model(X_train: pd.DataFrame,
                y_train: pd.Series,
                X_val: pd.DataFrame,
                y_val: pd.Series,
                config: Dict) -> lgb.Booster:
    """
    Train a LightGBM model
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        config: Model configuration
        
    Returns:
        Trained model
    """
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Train
    model = lgb.train(
        config['model']['lgbm_params'],
        train_data,
        num_boost_round=config['model']['n_estimators'],
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(config['model']['early_stopping_rounds'], verbose=False),
            lgb.log_evaluation(period=100)
        ]
    )
    
    return model


def evaluate_model(model: lgb.Booster,
                   X: pd.DataFrame,
                   y: pd.Series,
                   name: str = "Test") -> Dict:
    """
    Evaluate model performance
    
    Args:
        model: Trained model
        X: Features
        y: True targets
        name: Dataset name for printing
        
    Returns:
        Dictionary of metrics
    """
    y_pred = model.predict(X)
    
    # Calculate metrics
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    # Direction accuracy
    direction_acc = ((y > 0) == (y_pred > 0)).mean()
    
    # Information coefficient (correlation)
    ic = np.corrcoef(y, y_pred)[0, 1]
    
    metrics = {
        'mse': mse,
        'rmse': np.sqrt(mse),
        'mae': mae,
        'r2': r2,
        'direction_accuracy': direction_acc,
        'ic': ic
    }
    
    # Print
    print(f"\n{name} Metrics:")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MAE: {metrics['mae']:.6f}")
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"  Direction Accuracy: {metrics['direction_accuracy']:.2%}")
    print(f"  IC: {metrics['ic']:.4f}")
    
    return metrics


def train_horizon_model(df: pd.DataFrame,
                       horizon: int,
                       config: Dict,
                       feature_cols: List[str],
                       output_dir: str) -> Dict:
    """
    Train a model for a specific horizon
    
    Args:
        df: DataFrame with features and targets
        horizon: Prediction horizon
        config: Configuration
        feature_cols: List of feature columns
        output_dir: Directory to save models
        
    Returns:
        Dictionary with model and metrics
    """
    print(f"\n{'='*60}")
    print(f"Training model for {horizon}-day horizon")
    print(f"{'='*60}")
    
    target_col = f'target_{horizon}d'
    
    # Filter to rows with valid targets
    df_valid = df[df[target_col].notna()].copy()
    
    print(f"Valid samples: {len(df_valid):,}")
    print(f"Date range: {df_valid['date'].min()} to {df_valid['date'].max()}")
    
    # Create time series splits
    splitter = TimeSeriesSplit(
        train_years=config['backtest']['train_years'],
        test_years=config['backtest']['test_years']
    )
    
    splits = splitter.split(df_valid)
    print(f"Number of splits: {len(splits)}")
    
    # Train on final split (most recent)
    train_df, test_df = splits[-1]
    
    print(f"\nTrain period: {train_df['date'].min()} to {train_df['date'].max()}")
    print(f"Test period: {test_df['date'].min()} to {test_df['date'].max()}")
    print(f"Train samples: {len(train_df):,}")
    print(f"Test samples: {len(test_df):,}")
    
    # Prepare data
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df[target_col]
    
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df[target_col]
    
    # Train model
    print("\nTraining model...")
    model = train_model(X_train, y_train, X_test, y_test, config)
    
    # Evaluate
    train_metrics = evaluate_model(model, X_train, y_train, name="Train")
    test_metrics = evaluate_model(model, X_test, y_test, name="Test")
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 features:")
    print(importance_df.head(10).to_string(index=False))
    
    # Save model
    model_path = f"{output_dir}/model_{horizon}d.txt"
    utils.ensure_dir(Path(model_path).parent)
    model.save_model(model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Save metadata
    metadata = {
        'horizon': horizon,
        'target_col': target_col,
        'feature_cols': feature_cols,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'train_period': {
            'start': str(train_df['date'].min()),
            'end': str(train_df['date'].max()),
            'samples': len(train_df)
        },
        'test_period': {
            'start': str(test_df['date'].min()),
            'end': str(test_df['date'].max()),
            'samples': len(test_df)
        },
        'feature_importance': importance_df.head(20).to_dict('records'),
        'trained_at': datetime.now().isoformat()
    }
    
    metadata_path = f"{output_dir}/model_{horizon}d_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return {
        'model': model,
        'metrics': test_metrics,
        'metadata': metadata
    }


def load_model(model_path: str) -> lgb.Booster:
    """Load a trained model"""
    model = lgb.Booster(model_file=model_path)
    print(f"Loaded model from {model_path}")
    return model


def load_model_metadata(metadata_path: str) -> Dict:
    """Load model metadata"""
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    
    from src.utils import load_config
    from src.build_features import load_features, get_feature_columns
    
    # Load config
    config = utils.load_config('config.yaml')
    
    # Load features
    features_path = f"{config['data']['processed_dir']}/features.parquet"
    df = build_features.load_features(features_path)
    
    # Get feature columns
    feature_cols = build_features.get_feature_columns(df)
    print(f"\nNumber of features: {len(feature_cols)}")
    
    # Output directory
    model_dir = "models"
    utils.ensure_dir(model_dir)
    
    # Train models for each horizon
    results = {}
    for horizon in config['model']['horizons']:
        result = train_horizon_model(
            df=df,
            horizon=horizon,
            config=config,
            feature_cols=feature_cols,
            output_dir=model_dir
        )
        results[f'{horizon}d'] = result
    
    # Summary
    print(f"\n{'='*60}")
    print("Training Summary")
    print(f"{'='*60}")
    for horizon_name, result in results.items():
        print(f"\n{horizon_name}:")
        metrics = result['metrics']
        print(f"  Test IC: {metrics['ic']:.4f}")
        print(f"  Test Direction Acc: {metrics['direction_accuracy']:.2%}")
        print(f"  Test R²: {metrics['r2']:.4f}")
    
    print("\n✅ Model training complete!")