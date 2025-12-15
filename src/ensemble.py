"""
Ensemble Model System - Phase 2
Combines LightGBM, XGBoost, and Random Forest for better predictions
"""

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import utils


class EnsembleModel:
    """Ensemble of multiple models for robust predictions"""
    
    def __init__(self, model_type: str = 'weighted'):
        """
        Args:
            model_type: 'weighted', 'voting', or 'stacking'
        """
        self.model_type = model_type
        self.models = {}
        self.weights = {}
        self.feature_importance = {}
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        """
        Train all models in the ensemble
        
        Returns:
            Dict with model metrics
        """
        print("\n🎯 Training Ensemble Models...")
        
        # Model 1: LightGBM (Current champion)
        print("\n  Training LightGBM...")
        self.models['lgbm'] = self._train_lightgbm(X_train, y_train, X_val, y_val)
        lgbm_pred = self.models['lgbm'].predict(X_val)
        lgbm_score = self._calculate_direction_accuracy(y_val, lgbm_pred)
        print(f"    ✓ LightGBM Direction Accuracy: {lgbm_score:.2%}")
        
        # Model 2: XGBoost
        print("\n  Training XGBoost...")
        self.models['xgb'] = self._train_xgboost(X_train, y_train, X_val, y_val)
        xgb_pred = self.models['xgb'].predict(X_val)
        xgb_score = self._calculate_direction_accuracy(y_val, xgb_pred)
        print(f"    ✓ XGBoost Direction Accuracy: {xgb_score:.2%}")
        
        # Model 3: Random Forest
        print("\n  Training Random Forest...")
        self.models['rf'] = self._train_random_forest(X_train, y_train)
        rf_pred = self.models['rf'].predict(X_val)
        rf_score = self._calculate_direction_accuracy(y_val, rf_pred)
        print(f"    ✓ Random Forest Direction Accuracy: {rf_score:.2%}")
        
        # Calculate optimal weights based on validation performance
        total_score = lgbm_score + xgb_score + rf_score
        self.weights = {
            'lgbm': lgbm_score / total_score,
            'xgb': xgb_score / total_score,
            'rf': rf_score / total_score
        }
        
        print(f"\n  📊 Ensemble Weights:")
        print(f"    • LightGBM: {self.weights['lgbm']:.1%}")
        print(f"    • XGBoost:  {self.weights['xgb']:.1%}")
        print(f"    • Random Forest: {self.weights['rf']:.1%}")
        
        # Test ensemble on validation set
        ensemble_pred = self.predict(X_val)
        ensemble_score = self._calculate_direction_accuracy(y_val, ensemble_pred)
        
        print(f"\n  🎯 ENSEMBLE Direction Accuracy: {ensemble_score:.2%}")
        
        improvement = ensemble_score - max(lgbm_score, xgb_score, rf_score)
        if improvement > 0:
            print(f"  ✅ Improvement over best single model: +{improvement:.2%}")
        else:
            print(f"  ⚠️  Best single model performs better by {-improvement:.2%}")
        
        return {
            'lgbm_accuracy': lgbm_score,
            'xgb_accuracy': xgb_score,
            'rf_accuracy': rf_score,
            'ensemble_accuracy': ensemble_score,
            'improvement': improvement,
            'weights': self.weights
        }
    
    def _train_lightgbm(self, X_train, y_train, X_val, y_val):
        """Train LightGBM model"""
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 63,
            'learning_rate': 0.03,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'max_depth': 8,
            'min_data_in_leaf': 100,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'verbose': -1,
            'seed': 42
        }
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        return model
    
    def _train_xgboost(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model"""
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=100,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=50,
            eval_metric='rmse'
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        return model
    
    def _train_random_forest(self, X_train, y_train):
        """Train Random Forest model"""
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_split=100,
            min_samples_leaf=50,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        model.fit(X_train, y_train)
        
        return model
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make ensemble predictions
        
        Args:
            X: Features
            
        Returns:
            Weighted average predictions
        """
        if self.model_type == 'weighted':
            # Weighted average based on validation performance
            lgbm_pred = self.models['lgbm'].predict(X)
            xgb_pred = self.models['xgb'].predict(X)
            rf_pred = self.models['rf'].predict(X)
            
            ensemble_pred = (
                self.weights['lgbm'] * lgbm_pred +
                self.weights['xgb'] * xgb_pred +
                self.weights['rf'] * rf_pred
            )
            
            return ensemble_pred
        
        elif self.model_type == 'voting':
            # Simple average
            lgbm_pred = self.models['lgbm'].predict(X)
            xgb_pred = self.models['xgb'].predict(X)
            rf_pred = self.models['rf'].predict(X)
            
            return (lgbm_pred + xgb_pred + rf_pred) / 3
        
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def _calculate_direction_accuracy(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate direction accuracy"""
        return np.mean((y_true > 0) == (y_pred > 0))
    
    def save(self, model_dir: str, horizon: int):
        """Save ensemble models"""
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save each model
        self.models['lgbm'].save_model(str(model_path / f'ensemble_lgbm_{horizon}d.txt'))
        self.models['xgb'].save_model(str(model_path / f'ensemble_xgb_{horizon}d.json'))
        
        import joblib
        joblib.dump(self.models['rf'], str(model_path / f'ensemble_rf_{horizon}d.pkl'))
        
        # Save weights and metadata
        metadata = {
            'weights': self.weights,
            'model_type': self.model_type,
            'horizon': horizon,
            'created_at': datetime.now().isoformat()
        }
        
        with open(model_path / f'ensemble_metadata_{horizon}d.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Ensemble saved to {model_path}")
    
    @classmethod
    def load(cls, model_dir: str, horizon: int):
        """Load ensemble models"""
        model_path = Path(model_dir)
        
        ensemble = cls()
        
        # Load metadata
        with open(model_path / f'ensemble_metadata_{horizon}d.json', 'r') as f:
            metadata = json.load(f)
        
        ensemble.weights = metadata['weights']
        ensemble.model_type = metadata['model_type']
        
        # Load models
        ensemble.models['lgbm'] = lgb.Booster(model_file=str(model_path / f'ensemble_lgbm_{horizon}d.txt'))
        ensemble.models['xgb'] = xgb.XGBRegressor()
        ensemble.models['xgb'].load_model(str(model_path / f'ensemble_xgb_{horizon}d.json'))
        
        import joblib
        ensemble.models['rf'] = joblib.load(str(model_path / f'ensemble_rf_{horizon}d.pkl'))
        
        return ensemble


def train_ensemble_model(df_features: pd.DataFrame, feature_cols: List[str], 
                         horizon: int, config: Dict) -> Tuple[EnsembleModel, Dict]:
    """
    Train ensemble model for a given horizon
    
    Args:
        df_features: Feature dataframe
        feature_cols: List of feature columns
        horizon: Prediction horizon in days
        config: Configuration dict
        
    Returns:
        Trained ensemble model and metrics dict
    """
    print(f"\n{'='*60}")
    print(f"Training Ensemble for {horizon}-day horizon")
    print(f"{'='*60}")
    
    # Prepare data
    target_col = f'target_{horizon}d'
    df_valid = df_features.dropna(subset=[target_col]).copy()
    
    print(f"Valid samples: {len(df_valid):,}")
    print(f"Date range: {df_valid['date'].min()} to {df_valid['date'].max()}")
    
    # Split by time
    train_end_date = df_valid['date'].max() - pd.Timedelta(days=252)  # 1 year for testing
    
    train_df = df_valid[df_valid['date'] <= train_end_date]
    test_df = df_valid[df_valid['date'] > train_end_date]
    
    print(f"\nTrain period: {train_df['date'].min()} to {train_df['date'].max()}")
    print(f"Test period: {test_df['date'].min()} to {test_df['date'].max()}")
    print(f"Train samples: {len(train_df):,}")
    print(f"Test samples: {len(test_df):,}")
    
    # Further split train into train/val for ensemble weight calculation
    val_end_date = train_end_date - pd.Timedelta(days=126)  # 6 months for validation
    
    train_only = train_df[train_df['date'] <= val_end_date]
    val_df = train_df[train_df['date'] > val_end_date]
    
    X_train = train_only[feature_cols].fillna(0)
    y_train = train_only[target_col]
    
    X_val = val_df[feature_cols].fillna(0)
    y_val = val_df[target_col]
    
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df[target_col]
    
    # Train ensemble
    ensemble = EnsembleModel(model_type='weighted')
    train_metrics = ensemble.train(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    print(f"\n{'='*60}")
    print(f"Final Test Evaluation")
    print(f"{'='*60}")
    
    test_pred = ensemble.predict(X_test)
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_mae = mean_absolute_error(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)
    test_direction = np.mean((y_test > 0) == (test_pred > 0))
    test_ic = np.corrcoef(y_test, test_pred)[0, 1]
    
    print(f"\nTest Metrics:")
    print(f"  RMSE: {test_rmse:.6f}")
    print(f"  MAE: {test_mae:.6f}")
    print(f"  R²: {test_r2:.4f}")
    print(f"  Direction Accuracy: {test_direction:.2%}")
    print(f"  IC: {test_ic:.4f}")
    
    metrics = {
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'test_direction_accuracy': test_direction,
        'test_ic': test_ic,
        **train_metrics
    }
    
    return ensemble, metrics


if __name__ == "__main__":
    # Test ensemble
    import sys
    sys.path.insert(0, '.')
    
    import build_features
    
    config = utils.load_config('config.yaml')
    
    # Load features
    df = build_features.load_features('data/processed/features.parquet')
    feature_cols = build_features.get_feature_columns(df)
    
    # Train ensemble for 5-day horizon
    ensemble, metrics = train_ensemble_model(df, feature_cols, 5, config)
    
    print(f"\n{'='*60}")
    print("ENSEMBLE MODEL COMPLETE")
    print(f"{'='*60}")
    print(f"Test Direction Accuracy: {metrics['test_direction_accuracy']:.2%}")
    print(f"Improvement: {metrics['improvement']:.2%}")