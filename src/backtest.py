"""
Backtest trading strategies with realistic costs
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from pathlib import Path

import utils
import build_features
import train


class Backtest:
    """Backtesting engine for trading strategies"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.commission = config['backtest']['commission']
        self.slippage = config['backtest']['slippage']
        self.max_positions = config['backtest']['max_positions']
        self.min_weight = config['backtest']['min_weight']
        self.max_weight = config['backtest']['max_weight']
        
    def generate_signals(self,
                        df: pd.DataFrame,
                        model: object,
                        feature_cols: List[str],
                        horizon: int) -> pd.DataFrame:
        """
        Generate trading signals from model predictions
        
        Args:
            df: DataFrame with features
            model: Trained model
            feature_cols: List of feature columns
            horizon: Prediction horizon
            
        Returns:
            DataFrame with signals
        """
        # Make predictions
        X = df[feature_cols].fillna(0)
        df['prediction'] = model.predict(X)
        
        # Add actual returns for evaluation
        df['actual_return'] = df[f'target_{horizon}d']
        
        return df
    
    def rank_and_select(self,
                       df: pd.DataFrame,
                       date: pd.Timestamp,
                       method: str = 'top_n') -> pd.DataFrame:
        """
        Rank stocks and select top N for the portfolio
        
        Args:
            df: DataFrame with predictions
            date: Date to select for
            method: Selection method
            
        Returns:
            Selected stocks for this date
        """
        # Get data for this date
        date_df = df[df['date'] == date].copy()
        
        if len(date_df) == 0:
            return pd.DataFrame()
        
        # Remove any with NaN predictions or actuals
        date_df = date_df.dropna(subset=['prediction', 'actual_return'])
        
        if len(date_df) == 0:
            return pd.DataFrame()
        
        # Rank by prediction
        date_df = date_df.sort_values('prediction', ascending=False)
        
        if method == 'top_n':
            # Take top N
            selected = date_df.head(self.max_positions).copy()
            
        elif method == 'long_short':
            # Long top N, short bottom N
            n_per_side = self.max_positions // 2
            long = date_df.head(n_per_side).copy()
            long['position'] = 'long'
            short = date_df.tail(n_per_side).copy()
            short['position'] = 'short'
            selected = pd.concat([long, short])
            
        else:
            raise ValueError(f"Unknown selection method: {method}")
        
        return selected
    
    def calculate_portfolio_returns(self,
                                   selections: List[pd.DataFrame],
                                   allocation_method: str = 'equal') -> pd.DataFrame:
        """
        Calculate portfolio returns across all dates
        
        Args:
            selections: List of selected stocks per date
            allocation_method: How to allocate weights
            
        Returns:
            DataFrame with portfolio returns
        """
        portfolio_returns = []
        
        for selected_df in selections:
            if len(selected_df) == 0:
                continue
            
            # Calculate weights
            if allocation_method == 'equal':
                weights = np.ones(len(selected_df)) / len(selected_df)
                
            elif allocation_method == 'volatility':
                # Inverse volatility weighting
                if 'volatility_20d' in selected_df.columns:
                    vols = selected_df['volatility_20d'].fillna(selected_df['volatility_20d'].median())
                    inv_vol = 1 / (vols + 1e-8)
                    weights = inv_vol / inv_vol.sum()
                else:
                    weights = np.ones(len(selected_df)) / len(selected_df)
                    
            elif allocation_method == 'prediction':
                # Weight by prediction strength
                preds = selected_df['prediction'].values
                # Normalize to positive
                preds = preds - preds.min() + 0.01
                weights = preds / preds.sum()
            else:
                weights = np.ones(len(selected_df)) / len(selected_df)
            
            # Clip weights
            weights = np.clip(weights, self.min_weight, self.max_weight)
            weights = weights / weights.sum()  # Renormalize
            
            # Calculate portfolio return
            returns = selected_df['actual_return'].values
            portfolio_return = (weights * returns).sum()
            
            # Apply costs
            # Turnover cost (assume 100% turnover for simplicity in this basic version)
            cost = self.commission + self.slippage
            portfolio_return_after_cost = portfolio_return - cost
            
            portfolio_returns.append({
                'date': selected_df['date'].iloc[0],
                'return': portfolio_return,
                'return_after_cost': portfolio_return_after_cost,
                'n_positions': len(selected_df),
                'top_ticker': selected_df.iloc[0]['ticker'],
                'top_prediction': selected_df.iloc[0]['prediction']
            })
        
        return pd.DataFrame(portfolio_returns)
    
    def run_backtest(self,
                    df: pd.DataFrame,
                    model: object,
                    feature_cols: List[str],
                    horizon: int,
                    start_date: str = None,
                    end_date: str = None) -> Dict:
        """
        Run full backtest
        
        Args:
            df: DataFrame with features
            model: Trained model
            feature_cols: Feature columns
            horizon: Prediction horizon
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            Dictionary with results
        """
        print(f"\nRunning backtest for {horizon}-day horizon...")
        
        # Generate signals
        df_signals = self.generate_signals(df, model, feature_cols, horizon)
        
        # Filter date range
        if start_date:
            df_signals = df_signals[df_signals['date'] >= pd.to_datetime(start_date)]
        if end_date:
            df_signals = df_signals[df_signals['date'] <= pd.to_datetime(end_date)]
        
        # Get unique dates
        dates = sorted(df_signals['date'].unique())
        print(f"Backtesting period: {dates[0]} to {dates[-1]}")
        print(f"Number of days: {len(dates)}")
        
        # Select stocks for each date
        selections = []
        for date in dates:
            selected = self.rank_and_select(df_signals, date)
            if len(selected) > 0:
                selections.append(selected)
        
        print(f"Days with positions: {len(selections)}")
        
        # Calculate portfolio returns
        portfolio_df = self.calculate_portfolio_returns(selections)
        
        if len(portfolio_df) == 0:
            print("No valid portfolio returns!")
            return {}
        
        # Calculate metrics
        returns = portfolio_df['return_after_cost']
        
        metrics = {
            'total_return': (1 + returns).prod() - 1,
            'annual_return': (1 + returns).prod() ** (252 / len(returns)) - 1,
            'sharpe': utils.calculate_sharpe(returns),
            'max_drawdown': utils.calculate_max_drawdown(returns),
            'win_rate': (returns > 0).mean(),
            'avg_return': returns.mean(),
            'std_return': returns.std(),
            'num_trades': len(returns)
        }
        
        # Print metrics
        utils.print_backtest_metrics(returns, name=f"{horizon}d Horizon Strategy")
        
        return {
            'portfolio_returns': portfolio_df,
            'metrics': metrics,
            'selections': selections
        }
    
    def plot_results(self, portfolio_df: pd.DataFrame, output_path: str = None):
        """Plot backtest results"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Cumulative returns
        cum_returns = (1 + portfolio_df['return_after_cost']).cumprod()
        axes[0].plot(portfolio_df['date'], cum_returns, linewidth=2, label='Strategy')
        axes[0].axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        axes[0].set_title('Cumulative Returns', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Cumulative Return')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Drawdown
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        axes[1].fill_between(portfolio_df['date'], drawdown, 0, alpha=0.3, color='red')
        axes[1].plot(portfolio_df['date'], drawdown, color='red', linewidth=1)
        axes[1].set_title('Drawdown', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Drawdown')
        axes[1].set_xlabel('Date')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            ensure_dir(Path(output_path).parent)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {output_path}")
        
        plt.close()


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    
    from src.utils import load_config
    from src.build_features import load_features, get_feature_columns
    from src.train import load_model
    
    # Load config
    config = utils.load_config('config.yaml')
    
    # Load features
    features_path = f"{config['data']['processed_dir']}/features.parquet"
    df = build_features.load_features(features_path)
    
    # Get feature columns
    feature_cols = build_features.get_feature_columns(df)
    
    # Initialize backtester
    backtester = Backtest(config)
    
    # Run backtests for each horizon
    results = {}
    
    for horizon in config['model']['horizons']:
        print(f"\n{'='*60}")
        print(f"Backtesting {horizon}-day model")
        print(f"{'='*60}")
        
        # Load model
        model_path = f"models/model_{horizon}d.txt"
        model = train.load_model(model_path)
        
        # Run backtest on most recent 2 years
        two_years_ago = pd.Timestamp.now() - pd.DateOffset(years=2)
        
        result = backtester.run_backtest(
            df=df,
            model=model,
            feature_cols=feature_cols,
            horizon=horizon,
            start_date=two_years_ago.strftime("%Y-%m-%d")
        )
        
        results[f'{horizon}d'] = result
        
        # Plot
        if 'portfolio_returns' in result:
            backtester.plot_results(
                result['portfolio_returns'],
                output_path=f"results/backtest_{horizon}d.png"
            )
    
    # Compare horizons
    print(f"\n{'='*60}")
    print("Backtest Comparison")
    print(f"{'='*60}")
    
    comparison_df = pd.DataFrame({
        horizon: result['metrics']
        for horizon, result in results.items()
        if 'metrics' in result
    }).T
    
    print(comparison_df.to_string())
    
    print("\n✅ Backtesting complete!")