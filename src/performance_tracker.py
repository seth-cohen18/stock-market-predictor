"""
Performance Tracker - Measure Real-World Model Performance
Logs predictions vs actuals to calculate true accuracy over time
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
from typing import Dict, List, Optional
import matplotlib.pyplot as plt


class PerformanceTracker:
    """Track and analyze model performance over time"""
    
    def __init__(self, log_dir: str = 'data/performance'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_file = self.log_dir / 'predictions.csv'
        
        # Initialize predictions log if doesn't exist
        if not self.predictions_file.exists():
            df = pd.DataFrame(columns=[
                'date', 'ticker', 'horizon', 'predicted_return', 
                'actual_return', 'predicted_direction', 'actual_direction',
                'correct', 'model_type', 'confidence'
            ])
            df.to_csv(self.predictions_file, index=False)
    
    def log_prediction(self, date: str, ticker: str, horizon: int,
                       predicted_return: float, model_type: str = 'single',
                       confidence: float = 0.5):
        """
        Log a prediction (actual return will be filled in later)
        
        Args:
            date: Prediction date (YYYY-MM-DD)
            ticker: Stock ticker
            horizon: Days ahead (1, 5, etc.)
            predicted_return: Predicted return
            model_type: 'single', 'ensemble', 'lgbm', 'xgb', 'rf'
            confidence: Model confidence (0-1)
        """
        df = pd.read_csv(self.predictions_file)
        
        new_row = {
            'date': date,
            'ticker': ticker,
            'horizon': horizon,
            'predicted_return': predicted_return,
            'actual_return': np.nan,  # Fill in later
            'predicted_direction': 1 if predicted_return > 0 else 0,
            'actual_direction': np.nan,
            'correct': np.nan,
            'model_type': model_type,
            'confidence': confidence
        }
        
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(self.predictions_file, index=False)
    
    def update_actuals(self, price_data: pd.DataFrame):
        """
        Update predictions with actual returns once data is available
        
        Args:
            price_data: DataFrame with columns: date, ticker, close
        """
        df = pd.read_csv(self.predictions_file)
        df['date'] = pd.to_datetime(df['date'])
        
        # Only update rows without actuals
        to_update = df[df['actual_return'].isna()].copy()
        
        if len(to_update) == 0:
            return
        
        print(f"Updating {len(to_update)} predictions with actual returns...")
        
        # Merge with price data
        price_data['date'] = pd.to_datetime(price_data['date'])
        
        for idx, row in to_update.iterrows():
            ticker = row['ticker']
            pred_date = row['date']
            horizon = int(row['horizon'])
            
            # Get future date
            future_date = pred_date + timedelta(days=horizon)
            
            # Get prices
            ticker_prices = price_data[price_data['ticker'] == ticker].sort_values('date')
            
            start_price = ticker_prices[ticker_prices['date'] == pred_date]['close'].values
            end_price = ticker_prices[ticker_prices['date'] >= future_date]['close'].values
            
            if len(start_price) > 0 and len(end_price) > 0:
                actual_return = np.log(end_price[0] / start_price[0])
                actual_direction = 1 if actual_return > 0 else 0
                correct = 1 if actual_direction == row['predicted_direction'] else 0
                
                df.loc[idx, 'actual_return'] = actual_return
                df.loc[idx, 'actual_direction'] = actual_direction
                df.loc[idx, 'correct'] = correct
        
        df.to_csv(self.predictions_file, index=False)
        print("✓ Actuals updated")
    
    def get_performance_summary(self, days_back: int = 30) -> Dict:
        """
        Get performance summary for recent predictions
        
        Args:
            days_back: Number of days to look back
            
        Returns:
            Dict with performance metrics
        """
        df = pd.read_csv(self.predictions_file)
        df['date'] = pd.to_datetime(df['date'])
        
        # Filter to recent and complete predictions
        cutoff_date = datetime.now() - timedelta(days=days_back)
        df = df[(df['date'] >= cutoff_date) & (df['correct'].notna())]
        
        if len(df) == 0:
            return {
                'message': 'No complete predictions in time window',
                'total_predictions': 0
            }
        
        # Overall metrics
        total = len(df)
        accuracy = df['correct'].mean()
        
        # By horizon
        by_horizon = df.groupby('horizon').agg({
            'correct': ['mean', 'count'],
            'actual_return': 'mean'
        }).round(4)
        
        # By model type
        by_model = df.groupby('model_type').agg({
            'correct': ['mean', 'count']
        }).round(4)
        
        # Recent trend (last 7 days vs previous)
        last_7 = df[df['date'] >= (datetime.now() - timedelta(days=7))]
        prev_7 = df[(df['date'] >= (datetime.now() - timedelta(days=14))) & 
                    (df['date'] < (datetime.now() - timedelta(days=7)))]
        
        summary = {
            'total_predictions': total,
            'overall_accuracy': accuracy,
            'by_horizon': by_horizon.to_dict(),
            'by_model': by_model.to_dict(),
            'last_7_days': {
                'accuracy': last_7['correct'].mean() if len(last_7) > 0 else None,
                'count': len(last_7)
            },
            'prev_7_days': {
                'accuracy': prev_7['correct'].mean() if len(prev_7) > 0 else None,
                'count': len(prev_7)
            }
        }
        
        return summary
    
    def print_summary(self, days_back: int = 30):
        """Print formatted performance summary"""
        summary = self.get_performance_summary(days_back)
        
        if summary['total_predictions'] == 0:
            print("📊 No predictions available yet")
            return
        
        print("\n" + "="*60)
        print(f"📊 PERFORMANCE SUMMARY (Last {days_back} Days)")
        print("="*60)
        
        print(f"\n📈 Overall:")
        print(f"  Total Predictions: {summary['total_predictions']}")
        print(f"  Accuracy: {summary['overall_accuracy']:.2%}")
        
        print(f"\n⏱️  By Horizon:")
        for horizon, data in summary['by_horizon'].items():
            acc = data['correct']['mean']
            count = int(data['correct']['count'])
            avg_return = data['actual_return']['mean']
            print(f"  {horizon}-day: {acc:.2%} accuracy ({count} predictions, avg return: {avg_return:+.2%})")
        
        print(f"\n🤖 By Model:")
        for model, data in summary['by_model'].items():
            acc = data['correct']['mean']
            count = int(data['correct']['count'])
            print(f"  {model}: {acc:.2%} ({count} predictions)")
        
        if summary['last_7_days']['count'] > 0:
            print(f"\n📅 Recent Trend:")
            last_acc = summary['last_7_days']['accuracy']
            prev_acc = summary['prev_7_days']['accuracy']
            
            print(f"  Last 7 days: {last_acc:.2%} ({summary['last_7_days']['count']} predictions)")
            
            if prev_acc is not None:
                change = last_acc - prev_acc
                emoji = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                print(f"  Previous 7 days: {prev_acc:.2%}")
                print(f"  Change: {emoji} {change:+.2%}")
        
        print("="*60)
    
    def plot_performance(self, days_back: int = 30, save_path: Optional[str] = None):
        """
        Plot performance over time
        
        Args:
            days_back: Days to plot
            save_path: Optional path to save plot
        """
        df = pd.read_csv(self.predictions_file)
        df['date'] = pd.to_datetime(df['date'])
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        df = df[(df['date'] >= cutoff_date) & (df['correct'].notna())]
        
        if len(df) == 0:
            print("No data to plot")
            return
        
        # Rolling accuracy
        df = df.sort_values('date')
        df['rolling_accuracy'] = df['correct'].rolling(20, min_periods=5).mean()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Accuracy over time
        ax1.plot(df['date'], df['rolling_accuracy'], label='20-prediction Rolling Accuracy', linewidth=2)
        ax1.axhline(y=0.5, color='r', linestyle='--', label='Random (50%)')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Accuracy Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Predicted vs Actual Returns
        ax2.scatter(df['predicted_return'], df['actual_return'], alpha=0.5, s=10)
        ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax2.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
        ax2.plot([-0.1, 0.1], [-0.1, 0.1], 'r--', label='Perfect Prediction')
        ax2.set_xlabel('Predicted Return')
        ax2.set_ylabel('Actual Return')
        ax2.set_title('Predicted vs Actual Returns')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")
        else:
            plt.show()
    
    def export_report(self, days_back: int = 30) -> str:
        """
        Export detailed performance report
        
        Returns:
            Path to exported CSV report
        """
        df = pd.read_csv(self.predictions_file)
        df['date'] = pd.to_datetime(df['date'])
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        df = df[(df['date'] >= cutoff_date) & (df['correct'].notna())]
        
        report_path = self.log_dir / f'report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df.to_csv(report_path, index=False)
        
        print(f"✓ Report exported to {report_path}")
        return str(report_path)


def create_daily_snapshot(tracker: PerformanceTracker, 
                          recommendations: List[Dict],
                          horizon: int,
                          model_type: str = 'ensemble'):
    """
    Create daily snapshot of recommendations for tracking
    
    Args:
        tracker: PerformanceTracker instance
        recommendations: List of recommendation dicts
        horizon: Prediction horizon
        model_type: Model type used
    """
    today = datetime.now().strftime('%Y-%m-%d')
    
    for rec in recommendations:
        tracker.log_prediction(
            date=today,
            ticker=rec['ticker'],
            horizon=horizon,
            predicted_return=rec['predicted_return'],
            model_type=model_type,
            confidence=rec.get('prediction_confidence', 0.5)
        )
    
    print(f"✓ Logged {len(recommendations)} predictions for {today}")


if __name__ == "__main__":
    # Test tracker
    tracker = PerformanceTracker()
    tracker.print_summary(30)