"""
Daily Performance Tracker with Graphs
Tracks daily and weekly portfolio performance with visualizations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO
import base64


class PerformanceTracker:
    """Track portfolio performance with daily and weekly metrics"""
    
    def __init__(self):
        self.history_file = 'data/performance/portfolio_history.json'
        self.charts_dir = Path('data/performance/charts')
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        self.load_history()
    
    def load_history(self):
        """Load historical performance data"""
        try:
            with open(self.history_file, 'r') as f:
                self.history = json.load(f)
        except FileNotFoundError:
            self.history = {
                'daily_snapshots': [],
                'trades': [],
                'predictions_tracking': []
            }
    
    def save_history(self):
        """Save performance history"""
        Path(self.history_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def take_snapshot(self, portfolio_df: pd.DataFrame, portfolio_summary: Dict,
                     market_open: bool = False) -> Dict:
        """
        Take a snapshot of portfolio at a point in time
        
        Args:
            portfolio_df: Current portfolio positions
            portfolio_summary: Portfolio summary metrics
            market_open: True for 9:30 AM, False for 4:00 PM
        """
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'date': datetime.now().strftime('%Y-%m-%d'),
            'time': 'open' if market_open else 'close',
            'total_value': portfolio_summary['total_value'],
            'invested': portfolio_summary['invested'],
            'cash': portfolio_summary['cash'],
            'total_pnl': portfolio_summary['total_pnl'],
            'total_pnl_pct': portfolio_summary['total_pnl_pct'],
            'num_positions': portfolio_summary['num_positions'],
            'positions': []
        }
        
        # Add individual positions
        for _, row in portfolio_df.iterrows():
            snapshot['positions'].append({
                'ticker': row['ticker'],
                'shares': row['shares'],
                'price': row['current_price'],
                'value': row['current_value'],
                'pnl': row['pnl'],
                'pnl_pct': row['pnl_pct']
            })
        
        # Add to history
        self.history['daily_snapshots'].append(snapshot)
        
        # Keep only last 90 days
        cutoff_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        self.history['daily_snapshots'] = [
            s for s in self.history['daily_snapshots']
            if s['date'] >= cutoff_date
        ]
        
        self.save_history()
        return snapshot
    
    def calculate_daily_performance(self) -> Dict:
        """Calculate today's performance"""
        
        if len(self.history['daily_snapshots']) < 2:
            return {
                'has_data': False,
                'message': 'Need at least 2 snapshots (open + close) to calculate daily performance'
            }
        
        # Get today's snapshots
        today = datetime.now().strftime('%Y-%m-%d')
        today_snapshots = [s for s in self.history['daily_snapshots'] if s['date'] == today]
        
        if len(today_snapshots) < 2:
            # Get last close and current value
            all_snapshots = sorted(self.history['daily_snapshots'], key=lambda x: x['timestamp'])
            last_snapshot = all_snapshots[-1]
            prev_snapshot = all_snapshots[-2]
            
            opening_value = prev_snapshot['total_value']
            current_value = last_snapshot['total_value']
        else:
            # Use today's open and close
            opening = [s for s in today_snapshots if s['time'] == 'open']
            closing = [s for s in today_snapshots if s['time'] == 'close']
            
            if len(opening) == 0 or len(closing) == 0:
                return {'has_data': False, 'message': 'Missing open or close snapshot'}
            
            opening_value = opening[0]['total_value']
            current_value = closing[-1]['total_value']
        
        # Calculate daily change
        daily_change = current_value - opening_value
        daily_change_pct = (daily_change / opening_value) if opening_value > 0 else 0
        
        # Get position-level changes
        last_snapshot = sorted(self.history['daily_snapshots'], key=lambda x: x['timestamp'])[-1]
        
        position_changes = []
        for pos in last_snapshot['positions']:
            # Find previous price (simplified - would need more history for exact open price)
            position_changes.append({
                'ticker': pos['ticker'],
                'current_value': pos['value'],
                'pnl_today': pos['pnl'],  # This is total P&L, would need open price for daily
                'pnl_pct_today': pos['pnl_pct']
            })
        
        return {
            'has_data': True,
            'date': today,
            'opening_value': opening_value,
            'closing_value': current_value,
            'daily_change': daily_change,
            'daily_change_pct': daily_change_pct,
            'positions': position_changes
        }
    
    def calculate_weekly_performance(self) -> Dict:
        """Calculate last 7 days performance"""
        
        if len(self.history['daily_snapshots']) < 2:
            return {'has_data': False, 'message': 'Not enough history'}
        
        # Get snapshots from last 7 days
        week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        recent = sorted(
            [s for s in self.history['daily_snapshots'] if s['date'] >= week_ago],
            key=lambda x: x['timestamp']
        )
        
        if len(recent) < 2:
            return {'has_data': False, 'message': 'Not enough weekly history'}
        
        week_start_value = recent[0]['total_value']
        week_end_value = recent[-1]['total_value']
        
        weekly_change = week_end_value - week_start_value
        weekly_change_pct = (weekly_change / week_start_value) if week_start_value > 0 else 0
        
        return {
            'has_data': True,
            'start_date': recent[0]['date'],
            'end_date': recent[-1]['date'],
            'start_value': week_start_value,
            'end_value': week_end_value,
            'weekly_change': weekly_change,
            'weekly_change_pct': weekly_change_pct,
            'num_days': len(set(s['date'] for s in recent))
        }
    
    def create_performance_chart(self, chart_type: str = 'daily') -> str:
        """
        Create performance chart and return as base64 image
        
        Args:
            chart_type: 'daily' for today, 'weekly' for last 7 days
        
        Returns:
            Base64 encoded PNG image
        """
        if chart_type == 'daily':
            return self._create_daily_chart()
        elif chart_type == 'weekly':
            return self._create_weekly_chart()
        elif chart_type == 'positions':
            return self._create_positions_chart()
        else:
            return ""
    
    def _create_daily_chart(self) -> str:
        """Create today's performance line chart"""
        
        # Get today's data points
        today = datetime.now().strftime('%Y-%m-%d')
        today_data = [s for s in self.history['daily_snapshots'] if s['date'] == today]
        
        if len(today_data) < 2:
            return ""
        
        # Extract timestamps and values
        times = [datetime.fromisoformat(s['timestamp']) for s in today_data]
        values = [s['total_value'] for s in today_data]
        
        # Create chart
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(times, values, marker='o', linewidth=2, markersize=8, color='#2ecc71')
        ax.fill_between(times, values, alpha=0.3, color='#2ecc71')
        
        ax.set_title(f"Today's Portfolio Value - {today}", fontsize=14, fontweight='bold')
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%I:%M %p'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return image_base64
    
    def _create_weekly_chart(self) -> str:
        """Create weekly performance chart"""
        
        # Get last 7 days
        week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        weekly_data = sorted(
            [s for s in self.history['daily_snapshots'] if s['date'] >= week_ago],
            key=lambda x: x['timestamp']
        )
        
        if len(weekly_data) < 2:
            return ""
        
        # Get one snapshot per day (closing value)
        daily_closes = {}
        for s in weekly_data:
            date = s['date']
            if date not in daily_closes or s['time'] == 'close':
                daily_closes[date] = s['total_value']
        
        dates = sorted(daily_closes.keys())
        values = [daily_closes[d] for d in dates]
        
        # Create chart
        fig, ax = plt.subplots(figsize=(10, 5))
        
        date_objs = [datetime.strptime(d, '%Y-%m-%d') for d in dates]
        colors = ['#2ecc71' if i == 0 or values[i] >= values[i-1] else '#e74c3c' 
                 for i in range(len(values))]
        
        ax.bar(date_objs, values, color=colors, alpha=0.7, edgecolor='black')
        ax.plot(date_objs, values, marker='o', linewidth=2, markersize=8, color='#34495e')
        
        ax.set_title('Weekly Portfolio Value', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Format axes
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return image_base64
    
    def _create_positions_chart(self) -> str:
        """Create position performance breakdown"""
        
        if len(self.history['daily_snapshots']) == 0:
            return ""
        
        latest = self.history['daily_snapshots'][-1]
        positions = latest['positions']
        
        if len(positions) == 0:
            return ""
        
        # Extract data
        tickers = [p['ticker'] for p in positions]
        pnl_pcts = [p['pnl_pct'] * 100 for p in positions]
        colors = ['#2ecc71' if pnl >= 0 else '#e74c3c' for pnl in pnl_pcts]
        
        # Create chart
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(tickers, pnl_pcts, color=colors, alpha=0.7, edgecolor='black')
        
        ax.set_title('Position Performance (P&L %)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Gain/Loss (%)', fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, pnl_pcts)):
            ax.text(val, i, f' {val:+.1f}%', va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return image_base64
    
    def track_prediction_accuracy(self, predictions_df: pd.DataFrame, 
                                  actual_prices: pd.DataFrame):
        """Track how well predictions matched reality"""
        
        # Compare predictions to actual outcomes
        # This would need enhancement to store predictions and compare later
        pass


if __name__ == "__main__":
    print("Performance Tracker Module")
    print("Tracks daily and weekly portfolio performance with charts")