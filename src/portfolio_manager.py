"""
Portfolio Manager - Track holdings, performance, and P&L
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import yaml


class PortfolioManager:
    """Manage portfolio holdings and track performance"""
    
    def __init__(self, portfolio_file: str = 'portfolio.yaml'):
        """Load portfolio configuration"""
        self.portfolio_file = portfolio_file
        self.load_portfolio()
    
    def load_portfolio(self):
        """Load portfolio from YAML file"""
        try:
            with open(self.portfolio_file, 'r') as f:
                config = yaml.safe_load(f)
            
            self.cash = config.get('cash', 0)
            self.holdings = config.get('holdings', {})
            self.risk_profile = config.get('risk_profile', {})
            self.preferences = config.get('preferences', {})
            
            print(f"✓ Portfolio loaded: {len(self.holdings)} positions")
        except FileNotFoundError:
            print(f"⚠️  Portfolio file not found: {self.portfolio_file}")
            self.cash = 0
            self.holdings = {}
            self.risk_profile = {}
            self.preferences = {}
    
    def get_holdings_tickers(self) -> List[str]:
        """Get list of tickers in portfolio"""
        return list(self.holdings.keys())
    
    def update_with_prices(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Update portfolio with current prices
        
        Args:
            prices_df: DataFrame with latest prices (must have 'ticker', 'close')
        
        Returns:
            DataFrame with portfolio analysis
        """
        if len(self.holdings) == 0:
            return pd.DataFrame()
        
        # Get latest prices for holdings
        latest_date = prices_df['date'].max()
        current_prices = prices_df[prices_df['date'] == latest_date].copy()
        
        portfolio_data = []
        
        for ticker, position in self.holdings.items():
            shares = position['shares']
            avg_cost = position['avg_cost']
            
            # Get current price
            ticker_price = current_prices[current_prices['ticker'] == ticker]
            
            if len(ticker_price) == 0:
                print(f"⚠️  No price data for {ticker}")
                current_price = avg_cost  # Use avg cost as fallback
            else:
                current_price = ticker_price['close'].iloc[0]
            
            # Calculate metrics
            cost_basis = shares * avg_cost
            current_value = shares * current_price
            pnl = current_value - cost_basis
            pnl_pct = (pnl / cost_basis) if cost_basis > 0 else 0
            
            portfolio_data.append({
                'ticker': ticker,
                'shares': shares,
                'avg_cost': avg_cost,
                'current_price': current_price,
                'cost_basis': cost_basis,
                'current_value': current_value,
                'pnl': pnl,
                'pnl_pct': pnl_pct
            })
        
        portfolio_df = pd.DataFrame(portfolio_data)
        
        # Add portfolio weights
        total_value = portfolio_df['current_value'].sum()
        portfolio_df['weight'] = portfolio_df['current_value'] / total_value
        
        return portfolio_df
    
    def get_portfolio_summary(self, portfolio_df: pd.DataFrame) -> Dict:
        """Get overall portfolio summary"""
        
        if len(portfolio_df) == 0:
            return {
                'total_value': self.cash,
                'invested': 0,
                'cash': self.cash,
                'total_pnl': 0,
                'total_pnl_pct': 0,
                'num_positions': 0
            }
        
        invested = portfolio_df['current_value'].sum()
        total_value = invested + self.cash
        total_pnl = portfolio_df['pnl'].sum()
        total_cost = portfolio_df['cost_basis'].sum()
        total_pnl_pct = (total_pnl / total_cost) if total_cost > 0 else 0
        
        return {
            'total_value': total_value,
            'invested': invested,
            'cash': self.cash,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,
            'num_positions': len(portfolio_df)
        }
    
    def check_position_alerts(self, portfolio_df: pd.DataFrame) -> List[Dict]:
        """Check for position-based alerts (stop loss, take profit, etc.)"""
        
        alerts = []
        
        if len(portfolio_df) == 0:
            return alerts
        
        # Get thresholds from preferences
        take_profit = self.preferences.get('take_profit_threshold', 0.25)
        stop_loss = self.preferences.get('stop_loss_threshold', -0.15)
        rebalance = self.preferences.get('rebalance_threshold', 0.10)
        max_position = self.risk_profile.get('max_position_size', 0.20)
        
        for _, row in portfolio_df.iterrows():
            ticker = row['ticker']
            pnl_pct = row['pnl_pct']
            weight = row['weight']
            
            # Take profit alert
            if pnl_pct >= take_profit:
                alerts.append({
                    'ticker': ticker,
                    'type': 'TAKE_PROFIT',
                    'message': f"{ticker}: Up {pnl_pct:.1%}, consider taking profits",
                    'severity': 'info'
                })
            
            # Stop loss alert
            if pnl_pct <= stop_loss:
                alerts.append({
                    'ticker': ticker,
                    'type': 'STOP_LOSS',
                    'message': f"{ticker}: Down {pnl_pct:.1%}, consider cutting losses",
                    'severity': 'warning'
                })
            
            # Position size alert
            if weight > max_position:
                alerts.append({
                    'ticker': ticker,
                    'type': 'OVERWEIGHT',
                    'message': f"{ticker}: {weight:.1%} of portfolio (max {max_position:.0%}), consider trimming",
                    'severity': 'info'
                })
        
        return alerts
    
    def calculate_buying_power(self, portfolio_df: pd.DataFrame) -> Dict:
        """Calculate available buying power and constraints"""
        
        total_value = self.get_portfolio_summary(portfolio_df)['total_value']
        
        # Max amount that can be allocated to new position
        max_position_size = self.risk_profile.get('max_position_size', 0.20)
        min_position_size = self.risk_profile.get('min_position_size', 0.05)
        
        max_new_position = total_value * max_position_size
        min_new_position = total_value * min_position_size
        
        return {
            'available_cash': self.cash,
            'max_new_position': max_new_position,
            'min_new_position': min_new_position,
            'total_portfolio_value': total_value,
            'can_buy': self.cash >= min_new_position
        }
    
    def save_portfolio(self, portfolio_df: pd.DataFrame = None):
        """Save current portfolio state (for future: track trades)"""
        # This could be enhanced to save portfolio history
        # For now, portfolio.yaml is the source of truth
        pass


if __name__ == "__main__":
    # Test portfolio manager
    manager = PortfolioManager()
    
    print("\n📊 Portfolio Configuration:")
    print(f"Cash: ${manager.cash:,.2f}")
    print(f"Holdings: {manager.holdings}")
    print(f"Risk Profile: {manager.risk_profile}")
    print(f"Preferences: {manager.preferences}")