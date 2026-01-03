"""
Action Recommender - Generate BUY/SELL/HOLD recommendations with AI reasoning
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime


class ActionRecommender:
    """Generate actionable recommendations for portfolio"""
    
    def __init__(self, predictions_df: pd.DataFrame, portfolio_manager):
        """
        Initialize with predictions and portfolio context
        
        Args:
            predictions_df: DataFrame with model predictions
            portfolio_manager: PortfolioManager instance
        """
        self.predictions_df = predictions_df
        self.portfolio_manager = portfolio_manager
        self.preferences = portfolio_manager.preferences
    
    def recommend_actions_for_holdings(self, portfolio_df: pd.DataFrame) -> List[Dict]:
        """
        Generate BUY/SELL/HOLD recommendations for current holdings
        
        Returns:
            List of action recommendations with reasoning
        """
        recommendations = []
        
        for _, position in portfolio_df.iterrows():
            ticker = position['ticker']
            
            # Get prediction for this ticker
            pred = self.predictions_df[self.predictions_df['ticker'] == ticker]
            
            if len(pred) == 0:
                # No prediction available
                recommendations.append({
                    'ticker': ticker,
                    'action': 'HOLD',
                    'action_type': 'NO_DATA',
                    'confidence': 0.5,
                    'reasoning': 'No prediction data available',
                    'quantity': 0,
                    'current_shares': position['shares'],
                    'pnl_pct': position['pnl_pct']
                })
                continue
            
            pred = pred.iloc[0]
            
            # Get prediction values
            predicted_return = pred.get('predicted_return', 0)
            confidence = abs(predicted_return)  # Use magnitude as confidence
            
            # Current position metrics
            pnl_pct = position['pnl_pct']
            weight = position['weight']
            shares = position['shares']
            
            # Decision logic
            action = self._determine_action(
                predicted_return=predicted_return,
                pnl_pct=pnl_pct,
                weight=weight,
                confidence=confidence
            )
            
            # Generate reasoning
            reasoning = self._generate_reasoning(
                action=action,
                predicted_return=predicted_return,
                pnl_pct=pnl_pct,
                weight=weight,
                ticker=ticker
            )
            
            # Calculate quantity recommendation
            quantity = self._calculate_quantity(
                action=action,
                shares=shares,
                predicted_return=predicted_return,
                weight=weight
            )
            
            recommendations.append({
                'ticker': ticker,
                'action': action['action'],
                'action_type': action['type'],
                'confidence': min(confidence * 100, 99),  # Convert to percentage
                'reasoning': reasoning,
                'quantity': quantity,
                'current_shares': shares,
                'predicted_return': predicted_return,
                'pnl_pct': pnl_pct,
                'weight': weight
            })
        
        return recommendations
    
    def recommend_new_opportunities(self, portfolio_df: pd.DataFrame, 
                                   top_n: int = 5) -> List[Dict]:
        """
        Find new stocks to buy (not in portfolio)
        
        Args:
            portfolio_df: Current portfolio
            top_n: Number of opportunities to return
        
        Returns:
            List of BUY recommendations
        """
        # Get tickers not in portfolio
        current_tickers = set(portfolio_df['ticker'].tolist())
        available = self.predictions_df[~self.predictions_df['ticker'].isin(current_tickers)].copy()
        
        if len(available) == 0:
            return []
        
        # Filter by minimum confidence
        min_confidence = self.preferences.get('min_confidence', 0.65)
        
        # Only positive predictions
        available = available[available['predicted_return'] > 0].copy()
        
        # Sort by predicted return
        available = available.sort_values('predicted_return', ascending=False)
        
        # Get buying power constraints
        buying_power = self.portfolio_manager.calculate_buying_power(portfolio_df)
        
        opportunities = []
        
        for _, pred in available.head(top_n).iterrows():
            ticker = pred['ticker']
            predicted_return = pred['predicted_return']
            confidence = min(abs(predicted_return) * 100, 99)
            
            # Skip if confidence too low
            if confidence / 100 < min_confidence:
                continue
            
            # Calculate recommended investment
            current_price = pred.get('close', pred.get('current_price', 100))
            max_investment = buying_power['max_new_position']
            
            # Use 5-15% of portfolio for new positions
            target_investment = buying_power['total_portfolio_value'] * 0.10
            target_investment = min(target_investment, buying_power['available_cash'])
            
            shares = int(target_investment / current_price)
            
            if shares == 0:
                continue
            
            opportunities.append({
                'ticker': ticker,
                'action': 'BUY',
                'action_type': 'NEW_POSITION',
                'confidence': confidence,
                'reasoning': self._generate_buy_reasoning(pred, predicted_return),
                'quantity': shares,
                'price': current_price,
                'investment': shares * current_price,
                'predicted_return': predicted_return
            })
        
        return opportunities
    
    def _determine_action(self, predicted_return: float, pnl_pct: float,
                         weight: float, confidence: float) -> Dict:
        """Determine action based on multiple signals"""
        
        # Thresholds
        strong_sell_threshold = -0.05  # -5% predicted
        weak_sell_threshold = -0.02    # -2% predicted
        weak_buy_threshold = 0.02      # +2% predicted
        strong_buy_threshold = 0.05    # +5% predicted
        
        take_profit = self.preferences.get('take_profit_threshold', 0.25)
        stop_loss = self.preferences.get('stop_loss_threshold', -0.15)
        max_position = self.portfolio_manager.risk_profile.get('max_position_size', 0.20)
        
        # SELL conditions
        if predicted_return < strong_sell_threshold:
            return {'action': 'SELL', 'type': 'STRONG_NEGATIVE'}
        
        if pnl_pct <= stop_loss:
            return {'action': 'SELL', 'type': 'STOP_LOSS'}
        
        if pnl_pct >= take_profit and predicted_return < weak_buy_threshold:
            return {'action': 'SELL', 'type': 'TAKE_PROFIT'}
        
        if weight > max_position:
            return {'action': 'SELL', 'type': 'TRIM_OVERWEIGHT'}
        
        # BUY MORE conditions
        if predicted_return > strong_buy_threshold and weight < max_position:
            return {'action': 'BUY', 'type': 'ADD_STRONG'}
        
        if predicted_return > weak_buy_threshold and pnl_pct < 0:
            return {'action': 'BUY', 'type': 'AVERAGE_DOWN'}
        
        # HOLD (default)
        return {'action': 'HOLD', 'type': 'NEUTRAL'}
    
    def _generate_reasoning(self, action: Dict, predicted_return: float,
                           pnl_pct: float, weight: float, ticker: str) -> str:
        """Generate human-readable reasoning"""
        
        action_type = action['type']
        
        reasons = {
            'STRONG_NEGATIVE': f"AI predicts {predicted_return:.1%} decline, high conviction sell",
            'STOP_LOSS': f"Position down {pnl_pct:.1%}, cut losses before further decline",
            'TAKE_PROFIT': f"Up {pnl_pct:.1%}, lock in gains (AI predicts slowing momentum)",
            'TRIM_OVERWEIGHT': f"Position is {weight:.1%} of portfolio, reduce concentration risk",
            'ADD_STRONG': f"AI predicts {predicted_return:.1%} upside, add to winning position",
            'AVERAGE_DOWN': f"Strong predicted rebound ({predicted_return:.1%}), buy the dip",
            'NEUTRAL': f"AI predicts {predicted_return:.1%}, maintain current position"
        }
        
        return reasons.get(action_type, f"Hold and monitor {ticker}")
    
    def _generate_buy_reasoning(self, pred: pd.Series, predicted_return: float) -> str:
        """Generate reasoning for new buy opportunity"""
        
        reasons = []
        
        # Predicted return
        reasons.append(f"AI forecasts {predicted_return:.1%} return")
        
        # Add technical signals if available
        if 'momentum_20' in pred and pred['momentum_20'] > 0:
            reasons.append("positive momentum")
        
        if 'sector_return_20d' in pred and pred['sector_return_20d'] > 0:
            reasons.append("strong sector performance")
        
        if 'zscore_20' in pred and pred['zscore_20'] < -1:
            reasons.append("oversold entry point")
        
        return ", ".join(reasons)
    
    def _calculate_quantity(self, action: Dict, shares: int,
                           predicted_return: float, weight: float) -> int:
        """Calculate recommended quantity for action"""
        
        action_type = action['type']
        
        if action['action'] == 'SELL':
            # Sell strategies
            if action_type in ['STRONG_NEGATIVE', 'STOP_LOSS']:
                return shares  # Sell all
            elif action_type == 'TAKE_PROFIT':
                return int(shares * 0.5)  # Sell half
            elif action_type == 'TRIM_OVERWEIGHT':
                return int(shares * 0.25)  # Sell 25%
        
        elif action['action'] == 'BUY':
            # Buy strategies
            if action_type == 'ADD_STRONG':
                return int(shares * 0.25)  # Add 25% more
            elif action_type == 'AVERAGE_DOWN':
                return int(shares * 0.20)  # Add 20% more
        
        return 0  # HOLD


if __name__ == "__main__":
    print("Action Recommender Module")
    print("Use with portfolio_manager and predictions")