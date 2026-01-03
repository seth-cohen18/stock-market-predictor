"""
Generate Portfolio-Based Recommendations
Main script for GitHub Actions - provides actionable portfolio management
"""

import sys
import os
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from portfolio_manager import PortfolioManager
from action_recommender import ActionRecommender
from adaptive_trainer import AdaptiveTrainer
import pandas as pd
import numpy as np


def load_latest_data():
    """Load latest prices and features"""
    print("\n📂 Loading latest market data...")
    
    try:
        # Load prices
        prices = pd.read_parquet('data/raw/prices.parquet')
        print(f"✓ Loaded {len(prices):,} price records")
        
        # Load features
        features = pd.read_parquet('data/processed/features.parquet')
        print(f"✓ Loaded {len(features):,} feature records")
        
        # Get latest date
        latest_date = features['date'].max()
        print(f"✓ Latest data: {latest_date}")
        
        return prices, features, latest_date
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None, None


def load_trained_models():
    """Load the trained models and generate predictions"""
    print("\n🤖 Loading trained models...")
    
    try:
        import lightgbm as lgb
        
        # Load models
        model_1d = lgb.Booster(model_file='models/model_1d.txt')
        model_5d = lgb.Booster(model_file='models/model_5d.txt')
        
        print("✓ Models loaded successfully")
        return model_1d, model_5d
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return None, None


def generate_predictions(features: pd.DataFrame, model_5d) -> pd.DataFrame:
    """Generate predictions for all stocks"""
    print("\n🔮 Generating AI predictions...")
    
    try:
        from build_features import get_feature_columns
        
        # Get latest data for each ticker
        latest_date = features['date'].max()
        latest_features = features[features['date'] == latest_date].copy()
        
        # Get feature columns
        feature_cols = get_feature_columns(features)
        
        # Generate predictions
        X = latest_features[feature_cols].fillna(0)
        predictions = model_5d.predict(X)
        
        # Add to dataframe
        latest_features['predicted_return'] = predictions
        
        print(f"✓ Generated predictions for {len(latest_features)} stocks")
        
        return latest_features[['ticker', 'close', 'predicted_return'] + feature_cols]
        
    except Exception as e:
        print(f"❌ Error generating predictions: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def format_portfolio_report(portfolio_manager, portfolio_df, portfolio_actions,
                            new_opportunities, training_report, summary) -> str:
    """Format complete portfolio report"""
    
    output = []
    output.append("="*70)
    output.append("📊 DAILY PORTFOLIO ANALYSIS & AI RECOMMENDATIONS")
    output.append(f"Date: {datetime.now().strftime('%Y-%m-%d %I:%M %p ET')}")
    output.append("="*70)
    
    # Portfolio Summary
    output.append("\n💼 YOUR PORTFOLIO SUMMARY")
    output.append("-"*70)
    output.append(f"Total Value: ${summary['total_value']:,.2f}")
    output.append(f"  • Invested: ${summary['invested']:,.2f}")
    output.append(f"  • Cash Available: ${summary['cash']:,.2f}")
    output.append(f"  • Total P&L: ${summary['total_pnl']:+,.2f} ({summary['total_pnl_pct']:+.2%})")
    output.append(f"  • Positions: {summary['num_positions']}")
    
    # Current Holdings with Actions
    output.append("\n📈 YOUR HOLDINGS - AI RECOMMENDATIONS")
    output.append("-"*70)
    
    if len(portfolio_df) == 0:
        output.append("\n⚠️  No holdings found. Consider the opportunities below!")
    else:
        for i, action in enumerate(portfolio_actions, 1):
            ticker = action['ticker']
            position = portfolio_df[portfolio_df['ticker'] == ticker].iloc[0]
            
            # Action emoji
            action_emoji = {
                'BUY': '💚',
                'SELL': '🔴',
                'HOLD': '⏸️'
            }.get(action['action'], '•')
            
            output.append(f"\n{i}. {ticker}")
            output.append(f"   Shares: {action['current_shares']} | "
                         f"Cost: ${position['avg_cost']:.2f} | "
                         f"Current: ${position['current_price']:.2f}")
            output.append(f"   Value: ${position['current_value']:,.2f} | "
                         f"P&L: ${position['pnl']:+,.2f} ({position['pnl_pct']:+.2%})")
            
            # AI Recommendation
            output.append(f"   {action_emoji} AI ACTION: {action['action']}", end="")
            if action['quantity'] > 0:
                if action['action'] == 'SELL':
                    output.append(f" {action['quantity']} shares")
                elif action['action'] == 'BUY':
                    output.append(f" {action['quantity']} more shares")
            else:
                output.append("")
            
            output.append(f"   📊 AI Confidence: {action['confidence']:.0f}%")
            output.append(f"   💭 Reasoning: {action['reasoning']}")
            output.append(f"   🎯 Predicted Return: {action['predicted_return']:+.2%}")
    
    # New Opportunities
    if len(new_opportunities) > 0:
        output.append("\n\n💎 NEW OPPORTUNITIES (Stocks You Don't Own)")
        output.append("-"*70)
        
        for i, opp in enumerate(new_opportunities, 1):
            output.append(f"\n{i}. {opp['ticker']}")
            output.append(f"   Current Price: ${opp['price']:.2f}")
            output.append(f"   💚 AI RECOMMENDS: BUY {opp['quantity']} shares "
                         f"(${opp['investment']:,.2f})")
            output.append(f"   📊 AI Confidence: {opp['confidence']:.0f}%")
            output.append(f"   🎯 Expected Return: {opp['predicted_return']:+.2%}")
            output.append(f"   💭 Why: {opp['reasoning']}")
    
    # Portfolio Alerts
    alerts = portfolio_manager.check_position_alerts(portfolio_df)
    if len(alerts) > 0:
        output.append("\n\n⚠️  PORTFOLIO ALERTS")
        output.append("-"*70)
        for alert in alerts:
            output.append(f"  • {alert['message']}")
    
    # AI Training Report
    output.append("\n\n🤖 AI MODEL STATUS - PROOF OF ACTIVE LEARNING")
    output.append("-"*70)
    output.append(training_report)
    
    # Risk Warning
    output.append("\n" + "="*70)
    output.append("⚠️  IMPORTANT DISCLAIMERS")
    output.append("="*70)
    output.append("• These are AI-generated recommendations, not financial advice")
    output.append("• Past performance does not guarantee future results")
    output.append("• Always do your own research before making investment decisions")
    output.append("• Never invest more than you can afford to lose")
    output.append("• Consider consulting a financial advisor for personalized advice")
    
    # Footer
    output.append("\n" + "="*70)
    output.append("📧 Questions? Reply to this email")
    output.append("🔧 Update your portfolio: Edit portfolio.yaml and commit to GitHub")
    output.append("="*70)
    
    return "\n".join(output)


def main():
    """Main execution"""
    
    print("\n" + "="*70)
    print("🚀 PORTFOLIO-BASED STOCK PREDICTOR")
    print("="*70)
    
    try:
        # 1. Load portfolio
        print("\n📂 Step 1: Loading Portfolio Configuration")
        portfolio_manager = PortfolioManager('portfolio.yaml')
        
        if len(portfolio_manager.holdings) == 0:
            print("⚠️  No holdings in portfolio.yaml")
            print("    Add your stocks to portfolio.yaml to get started!")
        
        # 2. Load market data
        print("\n📂 Step 2: Loading Market Data")
        prices, features, latest_date = load_latest_data()
        
        if features is None:
            raise Exception("Could not load market data")
        
        # 3. Load models
        print("\n🤖 Step 3: Loading AI Models")
        model_1d, model_5d = load_trained_models()
        
        if model_5d is None:
            raise Exception("Could not load models")
        
        # 4. Generate predictions
        print("\n🔮 Step 4: Generating AI Predictions")
        predictions_df = generate_predictions(features, model_5d)
        
        if len(predictions_df) == 0:
            raise Exception("Could not generate predictions")
        
        # 5. Update portfolio with current prices
        print("\n💼 Step 5: Updating Portfolio Positions")
        portfolio_df = portfolio_manager.update_with_prices(prices)
        summary = portfolio_manager.get_portfolio_summary(portfolio_df)
        
        print(f"✓ Portfolio value: ${summary['total_value']:,.2f}")
        
        # 6. Generate action recommendations
        print("\n🎯 Step 6: Generating Action Recommendations")
        recommender = ActionRecommender(predictions_df, portfolio_manager)
        
        # Actions for current holdings
        portfolio_actions = recommender.recommend_actions_for_holdings(portfolio_df)
        print(f"✓ Generated {len(portfolio_actions)} holding recommendations")
        
        # New opportunities
        new_opportunities = recommender.recommend_new_opportunities(portfolio_df, top_n=5)
        print(f"✓ Found {len(new_opportunities)} new opportunities")
        
        # 7. Get training report
        print("\n📊 Step 7: Generating Training Report")
        trainer = AdaptiveTrainer()
        training_report = trainer.get_training_report()
        
        # 8. Format complete report
        print("\n📝 Step 8: Formatting Report")
        report = format_portfolio_report(
            portfolio_manager,
            portfolio_df,
            portfolio_actions,
            new_opportunities,
            training_report,
            summary
        )
        
        # 9. Save to file
        with open('recommendations.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("\n✅ Portfolio recommendations saved to recommendations.txt")
        print("\n" + "="*70)
        print(report)
        print("="*70)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        # Create error file
        with open('recommendations.txt', 'w') as f:
            f.write(f"❌ Could not generate portfolio recommendations\n")
            f.write(f"Error: {str(e)}\n")
            f.write(f"Time: {datetime.now()}\n")
        
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)