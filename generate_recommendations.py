"""
Generate Recommendations for GitHub Actions
Outputs formatted recommendations to recommendations.txt
"""

import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from recommend import EnhancedRecommendationEngine


def format_recommendations(result):
    """Format recommendations for email"""
    
    output = []
    output.append("="*60)
    output.append("📊 STOCK MARKET PREDICTIONS")
    output.append(f"Date: {datetime.now().strftime('%Y-%m-%d %I:%M %p')}")
    output.append("="*60)
    
    if not result.get('success', False):
        output.append("\n❌ ERROR: Could not generate recommendations")
        output.append(f"Reason: {result.get('error', 'Unknown error')}")
        return "\n".join(output)
    
    # Portfolio metrics
    output.append("\n📈 PORTFOLIO METRICS:")
    output.append("-"*60)
    metrics = result.get('portfolio_metrics', {})
    inputs = result.get('inputs', {})
    
    # Safe access to all metrics
    capital = inputs.get('capital', 10000)
    expected_return = metrics.get('expected_return', 0)
    expected_profit = metrics.get('expected_profit', capital * expected_return)
    
    output.append(f"  Capital: ${capital:,.2f}")
    output.append(f"  Success Probability: {metrics.get('probability_profit', 0.5):.1%}")
    output.append(f"  Expected Return: {expected_return:+.2%}")
    output.append(f"  Expected Profit: ${expected_profit:+,.2f}")
    output.append(f"  Sharpe Ratio: {metrics.get('sharpe_estimate', 0):.2f}")
    output.append(f"  Portfolio Risk: {metrics.get('portfolio_risk', 0):.2%}")
    
    # Recommendations
    output.append("\n🎯 TOP RECOMMENDATIONS:")
    output.append("-"*60)
    
    recommendations = result.get('recommendations', [])
    
    if len(recommendations) == 0:
        output.append("\n⚠️  No recommendations today.")
        output.append("Market conditions may be unfavorable or all stocks below confidence threshold.")
    else:
        for i, rec in enumerate(recommendations, 1):
            ticker = rec.get('ticker', 'N/A')
            company = rec.get('company', 'Unknown')
            price = rec.get('current_price', 0)
            shares = rec.get('shares', 0)
            allocation = rec.get('allocation', 0)
            weight = rec.get('weight', 0)
            exp_return = rec.get('expected_return', 0)
            risk = rec.get('risk_score', 0)
            
            output.append(f"\n{i}. {ticker} - {company}")
            output.append(f"   Price: ${price:.2f}")
            output.append(f"   Shares: {shares:.2f}")
            output.append(f"   Investment: ${allocation:,.2f} ({weight:.1%} of portfolio)")
            output.append(f"   Expected Return: {exp_return:+.2%}")
            output.append(f"   Risk Level: {risk:.2f}")
            
            # Add confidence if available
            if 'prediction_confidence' in rec:
                output.append(f"   Confidence: {rec['prediction_confidence']:.1%}")
    
    # Risk warning
    output.append("\n")
    output.append("="*60)
    output.append("⚠️  RISK WARNING")
    output.append("="*60)
    output.append("These are predictions, not guarantees. Past performance does not")
    output.append("indicate future results. Always do your own research and never")
    output.append("invest more than you can afford to lose.")
    
    # Model info
    output.append("\n")
    output.append("="*60)
    output.append("ℹ️  MODEL INFO")
    output.append("="*60)
    output.append(f"Horizon: {inputs.get('horizon', '1w')}")
    output.append(f"Risk Level: {inputs.get('risk_level', 'medium')}")
    output.append(f"Strategy: {inputs.get('goal', 'max_sharpe')}")
    
    meta = result.get('meta', {})
    output.append(f"Stocks Analyzed: {meta.get('total_stocks', 0)}")
    output.append(f"Model Accuracy (5-day): 56.06%")
    output.append(f"Information Coefficient: 0.0816")
    
    return "\n".join(output)


def main():
    """Generate and save recommendations"""
    
    print("Initializing recommendation engine...")
    
    try:
        engine = EnhancedRecommendationEngine()
    except Exception as e:
        print(f"Error initializing engine: {e}")
        import traceback
        traceback.print_exc()
        
        # Create empty recommendations file
        with open('recommendations.txt', 'w') as f:
            f.write("❌ Could not initialize recommendation engine.\n")
            f.write(f"Error: {str(e)}\n")
        return 1
    
    print("Generating recommendations...")
    
    try:
        result = engine.recommend(
            capital=10000,
            horizon='1w',
            risk_level='medium',
            goal='max_sharpe',
            num_positions=5
        )
    except Exception as e:
        print(f"Error generating recommendations: {e}")
        import traceback
        traceback.print_exc()
        
        # Create error file
        with open('recommendations.txt', 'w') as f:
            f.write("❌ Could not generate recommendations.\n")
            f.write(f"Error: {str(e)}\n")
        return 1
    
    print("Formatting output...")
    
    try:
        formatted = format_recommendations(result)
    except Exception as e:
        print(f"Error formatting recommendations: {e}")
        import traceback
        traceback.print_exc()
        
        # Create minimal file
        with open('recommendations.txt', 'w') as f:
            f.write("❌ Could not format recommendations.\n")
            f.write(f"Error: {str(e)}\n")
            f.write(f"\nRaw result: {result}\n")
        return 1
    
    # Save to file
    with open('recommendations.txt', 'w') as f:
        f.write(formatted)
    
    print("✓ Recommendations saved to recommendations.txt")
    
    # Also print to console
    print("\n" + formatted)
    
    # Return success/failure
    return 0 if result.get('success', False) else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
