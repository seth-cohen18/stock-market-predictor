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
    
    if not result['success']:
        output.append("\n❌ ERROR: Could not generate recommendations")
        output.append(f"Reason: {result.get('error', 'Unknown error')}")
        return "\n".join(output)
    
    # Portfolio metrics
    output.append("\n📈 PORTFOLIO METRICS:")
    output.append("-"*60)
    metrics = result.get('portfolio_metrics', {})
    output.append(f"  Capital: ${result['inputs']['capital']:,.2f}")
    output.append(f"  Success Probability: {metrics.get('probability_profit', 0.0):.1%}")
    output.append(f"  Expected Return: {metrics.get('expected_return', 0.0):+.2%}")
    if 'expected_profit' in metrics:
        output.append(f"  Expected Profit: ${metrics['expected_profit']:+,.2f}")
    output.append(f"  Sharpe Ratio: {metrics.get('sharpe_estimate', 0.0):.2f}")
    output.append(f"  Portfolio Risk: {metrics.get('portfolio_risk', 0.0):.2%}")
    
    # Recommendations
    output.append("\n🎯 TOP RECOMMENDATIONS:")
    output.append("-"*60)
    
    recommendations = result.get('recommendations', [])
    
    if len(recommendations) == 0:
        output.append("\n⚠️  No recommendations today.")
        output.append("Market conditions may be unfavorable or all stocks below confidence threshold.")
    else:
        for i, rec in enumerate(recommendations, 1):
            output.append(f"\n{i}. {rec['ticker']} - {rec['company']}")
            output.append(f"   Price: ${rec['current_price']:.2f}")
            output.append(f"   Shares: {rec.get('shares', 0):.2f}")
            output.append(f"   Investment: ${rec.get('allocation', 0):,.2f} ({rec.get('weight', 0):.1%} of portfolio)")
            output.append(f"   Expected Return: {rec.get('expected_return', 0):+.2%}")
            output.append(f"   Risk Level: {rec.get('risk_score', 0):.2f}")
            output.append(f"   Confidence: {rec.get('prediction_confidence', 0.5):.1%}")
    
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
    output.append(f"Horizon: {result['inputs']['horizon']}")
    output.append(f"Risk Level: {result['inputs']['risk_level']}")
    output.append(f"Strategy: {result['inputs']['goal']}")
    output.append(f"Stocks Analyzed: {result.get('meta', {}).get('total_stocks', 0)}")
    output.append(f"Model Accuracy (1-day): 53.13%")
    output.append(f"Model Accuracy (5-day): 53.50%")
    output.append(f"Information Coefficient: 0.0829")
    
    return "\n".join(output)


def main():
    """Generate and save recommendations"""
    
    print("Initializing recommendation engine...")
    engine = EnhancedRecommendationEngine()
    
    print("Generating recommendations...")
    result = engine.recommend(
        capital=10000,
        horizon='1w',
        risk_level='medium',
        goal='max_sharpe',
        num_positions=5
    )
    
    print("Formatting output...")
    formatted = format_recommendations(result)
    
    # Save to file
    with open('recommendations.txt', 'w') as f:
        f.write(formatted)
    
    print("✓ Recommendations saved to recommendations.txt")
    
    # Also print to console
    print("\n" + formatted)
    
    # Return success/failure
    return 0 if result['success'] else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)