"""
Interactive CLI for getting stock recommendations
Makes it easy to use without remembering command-line arguments
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from recommend import RecommendationEngine


def get_user_input():
    """Interactive prompts to get user inputs"""
    print("\n" + "="*70)
    print("     STOCK MARKET PREDICTION SYSTEM - INTERACTIVE MODE")
    print("="*70)
    
    # Capital
    while True:
        try:
            capital = input("\n💰 How much do you want to invest? ($): ")
            capital = float(capital.replace(',', '').replace('$', ''))
            if capital <= 0:
                print("   Please enter a positive amount")
                continue
            break
        except ValueError:
            print("   Please enter a valid number")
    
    # Horizon
    print("\n⏱️  Investment time horizon:")
    print("   1. One day (1d) - Very short-term")
    print("   2. One week (1w) - Short-term")
    print("   3. One month (1m) - Medium-term")
    
    horizon_map = {'1': '1d', '2': '1w', '3': '1m'}
    while True:
        choice = input("\n   Choose (1-3): ").strip()
        if choice in horizon_map:
            horizon = horizon_map[choice]
            break
        print("   Please choose 1, 2, or 3")
    
    # Risk level
    print("\n🎲 Risk tolerance:")
    print("   1. Low - Prefer stable stocks")
    print("   2. Medium - Balanced approach")
    print("   3. High - Accept volatility for potential returns")
    
    risk_map = {'1': 'low', '2': 'medium', '3': 'high'}
    while True:
        choice = input("\n   Choose (1-3): ").strip()
        if choice in risk_map:
            risk_level = risk_map[choice]
            break
        print("   Please choose 1, 2, or 3")
    
    # Goal
    print("\n🎯 Investment goal:")
    print("   1. Maximize expected return")
    print("   2. Maximize risk-adjusted return (recommended)")
    print("   3. Target a specific ROI percentage")
    
    goal_map = {'1': 'max_return', '2': 'max_sharpe', '3': 'prob_target'}
    while True:
        choice = input("\n   Choose (1-3): ").strip()
        if choice in goal_map:
            goal = goal_map[choice]
            break
        print("   Please choose 1, 2, or 3")
    
    # Target ROI if needed
    target_roi = None
    if goal == 'prob_target':
        while True:
            try:
                target = input("\n   What's your target return? (e.g., 3 for 3%): ")
                target_roi = float(target) / 100
                if target_roi <= 0 or target_roi > 1:
                    print("   Please enter a reasonable percentage (0-100)")
                    continue
                break
            except ValueError:
                print("   Please enter a valid number")
    
    # Number of positions
    print("\n📊 Number of stock positions:")
    while True:
        try:
            choice = input("   How many stocks? (1-10, default 5): ").strip()
            if not choice:
                num_positions = 5
                break
            num_positions = int(choice)
            if 1 <= num_positions <= 10:
                break
            print("   Please choose between 1 and 10")
        except ValueError:
            print("   Please enter a valid number")
    
    return {
        'capital': capital,
        'horizon': horizon,
        'risk_level': risk_level,
        'goal': goal,
        'target_roi': target_roi,
        'num_positions': num_positions
    }


def main():
    """Main interactive loop"""
    try:
        print("\nInitializing recommendation engine...")
        engine = RecommendationEngine()
        
        while True:
            # Get user inputs
            params = get_user_input()
            
            # Generate recommendations
            print("\n⏳ Generating recommendations...")
            result = engine.recommend(**params)
            
            # Print results
            engine.print_recommendations(result)
            
            # Ask if user wants to try again
            print("\n" + "="*70)
            again = input("\nWould you like to get another recommendation? (y/n): ")
            if again.lower() not in ['y', 'yes']:
                break
            print("\n")
        
        print("\n✅ Thank you for using the Stock Market Prediction System!")
        print("   Remember: These are predictions, not guarantees. Trade wisely!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Goodbye!")
    except FileNotFoundError as e:
        print(f"\n❌ Error: Model files not found.")
        print("   Please run 'python main.py' first to train the models.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()