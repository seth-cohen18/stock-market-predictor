"""
Performance Tracking Script
View real-world model performance and accuracy
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from performance_tracker import PerformanceTracker
from build_features import load_features
import pandas as pd
import argparse


def main():
    """View performance statistics"""
    
    parser = argparse.ArgumentParser(description='View model performance')
    parser.add_argument('--days', type=int, default=30, help='Days to look back (default: 30)')
    parser.add_argument('--update', action='store_true', help='Update predictions with latest price data')
    parser.add_argument('--plot', action='store_true', help='Generate performance plots')
    parser.add_argument('--export', action='store_true', help='Export detailed report')
    
    args = parser.parse_args()
    
    tracker = PerformanceTracker()
    
    # Update actuals if requested
    if args.update:
        print("Updating predictions with actual returns...")
        try:
            prices = pd.read_parquet('data/raw/prices.parquet')
            tracker.update_actuals(prices)
        except FileNotFoundError:
            print("❌ Price data not found. Run main.py or main_ensemble.py first.")
            return
    
    # Print summary
    tracker.print_summary(days_back=args.days)
    
    # Generate plot if requested
    if args.plot:
        print("\nGenerating performance plot...")
        plot_path = f'data/performance/performance_plot.png'
        tracker.plot_performance(days_back=args.days, save_path=plot_path)
    
    # Export report if requested
    if args.export:
        print("\nExporting detailed report...")
        report_path = tracker.export_report(days_back=args.days)
        print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()