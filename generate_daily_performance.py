"""
End of Day Performance Report
Generates daily performance report with charts at market close
"""

import sys
import os
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from portfolio_manager import PortfolioManager
from performance_tracker_daily import PerformanceTracker
import pandas as pd


def load_data():
    """Load latest market data"""
    try:
        prices = pd.read_parquet('data/raw/prices.parquet')
        return prices
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def format_performance_report(daily_perf: dict, weekly_perf: dict,
                              portfolio_df: pd.DataFrame, 
                              chart_daily: str, chart_weekly: str,
                              chart_positions: str) -> str:
    """Format HTML performance report with embedded charts"""
    
    # Determine if positive/negative
    daily_color = '#2ecc71' if daily_perf.get('daily_change', 0) >= 0 else '#e74c3c'
    daily_emoji = '🟢' if daily_perf.get('daily_change', 0) >= 0 else '🔴'
    weekly_color = '#2ecc71' if weekly_perf.get('weekly_change', 0) >= 0 else '#e74c3c'
    weekly_emoji = '🟢' if weekly_perf.get('weekly_change', 0) >= 0 else '🔴'
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 20px;
        }}
        .metric-box {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin: 15px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-title {{
            font-size: 14px;
            color: #666;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 28px;
            font-weight: bold;
            margin: 5px 0;
        }}
        .positive {{ color: #2ecc71; }}
        .negative {{ color: #e74c3c; }}
        .chart-container {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            text-align: center;
        }}
        .position-row {{
            display: flex;
            justify-content: space-between;
            padding: 10px;
            border-bottom: 1px solid #eee;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📈 Daily Performance Report</h1>
        <p>Market Close: {datetime.now().strftime('%B %d, %Y @ 4:00 PM ET')}</p>
    </div>
    
    <div class="metric-box">
        <div class="metric-title">💰 TODAY'S PERFORMANCE</div>
        <div class="metric-value" style="color: {daily_color}">
            {daily_emoji} ${daily_perf.get('daily_change', 0):+,.2f} ({daily_perf.get('daily_change_pct', 0):+.2%})
        </div>
        <div>
            <strong>Opening Value:</strong> ${daily_perf.get('opening_value', 0):,.2f}<br>
            <strong>Closing Value:</strong> ${daily_perf.get('closing_value', 0):,.2f}
        </div>
    </div>
    
    <div class="metric-box">
        <div class="metric-title">📊 WEEKLY PERFORMANCE (Last 7 Days)</div>
        <div class="metric-value" style="color: {weekly_color}">
            {weekly_emoji} ${weekly_perf.get('weekly_change', 0):+,.2f} ({weekly_perf.get('weekly_change_pct', 0):+.2%})
        </div>
        <div>
            <strong>Week Start:</strong> ${weekly_perf.get('start_value', 0):,.2f}<br>
            <strong>Week End:</strong> ${weekly_perf.get('end_value', 0):,.2f}
        </div>
    </div>
"""
    
    # Add daily chart
    if chart_daily:
        html += f"""
    <div class="chart-container">
        <h3>Today's Portfolio Value</h3>
        <img src="data:image/png;base64,{chart_daily}" style="max-width: 100%; height: auto;">
    </div>
"""
    
    # Add weekly chart
    if chart_weekly:
        html += f"""
    <div class="chart-container">
        <h3>Weekly Portfolio Value</h3>
        <img src="data:image/png;base64,{chart_weekly}" style="max-width: 100%; height: auto;">
    </div>
"""
    
    # Add positions breakdown
    if len(portfolio_df) > 0:
        html += """
    <div class="metric-box">
        <h3>📉 Stock-by-Stock Performance</h3>
        <table>
            <tr>
                <th>Ticker</th>
                <th>Shares</th>
                <th>Current Price</th>
                <th>Position Value</th>
                <th>Total P&L</th>
            </tr>
"""
        for _, row in portfolio_df.iterrows():
            pnl_color = '#2ecc71' if row['pnl'] >= 0 else '#e74c3c'
            html += f"""
            <tr>
                <td><strong>{row['ticker']}</strong></td>
                <td>{row['shares']:.0f}</td>
                <td>${row['current_price']:.2f}</td>
                <td>${row['current_value']:,.2f}</td>
                <td style="color: {pnl_color}; font-weight: bold;">
                    ${row['pnl']:+,.2f} ({row['pnl_pct']:+.2%})
                </td>
            </tr>
"""
        html += """
        </table>
    </div>
"""
    
    # Add positions chart
    if chart_positions:
        html += f"""
    <div class="chart-container">
        <h3>Position Performance Breakdown</h3>
        <img src="data:image/png;base64,{chart_positions}" style="max-width: 100%; height: auto;">
    </div>
"""
    
    html += """
    <div class="metric-box" style="background-color: #fff3cd; border-left: 4px solid #ffc107;">
        <strong>⚠️ Note:</strong> This report shows your portfolio's performance. 
        Past performance does not guarantee future results. Always do your own research.
    </div>
    
    <div style="text-align: center; margin-top: 30px; color: #666;">
        <p>Generated by AI Stock Predictor</p>
        <p>Questions? Reply to this email</p>
    </div>
</body>
</html>
"""
    
    return html


def generate_text_report(daily_perf: dict, weekly_perf: dict,
                        portfolio_df: pd.DataFrame) -> str:
    """Generate plain text version for email clients that don't support HTML"""
    
    daily_emoji = '🟢' if daily_perf.get('daily_change', 0) >= 0 else '🔴'
    weekly_emoji = '🟢' if weekly_perf.get('weekly_change', 0) >= 0 else '🔴'
    
    output = []
    output.append("="*70)
    output.append("📈 DAILY PERFORMANCE REPORT")
    output.append(f"Market Close: {datetime.now().strftime('%B %d, %Y @ 4:00 PM ET')}")
    output.append("="*70)
    
    output.append("\n💰 TODAY'S PERFORMANCE")
    output.append("-"*70)
    output.append(f"Opening Value: ${daily_perf.get('opening_value', 0):,.2f}")
    output.append(f"Closing Value: ${daily_perf.get('closing_value', 0):,.2f}")
    output.append(f"Daily Change: {daily_emoji} ${daily_perf.get('daily_change', 0):+,.2f} "
                 f"({daily_perf.get('daily_change_pct', 0):+.2%})")
    
    output.append("\n📊 WEEKLY PERFORMANCE (Last 7 Days)")
    output.append("-"*70)
    output.append(f"Week Start ({weekly_perf.get('start_date', 'N/A')}): "
                 f"${weekly_perf.get('start_value', 0):,.2f}")
    output.append(f"Week End ({weekly_perf.get('end_date', 'N/A')}): "
                 f"${weekly_perf.get('end_value', 0):,.2f}")
    output.append(f"Weekly Change: {weekly_emoji} ${weekly_perf.get('weekly_change', 0):+,.2f} "
                 f"({weekly_perf.get('weekly_change_pct', 0):+.2%})")
    
    if len(portfolio_df) > 0:
        output.append("\n📉 STOCK-BY-STOCK PERFORMANCE")
        output.append("-"*70)
        
        for i, row in portfolio_df.iterrows():
            pnl_emoji = '🟢' if row['pnl'] >= 0 else '🔴'
            output.append(f"\n{row['ticker']}:")
            output.append(f"  Shares: {row['shares']:.0f}")
            output.append(f"  Current Price: ${row['current_price']:.2f}")
            output.append(f"  Position Value: ${row['current_value']:,.2f}")
            output.append(f"  Total P&L: {pnl_emoji} ${row['pnl']:+,.2f} ({row['pnl_pct']:+.2%})")
    
    output.append("\n" + "="*70)
    output.append("📊 View charts and detailed dashboard:")
    output.append("Check the HTML attachment or visit the dashboard link")
    output.append("="*70)
    
    return "\n".join(output)


def main():
    """Generate end-of-day performance report"""
    
    print("\n" + "="*70)
    print("📊 END OF DAY PERFORMANCE REPORT")
    print("="*70)
    
    try:
        # 1. Load portfolio
        print("\n📂 Loading Portfolio...")
        portfolio_manager = PortfolioManager('portfolio.yaml')
        
        # 2. Load market data
        print("📂 Loading Market Data...")
        prices = load_data()
        
        if prices is None:
            raise Exception("Could not load price data")
        
        # 3. Update portfolio with current prices
        print("💼 Updating Portfolio Positions...")
        portfolio_df = portfolio_manager.update_with_prices(prices)
        summary = portfolio_manager.get_portfolio_summary(portfolio_df)
        
        # 4. Initialize performance tracker
        print("📊 Calculating Performance Metrics...")
        tracker = PerformanceTracker()
        
        # Take closing snapshot
        tracker.take_snapshot(portfolio_df, summary, market_open=False)
        
        # Calculate performance
        daily_perf = tracker.calculate_daily_performance()
        weekly_perf = tracker.calculate_weekly_performance()
        
        # 5. Generate charts
        print("📈 Generating Performance Charts...")
        chart_daily = tracker.create_performance_chart('daily')
        chart_weekly = tracker.create_performance_chart('weekly')
        chart_positions = tracker.create_performance_chart('positions')
        
        # 6. Format reports
        print("📝 Formatting Reports...")
        
        # HTML report
        html_report = format_performance_report(
            daily_perf, weekly_perf, portfolio_df,
            chart_daily, chart_weekly, chart_positions
        )
        
        with open('performance_report.html', 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        # Text report
        text_report = generate_text_report(daily_perf, weekly_perf, portfolio_df)
        
        with open('performance_report.txt', 'w', encoding='utf-8') as f:
            f.write(text_report)
        
        print("\n✅ Performance reports generated!")
        print("  • performance_report.html (with charts)")
        print("  • performance_report.txt (plain text)")
        
        # Print summary
        print("\n" + "="*70)
        print(text_report)
        print("="*70)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        # Create error file
        with open('performance_report.txt', 'w') as f:
            f.write(f"❌ Could not generate performance report\n")
            f.write(f"Error: {str(e)}\n")
        
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)