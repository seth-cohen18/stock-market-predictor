"""
Enhanced Recommendation Engine with beautiful UI, partial shares, and company names
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from scipy import stats
import yfinance as yf

import utils
import build_features
import train


# Company name mapping (top companies)
COMPANY_NAMES = {
    'AAPL': 'Apple Inc.', 'MSFT': 'Microsoft Corp.', 'GOOGL': 'Alphabet Inc.',
    'AMZN': 'Amazon.com Inc.', 'NVDA': 'NVIDIA Corp.', 'META': 'Meta Platforms',
    'TSLA': 'Tesla Inc.', 'BRK.B': 'Berkshire Hathaway', 'V': 'Visa Inc.',
    'JPM': 'JPMorgan Chase', 'WMT': 'Walmart Inc.', 'MA': 'Mastercard Inc.',
    'PG': 'Procter & Gamble', 'JNJ': 'Johnson & Johnson', 'UNH': 'UnitedHealth Group',
    'HD': 'Home Depot', 'BAC': 'Bank of America', 'XOM': 'Exxon Mobil',
    'CVX': 'Chevron Corp.', 'LLY': 'Eli Lilly', 'ABBV': 'AbbVie Inc.',
    'KO': 'Coca-Cola Co.', 'PEP': 'PepsiCo Inc.', 'COST': 'Costco Wholesale',
    'AVGO': 'Broadcom Inc.', 'MRK': 'Merck & Co.', 'TMO': 'Thermo Fisher',
    'ABT': 'Abbott Labs', 'ORCL': 'Oracle Corp.', 'CSCO': 'Cisco Systems',
    'AMD': 'Advanced Micro Devices', 'INTC': 'Intel Corp.', 'QCOM': 'Qualcomm Inc.',
    'TXN': 'Texas Instruments', 'NFLX': 'Netflix Inc.', 'DIS': 'Walt Disney',
    'NKE': 'Nike Inc.', 'CRM': 'Salesforce Inc.', 'ADBE': 'Adobe Inc.',
    'LOW': 'Lowe\'s Companies', 'UPS': 'United Parcel Service', 'BA': 'Boeing Co.',
    'GE': 'General Electric', 'MMM': '3M Company', 'CAT': 'Caterpillar Inc.',
    'SPY': 'S&P 500 ETF', 'QQQ': 'Nasdaq-100 ETF', 'IWM': 'Russell 2000 ETF',
    'LRCX': 'Lam Research', 'LYFT': 'Lyft Inc.', 'UBER': 'Uber Technologies',
    'SQ': 'Block Inc.', 'COIN': 'Coinbase Global', 'SHOP': 'Shopify Inc.'
}


def get_company_name(ticker: str) -> str:
    """Get company name for ticker, fetch if not in cache"""
    if ticker in COMPANY_NAMES:
        return COMPANY_NAMES[ticker]
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        name = info.get('longName') or info.get('shortName') or ticker
        COMPANY_NAMES[ticker] = name
        return name
    except:
        return ticker


class EnhancedRecommendationEngine:
    """Enhanced recommendation engine with better UI and features"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        self.config = utils.load_config(config_path)
        self.models = {}
        self.metadata = {}
        self.df_features = None
        self.feature_cols = None
        
        self._load_models()
        self._load_features()
    
    def _load_models(self):
        """Load all trained models"""
        print("\n🔄 Loading models...")
        for horizon in self.config['model']['horizons']:
            model_path = f"models/model_{horizon}d.txt"
            metadata_path = f"models/model_{horizon}d_metadata.json"
            
            try:
                self.models[horizon] = train.load_model(model_path)
                self.metadata[horizon] = train.load_model_metadata(metadata_path)
                print(f"   ✓ {horizon}-day model loaded")
            except Exception as e:
                print(f"   ✗ Failed to load {horizon}-day model: {e}")
    
    def _load_features(self):
        """Load feature data"""
        features_path = f"{self.config['data']['processed_dir']}/features.parquet"
        self.df_features = build_features.load_features(features_path)
        self.feature_cols = build_features.get_feature_columns(self.df_features)
        
        latest = self.df_features['date'].max()
        print(f"   ✓ Features loaded (latest: {latest.date()})")
    
    def get_latest_predictions(self, horizon: int, top_n: int = 50) -> pd.DataFrame:
        """Get predictions for the latest available date"""
        if horizon not in self.models:
            raise ValueError(f"No model for horizon {horizon}")
        
        latest_date = self.df_features['date'].max()
        df_latest = self.df_features[self.df_features['date'] == latest_date].copy()
        
        X = df_latest[self.feature_cols].fillna(0)
        df_latest['predicted_return'] = self.models[horizon].predict(X)
        
        if 'volatility_20d' in df_latest.columns:
            df_latest['predicted_volatility'] = df_latest['volatility_20d']
        else:
            df_latest['predicted_volatility'] = 0.20
        
        df_latest['risk_adjusted_score'] = (
            df_latest['predicted_return'] / (df_latest['predicted_volatility'] + 1e-8)
        )
        
        df_latest = df_latest.sort_values('predicted_return', ascending=False)
        
        return df_latest.head(top_n)
    
    def allocate_portfolio(self, selected: pd.DataFrame, capital: float, 
                          method: str = 'volatility', allow_fractional: bool = True) -> pd.DataFrame:
        """Allocate capital with support for fractional shares"""
        if len(selected) == 0:
            return selected
        
        selected = selected.copy()
        
        if method == 'equal':
            selected['weight'] = 1 / len(selected)
        elif method == 'volatility':
            inv_vol = 1 / (selected['predicted_volatility'] + 1e-8)
            selected['weight'] = inv_vol / inv_vol.sum()
        else:
            selected['weight'] = 1 / len(selected)
        
        max_weight = self.config['backtest']['max_weight']
        selected['weight'] = selected['weight'].clip(upper=max_weight)
        selected['weight'] = selected['weight'] / selected['weight'].sum()
        
        selected['allocation'] = selected['weight'] * capital
        
        if allow_fractional:
            # Fractional shares
            selected['shares'] = selected['allocation'] / selected['close']
            selected['actual_allocation'] = selected['shares'] * selected['close']
        else:
            # Whole shares only
            selected['shares'] = (selected['allocation'] / selected['close']).astype(int)
            selected['actual_allocation'] = selected['shares'] * selected['close']
        
        return selected
    
    def recommend(self, capital: float, horizon: str = '1w', risk_level: str = 'medium',
                 goal: str = 'max_return', target_roi: Optional[float] = None,
                 num_positions: Optional[int] = None, allow_fractional: bool = True) -> Dict:
        """Generate investment recommendations"""
        
        horizon_map = {'1d': 1, '1w': 5, '1m': 20}
        horizon_days = horizon_map.get(horizon, 5)
        
        if horizon_days not in self.models:
            available = list(self.models.keys())
            horizon_days = available[0] if available else 1
        
        predictions = self.get_latest_predictions(horizon_days, top_n=50)
        
        if risk_level == 'low':
            predictions = predictions[predictions['predicted_volatility'] < 0.25]
        
        if goal == 'max_return':
            predictions = predictions.sort_values('predicted_return', ascending=False)
        elif goal == 'max_sharpe':
            predictions = predictions.sort_values('risk_adjusted_score', ascending=False)
        elif goal == 'prob_target' and target_roi is not None:
            horizon_vol = predictions['predicted_volatility'] * np.sqrt(horizon_days / 252)
            prob_target = 1 - stats.norm.cdf(
                target_roi, loc=predictions['predicted_return'], scale=horizon_vol
            )
            predictions['prob_target'] = prob_target
            predictions = predictions.sort_values('prob_target', ascending=False)
        
        n_positions = num_positions or self.config['recommend']['top_n']
        selected = predictions.head(n_positions).copy()
        
        if len(selected) == 0:
            return {
                'success': False,
                'message': 'No suitable investments found with current filters'
            }
        
        selected = self.allocate_portfolio(selected, capital, allow_fractional=allow_fractional)
        
        portfolio_metrics = self.calculate_portfolio_metrics(selected, horizon_days)
        
        recommendations = []
        for _, row in selected.iterrows():
            rec = {
                'ticker': row['ticker'],
                'company': get_company_name(row['ticker']),
                'shares': row['shares'],
                'allocation': row['actual_allocation'],
                'weight': row['weight'],
                'current_price': row['close'],
                'predicted_return': row['predicted_return'],
                'predicted_volatility': row['predicted_volatility'],
                'risk_adjusted_score': row['risk_adjusted_score']
            }
            recommendations.append(rec)
        
        result = {
            'success': True,
            'recommendations': recommendations,
            'portfolio_metrics': portfolio_metrics,
            'inputs': {
                'capital': capital,
                'horizon': horizon,
                'horizon_days': horizon_days,
                'risk_level': risk_level,
                'goal': goal,
                'target_roi': target_roi,
                'fractional_shares': allow_fractional
            },
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'data_date': str(predictions['date'].iloc[0].date())
            }
        }
        
        return result
    
    def calculate_portfolio_metrics(self, selected: pd.DataFrame, horizon: int) -> Dict:
        """Calculate portfolio-level metrics"""
        if len(selected) == 0:
            return {}
        
        weights = selected['weight'].values
        expected_return = (weights * selected['predicted_return']).sum()
        portfolio_vol = np.sqrt((weights**2 * selected['predicted_volatility']**2).sum())
        horizon_vol = portfolio_vol * np.sqrt(horizon / 252)
        prob_profit = 1 - stats.norm.cdf(0, loc=expected_return, scale=horizon_vol)
        var_5 = stats.norm.ppf(0.05, loc=expected_return, scale=horizon_vol)
        
        return {
            'expected_return': expected_return,
            'expected_volatility': portfolio_vol,
            'horizon_volatility': horizon_vol,
            'probability_profit': prob_profit,
            'var_5_pct': var_5,
            'sharpe_estimate': expected_return / (portfolio_vol + 1e-8) * np.sqrt(252)
        }
    
    def print_recommendations(self, result: Dict):
        """Beautiful formatted output"""
        if not result['success']:
            print(f"\n❌ {result['message']}")
            return
        
        inputs = result['inputs']
        portfolio = result['portfolio_metrics']
        recs = result['recommendations']
        
        # Header
        print("\n" + "="*90)
        print("💰 INVESTMENT RECOMMENDATIONS".center(90))
        print("="*90)
        
        # Inputs section
        print(f"\n📊 YOUR PARAMETERS:")
        print(f"   💵 Capital:        ${inputs['capital']:,.2f}")
        print(f"   ⏱️  Time Horizon:   {inputs['horizon']} ({inputs['horizon_days']} trading days)")
        print(f"   🎲 Risk Level:     {inputs['risk_level'].title()}")
        print(f"   🎯 Goal:           {inputs['goal'].replace('_', ' ').title()}")
        if inputs.get('fractional_shares'):
            print(f"   📈 Shares:         Fractional shares allowed")
        
        # Portfolio metrics
        print(f"\n📈 PORTFOLIO METRICS:")
        prob_color = "🟢" if portfolio['probability_profit'] > 0.55 else "🟡" if portfolio['probability_profit'] > 0.50 else "🔴"
        print(f"   {prob_color} Probability of Profit:  {portfolio['probability_profit']:.1%}")
        print(f"   💹 Expected Return:        {portfolio['expected_return']:.2%}")
        print(f"   📉 Expected Volatility:    {portfolio['expected_volatility']:.1%}")
        print(f"   ⚖️  Sharpe Ratio:           {portfolio['sharpe_estimate']:.2f}")
        print(f"   ⚠️  Worst Case (5% VaR):   {portfolio['var_5_pct']:.2%}")
        
        # Recommendations table
        print(f"\n" + "="*90)
        print(f"🎯 RECOMMENDED POSITIONS ({len(recs)} stocks)".center(90))
        print("="*90)
        
        print(f"\n{'Ticker':<8} {'Company':<25} {'Shares':>10} {'Amount':>12} {'Weight':>8} {'Return':>8} {'Risk':>8}")
        print("-"*90)
        
        total_allocation = 0
        for rec in recs:
            company = rec['company'][:23] + '..' if len(rec['company']) > 25 else rec['company']
            shares_str = f"{rec['shares']:.4f}" if inputs.get('fractional_shares') else f"{int(rec['shares'])}"
            
            print(
                f"{rec['ticker']:<8} "
                f"{company:<25} "
                f"{shares_str:>10} "
                f"${rec['allocation']:>10,.2f} "
                f"{rec['weight']:>7.1%} "
                f"{rec['predicted_return']:>7.2%} "
                f"{rec['predicted_volatility']:>7.1%}"
            )
            total_allocation += rec['allocation']
        
        print("-"*90)
        print(f"{'TOTAL':<8} {'':<25} {'':<10} ${total_allocation:>10,.2f}")
        
        cash = inputs['capital'] - total_allocation
        if cash > 0:
            print(f"\n💵 Remaining Cash: ${cash:,.2f}")
        
        # Tips section
        print(f"\n" + "="*90)
        print("💡 TIPS TO IMPROVE YOUR RESULTS".center(90))
        print("="*90)
        self._print_tips(portfolio)
        
        # Disclaimer
        print(f"\n⚠️  IMPORTANT DISCLAIMER:")
        print(f"   • These are probabilistic forecasts based on historical patterns")
        print(f"   • Past performance does NOT guarantee future results")
        print(f"   • Always do your own research before investing")
        print(f"   • Consider consulting a financial advisor")
        print(f"   • Only invest money you can afford to lose")
        print(f"   • Diversification does not eliminate risk")
        print("="*90 + "\n")
    
    def _print_tips(self, portfolio: Dict):
        """Print personalized tips based on portfolio metrics"""
        prob = portfolio['probability_profit']
        sharpe = portfolio['sharpe_estimate']
        
        tips = []
        
        if prob < 0.52:
            tips.append("📊 Low win probability - Consider waiting for better setups")
            tips.append("🔄 Update data daily: run 'python main.py' each morning")
        
        if prob >= 0.52 and prob < 0.55:
            tips.append("✅ Decent probability - Small position sizes recommended")
            tips.append("📈 Track performance to validate the model over time")
        
        if prob >= 0.55:
            tips.append("🎯 Good probability - Model showing confidence in these picks")
            tips.append("💪 Consider scaling position sizes gradually")
        
        if sharpe < 0.5:
            tips.append("⚖️  Low Sharpe ratio - Risk may not justify returns")
            tips.append("🎲 Try 'max_sharpe' goal for better risk-adjusted picks")
        
        tips.append("🔍 Always verify companies fundamentals independently")
        tips.append("⏰ Set stop-losses to limit downside risk")
        tips.append("📱 Use paper trading first to validate the system")
        
        for i, tip in enumerate(tips[:5], 1):
            print(f"   {i}. {tip}")


if __name__ == "__main__":
    engine = EnhancedRecommendationEngine()
    
    result = engine.recommend(
        capital=5000,
        horizon='1w',
        risk_level='medium',
        goal='max_sharpe',
        allow_fractional=True
    )
    
    engine.print_recommendations(result)