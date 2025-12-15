"""
News Sentiment Analyzer - Phase 3
Get news sentiment from trusted financial sources
Uses free APIs: Alpha Vantage, NewsAPI, and web scraping
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time
import json
from pathlib import Path


class NewsSentimentAnalyzer:
    """Analyze news sentiment for stocks"""
    
    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        """
        Args:
            api_keys: Dict with 'alpha_vantage' and/or 'newsapi' keys
                     If None, will try to load from config or use free tier
        """
        self.api_keys = api_keys or {}
        self.cache_dir = Path('data/news_cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Trusted financial news sources
        self.trusted_sources = [
            'reuters.com', 'bloomberg.com', 'wsj.com', 'ft.com',
            'cnbc.com', 'marketwatch.com', 'seekingalpha.com',
            'fool.com', 'benzinga.com', 'investing.com'
        ]
    
    def get_stock_sentiment(self, ticker: str, days_back: int = 7) -> Dict:
        """
        Get aggregated sentiment for a stock
        
        Args:
            ticker: Stock ticker
            days_back: Days of news to analyze
            
        Returns:
            Dict with sentiment scores
        """
        # Check cache first
        cache_file = self.cache_dir / f"{ticker}_{datetime.now().strftime('%Y%m%d')}.json"
        
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        sentiment = {
            'ticker': ticker,
            'sentiment_score': 0.0,  # -1 to +1
            'sentiment_magnitude': 0.0,  # 0 to 1 (strength)
            'article_count': 0,
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0,
            'sources': []
        }
        
        # Try Alpha Vantage first
        if 'alpha_vantage' in self.api_keys:
            alpha_sentiment = self._get_alpha_vantage_sentiment(ticker)
            if alpha_sentiment:
                sentiment.update(alpha_sentiment)
        
        # Fallback: Simple web-based sentiment
        if sentiment['article_count'] == 0:
            web_sentiment = self._get_web_sentiment(ticker, days_back)
            sentiment.update(web_sentiment)
        
        # Cache result
        with open(cache_file, 'w') as f:
            json.dump(sentiment, f)
        
        return sentiment
    
    def _get_alpha_vantage_sentiment(self, ticker: str) -> Optional[Dict]:
        """Get sentiment from Alpha Vantage News Sentiment API"""
        api_key = self.api_keys.get('alpha_vantage')
        if not api_key:
            return None
        
        url = f"https://www.alphavantage.co/query"
        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': ticker,
            'apikey': api_key,
            'limit': 50
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if 'feed' not in data:
                return None
            
            articles = data['feed']
            
            sentiments = []
            sources = []
            
            for article in articles[:20]:  # Top 20 articles
                # Get ticker-specific sentiment
                ticker_sentiments = article.get('ticker_sentiment', [])
                
                for ts in ticker_sentiments:
                    if ts.get('ticker') == ticker:
                        score = float(ts.get('ticker_sentiment_score', 0))
                        sentiments.append(score)
                        
                        source = article.get('source', 'unknown')
                        if any(trusted in source.lower() for trusted in self.trusted_sources):
                            sources.append(source)
            
            if not sentiments:
                return None
            
            avg_sentiment = np.mean(sentiments)
            magnitude = np.std(sentiments)
            
            return {
                'sentiment_score': avg_sentiment,
                'sentiment_magnitude': magnitude,
                'article_count': len(sentiments),
                'positive_count': sum(1 for s in sentiments if s > 0.15),
                'negative_count': sum(1 for s in sentiments if s < -0.15),
                'neutral_count': sum(1 for s in sentiments if -0.15 <= s <= 0.15),
                'sources': sources[:5]
            }
        
        except Exception as e:
            print(f"Alpha Vantage error for {ticker}: {e}")
            return None
    
    def _get_web_sentiment(self, ticker: str, days_back: int) -> Dict:
        """
        Simple sentiment estimation without API
        Based on price action and volume (proxy for news impact)
        """
        try:
            import yfinance as yf
            
            stock = yf.Ticker(ticker)
            hist = stock.history(period=f"{days_back}d")
            
            if len(hist) < 2:
                return {'sentiment_score': 0.0, 'sentiment_magnitude': 0.0, 'article_count': 0}
            
            # Price momentum
            returns = hist['Close'].pct_change()
            avg_return = returns.mean()
            
            # Volume surge (news often causes volume spikes)
            volume_surge = (hist['Volume'].iloc[-1] / hist['Volume'].mean()) - 1
            
            # Combine signals
            sentiment_score = np.tanh(avg_return * 100)  # -1 to +1
            
            # Adjust by volume (high volume = high conviction)
            if volume_surge > 0.5:
                sentiment_score *= 1.2
            
            magnitude = min(abs(volume_surge), 1.0)
            
            return {
                'sentiment_score': np.clip(sentiment_score, -1, 1),
                'sentiment_magnitude': magnitude,
                'article_count': 1,  # Proxy
                'positive_count': 1 if sentiment_score > 0.2 else 0,
                'negative_count': 1 if sentiment_score < -0.2 else 0,
                'neutral_count': 1 if -0.2 <= sentiment_score <= 0.2 else 0,
                'sources': ['price_action']
            }
        
        except Exception as e:
            print(f"Web sentiment error for {ticker}: {e}")
            return {
                'sentiment_score': 0.0,
                'sentiment_magnitude': 0.0,
                'article_count': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'sources': []
            }
    
    def get_market_sentiment(self, tickers: List[str]) -> pd.DataFrame:
        """
        Get sentiment for multiple stocks
        
        Args:
            tickers: List of tickers
            
        Returns:
            DataFrame with sentiment data
        """
        sentiments = []
        
        print(f"\nAnalyzing news sentiment for {len(tickers)} stocks...")
        
        for i, ticker in enumerate(tickers):
            if i > 0 and i % 5 == 0:
                time.sleep(1)  # Rate limiting
                print(f"  Processed {i}/{len(tickers)}...")
            
            sentiment = self.get_stock_sentiment(ticker)
            sentiments.append(sentiment)
        
        df = pd.DataFrame(sentiments)
        
        print(f"✓ Sentiment analysis complete")
        print(f"  Average sentiment: {df['sentiment_score'].mean():.3f}")
        print(f"  Positive stocks: {(df['sentiment_score'] > 0.15).sum()}")
        print(f"  Negative stocks: {(df['sentiment_score'] < -0.15).sum()}")
        
        return df


def add_sentiment_features(df: pd.DataFrame, 
                           sentiment_analyzer: Optional[NewsSentimentAnalyzer] = None) -> pd.DataFrame:
    """
    Add news sentiment features to dataframe
    
    Args:
        df: Features dataframe with 'ticker' column
        sentiment_analyzer: NewsSentimentAnalyzer instance
        
    Returns:
        DataFrame with sentiment features added
    """
    if sentiment_analyzer is None:
        sentiment_analyzer = NewsSentimentAnalyzer()
    
    # Get unique tickers
    tickers = df['ticker'].unique().tolist()
    
    # Get sentiment data
    sentiment_df = sentiment_analyzer.get_market_sentiment(tickers)
    
    # Merge with main dataframe
    df = df.merge(
        sentiment_df[['ticker', 'sentiment_score', 'sentiment_magnitude', 
                     'positive_count', 'negative_count']],
        on='ticker',
        how='left'
    )
    
    # Fill missing with neutral
    df['sentiment_score'] = df['sentiment_score'].fillna(0)
    df['sentiment_magnitude'] = df['sentiment_magnitude'].fillna(0)
    df['positive_count'] = df['positive_count'].fillna(0)
    df['negative_count'] = df['negative_count'].fillna(0)
    
    # Create derived features
    df['sentiment_positive'] = (df['sentiment_score'] > 0.15).astype(int)
    df['sentiment_negative'] = (df['sentiment_score'] < -0.15).astype(int)
    df['sentiment_strength'] = abs(df['sentiment_score']) * df['sentiment_magnitude']
    
    return df


if __name__ == "__main__":
    # Test sentiment analyzer
    analyzer = NewsSentimentAnalyzer()
    
    # Test single stock
    result = analyzer.get_stock_sentiment('AAPL')
    print(f"\nAAPL Sentiment:")
    print(f"  Score: {result['sentiment_score']:.3f}")
    print(f"  Magnitude: {result['sentiment_magnitude']:.3f}")
    print(f"  Articles: {result['article_count']}")
    
    # Test multiple stocks
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    df = analyzer.get_market_sentiment(test_tickers)
    print("\nTop Sentiments:")
    print(df.sort_values('sentiment_score', ascending=False)[['ticker', 'sentiment_score', 'article_count']])
    #test