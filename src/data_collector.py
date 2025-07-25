import asyncio
import aiohttp
import sqlite3
import redis
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import os
from dotenv import load_dotenv

load_dotenv()

class DataCollector:
    def __init__(self):
        self.db_path = 'data/trading.db'
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        self.setup_database()
        
        # API Keys
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.news_api_key = os.getenv('NEWS_API_KEY')
        
    def setup_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Market data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                open_price REAL,
                high_price REAL,
                low_price REAL,
                close_price REAL,
                volume INTEGER,
                source TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # News data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT,
                url TEXT,
                source TEXT,
                published_at DATETIME,
                sentiment_score REAL,
                symbols TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol_timestamp ON market_data(symbol, timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_news_published ON news_data(published_at)')
        
        conn.commit()
        conn.close()
    
    async def fetch_stock_data(self, symbols: List[str]) -> Dict:
        """Fetch real-time stock data from multiple sources"""
        results = {}
        
        # Primary: Yahoo Finance (Free, unlimited)
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                hist = ticker.history(period="1d", interval="1m")
                
                if not hist.empty:
                    latest = hist.iloc[-1]
                    results[symbol] = {
                        'symbol': symbol,
                        'price': float(latest['Close']),
                        'open': float(latest['Open']),
                        'high': float(latest['High']),
                        'low': float(latest['Low']),
                        'volume': int(latest['Volume']),
                        'timestamp': latest.name.isoformat(),
                        'source': 'yahoo_finance'
                    }
                    
                    # Cache in Redis for 30 seconds
                    self.redis_client.setex(
                        f"stock:{symbol}", 
                        30, 
                        json.dumps(results[symbol])
                    )
                    
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
                
        return results
    
    async def fetch_crypto_data(self, symbols: List[str]) -> Dict:
        """Fetch crypto data from CoinGecko (Free: 10-50 calls/minute)"""
        results = {}
        
        async with aiohttp.ClientSession() as session:
            for symbol in symbols:
                try:
                    # CoinGecko API - Free tier
                    url = f"https://api.coingecko.com/api/v3/simple/price"
                    params = {
                        'ids': symbol.lower(),
                        'vs_currencies': 'usd',
                        'include_24hr_change': 'true',
                        'include_24hr_vol': 'true'
                    }
                    
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            if symbol.lower() in data:
                                crypto_data = data[symbol.lower()]
                                results[symbol] = {
                                    'symbol': symbol,
                                    'price': crypto_data['usd'],
                                    'volume': crypto_data.get('usd_24h_vol', 0),
                                    'change_24h': crypto_data.get('usd_24h_change', 0),
                                    'timestamp': datetime.utcnow().isoformat(),
                                    'source': 'coingecko'
                                }
                                
                                # Cache for 60 seconds
                                self.redis_client.setex(
                                    f"crypto:{symbol}", 
                                    60, 
                                    json.dumps(results[symbol])
                                )
                                
                    # Rate limiting for free tier
                    await asyncio.sleep(1.2)  # ~50 calls/minute max
                    
                except Exception as e:
                    print(f"Error fetching crypto {symbol}: {e}")
                    
        return results
    
    async def fetch_news_sentiment(self, symbols: List[str]) -> List[Dict]:
        """Fetch news data with sentiment analysis"""
        news_data = []
        
        async with aiohttp.ClientSession() as session:
            for symbol in symbols:
                try:
                    # NewsAPI - Free: 1000 requests/day
                    url = "https://newsapi.org/v2/everything"
                    params = {
                        'q': f"{symbol} stock",
                        'apiKey': self.news_api_key,
                        'language': 'en',
                        'sortBy': 'publishedAt',
                        'pageSize': 5,
                        'from': (datetime.now() - timedelta(hours=24)).isoformat()
                    }
                    
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            for article in data.get('articles', []):
                                # Simple sentiment analysis (can be enhanced with NLTK/spaCy)
                                sentiment = self.analyze_sentiment(article.get('title', '') + ' ' + article.get('description', ''))
                                
                                news_item = {
                                    'title': article['title'],
                                    'description': article.get('description', ''),
                                    'url': article['url'],
                                    'source': article['source']['name'],
                                    'published_at': article['publishedAt'],
                                    'sentiment_score': sentiment,
                                    'symbols': symbol
                                }
                                
                                news_data.append(news_item)
                                
                    # Rate limiting
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    print(f"Error fetching news for {symbol}: {e}")
                    
        return news_data
    
    def analyze_sentiment(self, text: str) -> float:
        """Simple sentiment analysis (-1 to 1 scale)"""
        # Simple keyword-based sentiment (can be enhanced with proper NLP)
        positive_words = ['buy', 'bull', 'bullish', 'up', 'rise', 'gain', 'profit', 'strong', 'growth']
        negative_words = ['sell', 'bear', 'bearish', 'down', 'fall', 'loss', 'weak', 'decline', 'crash']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count + negative_count == 0:
            return 0.0
        
        return (positive_count - negative_count) / (positive_count + negative_count)
    
    def store_market_data(self, data: Dict):
        """Store market data in SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for symbol, item in data.items():
            cursor.execute('''
                INSERT INTO market_data 
                (symbol, timestamp, open_price, high_price, low_price, close_price, volume, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                item['symbol'],
                item['timestamp'],
                item.get('open', item['price']),
                item.get('high', item['price']),
                item.get('low', item['price']),
                item['price'],
                item.get('volume', 0),
                item['source']
            ))
        
        conn.commit()
        conn.close()
    
    def store_news_data(self, news_list: List[Dict]):
        """Store news data in SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for news in news_list:
            cursor.execute('''
                INSERT INTO news_data 
                (title, description, url, source, published_at, sentiment_score, symbols)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                news['title'],
                news['description'],
                news['url'],
                news['source'],
                news['published_at'],
                news['sentiment_score'],
                news['symbols']
            ))
        
        conn.commit()
        conn.close()
    
    async def collect_all_data(self, stock_symbols: List[str], crypto_symbols: List[str] = None):
        """Main data collection orchestrator"""
        print(f"Starting data collection at {datetime.now()}")
        
        # Collect stock data
        if stock_symbols:
            stock_data = await self.fetch_stock_data(stock_symbols)
            self.store_market_data(stock_data)
            print(f"Collected stock data for {len(stock_data)} symbols")
        
        # Collect crypto data
        if crypto_symbols:
            crypto_data = await self.fetch_crypto_data(crypto_symbols)
            self.store_market_data(crypto_data)
            print(f"Collected crypto data for {len(crypto_data)} symbols")
        
        # Collect news data
        news_data = await self.fetch_news_sentiment(stock_symbols)
        self.store_news_data(news_data)
        print(f"Collected {len(news_data)} news articles")
        
        print(f"Data collection completed at {datetime.now()}")

# Usage Example
async def main():
    collector = DataCollector()
    await collector.collect_all_data(['AAPL', 'GOOGL', 'TSLA'], ['bitcoin', 'ethereum'])

if __name__ == "__main__":
    asyncio.run(main())