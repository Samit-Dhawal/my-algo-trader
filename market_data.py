# market_data.py
import yfinance as yf
import requests
import sqlite3
from datetime import datetime

class DataCollector:
    def __init__(self):
        self.db = sqlite3.connect('trading.db')
        self.setup_db()
    
    def setup_db(self):
        """Create a basic market_data table if not exists"""
        cursor = self.db.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER
            )
        ''')
        self.db.commit()

    def store_market_data(self, symbol, data):
        """Store the fetched OHLCV data into SQLite"""
        cursor = self.db.cursor()
        for index, row in data.iterrows():
            cursor.execute('''
                INSERT INTO market_data (symbol, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                index.strftime('%Y-%m-%d %H:%M:%S'),
                row['Open'],
                row['High'],
                row['Low'],
                row['Close'],
                int(row['Volume'])
            ))
        self.db.commit()

    def get_stock_data(self, symbol, period='1d'):
        """Free stock data from Yahoo Finance"""
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        self.store_market_data(symbol, data)
        return data
    
    def get_news_sentiment(self, symbol):
        """Free news from NewsAPI"""
        api_key = "YOUR_FREE_API_KEY"
        url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={api_key}"
        response = requests.get(url)
        return response.json()



# # market_data.py
# import yfinance as yf
# import requests
# import sqlite3
# from datetime import datetime

# class DataCollector:
#     def __init__(self):
#         self.db = sqlite3.connect('trading.db')
#         self.setup_db()
    
#     def get_stock_data(self, symbol, period='1d'):
#         """Free stock data from Yahoo Finance"""
#         ticker = yf.Ticker(symbol)
#         data = ticker.history(period=period)
#         self.store_market_data(symbol, data)
#         return data
    
#     def get_news_sentiment(self, symbol):
#         """Free news from NewsAPI"""
#         api_key = "YOUR_FREE_API_KEY"
#         url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={api_key}"
#         response = requests.get(url)
#         return response.json()