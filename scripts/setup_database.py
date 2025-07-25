#!/usr/bin/env python3
import sqlite3
import os
from datetime import datetime

def create_database():
    """Initialize the SQLite database with required tables"""
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/logs', exist_ok=True)
    os.makedirs('data/models', exist_ok=True)
    
    conn = sqlite3.connect('data/trading.db')
    cursor = conn.cursor()
    
    # Create tables
    tables = {
        'market_data': '''
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                open_price REAL NOT NULL,
                high_price REAL NOT NULL,
                low_price REAL NOT NULL,
                close_price REAL NOT NULL,
                volume INTEGER NOT NULL,
                vwap REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp)
            )
        ''',
        
        'signals': '''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                signal INTEGER NOT NULL,  -- -1: sell, 0: hold, 1: buy
                confidence REAL NOT NULL,
                method TEXT NOT NULL,
                parameters TEXT,  -- JSON string of parameters
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''',
        
        'orders': '''
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,  -- buy/sell
                quantity INTEGER NOT NULL,
                order_type TEXT NOT NULL,  -- market/limit
                price REAL,
                status TEXT NOT NULL,  -- pending/filled/cancelled
                broker_order_id TEXT,
                signal_id INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                filled_at DATETIME,
                FOREIGN KEY(signal_id) REFERENCES signals(id)
            )
        ''',
        
        'portfolio': '''
            CREATE TABLE IF NOT EXISTS portfolio (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                avg_cost REAL NOT NULL,
                current_price REAL,
                unrealized_pl REAL,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol)
            )
        ''',
        
        'trade_history': '''
            CREATE TABLE IF NOT EXISTS trade_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                price REAL NOT NULL,
                commission REAL DEFAULT 0,
                profit_loss REAL,
                order_id INTEGER,
                executed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(order_id) REFERENCES orders(id)
            )
        ''',
        
        'system_metrics': '''
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''',
        
        'news_sentiment': '''
            CREATE TABLE IF NOT EXISTS news_sentiment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                headline TEXT NOT NULL,
                content TEXT,
                sentiment_score REAL,
                source TEXT,
                published_at DATETIME,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        '''
    }
    
    # Create all tables
    for table_name, create_sql in tables.items():
        cursor.execute(create_sql)
        print(f"✅ Created table: {table_name}")
    
    # Create indexes for better performance
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp ON market_data(symbol, timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_signals_symbol_timestamp ON signals(symbol, timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status)",
        "CREATE INDEX IF NOT EXISTS idx_trade_history_symbol ON trade_history(symbol)"
    ]
    
    for index_sql in indexes:
        cursor.execute(index_sql)
    
    print("✅ Created database indexes")
    
    conn.commit()
    conn.close()
    print("🎉 Database setup completed successfully!")

if __name__ == "__main__":
    create_database()