import alpaca_trade_api as tradeapi
from typing import Dict, Optional, List
import os
from datetime import datetime
import sqlite3
import json
from enum import Enum

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderExecutor:
    def __init__(self, paper_trading: bool = True):
        self.paper_trading = paper_trading
        self.db_path = 'data/trading.db'
        
        # Initialize Alpaca API
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        if paper_trading:
            self.base_url = 'https://paper-api.alpaca.markets'
        else:
            self.base_url = 'https://api.alpaca.markets'
        
        self.api = tradeapi.REST(
            self.api_key,
            self.secret_key,
            self.base_url,
            api_version='v2'
        )
        
        self.setup_database()
    
    def setup_database(self):
        """Initialize orders table"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT UNIQUE,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                order_type TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                price REAL,
                stop_price REAL,
                status TEXT NOT NULL,
                filled_qty INTEGER DEFAULT 0,
                filled_price REAL,
                commission REAL DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                filled_at DATETIME,
                cancelled_at DATETIME,
                metadata TEXT
            )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status)')
        
        conn.commit()
        conn.close()
    
    def get_account_info(self) -> Dict:
        """Get account information"""
        try:
            account = self.api.get_account()
            return {
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'day_trade_count': int(account.day_trade_count),
                'trading_blocked': account.trading_blocked,
                'account_blocked': account.account_blocked,
                'pattern_day_trader': account.pattern_day_trader
            }
        except Exception as e:
            print(f"Error getting account info: {e}")
            return {}
    
    def get_positions(self) -> List[Dict]:
        """Get current positions"""
        try:
            positions = self.api.list_positions()
            return [
                {
                    'symbol': pos.symbol,
                    'quantity': int(pos.qty),
                    'side': 'long' if int(pos.qty) > 0 else 'short',
                    'market_value': float(pos.market_value),
                    'avg_entry_price': float(pos.avg_entry_price),
                    'unrealized_pl': float(pos.unrealized_pl),
                    'unrealized_plpc': float(pos.unrealized_plpc),
                    'current_price': float(pos.current_price)
                }
                for pos in positions
            ]
        except Exception as e:
            print(f"Error getting positions: {e}")
            return []
    
    def calculate_position_size(self, symbol: str, signal_confidence: float, 
                              current_price: float, risk_per_trade: float = 0.02) -> int:
        """Calculate position size based on risk management rules"""
        
        account_info = self.get_account_info()
        if not account_info:
            return 0
        
        # Get available buying power
        buying_power = account_info.get('buying_power', 0)
        portfolio_value = account_info.get('portfolio_value', 0)
        
        # Risk management: max 2% risk per trade
        max_risk_amount = portfolio_value * risk_per_trade
        
        # Position sizing based on confidence
        confidence_multiplier = min(signal_confidence, 1.0)
        position_value = min(
            buying_power * 0.25 * confidence_multiplier,  # Max 25% of buying power
            max_risk_amount * 10  # 10:1 risk-reward assumption
        )
        
        # Calculate number of shares
        shares = int(position_value / current_price)
        
        return max(0, shares)
    
    def place_order(self, symbol: str, side: OrderSide, quantity: int,
                   order_type: OrderType = OrderType.MARKET, 
                   limit_price: Optional[float] = None,
                   stop_price: Optional[float] = None,
                   time_in_force: str = 'day',
                   metadata: Dict = None) -> Dict:
        """Place an order through Alpaca API"""
        
        if quantity <= 0:
            return {'error': 'Invalid quantity'}
        
        try:
            # Prepare order parameters
            order_params = {
                'symbol': symbol,
                'qty': quantity,
                'side': side.value,
                'type': order_type.value,
                'time_in_force': time_in_force
            }
            
            if limit_price:
                order_params['limit_price'] = limit_price
            
            if stop_price:
                order_params['stop_price'] = stop_price
            
            # Submit order
            order = self.api.submit_order(**order_params)
            
            # Store order in database
            self.store_order({
                'order_id': order.id,
                'symbol': symbol,
                'side': side.value,
                'order_type': order_type.value,
                'quantity': quantity,
                'price': limit_price,
                'stop_price': stop_price,
                'status': order.status,
                'metadata': json.dumps(metadata) if metadata else None
            })
            
            return {
                'success': True,
                'order_id': order.id,
                'symbol': symbol,
                'side': side.value,
                'quantity': quantity,
                'status': order.status,
                'created_at': order.created_at
            }
            
        except Exception as e:
            print(f"Error placing order: {e}")
            return {'error': str(e)}
    
    def store_order(self, order_data: Dict):
        """Store order in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO orders 
            (order_id, symbol, side, order_type, quantity, price, stop_price, status, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            order_data['order_id'],
            order_data['symbol'],
            order_data['side'],
            order_data['order_type'],
            order_data['quantity'],
            order_data.get('price'),
            order_data.get('stop_price'),
            order_data['status'],
            order_data.get('metadata')
        ))
        
        conn.commit()
        conn.close()
    
    def get_order_status(self, order_id: str) -> Dict:
        """Get order status from Alpaca"""
        try:
            order = self.api.get_order(order_id)
            
            # Update database
            self.update_order_status(order_id, {
                'status': order.status,
                'filled_qty': int(order.filled_qty or 0),
                'filled_price': float(order.filled_avg_price or 0),
                'filled_at': order.filled_at
            })
            
            return {
                'order_id': order.id,
                'symbol': order.symbol,
                'status': order.status,
                'quantity': int(order.qty),
                'filled_qty': int(order.filled_qty or 0),
                'filled_price': float(order.filled_avg_price or 0) if order.filled_avg_price else None,
                'created_at': order.created_at,
                'filled_at': order.filled_at
            }
            
        except Exception as e:
            print(f"Error getting order status: {e}")
            return {'error': str(e)}
    
    def update_order_status(self, order_id: str, update_data: Dict):
        """Update order status in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE orders 
            SET status = ?, filled_qty = ?, filled_price = ?, filled_at = ?
            WHERE order_id = ?
        ''', (
            update_data['status'],
            update_data.get('filled_qty', 0),
            update_data.get('filled_price'),
            update_data.get('filled_at'),
            order_id
        ))
        
        conn.commit()
        conn.close()
    
    def cancel_order(self, order_id: str) -> Dict:
        """Cancel an order"""
        try:
            self.api.cancel_order(order_id)
            
            # Update database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE orders 
                SET status = 'cancelled', cancelled_at = CURRENT_TIMESTAMP
                WHERE order_id = ?
            ''', (order_id,))
            conn.commit()
            conn.close()
            
            return {'success': True, 'order_id': order_id, 'status': 'cancelled'}
            
        except Exception as e:
            print(f"Error cancelling order: {e}")
            return {'error': str(e)}
    
    def execute_signal(self, signal_data: Dict) -> Dict:
        """Execute trading signal"""
        symbol = signal_data['symbol']
        signal = signal_data['signal']
        confidence = signal_data.get('confidence', 0.5)
        current_price = signal_data.get('current_price', 0)
        
        if signal == 0:  # Hold
            return {'action': 'hold', 'symbol': symbol}
        
        # Check current position
        positions = self.get_positions()
        current_position = next((p for p in positions if p['symbol'] == symbol), None)
        
        if signal == 1:  # Buy signal
            if current_position and current_position['quantity'] > 0:
                return {'action': 'already_long', 'symbol': symbol}
            
            # Calculate position size
            quantity = self.calculate_position_size(symbol, confidence, current_price)
            
            if quantity > 0:
                result = self.place_order(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    quantity=quantity,
                    order_type=OrderType.MARKET,
                    metadata={
                        'signal_confidence': confidence,
                        'signal_method': signal_data.get('method', 'unknown')
                    }
                )
                return result
            else:
                return {'error': 'Insufficient buying power or invalid position size'}
        
        elif signal == -1:  # Sell signal
            if not current_position or current_position['quantity'] <= 0:
                return {'action': 'no_position', 'symbol': symbol}
            
            # Sell current position
            result = self.place_order(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=abs(current_position['quantity']),
                order_type=OrderType.MARKET,
                metadata={
                    'signal_confidence': confidence,
                    'signal_method': signal_data.get('method', 'unknown')
                }
            )
            return result
        
        return {'error': 'Invalid signal'}
    
    def get_trade_history(self, symbol: str = None, days: int = 30) -> List[Dict]:
        """Get trade history from database"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT * FROM orders
            WHERE created_at >= datetime('now', '-{} days')
        '''.format(days)
        
        if symbol:
            query += f" AND symbol = '{symbol}'"
        
        query += " ORDER BY created_at DESC"
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df.to_dict('records') if not df.empty else []