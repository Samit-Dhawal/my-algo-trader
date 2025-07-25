import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import sqlite3
from datetime import datetime, timedelta
import json
import os

class Backtester:
    def __init__(self, initial_capital: float = 10000.0, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.db_path = 'data/trading.db'
        
    def get_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get historical market data for backtesting"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT timestamp, open_price, high_price, low_price, close_price, volume
            FROM market_data
            WHERE symbol = ? AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp ASC
        '''
        
        df = pd.read_sql_query(query, conn, params=(symbol, start_date, end_date))
        conn.close()
        
        if df.empty:
            return df
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        return df
    
    def simulate_strategy(self, symbol: str, start_date: str, end_date: str, 
                         strategy_func, **strategy_params) -> Dict:
        """Run backtest simulation"""
        
        # Get historical data
        df = self.get_historical_data(symbol, start_date, end_date)
        
        if df.empty:
            return {'error': 'No historical data available'}
        
        # Initialize portfolio
        portfolio = {
            'cash': self.initial_capital,
            'shares': 0,
            'value': self.initial_capital,
            'trades': [],
            'equity_curve': []
        }
        
        # Apply strategy
        from .signal_engine import SignalEngine
        signal_engine = SignalEngine()
        
        # Calculate indicators for entire dataset
        df_with_indicators = signal_engine.calculate_technical_indicators(df.copy())
        
        for i in range(50, len(df_with_indicators)):  # Start after warmup period
            current_data = df_with_indicators.iloc[:i+1]
            current_price = current_data['Close'].iloc[-1]
            
            # Generate signal
            signal_data = strategy_func(current_data, **strategy_params)
            signal = signal_data['signal']
            confidence = signal_data.get('confidence', 0.5)
            
            # Execute trades based on signal
            if signal == 1 and portfolio['shares'] == 0:  # Buy signal
                # Calculate position size (risk management)
                position_size = min(
                    portfolio['cash'] * confidence * 0.95,  # Max 95% of cash, scaled by confidence
                    portfolio['cash'] * 0.2  # Max 20% per trade
                )
                
                if position_size > current_price:
                    shares_to_buy = int(position_size / current_price)
                    cost = shares_to_buy * current_price * (1 + self.commission)
                    
                    if cost <= portfolio['cash']:
                        portfolio['cash'] -= cost
                        portfolio['shares'] += shares_to_buy
                        
                        portfolio['trades'].append({
                            'timestamp': current_data.index[-1],
                            'type': 'BUY',
                            'shares': shares_to_buy,
                            'price': current_price,
                            'cost': cost,
                            'confidence': confidence
                        })
            
            elif signal == -1 and portfolio['shares'] > 0:  # Sell signal
                # Sell all shares
                revenue = portfolio['shares'] * current_price * (1 - self.commission)
                portfolio['cash'] += revenue
                
                portfolio['trades'].append({
                    'timestamp': current_data.index[-1],
                    'type': 'SELL',
                    'shares': portfolio['shares'],
                    'price': current_price,
                    'revenue': revenue,
                    'confidence': confidence
                })
                
                portfolio['shares'] = 0
            
            # Update portfolio value
            portfolio_value = portfolio['cash'] + (portfolio['shares'] * current_price)
            portfolio['equity_curve'].append({
                'timestamp': current_data.index[-1],
                'value': portfolio_value,
                'cash': portfolio['cash'],
                'shares_value': portfolio['shares'] * current_price
            })
        
        # Calculate final metrics
        return self.calculate_performance_metrics(portfolio, df)
    
    def calculate_performance_metrics(self, portfolio: Dict, price_data: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        equity_curve = pd.DataFrame(portfolio['equity_curve'])
        equity_curve.set_index('timestamp', inplace=True)
        
        if equity_curve.empty:
            return {'error': 'No trades executed'}
        
        # Basic metrics
        final_value = equity_curve['value'].iloc[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Calculate returns series
        equity_curve['returns'] = equity_curve['value'].pct_change().dropna()
        
        # Risk metrics
        volatility = equity_curve['returns'].std() * np.sqrt(252)  # Annualized
        
        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        excess_return = total_return - risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        running_max = equity_curve['value'].expanding().max()
        drawdown = (equity_curve['value'] - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        trades = portfolio['trades']
        if len(trades) >= 2:
            winning_trades = 0
            total_trades = 0
            
            for i in range(1, len(trades), 2):  # Assuming buy-sell pairs
                if i < len(trades):
                    buy_trade = trades[i-1]
                    sell_trade = trades[i]
                    
                    if sell_trade['revenue'] > buy_trade['cost']:
                        winning_trades += 1
                    total_trades += 1
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
        else:
            win_rate = 0
        
        # Buy and hold comparison
        buy_hold_return = (price_data['Close'].iloc[-1] - price_data['Close'].iloc[0]) / price_data['Close'].iloc[0]
        
        return {
            'symbol': price_data.name if hasattr(price_data, 'name') else 'Unknown',
            'start_date': equity_curve.index[0].strftime('%Y-%m-%d'),
            'end_date': equity_curve.index[-1].strftime('%Y-%m-%d'),
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'buy_hold_return': buy_hold_return,
            'buy_hold_return_pct': buy_hold_return * 100,
            'alpha': (total_return - buy_hold_return) * 100,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'total_trades': len(trades),
            'trades': trades,
            'equity_curve': equity_curve.to_dict('records')
        }

def simple_ma_crossover_strategy(df: pd.DataFrame, short_window: int = 10, long_window: int = 20) -> Dict:
    """Simple moving average crossover strategy for backtesting"""
    if len(df) < long_window:
        return {'signal': 0, 'confidence': 0}
    
    short_ma = df['Close'].rolling(window=short_window).mean().iloc[-1]
    long_ma = df['Close'].rolling(window=long_window).mean().iloc[-1]
    prev_short_ma = df['Close'].rolling(window=short_window).mean().iloc[-2]
    prev_long_ma = df['Close'].rolling(window=long_window).mean().iloc[-2]
    
    # Generate signals
    if short_ma > long_ma and prev_short_ma <= prev_long_ma:
        return {'signal': 1, 'confidence': 0.7}  # Buy signal
    elif short_ma < long_ma and prev_short_ma >= prev_long_ma:
        return {'signal': -1, 'confidence': 0.7}  # Sell signal
    else:
        return {'signal': 0, 'confidence': 0}  # Hold