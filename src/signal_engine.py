import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Tuple, Optional
import sqlite3
import json
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os

class SignalEngine:
    def __init__(self, db_path: str = 'data/trading.db'):
        self.db_path = db_path
        self.models = {}
        self.scalers = {}
        self.load_models()
    
    def load_models(self):
        """Load pre-trained ML models if they exist"""
        model_dir = 'models'
        if os.path.exists(model_dir):
            for filename in os.listdir(model_dir):
                if filename.endswith('_model.pkl'):
                    symbol = filename.replace('_model.pkl', '')
                    with open(f'{model_dir}/{filename}', 'rb') as f:
                        self.models[symbol] = pickle.load(f)
                    
                    scaler_file = f'{model_dir}/{symbol}_scaler.pkl'
                    if os.path.exists(scaler_file):
                        with open(scaler_file, 'rb') as f:
                            self.scalers[symbol] = pickle.load(f)
    
    def get_market_data(self, symbol: str, days: int = 100) -> pd.DataFrame:
        """Retrieve market data from database"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT timestamp, open_price, high_price, low_price, close_price, volume
            FROM market_data
            WHERE symbol = ? AND timestamp >= datetime('now', '-{} days')
            ORDER BY timestamp ASC
        '''.format(days)
        
        df = pd.read_sql_query(query, conn, params=(symbol,))
        conn.close()
        
        if df.empty:
            return df
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        return df
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        if df.empty:
            return df
        
        # Ensure we have numpy arrays
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        volume = df['Volume'].values
        
        # Moving Averages
        df['SMA_10'] = talib.SMA(close, timeperiod=10)
        df['SMA_20'] = talib.SMA(close, timeperiod=20)
        df['SMA_50'] = talib.SMA(close, timeperiod=50)
        df['EMA_12'] = talib.EMA(close, timeperiod=12)
        df['EMA_26'] = talib.EMA(close, timeperiod=26)
        
        # Momentum Indicators
        df['RSI'] = talib.RSI(close, timeperiod=14)
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(close)
        df['CCI'] = talib.CCI(high, low, close, timeperiod=14)
        df['MOM'] = talib.MOM(close, timeperiod=10)
        
        # Volatility Indicators
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(close)
        df['ATR'] = talib.ATR(high, low, close, timeperiod=14)
        
        # Volume Indicators
        df['OBV'] = talib.OBV(close, volume)
        df['AD'] = talib.AD(high, low, close, volume)
        
        # Trend Indicators
        df['ADX'] = talib.ADX(high, low, close, timeperiod=14)
        df['AROON_up'], df['AROON_down'] = talib.AROON(high, low, timeperiod=14)
        
        # Price Action
        df['PCT_CHANGE'] = df['Close'].pct_change()
        df['HIGH_LOW_PCT'] = (df['High'] - df['Low']) / df['Close']
        df['PRICE_POSITION'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        
        # Support/Resistance Levels
        df['SUPPORT'] = df['Low'].rolling(window=20).min()
        df['RESISTANCE'] = df['High'].rolling(window=20).max()
        df['DISTANCE_TO_SUPPORT'] = (df['Close'] - df['SUPPORT']) / df['Close']
        df['DISTANCE_TO_RESISTANCE'] = (df['RESISTANCE'] - df['Close']) / df['Close']
        
        return df
    
    def generate_traditional_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate signals using traditional technical analysis rules"""
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0
        signals['strength'] = 0.0
        signals['reason'] = ''
        
        if len(df) < 50:  # Need enough data for indicators
            return signals
        
        for i in range(50, len(df)):
            score = 0
            reasons = []
            
            # Moving Average Crossover Signals
            if df['SMA_10'].iloc[i] > df['SMA_20'].iloc[i] and df['SMA_10'].iloc[i-1] <= df['SMA_20'].iloc[i-1]:
                score += 2
                reasons.append('SMA_Cross_Bull')
            elif df['SMA_10'].iloc[i] < df['SMA_20'].iloc[i] and df['SMA_10'].iloc[i-1] >= df['SMA_20'].iloc[i-1]:
                score -= 2
                reasons.append('SMA_Cross_Bear')
            
            # RSI Signals
            rsi = df['RSI'].iloc[i]
            if rsi < 30 and df['RSI'].iloc[i-1] >= 30:
                score += 1
                reasons.append('RSI_Oversold')
            elif rsi > 70 and df['RSI'].iloc[i-1] <= 70:
                score -= 1
                reasons.append('RSI_Overbought')
            
            # MACD Signals
            if (df['MACD'].iloc[i] > df['MACD_signal'].iloc[i] and 
                df['MACD'].iloc[i-1] <= df['MACD_signal'].iloc[i-1]):
                score += 1.5
                reasons.append('MACD_Bull')
            elif (df['MACD'].iloc[i] < df['MACD_signal'].iloc[i] and 
                  df['MACD'].iloc[i-1] >= df['MACD_signal'].iloc[i-1]):
                score -= 1.5
                reasons.append('MACD_Bear')
            
            # Bollinger Band Signals
            if df['Close'].iloc[i] < df['BB_lower'].iloc[i]:
                score += 0.5
                reasons.append('BB_Oversold')
            elif df['Close'].iloc[i] > df['BB_upper'].iloc[i]:
                score -= 0.5
                reasons.append('BB_Overbought')
            
            # Volume Confirmation
            volume_avg = df['Volume'].iloc[i-20:i].mean()
            if df['Volume'].iloc[i] > volume_avg * 1.5:
                if score > 0:
                    score *= 1.2  # Amplify bullish signal with high volume
                elif score < 0:
                    score *= 1.2  # Amplify bearish signal with high volume
                reasons.append('High_Volume')
            
            # Trend Confirmation with ADX
            adx = df['ADX'].iloc[i]
            if adx > 25:  # Strong trend
                if df['Close'].iloc[i] > df['SMA_50'].iloc[i]:  # Uptrend
                    if score > 0:
                        score *= 1.1
                else:  # Downtrend
                    if score < 0:
                        score *= 1.1
                reasons.append('Strong_Trend')
            
            # Generate final signal
            if score >= 2:
                signals.loc[df.index[i], 'signal'] = 1  # Buy
            elif score <= -2:
                signals.loc[df.index[i], 'signal'] = -1  # Sell
            else:
                signals.loc[df.index[i], 'signal'] = 0  # Hold
            
            signals.loc[df.index[i], 'strength'] = abs(score) / 10  # Normalize to 0-1
            signals.loc[df.index[i], 'reason'] = ','.join(reasons)
        
        return signals
    
    def prepare_ml_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for machine learning model"""
        feature_columns = [
            'RSI', 'MACD', 'MACD_signal', 'CCI', 'MOM',
            'BB_upper', 'BB_middle', 'BB_lower', 'ATR',
            'ADX', 'AROON_up', 'AROON_down',
            'PCT_CHANGE', 'HIGH_LOW_PCT', 'PRICE_POSITION',
            'DISTANCE_TO_SUPPORT', 'DISTANCE_TO_RESISTANCE'
        ]
        
        # Fill NaN values with forward fill then backward fill
        feature_data = df[feature_columns].fillna(method='ffill').fillna(method='bfill')
        
        return feature_data.values
    
    def generate_ml_signals(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        """Generate signals using machine learning model"""
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0
        signals['probability'] = 0.0
        
        if symbol not in self.models or len(df) < 100:
            return signals
        
        try:
            # Prepare features
            features = self.prepare_ml_features(df)
            
            if symbol in self.scalers:
                features = self.scalers[symbol].transform(features)
            
            # Generate predictions
            predictions = self.models[symbol].predict(features)
            probabilities = self.models[symbol].predict_proba(features)
            
            # Map predictions to signals
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                if i < len(df):
                    signals.iloc[i]['signal'] = pred
                    signals.iloc[i]['probability'] = max(prob)
            
        except Exception as e:
            print(f"Error generating ML signals for {symbol}: {e}")
        
        return signals
    
    def combine_signals(self, traditional_signals: pd.DataFrame, 
                       ml_signals: pd.DataFrame, 
                       traditional_weight: float = 0.6,
                       ml_weight: float = 0.4) -> pd.DataFrame:
        """Combine traditional and ML signals with weighted approach"""
        combined = pd.DataFrame(index=traditional_signals.index)
        combined['signal'] = 0
        combined['confidence'] = 0.0
        combined['method'] = ''
        
        for i in range(len(combined)):
            trad_signal = traditional_signals.iloc[i]['signal']
            trad_strength = traditional_signals.iloc[i]['strength']
            
            ml_signal = ml_signals.iloc[i]['signal'] if i < len(ml_signals) else 0
            ml_prob = ml_signals.iloc[i]['probability'] if i < len(ml_signals) else 0
            
            # Weighted combination
            combined_score = (trad_signal * trad_strength * traditional_weight + 
                            ml_signal * ml_prob * ml_weight)
            
            # Generate final signal
            if combined_score > 0.3:
                combined.iloc[i]['signal'] = 1
                combined.iloc[i]['confidence'] = min(combined_score, 1.0)
                combined.iloc[i]['method'] = 'combined_buy'
            elif combined_score < -0.3:
                combined.iloc[i]['signal'] = -1
                combined.iloc[i]['confidence'] = min(abs(combined_score), 1.0)
                combined.iloc[i]['method'] = 'combined_sell'
            else:
                combined.iloc[i]['signal'] = 0
                combined.iloc[i]['confidence'] = 0
                combined.iloc[i]['method'] = 'hold'
        
        return combined
    
    def generate_signals(self, symbol: str) -> Dict:
        """Main signal generation function"""
        print(f"Generating signals for {symbol}")
        
        # Get market data
        df = self.get_market_data(symbol, days=100)
        
        if df.empty:
            return {
                'symbol': symbol,
                'signal': 0,
                'confidence': 0,
                'method': 'no_data',
                'timestamp': datetime.now().isoformat()
            }
        
        # Calculate technical indicators
        df = self.calculate_technical_indicators(df)
        
        # Generate traditional signals
        traditional_signals = self.generate_traditional_signals(df)
        
        # Generate ML signals (if model exists)
        ml_signals = self.generate_ml_signals(symbol, df)
        
        # Combine signals
        final_signals = self.combine_signals(traditional_signals, ml_signals)
        
        # Get latest signal
        latest_signal = final_signals.iloc[-1]
        
        return {
            'symbol': symbol,
            'signal': int(latest_signal['signal']),
            'confidence': float(latest_signal['confidence']),
            'method': latest_signal['method'],
            'current_price': float(df['Close'].iloc[-1]),
            'rsi': float(df['RSI'].iloc[-1]) if not pd.isna(df['RSI'].iloc[-1]) else None,
            'macd': float(df['MACD'].iloc[-1]) if not pd.isna(df['MACD'].iloc[-1]) else None,
            'volume': int(df['Volume'].iloc[-1]),
            'timestamp': datetime.now().isoformat()
        }
    
    def train_model(self, symbol: str, lookback_days: int = 365):
        """Train ML model for a specific symbol"""
        print(f"Training ML model for {symbol}")
        
        # Get historical data
        df = self.get_market_data(symbol, days=lookback_days)
        
        if len(df) < 100:
            print(f"Insufficient data for {symbol}")
            return False
        
        # Calculate indicators
        df = self.calculate_technical_indicators(df)
        
        # Prepare features
        features = self.prepare_ml_features(df)
        
        # Create labels (future returns)
        df['future_return'] = df['Close'].shift(-5) / df['Close'] - 1
        df['label'] = 0
        df.loc[df['future_return'] > 0.02, 'label'] = 1  # Buy if >2% return
        df.loc[df['future_return'] < -0.02, 'label'] = -1  # Sell if <-2% return
        
        # Remove NaN values
        valid_idx = ~(pd.isna(df['label']) | pd.isna(features).any(axis=1))
        X = features[valid_idx]
        y = df.loc[valid_idx, 'label'].values
        
        if len(X) < 50:
            print(f"Insufficient valid data for {symbol}")
            return False
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        
        # Save
# Save model and scaler
        os.makedirs('models', exist_ok=True)
        
        with open(f'models/{symbol}_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        with open(f'models/{symbol}_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        # Store in memory
        self.models[symbol] = model
        self.scalers[symbol] = scaler
        
        # Evaluate model
        from sklearn.metrics import classification_report
        y_pred = model.predict(X_scaled)
        print(f"Model training completed for {symbol}")
        print(classification_report(y, y_pred))
        
        return True
    
    def get_news_sentiment(self, symbol: str) -> float:
        """Get average news sentiment for symbol"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT AVG(sentiment_score) as avg_sentiment
            FROM news_data
            WHERE symbols LIKE ? AND published_at >= datetime('now', '-24 hours')
        '''
        
        result = pd.read_sql_query(query, conn, params=(f'%{symbol}%',))
        conn.close()
        
        if not result.empty and not pd.isna(result.iloc[0]['avg_sentiment']):
            return float(result.iloc[0]['avg_sentiment'])
        
        return 0.0