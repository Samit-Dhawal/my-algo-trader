# signals.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import talib

class SignalEngine:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=50)
        
    def technical_indicators(self, data):
        """Calculate technical indicators"""
        data['SMA_20'] = data['Close'].rolling(20).mean()
        data['SMA_50'] = data['Close'].rolling(50).mean()
        data['RSI'] = talib.RSI(data['Close'].values)
        data['MACD'], data['MACD_signal'], _ = talib.MACD(data['Close'].values)
        return data
    
    def generate_signal(self, data):
        """Generate buy/sell signals"""
        signals = []
        for i in range(len(data)):
            if data['SMA_20'].iloc[i] > data['SMA_50'].iloc[i] and data['RSI'].iloc[i] < 70:
                signals.append('BUY')
            elif data['SMA_20'].iloc[i] < data['SMA_50'].iloc[i] and data['RSI'].iloc[i] > 30:
                signals.append('SELL')
            else:
                signals.append('HOLD')
        return signals