# backtest.py

import pandas as pd
import numpy as np

class Backtester:
    def __init__(self, initial_balance=100000):
        self.initial_balance = initial_balance

    def run_backtest(self, data, signals):
        portfolio = self.simulate_trading(data, signals)
        return self.calculate_metrics(portfolio)

    def simulate_trading(self, data, signals):
        balance = self.initial_balance
        position = 0
        portfolio = []

        for i in range(len(signals)):
            price = data['Close'].iloc[i]
            signal = signals[i]

            if signal == "BUY" and balance >= price:
                position = balance / price
                balance = 0
            elif signal == "SELL" and position > 0:
                balance = position * price
                position = 0

            portfolio_value = balance + (position * price)
            portfolio.append(portfolio_value)

        return pd.Series(portfolio, index=data.index)

    def calculate_metrics(self, portfolio):
        returns = portfolio.pct_change().dropna()
        return {
            'final_value': portfolio.iloc[-1],
            'total_return': (portfolio.iloc[-1] / portfolio.iloc[0]) - 1,
            'sharpe_ratio': self.calculate_sharpe_ratio(returns),
            'max_drawdown': self.calculate_max_drawdown(returns)
        }

    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.0):
        if returns.std() == 0:
            return 0
        return (returns.mean() - risk_free_rate) / returns.std()

    def calculate_max_drawdown(self, returns):
        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        return drawdown.min()
