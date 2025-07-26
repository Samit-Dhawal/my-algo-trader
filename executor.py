# executor.py
import os
from dotenv import load_dotenv
from alpaca_trade_api import REST

load_dotenv()

class OrderExecutor:
    def __init__(self, paper=True):
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        self.api = REST(api_key, secret_key, 
                       base_url='https://paper-api.alpaca.markets/' if paper 
                       else 'https://api.alpaca.markets')
    
    def place_order(self, symbol, qty, side):
        """Place paper trading order"""
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='market',
                time_in_force='gtc'
            )
            return f"Order placed: {order.id} for {qty} shares of {symbol} on the {side} side."
        except Exception as e:
            # print(f"Order failed: {e}")
            return f"Order failed: {e}"
        

    def cancel_open_orders(self):
        try:
            open_orders = self.api.list_orders(status="open")
            for order in open_orders:
                self.api.cancel_order(order.id)
            return f"Cancelled {len(open_orders)} open orders."
        except Exception as e:
            return f"Error cancelling orders: {e}"

    def get_positions(self):
        try:
            positions = self.api.list_positions()
            result = [
                {
                    "symbol": pos.symbol,
                    "qty": pos.qty,
                    "avg_entry_price": pos.avg_entry_price
                }
                for pos in positions
            ]
            return True, result
        except Exception as e:
            return False, f"Error fetching positions: {e}"