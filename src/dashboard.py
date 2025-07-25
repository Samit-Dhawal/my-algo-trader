import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import json
from datetime import datetime, timedelta
import asyncio
import time
from .data_collector import DataCollector
from .signal_engine import SignalEngine
from .order_executor import OrderExecutor
from .backtester import Backtester

# Page configuration
st.set_page_config(
    page_title="Algorithmic Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class TradingDashboard:
    def __init__(self):
        self.db_path = 'data/trading.db'
        self.data_collector = DataCollector()
        self.signal_engine = SignalEngine()
        self.order_executor = OrderExecutor(paper_trading=True)
        self.backtester = Backtester()
    
    def run(self):
        st.title("🤖 Algorithmic Trading Dashboard")
        
        # Sidebar
        self.render_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Live Dashboard", 
            "📈 Signals", 
            "💼 Portfolio", 
            "🔄 Backtesting", 
            "⚙️ Settings"
        ])
        
        with tab1:
            self.render_live_dashboard()
        
        with tab2:
            self.render_signals_tab()
        
        with tab3:
            self.render_portfolio_tab()
        
        with tab4:
            self.render_backtesting_tab()
        
        with tab5:
            self.render_settings_tab()
    
    def render_sidebar(self):
        st.sidebar.title("Navigation")
        
        # System status
        st.sidebar.subheader("System Status")
        
        # Check database connection
        try:
            conn = sqlite3.connect(self.db_path)
            conn.close()
            st.sidebar.success("Database: Connected")
        except:
            st.sidebar.error("Database: Disconnected")
        
        # Check API status
        account_info = self.order_executor.get_account_info()
        if account_info:
            st.sidebar.success("Alpaca API: Connected")
            st.sidebar.info(f"Portfolio Value: ${account_info.get('portfolio_value', 0):,.2f}")
        else:
            st.sidebar.error("Alpaca API: Disconnected")
        
        # Manual controls
        st.sidebar.subheader("Manual Controls")
        
        if st.sidebar.button("🔄 Collect Data Now"):
            with st.sidebar:
                with st.spinner("Collecting data..."):
                    # This should be async in production
                    st.success("Data collection triggered")
        
        if st.sidebar.button("🎯 Generate Signals"):
            with st.sidebar:
                with st.spinner("Generating signals..."):
                    st.success("Signals generated")
    
    def render_live_dashboard(self):
        """Main dashboard view"""
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        account_info = self.order_executor.get_account_info()
        
        with col1:
            st.metric(
                "Portfolio Value", 
                f"${account_info.get('portfolio_value', 0):,.2f}",
                delta="0.00"  # Calculate daily change
            )
        
        with col2:
            st.metric(
                "Buying Power", 
                f"${account_info.get('buying_power', 0):,.2f}"
            )
        
        with col3:
                    st.metric(
                        "Day Trades", 
                        account_info.get('day_trade_count', 0),
                        delta=None
                    )
        
        with col4:
            positions = self.order_executor.get_positions()
            total_positions = len(positions)
            st.metric("Active Positions", total_positions)
        
        # Market overview chart
        st.subheader("📈 Market Overview")
        
        # Get recent market data for watchlist
        watchlist = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY']
        market_data = self.get_market_overview(watchlist)
        
        if not market_data.empty:
            fig = self.create_market_overview_chart(market_data)
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent trades table
        st.subheader("🔄 Recent Trades")
        recent_trades = self.order_executor.get_trade_history(days=7)
        
        if recent_trades:
            trades_df = pd.DataFrame(recent_trades)
            st.dataframe(
                trades_df[['symbol', 'side', 'quantity', 'status', 'created_at']].head(10),
                use_container_width=True
            )
        else:
            st.info("No recent trades found.")
        
        # Live signals
        st.subheader("🎯 Live Signals")
        self.render_live_signals()
    
    def get_market_overview(self, symbols: list) -> pd.DataFrame:
        """Get market data for overview"""
        conn = sqlite3.connect(self.db_path)
        
        data_frames = []
        for symbol in symbols:
            query = '''
                SELECT symbol, timestamp, close_price
                FROM market_data
                WHERE symbol = ? AND timestamp >= datetime('now', '-1 day')
                ORDER BY timestamp DESC
                LIMIT 24
            '''
            df = pd.read_sql_query(query, conn, params=(symbol,))
            if not df.empty:
                data_frames.append(df)
        
        conn.close()
        
        if data_frames:
            return pd.concat(data_frames, ignore_index=True)
        return pd.DataFrame()
    
    def create_market_overview_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create market overview chart"""
        fig = make_subplots(
            rows=len(data['symbol'].unique()),
            cols=1,
            shared_xaxes=True,
            subplot_titles=data['symbol'].unique(),
            vertical_spacing=0.05
        )
        
        for i, symbol in enumerate(data['symbol'].unique(), 1):
            symbol_data = data[data['symbol'] == symbol].copy()
            symbol_data['timestamp'] = pd.to_datetime(symbol_data['timestamp'])
            symbol_data = symbol_data.sort_values('timestamp')
            
            fig.add_trace(
                go.Scatter(
                    x=symbol_data['timestamp'],
                    y=symbol_data['close_price'],
                    mode='lines',
                    name=symbol,
                    line=dict(width=2)
                ),
                row=i, col=1
            )
        
        fig.update_layout(
            height=600,
            showlegend=False,
            title_text="Market Overview - Last 24 Hours"
        )
        
        return fig
    
    def render_live_signals(self):
        """Render live trading signals"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Current Signals**")
            
            # Get latest signals from database
            conn = sqlite3.connect(self.db_path)
            query = '''
                SELECT symbol, signal, confidence, method, timestamp
                FROM signals
                WHERE timestamp >= datetime('now', '-1 hour')
                ORDER BY timestamp DESC
                LIMIT 10
            '''
            signals_df = pd.read_sql_query(query, conn)
            conn.close()
            
            if not signals_df.empty:
                for _, signal in signals_df.iterrows():
                    signal_color = "🟢" if signal['signal'] == 1 else "🔴" if signal['signal'] == -1 else "⚪"
                    st.write(f"{signal_color} **{signal['symbol']}** - {signal['method']} (Confidence: {signal['confidence']:.2f})")
            else:
                st.info("No recent signals generated.")
        
        with col2:
            st.write("**Signal Generation**")
            
            symbol_input = st.text_input("Symbol", value="AAPL")
            if st.button("Generate Signal"):
                if symbol_input:
                    with st.spinner(f"Generating signal for {symbol_input}..."):
                        signal_result = self.signal_engine.generate_signal(symbol_input)
                        if 'error' not in signal_result:
                            st.success(f"Signal: {signal_result['signal']} (Confidence: {signal_result['confidence']:.2f})")
                        else:
                            st.error(f"Error: {signal_result['error']}")
    
    def render_signals_tab(self):
        """Signals analysis tab"""
        st.header("🎯 Signal Analysis")
        
        # Signal configuration
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Configuration")
            
            # Symbol selection
            symbols = st.multiselect(
                "Select Symbols",
                options=['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY', 'QQQ', 'IWM'],
                default=['AAPL', 'MSFT']
            )
            
            # Strategy selection
            strategy = st.selectbox(
                "Strategy",
                options=['ML_Ensemble', 'MA_Crossover', 'RSI_Divergence', 'MACD_Signal']
            )
            
            # Time frame
            timeframe = st.selectbox(
                "Timeframe",
                options=['1m', '5m', '15m', '1h', '1d'],
                index=2
            )
            
            if st.button("Run Analysis"):
                st.session_state['run_analysis'] = True
        
        with col2:
            st.subheader("Signal History")
            
            # Signal history chart
            if symbols:
                signal_history = self.get_signal_history(symbols[0], days=7)
                if not signal_history.empty:
                    fig = self.create_signal_chart(signal_history)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Signal performance metrics
        if symbols:
            st.subheader("📊 Signal Performance")
            self.render_signal_performance(symbols)
    
    def get_signal_history(self, symbol: str, days: int = 7) -> pd.DataFrame:
        """Get signal history for analysis"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT s.timestamp, s.signal, s.confidence, s.method,
                   m.close_price, m.volume
            FROM signals s
            LEFT JOIN market_data m ON s.symbol = m.symbol 
                AND datetime(s.timestamp) = datetime(m.timestamp)
            WHERE s.symbol = ? AND s.timestamp >= datetime('now', '-{} days')
            ORDER BY s.timestamp ASC
        '''.format(days)
        
        df = pd.read_sql_query(query, conn, params=(symbol,))
        conn.close()
        
        return df
    
    def create_signal_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create signal visualization chart"""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=['Price & Signals', 'Signal Confidence']
        )
        
        # Price chart
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(data['timestamp']),
                y=data['close_price'],
                mode='lines',
                name='Price',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Buy signals
        buy_signals = data[data['signal'] == 1]
        if not buy_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=pd.to_datetime(buy_signals['timestamp']),
                    y=buy_signals['close_price'],
                    mode='markers',
                    name='Buy Signal',
                    marker=dict(color='green', size=10, symbol='triangle-up')
                ),
                row=1, col=1
            )
        
        # Sell signals
        sell_signals = data[data['signal'] == -1]
        if not sell_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=pd.to_datetime(sell_signals['timestamp']),
                    y=sell_signals['close_price'],
                    mode='markers',
                    name='Sell Signal',
                    marker=dict(color='red', size=10, symbol='triangle-down')
                ),
                row=1, col=1
            )
        
        # Confidence chart
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(data['timestamp']),
                y=data['confidence'],
                mode='lines+markers',
                name='Confidence',
                line=dict(color='purple', width=2)
            ),
            row=2, col=1
        )
        
        fig.update_layout(height=600, showlegend=True)
        
        return fig
    
    def render_signal_performance(self, symbols: list):
        """Render signal performance metrics"""
        performance_data = []
        
        for symbol in symbols:
            # Calculate signal accuracy
            conn = sqlite3.connect(self.db_path)
            query = '''
                SELECT COUNT(*) as total_signals,
                       AVG(confidence) as avg_confidence,
                       method
                FROM signals
                WHERE symbol = ? AND timestamp >= datetime('now', '-30 days')
                GROUP BY method
            '''
            df = pd.read_sql_query(query, conn, params=(symbol,))
            conn.close()
            
            for _, row in df.iterrows():
                performance_data.append({
                    'Symbol': symbol,
                    'Method': row['method'],
                    'Total Signals': row['total_signals'],
                    'Avg Confidence': f"{row['avg_confidence']:.3f}"
                })
        
        if performance_data:
            performance_df = pd.DataFrame(performance_data)
            st.dataframe(performance_df, use_container_width=True)
    
    def render_portfolio_tab(self):
        """Portfolio management tab"""
        st.header("💼 Portfolio Management")
        
        # Portfolio overview
        col1, col2, col3 = st.columns(3)
        
        account_info = self.order_executor.get_account_info()
        positions = self.order_executor.get_positions()
        
        with col1:
            st.subheader("Account Summary")
            st.write(f"**Portfolio Value:** ${account_info.get('portfolio_value', 0):,.2f}")
            st.write(f"**Cash:** ${account_info.get('cash', 0):,.2f}")
            st.write(f"**Buying Power:** ${account_info.get('buying_power', 0):,.2f}")
        
        with col2:
            st.subheader("Risk Metrics")
            # Calculate portfolio risk metrics
            if positions:
                total_value = sum(pos['market_value'] for pos in positions)
                st.write(f"**Total Position Value:** ${total_value:,.2f}")
                st.write(f"**Number of Positions:** {len(positions)}")
        
        with col3:
            st.subheader("Quick Actions")
            if st.button("🔄 Refresh Positions"):
                st.experimental_rerun()
            
            if st.button("📊 Generate Report"):
                st.info("Portfolio report generated!")
        
        # Current positions
        if positions:
            st.subheader("📊 Current Positions")
            
            positions_df = pd.DataFrame(positions)
            positions_df['Unrealized P&L %'] = positions_df['unrealized_plpc'].apply(lambda x: f"{x:.2%}")
            positions_df['Market Value'] = positions_df['market_value'].apply(lambda x: f"${x:,.2f}")
            positions_df['Unrealized P&L'] = positions_df['unrealized_pl'].apply(lambda x: f"${x:,.2f}")
            
            st.dataframe(
                positions_df[['symbol', 'quantity', 'avg_entry_price', 'current_price', 
                             'Market Value', 'Unrealized P&L', 'Unrealized P&L %']],
                use_container_width=True
            )
            
            # Portfolio allocation chart
            if len(positions) > 1:
                fig = go.Figure(data=[
                    go.Pie(
                        labels=[pos['symbol'] for pos in positions],
                        values=[abs(pos['market_value']) for pos in positions],
                        hole=0.4
                    )
                ])
                fig.update_layout(title="Portfolio Allocation")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No current positions.")
        
        # Order history
        st.subheader("📋 Order History")
        
        # Date filter
        col1, col2 = st.columns(2)
        with col1:
            days_back = st.selectbox("Show orders from last:", [7, 30, 90, 365], index=1)
        
        order_history = self.order_executor.get_trade_history(days=days_back)
        
        if order_history:
            orders_df = pd.DataFrame(order_history)
            st.dataframe(
                orders_df[['symbol', 'side', 'quantity', 'price', 'status', 'created_at']],
                use_container_width=True
            )
        else:
            st.info("No order history found.")
    
    def render_backtesting_tab(self):
        """Backtesting interface"""
        st.header("🔄 Strategy Backtesting")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Backtest Configuration")
            
            # Backtest parameters
            symbol = st.selectbox("Symbol", ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY'])
            
            start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
            end_date = st.date_input("End Date", value=datetime.now())
            
            initial_capital = st.number_input("Initial Capital", value=10000, min_value=1000, step=1000)
            
            strategy_params = st.expander("Strategy Parameters")
            with strategy_params:
                short_window = st.slider("Short MA Window", 5, 20, 10)
                long_window = st.slider("Long MA Window", 20, 50, 20)
            
            run_backtest = st.button("🚀 Run Backtest", type="primary")
        
        with col2:
            st.subheader("Backtest Results")
            
            if run_backtest:
                with st.spinner("Running backtest..."):
                    # Import strategy function
                    from .backtester import simple_ma_crossover_strategy
                    
                    # Run backtest
                    results = self.backtester.simulate_strategy(
                        symbol=symbol,
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=end_date.strftime('%Y-%m-%d'),
                        strategy_func=simple_ma_crossover_strategy,
                        short_window=short_window,
                        long_window=long_window
                    )
                    
                    if 'error' not in results:
                        # Display results
                        metrics_col1, metrics_col2 = st.columns(2)
                        
                        with metrics_col1:
                            st.metric("Total Return", f"{results['total_return_pct']:.2f}%")
                            st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.3f}")
                            st.metric("Max Drawdown", f"{results['max_drawdown_pct']:.2f}%")
                        
                        with metrics_col2:
                            st.metric("Win Rate", f"{results['win_rate_pct']:.1f}%")
                            st.metric("Total Trades", results['total_trades'])
                            st.metric("Alpha vs Buy&Hold", f"{results['alpha']:.2f}%")
                        
                        # Equity curve chart
                        if results['equity_curve']:
                            equity_df = pd.DataFrame(results['equity_curve'])
                            equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=equity_df['timestamp'],
                                y=equity_df['value'],
                                mode='lines',
                                name='Strategy',
                                line=dict(color='blue', width=2)
                            ))
                            
                            fig.update_layout(
                                title="Equity Curve",
                                xaxis_title="Date",
                                yaxis_title="Portfolio Value ($)",
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Trade analysis
                        if results['trades']:
                            st.subheader("Trade Analysis")
                            trades_df = pd.DataFrame(results['trades'])
                            st.dataframe(trades_df, use_container_width=True)
                    else:
                        st.error(f"Backtest failed: {results['error']}")
    
    def render_settings_tab(self):
        """System settings and configuration"""
        st.header("⚙️ System Settings")
        
        tab1, tab2, tab3 = st.tabs(["API Settings", "Risk Management", "Alerts"])
        
        with tab1:
            st.subheader("API Configuration")
            
            # Alpaca settings
            st.write("**Alpaca API**")
            alpaca_key = st.text_input("API Key", type="password", placeholder="Enter your Alpaca API key")
            alpaca_secret = st.text_input("Secret Key", type="password", placeholder="Enter your Alpaca secret key")
            paper_trading = st.checkbox("Paper Trading", value=True)
            
            # Data source settings
            st.write("**Data Sources**")
            alpha_vantage_key = st.text_input("Alpha Vantage API Key", type="password")
            news_api_key = st.text_input("NewsAPI Key", type="password")
            
            if st.button("Save API Settings"):
                st.success("API settings saved successfully!")
        
        with tab2:
            st.subheader("Risk Management")
            
            max_position_size = st.slider("Max Position Size (%)", 1, 50, 20)
            max_daily_trades = st.number_input("Max Daily Trades", 1, 100, 10)
            stop_loss_pct = st.slider("Default Stop Loss (%)", 1, 20, 5)
            take_profit_pct = st.slider("Default Take Profit (%)", 1, 50, 15)
            
            if st.button("Save Risk Settings"):
                st.success("Risk management settings saved!")
        
        with tab3:
            st.subheader("Alert Settings")
            
            # Email alerts
            st.write("**Email Alerts**")
            email_enabled = st.checkbox("Enable Email Alerts")
            email_address = st.text_input("Email Address")
            smtp_server = st.text_input("SMTP Server", value="smtp.gmail.com")
            smtp_port = st.number_input("SMTP Port", value=587)
            
            # Telegram alerts
            st.write("**Telegram Alerts**")
            telegram_enabled = st.checkbox("Enable Telegram Alerts")
            telegram_token = st.text_input("Bot Token", type="password")
            telegram_chat_id = st.text_input("Chat ID")
            
            if st.button("Save Alert Settings"):
                st.success("Alert settings saved!")

# Utility functions for the dashboard
def format_currency(value):
    """Format currency values"""
    return f"${value:,.2f}"

def format_percentage(value):
    """Format percentage values"""
    return f"{value:.2f}%"

# Main dashboard runner
if __name__ == "__main__":
    dashboard = TradingDashboard()
    dashboard.run()