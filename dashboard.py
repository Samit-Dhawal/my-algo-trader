# === dashboard.py ===
import streamlit as st
import plotly.graph_objects as go
from market_data import DataCollector
from signals import SignalEngine
import pandas as pd
from executor import OrderExecutor
from backtest import Backtester
import os
from dotenv import load_dotenv
import streamlit_autorefresh

load_dotenv()
streamlit_autorefresh.st_autorefresh(interval=60000, key='refresh')

st.set_page_config(page_title="Algo Trading Dashboard", layout="wide")

st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)


symbols = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "V", "PG",
    "UNH", "KO", "PEP", "MA", "WMT", "DIS", "NFLX", "BA", "XOM", "IBM"
]

periods = [
    "1d", "5d", "1wk", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"
]

symbol = st.sidebar.selectbox("📈 Stock Symbol", symbols)
period = st.sidebar.selectbox("🕒 Period", periods)
custom_rows = st.sidebar.number_input(
    "🔢 Optional: Number of rows to display",
    min_value=1, max_value=500, step=1, value=None,
    placeholder="Leave blank to show all"
)

@st.cache_data(show_spinner=False)
def load_data(symbol, period):
    collector = DataCollector()
    return collector.get_stock_data(symbol, period)

@st.cache_data(show_spinner=False)
def compute_signals(data):
    engine = SignalEngine()
    df = engine.technical_indicators(data)
    signals = engine.generate_signal(df)
    return df, signals

col1, col2 = st.columns(2)

with col1:
    st.markdown("## 📊 Price Chart", unsafe_allow_html=True)
    with st.spinner("Fetching data..."):
        data = load_data(symbol, period)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'], high=data['High'],
        low=data['Low'], close=data['Close']
    ))

    fig.update_layout(
        hovermode='x unified',
        xaxis=dict(showspikes=True, spikemode='across', spikesnap='cursor'),
        yaxis=dict(showspikes=True, spikemode='across', spikesnap='cursor'),
        dragmode='zoom',
        margin=dict(l=10, r=10, t=10, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("## 🔔 Trading Signals", unsafe_allow_html=True)
    with st.spinner("Generating signals..."):
        data_with_indicators, signals = compute_signals(data)

    data_reset = data.reset_index()

    def signal_to_icon(signal):
        if signal == 'BUY':
            return '🟢 BUY'
        elif signal == 'SELL':
            return '🔴 SELL'
        elif signal == 'HOLD':
            return '🟡 HOLD'
        return signal

    signal_df = pd.DataFrame({
        'Date': data_reset['Date'],
        'Price': data_reset['Close'],
        'Signal': [signal_to_icon(sig) for sig in signals]
    }).sort_values(by='Date', ascending=False)

    if custom_rows:
        signal_df = signal_df.head(custom_rows)

    # def highlight_signal(val):
    #     if 'BUY' in val: return 'background-color: green; color: white'
    #     elif 'SELL' in val: return 'background-color: red; color: white'
    #     elif 'HOLD' in val: return 'background-color: yellow; color: black'
    #     return ''

    # styled_df = signal_df.style.map(highlight_signal, subset=['Signal'])
    st.dataframe(signal_df, use_container_width=True)

    # === Live Trade Execution ===
    st.subheader("⚡ Live Trade Execution")

    latest_signal = signals[-1]
    latest_price = data_with_indicators['Close'].iloc[-1]
    emoji = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '🟡'}.get(latest_signal, '')
    st.markdown(f"**Latest Signal:** {emoji} `{latest_signal}` at price `{latest_price:.2f}`")

    qty = st.number_input("Quantity", min_value=1, value=1, step=1)
    enable_live = st.checkbox("✅ Enable Live Trading")
    executor = OrderExecutor()


    # Put these 3 buttons side-by-side
    btn1, btn2, btn3 = st.columns(3)

    with btn1:
        if st.button("▶️ Execute Trade Now") and enable_live:
            if latest_signal in ['BUY', 'SELL']:
                result = executor.place_order(symbol, qty, latest_signal.lower())
                st.success(result)
            else:
                st.warning("No action taken. Signal is HOLD.")

    with btn2:
        if st.button("❌ Cancel All Open Orders"):
            cancel_result = executor.cancel_open_orders()
            st.info(cancel_result)

    with btn3:
        if st.button("📈 Show Current Positions"):
            success, positions = executor.get_positions()
            if success:
                if positions:
                    for pos in positions:
                        st.write(f"📌 {pos['symbol']}: {pos['qty']} shares at avg price {pos['avg_entry_price']}")
                else:
                    st.info("No open positions yet.")
            else:
                st.error(positions)


    # with btn4:
    #     st.write("")  # If you want some space or add another button here later


    st.subheader("📈 Backtest Performance")
    backtester = Backtester()
    with st.spinner("Running backtest..."):
        metrics = backtester.run_backtest(data_with_indicators, signals)

    st.metric("Total Return", f"{metrics['total_return'] * 100:.2f}%")
    st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
    st.metric("Max Drawdown", f"{metrics['max_drawdown'] * 100:.2f}%")
